#!/usr/bin/env python3
"""
maasv Graph Enrichment — One-Time Backfill

Reads existing memories (project, decision, learning, preference categories)
and re-extracts technology entities and relationships using the enriched
extraction prompt.

Usage (from the doris project directory, with its venv active):
    cd /Users/macmini/Projects/doris
    python /Users/macmini/Projects/maasv/scripts/enrich_graph.py

Options:
    --dry-run         Show what would be extracted without writing to DB
    --max-batches N   Process at most N batches (default: unlimited)
    --batch-size N    Memories per LLM call (default: 10)
    --resume          Resume from last checkpoint
    -v, --verbose     Show detailed extraction results

Cost estimate: ~$0.80 total (Haiku 4.5, ~180 batches, ~450K tokens)
Expected output: 20-50 new tech entities, 30-80 new relationships
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure both maasv and doris are importable
sys.path.insert(0, "/Users/macmini/Projects/maasv")
sys.path.insert(0, "/Users/macmini/Projects/doris")
os.chdir("/Users/macmini/Projects/doris")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("enrich_graph")

CHECKPOINT_PATH = Path("/Users/macmini/Projects/maasv/scripts/.enrich_checkpoint.json")
ENRICHABLE_CATEGORIES = {"project", "decision", "learning", "preference"}


def load_checkpoint() -> dict:
    """Load checkpoint state from disk."""
    if CHECKPOINT_PATH.exists():
        try:
            return json.loads(CHECKPOINT_PATH.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    return {"processed_ids": [], "batch_index": 0, "stats": {
        "entities_created": 0, "relationships_created": 0,
        "entities_skipped": 0, "batches_processed": 0,
        "memories_processed": 0, "errors": 0,
    }}


def save_checkpoint(state: dict):
    """Save checkpoint state to disk."""
    try:
        CHECKPOINT_PATH.write_text(json.dumps(state, indent=2))
    except OSError as e:
        logger.error(f"Failed to save checkpoint: {e}")


def get_enrichable_memories() -> list[dict]:
    """Fetch all active memories in enrichable categories."""
    from maasv.core.store import get_all_active
    all_memories = []
    for cat in sorted(ENRICHABLE_CATEGORIES):
        memories = get_all_active(category=cat)
        all_memories.extend(memories)
        logger.info(f"  {cat}: {len(memories)} memories")
    return all_memories


def build_batch_prompt(memories: list[dict], known_entities: dict[str, str]) -> str:
    """Build a prompt for batch extraction from multiple memories."""
    entities_section = ""
    if known_entities:
        lines = [f"- {name} ({etype})" for name, etype in known_entities.items()]
        entities_section = "\nKnown entities (avoid duplicates — use these exact names):\n" + "\n".join(lines) + "\n"

    memory_texts = []
    for i, mem in enumerate(memories, 1):
        subject = f" [{mem.get('subject', 'unknown')}]" if mem.get('subject') else ""
        memory_texts.append(f"Memory {i}{subject}: {mem['content']}")

    return f"""Extract technology entities and relationships from these memories for a knowledge graph.

Focus on:
1. **Technologies** — Programming languages, frameworks, databases, tools, services, platforms
2. **Technology relationships** — Which projects use which technologies
3. **Projects** — Only if not already in known entities

For each technology mentioned, extract the entity AND its relationship to any project.
{entities_section}
Return JSON:
```json
{{
    "entities": [
        {{
            "name": "Display Name",
            "type": "technology|project",
            "description": "Brief context from the memories",
            "confidence": 0.0-1.0
        }}
    ],
    "relationships": [
        {{
            "subject": "Project Name",
            "predicate": "uses_tech|built_with|written_in|runs_on|hosted_on|depends_on",
            "object": "Technology Name",
            "object_is_entity": true,
            "confidence": 0.0-1.0
        }}
    ]
}}
```

IMPORTANT:
- Only extract technologies with PROPER NAMES (e.g., "FastAPI" not "web framework")
- Use exact names from known entities when they match
- Confidence 0.9+ for explicitly stated facts, 0.6-0.8 for inferred
- Return empty arrays if no technologies found
- Do NOT extract people, places, or family relationships (already handled)

MEMORIES:
{chr(10).join(memory_texts)}
"""


def extract_batch(memories: list[dict], known_entities: dict[str, str], dry_run: bool = False) -> dict:
    """Run extraction on a batch of memories."""
    import maasv

    llm = maasv.get_llm()
    config = maasv.get_config()
    prompt = build_batch_prompt(memories, known_entities)

    try:
        content = llm.call(
            messages=[{"role": "user", "content": prompt}],
            model=config.extraction_model,
            max_tokens=2000,
            source="graph-enrichment",
        )

        # Parse JSON
        data = None
        try:
            data = json.loads(content.strip())
        except json.JSONDecodeError:
            try:
                if "```json" in content:
                    stripped = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    stripped = content.split("```")[1].split("```")[0]
                else:
                    stripped = content
                data = json.loads(stripped.strip())
            except (json.JSONDecodeError, IndexError) as e:
                logger.warning(f"Unparseable response: {e}")
                return {"entities": [], "relationships": [], "error": str(e)}

        if data is None:
            return {"entities": [], "relationships": [], "error": "No data parsed"}

        return {
            "entities": data.get("entities", []),
            "relationships": data.get("relationships", []),
        }

    except Exception as e:
        logger.error(f"Batch extraction failed: {e}")
        return {"entities": [], "relationships": [], "error": str(e)}


def store_extraction_results(result: dict, dry_run: bool = False) -> dict:
    """Store extracted entities and relationships, deduping against existing graph."""
    from maasv.core.store import find_entity_by_name, find_or_create_entity, add_relationship
    from maasv.extraction.entity_extraction import _is_garbage_entity

    stats = {"entities_created": 0, "relationships_created": 0, "entities_skipped": 0}
    entity_id_map = {}

    for entity in result.get("entities", []):
        name = entity.get("name", "").strip()
        entity_type = entity.get("type", "technology")
        confidence = entity.get("confidence", 0.7)

        if not name or _is_garbage_entity(name) or confidence < 0.5:
            stats["entities_skipped"] += 1
            continue

        existing = find_entity_by_name(name)
        if existing:
            entity_id_map[name] = existing["id"]
            stats["entities_skipped"] += 1
            continue

        if dry_run:
            logger.info(f"  [DRY RUN] Would create entity: {name} ({entity_type})")
            stats["entities_created"] += 1
            continue

        try:
            entity_id = find_or_create_entity(
                name=name,
                entity_type=entity_type,
                metadata={
                    "description": entity.get("description"),
                    "source": "enrichment_backfill",
                    "confidence": confidence,
                }
            )
            entity_id_map[name] = entity_id
            stats["entities_created"] += 1
            logger.info(f"  Created entity: {name} ({entity_type})")
        except Exception as e:
            logger.warning(f"  Failed to create entity {name}: {e}")

    for rel in result.get("relationships", []):
        subject_name = rel.get("subject", "").strip()
        predicate = rel.get("predicate", "").strip()
        object_name = rel.get("object", "").strip()
        confidence = rel.get("confidence", 0.7)

        if not subject_name or not predicate or not object_name:
            continue
        if confidence < 0.5:
            continue
        if _is_garbage_entity(subject_name) or _is_garbage_entity(object_name):
            continue

        if dry_run:
            logger.info(f"  [DRY RUN] Would create: {subject_name} -{predicate}-> {object_name}")
            stats["relationships_created"] += 1
            continue

        try:
            # Resolve subject
            subject_id = entity_id_map.get(subject_name)
            if not subject_id:
                subject_entity = find_entity_by_name(subject_name)
                if subject_entity:
                    subject_id = subject_entity["id"]
                else:
                    subject_id = find_or_create_entity(
                        name=subject_name,
                        entity_type="project",
                        metadata={"source": "enrichment_backfill"}
                    )

            # Resolve object
            object_id = entity_id_map.get(object_name)
            if not object_id:
                object_entity = find_entity_by_name(object_name)
                if object_entity:
                    object_id = object_entity["id"]
                else:
                    object_id = find_or_create_entity(
                        name=object_name,
                        entity_type="technology",
                        metadata={"source": "enrichment_backfill"}
                    )

            add_relationship(
                subject_id=subject_id,
                predicate=predicate,
                object_id=object_id,
                confidence=confidence,
                source="enrichment_backfill",
            )
            stats["relationships_created"] += 1
            logger.info(f"  Created: {subject_name} -{predicate}-> {object_name}")

        except Exception as e:
            logger.warning(f"  Failed to create relationship: {e}")

    return stats


def run_enrichment(
    dry_run: bool = False,
    max_batches: int = 0,
    batch_size: int = 10,
    resume: bool = False,
    verbose: bool = False,
):
    """Main enrichment loop."""
    # Initialize maasv through Doris's bridge
    from maasv_bridge import init_maasv
    init_maasv()

    import maasv
    config = maasv.get_config()
    known_entities = config.known_entities

    print()
    print("=" * 60)
    print("  maasv Graph Enrichment")
    print("=" * 60)
    print()

    if dry_run:
        print("  MODE: Dry run (no writes)")
    print(f"  Batch size: {batch_size}")
    if max_batches:
        print(f"  Max batches: {max_batches}")
    print()

    # Load checkpoint if resuming, otherwise fresh start
    if resume:
        state = load_checkpoint()
        if state["stats"]["batches_processed"] > 0:
            print(f"  Resuming from batch {state['batch_index']} "
                  f"({state['stats']['memories_processed']} memories already processed)")
    else:
        state = {
            "processed_ids": [], "batch_index": 0,
            "stats": {
                "entities_created": 0, "relationships_created": 0,
                "entities_skipped": 0, "batches_processed": 0,
                "memories_processed": 0, "errors": 0,
            }
        }

    # Fetch memories
    print("  Loading memories...")
    all_memories = get_enrichable_memories()
    print(f"  Total enrichable memories: {len(all_memories)}")
    print()

    # Filter out already-processed memories
    processed_set = set(state["processed_ids"])
    memories = [m for m in all_memories if m["id"] not in processed_set]
    print(f"  Remaining to process: {len(memories)}")

    if not memories:
        print("  Nothing to process!")
        return state["stats"]

    # Batch and process
    batches = [memories[i:i + batch_size] for i in range(0, len(memories), batch_size)]
    total_batches = min(len(batches), max_batches) if max_batches else len(batches)
    print(f"  Batches to process: {total_batches}")
    print()

    start_batch = state["batch_index"]
    for batch_idx, batch in enumerate(batches[:total_batches]):
        batch_num = start_batch + batch_idx + 1
        print(f"  Batch {batch_num}/{start_batch + total_batches} "
              f"({len(batch)} memories)...", end="", flush=True)

        start = time.time()
        result = extract_batch(batch, known_entities, dry_run=dry_run)
        elapsed = time.time() - start

        if result.get("error"):
            print(f" ERROR ({elapsed:.1f}s): {result['error']}")
            state["stats"]["errors"] += 1
        else:
            store_stats = store_extraction_results(result, dry_run=dry_run)
            n_ent = len(result.get("entities", []))
            n_rel = len(result.get("relationships", []))
            print(f" {n_ent} entities, {n_rel} relationships ({elapsed:.1f}s)")

            state["stats"]["entities_created"] += store_stats["entities_created"]
            state["stats"]["relationships_created"] += store_stats["relationships_created"]
            state["stats"]["entities_skipped"] += store_stats["entities_skipped"]

        state["stats"]["batches_processed"] += 1
        state["stats"]["memories_processed"] += len(batch)
        state["processed_ids"].extend(m["id"] for m in batch)
        state["batch_index"] = batch_num

        # Checkpoint after each batch
        if not dry_run:
            save_checkpoint(state)

        if verbose:
            for ent in result.get("entities", []):
                print(f"         + {ent['name']} ({ent.get('type', '?')}, "
                      f"conf={ent.get('confidence', 0):.1f})")
            for rel in result.get("relationships", []):
                print(f"         ~ {rel['subject']} -{rel['predicate']}-> {rel['object']}")

    # Summary
    stats = state["stats"]
    print()
    print("  " + "-" * 40)
    print(f"  Memories processed: {stats['memories_processed']}")
    print(f"  Entities created:   {stats['entities_created']}")
    print(f"  Entities skipped:   {stats['entities_skipped']}")
    print(f"  Relationships:      {stats['relationships_created']}")
    print(f"  Errors:             {stats['errors']}")
    print()

    # Clean up checkpoint on successful completion
    if not dry_run and not max_batches and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("  Checkpoint cleaned up (run complete)")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="maasv graph enrichment backfill")
    parser.add_argument("--dry-run", action="store_true", help="Show extractions without writing to DB")
    parser.add_argument("--max-batches", type=int, default=0, help="Max batches to process (0=all)")
    parser.add_argument("--batch-size", type=int, default=10, help="Memories per batch")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed results")
    args = parser.parse_args()

    run_enrichment(
        dry_run=args.dry_run,
        max_batches=args.max_batches,
        batch_size=args.batch_size,
        resume=args.resume,
        verbose=args.verbose,
    )
