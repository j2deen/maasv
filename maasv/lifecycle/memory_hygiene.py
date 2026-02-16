"""
Memory Hygiene - Sleep-Time Cleanup

Runs during idle periods to:
1. Deduplicate memories with high embedding similarity (>threshold)
2. Prune stale, low-confidence memories
3. Consolidate clusters of related memories

All operations are audited and can be run in dry-run mode.
"""

import logging
import json
import shutil
from datetime import datetime, timedelta
from typing import Callable, Optional
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger("maasv.lifecycle.memory_hygiene")


@dataclass
class HygieneStats:
    """Statistics from a hygiene run."""
    duplicates_found: int = 0
    duplicates_merged: int = 0
    stale_found: int = 0
    stale_pruned: int = 0
    clusters_found: int = 0
    clusters_consolidated: int = 0
    errors: list = field(default_factory=list)
    dry_run: bool = True
    backup_path: Optional[str] = None
    started_at: str = ""
    completed_at: str = ""


def run_memory_hygiene_job(data: dict, cancel_check: Callable[[], bool]) -> dict:
    """
    Run a memory hygiene job.

    Args:
        data: {
            "mode": "full" | "incremental",
            "dry_run": bool (default True),
            "dedup": bool (default True),
            "prune": bool (default True),
            "consolidate": bool (default False - expensive)
        }
        cancel_check: Function to check if job should stop

    Returns:
        {"stats": HygieneStats as dict, "cancelled": bool}
    """
    mode = data.get("mode", "incremental")
    dry_run = data.get("dry_run", True)
    do_dedup = data.get("dedup", True)
    do_prune = data.get("prune", True)
    do_consolidate = data.get("consolidate", False)

    stats = HygieneStats(
        dry_run=dry_run,
        started_at=datetime.now().isoformat()
    )

    if cancel_check():
        return {"stats": _stats_to_dict(stats), "cancelled": True}

    # Create backup before any destructive operations
    if not dry_run:
        backup_path = _create_backup()
        if backup_path:
            stats.backup_path = str(backup_path)
            logger.info(f"[MemoryHygiene] Created backup: {backup_path}")
        else:
            logger.error("[MemoryHygiene] Failed to create backup, aborting")
            stats.errors.append("Failed to create backup")
            return {"stats": _stats_to_dict(stats), "cancelled": False}

    # Step 1: Deduplicate
    if do_dedup and not cancel_check():
        try:
            dedup_stats = _deduplicate_memories(dry_run, cancel_check)
            stats.duplicates_found = dedup_stats["found"]
            stats.duplicates_merged = dedup_stats["merged"]
            logger.info(f"[MemoryHygiene] Dedup: found {stats.duplicates_found}, merged {stats.duplicates_merged}")
        except Exception as e:
            logger.error(f"[MemoryHygiene] Dedup failed: {e}", exc_info=True)
            stats.errors.append(f"Dedup error: {e}")

    if cancel_check():
        stats.completed_at = datetime.now().isoformat()
        return {"stats": _stats_to_dict(stats), "cancelled": True}

    # Step 2: Prune stale
    if do_prune and not cancel_check():
        try:
            prune_stats = _prune_stale_memories(dry_run, cancel_check)
            stats.stale_found = prune_stats["found"]
            stats.stale_pruned = prune_stats["pruned"]
            logger.info(f"[MemoryHygiene] Prune: found {stats.stale_found}, pruned {stats.stale_pruned}")
        except Exception as e:
            logger.error(f"[MemoryHygiene] Prune failed: {e}", exc_info=True)
            stats.errors.append(f"Prune error: {e}")

    if cancel_check():
        stats.completed_at = datetime.now().isoformat()
        return {"stats": _stats_to_dict(stats), "cancelled": True}

    # Step 3: Consolidate (only in full mode, expensive)
    if do_consolidate and mode == "full" and not cancel_check():
        try:
            consolidate_stats = _consolidate_clusters(dry_run, cancel_check)
            stats.clusters_found = consolidate_stats["found"]
            stats.clusters_consolidated = consolidate_stats["consolidated"]
            logger.info(f"[MemoryHygiene] Consolidate: found {stats.clusters_found}, consolidated {stats.clusters_consolidated}")
        except Exception as e:
            logger.error(f"[MemoryHygiene] Consolidate failed: {e}", exc_info=True)
            stats.errors.append(f"Consolidate error: {e}")

    stats.completed_at = datetime.now().isoformat()
    _log_hygiene_run(stats)

    return {"stats": _stats_to_dict(stats), "cancelled": False}


def _create_backup() -> Optional[Path]:
    """Create a backup of the database before modifications, retaining only the last N."""
    import maasv

    config = maasv.get_config()

    if not config.backup_dir:
        logger.warning("[MemoryHygiene] No backup_dir configured, skipping backup")
        return None

    try:
        backup_dir = config.backup_dir / "memory_hygiene"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"pre_hygiene_{timestamp}.db"
        shutil.copy2(config.db_path, backup_path)

        # Enforce retention: keep only the last N backups
        _enforce_backup_retention(backup_dir, config.max_hygiene_backups)

        return backup_path
    except Exception as e:
        logger.error(f"[MemoryHygiene] Backup failed: {e}")
        return None


def _enforce_backup_retention(backup_dir: Path, max_backups: int):
    """Delete old hygiene backups, keeping only the most recent max_backups."""
    try:
        backups = sorted(
            backup_dir.glob("pre_hygiene_*.db"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        for old_backup in backups[max_backups:]:
            old_backup.unlink()
            logger.info(f"[MemoryHygiene] Removed old backup: {old_backup.name}")

    except Exception as e:
        logger.warning(f"[MemoryHygiene] Backup retention cleanup failed: {e}")


def _is_protected(memory: dict) -> bool:
    """Check if a memory is protected from deletion."""
    import maasv

    config = maasv.get_config()

    category = memory.get("category", "").lower()
    subject = (memory.get("subject") or "").lower()
    confidence = memory.get("confidence", 1.0)

    # Never delete high-confidence memories
    if confidence >= 0.9:
        return True

    # Never delete protected categories
    if category in config.protected_categories:
        return True

    # Never delete memories about protected subjects
    if subject in config.protected_subjects:
        return True

    return False


def _deduplicate_memories(dry_run: bool, cancel_check: Callable[[], bool]) -> dict:
    """
    Find and merge duplicate memories using embedding similarity.

    Duplicates are memories with:
    - Same category
    - Embedding similarity > threshold

    When merging:
    - Keep the memory with highest confidence
    - Combine metadata
    - Mark other as superseded

    Uses pre-computed embeddings from memory_vectors table for efficiency.
    """
    import maasv
    from maasv.core.store import get_db

    config = maasv.get_config()
    similarity_threshold = config.similarity_threshold

    stats = {"found": 0, "merged": 0, "pairs": []}
    db = get_db()

    try:
        # Get all active memories with their categories
        memories = db.execute("""
            SELECT id, content, category, subject, confidence, metadata, created_at
            FROM memories
            WHERE superseded_by IS NULL
            ORDER BY category, created_at DESC
        """).fetchall()

        memories = [dict(m) for m in memories]
        logger.info(f"[MemoryHygiene] Checking {len(memories)} memories for duplicates")

        # Group by category to reduce comparison space
        by_category = {}
        for mem in memories:
            cat = mem["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(mem)

        duplicates_to_merge = []
        seen_pairs = set()

        for category, mems in by_category.items():
            if cancel_check():
                break

            if len(mems) < 2:
                continue

            # For each memory, find similar ones via vector search using existing embeddings
            for i, mem in enumerate(mems):
                if cancel_check():
                    break
                if _is_protected(mem):
                    continue

                # Get the existing embedding for this memory
                embedding_row = db.execute("""
                    SELECT embedding FROM memory_vectors WHERE id = ?
                """, (mem["id"],)).fetchone()

                if not embedding_row:
                    continue  # No embedding, skip

                # Find similar memories in same category using the existing embedding
                similar = db.execute("""
                    SELECT v.id, v.distance
                    FROM memory_vectors v
                    JOIN memories m ON v.id = m.id
                    WHERE m.superseded_by IS NULL
                    AND m.category = ?
                    AND m.id != ?
                    AND v.embedding MATCH ?
                    AND k = 10
                    ORDER BY distance
                """, (category, mem["id"], embedding_row["embedding"])).fetchall()

                for row in similar:
                    if cancel_check():
                        break

                    # sqlite-vec returns L2 distance, convert to similarity
                    # For normalized embeddings, similarity ≈ 1 - (distance^2 / 2)
                    distance = row["distance"]
                    similarity = 1 - (distance ** 2 / 2)

                    if similarity > similarity_threshold:
                        other_id = row["id"]
                        # Avoid duplicate pairs
                        pair_key = tuple(sorted([mem["id"], other_id]))
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            duplicates_to_merge.append((mem["id"], other_id, similarity))
                            stats["found"] += 1

        logger.info(f"[MemoryHygiene] Found {len(duplicates_to_merge)} duplicate pairs")

        # Merge duplicates
        for mem1_id, mem2_id, similarity in duplicates_to_merge:
            if cancel_check():
                break

            # Get both memories
            mem1 = db.execute("SELECT * FROM memories WHERE id = ?", (mem1_id,)).fetchone()
            mem2 = db.execute("SELECT * FROM memories WHERE id = ?", (mem2_id,)).fetchone()

            if not mem1 or not mem2:
                continue

            mem1 = dict(mem1)
            mem2 = dict(mem2)

            # Skip if either is protected
            if _is_protected(mem1) or _is_protected(mem2):
                continue

            # Keep the one with higher confidence (or newer if equal)
            if mem1.get("confidence", 1.0) >= mem2.get("confidence", 1.0):
                keep, remove = mem1, mem2
            else:
                keep, remove = mem2, mem1

            stats["pairs"].append({
                "keep": keep["id"],
                "remove": remove["id"],
                "similarity": similarity,
                "keep_content": keep["content"][:100],
                "remove_content": remove["content"][:100]
            })

            if not dry_run:
                # Mark the duplicate as superseded
                db.execute("""
                    UPDATE memories
                    SET superseded_by = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (keep["id"], remove["id"]))

                # Merge metadata if present
                keep_meta = json.loads(keep["metadata"]) if keep.get("metadata") else {}
                remove_meta = json.loads(remove["metadata"]) if remove.get("metadata") else {}
                if remove_meta:
                    merged_meta = {**remove_meta, **keep_meta}  # keep wins on conflicts
                    db.execute("""
                        UPDATE memories
                        SET metadata = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (json.dumps(merged_meta), keep["id"]))

                stats["merged"] += 1

        if not dry_run:
            db.commit()

    finally:
        db.close()

    return stats


def _prune_stale_memories(dry_run: bool, cancel_check: Callable[[], bool]) -> dict:
    """
    Remove stale, low-value memories.

    Candidates for pruning:
    - Confidence < min_confidence_threshold
    - Created > stale_days ago
    - Category not in protected list
    - Subject not in protected list
    """
    import maasv
    from maasv.core.store import get_db

    config = maasv.get_config()

    stats = {"found": 0, "pruned": 0, "candidates": []}
    db = get_db()

    try:
        cutoff_date = (datetime.now() - timedelta(days=config.stale_days)).isoformat()

        # Find stale, low-confidence memories
        candidates = db.execute("""
            SELECT id, content, category, subject, confidence, created_at
            FROM memories
            WHERE superseded_by IS NULL
            AND confidence < ?
            AND created_at < ?
        """, (config.min_confidence_threshold, cutoff_date)).fetchall()

        candidates = [dict(c) for c in candidates]
        logger.info(f"[MemoryHygiene] Found {len(candidates)} prune candidates")

        for mem in candidates:
            if cancel_check():
                break

            if _is_protected(mem):
                continue

            stats["found"] += 1
            stats["candidates"].append({
                "id": mem["id"],
                "content": mem["content"][:100],
                "category": mem["category"],
                "confidence": mem["confidence"],
                "age_days": (datetime.now() - datetime.fromisoformat(mem["created_at"])).days
            })

            if not dry_run:
                # Actually delete
                db.execute("DELETE FROM memory_vectors WHERE id = ?", (mem["id"],))
                db.execute("DELETE FROM memories WHERE id = ?", (mem["id"],))
                stats["pruned"] += 1

        if not dry_run:
            db.commit()

    finally:
        db.close()

    return stats


def _consolidate_clusters(dry_run: bool, cancel_check: Callable[[], bool]) -> dict:
    """
    Find clusters of related memories and consolidate into stronger single memories.

    This is expensive (O(n^2) embedding comparisons) so only runs in full mode.
    Clusters are memories with:
    - Same subject
    - Similarity > cluster_similarity threshold
    """
    import maasv
    from maasv.core.store import get_db, store_memory, get_embedding

    config = maasv.get_config()

    stats = {"found": 0, "consolidated": 0, "clusters": []}
    db = get_db()

    try:
        # Get memories grouped by subject (only those with subjects)
        memories = db.execute("""
            SELECT id, content, category, subject, confidence, metadata, created_at
            FROM memories
            WHERE superseded_by IS NULL
            AND subject IS NOT NULL
            AND subject != ''
            ORDER BY subject, created_at DESC
        """).fetchall()

        memories = [dict(m) for m in memories]

        # Group by subject
        by_subject = {}
        for mem in memories:
            subj = mem["subject"].lower()
            if subj not in by_subject:
                by_subject[subj] = []
            by_subject[subj].append(mem)

        for subject, mems in by_subject.items():
            if cancel_check():
                break

            if len(mems) < 3:  # Need at least 3 to form a meaningful cluster
                continue

            # Find clusters using vector similarity
            # Simple greedy clustering
            clusters = []
            used = set()

            for mem in mems:
                if cancel_check():
                    break
                if mem["id"] in used:
                    continue

                cluster = [mem]
                used.add(mem["id"])

                query_embedding = get_embedding(mem["content"])

                for other in mems:
                    if other["id"] in used:
                        continue

                    # Check similarity
                    other_embedding = get_embedding(other["content"])
                    # Cosine similarity (assuming normalized)
                    similarity = sum(a * b for a, b in zip(query_embedding, other_embedding))

                    if similarity > config.cluster_similarity:
                        cluster.append(other)
                        used.add(other["id"])

                if len(cluster) >= 3:
                    clusters.append(cluster)
                    stats["found"] += 1

            # Consolidate clusters
            for cluster in clusters:
                if cancel_check():
                    break

                # Create consolidated content
                contents = [m["content"] for m in cluster]
                consolidated_content = _summarize_cluster(contents, subject)

                if not consolidated_content:
                    continue

                # Get highest confidence
                max_confidence = max(m.get("confidence", 1.0) for m in cluster)
                category = cluster[0]["category"]

                stats["clusters"].append({
                    "subject": subject,
                    "count": len(cluster),
                    "ids": [m["id"] for m in cluster],
                    "consolidated": consolidated_content[:200]
                })

                if not dry_run:
                    # Create new consolidated memory
                    new_id = store_memory(
                        content=consolidated_content,
                        category=category,
                        subject=subject.title(),
                        source="consolidation",
                        confidence=max_confidence,
                        metadata={"consolidated_from": [m["id"] for m in cluster]}
                    )

                    # Mark old ones as superseded (unless protected)
                    for mem in cluster:
                        if not _is_protected(mem):
                            db.execute("""
                                UPDATE memories
                                SET superseded_by = ?, updated_at = CURRENT_TIMESTAMP
                                WHERE id = ?
                            """, (new_id, mem["id"]))

                    stats["consolidated"] += 1

        if not dry_run:
            db.commit()

    finally:
        db.close()

    return stats


def _summarize_cluster(contents: list[str], subject: str) -> Optional[str]:
    """
    Create a summary of clustered memories.

    For now, just concatenate unique facts. Could use LLM for smarter summarization.
    """
    # Deduplicate very similar content
    unique = []
    for content in contents:
        content_lower = content.lower().strip()
        is_dup = False
        for existing in unique:
            if content_lower in existing.lower() or existing.lower() in content_lower:
                is_dup = True
                break
        if not is_dup:
            unique.append(content)

    if len(unique) == 1:
        return unique[0]

    # Simple concatenation for now
    return f"[Consolidated] About {subject}: " + " | ".join(unique[:5])


def _log_hygiene_run(stats: HygieneStats):
    """Log hygiene run results for audit."""
    import maasv

    config = maasv.get_config()
    log_path = config.hygiene_log_path

    if not log_path:
        logger.debug("[MemoryHygiene] No hygiene_log_path configured, skipping log file")
        return

    try:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if log_path.exists():
            with open(log_path) as f:
                log = json.load(f)
        else:
            log = {"runs": []}

        log["runs"].append(_stats_to_dict(stats))

        # Keep last 100 runs
        log["runs"] = log["runs"][-100:]

        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

    except Exception as e:
        logger.error(f"[MemoryHygiene] Failed to log run: {e}")


def _stats_to_dict(stats: HygieneStats) -> dict:
    """Convert HygieneStats to dict for JSON serialization."""
    return {
        "duplicates_found": stats.duplicates_found,
        "duplicates_merged": stats.duplicates_merged,
        "stale_found": stats.stale_found,
        "stale_pruned": stats.stale_pruned,
        "clusters_found": stats.clusters_found,
        "clusters_consolidated": stats.clusters_consolidated,
        "errors": stats.errors,
        "dry_run": stats.dry_run,
        "backup_path": stats.backup_path,
        "started_at": stats.started_at,
        "completed_at": stats.completed_at
    }


# Convenience function for manual runs
def run_hygiene(
    mode: str = "incremental",
    dry_run: bool = True,
    dedup: bool = True,
    prune: bool = True,
    consolidate: bool = False
) -> dict:
    """
    Run memory hygiene manually (not as a sleep job).

    Args:
        mode: "incremental" or "full"
        dry_run: If True, only report what would be done
        dedup: Run deduplication
        prune: Run stale pruning
        consolidate: Run cluster consolidation (expensive)

    Returns:
        Stats dict
    """
    return run_memory_hygiene_job(
        data={
            "mode": mode,
            "dry_run": dry_run,
            "dedup": dedup,
            "prune": prune,
            "consolidate": consolidate
        },
        cancel_check=lambda: False
    )
