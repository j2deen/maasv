#!/usr/bin/env python3
"""
maasv Entity Type Cleanup

Reclassifies compound/inconsistent entity types and removes garbage entities.
Compound types like "person + place" are artifacts of LLM hedging during extraction.

Usage (from the doris project directory, with its venv active):
    cd /Users/macmini/Projects/doris
    python /Users/macmini/Projects/maasv/scripts/cleanup_entity_types.py

Options:
    --dry-run    Show what would be changed without modifying the DB
    -v           Verbose: show each entity being fixed
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, "/Users/macmini/Projects/maasv")
sys.path.insert(0, "/Users/macmini/Projects/doris")
os.chdir("/Users/macmini/Projects/doris")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("cleanup_entity_types")

# Maps bad entity_type → corrected type (None = delete the entity)
TYPE_FIXES = {
    "person + place": "person",
    "person, place": "person",
    "person and place": "person",
    "person/event": "person",
    "event/condition": "event",
    "event/date reference": "event",
    "event/place": "event",
    "place/program": "place",
    "project/system": "project",
    "news event": "event",
    "weather event": "event",
    "weather_event": "event",
    "people": "person",
    "restaurant": "place",
    "N/A": None,  # garbage entity — delete
}


def find_entities_to_fix(db) -> list[dict]:
    """Find all entities with compound or inconsistent types."""
    # Build a case-insensitive check for all known bad types
    entities = []
    for bad_type in TYPE_FIXES:
        rows = db.execute(
            "SELECT id, name, entity_type, canonical_name, metadata FROM entities WHERE LOWER(entity_type) = LOWER(?)",
            (bad_type,)
        ).fetchall()
        entities.extend([dict(r) for r in rows])

    # Also find any remaining compound types we might have missed
    extra = db.execute("""
        SELECT id, name, entity_type, canonical_name, metadata FROM entities
        WHERE entity_type LIKE '%+%'
        OR entity_type LIKE '%/%'
        OR entity_type LIKE '%, %'
        OR entity_type LIKE '% and %'
    """).fetchall()

    seen_ids = {e["id"] for e in entities}
    for row in extra:
        r = dict(row)
        if r["id"] not in seen_ids:
            entities.append(r)
            logger.warning(f"  Found unmapped compound type: {r['entity_type']!r} on entity {r['name']!r}")

    return entities


def get_relationship_count(db, entity_id: str) -> int:
    """Count active relationships for an entity (as subject or object)."""
    row = db.execute("""
        SELECT COUNT(*) as cnt FROM relationships
        WHERE valid_to IS NULL
        AND (subject_id = ? OR object_id = ?)
    """, (entity_id, entity_id)).fetchone()
    return row["cnt"]


def fix_entity_type(db, entity: dict, dry_run: bool, verbose: bool) -> str:
    """
    Fix one entity's type. Returns action taken: 'updated', 'deleted', 'skipped'.
    """
    bad_type = entity["entity_type"].lower()
    # Find matching fix (case-insensitive)
    new_type = None
    matched = False
    for key, val in TYPE_FIXES.items():
        if key.lower() == bad_type:
            new_type = val
            matched = True
            break

    if not matched:
        if verbose:
            logger.info(f"  SKIP (unmapped): {entity['name']!r} type={entity['entity_type']!r}")
        return "skipped"

    if new_type is None:
        # Delete this entity
        rel_count = get_relationship_count(db, entity["id"])
        if verbose:
            logger.info(f"  DELETE: {entity['name']!r} type={entity['entity_type']!r} ({rel_count} relationships)")

        if not dry_run:
            # Delete orphaned relationships first
            db.execute(
                "DELETE FROM relationships WHERE subject_id = ? OR object_id = ?",
                (entity["id"], entity["id"])
            )
            db.execute("DELETE FROM entities WHERE id = ?", (entity["id"],))

        return "deleted"

    # Update entity_type
    if verbose:
        logger.info(f"  UPDATE: {entity['name']!r} type={entity['entity_type']!r} → {new_type}")

    if not dry_run:
        db.execute(
            "UPDATE entities SET entity_type = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (new_type, entity["id"])
        )

    return "updated"


def main():
    parser = argparse.ArgumentParser(description="Clean up compound/inconsistent entity types")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without modifying DB")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show each entity being fixed")
    args = parser.parse_args()

    from maasv_bridge import init_maasv
    init_maasv()

    from maasv.core.store import get_db

    db = get_db()

    # Stats before
    total_entities = db.execute("SELECT count(*) FROM entities").fetchone()[0]
    type_dist_before = db.execute(
        "SELECT entity_type, count(*) as cnt FROM entities GROUP BY entity_type ORDER BY cnt DESC"
    ).fetchall()

    logger.info(f"=== Entity Type Cleanup ===")
    logger.info(f"Database: {db.execute('PRAGMA database_list').fetchone()[2]}")
    logger.info(f"Total entities: {total_entities}")
    logger.info(f"")
    logger.info(f"Type distribution before:")
    for row in type_dist_before:
        logger.info(f"  {row['entity_type']}: {row['cnt']}")

    # Find entities to fix
    entities = find_entities_to_fix(db)
    logger.info(f"\nFound {len(entities)} entities with compound/inconsistent types")

    if not entities:
        logger.info("Nothing to fix!")
        db.close()
        return

    # Fix each entity
    updated = 0
    deleted = 0
    deleted_rels = 0
    skipped = 0

    for entity in entities:
        # Count relationships before potential deletion (for stats)
        rel_count = get_relationship_count(db, entity["id"]) if TYPE_FIXES.get(entity["entity_type"].lower()) is None else 0

        action = fix_entity_type(db, entity, args.dry_run, args.verbose)
        if action == "updated":
            updated += 1
        elif action == "deleted":
            deleted += 1
            deleted_rels += rel_count
        elif action == "skipped":
            skipped += 1

    if not args.dry_run:
        db.commit()

    # Run relationship dedup (type changes may create duplicate entities that merge later)
    if not args.dry_run:
        logger.info("\nRunning relationship dedup after type fixes...")
        from maasv.lifecycle.memory_hygiene import _deduplicate_relationships
        rel_stats = _deduplicate_relationships(dry_run=False)
        logger.info(f"  Relationship dedup: found {rel_stats['found']} groups, removed {rel_stats['removed']}")

    # Stats after
    action_word = "Would" if args.dry_run else "Did"
    logger.info(f"\n=== Summary ===")
    logger.info(f"{action_word} update: {updated} entities")
    logger.info(f"{action_word} delete: {deleted} entities ({deleted_rels} orphaned relationships)")
    logger.info(f"Skipped (unmapped): {skipped}")

    if not args.dry_run:
        total_after = db.execute("SELECT count(*) FROM entities").fetchone()[0]
        type_dist_after = db.execute(
            "SELECT entity_type, count(*) as cnt FROM entities GROUP BY entity_type ORDER BY cnt DESC"
        ).fetchall()
        logger.info(f"\nTotal entities after: {total_after}")
        logger.info(f"Type distribution after:")
        for row in type_dist_after:
            logger.info(f"  {row['entity_type']}: {row['cnt']}")

        # Verify no compound types remain
        remaining = db.execute("""
            SELECT entity_type, count(*) as cnt FROM entities
            WHERE entity_type LIKE '%+%'
            OR entity_type LIKE '%/%'
            OR entity_type LIKE '%, %'
            OR entity_type LIKE '% and %'
            OR LOWER(entity_type) = 'n/a'
            GROUP BY entity_type
        """).fetchall()

        if remaining:
            logger.warning("REMAINING compound/bad types:")
            for row in remaining:
                logger.warning(f"  {row['entity_type']}: {row['cnt']}")
        else:
            logger.info("All compound/bad types cleaned up!")

    db.close()


if __name__ == "__main__":
    main()
