#!/usr/bin/env python3
"""
maasv Relationship Deduplication — One-Time Cleanup

Finds all (subject_id, predicate, object_id) groups with duplicates,
keeps the row with highest confidence per group, deletes the rest.

Usage (from the doris project directory, with its venv active):
    cd /Users/macmini/Projects/doris
    python /Users/macmini/Projects/maasv/scripts/dedup_relationships.py

Options:
    --dry-run    Show what would be deleted without modifying the DB
    -v           Verbose: show each group being deduped
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
logger = logging.getLogger("dedup_relationships")


def find_duplicate_groups(db) -> list[dict]:
    """Find all (subject_id, predicate, object_id) groups with count > 1."""
    rows = db.execute("""
        SELECT subject_id, predicate, object_id, count(*) as cnt
        FROM relationships
        WHERE valid_to IS NULL
        GROUP BY subject_id, predicate, object_id
        HAVING count(*) > 1
        ORDER BY cnt DESC
    """).fetchall()
    return [dict(r) for r in rows]


def dedup_group(db, subject_id: str, predicate: str, object_id: str, dry_run: bool, verbose: bool) -> int:
    """Deduplicate one relationship group. Returns number of rows deleted."""
    # Get all rows in this group
    rows = db.execute("""
        SELECT id, confidence, source, metadata, created_at
        FROM relationships
        WHERE subject_id = ? AND predicate = ? AND object_id = ?
        AND valid_to IS NULL
        ORDER BY confidence DESC, created_at ASC
    """, (subject_id, predicate, object_id)).fetchall()

    if len(rows) < 2:
        return 0

    # Keep the first (highest confidence, oldest for ties)
    keeper = rows[0]
    to_delete = [dict(r)["id"] for r in rows[1:]]

    if verbose:
        logger.info(f"  {subject_id} --{predicate}--> {object_id}: keeping {keeper['id']} (conf={keeper['confidence']}), deleting {len(to_delete)}")

    if not dry_run:
        placeholders = ",".join("?" * len(to_delete))
        db.execute(f"DELETE FROM relationships WHERE id IN ({placeholders})", to_delete)

    return len(to_delete)


def also_dedup_object_value_groups(db, dry_run: bool, verbose: bool) -> int:
    """Dedup groups keyed by (subject_id, predicate, object_value) for attribute relationships."""
    rows = db.execute("""
        SELECT subject_id, predicate, object_value, count(*) as cnt
        FROM relationships
        WHERE valid_to IS NULL AND object_id IS NULL AND object_value IS NOT NULL
        GROUP BY subject_id, predicate, object_value
        HAVING count(*) > 1
        ORDER BY cnt DESC
    """).fetchall()

    deleted = 0
    for group in rows:
        g = dict(group)
        all_rows = db.execute("""
            SELECT id, confidence, created_at
            FROM relationships
            WHERE subject_id = ? AND predicate = ? AND object_value = ?
            AND valid_to IS NULL AND object_id IS NULL
            ORDER BY confidence DESC, created_at ASC
        """, (g["subject_id"], g["predicate"], g["object_value"])).fetchall()

        if len(all_rows) < 2:
            continue

        keeper = all_rows[0]
        to_delete = [dict(r)["id"] for r in all_rows[1:]]

        if verbose:
            logger.info(f"  {g['subject_id']} --{g['predicate']}--> val:{g['object_value']}: keeping {keeper['id']}, deleting {len(to_delete)}")

        if not dry_run:
            placeholders = ",".join("?" * len(to_delete))
            db.execute(f"DELETE FROM relationships WHERE id IN ({placeholders})", to_delete)

        deleted += len(to_delete)

    return deleted


def main():
    parser = argparse.ArgumentParser(description="Deduplicate maasv relationships")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without modifying DB")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show each group being deduped")
    args = parser.parse_args()

    # Initialize maasv with doris config
    from maasv_bridge import init_maasv
    init_maasv()

    from maasv.core.store import get_db

    db = get_db()

    # Stats before
    total_before = db.execute("SELECT count(*) FROM relationships").fetchone()[0]
    active_before = db.execute("SELECT count(*) FROM relationships WHERE valid_to IS NULL").fetchone()[0]
    logger.info(f"Before: {total_before} total relationships ({active_before} active)")

    # Find entity-to-entity duplicate groups
    groups = find_duplicate_groups(db)
    logger.info(f"Found {len(groups)} duplicate groups (entity-to-entity)")

    # Dedup each group
    total_deleted = 0
    for g in groups:
        deleted = dedup_group(db, g["subject_id"], g["predicate"], g["object_id"], args.dry_run, args.verbose)
        total_deleted += deleted

    # Also handle object_value-based relationships
    val_deleted = also_dedup_object_value_groups(db, args.dry_run, args.verbose)
    total_deleted += val_deleted

    if not args.dry_run:
        db.commit()

    # Stats after
    total_after = db.execute("SELECT count(*) FROM relationships").fetchone()[0]
    active_after = db.execute("SELECT count(*) FROM relationships WHERE valid_to IS NULL").fetchone()[0]
    remaining_dupes = db.execute("""
        SELECT count(*) FROM (
            SELECT subject_id, predicate, object_id, count(*) as cnt
            FROM relationships WHERE valid_to IS NULL
            GROUP BY subject_id, predicate, object_id
            HAVING count(*) > 1
        )
    """).fetchone()[0]

    action = "Would delete" if args.dry_run else "Deleted"
    logger.info(f"{action} {total_deleted} duplicate rows")
    logger.info(f"After: {total_after} total relationships ({active_after} active)")
    logger.info(f"Remaining duplicate groups: {remaining_dupes}")

    db.close()


if __name__ == "__main__":
    main()
