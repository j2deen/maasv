"""
Graph Reorganization - Sleep-Time Compute

Optimizes the knowledge graph for faster retrieval:
- Tracks access patterns
- Pre-computes common traversal paths
- Cleans up stale/orphaned data
"""

import logging
import json
from typing import Callable
from datetime import datetime, timedelta

logger = logging.getLogger("maasv.lifecycle.reorganize")


def run_reorganize_job(data: dict, cancel_check: Callable[[], bool]) -> dict:
    """Run a graph reorganization job."""
    mode = data.get("mode", "incremental")
    focus_entities = data.get("focus_entities", [])

    results = {"optimizations": [], "cleaned": 0, "paths_cached": 0}

    if cancel_check():
        return {**results, "cancelled": True}

    _update_access_stats(focus_entities)
    results["optimizations"].append("updated_access_stats")

    if cancel_check():
        return {**results, "cancelled": True}

    paths_cached = _cache_common_paths()
    results["paths_cached"] = paths_cached
    results["optimizations"].append("cached_common_paths")

    if cancel_check():
        return {**results, "cancelled": True}

    if mode == "full":
        cleaned = _cleanup_orphans()
        results["cleaned"] = cleaned
        results["optimizations"].append("cleaned_orphans")

    if cancel_check():
        return {**results, "cancelled": True}

    strengthened = _strengthen_frequent_connections()
    if strengthened:
        results["optimizations"].append("strengthened_connections")

    return results


def _update_access_stats(focus_entities: list[str] = None):
    """Update access statistics for entities."""
    from maasv.core.store import get_db

    try:
        db = get_db()
        try:
            db.execute("ALTER TABLE entities ADD COLUMN access_count INTEGER DEFAULT 0")
        except Exception:
            pass
        try:
            db.execute("ALTER TABLE entities ADD COLUMN last_accessed_at TEXT")
        except Exception:
            pass
        db.commit()
        db.close()
        logger.debug("[Reorganize] Updated access stats schema")
    except Exception as e:
        logger.warning(f"[Reorganize] Failed to update access stats: {e}")


def _cache_common_paths() -> int:
    """Pre-compute and cache common traversal paths."""
    from maasv.core.store import get_db, find_entity_by_name, get_entity_relationships

    cached = 0

    try:
        # Find the primary user entity — use config's known_entities if available
        import maasv
        config = maasv.get_config()

        # Look for the first "person" in known_entities
        primary_person = None
        for name, etype in config.known_entities.items():
            if etype == "person":
                primary_person = name
                break

        if not primary_person:
            logger.debug("[Reorganize] No primary person in known_entities, skipping path caching")
            return 0

        person = find_entity_by_name(primary_person)
        if not person:
            logger.debug(f"[Reorganize] No {primary_person} entity found, skipping path caching")
            return 0

        rels = get_entity_relationships(person["id"], predicate=None, direction="outgoing")

        family_predicates = {"spouse", "child", "married_to", "parent_of", "sibling"}
        family_members = [r for r in rels if r.get("predicate") in family_predicates]
        if family_members:
            _store_cached_path("primary_family", family_members)
            cached += 1

        project_rels = [r for r in rels if r.get("predicate") == "works_on"]
        if project_rels:
            _store_cached_path("primary_projects", project_rels)
            cached += 1

        logger.info(f"[Reorganize] Cached {cached} common paths")
        return cached

    except Exception as e:
        logger.warning(f"[Reorganize] Failed to cache paths: {e}")
        return 0


def _store_cached_path(path_name: str, relationships: list[dict]):
    """Store a cached path for fast retrieval."""
    from maasv.core.store import get_db

    try:
        db = get_db()
        db.execute("""
            CREATE TABLE IF NOT EXISTS cached_paths (
                name TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                cached_at TEXT NOT NULL,
                expires_at TEXT
            )
        """)

        now = datetime.now()
        expires = now + timedelta(hours=1)

        simplified = []
        for r in relationships:
            simplified.append({
                "id": r.get("id"),
                "predicate": r.get("predicate"),
                "object_id": r.get("object_id"),
                "object_name": r.get("object_name"),
                "object_type": r.get("object_type"),
                "object_value": r.get("object_value")
            })

        db.execute("""
            INSERT OR REPLACE INTO cached_paths (name, data, cached_at, expires_at)
            VALUES (?, ?, ?, ?)
        """, (path_name, json.dumps(simplified), now.isoformat(), expires.isoformat()))

        db.commit()
        db.close()

    except Exception as e:
        logger.warning(f"[Reorganize] Failed to store cached path '{path_name}': {e}")


def get_cached_path(path_name: str) -> list[dict] | None:
    """Retrieve a cached path if still valid. Returns None if not cached or expired."""
    from maasv.core.store import get_db

    try:
        db = get_db()
        row = db.execute(
            "SELECT data, expires_at FROM cached_paths WHERE name = ?",
            (path_name,)
        ).fetchone()
        db.close()

        if not row:
            return None

        expires_at = datetime.fromisoformat(row["expires_at"])
        if datetime.now() > expires_at:
            return None

        return json.loads(row["data"])

    except Exception as e:
        logger.warning(f"[Reorganize] Failed to get cached path '{path_name}': {e}")
        return None


def _cleanup_orphans() -> int:
    """Clean up orphaned entities (no relationships, created >7 days ago)."""
    from maasv.core.store import get_db

    try:
        db = get_db()
        cutoff = (datetime.now() - timedelta(days=7)).isoformat()

        orphans = db.execute("""
            SELECT e.id FROM entities e
            WHERE e.created_at < ?
            AND NOT EXISTS (
                SELECT 1 FROM relationships r
                WHERE r.subject_id = e.id OR r.object_id = e.id
            )
        """, (cutoff,)).fetchall()

        orphan_ids = [row["id"] for row in orphans]

        if orphan_ids:
            placeholders = ",".join("?" * len(orphan_ids))
            db.execute(f"DELETE FROM entities WHERE id IN ({placeholders})", orphan_ids)
            db.commit()
            logger.info(f"[Reorganize] Cleaned {len(orphan_ids)} orphaned entities")

        db.close()
        return len(orphan_ids)

    except Exception as e:
        logger.warning(f"[Reorganize] Failed to cleanup orphans: {e}")
        return 0


def _strengthen_frequent_connections() -> bool:
    """Placeholder for connection strengthening based on co-access patterns."""
    logger.debug("[Reorganize] Connection strengthening not yet implemented")
    return False
