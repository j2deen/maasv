"""
maasv Memory Store (slim)

Memory CRUD operations: store, supersede, get, delete, update metadata.
Database infra lives in db.py, retrieval in retrieval.py, graph in graph.py.
"""

import logging
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from maasv.core.db import _db, get_embedding, serialize_embedding

logger = logging.getLogger(__name__)


def store_memory(
    content: str,
    category: str,
    subject: Optional[str] = None,
    source: str = "manual",
    confidence: float = 1.0,
    metadata: Optional[dict] = None,
    dedup_threshold: float = 0.05
) -> str:
    """
    Store a new memory with embedding, with dedup check.

    Args:
        content: The fact or memory to store
        category: Type of memory (family, preference, project, decision, etc.)
        subject: Who/what this is about (e.g., "John", "ProjectX")
        source: Where this came from (manual, conversation, extracted)
        confidence: How confident we are (0.0-1.0)
        metadata: Additional structured data
        dedup_threshold: Vector distance below which a memory is considered duplicate

    Returns:
        Memory ID (existing ID if duplicate found)
    """
    # Compute embedding first (needed for both dedup check and storage)
    embedding = get_embedding(content)

    with _db() as db:
        # Dedup check: find near-duplicate via vector similarity
        try:
            rows = db.execute(
                """
                SELECT v.id, m.content, m.category, distance
                FROM memory_vectors v
                JOIN memories m ON v.id = m.id
                WHERE m.superseded_by IS NULL
                AND v.embedding MATCH ?
                AND k = 3
                ORDER BY distance
                """,
                (serialize_embedding(embedding),)
            ).fetchall()

            for row in rows:
                if row['distance'] < dedup_threshold and row['category'] == category:
                    logger.info(
                        f"Dedup: skipping store, near-duplicate found: {row['id']} (dist={row['distance']:.4f})"
                    )
                    return row['id']
        except Exception:
            logger.debug("Dedup check failed, proceeding with store", exc_info=True)

        # No duplicate found — insert
        memory_id = f"mem_{uuid.uuid4().hex[:12]}"

        db.execute("""
            INSERT INTO memories (id, content, category, subject, source, confidence, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id,
            content,
            category,
            subject,
            source,
            confidence,
            json.dumps(metadata) if metadata else None
        ))

        db.execute(
            "INSERT INTO memory_vectors (id, embedding) VALUES (?, ?)",
            (memory_id, serialize_embedding(embedding))
        )

        db.commit()

    return memory_id


def supersede_memory(old_id: str, new_content: str, source: str = "correction") -> str:
    """Mark an old memory as superseded and create a new one."""
    with _db() as db:
        old = db.execute(
            "SELECT category, subject, metadata FROM memories WHERE id = ?",
            (old_id,)
        ).fetchone()

        if not old:
            raise ValueError(f"Memory {old_id} not found")

    new_id = store_memory(
        content=new_content,
        category=old['category'],
        subject=old['subject'],
        source=source,
        metadata=json.loads(old['metadata']) if old['metadata'] else None
    )

    with _db() as db:
        db.execute(
            "UPDATE memories SET superseded_by = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (new_id, old_id)
        )
        db.commit()

    return new_id


def get_all_active(category: Optional[str] = None) -> list[dict]:
    """Get all active (non-superseded) memories."""
    query = "SELECT * FROM memories WHERE superseded_by IS NULL"
    params = []

    if category:
        query += " AND category = ?"
        params.append(category)

    query += " ORDER BY created_at DESC"

    with _db() as db:
        rows = db.execute(query, params).fetchall()

    return [dict(row) for row in rows]


def get_recent_memories(
    hours: int = 48,
    categories: Optional[list[str]] = None,
    limit: int = 50
) -> list[dict]:
    """Get recent memories from the last N hours."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

    query = "SELECT * FROM memories WHERE superseded_by IS NULL AND created_at >= ?"
    params: list = [cutoff]

    if categories:
        placeholders = ",".join("?" * len(categories))
        query += f" AND category IN ({placeholders})"
        params.extend(categories)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with _db() as db:
        rows = db.execute(query, params).fetchall()

    return [dict(row) for row in rows]


def delete_memory(memory_id: str) -> bool:
    """Permanently delete a memory."""
    with _db() as db:
        db.execute("DELETE FROM memory_vectors WHERE id = ?", (memory_id,))
        cursor = db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        deleted = cursor.rowcount > 0
        db.commit()

    return deleted


def update_memory_metadata(memory_id: str, metadata_updates: dict) -> bool:
    """Update metadata for an existing memory (merge, not replace)."""
    with _db() as db:
        row = db.execute(
            "SELECT metadata FROM memories WHERE id = ?",
            (memory_id,)
        ).fetchone()

        if not row:
            return False

        current = json.loads(row['metadata']) if row['metadata'] else {}
        current.update(metadata_updates)

        db.execute(
            "UPDATE memories SET metadata = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (json.dumps(current), memory_id)
        )
        db.commit()
    return True
