"""
Memory Evolution - Sleep-Time Compute (A-MEM style)

New memories retroactively enrich old ones. For each memory stored since the
last run, find semantically related older memories via vector KNN and record
bidirectional links in metadata["related_ids"]. The memory network densifies
during idle time: what arrives today updates how yesterday's knowledge is
connected, without any LLM call.

Optionally (evolve_llm_refresh=True, requires an LLMProvider), linked older
memories get their metadata["tags"] refreshed so new context can reframe old
facts — the full A-MEM "memory evolution" loop.

Watermark lives in db_meta ("evolve_watermark"): each run only processes
memories created after the previous run.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Callable, Optional

logger = logging.getLogger("maev.lifecycle.evolve")

WATERMARK_KEY = "evolve_watermark"


def _get_watermark(db) -> Optional[tuple[str, str]]:
    """Composite (created_at, id) watermark. created_at has second granularity
    (SQLite CURRENT_TIMESTAMP), so ties are common — a timestamp-only watermark
    with a strict '>' filter would skip same-second memories forever."""
    row = db.execute(
        "SELECT value FROM db_meta WHERE key = ?", (WATERMARK_KEY,)
    ).fetchone()
    if not row:
        return None
    try:
        parsed = json.loads(row["value"])
        if isinstance(parsed, list) and len(parsed) == 2:
            return (parsed[0], parsed[1])
    except (json.JSONDecodeError, TypeError):
        pass
    # Legacy plain-timestamp watermark: id "" sorts before every real id,
    # so same-second rows are (re)considered rather than skipped
    return (row["value"], "")


def _set_watermark(db, created_at: str, mem_id: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    db.execute("""
        INSERT INTO db_meta (key, value, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
    """, (WATERMARK_KEY, json.dumps([created_at, mem_id]), now, now))


def _load_metadata(raw: Optional[str]) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _add_links(db, mem_id: str, new_ids: list[str], max_links: int) -> bool:
    """Merge new_ids into metadata['related_ids'] (bounded FIFO). True if changed."""
    row = db.execute(
        "SELECT metadata FROM memories WHERE id = ? AND superseded_by IS NULL",
        (mem_id,)
    ).fetchone()
    if row is None:
        return False

    meta = _load_metadata(row["metadata"])
    existing = meta.get("related_ids")
    existing = list(existing) if isinstance(existing, list) else []
    changed = False
    for rid in new_ids:
        if rid != mem_id and rid not in existing:
            existing.append(rid)
            changed = True
    if not changed:
        return False

    meta["related_ids"] = existing[-max_links:]
    db.execute(
        "UPDATE memories SET metadata = ? WHERE id = ?",
        (json.dumps(meta), mem_id)
    )
    return True


def _refresh_tags(db, memory: dict, new_content: str, model: str) -> bool:
    """LLM attribute refresh: re-tag an older memory in light of new context."""
    import maev
    from maev.utils import parse_llm_json

    try:
        llm = maev.get_llm()
    except RuntimeError:
        return False

    prompt = (
        "An older memory is being re-evaluated in light of a newly stored, related memory.\n"
        f"Older memory: {memory['content']}\n"
        f"New related memory: {new_content}\n"
        'Return JSON only: {"tags": ["3-6 short topical tags for the OLDER memory, '
        'reflecting any reframing the new context suggests"]}'
    )
    try:
        response = llm.call(
            [{"role": "user", "content": prompt}],
            model=model, max_tokens=200, source="memory-evolve",
        )
    except Exception:
        logger.debug("Evolve tag refresh LLM call failed", exc_info=True)
        return False

    parsed = parse_llm_json(response)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("tags"), list):
        return False
    tags = [str(t)[:40] for t in parsed["tags"][:6] if t]
    if not tags:
        return False

    # Re-read metadata from the DB: the row dict `memory` was captured by the
    # KNN SELECT before _add_links wrote related_ids — merging into that stale
    # copy would erase the backlink (and any other concurrent writer's keys).
    row = db.execute(
        "SELECT metadata FROM memories WHERE id = ? AND superseded_by IS NULL",
        (memory["id"],)
    ).fetchone()
    if row is None:
        return False
    meta = _load_metadata(row["metadata"])
    if meta.get("tags") == tags:
        return False
    meta["tags"] = tags
    db.execute("UPDATE memories SET metadata = ? WHERE id = ?",
               (json.dumps(meta), memory["id"]))
    return True


def run_evolve_job(data: dict, cancel_check: Callable[[], bool]) -> dict:
    """
    Link new memories to related older ones; optionally refresh old tags.

    Args:
        data: {"batch_size": int (optional override)}
        cancel_check: cooperative cancellation

    Returns stats: processed / links_created / tags_refreshed.
    """
    import maev
    from maev.core.db import _db

    config = maev.get_config()
    stats = {"processed": 0, "links_created": 0, "tags_refreshed": 0, "cancelled": False}
    if not config.evolve_enabled:
        return stats

    batch_size = data.get("batch_size") or config.evolve_batch_size
    link_threshold = config.evolve_link_threshold

    with _db() as db:
        watermark = _get_watermark(db)
        params: list = []
        where = "m.superseded_by IS NULL"
        if watermark:
            # Keyset pagination on (created_at, id) — matches the ORDER BY
            where += " AND (m.created_at > ? OR (m.created_at = ? AND m.id > ?))"
            params.extend([watermark[0], watermark[0], watermark[1]])
        params.append(batch_size)

        new_memories = db.execute(f"""
            SELECT m.id, m.content, m.created_at
            FROM memories m
            WHERE {where}
            ORDER BY m.created_at ASC, m.id ASC
            LIMIT ?
        """, params).fetchall()

        if not new_memories:
            return stats

        latest_processed = None
        for mem in new_memories:
            if cancel_check():
                stats["cancelled"] = True
                break
            latest_processed = (mem["created_at"], mem["id"])
            stats["processed"] += 1

            emb_row = db.execute(
                "SELECT embedding FROM memory_vectors WHERE id = ?", (mem["id"],)
            ).fetchone()
            if not emb_row:
                continue

            # KNN over active memories, excluding self; only OLDER neighbors so
            # each pair is linked once (the newer side drives the link).
            try:
                neighbors = db.execute("""
                    SELECT v.id, v.distance, m.content, m.metadata, m.created_at
                    FROM memory_vectors v
                    JOIN memories m ON v.id = m.id
                    WHERE m.superseded_by IS NULL
                    AND m.id != ?
                    AND v.embedding MATCH ?
                    AND k = ?
                    ORDER BY distance
                """, (mem["id"], emb_row["embedding"],
                      config.evolve_max_links)).fetchall()
            except Exception:
                logger.debug("Evolve KNN failed for %s", mem["id"], exc_info=True)
                continue

            related = []
            for row in neighbors:
                similarity = 1 - (row["distance"] ** 2 / 2)
                if similarity < link_threshold:
                    break  # ordered by distance — the rest are weaker
                if (row["created_at"], row["id"]) < (mem["created_at"], mem["id"]):
                    related.append(dict(row))

            if not related:
                continue

            related_ids = [r["id"] for r in related]
            if _add_links(db, mem["id"], related_ids, config.evolve_max_links):
                stats["links_created"] += len(related_ids)
            for r in related:
                _add_links(db, r["id"], [mem["id"]], config.evolve_max_links)

            if config.evolve_llm_refresh:
                for r in related:
                    if cancel_check():
                        stats["cancelled"] = True
                        break
                    if _refresh_tags(db, r, mem["content"], config.review_model):
                        stats["tags_refreshed"] += 1

        if latest_processed is not None:
            _set_watermark(db, latest_processed[0], latest_processed[1])
        db.commit()

    logger.info(
        "[Evolve] processed=%d links=%d tags=%d",
        stats["processed"], stats["links_created"], stats["tags_refreshed"]
    )
    return stats
