"""
maasv Memory Retrieval

3-signal retrieval pipeline (vector + BM25 + graph), RRF fusion,
cross-encoder reranking, diversity-aware selection, and tiered context.
"""

import logging
import math
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

from maasv.core.db import (
    _db,
    get_query_embedding,
    serialize_embedding,
    _record_memory_access,
    _escape_like,
    _sanitize_fts_input,
)

logger = logging.getLogger(__name__)


# ============================================================================
# IMPORTANCE SCORING
# ============================================================================

def _importance_score(
    candidates: list[dict],
    protected: set[str],
    now: datetime,
    vector_distances: dict[str, float],
    bm25_ids: set[str],
    graph_ids: set[str],
    rrf_weight: float = 0.0,
) -> tuple[list[dict], list[dict]]:
    """
    Score candidates by importance-weighted formula. Separates into primary
    (have vector distance) and supplementary (no vector distance) lists,
    both sorted by _imp_score descending.

    Scoring: (1 - distance) + 0.05 * importance * decay * ips_utility + signal term.
    The signal term is rrf_weight * normalized fused RRF score when rrf_weight > 0
    (signal STRENGTH — a rank-1 graph or BM25 hit carries real weight), else the
    legacy flat agreement bonus (signal membership only).
    """
    primary = []
    supplementary = []

    max_rrf = 0.0
    if rrf_weight > 0.0:
        max_rrf = max((m.get("rrf_score") or 0.0 for m in candidates), default=0.0)

    for mem in candidates:
        importance = mem.get('importance') or 0.5
        access_count = mem.get('access_count') or 0
        surfacing_count = mem.get('surfacing_count') or 0

        if mem.get('category') in protected:
            decay_factor = 1.0
        else:
            try:
                created = datetime.fromisoformat(mem['created_at'])
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                days_old = (now - created).total_seconds() / 86400
            except (ValueError, TypeError):
                days_old = 0
            decay_factor = math.exp(-days_old / 180)

        # IPS utility: access_count/surfacing_count measures conversion rate.
        # High ratio = surfaced rarely but used often = genuinely useful.
        # Cold-start fallback uses the old capped formula.
        if surfacing_count > 0:
            ips_utility = math.log(2 + access_count / surfacing_count)
        else:
            ips_utility = math.log(2 + min(access_count, 5))

        distance = vector_distances.get(mem['id'])

        if rrf_weight > 0.0 and max_rrf > 0.0:
            signal_term = rrf_weight * (mem.get("rrf_score") or 0.0) / max_rrf
        else:
            signal_count = 1
            if mem['id'] in bm25_ids:
                signal_count += 1
            if mem['id'] in graph_ids:
                signal_count += 1
            signal_term = (signal_count - 1) * 0.03

        if distance is not None:
            # Vector similarity is the primary signal. Importance, decay, and
            # usage are additive tiebreakers — they influence ordering among
            # close matches but can't override a strong vector match.
            vector_sim = 1.0 - distance
            tiebreaker = 0.05 * importance * decay_factor * ips_utility
            mem['_imp_score'] = vector_sim + tiebreaker + signal_term
            primary.append(mem)
        else:
            # No vector presence — fused-rank strength still earns a real score
            # so a top BM25/graph hit isn't buried behind every vector match.
            mem['_imp_score'] = signal_term + importance * decay_factor * ips_utility * 0.0001
            supplementary.append(mem)

    primary.sort(key=lambda m: m['_imp_score'], reverse=True)
    supplementary.sort(key=lambda m: m['_imp_score'], reverse=True)

    return primary, supplementary


def _redact_text(text: Optional[str]) -> Optional[str]:
    """Apply the configured output-redaction hook to one string (if any)."""
    if text is None:
        return None
    import maasv
    redact = maasv.get_config().redact_output
    if redact is None:
        return text
    try:
        return redact(text)
    except Exception:
        logger.warning("redact_output hook raised; withholding text", exc_info=True)
        return "[redacted]"


def _redact_memories(memories: list[dict]) -> list[dict]:
    """Redact content/subject on COPIES — never mutates cached/stored rows."""
    import maasv
    if maasv.get_config().redact_output is None:
        return memories
    redacted = []
    for mem in memories:
        m = dict(mem)
        m["content"] = _redact_text(m.get("content"))
        m["subject"] = _redact_text(m.get("subject"))
        redacted.append(m)
    return redacted


# ============================================================================
# MULTI-SIGNAL RETRIEVAL HELPERS
# ============================================================================

def _query_to_entity_fts(query: str) -> str:
    """
    Convert a natural-language query to OR-separated FTS5 terms for entity search.

    FTS5 defaults to AND, so "MyApp architecture" requires both words in entity
    names — which misses the "MyApp" entity. Converting to "MyApp OR architecture"
    ensures we find entities matching ANY query term.

    Strips FTS5 special characters and skips short/common words.
    """
    stop_words = {"the", "a", "an", "is", "of", "in", "on", "for", "and", "or", "to", "with"}
    words = re.findall(r'\w+', query)
    terms = [w for w in words if len(w) > 1 and w.lower() not in stop_words]
    if not terms:
        return query
    return " OR ".join(terms)


def _expand_query_from_graph(db, query: str) -> str:
    """
    Expand a query with related entity names from the knowledge graph.

    "MyApp architecture" -> graph says MyApp uses_tech FastAPI ->
    returns "MyApp architecture OR FastAPI"

    This provides redundant coverage with _find_memories_by_graph():
    if graph signal or BM25 fails independently, the other catches it.
    """
    entity_fts_query = _query_to_entity_fts(query)
    try:
        entities = db.execute("""
            SELECT e.id, e.name
            FROM entities_fts f
            JOIN entities e ON f.rowid = e.rowid
            WHERE entities_fts MATCH ?
            LIMIT 5
        """, (entity_fts_query,)).fetchall()
    except Exception:
        return query

    if not entities:
        return query

    entity_ids = [e["id"] for e in entities]
    placeholders = ",".join("?" * len(entity_ids))

    try:
        related = db.execute(f"""
            SELECT DISTINCT e.name
            FROM relationships r
            JOIN entities e ON (
                CASE
                    WHEN r.subject_id IN ({placeholders}) THEN r.object_id
                    ELSE r.subject_id
                END
            ) = e.id
            WHERE (r.subject_id IN ({placeholders}) OR r.object_id IN ({placeholders}))
            AND r.valid_to IS NULL
            LIMIT 10
        """, entity_ids * 3).fetchall()
    except Exception:
        return query

    # Build expanded query: original OR "related term 1" OR "related term 2"
    expansion_terms = []
    for row in related:
        name = row["name"]
        if name:
            clean = re.sub(r'[^\w\s]', '', name).strip()
            if clean and clean.lower() not in query.lower():
                expansion_terms.append(f'"{clean}"')

    if not expansion_terms:
        return query

    # FTS5 OR syntax
    return query + " OR " + " OR ".join(expansion_terms)


def _find_memories_by_bm25(db, query: str, limit: int = 50) -> list[dict]:
    """
    Return memories ranked by BM25 relevance from the FTS5 index.

    Uses bm25() scoring function with weights: content=10, category=1, subject=5.
    Only returns active (non-superseded) memories.
    Expands query with graph-connected entity names before searching.
    Returns dicts with 'id' key (required for RRF) and 'bm25_score'.
    """
    query = _sanitize_fts_input(query)
    if not query:
        return []
    expanded_query = _expand_query_from_graph(db, query)
    if expanded_query != query:
        logger.debug("BM25 query expanded: %s -> %s", query, expanded_query)

    try:
        rows = db.execute("""
            SELECT m.id, m.content, m.category, m.subject, m.confidence,
                   m.created_at, m.metadata, m.importance, m.access_count,
                   m.surfacing_count, m.origin, m.origin_interface,
                   bm25(memories_fts, 10.0, 1.0, 5.0) as bm25_score
            FROM memories_fts f
            JOIN memories m ON f.rowid = m.rowid
            WHERE memories_fts MATCH ?
            AND m.superseded_by IS NULL
            ORDER BY bm25(memories_fts, 10.0, 1.0, 5.0)
            LIMIT ?
        """, (expanded_query, limit)).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        logger.debug("BM25 search failed for expanded query: %s", expanded_query, exc_info=True)
        # Fallback: try original query without expansion
        if expanded_query != query:
            try:
                rows = db.execute("""
                    SELECT m.id, m.content, m.category, m.subject, m.confidence,
                           m.created_at, m.metadata, m.importance, m.access_count,
                           m.surfacing_count,
                           bm25(memories_fts, 10.0, 1.0, 5.0) as bm25_score
                    FROM memories_fts f
                    JOIN memories m ON f.rowid = m.rowid
                    WHERE memories_fts MATCH ?
                    AND m.superseded_by IS NULL
                    ORDER BY bm25(memories_fts, 10.0, 1.0, 5.0)
                    LIMIT ?
                """, (query, limit)).fetchall()
                return [dict(r) for r in rows]
            except Exception:
                logger.debug("BM25 fallback also failed for: %s", query, exc_info=True)
        return []


def _get_graph_expanded_names(db, query: str) -> set[str]:
    """
    Get entity names reachable via 1-hop graph expansion from query entities.
    Used by graph slot injection to score candidates by entity density.
    """
    entity_fts_query = _query_to_entity_fts(query)
    try:
        entities = db.execute("""
            SELECT e.id, e.name
            FROM entities_fts f
            JOIN entities e ON f.rowid = e.rowid
            WHERE entities_fts MATCH ?
            LIMIT 10
        """, (entity_fts_query,)).fetchall()
    except Exception:
        return set()

    if not entities:
        return set()

    direct_ids = [e["id"] for e in entities]
    direct_names = {e["name"] for e in entities if e["name"]}
    placeholders = ",".join("?" * len(direct_ids))

    expanded = set()
    try:
        rows = db.execute(f"""
            SELECT DISTINCT e.name
            FROM relationships r
            JOIN entities e ON (
                CASE WHEN r.subject_id IN ({placeholders}) THEN r.object_id
                     ELSE r.subject_id END
            ) = e.id
            WHERE (r.subject_id IN ({placeholders}) OR r.object_id IN ({placeholders}))
            AND r.valid_to IS NULL
            LIMIT 30
        """, direct_ids * 3).fetchall()
        for r in rows:
            if r["name"] and r["name"] not in direct_names:
                expanded.add(r["name"].lower())
    except Exception:
        logger.debug("Graph expansion query failed in _get_graph_expanded_names", exc_info=True)

    return expanded


def _find_memories_by_graph(db, query: str, limit: int = 50) -> list[dict]:
    """
    Find memories connected to entities mentioned in the query via graph traversal.

    Flow:
    1. Entity FTS finds entities matching query terms
    2. 1-hop expansion: follow non-noise relationships to related entities
    3. Build set of all entity names (direct + 1-hop related)
    4. Search memories_fts (FTS5) for content mentioning any entity name
    5. Fall back to subject LIKE matching if FTS yields too few results

    The 1-hop expansion is what enables "MyApp architecture" -> MyApp entity ->
    MyApp-uses_tech->FastAPI -> memories mentioning "FastAPI".

    Returns dicts with 'id' key (required for RRF) and 'graph_score'.
    """
    # Step 1: Find entities mentioned in the query via FTS
    # Convert to OR terms so "MyApp architecture" matches "MyApp" entity
    entity_fts_query = _query_to_entity_fts(query)
    try:
        entities = db.execute("""
            SELECT e.id, e.canonical_name, e.name, e.entity_type
            FROM entities_fts f
            JOIN entities e ON f.rowid = e.rowid
            WHERE entities_fts MATCH ?
            LIMIT 10
        """, (entity_fts_query,)).fetchall()
    except Exception:
        logger.debug("Entity FTS failed for query: %s", query, exc_info=True)
        return []

    if not entities:
        return []

    # Collect direct entity IDs and names
    direct_entity_ids = set()
    direct_entity_names = set()
    for e in entities:
        direct_entity_ids.add(e["id"])
        if e["name"]:
            direct_entity_names.add(e["name"])

    # Step 2: 1-hop expansion — follow non-noise relationships
    # Only collect EXPANDED entity names (from related entities, not the query matches)
    expanded_entity_names = set()
    if direct_entity_ids:
        placeholders = ",".join("?" * len(direct_entity_ids))
        try:
            related_rows = db.execute(f"""
                SELECT DISTINCT
                    e.name
                FROM relationships r
                JOIN entities e ON (
                    CASE
                        WHEN r.subject_id IN ({placeholders}) THEN r.object_id
                        ELSE r.subject_id
                    END
                ) = e.id
                WHERE (r.subject_id IN ({placeholders}) OR r.object_id IN ({placeholders}))
                AND r.valid_to IS NULL
                LIMIT 20
            """, sorted(direct_entity_ids) * 3).fetchall()

            for row in related_rows:
                if row["name"] and row["name"] not in direct_entity_names:
                    expanded_entity_names.add(row["name"])
        except Exception:
            logger.debug("1-hop expansion failed", exc_info=True)

    # Combine both sets for different search strategies
    all_entity_names = direct_entity_names | expanded_entity_names

    if not all_entity_names:
        return []

    # Step 3: Search memories_fts for content mentioning expanded entity names
    # AND a direct entity name. This cross-referencing ensures results are
    # relevant to the original query context (e.g., "FastAPI" AND "MyApp").
    # NOTE: Iterates per entity name (not batched) to differentiate graph_score
    # (1.0 contextual vs 0.8 plain). Could batch into single OR query if
    # graph_score differentiation is removed.
    # Without the AND, searching a large number of tech entities returns noise.
    fts_results = []
    seen_fts_ids = set()

    if expanded_entity_names and direct_entity_names:
        # Sort for deterministic iteration order (sets are unordered)
        sorted_expanded = sorted(expanded_entity_names)

        # Build the direct entity part of the FTS query: "MyApp" OR "architecture" etc.
        direct_terms = []
        for name in direct_entity_names:
            clean = re.sub(r'[^\w\s]', '', name).strip()
            if clean:
                direct_terms.append(f'"{clean}"')

        direct_clause = " OR ".join(direct_terms) if direct_terms else None

        for name in sorted_expanded:
            clean = re.sub(r'[^\w\s]', '', name).strip()
            if not clean or len(clean) < 2:
                continue

            # Search for memories mentioning this expanded entity
            # (with or without direct entity context depending on availability)
            fts_query = f'"{clean}"'
            if direct_clause:
                # Prefer memories mentioning both: "FastAPI" AND ("MyApp")
                fts_query_with_context = f'({direct_clause}) AND "{clean}"'
            else:
                fts_query_with_context = fts_query

            try:
                # Try contextual query first (e.g., "MyApp" AND "FastAPI")
                rows = db.execute("""
                    SELECT m.id, m.content, m.category, m.subject, m.confidence,
                           m.created_at, m.metadata, m.importance, m.access_count,
                           m.surfacing_count, m.origin, m.origin_interface,
                           1.0 as graph_score
                    FROM memories_fts f
                    JOIN memories m ON f.rowid = m.rowid
                    WHERE memories_fts MATCH ?
                    AND m.superseded_by IS NULL
                    LIMIT 3
                """, (fts_query_with_context,)).fetchall()

                if not rows:
                    # Fall back to just the expanded entity name
                    rows = db.execute("""
                        SELECT m.id, m.content, m.category, m.subject, m.confidence,
                               m.created_at, m.metadata, m.importance, m.access_count,
                               m.surfacing_count, m.origin, m.origin_interface,
                               0.8 as graph_score
                        FROM memories_fts f
                        JOIN memories m ON f.rowid = m.rowid
                        WHERE memories_fts MATCH ?
                        AND m.superseded_by IS NULL
                        LIMIT 3
                    """, (fts_query,)).fetchall()

                for row in rows:
                    r = dict(row)
                    if r["id"] not in seen_fts_ids:
                        fts_results.append(r)
                        seen_fts_ids.add(r["id"])
            except Exception:
                # If contextual query fails (syntax), try plain entity search
                try:
                    rows = db.execute("""
                        SELECT m.id, m.content, m.category, m.subject, m.confidence,
                               m.created_at, m.metadata, m.importance, m.access_count,
                               m.surfacing_count, m.origin, m.origin_interface,
                               0.8 as graph_score
                        FROM memories_fts f
                        JOIN memories m ON f.rowid = m.rowid
                        WHERE memories_fts MATCH ?
                        AND m.superseded_by IS NULL
                        LIMIT 3
                    """, (f'"{clean}"',)).fetchall()
                    for row in rows:
                        r = dict(row)
                        if r["id"] not in seen_fts_ids:
                            fts_results.append(r)
                            seen_fts_ids.add(r["id"])
                except Exception:
                    continue

            if len(fts_results) >= limit:
                break

    elif direct_entity_names:
        # No expansion — search for direct entity names only
        for name in direct_entity_names:
            clean = re.sub(r'[^\w\s]', '', name).strip()
            if not clean:
                continue
            try:
                rows = db.execute("""
                    SELECT m.id, m.content, m.category, m.subject, m.confidence,
                           m.created_at, m.metadata, m.importance, m.access_count,
                           m.surfacing_count, m.origin, m.origin_interface,
                           1.0 as graph_score
                    FROM memories_fts f
                    JOIN memories m ON f.rowid = m.rowid
                    WHERE memories_fts MATCH ?
                    AND m.superseded_by IS NULL
                    LIMIT 5
                """, (f'"{clean}"',)).fetchall()
                for row in rows:
                    r = dict(row)
                    if r["id"] not in seen_fts_ids:
                        fts_results.append(r)
                        seen_fts_ids.add(r["id"])
            except Exception:
                continue
            if len(fts_results) >= limit:
                break

    fts_results = fts_results[:limit]

    # Step 4: Fallback — subject LIKE matching (catches memories without FTS hits)
    # NOTE: where_clause is built from entity names via parameterized LIKE (?).
    # The column references are hardcoded strings, not user input.
    seen_ids = {r["id"] for r in fts_results}
    remaining = limit - len(fts_results)

    if remaining > 0:
        conditions = []
        params = []
        for name in all_entity_names:
            conditions.append("LOWER(m.subject) LIKE ? ESCAPE '\\'")
            params.append(f"%{_escape_like(name.lower())}%")

        if conditions:
            where_clause = " OR ".join(conditions)
            params.append(remaining)
            try:
                subject_rows = db.execute(f"""
                    SELECT DISTINCT m.id, m.content, m.category, m.subject, m.confidence,
                           m.created_at, m.metadata, m.importance, m.access_count,
                           m.surfacing_count, m.origin, m.origin_interface,
                           0.8 as graph_score
                    FROM memories m
                    WHERE m.superseded_by IS NULL
                    AND ({where_clause})
                    ORDER BY m.importance DESC, m.created_at DESC
                    LIMIT ?
                """, params).fetchall()

                for row in subject_rows:
                    row_dict = dict(row)
                    if row_dict["id"] not in seen_ids:
                        fts_results.append(row_dict)
                        seen_ids.add(row_dict["id"])
            except Exception:
                logger.debug("Subject fallback search failed", exc_info=True)

    return fts_results


def _find_memories_by_ppr(db, query: str, limit: int = 50) -> list[dict]:
    """
    Graph signal via Personalized PageRank (multi-hop).

    Flow:
    1. Entity FTS finds seed entities matching query terms
    2. PPR over the seed neighborhood scores every reachable entity —
       2-hop and 3-hop entities get proportionally smaller (nonzero) weight
    3. Top-scored entities map back to memories (content FTS + subject LIKE);
       each memory accumulates the PPR score of every entity that matched it
    4. Memories ranked by accumulated score

    Returns dicts with 'id' (required for RRF) and 'graph_score' (0..1).
    """
    import maasv
    config = maasv.get_config()

    entity_fts_query = _query_to_entity_fts(query)
    try:
        seeds = db.execute("""
            SELECT e.id, e.name
            FROM entities_fts f
            JOIN entities e ON f.rowid = e.rowid
            WHERE entities_fts MATCH ?
            LIMIT 10
        """, (entity_fts_query,)).fetchall()
    except Exception:
        logger.debug("Entity FTS failed for PPR seeds: %s", query, exc_info=True)
        return []

    if not seeds:
        return []

    from maasv.core.ppr import personalized_pagerank
    try:
        scores = personalized_pagerank(
            db,
            [s["id"] for s in seeds],
            alpha=config.ppr_alpha,
            iterations=config.ppr_iterations,
            max_nodes=config.ppr_max_nodes,
        )
    except Exception:
        # The graph signal must never take down retrieval — an empty result
        # makes find_similar_memories fall back to legacy one-hop expansion.
        logger.warning("PPR scoring failed; falling back to one-hop", exc_info=True)
        return []
    if not scores:
        return []

    scored_ids = sorted(scores, key=lambda eid: (-scores[eid], eid))[:config.ppr_top_entities]
    placeholders = ",".join("?" * len(scored_ids))
    name_rows = db.execute(
        f"SELECT id, name FROM entities WHERE id IN ({placeholders})", scored_ids
    ).fetchall()
    entity_names = {r["id"]: r["name"] for r in name_rows if r["name"]}

    mem_scores: dict[str, float] = {}
    mem_rows: dict[str, dict] = {}

    for eid in scored_ids:
        name = entity_names.get(eid)
        if not name:
            continue
        clean = re.sub(r'[^\w\s]', '', name).strip()
        if len(clean) < 2:
            continue

        rows: list = []
        try:
            rows = db.execute("""
                SELECT m.id, m.content, m.category, m.subject, m.confidence,
                       m.created_at, m.metadata, m.importance, m.access_count,
                       m.surfacing_count, m.origin, m.origin_interface
                FROM memories_fts f
                JOIN memories m ON f.rowid = m.rowid
                WHERE memories_fts MATCH ?
                AND m.superseded_by IS NULL
                LIMIT 5
            """, (f'"{clean}"',)).fetchall()
        except Exception:
            logger.debug("PPR memory FTS failed for entity: %s", name, exc_info=True)

        try:
            subject_rows = db.execute("""
                SELECT m.id, m.content, m.category, m.subject, m.confidence,
                       m.created_at, m.metadata, m.importance, m.access_count,
                       m.surfacing_count, m.origin, m.origin_interface
                FROM memories m
                WHERE m.superseded_by IS NULL
                AND LOWER(m.subject) LIKE ? ESCAPE '\\'
                LIMIT 5
            """, (f"%{_escape_like(name.lower())}%",)).fetchall()
        except Exception:
            subject_rows = []

        matched_ids = set()
        for row in list(rows) + list(subject_rows):
            r = dict(row)
            if r["id"] in matched_ids:
                continue  # one credit per entity per memory
            matched_ids.add(r["id"])
            mem_scores[r["id"]] = mem_scores.get(r["id"], 0.0) + scores[eid]
            mem_rows.setdefault(r["id"], r)

    if not mem_scores:
        return []

    max_score = max(mem_scores.values())
    ranked = sorted(mem_scores, key=lambda mid: (-mem_scores[mid], mid))[:limit]
    results = []
    for mid in ranked:
        r = mem_rows[mid]
        r["graph_score"] = mem_scores[mid] / max_score if max_score > 0 else 0.0
        results.append(r)
    return results


def _reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    Each list is a sequence of dicts with at least an 'id' key.
    RRF score for each item = sum over lists of 1/(k + rank + 1).
    Returns fused list sorted by combined RRF score descending.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list):
            item_id = item["id"]
            rrf_score = 1.0 / (k + rank + 1)
            scores[item_id] = scores.get(item_id, 0.0) + rrf_score
            if item_id not in items:
                items[item_id] = item

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for item_id, score in fused:
        entry = items[item_id].copy()
        entry["rrf_score"] = score
        result.append(entry)

    return result


# ============================================================================
# MAIN RETRIEVAL FUNCTION
# ============================================================================

MAX_RETRIEVAL_LIMIT = 200


def find_similar_memories(
    query: str,
    limit: int = 5,
    category: Optional[str] = None,
    subject: Optional[str] = None,
    origin: Optional[str] = None,
    origin_interface: Optional[str] = None,
) -> list[dict]:
    """
    Find memories using 3-signal retrieval with cross-encoder reranking.

    Pipeline:
    1. Dense vector similarity -> top N candidates
    2. BM25 keyword matching (FTS5) -> top N candidates
    3. Graph connectivity (entity mentions -> subject match) -> top N candidates
    4. RRF fusion -> unified candidate pool
    5. Filter by category/subject/origin (if specified)
    6. Cross-encoder reranking (query-document relevance scoring)
       Fallback: importance-weighted formula if cross-encoder unavailable
    7. Diversity-aware selection (Jaccard dedup)
    8. Record access
    """
    # Hard cap on limit to prevent excessive resource usage
    if limit < 1:
        limit = 1
    limit = min(limit, MAX_RETRIEVAL_LIMIT)

    # Per-signal retrieval depth. Higher than Phase 1's 3x because multi-signal
    # fusion benefits from broader candidate pools — BM25 and graph may surface
    # relevant results at deeper ranks. Cap keeps total candidates manageable
    # (~50-75 unique after RRF dedup).
    RETRIEVAL_DEPTH = max(limit * 5, 25)

    import maasv
    protected = maasv.get_config().protected_categories
    now = datetime.now(timezone.utc)

    with _db() as db:
        # === Signal 1: Dense vector similarity ===
        query_embedding = get_query_embedding(query)
        vector_rows = db.execute("""
            SELECT
                v.id, m.content, m.category, m.subject, m.confidence,
                m.created_at, m.metadata, m.importance, m.access_count,
                m.surfacing_count, m.origin, m.origin_interface,
                distance
            FROM memory_vectors v
            JOIN memories m ON v.id = m.id
            WHERE m.superseded_by IS NULL
            AND v.embedding MATCH ?
            AND k = ?
            ORDER BY distance
        """, (serialize_embedding(query_embedding), RETRIEVAL_DEPTH)).fetchall()
        vector_results = [dict(row) for row in vector_rows]

        # === Signal 2: BM25 keyword matching ===
        bm25_results = _find_memories_by_bm25(db, query, limit=RETRIEVAL_DEPTH)

        # === Signal 3: Graph connectivity ===
        if maasv.get_config().graph_retrieval == "ppr":
            graph_results = _find_memories_by_ppr(db, query, limit=RETRIEVAL_DEPTH)
            if not graph_results:
                # Sparse/empty graph — legacy expansion still catches subject matches
                graph_results = _find_memories_by_graph(db, query, limit=RETRIEVAL_DEPTH)
        else:
            graph_results = _find_memories_by_graph(db, query, limit=RETRIEVAL_DEPTH)

        # === Fusion: Reciprocal Rank Fusion ===
        signals = [vector_results, bm25_results, graph_results]
        # Only include non-empty signals
        active_signals = [s for s in signals if s]

        if not active_signals:
            return []

        if len(active_signals) == 1:
            # Single signal — skip RRF overhead
            candidates = active_signals[0]
        else:
            candidates = _reciprocal_rank_fusion(active_signals, k=60)

        # === Filter by category/subject/origin ===
        if category:
            candidates = [c for c in candidates if c['category'] == category]
        if subject:
            candidates = [c for c in candidates if c.get('subject') and subject.lower() in c['subject'].lower()]
        if origin:
            candidates = [c for c in candidates if c.get('origin') == origin]
        if origin_interface:
            candidates = [c for c in candidates if c.get('origin_interface') == origin_interface]

        # === Reranking ===
        # Try cross-encoder first (best quality). Falls back to importance-weighted
        # formula if cross-encoder is unavailable.
        from maasv.core.reranker import rerank as ce_rerank
        ce_scores = ce_rerank(query, candidates)

        vector_distances = {r['id']: r['distance'] for r in vector_results}
        bm25_ids = {r['id'] for r in bm25_results}
        graph_ids = {r['id'] for r in graph_results}

        # === Importance scoring ===
        # Try learned ranker first; falls back to heuristic formula.
        from maasv.core.learned_ranker import score as learned_score
        lr_result = learned_score(
            candidates, protected, now, vector_distances, bm25_ids, graph_ids
        )
        if lr_result is not None:
            primary, supplementary = lr_result
        else:
            primary, supplementary = _importance_score(
                candidates, protected, now, vector_distances, bm25_ids, graph_ids,
                rrf_weight=maasv.get_config().rrf_rank_weight,
            )

        if ce_scores is not None:
            # === Two-stage reranking ===
            # Stage 1: importance scoring (done above).
            # Stage 2: CE reshuffles within top tier only.
            #
            # The MS MARCO cross-encoder prefers short exact matches over
            # informationally rich memories. Pure CE scoring regresses quality
            # because it displaces well-established, high-access memories with
            # semantically precise but shallow matches. Two-stage prevents this:
            # importance determines WHICH memories are candidates, CE only
            # refines the ORDER within that set.

            rerank_size = min(limit * 2, len(primary) + len(supplementary))
            importance_ranked = (primary + supplementary)[:rerank_size]

            # Map CE scores to this subset by memory ID
            ce_score_map = {}
            for mem, score in zip(candidates, ce_scores):
                ce_score_map[mem['id']] = score

            def _sigmoid(x):
                if x >= 0:
                    return 1.0 / (1.0 + math.exp(-x))
                exp_x = math.exp(x)
                return exp_x / (1.0 + exp_x)

            # Min-max normalize importance scores within the rerank window
            imp_scores = [m['_imp_score'] for m in importance_ranked]
            imp_min = min(imp_scores) if imp_scores else 0
            imp_max = max(imp_scores) if imp_scores else 1
            imp_range = imp_max - imp_min if imp_max > imp_min else 1.0

            for mem in importance_ranked:
                ce_raw = ce_score_map.get(mem['id'], 0.0)
                ce_norm = _sigmoid(ce_raw)
                imp_norm = (mem['_imp_score'] - imp_min) / imp_range

                # Importance-dominant blend: CE is a tiebreaker, not the decider.
                # 0.75 importance + 0.25 CE ensures the existing 9/10 baseline
                # is preserved while CE can swap close-ranked candidates.
                mem['_score'] = 0.75 * imp_norm + 0.25 * ce_norm

            importance_ranked.sort(key=lambda m: m['_score'], reverse=True)

            # Append any remaining candidates after the rerank window
            reranked_ids = {m['id'] for m in importance_ranked}
            remainder = [m for m in (primary + supplementary)[rerank_size:]
                         if m['id'] not in reranked_ids]
            scored_pool = importance_ranked + remainder
        else:
            # === Fallback: importance-weighted reranking ===
            # Copy _imp_score to _score for downstream compatibility
            for mem in primary + supplementary:
                mem['_score'] = mem['_imp_score']
            scored_pool = primary + supplementary

        # === Diversity-aware selection (optional) ===
        # When diversity_threshold > 0, greedily select from scored candidates,
        # skipping those too similar (by Jaccard) to already-selected results.
        config = maasv.get_config()
        if config.diversity_threshold > 0:
            result = []
            selected_words = []
            threshold = config.diversity_threshold
            for mem in scored_pool:
                if len(result) >= limit:
                    break
                mem_words = set(re.findall(r'\w+', mem.get('content', '').lower()))
                is_diverse = True
                for sw in selected_words:
                    if not mem_words or not sw:
                        continue
                    intersection = len(mem_words & sw)
                    union = len(mem_words | sw)
                    jaccard = intersection / union if union > 0 else 0
                    if jaccard > threshold:
                        is_diverse = False
                        break
                if is_diverse:
                    result.append(mem)
                    selected_words.append(mem_words)
        else:
            result = scored_pool[:limit]

        # === Graph slot injection (optional) ===
        # When enabled, if the graph signal found content via 1-hop expansion
        # that didn't make it into results, inject the best graph match into
        # the last slot. The graph signal always contributes through normal
        # RRF fusion regardless of this setting.
        if config.graph_slot_injection and graph_results and len(result) >= limit:
            result_ids = {m['id'] for m in result}
            result_content = " ".join(m.get('content', '').lower() for m in result)
            graph_only = [m for m in graph_results if m['id'] not in result_ids]

            if graph_only:
                expanded_names = _get_graph_expanded_names(db, query)
                if expanded_names:
                    novel_names = {n for n in expanded_names
                                   if n not in result_content}
                    if novel_names:
                        query_terms = [t.lower() for t in query.split() if len(t) >= 3]
                        best_candidate = None
                        best_score = (0, 0)
                        for gm in graph_only:
                            content_lower = gm.get('content', '').lower()
                            query_count = sum(1 for t in query_terms if t in content_lower)
                            if query_terms and query_count == 0:
                                continue
                            novel_count = sum(1 for n in novel_names if n in content_lower)
                            score = (novel_count, query_count)
                            if score > best_score:
                                best_score = score
                                best_candidate = gm
                        if best_candidate and best_score[0] > 0:
                            result[-1] = best_candidate

        # === Fusion rescue ===
        # Supplementary (no-vector-distance) candidates sort after every
        # primary candidate, so a rank-1 graph/BM25 hit is unreachable once
        # the corpus exceeds the vector window. Candidates in the top N of
        # their own signal claim up to fusion_rescue_slots tail slots,
        # strongest fused score first.
        rescue_n = config.fusion_rescue_top_n
        rescue_slots = min(config.fusion_rescue_slots, max(0, limit - 1))
        if rescue_n > 0 and rescue_slots > 0 and supplementary and len(result) >= limit:
            result_ids = {m['id'] for m in result}
            top_signal_ids = (
                {r['id'] for r in graph_results[:rescue_n]}
                | {r['id'] for r in bm25_results[:rescue_n]}
            )
            eligible = sorted(
                (m for m in supplementary
                 if m['id'] not in result_ids and m['id'] in top_signal_ids),
                key=lambda m: (-(m.get('rrf_score') or 0.0), m['id']),
            )
            for i, cand in enumerate(eligible[:rescue_slots]):
                result[-(i + 1)] = cand

        # Clean up internal scoring fields, expose relevance
        for mem in result:
            # Expose relevance from L2 distance on normalized vectors.
            # For unit vectors: L2² = 2 - 2·cos(θ), so cos(θ) = 1 - L2²/2.
            dist = mem.pop('distance', None)
            if dist is not None:
                cosine_sim = 1.0 - (dist * dist) / 2.0
                mem['relevance'] = round(cosine_sim, 4)
            mem.pop('_score', None)
            mem.pop('_imp_score', None)
            mem.pop('rrf_score', None)
            mem.pop('bm25_score', None)
            mem.pop('graph_score', None)

        _record_memory_access(db, [r['id'] for r in result])

        # Log retrieval for learned ranker training data (best-effort)
        try:
            from maasv.core.learned_ranker import log_retrieval
            log_retrieval(
                query=query,
                candidates=candidates,
                returned_ids=[r['id'] for r in result],
                vector_distances=vector_distances,
                bm25_ids=bm25_ids,
                graph_ids=graph_ids,
                protected=protected,
                now=now,
            )
        except Exception:
            pass

    return _redact_memories(result)


# ============================================================================
# TIERED MEMORY CONTEXT
# ============================================================================

def _get_category_priority() -> dict[str, int]:
    """Get category priority from config."""
    import maasv
    return maasv.get_config().category_priority

_core_memories_cache: list[dict] = []
_cache_timestamp: float = 0
_cache_lock = threading.Lock()
CACHE_TTL = 300  # 5 minutes


def get_core_memories(refresh: bool = False) -> list[dict]:
    """Get core memories (family, identity, preference). Cached for 5 minutes."""
    global _core_memories_cache, _cache_timestamp
    import time

    now = time.time()
    if not refresh and _core_memories_cache and (now - _cache_timestamp) < CACHE_TTL:
        return _core_memories_cache

    with _cache_lock:
        # Double-check after acquiring lock
        now = time.time()
        if not refresh and _core_memories_cache and (now - _cache_timestamp) < CACHE_TTL:
            return _core_memories_cache

        with _db() as db:
            rows = db.execute("""
                SELECT id, content, category, subject, confidence, created_at, importance
                FROM memories
                WHERE superseded_by IS NULL
                AND category IN ('family', 'identity', 'preference')
                ORDER BY
                    CASE category
                        WHEN 'family' THEN 1
                        WHEN 'identity' THEN 2
                        WHEN 'preference' THEN 3
                    END,
                    importance DESC,
                    created_at DESC
            """).fetchall()

        _core_memories_cache = [dict(row) for row in rows]
        _cache_timestamp = now

        return _core_memories_cache


def get_tiered_memory_context(
    query: str = None,
    core_limit: int = 10,
    relevant_limit: int = 5,
    use_semantic: bool = False,
    token_budget: Optional[int] = None,
    compact: bool = False,
) -> str:
    """
    Smart memory retrieval with tiered approach for low latency.

    Tier 1: Core memories (family, identity, prefs) - cached, instant
    Tier 2: Query-relevant via FTS keyword search - fast (~2ms)
    Tier 3: Semantic search - slow (~400ms), only if use_semantic=True

    token_budget: approximate cap (via utils.estimate_tokens) on the returned
    context. Memories are packed greedily in tier order — core facts first,
    query-relevant next, filler last — and packing stops at the budget, so the
    most salient facts survive. At least one fact is always included.

    compact: group facts by subject ("Marcus: fact; fact") instead of one
    bulleted line each — same information, fewer tokens.
    """
    seen_ids = set()
    memories = []

    # Under a token budget with a query, query-relevant facts pack FIRST:
    # a tight budget otherwise fills up with core facts and cuts exactly the
    # memories the query needs. Without a budget, tier order is unchanged.
    relevant_first = token_budget is not None and bool(query)

    core = get_core_memories()[:core_limit]

    if not relevant_first:
        # Tier 1: Always include core memories (cached)
        for mem in core:
            if mem['id'] not in seen_ids:
                memories.append(mem)
                seen_ids.add(mem['id'])

    # Tier 2: Add query-relevant memories via FTS (fast)
    if query and len(memories) < core_limit + relevant_limit:
        try:
            tokens = [_sanitize_fts_input(w) for w in query.split()[:5]]
            tokens = [t for t in tokens if t]
            keywords = ' OR '.join(tokens) if tokens else None
            fts_results = search_fts(keywords, limit=relevant_limit) if keywords else []
            for mem in fts_results:
                if mem['id'] not in seen_ids:
                    memories.append(mem)
                    seen_ids.add(mem['id'])
                    if len(memories) >= core_limit + relevant_limit:
                        break
        except Exception:
            logger.debug("FTS keyword search failed in tiered context", exc_info=True)

    # Tier 3: Semantic search as fallback (SLOW)
    if use_semantic and query and len(memories) < core_limit + relevant_limit:
        remaining = (core_limit + relevant_limit) - len(memories)
        semantic_results = find_similar_memories(query, limit=remaining)
        for mem in semantic_results:
            if mem['id'] not in seen_ids:
                memories.append(mem)
                seen_ids.add(mem['id'])

    if relevant_first:
        # Core memories follow the query-relevant ones
        for mem in core:
            if mem['id'] not in seen_ids and len(memories) < core_limit + relevant_limit:
                memories.append(mem)
                seen_ids.add(mem['id'])

    # Fill remaining slots with other memories by priority.
    # Fetch a bounded set (not all 5K+) ordered by importance, then sort by category priority in Python.
    remaining_slots = (core_limit + relevant_limit) - len(memories)
    if remaining_slots > 0:
        category_priority = _get_category_priority()
        # Over-fetch to account for seen_ids filtering, but cap at a reasonable limit
        fetch_limit = remaining_slots * 3

        with _db() as db:
            filler_rows = db.execute("""
                SELECT id, content, category, subject, confidence, created_at, importance
                FROM memories
                WHERE superseded_by IS NULL
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
            """, (fetch_limit,)).fetchall()

        filler_mems = [dict(row) for row in filler_rows if row['id'] not in seen_ids]
        filler_mems.sort(key=lambda m: category_priority.get(m['category'], 99))

        for mem in filler_mems:
            memories.append(mem)
            seen_ids.add(mem['id'])
            if len(memories) >= core_limit + relevant_limit:
                break

    if not memories:
        return ""

    from maasv.utils import estimate_tokens

    header = "Remembered facts:"

    if token_budget is not None:
        # Greedy pack PER FACT in salience (tier) order BEFORE any compact
        # grouping — grouping first would merge unbudgeted facts into a line
        # that ships whole and blows the cap. Header + first fact always
        # included. Cost is the fact's actual incremental rendering: in
        # compact mode a fact joining an existing subject group costs only
        # "; fact", so compact packing fits more facts under the same budget.
        selected = []
        seen_subjects = set()
        used = estimate_tokens(header)
        for i, mem in enumerate(memories):
            subject = mem.get('subject') or ""
            if compact:
                if subject and subject in seen_subjects:
                    rendered = f"; {mem['content']}"
                elif subject:
                    rendered = f"{subject}: {mem['content']}"
                else:
                    rendered = f"- {mem['content']}"
            else:
                subject_str = f"[{subject}] " if subject else ""
                rendered = f"- {subject_str}{mem['content']}"
            cost = estimate_tokens(rendered)
            if i > 0 and used + cost > token_budget:
                break
            selected.append(mem)
            if subject:
                seen_subjects.add(subject)
            used += cost
        memories = selected

    if compact:
        # Group by subject: "Marcus: fact; fact" — repeated subjects and
        # bullet scaffolding are pure token overhead. Insertion (tier) order
        # is preserved for both groups and facts within a group.
        groups: dict = {}
        order: list = []
        for mem in memories:
            key = mem.get('subject') or ""
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(mem['content'])
        lines = [header]
        for key in order:
            joined = "; ".join(groups[key])
            lines.append(f"{key}: {joined}" if key else f"- {joined}")
    else:
        lines = [header]
        for mem in memories:
            subject_str = f"[{mem['subject']}] " if mem.get('subject') else ""
            lines.append(f"- {subject_str}{mem['content']}")

    return _redact_text("\n".join(lines))


# ============================================================================
# SIMPLE FTS SEARCH (used by tiered context and externally)
# ============================================================================

def search_fts(query: str, limit: int = 10, category: Optional[str] = None) -> list[dict]:
    """Full-text search across memories, optionally filtered by category."""
    import sqlite3

    query = _sanitize_fts_input(query)
    if not query:
        return []

    with _db() as db:
        try:
            if category:
                rows = db.execute("""
                    SELECT
                        m.id, m.content, m.category, m.subject,
                        m.confidence, m.created_at
                    FROM memories_fts f
                    JOIN memories m ON f.rowid = m.rowid
                    WHERE memories_fts MATCH ?
                    AND m.superseded_by IS NULL
                    AND m.category = ?
                    ORDER BY rank
                    LIMIT ?
                """, (query, category, limit)).fetchall()
            else:
                rows = db.execute("""
                    SELECT
                        m.id, m.content, m.category, m.subject,
                        m.confidence, m.created_at
                    FROM memories_fts f
                    JOIN memories m ON f.rowid = m.rowid
                    WHERE memories_fts MATCH ?
                    AND m.superseded_by IS NULL
                    ORDER BY rank
                    LIMIT ?
                """, (query, limit)).fetchall()
        except sqlite3.OperationalError:
            logger.debug("FTS5 query failed (bad syntax?): %s", query, exc_info=True)
            return []

    return _redact_memories([dict(row) for row in rows])


def find_by_subject(subject: str, active_only: bool = True) -> list[dict]:
    """Find all memories about a specific subject."""
    escaped = _escape_like(subject)
    query = """
        SELECT id, content, category, subject, confidence, created_at, metadata
        FROM memories
        WHERE subject LIKE ? ESCAPE '\\'
    """
    if active_only:
        query += " AND superseded_by IS NULL"
    query += " ORDER BY created_at DESC"

    with _db() as db:
        rows = db.execute(query, (f"%{escaped}%",)).fetchall()

    return _redact_memories([dict(row) for row in rows])
