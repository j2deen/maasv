"""
maasv Memory Store

SQLite + sqlite-vec for persistent memory with semantic search.
Hybrid approach: vector embeddings for fuzzy matching, FTS5 for keywords.

All database paths and embedding calls come from the initialized config/providers.
"""

import logging
import sqlite3
import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional, Callable

logger = logging.getLogger(__name__)


def _get_db_path():
    """Get the configured database path."""
    import maasv
    return maasv.get_config().db_path


def _get_embed_dims():
    """Get the configured embedding dimensions."""
    import maasv
    return maasv.get_config().embed_dims


def get_db() -> sqlite3.Connection:
    """Get database connection with sqlite-vec loaded."""
    import sqlite_vec

    db = sqlite3.connect(str(_get_db_path()))
    db.row_factory = sqlite3.Row

    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    return db


@contextmanager
def _db():
    """Context manager for database connections — ensures close on exception."""
    db = get_db()
    try:
        yield db
    finally:
        db.close()


def _ensure_migration_table(db: sqlite3.Connection):
    """Create the schema_migrations tracking table if it doesn't exist."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.commit()


def run_migration(db: sqlite3.Connection, version: int, description: str, migrate_fn: Callable[[sqlite3.Connection], None]):
    """
    Run a schema migration if it hasn't been applied yet.

    Checks schema_migrations for the version. If not present, runs migrate_fn
    inside a transaction and records the version. If already applied, skips silently.

    Args:
        db: Open database connection (with sqlite-vec loaded)
        version: Integer migration version (must be unique, monotonically increasing)
        description: Human-readable description of what this migration does
        migrate_fn: Callable that takes a db connection and performs the migration
    """
    existing = db.execute(
        "SELECT version FROM schema_migrations WHERE version = ?", (version,)
    ).fetchone()
    if existing:
        return

    logger.info(f"Running migration {version}: {description}")
    try:
        migrate_fn(db)
        db.execute(
            "INSERT INTO schema_migrations (version, description) VALUES (?, ?)",
            (version, description)
        )
        db.commit()
        logger.info(f"Migration {version} applied successfully")
    except Exception:
        db.rollback()
        logger.error(f"Migration {version} failed, rolled back", exc_info=True)
        raise


def init_db():
    """Initialize database schema."""
    db = get_db()
    embed_dims = _get_embed_dims()

    # Core memories table
    db.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            category TEXT NOT NULL,
            subject TEXT,
            source TEXT NOT NULL DEFAULT 'manual',
            confidence REAL DEFAULT 1.0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            superseded_by TEXT,
            metadata TEXT
        )
    """)

    # Vector embeddings table (sqlite-vec)
    db.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors USING vec0(
            id TEXT PRIMARY KEY,
            embedding FLOAT[{embed_dims}]
        )
    """)

    # === GRAPH MEMORY TABLES ===
    db.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            canonical_name TEXT,
            metadata TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS relationships (
            id TEXT PRIMARY KEY,
            subject_id TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object_id TEXT,
            object_value TEXT,
            valid_from TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            valid_to TEXT,
            confidence REAL DEFAULT 1.0,
            source TEXT,
            metadata TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (subject_id) REFERENCES entities(id)
        )
    """)

    # Indexes for efficient graph queries
    db.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_relationships_subject ON relationships(subject_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_relationships_object ON relationships(object_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_relationships_predicate ON relationships(predicate)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_relationships_valid ON relationships(valid_to)")

    # Full-text search for entities
    db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
            name,
            entity_type,
            canonical_name,
            content='entities',
            content_rowid='rowid'
        )
    """)

    # Triggers for entity FTS sync
    db.execute("""
        CREATE TRIGGER IF NOT EXISTS entities_ai AFTER INSERT ON entities BEGIN
            INSERT INTO entities_fts(rowid, name, entity_type, canonical_name)
            VALUES (NEW.rowid, NEW.name, NEW.entity_type, NEW.canonical_name);
        END
    """)

    db.execute("""
        CREATE TRIGGER IF NOT EXISTS entities_ad AFTER DELETE ON entities BEGIN
            INSERT INTO entities_fts(entities_fts, rowid, name, entity_type, canonical_name)
            VALUES ('delete', OLD.rowid, OLD.name, OLD.entity_type, OLD.canonical_name);
        END
    """)

    db.execute("""
        CREATE TRIGGER IF NOT EXISTS entities_au AFTER UPDATE ON entities BEGIN
            INSERT INTO entities_fts(entities_fts, rowid, name, entity_type, canonical_name)
            VALUES ('delete', OLD.rowid, OLD.name, OLD.entity_type, OLD.canonical_name);
            INSERT INTO entities_fts(rowid, name, entity_type, canonical_name)
            VALUES (NEW.rowid, NEW.name, NEW.entity_type, NEW.canonical_name);
        END
    """)

    # Full-text search for memories
    db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content,
            category,
            subject,
            content='memories',
            content_rowid='rowid'
        )
    """)

    # Triggers to keep FTS in sync
    db.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content, category, subject)
            VALUES (NEW.rowid, NEW.content, NEW.category, NEW.subject);
        END
    """)

    db.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, category, subject)
            VALUES ('delete', OLD.rowid, OLD.content, OLD.category, OLD.subject);
        END
    """)

    db.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, category, subject)
            VALUES ('delete', OLD.rowid, OLD.content, OLD.category, OLD.subject);
            INSERT INTO memories_fts(rowid, content, category, subject)
            VALUES (NEW.rowid, NEW.content, NEW.category, NEW.subject);
        END
    """)

    db.commit()

    # --- Migration tracking and schema migrations ---
    _ensure_migration_table(db)

    # Version 0: Mark the Phase 0 baseline as tracked
    run_migration(db, 0, "Phase 0 baseline", lambda _db: None)

    # Version 1: Bitemporal columns
    def _migrate_bitemporal(db: sqlite3.Connection):
        db.execute("ALTER TABLE memories ADD COLUMN ingested_at TEXT DEFAULT NULL")
        db.execute("UPDATE memories SET ingested_at = created_at WHERE ingested_at IS NULL")
        db.execute("ALTER TABLE relationships ADD COLUMN ingested_at TEXT DEFAULT NULL")
        db.execute("UPDATE relationships SET ingested_at = created_at WHERE ingested_at IS NULL")
        db.execute("ALTER TABLE relationships ADD COLUMN change_reason TEXT DEFAULT NULL")

    run_migration(db, 1, "Bitemporal columns (ingested_at, change_reason)", _migrate_bitemporal)

    # Version 2: Importance scoring + decay columns
    def _migrate_importance(db: sqlite3.Connection):
        db.execute("ALTER TABLE memories ADD COLUMN importance REAL DEFAULT 0.5")
        db.execute("ALTER TABLE memories ADD COLUMN access_count INTEGER DEFAULT 0")
        db.execute("ALTER TABLE memories ADD COLUMN last_accessed_at TEXT DEFAULT NULL")
        # Backfill: family/identity = 1.0, decision = 0.8
        db.execute("UPDATE memories SET importance = 1.0 WHERE category IN ('family', 'identity')")
        db.execute("UPDATE memories SET importance = 0.8 WHERE category = 'decision'")

    run_migration(db, 2, "Importance scoring + decay columns", _migrate_importance)

    db.close()


# ============================================================================
# EMBEDDING HELPERS
# ============================================================================

def get_embedding(text: str) -> list[float]:
    """Get embedding vector for a document/memory via the configured EmbedProvider."""
    import maasv
    return maasv.get_embed().embed(text)


def get_query_embedding(text: str) -> list[float]:
    """Get embedding vector for a search query (may use instruction prefix)."""
    import maasv
    return maasv.get_embed().embed_query(text)


def serialize_embedding(embedding: list[float]) -> bytes:
    """Convert embedding to binary format for sqlite-vec."""
    from sqlite_vec import serialize_float32
    return serialize_float32(embedding)


# ============================================================================
# ACCESS TRACKING
# ============================================================================

def _record_memory_access(db: sqlite3.Connection, memory_ids: list[str]):
    """Increment access_count and set last_accessed_at for retrieved memories."""
    if not memory_ids:
        return
    now = datetime.now(timezone.utc).isoformat()
    placeholders = ",".join("?" * len(memory_ids))
    try:
        db.execute(
            f"UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? "
            f"WHERE id IN ({placeholders})",
            [now] + memory_ids
        )
        db.commit()
    except Exception:
        logger.warning("Failed to record memory access", exc_info=True)


def _record_entity_access(db: sqlite3.Connection, entity_ids: list[str]):
    """Increment access_count and set last_accessed_at for retrieved entities."""
    if not entity_ids:
        return
    now = datetime.now(timezone.utc).isoformat()
    placeholders = ",".join("?" * len(entity_ids))
    try:
        db.execute(
            f"UPDATE entities SET access_count = access_count + 1, last_accessed_at = ? "
            f"WHERE id IN ({placeholders})",
            [now] + entity_ids
        )
        db.commit()
    except Exception:
        logger.warning("Failed to record entity access", exc_info=True)


# ============================================================================
# MEMORY STORAGE & SEARCH
# ============================================================================

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
        subject: Who/what this is about (e.g., "Levi", "TerryAnn")
        source: Where this came from (manual, conversation, extracted)
        confidence: How confident we are (0.0-1.0)
        metadata: Additional structured data
        dedup_threshold: Vector distance below which a memory is considered duplicate

    Returns:
        Memory ID (existing ID if duplicate found)
    """
    import logging
    logger = logging.getLogger(__name__)

    # Compute embedding first (needed for both dedup check and storage)
    embedding = get_embedding(content)

    # Dedup check: find near-duplicate via vector similarity
    with _db() as db:
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
            pass  # If dedup check fails, proceed with store

    with _db() as db:
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


def find_similar_memories(
    query: str,
    limit: int = 5,
    category: Optional[str] = None,
    subject: Optional[str] = None
) -> list[dict]:
    """
    Find memories using 3-signal retrieval with importance-weighted reranking.

    Pipeline:
    1. Dense vector similarity → top N candidates (primary signal)
    2. BM25 keyword matching (FTS5) → top N candidates
    3. Graph connectivity (entity mentions → subject match) → top N candidates
    4. RRF fusion → unified candidate pool (deduped)
    5. Filter by category/subject (if specified)
    6. Split: primary (vector-retrieved) scored with Phase 1 importance formula,
       supplementary (BM25/graph-only) scored separately
    7. Primary results first, supplementary fills remaining slots
    8. Record access

    Multi-signal agreement: primary results found by multiple signals get a
    small additive bonus. Protected categories (family, identity) exempt from decay.
    """
    import math
    import re

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

        # === Filter by category/subject ===
        if category:
            candidates = [c for c in candidates if c['category'] == category]
        if subject:
            candidates = [c for c in candidates if c.get('subject') and subject.lower() in c['subject'].lower()]

        # === Importance-weighted reranking ===
        # Strategy: vector distance is primary quality signal (Phase 1 formula).
        # BM25/graph agreement provides a multiplicative boost.
        # Non-vector results are scored separately and appended as diversity fills.
        vector_distances = {r['id']: r['distance'] for r in vector_results}
        bm25_ids = {r['id'] for r in bm25_results}
        graph_ids = {r['id'] for r in graph_results}

        primary = []    # Vector-retrieved candidates (scored with distance)
        supplementary = []  # BM25/graph-only candidates (scored by importance)

        for mem in candidates:
            importance = mem.get('importance') or 0.5
            access_count = mem.get('access_count') or 0

            # Decay: protected categories don't decay
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

            # Dampen access_count to prevent feedback loops: frequently-retrieved
            # memories would otherwise snowball (acc=7 → 3.17x boost over acc=0).
            # Cap at 5 accesses (max 2.81x boost) so new/rare memories can compete.
            dampened_access = min(access_count, 5)

            distance = vector_distances.get(mem['id'])
            if distance is not None:
                # Primary: scored with Phase 1 formula.
                # (1 - distance) can be negative for cosine dist > 1.0 — that's fine,
                # negative scores still rank correctly relative to other vector results.
                base_score = (1.0 - distance) * importance * decay_factor * math.log(2 + dampened_access)

                # Multi-signal agreement boost: memories found by multiple signals
                # get an additive bonus (not multiplicative — multiplying negative
                # scores would invert the ranking).
                signal_count = 1  # already in vector
                if mem['id'] in bm25_ids:
                    signal_count += 1
                if mem['id'] in graph_ids:
                    signal_count += 1
                agreement_bonus = (signal_count - 1) * 0.01

                mem['_score'] = base_score + agreement_bonus
                primary.append(mem)
            else:
                # Supplementary: found only by BM25/graph, no vector match.
                # These fill remaining slots after all vector results.
                mem['_score'] = importance * decay_factor * math.log(2 + dampened_access) * 0.0001
                supplementary.append(mem)

        # Sort each group independently
        primary.sort(key=lambda m: m['_score'], reverse=True)
        supplementary.sort(key=lambda m: m['_score'], reverse=True)

        # === Diversity-aware selection (MMR-inspired) ===
        # Greedily select from primary, skipping candidates too similar to
        # already-selected results. This prevents near-duplicate memories
        # (e.g., 5 copies of "Adam has a Mac Mini") from filling all slots.
        # Similarity is measured by word overlap (Jaccard). Threshold adapts:
        # short memories (< 12 words) use 0.5 (they're easily near-duplicates),
        # longer memories use 0.7 (more content means legitimate diversity).
        result = []
        selected_words = []  # list of word-sets for selected results

        for pool in [primary, supplementary]:
            if len(result) >= limit:
                break
            for mem in pool:
                if len(result) >= limit:
                    break
                # Check diversity: is this too similar to any selected result?
                mem_words = set(re.findall(r'\w+', mem.get('content', '').lower()))
                threshold = 0.5 if len(mem_words) < 12 else 0.7
                is_diverse = True
                for sw in selected_words:
                    if not mem_words or not sw:
                        continue
                    intersection = len(mem_words & sw)
                    union = len(mem_words | sw)
                    jaccard = intersection / union if union > 0 else 0
                    # Also check asymmetric containment: if one memory's words
                    # are mostly a subset of another, it's redundant
                    smaller = min(len(mem_words), len(sw))
                    containment = intersection / smaller if smaller > 0 else 0
                    if jaccard > threshold or containment > 0.8:
                        is_diverse = False
                        break
                if is_diverse:
                    result.append(mem)
                    selected_words.append(mem_words)

        # Clean up internal scoring fields
        for mem in result:
            mem.pop('_score', None)
            mem.pop('rrf_score', None)
            mem.pop('bm25_score', None)
            mem.pop('graph_score', None)
            mem.pop('distance', None)

        _record_memory_access(db, [r['id'] for r in result])

    return result


def find_by_subject(subject: str, active_only: bool = True) -> list[dict]:
    """Find all memories about a specific subject."""
    query = """
        SELECT id, content, category, subject, confidence, created_at, metadata
        FROM memories
        WHERE subject LIKE ?
    """
    if active_only:
        query += " AND superseded_by IS NULL"
    query += " ORDER BY created_at DESC"

    with _db() as db:
        rows = db.execute(query, (f"%{subject}%",)).fetchall()

    return [dict(row) for row in rows]


def search_fts(query: str, limit: int = 10, category: Optional[str] = None) -> list[dict]:
    """Full-text search across memories, optionally filtered by category."""
    with _db() as db:
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

    return [dict(row) for row in rows]


# ============================================================================
# MULTI-SIGNAL RETRIEVAL (Phase 2)
# ============================================================================

def _find_memories_by_bm25(db: sqlite3.Connection, query: str, limit: int = 50) -> list[dict]:
    """
    Return memories ranked by BM25 relevance from the FTS5 index.

    Uses bm25() scoring function with weights: content=10, category=1, subject=5.
    Only returns active (non-superseded) memories.
    Returns dicts with 'id' key (required for RRF) and 'bm25_score'.
    """
    try:
        rows = db.execute("""
            SELECT m.id, m.content, m.category, m.subject, m.confidence,
                   m.created_at, m.metadata, m.importance, m.access_count,
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
        logger.debug("BM25 search failed for query: %s", query, exc_info=True)
        return []


def _find_memories_by_graph(db: sqlite3.Connection, query: str, limit: int = 50) -> list[dict]:
    """
    Find memories connected to entities mentioned in the query.

    Flow:
    1. Match query terms against entity names via entities_fts
    2. Find memories whose subject matches any discovered entity's canonical_name or name
    3. Also search memory content for direct entity name mentions

    Note: No hop expansion yet — direct entity matches only. Hop expansion
    added too much noise (TerryAnn → Adam → family memories). Will add
    weighted hop expansion in a future iteration when graph is denser.

    Returns dicts with 'id' key (required for RRF) and 'graph_score'.
    """
    # Step 1: Find entities mentioned in the query via FTS
    try:
        entities = db.execute("""
            SELECT e.id, e.canonical_name, e.name
            FROM entities_fts f
            JOIN entities e ON f.rowid = e.rowid
            WHERE entities_fts MATCH ?
            LIMIT 10
        """, (query,)).fetchall()
    except Exception:
        logger.debug("Entity FTS failed for query: %s", query, exc_info=True)
        return []

    if not entities:
        return []

    entity_names = set()
    for e in entities:
        if e["canonical_name"]:
            entity_names.add(e["canonical_name"])
        if e["name"]:
            entity_names.add(e["name"].lower())

    if not entity_names:
        return []

    # Step 2: Find memories whose subject matches any entity name
    conditions = []
    params = []
    for name in entity_names:
        conditions.append("LOWER(m.subject) LIKE ?")
        params.append(f"%{name.lower()}%")

    where_clause = " OR ".join(conditions)
    params.append(limit)

    rows = db.execute(f"""
        SELECT DISTINCT m.id, m.content, m.category, m.subject, m.confidence,
               m.created_at, m.metadata, m.importance, m.access_count,
               1.0 as graph_score
        FROM memories m
        WHERE m.superseded_by IS NULL
        AND ({where_clause})
        ORDER BY m.importance DESC, m.created_at DESC
        LIMIT ?
    """, params).fetchall()

    return [dict(r) for r in rows]


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


def check_conflicts(content: str, subject: Optional[str] = None) -> list[dict]:
    """Check if new memory conflicts with existing ones."""
    conflicts = []

    similar = find_similar_memories(content, limit=5)
    conflicts.extend(similar)

    if subject:
        by_subject = find_by_subject(subject)
        for mem in by_subject:
            if not any(c['id'] == mem['id'] for c in conflicts):
                conflicts.append(mem)

    return conflicts


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


def get_unprocessed_thoughts() -> list[dict]:
    """Get all unprocessed brain dump thoughts."""
    with _db() as db:
        rows = db.execute("""
            SELECT id, content, category, subject, source, confidence, created_at, metadata
            FROM memories
            WHERE category = 'thought'
            AND superseded_by IS NULL
            ORDER BY created_at ASC
        """).fetchall()

    unprocessed = []
    for row in rows:
        row_dict = dict(row)
        metadata = json.loads(row_dict['metadata']) if row_dict['metadata'] else {}
        if not metadata.get('processed', False):
            row_dict['_metadata'] = metadata
            unprocessed.append(row_dict)

    return unprocessed


# ============================================================================
# TIERED MEMORY CONTEXT
# ============================================================================

CATEGORY_PRIORITY = {
    'family': 1,
    'identity': 2,
    'preference': 3,
    'project': 4,
    'decision': 5,
    'person': 6,
    'learning': 7,
    'history': 8,
    'home': 9,
    'conversation': 10,
}

_core_memories_cache: list[dict] = []
_cache_timestamp: float = 0
CACHE_TTL = 300  # 5 minutes


def get_core_memories(refresh: bool = False) -> list[dict]:
    """Get core memories (family, identity, preference). Cached for 5 minutes."""
    global _core_memories_cache, _cache_timestamp
    import time

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
    use_semantic: bool = False
) -> str:
    """
    Smart memory retrieval with tiered approach for low latency.

    Tier 1: Core memories (family, identity, prefs) - cached, instant
    Tier 2: Query-relevant via FTS keyword search - fast (~2ms)
    Tier 3: Semantic search - slow (~400ms), only if use_semantic=True
    """
    seen_ids = set()
    memories = []

    # Tier 1: Always include core memories (cached)
    core = get_core_memories()[:core_limit]
    for mem in core:
        if mem['id'] not in seen_ids:
            memories.append(mem)
            seen_ids.add(mem['id'])

    # Tier 2: Add query-relevant memories via FTS (fast)
    if query and len(memories) < core_limit + relevant_limit:
        try:
            keywords = ' OR '.join(query.split()[:5])
            fts_results = search_fts(keywords, limit=relevant_limit)
            for mem in fts_results:
                if mem['id'] not in seen_ids:
                    memories.append(mem)
                    seen_ids.add(mem['id'])
                    if len(memories) >= core_limit + relevant_limit:
                        break
        except Exception:
            pass

    # Tier 3: Semantic search as fallback (SLOW)
    if use_semantic and query and len(memories) < core_limit + relevant_limit:
        remaining = (core_limit + relevant_limit) - len(memories)
        semantic_results = find_similar_memories(query, limit=remaining)
        for mem in semantic_results:
            if mem['id'] not in seen_ids:
                memories.append(mem)
                seen_ids.add(mem['id'])

    # Fill remaining slots with other memories by priority
    if len(memories) < core_limit + relevant_limit:
        all_mems = get_all_active()
        all_mems.sort(key=lambda m: CATEGORY_PRIORITY.get(m['category'], 99))
        for mem in all_mems:
            if mem['id'] not in seen_ids:
                memories.append(mem)
                seen_ids.add(mem['id'])
                if len(memories) >= core_limit + relevant_limit:
                    break

    if not memories:
        return ""

    lines = ["Remembered facts:"]
    for mem in memories:
        subject = f"[{mem['subject']}] " if mem.get('subject') else ""
        lines.append(f"- {subject}{mem['content']}")

    return "\n".join(lines)


# ============================================================================
# GRAPH MEMORY FUNCTIONS
# ============================================================================

def create_entity(
    name: str,
    entity_type: str,
    canonical_name: Optional[str] = None,
    metadata: Optional[dict] = None
) -> str:
    """Create a new entity in the knowledge graph."""
    entity_id = f"ent_{uuid.uuid4().hex[:12]}"

    if canonical_name is None:
        canonical_name = name.lower().strip().replace(" ", "_")

    with _db() as db:
        db.execute("""
            INSERT INTO entities (id, name, entity_type, canonical_name, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            entity_id, name, entity_type, canonical_name,
            json.dumps(metadata) if metadata else None
        ))
        db.commit()
    return entity_id


def get_entity(entity_id: str) -> Optional[dict]:
    """Get an entity by ID."""
    with _db() as db:
        row = db.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()

    if row:
        result = dict(row)
        if result.get('metadata'):
            result['metadata'] = json.loads(result['metadata'])
        return result
    return None


def find_entity_by_name(name: str, entity_type: Optional[str] = None) -> Optional[dict]:
    """Find an entity by name (case-insensitive)."""
    canonical = name.lower().strip().replace(" ", "_")

    with _db() as db:
        if entity_type:
            row = db.execute(
                "SELECT * FROM entities WHERE canonical_name = ? AND entity_type = ?",
                (canonical, entity_type)
            ).fetchone()
        else:
            row = db.execute(
                "SELECT * FROM entities WHERE canonical_name = ?",
                (canonical,)
            ).fetchone()

        if row:
            result = dict(row)
            _record_entity_access(db, [result['id']])
            if result.get('metadata'):
                result['metadata'] = json.loads(result['metadata'])
            return result
    return None


def find_or_create_entity(
    name: str,
    entity_type: str,
    metadata: Optional[dict] = None
) -> str:
    """Find existing entity or create new one. Returns entity ID."""
    existing = find_entity_by_name(name)
    if existing:
        return existing['id']
    return create_entity(name, entity_type, metadata=metadata)


def search_entities(
    query: str,
    entity_type: Optional[str] = None,
    limit: int = 10
) -> list[dict]:
    """Search entities using FTS."""
    with _db() as db:
        try:
            if entity_type:
                rows = db.execute("""
                    SELECT e.*
                    FROM entities_fts f
                    JOIN entities e ON f.rowid = e.rowid
                    WHERE entities_fts MATCH ?
                    AND e.entity_type = ?
                    ORDER BY rank
                    LIMIT ?
                """, (query, entity_type, limit)).fetchall()
            else:
                rows = db.execute("""
                    SELECT e.*
                    FROM entities_fts f
                    JOIN entities e ON f.rowid = e.rowid
                    WHERE entities_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (query, limit)).fetchall()
        except Exception:
            if entity_type:
                rows = db.execute("""
                    SELECT * FROM entities
                    WHERE name LIKE ? AND entity_type = ?
                    LIMIT ?
                """, (f"%{query}%", entity_type, limit)).fetchall()
            else:
                rows = db.execute("""
                    SELECT * FROM entities WHERE name LIKE ? LIMIT ?
                """, (f"%{query}%", limit)).fetchall()

    results = []
    for row in rows:
        result = dict(row)
        if result.get('metadata'):
            result['metadata'] = json.loads(result['metadata'])
        results.append(result)

    return results


def get_entities_by_type(entity_type: str, limit: int = 50) -> list[dict]:
    """Get all entities of a given type."""
    with _db() as db:
        rows = db.execute(
            "SELECT * FROM entities WHERE entity_type = ? ORDER BY name LIMIT ?",
            (entity_type, limit)
        ).fetchall()

    results = []
    for row in rows:
        result = dict(row)
        if result.get('metadata'):
            result['metadata'] = json.loads(result['metadata'])
        results.append(result)

    return results


def add_relationship(
    subject_id: str,
    predicate: str,
    object_id: Optional[str] = None,
    object_value: Optional[str] = None,
    valid_from: Optional[str] = None,
    confidence: float = 1.0,
    source: Optional[str] = None,
    metadata: Optional[dict] = None
) -> str:
    """Add a temporal relationship between entities."""
    if object_id is None and object_value is None:
        raise ValueError("Must provide either object_id or object_value")

    rel_id = f"rel_{uuid.uuid4().hex[:12]}"

    if valid_from is None:
        valid_from = datetime.now(timezone.utc).isoformat()

    with _db() as db:
        db.execute("""
            INSERT INTO relationships
            (id, subject_id, predicate, object_id, object_value, valid_from, confidence, source, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rel_id, subject_id, predicate, object_id, object_value,
            valid_from, confidence, source,
            json.dumps(metadata) if metadata else None
        ))
        db.commit()
    return rel_id


def expire_relationship(
    relationship_id: str,
    valid_to: Optional[str] = None
) -> bool:
    """Mark a relationship as expired (no longer current)."""
    if valid_to is None:
        valid_to = datetime.now(timezone.utc).isoformat()

    with _db() as db:
        cursor = db.execute(
            "UPDATE relationships SET valid_to = ? WHERE id = ? AND valid_to IS NULL",
            (valid_to, relationship_id)
        )
        updated = cursor.rowcount > 0
        db.commit()
    return updated


def get_entity_relationships(
    entity_id: str,
    include_expired: bool = False,
    predicate: Optional[str] = None,
    direction: str = "both"
) -> list[dict]:
    """Get all relationships for an entity."""
    results = []
    queries = []
    params_list = []

    if direction in ("outgoing", "both"):
        query = """
            SELECT r.*,
                   e_subj.name as subject_name, e_subj.entity_type as subject_type,
                   e_obj.name as object_name, e_obj.entity_type as object_type
            FROM relationships r
            JOIN entities e_subj ON r.subject_id = e_subj.id
            LEFT JOIN entities e_obj ON r.object_id = e_obj.id
            WHERE r.subject_id = ?
        """
        params = [entity_id]
        if not include_expired:
            query += " AND r.valid_to IS NULL"
        if predicate:
            query += " AND r.predicate = ?"
            params.append(predicate)
        queries.append(query)
        params_list.append(params)

    if direction in ("incoming", "both"):
        query = """
            SELECT r.*,
                   e_subj.name as subject_name, e_subj.entity_type as subject_type,
                   e_obj.name as object_name, e_obj.entity_type as object_type
            FROM relationships r
            JOIN entities e_subj ON r.subject_id = e_subj.id
            LEFT JOIN entities e_obj ON r.object_id = e_obj.id
            WHERE r.object_id = ?
        """
        params = [entity_id]
        if not include_expired:
            query += " AND r.valid_to IS NULL"
        if predicate:
            query += " AND r.predicate = ?"
            params.append(predicate)
        queries.append(query)
        params_list.append(params)

    seen_ids = set()
    with _db() as db:
        for query, params in zip(queries, params_list):
            rows = db.execute(query, params).fetchall()
            for row in rows:
                row_dict = dict(row)
                if row_dict['id'] not in seen_ids:
                    if row_dict.get('metadata'):
                        row_dict['metadata'] = json.loads(row_dict['metadata'])
                    results.append(row_dict)
                    seen_ids.add(row_dict['id'])

        # Track access on the queried entity itself
        _record_entity_access(db, [entity_id])

    return results


# Causal predicate sets for chain traversal
_FORWARD_CAUSAL = {"led_to", "resulted_in", "enabled_by"}
_BACKWARD_CAUSAL = {"caused_by", "motivated_by", "blocked_by"}
_ALL_CAUSAL = _FORWARD_CAUSAL | _BACKWARD_CAUSAL | {"chose_over"}


def get_causal_chain(
    entity_id: str,
    direction: str = "both",
    max_hops: int = 3
) -> list[dict]:
    """
    Traverse causal edges from an entity.

    Args:
        entity_id: Starting entity
        direction: "forward" (led_to, resulted_in, enabled_by),
                   "backward" (caused_by, motivated_by, blocked_by),
                   or "both"
        max_hops: Maximum traversal depth

    Returns:
        Ordered list of {"entity": {...}, "relationship": {...}, "hop": int}
    """
    if direction == "forward":
        predicates = _FORWARD_CAUSAL
    elif direction == "backward":
        predicates = _BACKWARD_CAUSAL
    else:
        predicates = _ALL_CAUSAL

    chain = []
    visited = {entity_id}
    frontier = [entity_id]

    with _db() as db:
        for hop in range(1, max_hops + 1):
            next_frontier = []
            for current_id in frontier:
                placeholders = ",".join("?" * len(predicates))
                rows = db.execute(f"""
                    SELECT r.*,
                           e.id as next_entity_id, e.name as next_entity_name,
                           e.entity_type as next_entity_type
                    FROM relationships r
                    JOIN entities e ON (
                        CASE WHEN r.subject_id = ? THEN r.object_id ELSE r.subject_id END
                    ) = e.id
                    WHERE (r.subject_id = ? OR r.object_id = ?)
                    AND r.predicate IN ({placeholders})
                    AND r.valid_to IS NULL
                """, [current_id, current_id, current_id] + list(predicates)).fetchall()

                for row in rows:
                    row_dict = dict(row)
                    next_id = row_dict['next_entity_id']
                    if next_id and next_id not in visited:
                        visited.add(next_id)
                        next_frontier.append(next_id)
                        chain.append({
                            "entity": {
                                "id": next_id,
                                "name": row_dict['next_entity_name'],
                                "type": row_dict['next_entity_type'],
                            },
                            "relationship": {
                                "id": row_dict['id'],
                                "predicate": row_dict['predicate'],
                                "subject_id": row_dict['subject_id'],
                                "object_id": row_dict['object_id'],
                                "confidence": row_dict['confidence'],
                            },
                            "hop": hop,
                        })

            frontier = next_frontier
            if not frontier:
                break

    return chain


def update_relationship_value(
    subject_id: str,
    predicate: str,
    new_value: str,
    source: Optional[str] = None
) -> tuple[Optional[str], str]:
    """Update a relationship by expiring the old one and creating a new one."""
    with _db() as db:
        current = db.execute("""
            SELECT id FROM relationships
            WHERE subject_id = ? AND predicate = ? AND valid_to IS NULL
        """, (subject_id, predicate)).fetchone()

    old_id = None
    if current:
        old_id = current['id']
        expire_relationship(old_id)

    new_id = add_relationship(
        subject_id=subject_id,
        predicate=predicate,
        object_value=new_value,
        source=source
    )

    return (old_id, new_id)


def graph_query(
    subject_type: Optional[str] = None,
    predicate: Optional[str] = None,
    object_type: Optional[str] = None,
    include_expired: bool = False,
    limit: int = 50
) -> list[dict]:
    """Query the graph with pattern matching."""
    query = """
        SELECT r.*,
               e_subj.name as subject_name, e_subj.entity_type as subject_type,
               e_obj.name as object_name, e_obj.entity_type as object_type
        FROM relationships r
        JOIN entities e_subj ON r.subject_id = e_subj.id
        LEFT JOIN entities e_obj ON r.object_id = e_obj.id
        WHERE 1=1
    """
    params = []

    if not include_expired:
        query += " AND r.valid_to IS NULL"
    if subject_type:
        query += " AND e_subj.entity_type = ?"
        params.append(subject_type)
    if predicate:
        query += " AND r.predicate = ?"
        params.append(predicate)
    if object_type:
        query += " AND e_obj.entity_type = ?"
        params.append(object_type)

    query += " ORDER BY r.created_at DESC LIMIT ?"
    params.append(limit)

    with _db() as db:
        rows = db.execute(query, params).fetchall()

    results = []
    for row in rows:
        row_dict = dict(row)
        if row_dict.get('metadata'):
            row_dict['metadata'] = json.loads(row_dict['metadata'])
        results.append(row_dict)

    return results


def get_entity_profile(entity_id: str) -> dict:
    """Get a complete profile for an entity including all current relationships."""
    entity = get_entity(entity_id)
    if not entity:
        return {}

    relationships = get_entity_relationships(entity_id, include_expired=False)

    profile = {
        "entity": entity,
        "relationships": {},
        "related_entities": []
    }

    related_ids = set()
    for rel in relationships:
        pred = rel['predicate']
        if pred not in profile['relationships']:
            profile['relationships'][pred] = []

        entry = {
            "id": rel['id'],
            "valid_from": rel['valid_from'],
            "confidence": rel['confidence']
        }

        if rel['object_id']:
            entry['entity_id'] = rel['object_id']
            entry['entity_name'] = rel.get('object_name')
            entry['entity_type'] = rel.get('object_type')
            related_ids.add(rel['object_id'])
        else:
            entry['value'] = rel['object_value']

        profile['relationships'][pred].append(entry)

    for eid in related_ids:
        related = get_entity(eid)
        if related:
            profile['related_entities'].append(related)

    return profile
