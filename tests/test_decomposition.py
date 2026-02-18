"""
Integration tests for the maasv decomposition.

Verifies that all modules (db, store, retrieval, graph, wisdom) work correctly
after being extracted from the monolithic store.py.

Uses a shared temp DB per session via module-scoped fixture.
"""

import sys
import tempfile
from pathlib import Path

import pytest


# ============================================================================
# MOCK PROVIDERS
# ============================================================================

class MockEmbedProvider:
    """Deterministic embeddings for testing. Hashes text into a vector."""
    def __init__(self, dims=64):
        self.dims = dims
        self.call_count = 0

    def embed(self, text: str) -> list[float]:
        self.call_count += 1
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        vec = [b / 255.0 for b in h]
        while len(vec) < self.dims:
            vec.extend(vec)
        return vec[:self.dims]

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)


class MockLLMProvider:
    """Mock LLM that returns canned JSON responses."""
    def __init__(self):
        self.call_count = 0

    def call(self, messages, model, max_tokens, source=""):
        self.call_count += 1
        return "[]"


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def maasv_db(tmp_path_factory):
    """Initialize maasv with a fresh test database (shared across module)."""
    from maasv.config import MaasvConfig
    import maasv

    tmpdir = tmp_path_factory.mktemp("maasv_test")
    db_path = tmpdir / "test.db"

    config = MaasvConfig(
        db_path=db_path,
        embed_dims=64,
        extraction_model="test-model",
        inference_model="test-model",
        review_model="test-model",
        cross_encoder_enabled=False,
    )

    llm = MockLLMProvider()
    embed = MockEmbedProvider(dims=64)

    maasv.init(config=config, llm=llm, embed=embed)
    return {"llm": llm, "embed": embed, "db_path": db_path}


# ============================================================================
# db.py tests
# ============================================================================

class TestDB:
    def test_connection(self, maasv_db):
        from maasv.core.db import get_db
        db = get_db()
        assert db is not None
        row = db.execute("SELECT vec_version()").fetchone()
        assert row is not None
        db.close()

    def test_plain_connection(self, maasv_db):
        from maasv.core.db import get_plain_db
        db = get_plain_db()
        assert db is not None
        db.execute("SELECT 1").fetchone()
        db.close()

    def test_tables_exist(self, maasv_db):
        from maasv.core.db import get_db
        db = get_db()
        tables = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {row['name'] for row in tables}
        assert 'memories' in table_names
        assert 'entities' in table_names
        assert 'relationships' in table_names
        assert 'schema_migrations' in table_names
        db.close()

    def test_embeddings(self, maasv_db):
        from maasv.core.db import get_embedding, get_query_embedding, serialize_embedding
        emb = get_embedding("test text")
        assert len(emb) == 64
        assert isinstance(emb[0], float)
        q_emb = get_query_embedding("test query")
        assert len(q_emb) == 64
        serialized = serialize_embedding(emb)
        assert isinstance(serialized, bytes)


# ============================================================================
# store.py tests
# ============================================================================

class TestStore:
    def test_store_memory(self, maasv_db):
        from maasv.core.store import store_memory
        mid = store_memory(
            content="Adam lives on the Upper West Side",
            category="identity",
            subject="Adam",
            source="test",
        )
        assert mid.startswith("mem_")

    def test_store_dedup(self, maasv_db):
        from maasv.core.store import store_memory
        mid1 = store_memory(content="Adam lives on the Upper West Side", category="identity")
        mid2 = store_memory(content="Adam lives on the Upper West Side", category="identity")
        assert mid1 == mid2

    def test_get_all_active(self, maasv_db):
        from maasv.core.store import get_all_active
        active = get_all_active()
        assert len(active) >= 1
        assert any("Upper West Side" in m['content'] for m in active)

    def test_get_recent_memories(self, maasv_db):
        from maasv.core.store import get_recent_memories
        recent = get_recent_memories(hours=48)
        assert len(recent) >= 1

    def test_supersede_memory(self, maasv_db):
        from maasv.core.store import store_memory, supersede_memory, get_all_active
        old_id = store_memory(content="Gabby works at BigCorp", category="family", subject="Gabby")
        new_id = supersede_memory(old_id, "Gabby works at AcmeCo")
        assert new_id != old_id
        active = get_all_active()
        active_ids = {m['id'] for m in active}
        assert new_id in active_ids
        assert old_id not in active_ids

    def test_update_metadata(self, maasv_db):
        from maasv.core.store import store_memory, update_memory_metadata
        mid = store_memory(content="Test metadata update", category="test", metadata={"key1": "val1"})
        result = update_memory_metadata(mid, {"key2": "val2"})
        assert result is True

    def test_delete_memory(self, maasv_db):
        from maasv.core.store import store_memory, delete_memory, get_all_active
        mid = store_memory(content="This memory will be deleted xyz123", category="test")
        assert delete_memory(mid) is True
        active = get_all_active()
        assert not any(m['id'] == mid for m in active)


# ============================================================================
# graph.py tests
# ============================================================================

class TestGraph:
    def test_create_entity(self, maasv_db):
        from maasv.core.graph import create_entity, get_entity
        eid = create_entity("Adam", "person")
        assert eid.startswith("ent_")
        entity = get_entity(eid)
        assert entity['name'] == "Adam"
        assert entity['entity_type'] == "person"

    def test_find_entity_by_name(self, maasv_db):
        from maasv.core.graph import find_entity_by_name
        entity = find_entity_by_name("Adam")
        assert entity is not None
        assert entity['name'] == "Adam"

    def test_find_or_create_entity(self, maasv_db):
        from maasv.core.graph import find_or_create_entity, get_entity
        eid1 = find_or_create_entity("Adam", "person")
        eid2 = find_or_create_entity("Doris", "project")
        assert eid1 != eid2
        doris = get_entity(eid2)
        assert doris['name'] == "Doris"

    def test_normalize_entity_name(self, maasv_db):
        from maasv.core.graph import normalize_entity_name
        assert normalize_entity_name("React-Native") == normalize_entity_name("react_native")
        assert normalize_entity_name("fastapi.dev") == "fastapi"
        assert normalize_entity_name("projects") == "project"

    def test_add_relationship(self, maasv_db):
        from maasv.core.graph import find_or_create_entity, add_relationship, get_entity_relationships
        adam_id = find_or_create_entity("Adam", "person")
        doris_id = find_or_create_entity("Doris", "project")
        rel_id = add_relationship(adam_id, "works_on", object_id=doris_id, source="test")
        assert rel_id.startswith("rel_")
        rels = get_entity_relationships(adam_id, direction="outgoing")
        assert any(r['predicate'] == "works_on" for r in rels)

    def test_relationship_dedup(self, maasv_db):
        from maasv.core.graph import find_or_create_entity, add_relationship
        adam_id = find_or_create_entity("Adam", "person")
        doris_id = find_or_create_entity("Doris", "project")
        rel1 = add_relationship(adam_id, "works_on", object_id=doris_id)
        rel2 = add_relationship(adam_id, "works_on", object_id=doris_id)
        assert rel1 == rel2

    def test_expire_relationship(self, maasv_db):
        from maasv.core.graph import find_or_create_entity, add_relationship, expire_relationship, get_entity_relationships
        a_id = find_or_create_entity("TestExpireA", "thing")
        b_id = find_or_create_entity("TestExpireB", "thing")
        rel_id = add_relationship(a_id, "test_rel", object_id=b_id)
        assert expire_relationship(rel_id) is True
        rels = get_entity_relationships(a_id, include_expired=False)
        assert not any(r['id'] == rel_id for r in rels)

    def test_graph_query(self, maasv_db):
        from maasv.core.graph import graph_query
        results = graph_query(subject_type="person", predicate="works_on")
        assert len(results) >= 1
        assert results[0]['subject_name'] == "Adam"

    def test_entity_profile(self, maasv_db):
        from maasv.core.graph import find_entity_by_name, get_entity_profile
        adam = find_entity_by_name("Adam")
        profile = get_entity_profile(adam['id'])
        assert 'entity' in profile
        assert 'relationships' in profile
        assert profile['entity']['name'] == "Adam"

    def test_search_entities(self, maasv_db):
        from maasv.core.graph import search_entities
        results = search_entities("Adam")
        assert len(results) >= 1

    def test_merge_entity(self, maasv_db):
        from maasv.core.graph import create_entity, merge_entity, add_relationship, get_entity
        keeper = create_entity("MergeKeeper", "thing")
        dup = create_entity("MergeDup", "thing")
        add_relationship(dup, "test_pred", object_value="test_val")
        stats = merge_entity(keeper, [dup])
        assert stats['entities_deleted'] == 1
        assert get_entity(dup) is None


# ============================================================================
# retrieval.py tests
# ============================================================================

class TestRetrieval:
    def test_find_similar_memories(self, maasv_db):
        from maasv.core.retrieval import find_similar_memories
        results = find_similar_memories("Adam Upper West Side", limit=3)
        assert len(results) >= 1
        assert any("Upper West Side" in m['content'] for m in results)

    def test_search_fts(self, maasv_db):
        from maasv.core.retrieval import search_fts
        results = search_fts("Upper West Side", limit=5)
        assert len(results) >= 1

    def test_find_by_subject(self, maasv_db):
        from maasv.core.retrieval import find_by_subject
        results = find_by_subject("Adam")
        assert len(results) >= 1

    def test_get_core_memories(self, maasv_db):
        from maasv.core.retrieval import get_core_memories
        core = get_core_memories(refresh=True)
        assert len(core) >= 1

    def test_tiered_memory_context(self, maasv_db):
        from maasv.core.retrieval import get_tiered_memory_context
        context = get_tiered_memory_context(query="Adam")
        assert "Remembered facts:" in context
        assert len(context) > 20


# ============================================================================
# wisdom.py tests
# ============================================================================

class TestWisdom:
    def test_wisdom_tables(self, maasv_db):
        from maasv.core.db import get_db
        db = get_db()
        tables = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {row['name'] for row in tables}
        assert 'wisdom' in table_names
        db.close()

    def test_log_reasoning(self, maasv_db):
        from maasv.core.wisdom import log_reasoning
        entry_id = log_reasoning(
            action_type="test_action",
            reasoning="Testing the wisdom module",
            action_data={"key": "value"},
        )
        assert entry_id is not None

    def test_search_wisdom(self, maasv_db):
        from maasv.core.wisdom import search_wisdom
        results = search_wisdom("test")
        assert isinstance(results, list)


# ============================================================================
# __init__.py re-exports
# ============================================================================

class TestReexports:
    def test_core_reexports(self, maasv_db):
        from maasv.core import (
            store_memory, find_similar_memories, find_by_subject, search_fts,
            get_all_active, get_recent_memories, delete_memory, supersede_memory,
            create_entity, get_entity, find_entity_by_name, find_or_create_entity,
            search_entities, add_relationship, expire_relationship,
            get_entity_relationships, get_causal_chain, graph_query, get_entity_profile,
            log_reasoning, record_outcome, add_feedback, get_relevant_wisdom, search_wisdom,
        )
        assert callable(store_memory)
        assert callable(find_similar_memories)
        assert callable(create_entity)
        assert callable(graph_query)
        assert callable(log_reasoning)


# ============================================================================
# lifecycle import paths
# ============================================================================

class TestLifecycleImports:
    def test_inference_imports(self, maasv_db):
        from maasv.lifecycle import inference

    def test_memory_hygiene_imports(self, maasv_db):
        from maasv.lifecycle import memory_hygiene

    def test_reorganize_imports(self, maasv_db):
        from maasv.lifecycle import reorganize


# ============================================================================
# Backwards-compatible CLI entry point
# ============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
