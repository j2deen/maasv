"""Regression tests for the confirmed adversarial-review findings.

Each test reproduces a defect the review verified with a live repro; they
must keep failing loudly if any fix regresses.
"""

import json

import pytest


class TestPPRCapCrash:
    """Finding: edges referencing max_nodes-trimmed endpoints crashed PPR."""

    def test_hub_exceeding_max_nodes_does_not_crash(self, tmp_path):
        import maasv
        from maasv.config import MaasvConfig
        from tests.test_decomposition import MockLLMProvider, MockEmbedProvider

        config = MaasvConfig(db_path=tmp_path / "hub.db", embed_dims=64)
        maasv.init(config=config, llm=MockLLMProvider(), embed=MockEmbedProvider(dims=64))

        from maasv.core.graph import find_or_create_entity, add_relationship
        from maasv.core.db import _db
        from maasv.core.ppr import personalized_pagerank, build_subgraph

        hub = find_or_create_entity("HubSeed", "person")
        for i in range(12):
            spoke = find_or_create_entity(f"Spoke{i}", "project")
            add_relationship(hub, "works_on", object_id=spoke)

        with _db() as db:
            # max_nodes smaller than the neighborhood: previously KeyError
            nodes, edges = build_subgraph(db, [hub], max_nodes=5)
            assert all(a in nodes and b in nodes for a, b, _ in edges)
            scores = personalized_pagerank(db, [hub], max_nodes=5)
        assert scores[hub] > 0
        assert abs(sum(scores.values()) - 1.0) < 1e-4


class TestFusionRescue:
    """Finding: graph/BM25-only hits could never reach top-k on large corpora."""

    def test_ppr_hit_survives_distractor_flood(self):
        from evals.corpus import build_corpus, Memory
        from evals.harness import _setup
        import tempfile
        from pathlib import Path

        corpus = build_corpus()
        # Vocabulary-overlapping distractors push the gold out of the vector
        # window (RETRIEVAL_DEPTH=25 at limit=5)
        for i in range(40):
            corpus.memories.append(Memory(
                f"dist{i}",
                f"Item {i}: the team leads the project review meeting notes "
                f"and depends on the planning board",
                "history",
            ))

        with tempfile.TemporaryDirectory() as tmp:
            key_to_id = _setup(Path(tmp) / "flood.db", corpus, None)
            from maasv.core.retrieval import find_similar_memories

            results = find_similar_memories(
                "Who is responsible for the thing that depends on Postgres?", limit=5
            )
            ids = [r["id"] for r in results]
            assert key_to_id["priya_leads_atlas"] in ids


class TestBitemporalBackfill:
    """Finding: backfilled facts expired the current fact and corrupted as-of."""

    @pytest.fixture()
    def bf_db(self, tmp_path):
        import maasv
        from maasv.config import MaasvConfig
        from tests.test_decomposition import MockLLMProvider, MockEmbedProvider

        config = MaasvConfig(db_path=tmp_path / "bf.db", embed_dims=64)
        maasv.init(config=config, llm=MockLLMProvider(), embed=MockEmbedProvider(dims=64))
        from maasv.core.graph import find_or_create_entity
        return find_or_create_entity("BfAlice", "person")

    def test_backfill_keeps_current_fact_active(self, bf_db):
        from maasv.core.graph import (
            add_relationship, get_entity_relationships, get_relationship_history,
        )

        add_relationship(bf_db, "lives_in", object_value="NYC",
                         valid_from="2026-06-01T00:00:00+00:00")
        add_relationship(bf_db, "lives_in", object_value="Toronto",
                         valid_from="2026-01-01T00:00:00+00:00")  # historical backfill

        active = get_entity_relationships(bf_db, predicate="lives_in", direction="outgoing")
        assert len(active) == 1
        assert active[0]["object_value"] == "NYC"  # current fact untouched

        history = get_relationship_history(bf_db, predicate="lives_in")
        toronto = next(h for h in history if h["object_value"] == "Toronto")
        # Backfill lands pre-closed at the next fact's valid_from
        assert toronto["valid_to"] == "2026-06-01T00:00:00+00:00"
        assert toronto["change_reason"] == "backfilled_historical"

        nyc = next(h for h in history if h["object_value"] == "NYC")
        assert nyc["valid_to"] is None  # no negative interval

    def test_as_of_correct_after_backfill(self, bf_db):
        from maasv.core.graph import add_relationship, get_entity_relationships

        add_relationship(bf_db, "lives_in", object_value="NYC",
                         valid_from="2026-06-01T00:00:00+00:00")
        add_relationship(bf_db, "lives_in", object_value="Toronto",
                         valid_from="2026-01-01T00:00:00+00:00")

        during = get_entity_relationships(
            bf_db, predicate="lives_in", direction="outgoing",
            as_of="2026-03-01T00:00:00+00:00")
        assert [r["object_value"] for r in during] == ["Toronto"]

        after = get_entity_relationships(
            bf_db, predicate="lives_in", direction="outgoing",
            as_of="2026-07-01T00:00:00+00:00")
        assert [r["object_value"] for r in after] == ["NYC"]


class TestEvolveFixes:
    def test_tag_refresh_preserves_backlink(self, tmp_path):
        """Finding: _refresh_tags wrote stale metadata, erasing related_ids."""
        import maasv
        from maasv.config import MaasvConfig
        from maasv.core.store import store_memory
        from maasv.lifecycle.evolve import run_evolve_job
        from tests.test_evolve import ClusterEmbed, TagLLM, _set_created, _get_meta

        config = MaasvConfig(db_path=tmp_path / "tagfix.db", embed_dims=64,
                             evolve_llm_refresh=True)
        maasv.init(config=config, llm=TagLLM(), embed=ClusterEmbed(dims=64))

        a1 = store_memory("The planet Mercury is closest to the sun", category="learning")
        a2 = store_memory("The planet Neptune is farthest out", category="learning")
        _set_created(a1, "2026-01-01 00:00:01")
        _set_created(a2, "2026-01-01 00:00:02")

        stats = run_evolve_job({}, cancel_check=lambda: False)
        assert stats["tags_refreshed"] >= 1
        meta = _get_meta(a1)
        assert meta.get("tags") == ["planets", "astronomy"]
        assert a2 in meta.get("related_ids", [])  # backlink survives the refresh

    def test_same_second_memories_all_processed(self, tmp_path):
        """Finding: timestamp-only watermark skipped same-second rows forever."""
        import maasv
        from maasv.config import MaasvConfig
        from maasv.core.store import store_memory
        from maasv.lifecycle.evolve import run_evolve_job
        from tests.test_evolve import ClusterEmbed, _set_created
        from tests.test_decomposition import MockLLMProvider

        config = MaasvConfig(db_path=tmp_path / "wm.db", embed_dims=64,
                             evolve_batch_size=2)
        maasv.init(config=config, llm=MockLLMProvider(), embed=ClusterEmbed(dims=64))

        ids = [
            store_memory(f"The planet catalogue entry {i}", category="learning")
            for i in range(3)
        ]
        for mid in ids:
            _set_created(mid, "2026-01-01 00:00:01")  # all share one second

        total = 0
        for _ in range(3):
            total += run_evolve_job({}, cancel_check=lambda: False)["processed"]
        assert total == 3  # batch 2, then 1, then 0 — nobody skipped


class TestCompactBudgetInteraction:
    """Finding: subject-grouping before packing shipped a mega-line over budget."""

    @pytest.fixture()
    def cb_db(self, tmp_path):
        import maasv
        from maasv.config import MaasvConfig
        from tests.test_decomposition import MockLLMProvider, MockEmbedProvider

        config = MaasvConfig(db_path=tmp_path / "cb.db", embed_dims=64)
        maasv.init(config=config, llm=MockLLMProvider(), embed=MockEmbedProvider(dims=64))

        from maasv.core.store import store_memory
        from maasv.core.retrieval import get_core_memories
        # All subjectless: previously merged into ONE always-included line
        store_memory("Denver offsite planning starts next month", category="history")
        for i in range(12):
            store_memory(f"Miscellaneous project note number {i} about infra", category="project")
        get_core_memories(refresh=True)
        return True

    def test_compact_respects_budget(self, cb_db):
        from maasv.core.retrieval import get_tiered_memory_context
        from maasv.utils import estimate_tokens

        ctx = get_tiered_memory_context(
            query="Denver offsite", token_budget=40, compact=True
        )
        assert estimate_tokens(ctx) <= 40 + 25  # header + first-fact tolerance
        assert ctx.count("Miscellaneous") <= 2  # filler no longer rides along

    def test_compact_budget_equals_plain_budget_selection(self, cb_db):
        from maasv.core.retrieval import get_tiered_memory_context

        plain = get_tiered_memory_context(query="Denver offsite", token_budget=40)
        compact = get_tiered_memory_context(query="Denver offsite", token_budget=40, compact=True)
        # Same facts selected either way; compact only changes rendering
        plain_facts = {l.split("] ")[-1].lstrip("- ") for l in plain.splitlines()[1:]}
        for fact in plain_facts:
            assert fact in compact
