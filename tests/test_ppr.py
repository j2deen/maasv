"""Tests for Personalized PageRank graph retrieval."""

import pytest


@pytest.fixture(scope="module")
def ppr_db(tmp_path_factory):
    """Fresh maasv db with a chain graph: A -> B -> C -> D, plus isolated E."""
    from maasv.config import MaasvConfig
    import maasv
    from tests.test_decomposition import MockLLMProvider, MockEmbedProvider

    db_path = tmp_path_factory.mktemp("ppr_test") / "test.db"
    config = MaasvConfig(
        db_path=db_path,
        embed_dims=64,
        extra_predicates={"test_pred", "test_rel"},
    )
    maasv.init(config=config, llm=MockLLMProvider(), embed=MockEmbedProvider(dims=64))

    from maasv.core.graph import find_or_create_entity, add_relationship

    ids = {}
    for name in ("ChainA", "ChainB", "ChainC", "ChainD", "IsolatedE"):
        ids[name] = find_or_create_entity(name, "project")
    add_relationship(ids["ChainA"], "depends_on", object_id=ids["ChainB"])
    add_relationship(ids["ChainB"], "depends_on", object_id=ids["ChainC"])
    add_relationship(ids["ChainC"], "depends_on", object_id=ids["ChainD"])
    return ids


class TestPersonalizedPagerank:
    def test_score_decays_with_distance(self, ppr_db):
        from maasv.core.db import _db
        from maasv.core.ppr import personalized_pagerank

        with _db() as db:
            scores = personalized_pagerank(db, [ppr_db["ChainA"]])
        assert scores[ppr_db["ChainA"]] > scores[ppr_db["ChainB"]]
        assert scores[ppr_db["ChainB"]] > scores[ppr_db["ChainC"]]
        assert scores[ppr_db["ChainC"]] > scores[ppr_db["ChainD"]]

    def test_multihop_reach(self, ppr_db):
        """3-hop node gets nonzero score — the thing 1-hop expansion can't do."""
        from maasv.core.db import _db
        from maasv.core.ppr import personalized_pagerank

        with _db() as db:
            scores = personalized_pagerank(db, [ppr_db["ChainA"]])
        assert scores[ppr_db["ChainD"]] > 0.0

    def test_isolated_node_excluded(self, ppr_db):
        from maasv.core.db import _db
        from maasv.core.ppr import personalized_pagerank

        with _db() as db:
            scores = personalized_pagerank(db, [ppr_db["ChainA"]])
        assert ppr_db["IsolatedE"] not in scores

    def test_empty_seeds(self, ppr_db):
        from maasv.core.db import _db
        from maasv.core.ppr import personalized_pagerank

        with _db() as db:
            assert personalized_pagerank(db, []) == {}

    def test_scores_sum_to_one(self, ppr_db):
        from maasv.core.db import _db
        from maasv.core.ppr import personalized_pagerank

        with _db() as db:
            scores = personalized_pagerank(db, [ppr_db["ChainB"]])
        assert abs(sum(scores.values()) - 1.0) < 1e-4

    def test_subgraph_bounds(self, ppr_db):
        from maasv.core.db import _db
        from maasv.core.ppr import build_subgraph

        with _db() as db:
            nodes, edges = build_subgraph(db, [ppr_db["ChainA"]], max_nodes=2)
        assert len(nodes) <= 3  # seeds + capped frontier growth
