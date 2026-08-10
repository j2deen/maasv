"""Bi-temporal facts: functional-predicate invalidation, as-of queries, history."""

import pytest


@pytest.fixture(scope="module")
def bt_db(tmp_path_factory):
    from maasv.config import MaasvConfig
    import maasv
    from tests.test_decomposition import MockLLMProvider, MockEmbedProvider

    db_path = tmp_path_factory.mktemp("bitemporal_test") / "test.db"
    config = MaasvConfig(
        db_path=db_path,
        embed_dims=64,
        extra_functional_predicates={"custom_single"},
        extra_predicates={"custom_single"},
    )
    maasv.init(config=config, llm=MockLLMProvider(), embed=MockEmbedProvider(dims=64))

    from maasv.core.graph import find_or_create_entity
    return {
        "alice": find_or_create_entity("BtAlice", "person"),
        "toronto": find_or_create_entity("BtToronto", "place"),
        "nyc": find_or_create_entity("BtNYC", "place"),
        "atlas": find_or_create_entity("BtAtlas", "project"),
        "beacon": find_or_create_entity("BtBeacon", "project"),
    }


T1 = "2026-01-01T00:00:00+00:00"
T2 = "2026-03-01T00:00:00+00:00"
T_BETWEEN = "2026-02-01T00:00:00+00:00"
T_AFTER = "2026-04-01T00:00:00+00:00"


class TestFunctionalInvalidation:
    def test_new_fact_supersedes_old(self, bt_db):
        from maasv.core.graph import add_relationship, get_entity_relationships, get_relationship_history

        r1 = add_relationship(bt_db["alice"], "lives_in", object_id=bt_db["toronto"], valid_from=T1)
        r2 = add_relationship(bt_db["alice"], "lives_in", object_id=bt_db["nyc"], valid_from=T2)
        assert r1 != r2

        active = get_entity_relationships(bt_db["alice"], predicate="lives_in", direction="outgoing")
        assert len(active) == 1
        assert active[0]["object_id"] == bt_db["nyc"]

        history = get_relationship_history(bt_db["alice"], predicate="lives_in")
        assert len(history) == 2
        old = next(h for h in history if h["id"] == r1)
        assert old["valid_to"] == T2  # closed exactly at the new fact's valid_from
        assert old["change_reason"] == "superseded_by_new_fact"

    def test_same_object_dedups_not_supersedes(self, bt_db):
        from maasv.core.graph import add_relationship, get_relationship_history

        r2a = add_relationship(bt_db["alice"], "lives_in", object_id=bt_db["nyc"], valid_from=T_AFTER)
        history = get_relationship_history(bt_db["alice"], predicate="lives_in")
        assert len(history) == 2  # no third row — dedup returned existing id
        active = [h for h in history if h["valid_to"] is None]
        assert len(active) == 1 and active[0]["id"] == r2a

    def test_multivalued_predicate_coexists(self, bt_db):
        from maasv.core.graph import add_relationship, get_entity_relationships

        add_relationship(bt_db["alice"], "works_on", object_id=bt_db["atlas"], valid_from=T1)
        add_relationship(bt_db["alice"], "works_on", object_id=bt_db["beacon"], valid_from=T2)
        active = get_entity_relationships(bt_db["alice"], predicate="works_on", direction="outgoing")
        assert len(active) == 2

    def test_extra_functional_predicate_from_config(self, bt_db):
        from maasv.core.graph import add_relationship, get_entity_relationships

        add_relationship(bt_db["alice"], "custom_single", object_value="v1", valid_from=T1)
        add_relationship(bt_db["alice"], "custom_single", object_value="v2", valid_from=T2)
        active = get_entity_relationships(bt_db["alice"], predicate="custom_single", direction="outgoing")
        assert len(active) == 1
        assert active[0]["object_value"] == "v2"


class TestAsOfQueries:
    def test_as_of_returns_past_belief(self, bt_db):
        from maasv.core.graph import get_entity_relationships

        then = get_entity_relationships(
            bt_db["alice"], predicate="lives_in", direction="outgoing", as_of=T_BETWEEN
        )
        assert len(then) == 1
        assert then[0]["object_id"] == bt_db["toronto"]

    def test_as_of_boundary_semantics(self, bt_db):
        from maasv.core.graph import get_entity_relationships

        # At exactly T2 the old fact is closed (valid_to > as_of fails)
        # and the new fact is valid (valid_from <= as_of holds)
        at_t2 = get_entity_relationships(
            bt_db["alice"], predicate="lives_in", direction="outgoing", as_of=T2
        )
        assert len(at_t2) == 1
        assert at_t2[0]["object_id"] == bt_db["nyc"]

    def test_as_of_before_any_fact(self, bt_db):
        from maasv.core.graph import get_entity_relationships

        before = get_entity_relationships(
            bt_db["alice"], predicate="lives_in", direction="outgoing",
            as_of="2025-06-01T00:00:00+00:00"
        )
        assert before == []


class TestExpireChangeReason:
    def test_expire_records_reason(self, bt_db):
        from maasv.core.graph import add_relationship, expire_relationship, get_relationship_history

        rid = add_relationship(bt_db["alice"], "uses", object_value="vim", valid_from=T1)
        assert expire_relationship(rid, change_reason="user_correction") is True
        history = get_relationship_history(bt_db["alice"], predicate="uses")
        row = next(h for h in history if h["id"] == rid)
        assert row["valid_to"] is not None
        assert row["change_reason"] == "user_correction"

    def test_expire_already_expired_is_noop(self, bt_db):
        from maasv.core.graph import add_relationship, expire_relationship

        rid = add_relationship(bt_db["alice"], "uses", object_value="emacs", valid_from=T1)
        assert expire_relationship(rid) is True
        assert expire_relationship(rid, change_reason="late") is False
