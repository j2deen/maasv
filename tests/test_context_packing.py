"""Token-budgeted, compact context packing in get_tiered_memory_context."""

import pytest

from maev.utils import estimate_tokens


@pytest.fixture(scope="module")
def packing_db(tmp_path_factory):
    import maev
    from maev.config import MaevConfig
    from tests.test_decomposition import MockLLMProvider, MockEmbedProvider

    db_path = tmp_path_factory.mktemp("packing_test") / "test.db"
    config = MaevConfig(db_path=db_path, embed_dims=64)
    maev.init(config=config, llm=MockLLMProvider(), embed=MockEmbedProvider(dims=64))

    from maev.core.store import store_memory
    from maev.core.retrieval import get_core_memories

    store_memory("Gabby is my wife", category="family", subject="Gabby")
    store_memory("We live on the Upper West Side", category="identity", subject="Home")
    store_memory("I prefer window seats on flights", category="preference")
    store_memory("Quarterly offsite is in Denver this year", category="history", subject="Offsite")
    store_memory("The Denver office parking needs a badge", category="history", subject="Offsite")
    for i in range(8):
        store_memory(f"Miscellaneous project note number {i} about infra", category="project")
    get_core_memories(refresh=True)
    return db_path


class TestTokenBudget:
    def test_budget_respected(self, packing_db):
        from maev.core.retrieval import get_tiered_memory_context
        ctx = get_tiered_memory_context(query="Denver offsite", token_budget=60)
        # header + first fact are always included; small tolerance over budget
        assert estimate_tokens(ctx) <= 60 + 25

    def test_budget_reduces_tokens(self, packing_db):
        from maev.core.retrieval import get_tiered_memory_context
        full = get_tiered_memory_context(query="Denver offsite")
        tight = get_tiered_memory_context(query="Denver offsite", token_budget=60)
        assert estimate_tokens(tight) < estimate_tokens(full)

    def test_query_relevant_survives_tight_budget(self, packing_db):
        from maev.core.retrieval import get_tiered_memory_context
        ctx = get_tiered_memory_context(query="Denver offsite parking", token_budget=50)
        assert "Denver" in ctx  # relevant-first packing beats core facts

    def test_no_budget_keeps_core_first(self, packing_db):
        from maev.core.retrieval import get_tiered_memory_context
        ctx = get_tiered_memory_context(query="Denver offsite")
        # Unbudgeted: legacy tier order — core (family) facts lead
        gabby_pos = ctx.find("Gabby")
        denver_pos = ctx.find("Denver")
        assert gabby_pos != -1 and denver_pos != -1
        assert gabby_pos < denver_pos

    def test_at_least_one_fact(self, packing_db):
        from maev.core.retrieval import get_tiered_memory_context
        ctx = get_tiered_memory_context(query="Denver", token_budget=1)
        assert len(ctx.splitlines()) >= 2  # header + one fact, budget or not


class TestCompact:
    def test_compact_groups_by_subject(self, packing_db):
        from maev.core.retrieval import get_tiered_memory_context
        ctx = get_tiered_memory_context(query="Denver offsite", compact=True)
        assert ctx.count("Offsite:") == 1  # both Offsite facts on one line
        assert "; " in ctx

    def test_compact_saves_tokens(self, packing_db):
        from maev.core.retrieval import get_tiered_memory_context
        normal = get_tiered_memory_context(query="Denver offsite")
        compact = get_tiered_memory_context(query="Denver offsite", compact=True)
        assert estimate_tokens(compact) < estimate_tokens(normal)

    def test_compact_keeps_all_content(self, packing_db):
        from maev.core.retrieval import get_tiered_memory_context
        normal = get_tiered_memory_context(query="Denver offsite")
        compact = get_tiered_memory_context(query="Denver offsite", compact=True)
        for line in normal.splitlines()[1:]:
            fact = line.split("] ")[-1].lstrip("- ")
            assert fact in compact
