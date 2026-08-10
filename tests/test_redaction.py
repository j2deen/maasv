"""Output-redaction hook: applied at retrieval boundaries, never at rest."""

import pytest


SECRET = "555-0142"


def _scrub(text: str) -> str:
    return text.replace(SECRET, "[PHONE]")


@pytest.fixture(scope="module")
def redact_db(tmp_path_factory):
    import maasv
    from maasv.config import MaasvConfig
    from tests.test_decomposition import MockLLMProvider, MockEmbedProvider

    db_path = tmp_path_factory.mktemp("redact_test") / "test.db"
    config = MaasvConfig(db_path=db_path, embed_dims=64, redact_output=_scrub)
    maasv.init(config=config, llm=MockLLMProvider(), embed=MockEmbedProvider(dims=64))

    from maasv.core.store import store_memory
    from maasv.core.retrieval import get_core_memories
    mid = store_memory(
        f"Marcus phone number is {SECRET}", category="person", subject="Marcus"
    )
    get_core_memories(refresh=True)
    return {"mid": mid}


class TestRedaction:
    def test_find_similar_redacts(self, redact_db):
        from maasv.core.retrieval import find_similar_memories
        results = find_similar_memories("Marcus phone number", limit=5)
        assert results
        assert all(SECRET not in r["content"] for r in results)
        assert any("[PHONE]" in r["content"] for r in results)

    def test_tiered_context_redacts(self, redact_db):
        from maasv.core.retrieval import get_tiered_memory_context
        context = get_tiered_memory_context(query="Marcus phone")
        assert SECRET not in context

    def test_search_fts_redacts(self, redact_db):
        from maasv.core.retrieval import search_fts
        results = search_fts("phone")
        assert results
        assert all(SECRET not in r["content"] for r in results)

    def test_find_by_subject_redacts(self, redact_db):
        from maasv.core.retrieval import find_by_subject
        results = find_by_subject("Marcus")
        assert results
        assert all(SECRET not in r["content"] for r in results)

    def test_stored_data_untouched(self, redact_db):
        from maasv.core.db import _db
        with _db() as db:
            row = db.execute(
                "SELECT content FROM memories WHERE id = ?", (redact_db["mid"],)
            ).fetchone()
        assert SECRET in row["content"]  # at-rest content is NOT redacted

    def test_raising_hook_fails_closed(self, redact_db):
        import maasv
        from maasv.core.retrieval import find_similar_memories

        def boom(text):
            raise RuntimeError("scrubber down")

        config = maasv.get_config()
        original = config.redact_output
        config.redact_output = boom
        try:
            results = find_similar_memories("Marcus phone number", limit=5)
            assert all(r["content"] == "[redacted]" for r in results)
        finally:
            config.redact_output = original
