"""Memory evolution (A-MEM style): linking, watermark, LLM tag refresh."""

import hashlib
import json
import math

import pytest


class ClusterEmbed:
    """Embeddings with controllable semantics: texts sharing a cluster keyword
    land at cosine ~0.86 (above the 0.70 link floor, far above store-dedup's
    L2 0.05); different clusters are near-orthogonal."""

    CLUSTERS = {"planet": 0, "banana": 1, "guitar": 2}

    def __init__(self, dims: int = 64):
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for kw, idx in self.CLUSTERS.items():
            if kw in text.lower():
                vec[idx] = 1.0
                break
        # Deterministic per-text noise in the tail dims keeps members distinct
        h = hashlib.sha256(text.encode()).digest()
        noise = [b / 255.0 - 0.5 for b in h[: self.dims - 8]]
        norm = math.sqrt(sum(x * x for x in noise)) or 1.0
        for i, x in enumerate(noise):
            vec[8 + i % (self.dims - 8)] += 0.55 * x / norm
        total = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / total for x in vec]

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)


class TagLLM:
    """LLM stub that always proposes tags."""

    def call(self, messages, model, max_tokens, source=""):
        return json.dumps({"tags": ["planets", "astronomy"]})


def _no_cancel() -> bool:
    return False


def _init(tmp_path, **config_kwargs):
    import maev
    from maev.config import MaevConfig
    from tests.test_decomposition import MockLLMProvider

    config = MaevConfig(db_path=tmp_path / "evolve.db", embed_dims=64, **config_kwargs)
    llm = config_kwargs.pop("_llm", None) if "_llm" in config_kwargs else None
    maev.init(config=config, llm=llm or MockLLMProvider(), embed=ClusterEmbed(dims=64))


def _set_created(mem_id: str, ts: str) -> None:
    from maev.core.db import _db
    with _db() as db:
        db.execute("UPDATE memories SET created_at = ? WHERE id = ?", (ts, mem_id))
        db.commit()


def _get_meta(mem_id: str) -> dict:
    from maev.core.db import _db
    with _db() as db:
        row = db.execute("SELECT metadata FROM memories WHERE id = ?", (mem_id,)).fetchone()
    return json.loads(row["metadata"]) if row and row["metadata"] else {}


class TestEvolve:
    def test_links_related_skips_unrelated(self, tmp_path):
        _init(tmp_path)
        from maev.core.store import store_memory
        from maev.lifecycle.evolve import run_evolve_job

        a1 = store_memory("The planet Mars has two moons", category="learning")
        a2 = store_memory("The planet Venus has a thick atmosphere", category="learning")
        b1 = store_memory("Banana bread needs ripe bananas", category="learning")
        _set_created(a1, "2026-01-01 00:00:01")
        _set_created(a2, "2026-01-01 00:00:02")
        _set_created(b1, "2026-01-01 00:00:03")

        stats = run_evolve_job({}, cancel_check=_no_cancel)
        assert stats["processed"] == 3
        assert stats["links_created"] >= 1

        assert a1 in _get_meta(a2).get("related_ids", [])   # newer side links back
        assert a2 in _get_meta(a1).get("related_ids", [])   # older side updated too
        assert "related_ids" not in _get_meta(b1)           # cross-cluster: no link

    def test_watermark_prevents_reprocessing(self, tmp_path):
        _init(tmp_path)
        from maev.core.store import store_memory
        from maev.lifecycle.evolve import run_evolve_job

        m = store_memory("The planet Jupiter is the largest", category="learning")
        _set_created(m, "2026-01-01 00:00:01")
        assert run_evolve_job({}, cancel_check=_no_cancel)["processed"] == 1
        assert run_evolve_job({}, cancel_check=_no_cancel)["processed"] == 0

    def test_second_batch_links_across_watermark(self, tmp_path):
        _init(tmp_path)
        from maev.core.store import store_memory
        from maev.lifecycle.evolve import run_evolve_job

        old = store_memory("Guitar strings are tuned E A D G B E", category="learning")
        _set_created(old, "2026-01-01 00:00:01")
        run_evolve_job({}, cancel_check=_no_cancel)

        new = store_memory("Guitar chords use three or more notes", category="learning")
        _set_created(new, "2026-01-01 00:00:05")
        stats = run_evolve_job({}, cancel_check=_no_cancel)
        assert stats["processed"] == 1
        assert old in _get_meta(new).get("related_ids", [])
        assert new in _get_meta(old).get("related_ids", [])

    def test_disabled_is_noop(self, tmp_path):
        _init(tmp_path, evolve_enabled=False)
        from maev.core.store import store_memory
        from maev.lifecycle.evolve import run_evolve_job

        store_memory("The planet Saturn has rings", category="learning")
        assert run_evolve_job({}, cancel_check=_no_cancel)["processed"] == 0

    def test_cancel_stops_early(self, tmp_path):
        _init(tmp_path)
        from maev.core.store import store_memory
        from maev.lifecycle.evolve import run_evolve_job

        for i in range(3):
            m = store_memory(f"The planet number {i} is hypothetical", category="learning")
            _set_created(m, f"2026-01-01 00:00:0{i + 1}")
        stats = run_evolve_job({}, cancel_check=lambda: True)
        assert stats["cancelled"] is True
        assert stats["processed"] <= 1

    def test_llm_tag_refresh(self, tmp_path):
        import maev
        from maev.config import MaevConfig
        from maev.core.store import store_memory
        from maev.lifecycle.evolve import run_evolve_job

        config = MaevConfig(
            db_path=tmp_path / "evolve_llm.db", embed_dims=64, evolve_llm_refresh=True
        )
        maev.init(config=config, llm=TagLLM(), embed=ClusterEmbed(dims=64))

        a1 = store_memory("The planet Mercury is closest to the sun", category="learning")
        a2 = store_memory("The planet Neptune is farthest out", category="learning")
        _set_created(a1, "2026-01-01 00:00:01")
        _set_created(a2, "2026-01-01 00:00:02")

        stats = run_evolve_job({}, cancel_check=_no_cancel)
        assert stats["tags_refreshed"] >= 1
        assert _get_meta(a1).get("tags") == ["planets", "astronomy"]
