"""Eval harness: recall + token cost for maasv retrieval, with control arms.

Three arms per run:
- retrieval:    find_similar_memories(query, limit=k) — the main pipeline
- tiered:       get_tiered_memory_context(query) — what a host would inject
- full_context: every active memory concatenated — the long-context control

Metrics: recall@1, recall@k, MRR, mean tokens injected per query. A gold hit
means any of the QA's gold memories appears in the arm's output. Per-type
breakdown (keyword/paraphrase/graph_1hop/graph_2hop) shows which retrieval
signal a change helped or hurt.
"""

import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from evals.corpus import Corpus, build_corpus
from evals.providers import HashedBowEmbed, NullLLM

EMBED_DIMS = 256
BUDGET_TOKENS = 120  # budgeted-tiered arm: token_budget passed to context packing


def approx_tokens(text: str) -> int:
    """Model-free token estimate (~4 chars/token, the standard heuristic)."""
    return max(1, round(len(text) / 4))


def _setup(db_path: Path, corpus: Corpus, config_overrides: Optional[dict] = None) -> dict[str, str]:
    """Init a fresh maasv db, load the corpus. Returns memory key -> id map.

    Memory/entity IDs are pinned to a deterministic sequence for the duration
    of corpus loading: several ranking tie-breakers fall back to the ID, and
    random uuid4 IDs would make exact-tie ordering differ run to run.
    """
    import itertools
    import uuid as _uuid
    import maasv
    from maasv.config import MaasvConfig

    kwargs = dict(
        db_path=db_path,
        embed_dims=EMBED_DIMS,
        cross_encoder_enabled=False,
        learned_ranker_enabled=False,  # deterministic evals: heuristic scoring only
        # All corpus categories protected -> decay_factor is exactly 1.0 for
        # every memory. Otherwise decay depends on wall-clock datetime.now()
        # and consecutive runs differ at ~1e-9 — enough to flip float-tied
        # ranks and make eval runs non-reproducible.
        protected_categories={
            "project", "person", "preference", "history", "identity",
            "family", "learning",
        },
    )
    kwargs.update(config_overrides or {})
    maasv.init(config=MaasvConfig(**kwargs), llm=NullLLM(), embed=HashedBowEmbed(dims=EMBED_DIMS))

    from maasv.core.store import store_memory
    from maasv.core.graph import find_or_create_entity, add_relationship
    from maasv.core.retrieval import get_core_memories

    counter = itertools.count(1)
    real_uuid4 = _uuid.uuid4
    # Counter in the HIGH bits: id generators take uuid4().hex[:12], the
    # first 12 hex chars, which low-bit counters would leave all-zero
    _uuid.uuid4 = lambda: _uuid.UUID(int=next(counter) << 96)
    try:
        key_to_id: dict[str, str] = {}
        for mem in corpus.memories:
            key_to_id[mem.key] = store_memory(
                mem.content, category=mem.category, subject=mem.subject, source="eval"
            )

        entity_ids: dict[str, str] = {}
        for name, etype in corpus.entities:
            entity_ids[name] = find_or_create_entity(name, etype)
        for subj, pred, obj in corpus.relationships:
            add_relationship(entity_ids[subj], pred, object_id=entity_ids[obj])
    finally:
        _uuid.uuid4 = real_uuid4

    # Pin every timestamp to one instant: created_at ties and decay factors are
    # then identical run-to-run, so evals can't flake on second boundaries.
    from maasv.core.db import _db as _db_ctx
    frozen = "2026-01-01 00:00:00"
    with _db_ctx() as db:
        db.execute("UPDATE memories SET created_at=?, updated_at=?, ingested_at=?", (frozen,) * 3)
        db.execute("UPDATE relationships SET created_at=?, valid_from=?, ingested_at=?", (frozen,) * 3)
        db.execute("UPDATE entities SET created_at=?, updated_at=?", (frozen,) * 2)
        db.commit()

    get_core_memories(refresh=True)  # bust module-level cache from prior runs
    return key_to_id


def _score_ranked(returned_ids: list[str], gold_ids: set[str], k: int) -> dict:
    rank = next((i + 1 for i, rid in enumerate(returned_ids) if rid in gold_ids), None)
    return {
        "hit_at_1": rank == 1,
        "hit_at_k": rank is not None and rank <= k,
        "rr": (1.0 / rank) if rank else 0.0,
        "rank": rank,
    }


def run_eval(k: int = 5, config_overrides: Optional[dict] = None,
             corpus: Optional[Corpus] = None) -> dict:
    """Run all arms over the corpus. Returns a metrics dict (JSON-safe)."""
    corpus = corpus or build_corpus()

    with tempfile.TemporaryDirectory() as tmp:
        key_to_id = _setup(Path(tmp) / "eval.db", corpus, config_overrides)
        id_to_content = {key_to_id[m.key]: m.content for m in corpus.memories}

        from maasv.core.retrieval import find_similar_memories, get_tiered_memory_context
        from maasv.core.store import get_all_active

        full_context_text = "\n".join(m["content"] for m in get_all_active())
        full_context_tokens = approx_tokens(full_context_text)

        per_question = []
        for qa in corpus.qas:
            gold_ids = {key_to_id[key] for key in qa.gold_keys}
            gold_contents = [id_to_content[gid] for gid in gold_ids]

            results = find_similar_memories(qa.question, limit=k)
            returned_ids = [r["id"] for r in results]
            scores = _score_ranked(returned_ids, gold_ids, k)
            retrieval_tokens = approx_tokens("\n".join(r["content"] for r in results))

            context = get_tiered_memory_context(query=qa.question)
            tiered_hit = any(g in context for g in gold_contents)
            tiered_tokens = approx_tokens(context)

            budgeted = get_tiered_memory_context(
                query=qa.question, token_budget=BUDGET_TOKENS, compact=True
            )
            budget_hit = any(g in budgeted for g in gold_contents)
            budget_tokens = approx_tokens(budgeted)

            per_question.append({
                "question": qa.question,
                "type": qa.qa_type,
                **scores,
                "retrieval_tokens": retrieval_tokens,
                "tiered_hit": tiered_hit,
                "tiered_tokens": tiered_tokens,
                "budget_hit": budget_hit,
                "budget_tokens": budget_tokens,
            })

    n = len(per_question)
    types = sorted({q["type"] for q in per_question})

    def _agg(rows):
        m = len(rows)
        return {
            "n": m,
            "recall_at_1": sum(r["hit_at_1"] for r in rows) / m,
            f"recall_at_{k}": sum(r["hit_at_k"] for r in rows) / m,
            "mrr": sum(r["rr"] for r in rows) / m,
            "mean_tokens": sum(r["retrieval_tokens"] for r in rows) / m,
        }

    return {
        "k": k,
        "n_questions": n,
        "n_memories": len(corpus.memories),
        "retrieval": _agg(per_question),
        "retrieval_by_type": {t: _agg([q for q in per_question if q["type"] == t]) for t in types},
        "tiered": {
            "gold_in_context_rate": sum(q["tiered_hit"] for q in per_question) / n,
            "mean_tokens": sum(q["tiered_tokens"] for q in per_question) / n,
        },
        "tiered_budget": {
            "budget": BUDGET_TOKENS,
            "gold_in_context_rate": sum(q["budget_hit"] for q in per_question) / n,
            "mean_tokens": sum(q["budget_tokens"] for q in per_question) / n,
        },
        "full_context": {
            "gold_in_context_rate": 1.0,
            "mean_tokens": full_context_tokens,
        },
        "per_question": per_question,
    }


def format_report(metrics: dict) -> str:
    """Human-readable eval report."""
    k = metrics["k"]
    lines = [
        f"maasv eval — {metrics['n_questions']} questions over {metrics['n_memories']} memories (k={k})",
        "",
        f"{'arm / type':<22}{'R@1':>7}{'R@' + str(k):>7}{'MRR':>7}{'tokens':>9}",
        "-" * 52,
    ]
    r = metrics["retrieval"]
    lines.append(f"{'retrieval (all)':<22}{r['recall_at_1']:>7.2f}{r[f'recall_at_{k}']:>7.2f}"
                 f"{r['mrr']:>7.2f}{r['mean_tokens']:>9.0f}")
    for t, tr in metrics["retrieval_by_type"].items():
        lines.append(f"{'  ' + t:<22}{tr['recall_at_1']:>7.2f}{tr[f'recall_at_{k}']:>7.2f}"
                     f"{tr['mrr']:>7.2f}{tr['mean_tokens']:>9.0f}")
    t = metrics["tiered"]
    tb = metrics["tiered_budget"]
    f = metrics["full_context"]
    lines += [
        "-" * 52,
        f"{'tiered context':<22}{'':>7}{t['gold_in_context_rate']:>7.2f}{'':>7}{t['mean_tokens']:>9.0f}",
        f"{'tiered (budget ' + str(tb['budget']) + ')':<22}{'':>7}{tb['gold_in_context_rate']:>7.2f}{'':>7}{tb['mean_tokens']:>9.0f}",
        f"{'full-context control':<22}{'':>7}{f['gold_in_context_rate']:>7.2f}{'':>7}{f['mean_tokens']:>9.0f}",
    ]
    misses = [q for q in metrics["per_question"] if not q["hit_at_k"]]
    if misses:
        lines += ["", f"misses @{k}:"]
        for q in misses:
            lines.append(f"  [{q['type']}] {q['question']}")
    return "\n".join(lines)
