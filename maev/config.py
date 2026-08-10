"""
maev configuration.

All paths, model names, and tuning parameters are set here.
No hardcoded values in the rest of the package.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


@dataclass
class MaevConfig:
    """Configuration for the maev cognition layer."""

    # Database
    db_path: Path

    # Embedding
    embed_dims: int = 1024
    embed_model: str = "qwen3-embedding:8b"  # recorded in DB to prevent model mismatch

    # Models (passed to LLMProvider.call — provider decides how to route).
    # Defaults current as of August 2026: claude-sonnet-5 balances quality and
    # cost for background extraction/inference/review. Swap for
    # claude-haiku-4-5 (fastest/cheapest) or claude-opus-5 (highest quality),
    # or any local model name your provider understands.
    extraction_model: str = "claude-sonnet-5"
    inference_model: str = "claude-sonnet-5"
    review_model: str = "claude-sonnet-5"

    # Memory hygiene
    backup_dir: Optional[Path] = None
    max_hygiene_backups: int = 3
    protected_categories: set[str] = field(default_factory=lambda: {"identity", "family"})
    protected_subjects: set[str] = field(default_factory=set)

    # Hygiene thresholds
    similarity_threshold: float = 0.95
    stale_days: int = 30
    min_confidence_threshold: float = 0.5
    cluster_similarity: float = 0.85

    # Retrieval tuning
    # Jaccard content-overlap ceiling between selected results (0.0 disables).
    # Default 0.7: near-duplicate memories stop crowding out distinct facts —
    # on a 176-memory eval corpus, repeated boilerplate otherwise fills 3 of
    # the top 5 slots for some queries.
    diversity_threshold: float = 0.7
    graph_slot_injection: bool = False  # Force-inject a graph result into last slot

    # Graph retrieval signal: "ppr" (Personalized PageRank, multi-hop; falls back
    # to one_hop when the graph yields nothing) or "one_hop" (legacy expansion)
    graph_retrieval: str = "ppr"
    ppr_alpha: float = 0.5        # walk-continuation probability (1-alpha restarts at seeds)
    ppr_iterations: int = 20      # power-iteration cap (converges early via L1 check)
    ppr_max_nodes: int = 500      # BFS subgraph bound
    ppr_top_entities: int = 12    # entities mapped back to memories

    # Weight of the normalized RRF fused score in final ranking. Lets a memory
    # that ranks highly in BM25/graph (e.g. multi-hop PPR hits) compete with
    # lexically-closer vector matches. 0.0 = legacy flat agreement bonus.
    rrf_rank_weight: float = 0.15

    # Fusion rescue: candidates with NO vector-search presence (graph/BM25-only
    # hits — e.g. multi-hop PPR results lexically far from the query) sort after
    # every vector candidate, so on corpora larger than the vector window they
    # can never reach the top-k. Candidates ranking in the top N of the graph
    # or BM25 signal claim up to fusion_rescue_slots tail slots of the result
    # (strongest fused score first). top_n=0 or slots=0 disables.
    fusion_rescue_top_n: int = 5
    fusion_rescue_slots: int = 2

    # Final reorder of the SELECTED top-k (heuristic path only — the
    # cross-encoder owns ordering when enabled). Selection guarantees
    # relevance; within the k, fused multi-signal strength and query-term
    # overlap order better than raw vector similarity, which lets one-signal
    # lexical near-misses steal rank 1. Weights: vector sim / normalized
    # fused RRF / query-term overlap. rerank_selected=False restores the
    # importance-score order.
    rerank_selected: bool = True
    rerank_selected_wv: float = 0.25
    rerank_selected_wr: float = 2.0
    rerank_selected_wq: float = 1.0
    # Entity novelty (multi-hop answer detection): a memory mentioning
    # graph-relevant entities ABSENT from the query text likely carries the
    # answer; one restating the queried entity is likely a bridge fact
    # echoing the question. wn rewards novel entity mass, wk penalizes
    # known (query-mentioned) entity mass.
    rerank_selected_wn: float = 1.0
    rerank_selected_wk: float = 0.5

    # Cross-encoder reranking (opt-in: requires sentence-transformers + torch ~2GB)
    cross_encoder_enabled: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Sleep worker
    idle_threshold_seconds: int = 30
    idle_check_interval: int = 5

    # Memory evolution (A-MEM style): new memories link to related older ones
    # during sleep-time; optionally the LLM re-tags the older side.
    evolve_enabled: bool = True
    evolve_link_threshold: float = 0.70  # cosine floor for a link (below dedup's 0.95)
    evolve_max_links: int = 10           # bound on related_ids per memory (and KNN k)
    evolve_batch_size: int = 100         # new memories per run
    evolve_llm_refresh: bool = False     # LLM tag refresh of linked older memories

    # Learned ranker
    learned_ranker_enabled: bool = True
    learned_ranker_min_samples: int = 100
    learned_ranker_shadow_mode: bool = True
    learned_ranker_lr: float = 0.01
    learned_ranker_max_steps: int = 50
    learned_ranker_ips_clamp: float = 50.0
    learned_ranker_auto_graduate: bool = False
    learned_ranker_graduation_min_comparisons: int = 50
    learned_ranker_graduation_min_ndcg: float = 0.5
    learned_ranker_graduation_min_tau: float = -0.3
    learned_ranker_graduation_max_tau_std: float = 0.3

    # Known entities for extraction prompts (name -> type)
    known_entities: dict[str, str] = field(default_factory=dict)

    # Hygiene log path (optional — if None, no log file written)
    hygiene_log_path: Optional[Path] = None

    # Extra predicates to extend VALID_PREDICATES (for host apps with existing data)
    extra_predicates: set[str] = field(default_factory=set)

    # Extra single-valued predicates: new facts auto-invalidate the previous
    # active fact for the same (subject, predicate). See graph.FUNCTIONAL_PREDICATES.
    extra_functional_predicates: set[str] = field(default_factory=set)

    # Action type groupings for wisdom "similar enough" matching
    action_families: dict[str, list[str]] = field(default_factory=dict)

    # Output redaction hook for sensitivity-split deployments: applied to
    # memory content/subject at the retrieval boundary (find_similar_memories,
    # get_tiered_memory_context, search_fts, find_by_subject) — the text that
    # leaves maev toward an LLM prompt. Stored data is never modified. Wire a
    # PII scrubber (e.g. Presidio) here and cloud models only ever see
    # redacted facts while local extraction sees raw text.
    redact_output: Optional[Callable[[str], str]] = None

    # Category priority for tiered memory context (lower = higher priority)
    category_priority: dict[str, int] = field(default_factory=lambda: {
        'identity': 1,
        'family': 2,
        'preference': 3,
        'project': 4,
        'decision': 5,
        'person': 6,
        'learning': 7,
        'history': 8,
        'home': 9,
        'conversation': 10,
    })
