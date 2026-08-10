"""
maasv configuration.

All paths, model names, and tuning parameters are set here.
No hardcoded values in the rest of the package.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class MaasvConfig:
    """Configuration for the maasv cognition layer."""

    # Database
    db_path: Path

    # Embedding
    embed_dims: int = 1024
    embed_model: str = "qwen3-embedding:8b"  # recorded in DB to prevent model mismatch

    # Models (passed to LLMProvider.call — provider decides how to route)
    extraction_model: str = "claude-haiku-4-5-20251001"
    inference_model: str = "claude-haiku-4-5-20251001"
    review_model: str = "claude-haiku-4-5-20251001"

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
    diversity_threshold: float = 0.0  # Jaccard threshold for dedup (0.0 = disabled, 0.7 = moderate)
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
