"""
Cross-encoder reranker for maasv retrieval.

Lazy-loads a small cross-encoder model on first use. The model scores
(query, document) pairs for relevance, providing much better ranking than
vector distance alone.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (22MB, ~50ms for 50 pairs on M4)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_reranker = None
_reranker_failed = False


def _get_reranker():
    """Lazy-load the cross-encoder model. Returns None if unavailable."""
    global _reranker, _reranker_failed

    if _reranker is not None:
        return _reranker

    if _reranker_failed:
        return None

    import maasv
    config = maasv.get_config()
    if not config.cross_encoder_enabled:
        return None

    try:
        from sentence_transformers import CrossEncoder
        model_name = config.cross_encoder_model
        logger.info(f"Loading cross-encoder model: {model_name}")
        _reranker = CrossEncoder(model_name)
        logger.info("Cross-encoder loaded successfully")
        return _reranker
    except ImportError:
        logger.warning("sentence-transformers not installed — cross-encoder disabled")
        _reranker_failed = True
        return None
    except Exception:
        logger.error("Failed to load cross-encoder model", exc_info=True)
        _reranker_failed = True
        return None


def rerank(
    query: str,
    candidates: list[dict],
    content_key: str = "content",
) -> Optional[list[float]]:
    """
    Score candidates against the query using the cross-encoder.

    Returns a list of float scores (one per candidate), or None if the
    cross-encoder is unavailable. Higher scores = more relevant.

    Scores are raw logits from the cross-encoder (can be negative).
    Callers should use them for relative ranking, not as absolute values.
    """
    reranker = _get_reranker()
    if reranker is None:
        return None

    if not candidates:
        return []

    pairs = [(query, mem.get(content_key, "")) for mem in candidates]

    try:
        scores = reranker.predict(pairs)
        return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
    except Exception:
        logger.error("Cross-encoder prediction failed", exc_info=True)
        return None
