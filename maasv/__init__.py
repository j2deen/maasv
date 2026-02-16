"""
maasv — Cognition layer for AI agents.

Stores, structures, connects, consolidates, retrieves, decays, forgets, and learns.

Usage:
    import maasv
    from maasv.config import MaasvConfig

    config = MaasvConfig(db_path=Path("data/my.db"))
    maasv.init(config, llm=my_llm_provider, embed=my_embed_provider)

    # Now use maasv.core.store, maasv.core.wisdom, etc.
"""

from maasv.config import MaasvConfig
from maasv.protocols import LLMProvider, EmbedProvider

__version__ = "0.1.0"

_config: MaasvConfig | None = None
_llm: LLMProvider | None = None
_embed: EmbedProvider | None = None
_initialized: bool = False


def init(config: MaasvConfig, llm: LLMProvider, embed: EmbedProvider) -> None:
    """
    Initialize maasv with configuration and providers.

    Must be called before using any maasv functionality.

    Args:
        config: Database path, model names, tuning parameters
        llm: Provider for LLM calls (entity extraction, inference, review)
        embed: Provider for text embeddings (storage, search, hygiene)
    """
    global _config, _llm, _embed, _initialized

    _config = config
    _llm = llm
    _embed = embed
    _initialized = True

    # Initialize database schema
    from maasv.core.store import init_db
    init_db()

    from maasv.core.wisdom import ensure_wisdom_tables
    ensure_wisdom_tables()


def get_config() -> MaasvConfig:
    """Get the current config. Raises if not initialized."""
    if not _initialized or _config is None:
        raise RuntimeError("maasv not initialized. Call maasv.init() first.")
    return _config


def get_llm() -> LLMProvider:
    """Get the LLM provider. Raises if not initialized."""
    if not _initialized or _llm is None:
        raise RuntimeError("maasv not initialized. Call maasv.init() first.")
    return _llm


def get_embed() -> EmbedProvider:
    """Get the embedding provider. Raises if not initialized."""
    if not _initialized or _embed is None:
        raise RuntimeError("maasv not initialized. Call maasv.init() first.")
    return _embed
