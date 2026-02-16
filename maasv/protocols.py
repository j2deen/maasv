"""
Provider protocols for dependency injection.

Consumers (e.g., Doris) implement these and pass them to maasv.init().
maasv never imports LLM or embedding libraries directly.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Provider for LLM calls (entity extraction, inference, review)."""

    def call(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int,
        source: str = "",
    ) -> str:
        """
        Call an LLM and return the text response.

        Args:
            messages: Chat messages [{"role": "user", "content": "..."}]
            model: Model identifier (e.g., "claude-haiku-4-5-20251001")
            max_tokens: Maximum tokens in response
            source: Label for tracking/billing (e.g., "entity-extract")

        Returns:
            The text content of the LLM response.
        """
        ...


@runtime_checkable
class EmbedProvider(Protocol):
    """Provider for text embeddings (memory storage, search, hygiene)."""

    def embed(self, text: str) -> list[float]:
        """
        Get embedding vector for text.

        Args:
            text: The text to embed

        Returns:
            Embedding vector as list of floats
        """
        ...
