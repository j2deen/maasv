"""
maasv quickstart: store memories, build a knowledge graph, retrieve with context.

This example uses mock providers so you can run it without any API keys or
embedding models. In production, you'd swap these for real providers
(see the EmbedProvider and LLMProvider protocols in maasv/protocols.py).

    python examples/quickstart.py
"""

import hashlib
import tempfile
from pathlib import Path

import maasv
from maasv.config import MaasvConfig


# -- Step 0: Implement the two provider protocols ---------------------------
# maasv doesn't bundle an LLM (and it's optional — only extraction, inference,
# and review need one). Embeddings default to the built-in Ollama provider,
# or you implement EmbedProvider yourself. These mocks let you run the example
# without any external dependencies (no Ollama, no API keys).

class LocalEmbedProvider:
    """Hash-based embeddings for demo purposes. Not useful for real retrieval."""

    def __init__(self, dims: int = 64):
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        vec = [b / 255.0 for b in h]
        while len(vec) < self.dims:
            vec.extend(vec)
        return vec[: self.dims]

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)


class LocalLLMProvider:
    """Stub LLM that returns empty JSON. Entity extraction won't produce
    results with this, but everything else works fine."""

    def call(self, messages, model, max_tokens, source=""):
        return "[]"


# -- Step 1: Initialize maasv ----------------------------------------------

with tempfile.TemporaryDirectory() as tmp:
    db_path = Path(tmp) / "demo.db"

    config = MaasvConfig(
        db_path=db_path,
        embed_dims=64,
        cross_encoder_enabled=False,
    )

    maasv.init(
        config=config,
        llm=LocalLLMProvider(),
        embed=LocalEmbedProvider(dims=64),
    )

    # -- Step 2: Store some memories ----------------------------------------

    from maasv.core.store import store_memory

    store_memory("Gabby is my wife", category="family", subject="Gabby")
    store_memory("Levi is 8, Dani is 5", category="family", subject="Kids")
    store_memory("We live on the Upper West Side", category="identity", subject="Home")
    store_memory("I prefer Python over JavaScript", category="preference")
    store_memory("Started building Doris in January 2025", category="project", subject="Doris")
    store_memory("Doris uses Claude as her brain", category="project", subject="Doris")

    print("Stored 6 memories.\n")

    # -- Step 3: Build the knowledge graph ----------------------------------

    from maasv.core.graph import find_or_create_entity, add_relationship

    adam = find_or_create_entity("Adam", "person")
    gabby = find_or_create_entity("Gabby", "person")
    doris = find_or_create_entity("Doris", "project")
    claude = find_or_create_entity("Claude", "technology")
    uws = find_or_create_entity("Upper West Side", "place")

    add_relationship(adam, "married_to", object_id=gabby)
    add_relationship(adam, "works_on", object_id=doris)
    add_relationship(adam, "lives_in", object_id=uws)
    add_relationship(doris, "uses_tech", object_id=claude)

    print("Built knowledge graph: 5 entities, 4 relationships.\n")

    # -- Step 4: Retrieve with 3-signal fusion ------------------------------

    from maasv.core.retrieval import find_similar_memories

    results = find_similar_memories("Tell me about Doris", limit=3)
    print("Query: 'Tell me about Doris'")
    for mem in results:
        print(f"  [{mem['category']}] {mem['content']}")

    print()

    # -- Step 5: Tiered context (what you'd inject into an LLM prompt) ------

    from maasv.core.retrieval import get_tiered_memory_context

    context = get_tiered_memory_context(query="family dinner plans")
    print("Tiered context for 'family dinner plans':")
    print(context)
    print()

    # -- Step 6: Log a decision to the wisdom system ------------------------

    from maasv.core.wisdom import log_reasoning, record_outcome, add_feedback

    wisdom_id = log_reasoning(
        action_type="restaurant_recommendation",
        reasoning="User asked for Italian near home. Picked Carmine's because "
                  "it's family-style and they have kids.",
        context="Family dinner, Upper West Side",
    )

    record_outcome(wisdom_id, "success", "They loved it, kids ate well")
    add_feedback(wisdom_id, score=5, notes="Perfect pick for family dinner")

    print("Logged wisdom entry with outcome and feedback.")
    print("\nDone. In production, the sleep worker would now run entity extraction,")
    print("inference, review, and hygiene jobs in the background.")
