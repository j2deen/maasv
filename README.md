# maasv

**The cognition layer for AI agents.**

Your AI remembers things now. It knows your name, your preferences, bits and pieces from past conversations. That's table stakes. But it doesn't *think* about what it knows. It doesn't connect dots across sources, notice patterns over time, or learn from what worked and what didn't. It remembers what you said. maasv understands what you meant.

maasv is the layer that closes that gap. Cognition for AI agents. Not just storage and retrieval. A full lifecycle that extracts, structures, connects, consolidates, and prunes knowledge over time.

## What changes

Without maasv, your agent answers questions. With it, your agent *notices things*.

It remembers that the person you're meeting tomorrow was mentioned in a conversation three weeks ago, and surfaces the context before you ask. It connects a complaint from a customer in March to a feature request from their team in June. It knows you tried a particular approach before and it didn't work, so it suggests something different this time.

This isn't retrieval-augmented generation. RAG finds documents. maasv builds understanding. A living knowledge graph that grows, consolidates, and prunes itself over time. Data comes in from disparate sources, gets structured into entities and relationships, and what comes back out isn't just information. It's insight.

Your agent doesn't just have access to what it's been told. It has *perspective*.

## Where this came from

I built [Doris](https://github.com/ascottbell/doris), a personal AI assistant for me and my family. She helps with schedules, remembers preferences, keeps track of projects, flags emails, sets reminders, sends directions, knows the kids' birthdays, and she sends me relevant messages at the right times, all proactively.

The memory system ended up being the most interesting part. Not the LLM, not the tool calling, not the integrations. The memory. Because memory is what makes an agent feel like it actually knows you.

So I pulled it out into its own package. maasv is the engine that powers Doris's cognition, and now it can power yours too.

## The lifecycle

Most memory tools store and retrieve. That's two steps. maasv owns six:

**Extract.** Entities, relationships, and facts are pulled from conversations by your LLM. People, places, projects, technologies, and how they connect to each other. Not keywords. Structure.

**Store.** Memories are embedded, categorized, and deduplicated on the way in. Each one carries metadata: confidence, importance, subject, access history. Not a vector dump.

**Consolidate.** During idle time, maasv merges near-duplicates, clusters related memories, resolves vague references to specific entities, and pre-computes common graph paths. Your agent's understanding gets sharper while nobody's using it.

**Retrieve.** Three signals fused together: dense vector search (semantic similarity), BM25 keyword matching (exact terms via FTS5), and graph connectivity (1-hop entity expansion). Merged with Reciprocal Rank Fusion, optionally reranked by a cross-encoder. This is how your agent finds the thing it didn't know it was looking for.

**Decay.** Memories that stop being accessed lose confidence over time. Protected categories (identity, family, core preferences) are exempt. Everything else has to earn its place.

**Forget.** Stale, low-confidence memories are pruned. Orphaned entities are cleaned up. The knowledge graph stays lean. This is the part most memory systems skip, and it's why most memory systems get worse over time instead of better.

## Install

```bash
pip install maasv
```

One dependency: `sqlite-vec` for vector search. Everything runs locally in a single SQLite database. No external services, no API keys for the engine itself.

Optional, for cross-encoder reranking:
```bash
pip install "maasv[reranking]"
```

## Quick start

maasv doesn't bundle an LLM or embedding model. You bring your own by implementing two protocols:

```python
from pathlib import Path
import maasv
from maasv.config import MaasvConfig

# Your providers implement LLMProvider and EmbedProvider protocols
# (see maasv/protocols.py -- it's two methods each)
config = MaasvConfig(db_path=Path("memory.db"), embed_dims=1024)
maasv.init(config=config, llm=my_llm, embed=my_embedder)

# Store a memory
from maasv.core.store import store_memory
store_memory("Alice prefers morning meetings", category="preferences", subject="Alice")

# Build the graph
from maasv.core.graph import find_or_create_entity, add_relationship
alice = find_or_create_entity("Alice", "person")
project_x = find_or_create_entity("ProjectX", "project")
add_relationship(alice, "works_on", object_id=project_x)

# Retrieve (3-signal fusion)
from maasv.core.retrieval import find_similar_memories
results = find_similar_memories("who's working on ProjectX?", limit=5)

# Or get tiered context for your LLM prompt
from maasv.core.retrieval import get_tiered_memory_context
context = get_tiered_memory_context(query="meeting prep for Alice")
```

See [`examples/quickstart.py`](examples/quickstart.py) for a complete runnable example with mock providers.

## Architecture

```
Your Agent
    |
    v
maasv.init(config, llm, embed)
    |
    +-- core/
    |   +-- store.py        Memory CRUD (store, supersede, delete)
    |   +-- retrieval.py    3-signal retrieval + reranking + tiered context
    |   +-- graph.py        Knowledge graph (entities, relationships, traversal)
    |   +-- wisdom.py       Experiential learning (log, outcome, feedback)
    |   +-- db.py           SQLite + sqlite-vec, migrations, access tracking
    |   +-- reranker.py     Optional cross-encoder (lazy-loaded)
    |
    +-- extraction/
    |   +-- entity_extraction.py   LLM-powered entity/relationship extraction
    |
    +-- lifecycle/
        +-- worker.py         Sleep-time job queue (background thread)
        +-- memory_hygiene.py Dedup, prune, consolidate, entity cleanup
        +-- reorganize.py     Graph optimization, path caching, orphan cleanup
        +-- inference.py      Resolve vague references to specific entities
        +-- review.py         Second-pass conversation analysis
```

Everything talks to one SQLite database. No Redis, no Postgres, no external services. The entire state of an agent's memory is a single `.db` file you can copy, back up, or throw away.

## The provider protocols

maasv never imports an LLM or embedding library directly. You implement two protocols:

```python
class LLMProvider(Protocol):
    def call(self, messages: list[dict], model: str, max_tokens: int, source: str = "") -> str: ...

class EmbedProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def embed_query(self, text: str) -> list[float]: ...
```

This means maasv works with any model from any provider. Claude, GPT, Gemini, local models, whatever. Your agent, your choice.

## Configuration

```python
from maasv.config import MaasvConfig

config = MaasvConfig(
    db_path=Path("memory.db"),
    embed_dims=1024,                    # Must match your embedding model

    # Models (names passed to your LLMProvider -- it decides what to do with them)
    extraction_model="claude-haiku-4-5-20251001",
    inference_model="claude-haiku-4-5-20251001",
    review_model="claude-haiku-4-5-20251001",

    # Hygiene tuning
    similarity_threshold=0.95,          # Dedup threshold (cosine)
    stale_days=30,                      # Prune after N days
    min_confidence_threshold=0.5,       # Prune below this confidence
    protected_categories={"identity", "family"},  # Never auto-delete

    # Cross-encoder (opt-in)
    cross_encoder_enabled=False,
    cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",

    # Sleep worker timing
    idle_threshold_seconds=30,
    idle_check_interval=5,

    # Known entities (helps extraction avoid duplicates)
    known_entities={"Alice": "person", "ProjectX": "project"},
)
```

## Status

This is running in production powering [Doris](https://github.com/ascottbell/doris), but the public API may shift as more people use it. The core concepts (memory, graph, retrieval, wisdom, lifecycle) are stable. The edges are still being refined.

## License

Business Source License 1.1. Free for personal, internal, educational, and non-commercial use. Commercial use requires a license. Contact admin@maasv.ai. Converts to Apache 2.0 on 2030-02-16. See [LICENSE](LICENSE) for details.

## Related

- **[Doris](https://github.com/ascottbell/doris)** The AI assistant maasv was built for. If maasv is the cognition layer, Doris is the person using it.
