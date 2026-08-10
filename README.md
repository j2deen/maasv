# maasv

**A cognition layer for AI agents.**

maasv gives your agent a real memory — not just storage and retrieval, but a full lifecycle that extracts, structures, connects, consolidates, prunes, and learns from knowledge over time. Entities and relationships are pulled from conversations, organized into a knowledge graph, and actively maintained in the background. What comes back out when you query isn't just relevant documents — it's structured understanding with context.

## What it does

Your agent remembers that the person you're meeting tomorrow was mentioned in a conversation three weeks ago, and surfaces the context before you ask. It connects a complaint from a customer in March to a feature request from their team in June. It knows you tried a particular approach before and it didn't work, so it suggests something different this time.

The knowledge graph grows, consolidates, and prunes itself over time. Data comes in from disparate sources, gets structured into entities and relationships, and the connections between them become queryable. Your agent builds perspective across conversations, not just within them.

## Where this came from

I built Doris, a personal AI assistant for me and my family. She helps with schedules, remembers preferences, keeps track of projects, flags emails, sets reminders, sends directions, knows the kids' birthdays, and she sends me relevant messages at the right times, all proactively.

The memory system ended up being the most interesting part. Not the LLM, not the tool calling, not the integrations. The memory. Because memory is what makes an agent feel like it actually knows you.

So I pulled it out into its own package. maasv is the engine that powers Doris's cognition, and now it can power yours too.

## The lifecycle

Most memory tools store and retrieve. That's two steps. maasv owns seven:

**Extract.** Entities, relationships, and facts are pulled from conversations by your LLM. People, places, projects, technologies, and how they connect to each other. Not keywords. Structure.

**Store.** Memories are embedded, categorized, and deduplicated on the way in. Each one carries metadata: confidence, importance, subject, and access history.

**Consolidate.** During idle time, maasv merges near-duplicates, clusters related memories, resolves vague references to specific entities, and pre-computes common graph paths. Your agent's understanding gets sharper while nobody's using it.

**Retrieve.** Three signals fused together: dense vector search (semantic similarity), BM25 keyword matching (exact terms via FTS5), and graph connectivity (1-hop entity expansion). Merged with Reciprocal Rank Fusion, scored for importance, then optionally refined by a cross-encoder in a two-stage rerank — importance decides which memories are candidates, the cross-encoder only reorders within that set. This is how your agent finds the thing it didn't know it was looking for.

**Decay.** Memories that stop being accessed lose confidence over time. Protected categories (identity, family, core preferences) are exempt. Everything else has to earn its place.

**Forget.** Stale, low-confidence memories are pruned. Orphaned entities are cleaned up. The knowledge graph stays lean. Without active forgetting, memory systems tend to get noisier over time — maasv gets sharper.

**Learn.** Two loops. The wisdom system captures reasoning before actions, records outcomes, and takes feedback, so past experience informs future decisions. And a learned ranker — a small neural network (81 parameters, trained by a bundled pure-Python autograd engine) — learns from your actual retrieval usage which memories matter. It runs in shadow mode until its rankings beat the heuristic, and only then takes over.

## Install

```bash
pip install maasv
```

One dependency: `sqlite-vec` for vector search. Everything runs locally in a single SQLite database. No external services, no API keys for the engine itself.

Optional extras:
```bash
pip install "maasv[reranking]"   # cross-encoder reranking (pulls in torch, ~2GB)
pip install "maasv[mcp]"         # MCP server for Claude Desktop, Claude Code, etc.
pip install "maasv[server]"      # REST API server (FastAPI + uvicorn)
pip install "maasv[anthropic]"   # Anthropic LLM provider for the servers
pip install "maasv[openai]"      # OpenAI LLM/embedding provider for the servers
pip install "maasv[voyage]"      # Voyage embedding provider for the servers
pip install "maasv[all]"         # anthropic + mcp + server + voyage
```

## Quick start

maasv doesn't bundle an LLM — you bring your own by implementing a one-method protocol, and it's optional (only entity extraction, inference, and review need it). Embeddings default to a built-in Ollama provider (`qwen3-embedding:8b` on `localhost:11434`), or you implement the two-method `EmbedProvider` protocol yourself:

```python
from pathlib import Path
import maasv
from maasv.config import MaasvConfig

config = MaasvConfig(db_path=Path("memory.db"), embed_dims=1024)

# Simplest init: local Ollama embeddings, no LLM (extraction/inference disabled)
maasv.init(config=config)

# Or bring your own providers (see maasv/protocols.py)
maasv.init(config=config, llm=my_llm, embed=my_embedder)

# Store a memory
from maasv.core.store import store_memory
store_memory("Alice prefers morning meetings", category="preference", subject="Alice")

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
    |   +-- store.py           Memory CRUD (store, supersede, delete)
    |   +-- retrieval.py       3-signal retrieval + reranking + tiered context
    |   +-- graph.py           Knowledge graph (entities, relationships, traversal)
    |   +-- wisdom.py          Experiential learning (log, outcome, feedback)
    |   +-- db.py              SQLite + sqlite-vec, migrations, access tracking
    |   +-- reranker.py        Optional cross-encoder (lazy-loaded)
    |   +-- learned_ranker.py  Learned ranking model (shadow mode, graduation gates)
    |   +-- autograd.py        Minimal autograd engine (micrograd port, zero deps)
    |
    +-- extraction/
    |   +-- entity_extraction.py   LLM-powered entity/relationship extraction
    |
    +-- lifecycle/
    |   +-- worker.py         Sleep-time job queue (background thread)
    |   +-- memory_hygiene.py Dedup, prune, consolidate, entity cleanup
    |   +-- reorganize.py     Graph optimization, path caching, orphan cleanup
    |   +-- inference.py      Resolve vague references to specific entities
    |   +-- review.py         Second-pass conversation analysis
    |   +-- learn.py          Label retrieval logs, train ranker, check graduation
    |
    +-- providers/
    |   +-- ollama.py         Built-in Ollama embed provider (default)
    |
    +-- mcp_server/           MCP server: 20 tools (memory, graph, wisdom, extraction)
    +-- server/               REST API server (FastAPI: memory, graph, wisdom,
                              extraction, health routers)
```

Everything talks to one SQLite database. No Redis, no Postgres, no external services. The entire state of an agent's memory is a single `.db` file you can copy, back up, or throw away.

## The provider protocols

maasv never imports an LLM library directly. You implement two protocols:

```python
class LLMProvider(Protocol):
    def call(self, messages: list[dict], model: str, max_tokens: int, source: str = "") -> str: ...

class EmbedProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def embed_query(self, text: str) -> list[float]: ...  # optional; falls back to embed()
```

This means maasv works with any model from any provider. Claude, GPT, Gemini, local models, whatever. Your agent, your choice.

Notes:

- `llm` is optional. Without it, storage, retrieval, graph, and wisdom all work — only LLM-powered features (entity extraction, inference, review) are unavailable.
- `embed` is optional too: it defaults to the built-in Ollama provider. You can pass the shortcut string `"ollama"` with `embed_model=` / `embed_base_url=` overrides, or your own `EmbedProvider`.
- At init, maasv validates that your embedder's output matches `config.embed_dims` and warns if vectors aren't L2-normalized (retrieval thresholds assume they are).

## Configuration

```python
from maasv.config import MaasvConfig

config = MaasvConfig(
    db_path=Path("memory.db"),
    embed_dims=1024,                    # Must match your embedding model
    embed_model="qwen3-embedding:8b",   # Recorded in DB to prevent model mismatch

    # Models (names passed to your LLMProvider -- it decides what to do with them)
    extraction_model="claude-haiku-4-5-20251001",
    inference_model="claude-haiku-4-5-20251001",
    review_model="claude-haiku-4-5-20251001",

    # Hygiene tuning
    similarity_threshold=0.95,          # Dedup threshold (cosine)
    stale_days=30,                      # Prune after N days
    min_confidence_threshold=0.5,       # Prune below this confidence
    cluster_similarity=0.85,            # Consolidation cluster threshold
    protected_categories={"identity", "family"},  # Never auto-delete
    protected_subjects=set(),           # Subjects that are never auto-deleted
    backup_dir=None,                    # Optional: back up DB before hygiene runs
    hygiene_log_path=None,              # Optional: audit log for hygiene actions

    # Retrieval tuning
    diversity_threshold=0.0,            # Jaccard dedup (0.0 = off, 0.7 = moderate)
    category_priority={"identity": 1, "family": 2, "preference": 3},  # Tiered context order

    # Cross-encoder (opt-in)
    cross_encoder_enabled=False,
    cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",

    # Learned ranker (on by default, shadow mode until it graduates)
    learned_ranker_enabled=True,
    learned_ranker_shadow_mode=True,
    learned_ranker_min_samples=100,     # Heuristic fallback below this
    learned_ranker_auto_graduate=False, # Auto-promote when graduation gates pass

    # Sleep worker timing
    idle_threshold_seconds=30,
    idle_check_interval=5,

    # Graph
    extra_predicates=set(),             # Extend the built-in predicate allowlist
    known_entities={"Alice": "person", "ProjectX": "project"},  # Helps extraction avoid duplicates

    # Wisdom
    action_families={},                 # Action type groupings for "similar enough" matching
)
```

See `maasv/config.py` for the full list, including learned ranker training and graduation thresholds.

## Privacy: sensitivity-split routing

maasv is built for setups where the memory corpus contains things that must never leave the machine — desktop context, enterprise records, family details. Everything lives in one local SQLite file, and two hooks make the hybrid local+cloud pattern work. The principle: **split by sensitivity, not by capability**.

**1. Route LLM work by task.** Your `LLMProvider.call()` receives a per-task model name (`extraction_model`, `inference_model`, `review_model` — all just strings you configure). Point extraction and inference — the tasks that see raw conversation text — at a local model, and let a frontier model handle only whatever your agent does with retrieved facts:

```python
class SplitProvider:
    def call(self, messages, model, max_tokens, source=""):
        if model.startswith("ollama/"):
            return call_ollama(messages, model.removeprefix("ollama/"), max_tokens)
        return call_cloud(messages, model, max_tokens)

config = MaasvConfig(
    db_path=Path("memory.db"),
    extraction_model="ollama/qwen3:8b",   # raw text stays local
    inference_model="ollama/qwen3:8b",    # raw text stays local
    review_model="ollama/qwen3:8b",       # raw text stays local
)
```

With this config, no raw conversation content is ever sent to a cloud API by maasv itself.

**2. Redact at the retrieval boundary.** `redact_output` is a callback applied to memory content the moment it leaves maasv toward a prompt (`find_similar_memories`, `get_tiered_memory_context`, `search_fts`, `find_by_subject`). Stored data is never modified — only the outbound copy. Wire in any scrubber; [Microsoft Presidio](https://microsoft.github.io/presidio/) is the standard open-source choice:

```python
config = MaasvConfig(
    db_path=Path("memory.db"),
    redact_output=lambda text: presidio_anonymize(text),  # or your own regex scrubber
)
```

If the hook raises, maasv fails closed: the text is replaced with `[redacted]` rather than passed through.

Combined, the cloud model only ever sees redacted, already-extracted facts — never the corpus, never raw conversations.

## Servers

You don't have to embed maasv as a library — it also ships two servers (both optional extras).

**MCP server** (`pip install "maasv[mcp]"`) exposes the cognition layer to any MCP client — Claude Desktop, Claude Code, ChatGPT — as 20 tools across 4 domains: memory (6), graph (9), wisdom (4), and extraction (1).

```bash
maasv-mcp                              # STDIO (local clients)
MAASV_TRANSPORT=http maasv-mcp         # HTTP (remote; requires MAASV_AUTH_TOKEN)
```

**REST server** (`pip install "maasv[server]"`) is a FastAPI app with routers for memory, graph, wisdom, extraction, and health, plus optional bearer-token auth.

```bash
maasv-server                           # defaults to 127.0.0.1:18790
```

Both configure via `MAASV_`-prefixed environment variables (or a `.env` file): `MAASV_DB_PATH`, `MAASV_LLM_PROVIDER` (`anthropic` or `openai`), `MAASV_LLM_API_KEY`, `MAASV_EMBED_PROVIDER` (`ollama`, `voyage`, or `openai`), and friends. Since the servers own the process, they construct providers for you from those variables — this is where the `anthropic`, `openai`, and `voyage` extras come in.

## Status

This is running in production powering Doris, but the public API may shift as more people use it. The core concepts (memory, graph, retrieval, wisdom, lifecycle) are stable. The edges are still being refined.

> Note: this repository preserves maasv 0.2.0. The original upstream (`ascottbell/maasv`) and the Doris repository are no longer public.

## License

Business Source License 1.1. Free for personal, internal, educational, and non-commercial use. Commercial use requires a license. Contact admin@maasv.ai. Converts to Apache 2.0 on 2030-02-16. See [LICENSE](LICENSE) for details.

## Related

- **Doris** — The AI assistant maasv was built for. If maasv is the cognition layer, Doris is the person using it. (Repository no longer public.)
