# MAEV

**Memory Architecture for Evolving Agents.** Pronounced "mave."

maev gives your agent a real memory — not just storage and retrieval, but a full lifecycle that extracts, structures, connects, consolidates, prunes, and learns from knowledge over time. Entities and relationships are pulled from conversations, organized into a knowledge graph, and actively maintained in the background. What comes back out when you query isn't just relevant documents — it's structured understanding with context.

## What it does

Your agent remembers that the person you're meeting tomorrow was mentioned in a conversation three weeks ago, and surfaces the context before you ask. It connects a complaint from a customer in March to a feature request from their team in June. It knows you tried a particular approach before and it didn't work, so it suggests something different this time.

The knowledge graph grows, consolidates, and prunes itself over time. Data comes in from disparate sources, gets structured into entities and relationships, and the connections between them become queryable. Your agent builds perspective across conversations, not just within them.

## Where this came from

MAEV is a continuation of [maasv](https://github.com/j2deen/maasv) 0.2.0 by Adam Bell — the memory engine he extracted from Doris, the personal AI assistant he built for his family. His original insight holds up: the memory system, not the LLM or the tool calling, is what makes an agent feel like it actually knows you. Credit for the architecture this project stands on — the single-SQLite design, the provider protocols, the sleep-time lifecycle, the wisdom system — belongs to that work.

The original repository went private at 0.2.0. This project preserves that snapshot and continues development under a new name, with substantial additions since the fork: Personalized PageRank multi-hop graph retrieval, bi-temporal knowledge updates with as-of queries, A-MEM-style memory evolution, a token-budgeted context packer, an output-redaction privacy boundary, a revived BM25 signal, evidence-based result ordering, and a deterministic eval harness that measures all of it.

## The lifecycle

Most memory tools store and retrieve. That's two steps. maev owns seven:

**Extract.** Entities, relationships, and facts are pulled from conversations by your LLM. People, places, projects, technologies, and how they connect to each other. Not keywords. Structure.

**Store.** Memories are embedded, categorized, and deduplicated on the way in. Each one carries metadata: confidence, importance, subject, and access history.

**Consolidate.** During idle time, maev merges near-duplicates, clusters related memories, resolves vague references to specific entities, and pre-computes common graph paths. New memories also evolve old ones (A-MEM style): each newly stored memory gets linked bidirectionally to semantically related older memories, and optionally the LLM re-tags the older side in light of the new context. Your agent's understanding gets sharper while nobody's using it.

**Retrieve.** Three signals fused together: dense vector search (semantic similarity), BM25 keyword matching (exact terms via FTS5), and graph connectivity via Personalized PageRank — a HippoRAG-style multi-hop walk from query entities, so a fact two or three hops away still earns retrieval weight (legacy 1-hop expansion remains as a fallback). Merged with Reciprocal Rank Fusion, scored for importance, then optionally refined by a cross-encoder in a two-stage rerank — importance decides which memories are candidates, the cross-encoder only reorders within that set. A fusion-rescue pass guarantees that strong graph/BM25 hits with no vector-search presence can still claim result slots. This is how your agent finds the thing it didn't know it was looking for.

Knowledge updates are bi-temporal: single-valued predicates (`lives_in`, `works_at`, `married_to`, ...) auto-close the previous fact's validity interval when a conflicting fact arrives, backfilled historical facts land pre-closed in the right slot of the timeline, `get_entity_relationships(as_of=...)` answers "what did we believe on June 1st?", and `get_relationship_history()` returns the full audit trail with change reasons.

**Decay.** Memories that stop being accessed lose confidence over time. Protected categories (identity, family, core preferences) are exempt. Everything else has to earn its place.

**Forget.** Stale, low-confidence memories are pruned. Orphaned entities are cleaned up. The knowledge graph stays lean. Without active forgetting, memory systems tend to get noisier over time — maev gets sharper.

**Learn.** Two loops. The wisdom system captures reasoning before actions, records outcomes, and takes feedback, so past experience informs future decisions. And a learned ranker — a small neural network (81 parameters, trained by a bundled pure-Python autograd engine) — learns from your actual retrieval usage which memories matter. It runs in shadow mode until its rankings beat the heuristic, and only then takes over.

## Install

```bash
pip install maev
```

One dependency: `sqlite-vec` for vector search. Everything runs locally in a single SQLite database. No external services, no API keys for the engine itself.

Optional extras:
```bash
pip install "maev[reranking]"   # cross-encoder reranking (pulls in torch, ~2GB)
pip install "maev[mcp]"         # MCP server for Claude Desktop, Claude Code, etc.
pip install "maev[server]"      # REST API server (FastAPI + uvicorn)
pip install "maev[anthropic]"   # Anthropic LLM provider for the servers
pip install "maev[openai]"      # OpenAI LLM/embedding provider for the servers
pip install "maev[voyage]"      # Voyage embedding provider for the servers
pip install "maev[all]"         # anthropic + mcp + server + voyage
```

## Quick start

maev doesn't bundle an LLM — you bring your own by implementing a one-method protocol, and it's optional (only entity extraction, inference, and review need it). Embeddings default to a built-in Ollama provider (`qwen3-embedding:8b` on `localhost:11434`), or you implement the two-method `EmbedProvider` protocol yourself:

```python
from pathlib import Path
import maev
from maev.config import MaevConfig

config = MaevConfig(db_path=Path("memory.db"), embed_dims=1024)

# Simplest init: local Ollama embeddings, no LLM (extraction/inference disabled)
maev.init(config=config)

# Or bring your own providers (see maev/protocols.py)
maev.init(config=config, llm=my_llm, embed=my_embedder)

# Store a memory
from maev.core.store import store_memory
store_memory("Alice prefers morning meetings", category="preference", subject="Alice")

# Build the graph
from maev.core.graph import find_or_create_entity, add_relationship
alice = find_or_create_entity("Alice", "person")
project_x = find_or_create_entity("ProjectX", "project")
add_relationship(alice, "works_on", object_id=project_x)

# Retrieve (3-signal fusion)
from maev.core.retrieval import find_similar_memories
results = find_similar_memories("who's working on ProjectX?", limit=5)

# Or get tiered context for your LLM prompt — optionally packed to a token
# budget (query-relevant facts first) and compact-grouped by subject
from maev.core.retrieval import get_tiered_memory_context
context = get_tiered_memory_context(query="meeting prep for Alice")
tight = get_tiered_memory_context(query="meeting prep for Alice",
                                  token_budget=150, compact=True)

# Bi-temporal graph queries: time-travel and audit history
from maev.core.graph import get_entity_relationships, get_relationship_history
then = get_entity_relationships(alice, as_of="2026-06-01T00:00:00+00:00")
timeline = get_relationship_history(alice, predicate="works_at")
```

See [`examples/quickstart.py`](examples/quickstart.py) for a complete runnable example with mock providers.

## Architecture

```
Your Agent
    |
    v
maev.init(config, llm, embed)
    |
    +-- core/
    |   +-- store.py           Memory CRUD (store, supersede, delete)
    |   +-- retrieval.py       3-signal retrieval + reranking + tiered context
    |   +-- ppr.py             Personalized PageRank graph signal (multi-hop, pure Python)
    |   +-- graph.py           Knowledge graph (entities, bi-temporal relationships,
    |   |                      as-of queries, history, traversal)
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
    |   +-- evolve.py         Memory evolution: link new memories to related old
    |                         ones, optional LLM re-tagging (A-MEM style)
    |
    +-- providers/
    |   +-- ollama.py         Built-in Ollama embed provider (default)
    |
    +-- mcp_server/           MCP server: 20 tools (memory, graph, wisdom, extraction)
    +-- server/               REST API server (FastAPI: memory, graph, wisdom,
                              extraction, health routers)

evals/                        Dev-only eval harness (not shipped in the package):
                              recall@k / MRR / tokens-injected on a deterministic
                              corpus, with a full-context control arm
```

Everything talks to one SQLite database. No Redis, no Postgres, no external services. The entire state of an agent's memory is a single `.db` file you can copy, back up, or throw away.

## The provider protocols

maev never imports an LLM library directly. You implement two protocols:

```python
class LLMProvider(Protocol):
    def call(self, messages: list[dict], model: str, max_tokens: int, source: str = "") -> str: ...

class EmbedProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def embed_query(self, text: str) -> list[float]: ...  # optional; falls back to embed()
```

This means maev works with any model from any provider. Claude, GPT, Gemini, local models, whatever. Your agent, your choice.

Notes:

- `llm` is optional. Without it, storage, retrieval, graph, and wisdom all work — only LLM-powered features (entity extraction, inference, review) are unavailable.
- `embed` is optional too: it defaults to the built-in Ollama provider. You can pass the shortcut string `"ollama"` with `embed_model=` / `embed_base_url=` overrides, or your own `EmbedProvider`.
- At init, maev validates that your embedder's output matches `config.embed_dims` and warns if vectors aren't L2-normalized (retrieval thresholds assume they are).

## Configuration

```python
from maev.config import MaevConfig

config = MaevConfig(
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
    graph_retrieval="ppr",              # "ppr" (multi-hop PageRank) or "one_hop" (legacy)
    rrf_rank_weight=0.15,               # Fused-rank strength in final scoring
    fusion_rescue_top_n=5,              # Rescue graph/BM25-only hits from this signal depth
    fusion_rescue_slots=2,              # ...into up to this many tail result slots

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
    extra_functional_predicates=set(),  # Single-valued predicates (auto-invalidate old facts)
    known_entities={"Alice": "person", "ProjectX": "project"},  # Helps extraction avoid duplicates

    # Memory evolution (sleep-time linking of new memories to related old ones)
    evolve_enabled=True,
    evolve_link_threshold=0.70,         # Cosine floor for a link
    evolve_llm_refresh=False,           # LLM re-tags linked older memories

    # Wisdom
    action_families={},                 # Action type groupings for "similar enough" matching
)
```

See `maev/config.py` for the full list, including learned ranker training and graduation thresholds.

## Privacy: sensitivity-split routing

maev is built for setups where the memory corpus contains things that must never leave the machine — desktop context, enterprise records, family details. Everything lives in one local SQLite file, and two hooks make the hybrid local+cloud pattern work. The principle: **split by sensitivity, not by capability**.

**1. Route LLM work by task.** Your `LLMProvider.call()` receives a per-task model name (`extraction_model`, `inference_model`, `review_model` — all just strings you configure). Point extraction and inference — the tasks that see raw conversation text — at a local model, and let a frontier model handle only whatever your agent does with retrieved facts:

```python
class SplitProvider:
    def call(self, messages, model, max_tokens, source=""):
        if model.startswith("ollama/"):
            return call_ollama(messages, model.removeprefix("ollama/"), max_tokens)
        return call_cloud(messages, model, max_tokens)

config = MaevConfig(
    db_path=Path("memory.db"),
    extraction_model="ollama/qwen3:8b",   # raw text stays local
    inference_model="ollama/qwen3:8b",    # raw text stays local
    review_model="ollama/qwen3:8b",       # raw text stays local
)
```

With this config, no raw conversation content is ever sent to a cloud API by maev itself.

**2. Redact at the retrieval boundary.** `redact_output` is a callback applied to memory content the moment it leaves maev toward a prompt (`find_similar_memories`, `get_tiered_memory_context`, `search_fts`, `find_by_subject`). Stored data is never modified — only the outbound copy. Wire in any scrubber; [Microsoft Presidio](https://microsoft.github.io/presidio/) is the standard open-source choice:

```python
config = MaevConfig(
    db_path=Path("memory.db"),
    redact_output=lambda text: presidio_anonymize(text),  # or your own regex scrubber
)
```

If the hook raises, maev fails closed: the text is replaced with `[redacted]` rather than passed through.

Combined, a host app that builds cloud prompts from those four retrieval functions sends only redacted, already-extracted facts — never the corpus, never raw conversations.

**Scope — read this before relying on it.** `redact_output` covers exactly the four retrieval functions listed above: the surfaces designed for prompt assembly. It does NOT apply to:

- Direct CRUD/graph reads (`get_all_active`, `get_recent_memories`, `get_entity_profile`, `graph_query`, `get_relationship_history`) — these return raw stored content, on purpose: local lifecycle jobs (extraction, review, hygiene) need unredacted text to work.
- The bundled MCP and REST servers. They configure themselves from environment variables and a redaction hook is a Python callable, so they run without one — their tools (e.g. `maev_memory_facts`, `maev_graph_entity_profile`) and routes (e.g. `GET /memory/{id}`) return raw stored content. Do not point a cloud-model MCP client at `maev-mcp` and expect redaction; for sensitive corpora, embed maev as a library behind your own redaction-aware server.

## Servers

You don't have to embed maev as a library — it also ships two servers (both optional extras).

**MCP server** (`pip install "maev[mcp]"`) exposes the cognition layer to any MCP client — Claude Desktop, Claude Code, ChatGPT — as 20 tools across 4 domains: memory (6), graph (9), wisdom (4), and extraction (1).

```bash
maev-mcp                              # STDIO (local clients)
MAEV_TRANSPORT=http maev-mcp         # HTTP (remote; requires MAEV_AUTH_TOKEN)
```

**REST server** (`pip install "maev[server]"`) is a FastAPI app with routers for memory, graph, wisdom, extraction, and health, plus optional bearer-token auth.

```bash
maev-server                           # defaults to 127.0.0.1:18790
```

Both configure via `MAEV_`-prefixed environment variables (or a `.env` file): `MAEV_DB_PATH`, `MAEV_LLM_PROVIDER` (`anthropic` or `openai`), `MAEV_LLM_API_KEY`, `MAEV_EMBED_PROVIDER` (`ollama`, `voyage`, or `openai`), and friends. Since the servers own the process, they construct providers for you from those variables — this is where the `anthropic`, `openai`, and `voyage` extras come in.

## Evals

The repo (not the pip package) ships a deterministic eval harness — the answer to a field where most memory-system numbers are vendor-self-reported:

```bash
python -m evals.run_eval          # human-readable report
python -m evals.run_eval --json   # full metrics
```

It scores recall@1/@k, MRR, and tokens injected per query across three arms: the main retrieval pipeline, tiered context, and a full-context control (every memory concatenated — what you'd pay without a memory system). Questions are tagged by retrieval mechanism (keyword / paraphrase / graph 1-hop / graph 2-hop) so you can see which signal a change helped or hurt. Runs are byte-for-byte reproducible: pinned IDs, pinned timestamps, hash-seed-independent scoring. If you change retrieval code, run this before and after.

## Status

Active development. The core concepts (memory, graph, retrieval, wisdom, lifecycle) are inherited from maasv and stable; the retrieval pipeline and lifecycle have evolved substantially since the fork (see "Where this came from"). Every retrieval change is gated by the eval harness. The public API may still shift.

## License

Business Source License 1.1, inherited from the original maasv work — the LICENSE file is preserved unmodified, including the original licensor. Free for personal, internal, educational, and non-commercial use. Commercial use of the original work requires a license from the original licensor (admin@maasv.ai, per the LICENSE). Converts to Apache 2.0 on 2030-02-16. See [LICENSE](LICENSE) for details.

## Related

- **maasv** — Adam Bell's original memory engine this project descends from; preserved at 0.2.0 in this repo's history. (Original upstream repository no longer public.)
- **Doris** — The AI assistant maasv was built for, and the origin of the architecture. (Repository no longer public.)
