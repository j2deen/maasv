# maev Memory

Structured long-term memory for OpenClaw agents, powered by [maev](https://github.com/j2deen/maev).

Replaces the default memory backend with a cognition layer that includes 3-signal retrieval (semantic + keyword + knowledge graph), entity extraction, temporal versioning, and experiential learning.

**maev is entirely self-hosted.** There is no maev cloud service. You run the server on your own machine, and all data is stored in a SQLite file on your local disk that you own and control. Nothing is sent to maev.

## Install

This skill requires the `@maev-ai/evolving-agents` plugin and a running maev server.

### 1. Start the server

From a checkout of [j2deen/maev](https://github.com/j2deen/maev):

```bash
pip install -e ".[server,anthropic,voyage]"
# configure via MAEV_* environment variables or a .env file (see below)
maev-server
```

### 2. Install the plugin

Install from npm (or from this repository's `openclaw/` directory):

```bash
openclaw plugins install @maev-ai/evolving-agents
```

### 3. Activate

```json5
// ~/.openclaw/openclaw.json
{
  plugins: {
    slots: { memory: "memory-maev" },
    entries: {
      "memory-maev": {
        enabled: true,
        config: {
          serverUrl: "http://127.0.0.1:18790",
          autoRecall: true,
          autoCapture: true,
          enableGraph: true
        }
      }
    }
  }
}
```

## Required Credentials

The maev server needs an LLM provider (for entity extraction) and an embedding provider (for semantic search). Configure these in your `.env` file:

| Variable | Required | Purpose |
|----------|----------|---------|
| `MAEV_LLM_PROVIDER` | Yes | `anthropic` or `openai` |
| `MAEV_LLM_API_KEY` | For cloud LLMs | LLM calls for entity extraction |
| `MAEV_EMBED_PROVIDER` | Yes | `ollama` (default), `voyage`, or `openai` |
| `MAEV_EMBED_API_KEY` | For cloud embeddings | Embedding generation (Voyage/OpenAI) |
| `MAEV_API_KEY` | Optional | Protects maev server endpoints with auth (`X-Maev-Key`) |

**For fully local operation** (no cloud calls), use `ollama` as your embed provider and a local LLM. maev is optimized for [Qwen3-Embedding-8B](https://ollama.com/library/qwen3-embedding) via Ollama, with built-in Matryoshka dimension truncation. See the [maev README](https://github.com/j2deen/maev) for local setup.

## Data & Network Behavior

- **maev has no cloud service.** The server runs on your machine, the database is a SQLite file on your disk. You own all of it.
- **The only external calls are to your own LLM/embedding provider** (Anthropic, OpenAI, Voyage) — using your own API keys, from your own machine. If you use `ollama`, zero data leaves your machine.
- **The plugin talks only to localhost** (`127.0.0.1:18790`). It makes no external network calls.
- **autoCapture** sends conversation summaries to your local maev server for entity extraction. Extracted entities are stored in your local SQLite database.
- **autoRecall** reads from your local SQLite database and injects relevant memories into the agent's context.
- **No telemetry, no analytics, no phone-home.** maev does not collect or transmit any data.

## What You Get

- **`memory_search`** — 3-signal retrieval across your memory store
- **`memory_store`** — Dedup-aware memory storage
- **`memory_forget`** — Permanent deletion
- **`memory_graph`** — Knowledge graph: entity search, profiles, relationships
- **`memory_wisdom`** — Log reasoning, record outcomes, search past decisions

## Links

- **Plugin (npm):** [@maev-ai/evolving-agents](https://www.npmjs.com/package/@maev-ai/evolving-agents)
- **Server + core:** [github.com/j2deen/maev](https://github.com/j2deen/maev) (not yet on PyPI)
