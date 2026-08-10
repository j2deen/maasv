# openclaw-maev

OpenClaw memory plugin powered by [MAEV](https://github.com/j2deen/maev) — Memory Architecture for Evolving Agents.

Gives OpenClaw agents structured long-term memory backed by SQLite: 3-signal retrieval (with multi-hop Personalized PageRank), a knowledge graph with bi-temporal versioning, and experiential learning. All state lives locally in SQLite. LLM and embedding calls go to your configured provider (cloud by default, local supported).

> **Where this came from:** this plugin is the MAEV continuation of Adam Bell's
> [openclaw-maasv](https://github.com/ascottbell/openclaw-maasv), vendored into the
> MAEV repository so plugin and engine ship as a single artifact. The plugin's own
> BSL 1.1 LICENSE (licensor Adam Bell) is preserved unmodified in this directory.

## Prerequisites

A running maev server instance:
```bash
pip install "maev[server,anthropic,voyage]"
maev-server
```
See the [MAEV README](../README.md) for full setup details.

## Setup

1. Install the plugin from npm (or from this repository's `openclaw/` directory):
```bash
openclaw plugins install @maev-ai/evolving-agents
```

2. Activate the memory slot:
```json5
// ~/.openclaw/openclaw.json
{
  plugins: {
    slots: { memory: "memory-maev" },
    entries: {
      "memory-maev": {
        enabled: true,
        // OpenClaw 2026.7.x: non-bundled plugins must opt in to typed hooks.
        // Without these, auto-recall and auto-capture are silently blocked.
        hooks: {
          allowConversationAccess: true, // agent_end may read the conversation (auto-capture)
          allowPromptInjection: true,    // before_agent_start may prepend memory context (auto-recall)
          timeoutMs: 180000              // entity extraction can exceed the default hook timeout
        },
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

3. (OpenClaw 2026.7.x) Expose the optional tools. `memory_graph` and
`memory_wisdom` are registered as *optional* and stay hidden from agents until
allowlisted:
```json5
// ~/.openclaw/openclaw.json
{
  tools: {
    alsoAllow: ["memory_graph", "memory_wisdom"]
  }
}
```

> **Version note:** use plugin `>= 0.1.1` on OpenClaw 2026.7.x — `0.1.0`
> shipped TypeScript-only (rejected by the 2026.7.x npm plugin loader), lacked
> the `contracts.tools` declaration, and predates the 2026.7.x typed-hook
> event shapes, so its auto-recall/auto-capture never fired there.

## Tools

### Core (always available)
- **`memory_search`** — Retrieval using semantic similarity, keyword matching, and graph connectivity
- **`memory_store`** — Store memories with automatic deduplication
- **`memory_forget`** — Delete a memory by ID

### Knowledge Graph (enableGraph: true)
- **`memory_graph`** — Search entities, view entity profiles with relationships, create relationships

### Wisdom (enableWisdom: true)
- **`memory_wisdom`** — Log reasoning, record outcomes, attach feedback, search past wisdom

## Auto-Recall & Auto-Capture

When enabled, the plugin automatically:
- **Recalls** relevant memories before each agent turn (configurable via `maxRecallResults` and `maxRecallTokens`)
- **Captures** entities and facts from conversations after each session

Both can be toggled independently in the config.

## CLI

```bash
openclaw maev health           # Check connection
openclaw maev stats            # Detailed statistics
openclaw maev search "query"   # Search memories
```

## Architecture

```
[openclaw-maev]          <- This plugin (TypeScript, npm)
     |  HTTP calls
     v
[maev-server]            <- Python HTTP service (FastAPI)
     |  Python import
     v
[maev]                   <- Cognition library (pip)
     |
     v
[SQLite + sqlite-vec]     <- All state lives here
```

The plugin sends raw text. maev-server owns embeddings.
