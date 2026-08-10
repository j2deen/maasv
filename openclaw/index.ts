/**
 * openclaw-maev: Memory plugin powered by the maev cognition layer.
 *
 * Replaces OpenClaw's flat-file memory with:
 * - 3-signal retrieval (vector + BM25 + knowledge graph)
 * - Entity extraction and knowledge graph with temporal versioning
 * - Background lifecycle management (dedup, decay, consolidate, reorganize)
 * - Experiential learning (log reasoning → record outcomes → feedback loop)
 * - Fully local, zero cloud dependency
 *
 * Architecture:
 *   [This plugin] → HTTP → [maev-server] → Python import → [maev] → [SQLite]
 *
 * maev-server owns embeddings. This plugin sends raw text.
 */

import { MaevClient } from "./client.js";
import { createMemorySearch } from "./tools/memory-search.js";
import { createMemoryStore } from "./tools/memory-store.js";
import { createMemoryForget } from "./tools/memory-forget.js";
import { createMemoryGraph } from "./tools/memory-graph.js";
import { createMemoryWisdom } from "./tools/memory-wisdom.js";
import type { PluginConfig } from "./types.js";

const DEFAULT_CONFIG: PluginConfig = {
  serverUrl: "http://127.0.0.1:18790",
  autoRecall: true,
  autoCapture: true,
  maxRecallResults: 5,
  maxRecallTokens: 2000,
  enableGraph: true,
  enableWisdom: false,
};

export default {
  id: "memory-maev",
  name: "Memory (maev)",
  description:
    "Cognition layer — entities, knowledge graph, lifecycle management, wisdom",
  kind: "memory" as const,

  register(api: any) {
    const rawConfig = api.pluginConfig ?? {};
    const config: PluginConfig = { ...DEFAULT_CONFIG, ...rawConfig };
    const client = new MaevClient(config);
    const logger = api.logger;

    // --- Background Service ---

    api.registerService({
      id: "memory-maev",
      async start() {
        try {
          const health = await client.health();
          logger.info(`maev connected: ${health.status}`);
        } catch (err) {
          logger.error(
            `Failed to connect to maev-server at ${config.serverUrl}: ${(err as Error).message}`,
          );
        }
      },
      stop() {
        logger.info("maev memory service stopped");
      },
    });

    // --- Core Memory Tools (always registered) ---

    api.registerTool(createMemorySearch(client));
    api.registerTool(createMemoryStore(client));
    api.registerTool(createMemoryForget(client));

    // --- Optional: Knowledge Graph ---

    if (config.enableGraph) {
      api.registerTool(createMemoryGraph(client), { optional: true });
    }

    // --- Optional: Wisdom ---

    if (config.enableWisdom) {
      api.registerTool(createMemoryWisdom(client), { optional: true });
    }

    // --- Auto-Recall Hook (before_agent_start) ---

    api.on("before_agent_start", async (event: any) => {
      if (!config.autoRecall) return;

      // Extract the user's latest message
      const userMessage = extractUserMessage(event);
      if (!userMessage) return;

      try {
        // Use maev's tiered context — returns pre-prioritized,
        // identity > family > preference > project > relevant content
        const { context } = await client.getContext({
          // Server caps query at 2000 chars — longer prompts 422 otherwise.
          query: userMessage.slice(0, 2000),
          core_limit: config.maxRecallResults,
          relevant_limit: Math.ceil(config.maxRecallResults / 2),
          use_semantic: true,
        });

        if (context && context.length > 0) {
          // Enforce maxRecallTokens (~4 chars per token)
          const maxChars = config.maxRecallTokens * 4;
          const truncated =
            context.length > maxChars
              ? context.slice(0, maxChars) + "\n[truncated]"
              : context;
          return {
            prependContext: `<long_term_memory>\n${truncated}\n</long_term_memory>`,
          };
        }
      } catch (err) {
        logger.warn(`Auto-recall failed: ${(err as Error).message}`);
      }
    });

    // --- Auto-Capture Hook (agent_end) ---

    api.on("agent_end", async (event: any) => {
      if (!config.autoCapture) return;

      try {
        const conversationText = extractConversation(event);
        if (!conversationText || conversationText.length < 50) return;

        // Send to maev extraction pipeline
        // Handles entity extraction, relationship building, memory storage,
        // dedup, confidence scoring, and graph updates internally
        await client.extract(conversationText);
      } catch (err) {
        logger.warn(`Auto-capture failed: ${(err as Error).message}`);
      }
    });

    // --- Gateway RPC Methods ---

    api.registerGatewayMethod(
      "maev.status",
      async ({ respond }: { respond: (ok: boolean, data: unknown) => void }) => {
        try {
          const health = await client.health();
          respond(true, health);
        } catch (err) {
          respond(false, { error: (err as Error).message });
        }
      },
    );

    api.registerGatewayMethod(
      "maev.stats",
      async ({ respond }: { respond: (ok: boolean, data: unknown) => void }) => {
        try {
          const s = await client.stats();
          respond(true, s);
        } catch (err) {
          respond(false, { error: (err as Error).message });
        }
      },
    );

    // --- CLI Commands ---

    api.registerCli(
      ({ program }: { program: any }) => {
        const maev = program
          .command("maev")
          .description("maev memory management");

        maev
          .command("health")
          .description("Check maev-server connection")
          .action(async () => {
            try {
              const health = await client.health();
              console.log(`Status: ${health.status}`);
            } catch (err) {
              console.error(`Connection failed: ${(err as Error).message}`);
            }
          });

        maev
          .command("stats")
          .description("Show detailed statistics")
          .action(async () => {
            try {
              const s = await client.stats();
              console.log(JSON.stringify(s, null, 2));
            } catch (err) {
              console.error(`Failed: ${(err as Error).message}`);
            }
          });

        maev
          .command("search")
          .argument("<query>", "Search query")
          .option("-n, --limit <number>", "Max results", "5")
          .action(async (query: string, opts: { limit: string }) => {
            try {
              const { results } = await client.searchMemories({
                query,
                limit: parseInt(opts.limit, 10),
              });
              if (results.length === 0) {
                console.log("No memories found.");
                return;
              }
              for (const m of results) {
                const subject = m.subject ? `[${m.subject}] ` : "";
                console.log(`${m.id}: ${subject}${m.content}`);
              }
            } catch (err) {
              console.error(`Search failed: ${(err as Error).message}`);
            }
          });
      },
      { commands: ["maev"] },
    );
  },
};

// --- Helpers ---

function messageContentToText(content: any): string {
  if (typeof content === "string") return content;
  // OpenClaw 2026.7.x uses content-block arrays: [{type:"text", text:"..."}]
  if (Array.isArray(content)) {
    const text = content
      .filter((block: any) => block?.type === "text" && typeof block.text === "string")
      .map((block: any) => block.text)
      .join("\n");
    if (text) return text;
  }
  return JSON.stringify(content);
}

function resolveEventMessages(event: any): any[] | null {
  // OpenClaw 2026.7.x: agent_end passes { messages }; older builds nested
  // them under event.context.messages.
  if (Array.isArray(event?.messages)) return event.messages;
  if (Array.isArray(event?.context?.messages)) return event.context.messages;
  return null;
}

function extractUserMessage(event: any): string | null {
  // OpenClaw 2026.7.x: before_agent_start passes { prompt }
  if (typeof event?.prompt === "string" && event.prompt.trim()) return event.prompt;
  if (typeof event?.userMessage === "string") return event.userMessage;
  const messages = resolveEventMessages(event);
  if (messages) {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        return messageContentToText(messages[i].content);
      }
    }
  }
  return null;
}

// Only the newest exchange is worth extracting each turn — earlier messages
// were captured after their own turns. Re-sending the whole history made
// extraction take minutes on local models (starving the shared Ollama the
// chat model runs on) and the oversized output usually failed to parse.
const CAPTURE_MAX_CHARS = 4000;

function extractConversation(event: any): string | null {
  const messages = resolveEventMessages(event);
  if (!messages) return null;

  // Walk back to the last user message; capture it plus everything after
  // (the assistant's reply, including multi-part replies).
  let start = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") {
      start = i;
      break;
    }
  }
  if (start === -1) return null;

  const parts: string[] = [];
  for (let i = start; i < messages.length; i++) {
    const msg = messages[i];
    if (msg.role === "user" || msg.role === "assistant") {
      parts.push(`${msg.role}: ${messageContentToText(msg.content)}`);
    }
  }
  if (parts.length === 0) return null;

  const text = parts.join("\n\n");
  return text.length > CAPTURE_MAX_CHARS ? text.slice(0, CAPTURE_MAX_CHARS) : text;
}
