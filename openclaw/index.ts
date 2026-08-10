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
          query: userMessage,
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

function extractUserMessage(event: any): string | null {
  // OpenClaw passes the user's message in various event shapes
  if (typeof event?.userMessage === "string") return event.userMessage;
  if (event?.context?.messages) {
    const messages = event.context.messages;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        return typeof messages[i].content === "string"
          ? messages[i].content
          : JSON.stringify(messages[i].content);
      }
    }
  }
  return null;
}

function extractConversation(event: any): string | null {
  if (!event?.context?.messages) return null;

  const messages = event.context.messages;
  const parts: string[] = [];

  for (const msg of messages) {
    if (msg.role === "user" || msg.role === "assistant") {
      const content =
        typeof msg.content === "string"
          ? msg.content
          : JSON.stringify(msg.content);
      parts.push(`${msg.role}: ${content}`);
    }
  }

  return parts.length > 0 ? parts.join("\n\n") : null;
}
