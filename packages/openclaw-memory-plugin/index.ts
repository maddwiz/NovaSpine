import { randomUUID } from "node:crypto";
import { appendFile, mkdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk/memory-core";

type NovaSpinePluginConfig = {
  baseUrl: string;
  apiToken?: string;
  autoRecall: boolean;
  autoCapture: boolean;
  sessionIngestOnReset: boolean;
  sessionSnapshotOnReset: boolean;
  guidance: boolean;
  recallTopK: number;
  recallMinScore: number;
  recallFormat: "xml" | "plain";
  captureMaxMessages: number;
  captureMinChars: number;
  captureCooldownMs: number;
  ticketsTtlMs: number;
  roles?: string[];
  timeoutMs: number;
};

type RecallMemory = {
  id?: string;
  content: string;
  role: string;
  session_id: string;
  score: number;
  metadata?: Record<string, unknown>;
};

type RecallResponse = {
  memories: RecallMemory[];
  count: number;
  query: string;
};

type AugmentResponse = {
  context: string;
  count: number;
  memories: RecallMemory[];
};

type HealthResponse = {
  status?: string;
  service?: string;
};

type StatusResponse = Record<string, unknown>;

type ExplainMemory = RecallMemory & {
  why_recalled?: {
    reasons?: string[];
    score?: number;
  };
  provenance?: {
    source_kind?: string;
    session_id?: string;
    source_file?: string;
    source_id?: string;
    created_at?: string;
  };
};

type ExplainResponse = {
  memories: ExplainMemory[];
  count: number;
  query: string;
};

type FactItem = {
  id: string;
  source_chunk_id: string;
  entity: string;
  relation: string;
  value: string;
  date?: string;
  confidence?: number;
  status?: string;
  metadata?: Record<string, unknown>;
  created_at?: string;
};

type CurrentFactsResponse = {
  facts: FactItem[];
  count: number;
};

type TruthGroup = {
  entity: string;
  relation: string;
  current_facts: FactItem[];
  historical_facts: FactItem[];
};

type TruthResponse = {
  fact_groups: TruthGroup[];
  count: number;
};

type FactConflict = {
  entity: string;
  relation: string;
  value_count: number;
  current_facts: FactItem[];
  historical_facts: FactItem[];
};

type FactConflictsResponse = {
  conflicts: FactConflict[];
  count: number;
};

type FactResolveResponse = {
  ok: boolean;
  winner_fact: FactItem;
  superseded_facts: FactItem[];
  resolved_at: string;
  resolution_id: string;
};

type SessionIngestResponse = {
  session_id: string;
  chunks_ingested: number;
  roles?: Record<string, number>;
};

type ResolutionTicket = {
  token: string;
  mode: "conflict" | "override";
  entity: string;
  relation: string;
  facts: FactItem[];
  created_at: string;
  expires_at: string;
  status: "pending" | "resolved";
  resolved_at?: string;
};

type ResolutionStore = {
  version: 1;
  tickets: ResolutionTicket[];
};

const LOG_COOLDOWN_MS = 60_000;
const DIRECT_RECALL_PATTERN =
  /(remember|phrase|recall|what did i|what do you remember|told you|previous conversation|previous chat|past chat)/i;
const STORE_MEMORY_PATTERN = /^(please\s+)?remember\b/i;
const MEMORY_EXPLAIN_PATTERN =
  /(why.*remember|why.*recalled|where.*memory come|where did that come from|provenance|how do you know|why do you know)/i;
const FACT_RESOLUTION_PATTERN =
  /(resolve.*conflict|which.*current|what.*current now|supersede|retire.*old|outdated.*fact|conflicting facts?)/i;
const configSchema = {
  type: "object",
  additionalProperties: false,
  properties: {
    baseUrl: { type: "string" },
    apiToken: { type: "string" },
    autoRecall: { type: "boolean" },
    autoCapture: { type: "boolean" },
    sessionIngestOnReset: { type: "boolean" },
    sessionSnapshotOnReset: { type: "boolean" },
    guidance: { type: "boolean" },
    recallTopK: { type: "number", minimum: 1, maximum: 20 },
    recallMinScore: { type: "number", minimum: 0, maximum: 1 },
    recallFormat: { type: "string", enum: ["xml", "plain"] },
    captureMaxMessages: { type: "number", minimum: 1, maximum: 10 },
    captureMinChars: { type: "number", minimum: 1, maximum: 5000 },
    captureCooldownMs: { type: "number", minimum: 1000, maximum: 86_400_000 },
    ticketsTtlMs: { type: "number", minimum: 60_000, maximum: 7 * 24 * 60 * 60_000 },
    roles: { type: "array", items: { type: "string" } },
    timeoutMs: { type: "number", minimum: 1000, maximum: 60000 },
  },
} as const;

const recallToolSchema = {
  type: "object",
  properties: {
    query: { type: "string", description: "What memory to search for" },
    limit: { type: "number", description: "Maximum results to return", minimum: 1, maximum: 20 },
    sessionFilter: { type: "string", description: "Optional session id filter" },
  },
  required: ["query"],
} as const;

const storeToolSchema = {
  type: "object",
  properties: {
    text: { type: "string", description: "Text to store in NovaSpine" },
    sourceId: { type: "string", description: "Optional source identifier" },
    metadata: { type: "object", additionalProperties: true },
  },
  required: ["text"],
} as const;

function asRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return value as Record<string, unknown>;
}

function readString(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

function readOptionalString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function readBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function readNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function readRoles(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const roles = value.filter((item): item is string => typeof item === "string" && item.trim().length > 0);
  return roles.length > 0 ? roles : undefined;
}

function normalizeConfig(pluginConfig: unknown): NovaSpinePluginConfig {
  const raw = asRecord(pluginConfig);
  return {
    baseUrl: readString(raw.baseUrl, "http://127.0.0.1:8420").replace(/\/+$/, ""),
    apiToken: readOptionalString(raw.apiToken),
    autoRecall: readBoolean(raw.autoRecall, true),
    autoCapture: readBoolean(raw.autoCapture, false),
    sessionIngestOnReset: readBoolean(raw.sessionIngestOnReset, readBoolean(raw.autoCapture, false)),
    sessionSnapshotOnReset: readBoolean(raw.sessionSnapshotOnReset, true),
    guidance: readBoolean(raw.guidance, true),
    recallTopK: Math.max(1, Math.min(20, Math.floor(readNumber(raw.recallTopK, 5)))),
    recallMinScore: Math.max(0, Math.min(1, readNumber(raw.recallMinScore, 0.005))),
    recallFormat: raw.recallFormat === "plain" ? "plain" : "xml",
    captureMaxMessages: Math.max(1, Math.min(10, Math.floor(readNumber(raw.captureMaxMessages, 6)))),
    captureMinChars: Math.max(1, Math.min(5000, Math.floor(readNumber(raw.captureMinChars, 24)))),
    captureCooldownMs: Math.max(1000, Math.min(86_400_000, Math.floor(readNumber(raw.captureCooldownMs, 300_000)))),
    ticketsTtlMs: Math.max(60_000, Math.min(7 * 24 * 60 * 60_000, Math.floor(readNumber(raw.ticketsTtlMs, 24 * 60 * 60_000)))),
    roles: readRoles(raw.roles),
    timeoutMs: Math.max(1000, Math.min(60000, Math.floor(readNumber(raw.timeoutMs, 12_000)))),
  };
}

function buildHeaders(cfg: NovaSpinePluginConfig): Record<string, string> {
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (cfg.apiToken) headers.authorization = `Bearer ${cfg.apiToken}`;
  return headers;
}

async function requestJson<T>(
  cfg: NovaSpinePluginConfig,
  route: string,
  init: { method?: string; body?: unknown } = {},
): Promise<T> {
  const response = await fetch(`${cfg.baseUrl}${route}`, {
    method: init.method ?? "GET",
    headers: buildHeaders(cfg),
    body: init.body === undefined ? undefined : JSON.stringify(init.body),
    signal: AbortSignal.timeout(cfg.timeoutMs),
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(detail ? `${response.status} ${response.statusText}: ${detail}` : `${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

function formatTool(details: unknown) {
  return {
    content: [{ type: "text" as const, text: JSON.stringify(details, null, 2) }],
    details,
  };
}

function extractTextBlocks(content: unknown): string[] {
  if (typeof content === "string") return [content];
  if (!Array.isArray(content)) return [];
  const blocks: string[] = [];
  for (const item of content) {
    if (!item || typeof item !== "object") continue;
    const record = item as Record<string, unknown>;
    if (record.type === "text" && typeof record.text === "string" && record.text.trim()) {
      blocks.push(record.text.trim());
    }
  }
  return blocks;
}

function summarizeText(raw: string, limit = 280): string {
  const cleaned = String(raw || "")
    .replace(/<relevant-memories>[\s\S]*?<\/relevant-memories>/gi, " ")
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (!cleaned) return "";
  return cleaned.length <= limit ? cleaned : `${cleaned.slice(0, Math.max(0, limit - 1))}…`;
}

function summarizeMessages(messages: unknown[], maxMessages: number, minChars: number): string[] {
  const selected: string[] = [];
  for (const message of messages) {
    if (!message || typeof message !== "object") continue;
    const record = message as Record<string, unknown>;
    const role = typeof record.role === "string" ? record.role : "";
    if (role !== "user" && role !== "assistant") continue;
    const text = summarizeText(extractTextBlocks(record.content).join("\n"));
    if (text.length >= minChars) selected.push(`${role}: ${text}`);
  }
  return selected.slice(-maxMessages);
}

function formatMemories(memories: RecallMemory[]): string {
  return memories
    .map((memory, index) => {
      const sessionId = memory.session_id || "unknown-session";
      return `${index + 1}. [${memory.role}] ${memory.content} (${(memory.score * 100).toFixed(0)}%, ${sessionId})`;
    })
    .join("\n");
}

function logWarnOnce(
  logger: OpenClawPluginApi["logger"],
  state: Map<string, number>,
  key: string,
  message: string,
) {
  const now = Date.now();
  const last = state.get(key) || 0;
  if (now - last < LOG_COOLDOWN_MS) return;
  state.set(key, now);
  logger.warn(message);
}

function workspaceMemoryDir(api: OpenClawPluginApi): string {
  return api.resolvePath("~/.openclaw/workspace/memory");
}

function memoryStateDir(api: OpenClawPluginApi): string {
  return api.resolvePath("~/.openclaw/plugin-state/novaspine-memory");
}

function resolutionStorePath(api: OpenClawPluginApi): string {
  return path.join(memoryStateDir(api), "resolution-tickets.json");
}

async function writeWorkspaceMemory(api: OpenClawPluginApi, text: string, metadata: Record<string, unknown> = {}) {
  const dir = workspaceMemoryDir(api);
  await mkdir(dir, { recursive: true });
  const now = new Date();
  const date = now.toISOString().slice(0, 10);
  const time = now.toISOString().slice(11, 19);
  const target = path.join(dir, `${date}.md`);
  const pairs = Object.entries(metadata)
    .filter(([, value]) => value !== undefined && value !== null && String(value).trim())
    .map(([key, value]) => `${key}=${String(value)}`);
  const entry = [
    "",
    `## ${time} UTC`,
    text.trim(),
    pairs.length ? `metadata: ${pairs.join(", ")}` : "",
    "",
  ]
    .filter(Boolean)
    .join("\n");
  await appendFile(target, entry, "utf-8");
}

async function computeFingerprint(filePath: string): Promise<string | undefined> {
  try {
    const info = await stat(filePath);
    return info.isFile() ? `${Math.floor(info.mtimeMs)}:${info.size}` : undefined;
  } catch {
    return undefined;
  }
}

async function loadResolutionStore(api: OpenClawPluginApi): Promise<ResolutionStore> {
  try {
    const raw = await readFile(resolutionStorePath(api), "utf-8");
    const parsed = JSON.parse(raw) as Partial<ResolutionStore>;
    const now = Date.now();
    return {
      version: 1,
      tickets: Array.isArray(parsed.tickets)
        ? parsed.tickets.filter((ticket) => {
            const expiresAt = Date.parse(String(ticket.expires_at || ""));
            return ticket.status !== "pending" || !Number.isFinite(expiresAt) || expiresAt > now;
          })
        : [],
    };
  } catch {
    return { version: 1, tickets: [] };
  }
}

async function saveResolutionStore(api: OpenClawPluginApi, store: ResolutionStore) {
  const target = resolutionStorePath(api);
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target, `${JSON.stringify(store, null, 2)}\n`, "utf-8");
}

function formatFact(fact: FactItem, index: number): string {
  const shortId = fact.id ? fact.id.slice(0, 8) : "unknown";
  const confidence = Number.isFinite(Number(fact.confidence)) ? Number(fact.confidence).toFixed(2) : "n/a";
  return `${index + 1}. [${shortId}] ${fact.value}${fact.date ? ` (date=${fact.date})` : ""} [${fact.status || "current"}, confidence=${confidence}]`;
}

function buildResolutionTicket(mode: "conflict" | "override", entity: string, relation: string, facts: FactItem[], ttlMs: number): ResolutionTicket {
  const now = new Date();
  return {
    token: randomUUID(),
    mode,
    entity,
    relation,
    facts,
    created_at: now.toISOString(),
    expires_at: new Date(now.getTime() + ttlMs).toISOString(),
    status: "pending",
  };
}

async function prepareResolutionTicket(
  api: OpenClawPluginApi,
  cfg: NovaSpinePluginConfig,
  params: { entity: string; relation?: string },
): Promise<{ ok: boolean; ticket?: ResolutionTicket; error?: string }> {
  if (!params.entity.trim()) return { ok: false, error: "entity is required" };
  const relation = params.relation?.trim() || "";
  const truth = await requestJson<TruthResponse>(
    cfg,
    `/api/v1/facts/truth?${new URLSearchParams({
      entity: params.entity.trim(),
      ...(relation ? { relation } : {}),
      limit: "20",
    }).toString()}`,
  );
  const groups = truth.fact_groups.filter((group) => group.entity.toLowerCase() === params.entity.trim().toLowerCase());
  const narrowed = relation
    ? groups.filter((group) => group.relation.toLowerCase() === relation.toLowerCase())
    : groups;
  if (narrowed.length === 0) {
    return { ok: false, error: "No matching facts found for that entity/relation." };
  }
  if (!relation && narrowed.length > 1) {
    return {
      ok: false,
      error: `Multiple relations found for ${params.entity.trim()}: ${narrowed.map((group) => group.relation).join(", ")}.`,
    };
  }
  const group = narrowed[0];
  const candidateFacts = [...group.current_facts, ...group.historical_facts];
  const mode = group.current_facts.length > 1 ? "conflict" : "override";
  const store = await loadResolutionStore(api);
  const ticket = buildResolutionTicket(mode, group.entity, group.relation, candidateFacts, cfg.ticketsTtlMs);
  store.tickets.push(ticket);
  await saveResolutionStore(api, store);
  return { ok: true, ticket };
}

async function applyResolutionTicket(
  api: OpenClawPluginApi,
  cfg: NovaSpinePluginConfig,
  params: { approvalToken: string; winner: string; reason?: string; userConfirmation: string },
): Promise<FactResolveResponse> {
  const store = await loadResolutionStore(api);
  const ticket = store.tickets.find((item) => item.token === params.approvalToken && item.status === "pending");
  if (!ticket) throw new Error("Unknown or expired resolution ticket.");
  const winnerInput = params.winner.trim();
  let winnerFact = ticket.facts.find((fact) => fact.id === winnerInput || fact.id.startsWith(winnerInput));
  if (!winnerFact) {
    const optionNumber = Number.parseInt(winnerInput, 10);
    if (Number.isFinite(optionNumber) && optionNumber >= 1 && optionNumber <= ticket.facts.length) {
      winnerFact = ticket.facts[optionNumber - 1];
    }
  }
  if (!winnerFact) throw new Error("Winner must be a fact id, id prefix, or 1-based option number.");
  const loserIds = ticket.facts.map((fact) => fact.id).filter((id) => id !== winnerFact?.id);
  const response = await requestJson<FactResolveResponse>(cfg, "/api/v1/facts/resolve", {
    method: "POST",
    body: {
      winner_fact_id: winnerFact.id,
      loser_fact_ids: loserIds,
      reason: params.reason || "",
      user_confirmation: params.userConfirmation,
      resolution_ticket_id: ticket.token,
    },
  });
  ticket.status = "resolved";
  ticket.resolved_at = new Date().toISOString();
  await saveResolutionStore(api, store);
  return response;
}

const NOVASPINE_GUIDANCE = [
  "NovaSpine memory is available.",
  "Use injected memories only when they materially improve the answer.",
  "For prior-session recall, remembered phrases, stable preferences, or 'what did I tell you to remember?' questions, call novaspine_recall.",
  "If the user asks you to remember a stable preference or fact, call novaspine_store.",
  "For provenance or 'why did you remember that?' questions, call novaspine_explain.",
  "For facts that may have changed over time, call novaspine_current_facts.",
  "For conflicting current facts, call novaspine_resolution_prepare, ask the user which option should stay current, then call novaspine_resolution_apply after explicit approval.",
  "Prefer novaspine_recall over generic memory_search for cross-session conversational memory.",
  "Do not claim memory certainty when no relevant NovaSpine memories were returned.",
].join("\n");

const novaspineMemoryPlugin = {
  id: "novaspine-memory",
  name: "NovaSpine Memory",
  description: "NovaSpine local-first memory plugin for OpenClaw",
  kind: "memory" as const,
  configSchema,

  register(api: OpenClawPluginApi) {
    const cfg = normalizeConfig(api.pluginConfig);
    const logCooldowns = new Map<string, number>();
    const resetCooldowns = new Map<string, { at: number; fingerprint?: string }>();

    api.registerService({
      id: "novaspine-memory",
      async start({ logger }) {
        try {
          const health = await requestJson<HealthResponse>(cfg, "/api/v1/health");
          logger.info(
            `novaspine-memory: connected to ${cfg.baseUrl} (${health.service ?? "unknown-service"}: ${health.status ?? "unknown"})`,
          );
        } catch (error) {
          logger.warn(`novaspine-memory: health check failed: ${String(error)}`);
        }
      },
    });

    api.registerTool(
      {
        name: "novaspine_status",
        label: "NovaSpine Status",
        description: "Check NovaSpine health and memory counts.",
        parameters: { type: "object", properties: {}, additionalProperties: false },
        async execute() {
          try {
            const [health, status] = await Promise.all([
              requestJson<HealthResponse>(cfg, "/api/v1/health"),
              requestJson<StatusResponse>(cfg, "/api/v1/status/full"),
            ]);
            return formatTool({
              ok: true,
              baseUrl: cfg.baseUrl,
              health,
              status,
              autoRecall: cfg.autoRecall,
              sessionIngestOnReset: cfg.sessionIngestOnReset,
              sessionSnapshotOnReset: cfg.sessionSnapshotOnReset,
            });
          } catch (error) {
            return formatTool({ ok: false, error: String(error) });
          }
        },
      },
      { name: "novaspine_status" },
    );

    api.registerTool(
      {
        name: "novaspine_recall",
        label: "NovaSpine Recall",
        description: "Search NovaSpine long-term memory for prior context and preferences.",
        parameters: recallToolSchema,
        async execute(_toolCallId, params) {
          try {
            const response = await requestJson<RecallResponse>(cfg, "/api/v1/memory/recall", {
              method: "POST",
              body: {
                query: String((params as { query: string }).query || "").trim(),
                top_k: Math.max(1, Math.min(20, Math.floor(Number((params as { limit?: number }).limit || cfg.recallTopK)))),
                session_filter: readOptionalString((params as { sessionFilter?: string }).sessionFilter),
              },
            });
            if (!response.memories.length) {
              return {
                content: [{ type: "text" as const, text: "No relevant NovaSpine memories found." }],
                details: { count: 0, query: response.query },
              };
            }
            return {
              content: [{ type: "text" as const, text: `Found ${response.memories.length} NovaSpine memories:\n\n${formatMemories(response.memories)}` }],
              details: response,
            };
          } catch (error) {
            return formatTool({ ok: false, error: String(error) });
          }
        },
      },
      { name: "novaspine_recall" },
    );

    api.registerTool(
      {
        name: "novaspine_store",
        label: "NovaSpine Store",
        description: "Store a high-value fact or decision in NovaSpine memory.",
        parameters: storeToolSchema,
        async execute(_toolCallId, params) {
          try {
            const input = params as { text: string; sourceId?: string; metadata?: Record<string, unknown> };
            const response = await requestJson<{ chunk_ids: string[]; count: number }>(cfg, "/api/v1/memory/ingest", {
              method: "POST",
              body: {
                text: input.text.trim(),
                source_id: input.sourceId || "openclaw:tool",
                metadata: {
                  source: "openclaw-tool",
                  stored_via: "novaspine_store",
                  ...(input.metadata || {}),
                },
              },
            });
            await writeWorkspaceMemory(api, `Remembered item: ${input.text.trim()}`, {
              source: "novaspine_store",
              ...(input.metadata || {}),
            });
            return {
              content: [{ type: "text" as const, text: `Stored ${response.count} chunk(s) in NovaSpine.` }],
              details: response,
            };
          } catch (error) {
            return formatTool({ ok: false, error: String(error) });
          }
        },
      },
      { name: "novaspine_store" },
    );

    api.registerTool(
      {
        name: "novaspine_explain",
        label: "NovaSpine Explain",
        description: "Explain why a memory was recalled and show provenance details.",
        parameters: recallToolSchema,
        async execute(_toolCallId, params) {
          try {
            const input = params as { query: string; limit?: number; sessionFilter?: string };
            const response = await requestJson<ExplainResponse>(cfg, "/api/v1/memory/explain", {
              method: "POST",
              body: {
                query: input.query.trim(),
                top_k: Math.max(1, Math.min(20, Math.floor(Number(input.limit || cfg.recallTopK)))),
                session_filter: readOptionalString(input.sessionFilter),
              },
            });
            return formatTool(response);
          } catch (error) {
            return formatTool({ ok: false, error: String(error) });
          }
        },
      },
      { name: "novaspine_explain" },
    );

    api.registerTool(
      {
        name: "novaspine_current_facts",
        label: "NovaSpine Current Facts",
        description: "Read the current fact view for an entity, especially when facts may have changed over time.",
        parameters: {
          type: "object",
          properties: {
            entity: { type: "string" },
            relation: { type: "string" },
            limit: { type: "number", minimum: 1, maximum: 100 },
          },
          required: ["entity"],
        },
        async execute(_toolCallId, params) {
          try {
            const input = params as { entity: string; relation?: string; limit?: number };
            const query = new URLSearchParams({
              entity: input.entity.trim(),
              ...(input.relation?.trim() ? { relation: input.relation.trim() } : {}),
              limit: String(Math.max(1, Math.min(100, Math.floor(Number(input.limit || 10))))),
            });
            const response = await requestJson<CurrentFactsResponse>(cfg, `/api/v1/facts/current?${query.toString()}`);
            return formatTool(response);
          } catch (error) {
            return formatTool({ ok: false, error: String(error) });
          }
        },
      },
      { name: "novaspine_current_facts" },
    );

    api.registerTool(
      {
        name: "novaspine_fact_conflicts",
        label: "NovaSpine Fact Conflicts",
        description: "List current fact conflicts where multiple active values still exist.",
        parameters: {
          type: "object",
          properties: {
            limit: { type: "number", minimum: 1, maximum: 100 },
          },
        },
        async execute(_toolCallId, params) {
          try {
            const limit = Math.max(1, Math.min(100, Math.floor(Number((params as { limit?: number }).limit || 10))));
            const response = await requestJson<FactConflictsResponse>(cfg, `/api/v1/facts/conflicts?limit=${limit}`);
            return formatTool(response);
          } catch (error) {
            return formatTool({ ok: false, error: String(error) });
          }
        },
      },
      { name: "novaspine_fact_conflicts" },
    );

    api.registerTool(
      {
        name: "novaspine_resolution_prepare",
        label: "NovaSpine Resolution Prepare",
        description: "Prepare a human-approved fact resolution ticket before asking the user which fact should stay current.",
        parameters: {
          type: "object",
          properties: {
            entity: { type: "string" },
            relation: { type: "string" },
          },
          required: ["entity"],
        },
        async execute(_toolCallId, params) {
          try {
            const input = params as { entity: string; relation?: string };
            const prepared = await prepareResolutionTicket(api, cfg, {
              entity: input.entity,
              relation: input.relation,
            });
            return formatTool(prepared.ok ? prepared : { ok: false, error: prepared.error });
          } catch (error) {
            return formatTool({ ok: false, error: String(error) });
          }
        },
      },
      { name: "novaspine_resolution_prepare" },
    );

    api.registerTool(
      {
        name: "novaspine_resolution_apply",
        label: "NovaSpine Resolution Apply",
        description: "Apply a prepared fact-resolution ticket after the user explicitly approved a winner.",
        parameters: {
          type: "object",
          properties: {
            approvalToken: { type: "string" },
            winner: { type: "string" },
            reason: { type: "string" },
            userConfirmation: { type: "string" },
          },
          required: ["approvalToken", "winner", "userConfirmation"],
        },
        async execute(_toolCallId, params) {
          try {
            const input = params as {
              approvalToken: string;
              winner: string;
              reason?: string;
              userConfirmation: string;
            };
            const response = await applyResolutionTicket(api, cfg, input);
            return formatTool(response);
          } catch (error) {
            return formatTool({ ok: false, error: String(error) });
          }
        },
      },
      { name: "novaspine_resolution_apply" },
    );

    api.registerCli(
      ({ program }) => {
        const memory = program.command("novaspine").description("NovaSpine memory plugin commands");
        memory.command("health").description("Check the NovaSpine API health").action(async () => {
          const health = await requestJson<HealthResponse>(cfg, "/api/v1/health");
          console.log(JSON.stringify(health, null, 2));
        });
        memory.command("status").description("Show NovaSpine full status").action(async () => {
          const status = await requestJson<StatusResponse>(cfg, "/api/v1/status/full");
          console.log(JSON.stringify(status, null, 2));
        });
        memory.command("recall").description("Recall memories from NovaSpine").argument("<query>").action(async (query) => {
          const response = await requestJson<RecallResponse>(cfg, "/api/v1/memory/recall", {
            method: "POST",
            body: { query, top_k: cfg.recallTopK },
          });
          console.log(JSON.stringify(response, null, 2));
        });
        memory.command("explain").description("Explain recalled memories").argument("<query>").action(async (query) => {
          const response = await requestJson<ExplainResponse>(cfg, "/api/v1/memory/explain", {
            method: "POST",
            body: { query, top_k: cfg.recallTopK },
          });
          console.log(JSON.stringify(response, null, 2));
        });
        memory.command("current").description("Show current facts for an entity").argument("<entity>").argument("[relation]").action(async (entity, relation) => {
          const query = new URLSearchParams({ entity, ...(relation ? { relation } : {}), limit: "20" });
          const response = await requestJson<CurrentFactsResponse>(cfg, `/api/v1/facts/current?${query.toString()}`);
          console.log(JSON.stringify(response, null, 2));
        });
        memory.command("conflicts").description("Show current fact conflicts").action(async () => {
          const response = await requestJson<FactConflictsResponse>(cfg, "/api/v1/facts/conflicts?limit=20");
          console.log(JSON.stringify(response, null, 2));
        });
      },
      { commands: ["novaspine"] },
    );

    api.on("before_prompt_build", async (event) => {
      const result: { prependContext?: string; prependSystemContext?: string } = {};
      if (cfg.guidance) result.prependSystemContext = NOVASPINE_GUIDANCE;
      const prompt = readString(event.prompt, "");
      if (!prompt || prompt.length < 8) return Object.keys(result).length ? result : undefined;

      if (DIRECT_RECALL_PATTERN.test(prompt)) {
        if (STORE_MEMORY_PATTERN.test(prompt) && !/\?/.test(prompt)) {
          result.prependSystemContext = [
            NOVASPINE_GUIDANCE,
            "The user is asking you to remember a stable fact or preference.",
            "Call novaspine_store with a compact memory worth keeping across sessions.",
          ].join("\n");
          return result;
        }
        result.prependSystemContext = [
          NOVASPINE_GUIDANCE,
          "The user is asking about prior-session memory or a remembered item.",
          "Do not rely only on injected snippets.",
          "Call novaspine_recall and prefer the most recent matching memory when multiple matches exist.",
        ].join("\n");
        return result;
      }
      if (MEMORY_EXPLAIN_PATTERN.test(prompt)) {
        result.prependSystemContext = [
          NOVASPINE_GUIDANCE,
          "The user is asking why a memory was recalled or where it came from.",
          "Call novaspine_explain so your answer can cite provenance instead of guessing.",
        ].join("\n");
        return result;
      }
      if (FACT_RESOLUTION_PATTERN.test(prompt)) {
        result.prependSystemContext = [
          NOVASPINE_GUIDANCE,
          "The current prompt may require resolving conflicting or changed facts.",
          "First call novaspine_resolution_prepare, then ask the user which option should remain current.",
          "Do not call novaspine_resolution_apply until the user explicitly approves the winner.",
        ].join("\n");
        return result;
      }
      if (!cfg.autoRecall) return Object.keys(result).length ? result : undefined;

      try {
        const response = await requestJson<AugmentResponse>(cfg, "/api/v1/memory/augment", {
          method: "POST",
          body: {
            query: prompt,
            top_k: cfg.recallTopK,
            min_score: cfg.recallMinScore,
            format: cfg.recallFormat,
            roles: cfg.roles,
          },
        });
        if (response.count > 0 && response.context.trim()) {
          result.prependContext = `[NovaSpine Recall]\n${response.context}\n`;
        }
      } catch (error) {
        logWarnOnce(api.logger, logCooldowns, "novaspine-memory.inject", `novaspine-memory: recall failed: ${String(error)}`);
      }

      return Object.keys(result).length ? result : undefined;
    });

    api.on("before_tool_call", async (event) => {
      if (event.toolName !== "memory_search") return;
      const query = readString(asRecord(event.params).query, "");
      if (!query) return;
      if (DIRECT_RECALL_PATTERN.test(query)) {
        return {
          block: true,
          blockReason: "Use novaspine_recall for prior-session memory and remembered phrases.",
        };
      }
      if (MEMORY_EXPLAIN_PATTERN.test(query)) {
        return {
          block: true,
          blockReason: "Use novaspine_explain for provenance and memory explanations.",
        };
      }
      return;
    });

    api.on("before_reset", async (event) => {
      const sessionFile = readOptionalString(event.sessionFile);
      const fingerprint = sessionFile ? await computeFingerprint(sessionFile) : undefined;
      const cooldownKey = sessionFile || "session-reset";
      const last = resetCooldowns.get(cooldownKey);
      if (last && Date.now() - last.at < cfg.captureCooldownMs && last.fingerprint === fingerprint) {
        return;
      }
      resetCooldowns.set(cooldownKey, { at: Date.now(), fingerprint });

      if (cfg.sessionIngestOnReset && sessionFile) {
        try {
          await requestJson<SessionIngestResponse>(cfg, "/api/v1/sessions/ingest", {
            method: "POST",
            body: { path: sessionFile },
          });
        } catch (error) {
          logWarnOnce(api.logger, logCooldowns, "novaspine-memory.reset-ingest", `novaspine-memory: reset ingest failed: ${String(error)}`);
        }
      }

      if (!cfg.sessionSnapshotOnReset || !event.messages?.length) return;
      try {
        const snapshot = summarizeMessages(event.messages, cfg.captureMaxMessages, cfg.captureMinChars);
        if (!snapshot.length) return;
        await writeWorkspaceMemory(
          api,
          `Session snapshot before reset:\n${snapshot.join("\n")}`,
          {
            source: "before_reset",
            reason: event.reason || "reset",
            session_file: sessionFile || "",
          },
        );
      } catch (error) {
        logWarnOnce(api.logger, logCooldowns, "novaspine-memory.snapshot", `novaspine-memory: reset snapshot failed: ${String(error)}`);
      }
    });
  },
};

export default novaspineMemoryPlugin;
