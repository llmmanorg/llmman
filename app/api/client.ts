// The only code in the app that talks HTTP: llmman's routes, its error
// shapes, reasoning fields and truncated-stream detection. Screens get an
// `ApiClient` for a server and never call fetch themselves.
//
// Streaming needs `fetch` from `expo/fetch`: React Native's built-in fetch
// cannot read a response body as it arrives.

import { fetch as expoFetch } from "expo/fetch";

/** A hosted model is addressed as `llmman.provider/<provider>/<model>`. */
export const REMOTE_PREFIX = "llmman.provider/";

export function remoteRef(providerId: string, modelId: string): string {
  return `${REMOTE_PREFIX}${providerId}/${modelId}`;
}

/** `{provider, model}` for a provider ref, or null for a local one. */
export function splitRemoteRef(ref: string | null | undefined): { provider: string; model: string } | null {
  if (!ref || !ref.startsWith(REMOTE_PREFIX)) return null;
  const rest = ref.slice(REMOTE_PREFIX.length);
  const slash = rest.indexOf("/");
  if (slash < 0) return null;
  return { provider: rest.slice(0, slash), model: rest.slice(slash + 1) };
}

/**
 * What went wrong, for a message the user can act on:
 * - `http`: the daemon answered with a non-2xx status (`status` is set).
 * - `network`: no answer at all (wrong address, server down, no LAN).
 * - `timeout`: no answer within the request's time limit.
 * - `protocol`: an answer that is not what llmman sends (bad JSON, a
 *   stream that ends early, an `{error}` inside a 200 stream).
 */
export type ApiErrorKind = "http" | "network" | "timeout" | "protocol";

export class ApiError extends Error {
  readonly status: number;
  readonly kind: ApiErrorKind;
  /** The 401 came from llmman's own key check (`WWW-Authenticate: Bearer realm="llmman"`). */
  readonly llmmanAuth: boolean;

  constructor(
    message: string,
    status: number,
    kind: ApiErrorKind = status ? "http" : "network",
    llmmanAuth = false,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.kind = kind;
    this.llmmanAuth = llmmanAuth;
  }

  /**
   * A missing or wrong llmman API key. The daemon relays a backend's or
   * provider's status verbatim, so a 401 without llmman's realm is a
   * hosted provider refusing its own key, not this one.
   */
  get unauthorized(): boolean {
    return this.status === 401 && this.llmmanAuth;
  }
}

/** Whether `e` is a cancellation the caller asked for (Stop), not a failure. */
export function isAbort(e: unknown): boolean {
  return e instanceof Error && e.name === "AbortError";
}

// ---- Transport -----------------------------------------------------------

/** The part of a fetch Response the client uses; satisfied by expo/fetch and WHATWG. */
export interface ResponseLike {
  ok: boolean;
  status: number;
  statusText: string;
  headers: { get(name: string): string | null };
  body: ReadableStream<Uint8Array> | null;
  text(): Promise<string>;
}

export interface RequestInitLike {
  method?: string;
  /** `manual` returns a redirect as is (expo/fetch supports it natively), e.g. to load its target without the key. */
  redirect?: "follow" | "manual" | "error";
  headers?: Record<string, string>;
  body?: string;
  signal?: AbortSignal;
}

export type FetchLike = (url: string, init?: RequestInitLike) => Promise<ResponseLike>;

export interface ServerConnection {
  /** Base URL, `http(s)://host:port` with an optional path prefix, no trailing slash. */
  url: string;
  /** API key, sent as a bearer when set. */
  key?: string;
}

/**
 * Where a client's key comes from when it can change under it: the web
 * app's own daemon, whose key the page asks for on llmman's 401.
 */
export interface KeySource {
  /** The key to send now; empty or undefined for none. */
  get(): string | undefined;
  /** Asks for a key after llmman refused (or wanted) one; resolves true when one was entered. */
  ask(): Promise<boolean>;
}

/**
 * Whether `r` is llmman refusing this client's key. A provider's or
 * backend's 401 is relayed without llmman's realm: an error to show, not
 * a key to ask for.
 */
export function wantsKey(r: ResponseLike): boolean {
  return r.status === 401 && /realm="llmman"/i.test(r.headers.get("www-authenticate") ?? "");
}

export interface RequestOptions {
  signal?: AbortSignal;
  /** Milliseconds before the request is abandoned with a `timeout` error. */
  timeoutMs?: number;
}

/** The error a non-2xx response carries: `{error}`, `{error:{message}}`, `{message}` or text. */
export async function errorFrom(response: ResponseLike): Promise<ApiError> {
  const text = await response.text().catch(() => "");
  let message: unknown = text || `${response.status} ${response.statusText}`.trim();
  try {
    const body = JSON.parse(text);
    message = body?.error?.message || body?.error || body?.message || message;
    if (typeof message !== "string") message = JSON.stringify(message);
  } catch {
    // not JSON: keep the text
  }
  const realm = /realm="llmman"/i.test(response.headers.get("www-authenticate") ?? "");
  return new ApiError(String(message).trim(), response.status, "http", response.status === 401 && realm);
}

/**
 * A streaming body's non-empty lines, split on `\n` (a trailing `\r`
 * dropped). `signal` aborting stops the read with an AbortError; a body
 * that fails mid-way (connection dropped) becomes a `network` ApiError.
 */
export async function* lines(
  body: ReadableStream<Uint8Array> | null,
  signal?: AbortSignal,
): AsyncGenerator<string> {
  if (!body) return;
  if (signal?.aborted) throw abortError();
  const reader = body.getReader();
  const onAbort = () => void reader.cancel().catch(() => {});
  signal?.addEventListener("abort", onAbort, { once: true });
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      let chunk: ReadableStreamReadResult<Uint8Array>;
      try {
        chunk = await reader.read();
      } catch (e) {
        if (signal?.aborted || isAbort(e)) throw abortError();
        const detail = e instanceof Error && e.message ? `: ${e.message}` : "";
        throw new ApiError(`the connection dropped${detail}`, 0, "network");
      }
      if (signal?.aborted) throw abortError();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      let nl;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl).replace(/\r$/, "");
        buffer = buffer.slice(nl + 1);
        if (line) yield line;
        if (signal?.aborted) throw abortError();
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) yield buffer;
  } finally {
    signal?.removeEventListener("abort", onAbort);
    // Also releases the reader when the consumer stopped early.
    await reader.cancel().catch(() => {});
  }
}

// ---- Daemon types --------------------------------------------------------

export interface VersionInfo {
  version: string;
  exe?: string;
  pid?: number;
}

export interface LocalModel {
  id: string;
  loaded: boolean;
}

export interface TagModel {
  name: string;
  size?: number;
  details?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface Provider {
  id: string;
  name: string;
  base_url?: string;
  wire?: string;
  key_set?: boolean;
  /** Whether a request would be routed upstream (the daemon holds a usable key). */
  key_usable: boolean;
  key_optional?: boolean;
  /** The variable the provider's key comes from on the computer, e.g. `ANTHROPIC_API_KEY`. */
  key_env?: string;
  /** Number of models the provider lists. */
  models?: number;
}

export interface ProviderModel {
  id: string;
  /** The effort levels the model takes (`reasoning_effort`), e.g. low…max; absent when unknown. */
  thinking?: string[];
  [key: string]: unknown;
}

/** What `PUT|DELETE /llmman/providers/:id/key` answer (never the key). */
export interface ProviderKeyResult {
  id: string;
  key_usable: boolean;
  key_env?: string;
  /** The provider's variable is set on the computer and wins over the stored key. */
  env_override: boolean;
}

export interface ShowInfo {
  capabilities?: string[];
  [key: string]: unknown;
}

/**
 * One `/api/pull` NDJSON line. A `{notice}` line (e.g. a verification note
 * just before `success`) has no `status`.
 */
export interface PullProgress {
  status?: string;
  notice?: string;
  digest?: string;
  total?: number;
  completed?: number;
}

export type ChatRole = "system" | "user" | "assistant";

/** OpenAI's content parts: text, and an image as a URL (here a `data:` URL). */
export type ContentPart = { type: "text"; text: string } | { type: "image_url"; image_url: { url: string } };

export interface ChatMessage {
  role: ChatRole;
  /** A string, or parts when the message carries images. */
  content: string | ContentPart[];
}

export interface ChatDelta {
  content: string;
  reasoning: string;
}

export interface ChatOptions {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  maxTokens?: number;
  /** `reasoning_effort`, one of the model's thinking levels; unset: its default. */
  reasoningEffort?: string;
  signal?: AbortSignal;
  onDelta?: (delta: ChatDelta) => void;
}

export interface ChatResult {
  finishReason: string | null;
}

// ---- Client --------------------------------------------------------------

export class ApiClient {
  readonly url: string;
  private readonly key: string;
  private readonly fetchImpl: FetchLike;
  private readonly keySource: KeySource | undefined;

  constructor(
    conn: ServerConnection,
    fetchImpl: FetchLike = expoFetch as unknown as FetchLike,
    opts: { keySource?: KeySource } = {},
  ) {
    this.url = conn.url.replace(/\/+$/, "");
    // A key pasted with a trailing newline or space would be rejected; no
    // llmman key contains whitespace at its ends.
    this.key = (conn.key ?? "").trim();
    this.fetchImpl = fetchImpl;
    this.keySource = opts.keySource;
  }

  /** The key sent now: the key source's when there is one. */
  currentKey(): string {
    return this.keySource ? (this.keySource.get() ?? "").trim() : this.key;
  }

  /**
   * Headers that authenticate a request made outside this client. Only for
   * a route that answers itself: a native loader (Image) follows redirects
   * with the same headers, so never attach them to one that may redirect
   * off the daemon (e.g. /llmman/search/avatar: resolve it with
   * `redirect: "manual"` and load the target without headers).
   */
  authHeaders(): Record<string, string> {
    const key = this.currentKey();
    return key ? { authorization: `Bearer ${key}` } : {};
  }

  /** An absolute URL for `path` on this server. */
  urlFor(path: string): string {
    return `${this.url}/${path.replace(/^\/+/, "")}`;
  }

  /**
   * `fetch` against the server with the bearer key. Transport failures
   * become `network`/`timeout` ApiErrors; the caller's own abort is
   * rethrown as an AbortError (see `isAbort`). The caller's signal stays
   * wired to the request after the headers arrive, so it also stops a
   * streaming body; `timeoutMs` covers only the wait for the headers.
   */
  async request(path: string, init: RequestInitLike = {}, opts: RequestOptions = {}): Promise<ResponseLike> {
    const sent = this.currentKey();
    const r = await this.send(path, init, opts);
    // llmman's own 401, with a key source: ask once (one prompt shared by
    // every request that got one) and retry once. A
    // key changed meanwhile (another request's prompt) is retried unasked.
    if (this.keySource && wantsKey(r) && (this.currentKey() !== sent || (await this.keySource.ask()))) {
      return this.send(path, init, opts);
    }
    return r;
  }

  private async send(path: string, init: RequestInitLike, opts: RequestOptions): Promise<ResponseLike> {
    const headers: Record<string, string> = { ...init.headers, ...this.authHeaders() };
    const outer = opts.signal ?? init.signal;
    if (outer?.aborted) throw abortError();

    let signal = outer;
    let timedOut = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    if (opts.timeoutMs && opts.timeoutMs > 0) {
      const controller = new AbortController();
      outer?.addEventListener("abort", () => controller.abort(), { once: true });
      timer = setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, opts.timeoutMs);
      signal = controller.signal;
    }

    try {
      // Called unbound: on the web expo/fetch is the browser's fetch, which
      // throws "Illegal invocation" when called as a method of the client.
      const fetchImpl = this.fetchImpl;
      return await fetchImpl(this.urlFor(path), { ...init, headers, signal });
    } catch (e) {
      if (timedOut) throw new ApiError(`no answer from ${this.url} in time`, 0, "timeout");
      if (outer?.aborted || isAbort(e)) throw abortError();
      const detail = e instanceof Error && e.message ? `: ${e.message}` : "";
      throw new ApiError(`could not reach ${this.url}${detail}`, 0, "network");
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  async getJson<T>(path: string, opts: RequestOptions = {}): Promise<T> {
    const r = await this.request(path, {}, opts);
    if (!r.ok) throw await errorFrom(r);
    return parseJson<T>(await r.text());
  }

  async postJson(path: string, body: unknown, opts: RequestOptions = {}): Promise<ResponseLike> {
    const r = await this.request(
      path,
      { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) },
      opts,
    );
    if (!r.ok) throw await errorFrom(r);
    return r;
  }

  /** `GET /api/version` → `{version, exe, pid}`. */
  async version(opts?: RequestOptions): Promise<VersionInfo> {
    const body = await this.getJson<Partial<VersionInfo>>("api/version", opts);
    if (typeof body?.version !== "string") {
      throw new ApiError("this server answered, but not like llmman serve", 200, "protocol");
    }
    return body as VersionInfo;
  }

  /** Local models from `GET /v1/models`, with the daemon's loaded flag. */
  async listLocal(opts?: RequestOptions): Promise<LocalModel[]> {
    const body = await this.getJson<{ data?: { id: string; status?: { value?: string } }[] }>(
      "v1/models",
      opts,
    );
    return (body.data || []).map((m) => ({ id: m.id, loaded: m.status?.value === "loaded" }));
  }

  /** `GET /api/tags`: local models with sizes and formats. */
  async listLocalDetailed(opts?: RequestOptions): Promise<TagModel[]> {
    const body = await this.getJson<{ models?: TagModel[] }>("api/tags", opts);
    return body.models || [];
  }

  /** Providers the daemon knows, from `GET /llmman/providers`. */
  async listProviders(opts?: RequestOptions): Promise<Provider[]> {
    const body = await this.getJson<{ providers?: Provider[] }>("llmman/providers", opts);
    return body.providers || [];
  }

  /** One provider's models: `GET /llmman/providers/:id`. */
  async providerModels(id: string, opts?: RequestOptions): Promise<ProviderModel[]> {
    const body = await this.getJson<{ models?: ProviderModel[] }>(
      `llmman/providers/${encodeURIComponent(id)}`,
      opts,
    );
    return body.models || [];
  }

  /**
   * Stores a provider's API key on the computer (`PUT /llmman/providers/:id/key`);
   * the daemon spends it from its next request. The key is not kept here.
   */
  async setProviderKey(id: string, apiKey: string, opts?: RequestOptions): Promise<ProviderKeyResult> {
    const r = await this.request(
      `llmman/providers/${encodeURIComponent(id)}/key`,
      {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ api_key: apiKey }),
      },
      opts,
    );
    if (!r.ok) throw await errorFrom(r);
    return parseJson<ProviderKeyResult>(await r.text());
  }

  /** Removes the key stored on the computer (`DELETE /llmman/providers/:id/key`). */
  async deleteProviderKey(id: string, opts?: RequestOptions): Promise<ProviderKeyResult> {
    const r = await this.request(
      `llmman/providers/${encodeURIComponent(id)}/key`,
      { method: "DELETE" },
      opts,
    );
    if (!r.ok) throw await errorFrom(r);
    return parseJson<ProviderKeyResult>(await r.text());
  }

  /** `GET /llmman/node`: this daemon's model memory, and what it has loaded and stored. */
  nodeInfo<T = { memory?: number; [key: string]: unknown }>(opts?: RequestOptions): Promise<T> {
    return this.getJson<T>("llmman/node", opts);
  }

  /** `POST /api/show`. */
  async showModel(model: string, opts?: RequestOptions): Promise<ShowInfo> {
    const r = await this.postJson("api/show", { model }, opts);
    return parseJson<ShowInfo>(await r.text());
  }

  /** Load a model without generating (Ollama's empty-prompt convention). */
  async loadModel(model: string, opts?: RequestOptions): Promise<void> {
    await this.postJson("api/generate", { model, prompt: "", stream: false }, opts);
  }

  /** Unload immediately (`keep_alive: 0` with an empty prompt). */
  async unloadModel(model: string, opts?: RequestOptions): Promise<void> {
    await this.postJson("api/generate", { model, prompt: "", stream: false, keep_alive: 0 }, opts);
  }

  /** `DELETE /api/delete`. */
  async deleteModel(model: string, opts?: RequestOptions): Promise<void> {
    const r = await this.request(
      "api/delete",
      { method: "DELETE", headers: { "content-type": "application/json" }, body: JSON.stringify({ model }) },
      opts,
    );
    if (!r.ok) throw await errorFrom(r);
  }

  /**
   * `POST /api/pull`; `onProgress` gets each NDJSON `{status, digest?,
   * total?, completed?}` line. Rejects on the daemon's `{error}` line, a
   * transport failure, or a stream that ends before `success`.
   */
  async pull(model: string, onProgress?: (p: PullProgress) => void, signal?: AbortSignal): Promise<void> {
    const r = await this.postJson("api/pull", { model, stream: true }, { signal });
    for await (const line of lines(r.body, signal)) {
      let event: PullProgress & { error?: unknown };
      try {
        event = JSON.parse(line);
      } catch {
        continue;
      }
      if (event.error) throw new ApiError(String(event.error), 200, "protocol");
      onProgress?.(event);
      if (event.status === "success") return;
    }
    throw new ApiError("the pull ended before the daemon reported success", 200, "protocol");
  }

  /**
   * `POST /v1/chat/completions`, streaming. `onDelta` gets `{content,
   * reasoning}` per chunk (backends spell the reasoning field three ways).
   * Resolves with `{finishReason}`; a stream that ends without `[DONE]` or
   * a finish reason is truncated and rejects, as the daemon itself treats it.
   */
  async chat({
    model,
    messages,
    temperature,
    maxTokens,
    reasoningEffort,
    signal,
    onDelta,
  }: ChatOptions): Promise<ChatResult> {
    const body: Record<string, unknown> = { model, messages, stream: true };
    if (Number.isFinite(temperature)) body.temperature = temperature;
    if (Number.isFinite(maxTokens) && (maxTokens as number) > 0) body.max_tokens = maxTokens;
    if (reasoningEffort) body.reasoning_effort = reasoningEffort;

    const r = await this.postJson("v1/chat/completions", body, { signal });
    let finishReason: string | null = null;
    let done = false;
    for await (const line of lines(r.body, signal)) {
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") {
        done = true;
        break;
      }
      let chunk: ChatChunk;
      try {
        chunk = JSON.parse(data);
      } catch {
        continue;
      }
      if (chunk.error) {
        const e = chunk.error;
        const message = typeof e === "string" ? e : e?.message || JSON.stringify(e);
        throw new ApiError(message, 200, "protocol");
      }
      for (const choice of chunk.choices || []) {
        const delta = choice.delta || {};
        const reasoning = delta.reasoning_content ?? delta.reasoning ?? delta.thinking;
        if (delta.content || reasoning) {
          onDelta?.({ content: delta.content || "", reasoning: reasoning || "" });
        }
        if (choice.finish_reason) finishReason = choice.finish_reason;
      }
    }
    if (!done && !finishReason) {
      throw new ApiError("the reply ended before the model finished", 200, "protocol");
    }
    return { finishReason };
  }
}

interface ChatChunk {
  error?: { message?: string } | string;
  choices?: {
    delta?: { content?: string; reasoning_content?: string; reasoning?: string; thinking?: string };
    finish_reason?: string | null;
  }[];
}

export function parseJson<T>(text: string): T {
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new ApiError("this server answered, but not like llmman serve", 200, "protocol");
  }
}

export function abortError(): Error {
  const e = new Error("aborted");
  e.name = "AbortError";
  return e;
}
