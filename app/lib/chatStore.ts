// Open conversations and their replies in flight. A reply belongs to its
// conversation, not to a screen: leaving the chat keeps it streaming, and
// Stop works from wherever the conversation is shown. Screens read it with
// `useConversation(id)`.

import { useEffect, useState, useSyncExternalStore } from "react";

import { isAbort, splitRemoteRef, type ApiClient, type ChatMessage } from "@/api/client";
import { explain } from "@/api/errors";
import { generateAudio, generateImage, generateVideo } from "@/api/media";
import type { ChatSettings } from "@/lib/chatSettings";
import {
  deleteConversation,
  loadConversation,
  saveConversation,
  titleFrom,
  type Conversation,
  type Message,
} from "@/lib/conversations";
import {
  optionsFor,
  pngSize,
  wavSeconds,
  type GeneratedMedia,
  type MediaKind,
  type MediaOptions,
} from "@/lib/media";
import { removeMedia, storeMedia } from "@/lib/mediaFiles";

export type Phase = "idle" | "loading-model" | "thinking" | "streaming" | "generating";

/** A media generation in progress: what the status line under the reply says. */
export interface Generation {
  kind: MediaKind;
  model: string;
  /** The model was not loaded: the daemon loads it first. */
  loading: boolean;
  startedAt: number;
  /** Denoising steps, once the backend reports them (images only). */
  step?: number;
  total?: number;
}

export interface ConversationState {
  conv: Conversation | null;
  phase: Phase;
  generation?: Generation;
}

const states = new Map<string, ConversationState>();
const running = new Map<string, AbortController>();
const listeners = new Set<() => void>();
const IDLE: ConversationState = { conv: null, phase: "idle" };

function emit() {
  for (const l of listeners) l();
}

function set(id: string, next: Partial<ConversationState>) {
  states.set(id, { ...(states.get(id) ?? IDLE), ...next });
  emit();
}

/** Calls `fn` after any change to any conversation; returns an unsubscribe. */
export function subscribe(fn: () => void): () => void {
  listeners.add(fn);
  return () => void listeners.delete(fn);
}

/** A conversation's current state, outside React. */
export function getConversationState(id: string | undefined): ConversationState {
  return id ? (states.get(id) ?? IDLE) : IDLE;
}

export function useConversation(id: string | undefined): ConversationState {
  return useSyncExternalStore(subscribe, () => getConversationState(id));
}

/**
 * A conversation, read from storage when it is not in the store yet (a
 * screen opened by its address, after a reload); `missing` once there is
 * no such conversation.
 */
export function useStoredConversation(id: string | undefined): ConversationState & { missing: boolean } {
  const state = useConversation(id);
  const [missing, setMissing] = useState<string | null>(null);
  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    void openConversation(id).then((found) => !cancelled && !found && setMissing(id));
    return () => {
      cancelled = true;
    };
  }, [id]);
  return { ...state, missing: !id || missing === id };
}

/** Whether any reply is streaming (e.g. to warn before removing a server). */
export function isRunning(id: string): boolean {
  return running.has(id);
}

/** Whether a reply is streaming in any conversation (the web warns before leaving the page). */
export function anyRunning(): boolean {
  return running.size > 0;
}

/** Puts a conversation in the store: a new one, or one read from storage. */
export async function openConversation(id: string, fresh?: Conversation): Promise<Conversation | null> {
  const have = states.get(id)?.conv;
  if (have) return have;
  const conv = fresh ?? (await loadConversation(id));
  if (conv && !states.get(id)?.conv) set(id, { conv, phase: "idle" });
  return states.get(id)?.conv ?? null;
}

/** Forgets a conversation (after it was deleted). */
export function dropConversation(id: string) {
  running.get(id)?.abort();
  states.delete(id);
  emit();
}

/** Another model for the chat; its thinking level is the new model's default again. */
export function setModel(id: string, model: string) {
  const conv = states.get(id)?.conv;
  if (!conv || conv.model === model) return;
  const { effort: _old, ...rest } = conv;
  set(id, { conv: { ...rest, model } });
  void saveConversation(states.get(id)!.conv!).catch(() => {});
}

/** The chat's thinking level; undefined for the model's default. */
export function setEffort(id: string, effort: string | undefined) {
  const conv = states.get(id)?.conv;
  if (!conv || conv.effort === effort) return;
  const { effort: _old, ...rest } = conv;
  set(id, { conv: effort ? { ...rest, effort } : rest });
  void saveConversation(states.get(id)!.conv!).catch(() => {});
}

/** The chat's media generation options (the settings sheet, the kind toggle). */
export function setMediaOptions(id: string, media: MediaOptions) {
  const conv = states.get(id)?.conv;
  if (!conv) return;
  const next = { ...conv, media };
  set(id, { conv: next });
  void saveConversation(next).catch(() => {});
}

export function setSettings(id: string, settings: ChatSettings) {
  const conv = states.get(id)?.conv;
  if (!conv) return;
  const { settings: _old, ...rest } = conv;
  const next = Object.keys(settings).length ? { ...rest, settings } : rest;
  set(id, { conv: next });
  void saveConversation(next).catch(() => {});
}

/** Forgets conversations that were deleted with their server. */
/** A chat's new title; blank goes back to the one from its first message. */
export async function renameConversation(id: string, title: string): Promise<void> {
  const conv = await openConversation(id);
  if (!conv) return;
  const first = conv.messages.find((m) => m.role === "user")?.content ?? "";
  const next = { ...states.get(id)!.conv!, title: title.trim() || titleFrom(first) };
  set(id, { conv: next });
  await saveConversation(next);
}

export function dropConversations(ids: string[]) {
  for (const id of ids) {
    running.get(id)?.abort();
    states.delete(id);
  }
  emit();
}

export function stop(id: string) {
  running.get(id)?.abort();
}

/**
 * The history sent to the model: the system prompt if any, then answered
 * turns only (no failed or empty replies). Media generation prompts and
 * their results are not part of a chat.
 */
export function historyFor(messages: Message[], systemPrompt?: string): ChatMessage[] {
  const out: ChatMessage[] = [];
  if (systemPrompt?.trim()) out.push({ role: "system", content: systemPrompt.trim() });
  for (const m of messages) {
    if (m.generate || m.request || m.media) continue;
    if (m.role === "user" && m.images?.length) {
      // Images go as OpenAI `image_url` parts, after the text.
      out.push({
        role: "user",
        content: [
          ...(m.content ? [{ type: "text" as const, text: m.content }] : []),
          ...m.images.map((url) => ({ type: "image_url" as const, image_url: { url } })),
        ],
      });
    } else if (m.role === "user" || (m.role === "assistant" && m.content)) {
      out.push({ role: m.role, content: m.content });
    }
  }
  return out;
}

/** Adds the user's message and streams the reply. */
/** How a turn is sent: whether the model is loaded (for the status), and the server's name for errors. */
export interface TurnOptions {
  loaded?: boolean;
  server?: { name?: string; url: string };
  /** The media the model generates (from /api/show); set for a diffusion model, whose turns generate. */
  mediaKinds?: MediaKind[];
}

export async function send(
  client: ApiClient,
  id: string,
  text: string,
  opts: TurnOptions = {},
  images: string[] = [],
): Promise<void> {
  const conv = states.get(id)?.conv;
  if (!conv || running.has(id) || (!text.trim() && !images.length)) return;
  // Claimed before any await, so a second tap cannot start another reply.
  const controller = new AbortController();
  running.set(id, controller);
  const user: Message = {
    role: "user",
    content: text.trim(),
    ...(images.length ? { images } : {}),
    at: Date.now(),
  };
  const next: Conversation = {
    ...conv,
    title: conv.title || titleFrom(text) || "Image",
    messages: [...conv.messages, user],
    updatedAt: Date.now(),
  };
  set(id, { conv: next });
  try {
    await saveConversation(next);
  } catch {
    // Storage trouble must not lose the reply; the end of the turn saves again.
  }
  await generate(client, id, controller, opts);
}

/**
 * A conversation cut at message `index`: that message and everything after
 * it go (as the web UI's Edit and Retry cut it).
 */
export function cutAt(messages: Message[], index: number): { kept: Message[]; dropped: Message[] } {
  return { kept: messages.slice(0, index), dropped: messages.slice(index) };
}

/**
 * Writes a reply again: the last one, or the reply at `index`, cutting the
 * conversation after it, as the web UI's Retry does.
 */
export async function retry(
  client: ApiClient,
  id: string,
  opts: TurnOptions = {},
  index?: number,
): Promise<void> {
  const conv = states.get(id)?.conv;
  if (!conv || running.has(id)) return;
  const at = index ?? conv.messages.length - 1;
  const target = conv.messages[at];
  const { kept, dropped } =
    target?.role === "assistant" ? cutAt(conv.messages, at) : cutAt(conv.messages, at + 1);
  if (kept[kept.length - 1]?.role !== "user") return;
  const controller = new AbortController();
  running.set(id, controller);
  set(id, { conv: { ...conv, messages: kept } });
  // The replaced results' files go with them.
  for (const m of dropped) if (m.media) void removeMedia(m.media);
  await generate(client, id, controller, opts);
}

/**
 * Takes a user message back to edit: the conversation is cut at it, and its
 * text and images are returned for the composer. A conversation left with
 * nothing is removed from storage until it is sent again.
 */
export async function editFrom(
  id: string,
  index: number,
): Promise<{ text: string; images: string[] } | null> {
  const conv = states.get(id)?.conv;
  const msg = conv?.messages[index];
  if (!conv || !msg || msg.role !== "user" || running.has(id)) return null;
  const { kept, dropped } = cutAt(conv.messages, index);
  for (const m of dropped) if (m.media) void removeMedia(m.media);
  const next: Conversation = { ...conv, messages: kept, updatedAt: Date.now() };
  set(id, { conv: next });
  if (kept.length) await saveConversation(next).catch(() => {});
  else await deleteConversation(id).catch(() => {});
  return { text: msg.content, images: msg.images ?? [] };
}

/** How often the screen sees new tokens; they arrive faster than is worth rendering. */
const FLUSH_MS = 60;

async function generate(client: ApiClient, id: string, controller: AbortController, opts: TurnOptions) {
  const start = states.get(id)!.conv!;
  // Mark the prompt: a generation prompt is not a chat turn (and a Retry
  // may switch it either way, with another model).
  const generating = !!opts.mediaKinds?.length;
  const at = start.messages.findLastIndex((m) => m.role === "user");
  const prompt = start.messages[at];
  if (prompt && !!prompt.generate !== generating) {
    const messages = start.messages.slice();
    const { generate: _was, ...rest } = prompt;
    messages[at] = generating ? { ...rest, generate: true } : rest;
    set(id, { conv: { ...start, messages } });
  }
  if (generating) return generateMedia(client, id, controller, opts);
  return chatTurn(client, id, controller, opts);
}

async function chatTurn(
  client: ApiClient,
  id: string,
  controller: AbortController,
  { loaded = true, server = { url: client.url } }: TurnOptions,
) {
  const start = states.get(id)!.conv!;
  const model = start.model;
  const history = historyFor(start.messages, start.settings?.systemPrompt);

  const reply: Message = { role: "assistant", content: "", reasoning: "", model, at: Date.now() };
  const index = start.messages.length;
  const remote = splitRemoteRef(model);
  const began = Date.now();
  set(id, {
    conv: { ...start, messages: [...start.messages, reply] },
    phase: !remote && !loaded ? "loading-model" : "thinking",
  });

  let pending = { content: "", reasoning: "" };
  // When the reasoning started and when the answer did: "Thought for 12s".
  let thinkStart: number | null = null;
  let thinkEnd: number | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  const write = (patch: Partial<Message>) => {
    const conv = states.get(id)?.conv;
    if (!conv) return;
    const messages = conv.messages.slice();
    messages[index] = { ...messages[index]!, ...patch };
    set(id, { conv: { ...conv, messages, updatedAt: Date.now() } });
  };
  const flush = () => {
    timer = null;
    if (!pending.content && !pending.reasoning) return;
    const cur = states.get(id)?.conv?.messages[index];
    if (!cur) return;
    write({
      content: cur.content + pending.content,
      reasoning: (cur.reasoning ?? "") + pending.reasoning,
    });
    pending = { content: "", reasoning: "" };
  };

  try {
    const result = await client.chat({
      model,
      messages: history,
      temperature: start.settings?.temperature,
      maxTokens: start.settings?.maxTokens,
      reasoningEffort: start.effort,
      signal: controller.signal,
      onDelta: ({ content, reasoning }) => {
        if (states.get(id)?.phase !== "streaming") set(id, { phase: "streaming" });
        if (reasoning && thinkStart === null) thinkStart = Date.now();
        if (content && thinkStart !== null && thinkEnd === null) thinkEnd = Date.now();
        pending.content += content;
        pending.reasoning += reasoning;
        timer ??= setTimeout(flush, FLUSH_MS);
      },
    });
    flush();
    write({ finishReason: result.finishReason });
  } catch (e) {
    flush();
    if (isAbort(e)) write({ stopped: true });
    else {
      const { title, detail, keyRejected } = explain(e, server);
      write({
        error: title,
        ...(detail ? { errorDetail: detail } : {}),
        ...(keyRejected ? { keyRejected } : {}),
      });
    }
  } finally {
    if (timer) clearTimeout(timer);
    running.delete(id);
    const conv = states.get(id)?.conv;
    const msg = conv?.messages[index];
    if (conv && msg) {
      const empty = !msg.content && !msg.reasoning;
      if (empty && msg.stopped) {
        // Stopped before a token: drop the empty reply, keep the prompt.
        set(id, { conv: { ...conv, messages: conv.messages.filter((_, k) => k !== index) } });
      } else if (empty && !msg.error) {
        write({ error: "The model returned nothing." });
      }
      if (thinkStart !== null && msg.reasoning) write({ thoughtMs: (thinkEnd ?? Date.now()) - thinkStart });
      // "SmolLM2 135M · 1.2s" under the reply, unless it was dropped above.
      if (states.get(id)?.conv?.messages[index]?.role === "assistant")
        write({ durationMs: Date.now() - began });
      set(id, { phase: "idle" });
      const final = states.get(id)?.conv;
      if (final) await saveConversation(final).catch(() => {});
    } else {
      set(id, { phase: "idle" });
    }
  }
}

/**
 * The media counterpart of a chat turn: the last user message is the
 * prompt, the reply is one picture, clip or sound, stored with the chat.
 */
async function generateMedia(
  client: ApiClient,
  id: string,
  controller: AbortController,
  { loaded = true, server = { url: client.url }, mediaKinds = [] }: TurnOptions,
) {
  const start = states.get(id)!.conv!;
  const model = start.model;
  const options = optionsFor(start.media, mediaKinds);
  const { kind } = options;
  const prompt = start.messages.findLast((m) => m.role === "user")?.content ?? "";
  const reply: Message = { role: "assistant", content: "", model, at: Date.now(), prompt, request: options };
  const index = start.messages.length;
  let generation: Generation = { kind, model, loading: !loaded, startedAt: Date.now() };
  set(id, { conv: { ...start, messages: [...start.messages, reply] }, phase: "generating", generation });

  const write = (patch: Partial<Message>) => {
    const conv = states.get(id)?.conv;
    if (!conv?.messages[index]) return;
    const messages = conv.messages.slice();
    messages[index] = { ...messages[index]!, ...patch };
    set(id, { conv: { ...conv, messages, updatedAt: Date.now() } });
  };

  const { kind: _kind, ...knobs } = options;
  const request = { ...knobs, model, prompt, signal: controller.signal };
  try {
    let media: Omit<GeneratedMedia, "path" | "blob">;
    let bytes: Uint8Array;
    if (kind === "video") {
      const r = await generateVideo(client, request);
      bytes = r.bytes;
      media = {
        kind,
        mime: r.mime,
        bytes: r.bytes.length,
        ...(r.width && r.height ? { width: r.width, height: r.height } : {}),
        ...(r.seconds ? { seconds: r.seconds } : {}),
        ...(r.fps ? { fps: r.fps } : {}),
        ...(r.hasAudio !== undefined ? { hasAudio: r.hasAudio } : {}),
        ...(r.revisedPrompt ? { revisedPrompt: r.revisedPrompt } : {}),
      };
    } else if (kind === "audio") {
      const r = await generateAudio(client, request);
      bytes = r.bytes;
      const seconds = wavSeconds(r.bytes) ?? options.seconds;
      media = { kind, mime: r.mime, bytes: r.bytes.length, ...(seconds ? { seconds } : {}) };
    } else {
      const r = await generateImage(client, {
        ...request,
        onProgress: ({ step, total }) => {
          generation = { ...generation, step, total };
          set(id, { generation });
        },
      });
      bytes = r.bytes;
      media = {
        kind,
        mime: r.mime,
        bytes: r.bytes.length,
        ...(pngSize(r.bytes) ?? {}),
        ...(r.revisedPrompt ? { revisedPrompt: r.revisedPrompt } : {}),
      };
    }
    // Stopped while the bytes came in: nothing is kept.
    if (controller.signal.aborted) throw abortLike();
    const where = await storeMedia(id, kind, bytes, media.mime);
    write({ media: { ...media, ...where } });
  } catch (e) {
    if (isAbort(e)) write({ stopped: true });
    else {
      const { title, detail, keyRejected } = explain(e, server);
      write({
        error: title,
        ...(detail ? { errorDetail: detail } : {}),
        ...(keyRejected ? { keyRejected } : {}),
      });
    }
  } finally {
    running.delete(id);
    const conv = states.get(id)?.conv;
    const msg = conv?.messages[index];
    // A stopped image is dropped, as the web UI drops it (the daemon stops
    // too). A video or sound keeps a note: the daemon finishes it anyway.
    if (conv && msg?.stopped && kind === "image") {
      set(id, { conv: { ...conv, messages: conv.messages.filter((_, k) => k !== index) } });
    }
    set(id, { phase: "idle", generation: undefined });
    const final = states.get(id)?.conv;
    if (final) await saveConversation(final).catch(() => {});
  }
}

function abortLike(): Error {
  const e = new Error("aborted");
  e.name = "AbortError";
  return e;
}

/** For tests: forget everything. */
export function __resetChatStore() {
  for (const c of running.values()) c.abort();
  running.clear();
  states.clear();
  emit();
}
