// Conversations live on the device, per server: an index of summaries, and
// each conversation's messages under its own key, so the list loads without
// reading every chat. Where: AsyncStorage on a phone, IndexedDB on the web
// (lib/chatStorage.ts and .web.ts).

import type { ChatSettings } from "@/lib/chatSettings";
import { importWebUiChats as importFromWebUi, storage } from "@/lib/chatStorage";
import { newId } from "@/lib/id";
import type { GeneratedMedia, MediaOptions } from "@/lib/media";
import { removeConversationMedia } from "@/lib/mediaFiles";

export interface Message {
  role: "user" | "assistant";
  content: string;
  /** Images attached to a user message, as `data:` URLs (sent as OpenAI `image_url` parts). */
  images?: string[];
  /** The model's reasoning, kept apart from the answer and shown collapsed. */
  reasoning?: string;
  /** How long the model reasoned before it answered. */
  thoughtMs?: number;
  /** The model that wrote an assistant message. */
  model?: string;
  /** How long an assistant message took, from the request to its last token. */
  durationMs?: number;
  at: number;
  finishReason?: string | null;
  /** Set when the reply failed: a plain headline (api/errors.ts explain). */
  error?: string;
  /** The raw error under the headline, shown small. */
  errorDetail?: string;
  /** llmman refused the phone's key: offer to update it rather than retry. */
  keyRejected?: boolean;
  /** The user pressed Stop. */
  stopped?: boolean;
  /** A user message that was a media generation prompt, not a chat turn. */
  generate?: boolean;
  /** A generated reply: the prompt it was made from. */
  prompt?: string;
  /** A generated reply: the options it was made with. */
  request?: MediaOptions;
  /** A generated reply: the picture, clip or sound. */
  media?: GeneratedMedia;
}

export interface Conversation {
  id: string;
  /** URL of the server the conversation belongs to. */
  server: string;
  title: string;
  model: string;
  /** The model's thinking level for this chat (`reasoning_effort`); unset: its default. */
  effort?: string;
  /** System prompt, temperature, max tokens; started from the server's defaults. */
  settings?: ChatSettings;
  /** The options of media generation runs, with a media generation model. */
  media?: MediaOptions;
  createdAt: number;
  updatedAt: number;
  messages: Message[];
}

export type ConversationSummary = Omit<Conversation, "messages"> & { preview: string };

export function newConversation(server: string, model: string, settings?: ChatSettings): Conversation {
  const now = Date.now();
  return {
    id: newId(),
    server,
    title: "",
    model,
    ...(settings && Object.keys(settings).length ? { settings: { ...settings } } : {}),
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

/** A title from the first prompt: its first line without markdown markers, shortened at a word. */
export function titleFrom(text: string, max = 48): string {
  const line = text
    .trim()
    .split("\n")[0]!
    .replace(/^\s*(#{1,6}|>|[-*+]|\d+[.)])\s+/, "")
    .replace(/(\*\*|__|~~|`+|\*)/g, "")
    .replace(/\s+/g, " ")
    .trim();
  if (line.length <= max) return line;
  const cut = line.slice(0, max);
  const space = cut.lastIndexOf(" ");
  return `${(space > max / 2 ? cut.slice(0, space) : cut).trimEnd()}…`;
}

function summarize(conv: Conversation): ConversationSummary {
  const { messages, ...rest } = conv;
  const last = [...messages].reverse().find((m) => m.content.trim());
  return { ...rest, preview: last ? last.content.trim().slice(0, 120) : "" };
}

/** The stored index; a read that fails throws. */
async function readIndexStrict(): Promise<ConversationSummary[]> {
  const parsed = await storage.getIndex();
  return Array.isArray(parsed) ? parsed.filter((s) => s && typeof s.id === "string") : [];
}

/** The stored index for showing, empty when it can't be read. */
async function readIndex(): Promise<ConversationSummary[]> {
  try {
    return await readIndexStrict();
  } catch {
    return [];
  }
}

const indexListeners = new Set<() => void>();

/** Calls `fn` after the conversation list changes (the drawer's list); returns an unsubscribe. */
export function onConversationsChanged(fn: () => void): () => void {
  indexListeners.add(fn);
  return () => indexListeners.delete(fn);
}

// Index writes are serialized: two chats saving at once must not drop one.
// An index that can't be read fails the write: writing over it would leave
// only the chat being saved.
let indexQueue: Promise<unknown> = Promise.resolve();
function updateIndex(fn: (index: ConversationSummary[]) => ConversationSummary[]): Promise<void> {
  const next = indexQueue.then(async () => {
    await storage.setIndex(fn(await readIndexStrict()));
    for (const l of indexListeners) l();
  });
  indexQueue = next.catch(() => {});
  return next;
}

/** A server's conversations, newest first. */
export async function listConversations(server: string): Promise<ConversationSummary[]> {
  await indexQueue;
  return (await readIndex()).filter((s) => s.server === server).sort((a, b) => b.updatedAt - a.updatedAt);
}

export async function loadConversation(id: string): Promise<Conversation | null> {
  try {
    return await storage.getChat(id);
  } catch {
    return null;
  }
}

/** Saves a conversation that has messages; an empty one is not kept. */
export async function saveConversation(conv: Conversation): Promise<void> {
  if (!conv.messages.length) return;
  await storage.setChat(conv);
  const summary = summarize(conv);
  await updateIndex((index) => [summary, ...index.filter((s) => s.id !== conv.id)]);
}

export async function deleteConversation(id: string): Promise<void> {
  await storage.removeChats([id]);
  await updateIndex((index) => index.filter((s) => s.id !== id));
  await removeConversationMedia([id]);
}

/** Removes a server's conversations; returns their ids. */
export async function deleteServerConversations(server: string): Promise<string[]> {
  await indexQueue;
  const ids = (await readIndex()).filter((s) => s.server === server).map((s) => s.id);
  await storage.removeChats(ids);
  await updateIndex((index) => index.filter((s) => s.server !== server));
  await removeConversationMedia(ids);
  return ids;
}

/** Every conversation on this device, every server's, newest first (Settings → Export). */
export async function allConversations(): Promise<Conversation[]> {
  await indexQueue;
  const index = (await readIndex()).sort((a, b) => b.updatedAt - a.updatedAt);
  const out: Conversation[] = [];
  for (const s of index) {
    const conv = await loadConversation(s.id);
    if (conv) out.push(conv);
  }
  return out;
}

/**
 * The export file: every conversation as JSON, as the web UI's Export
 * writes it. Media is left out (it has no JSON form), marked as omitted:
 * attached images, and a generated result's bytes (a Blob on the web, a
 * file on a phone), whose kind, type and size stay.
 */
export function exportJson(convs: Conversation[]): string {
  const omitMedia = (key: string, value: unknown) => {
    if (key === "images" && value) return { omitted: true };
    if (typeof Blob !== "undefined" && value instanceof Blob) {
      return { type: value.type, size: value.size, omitted: true };
    }
    if (key === "media" && value && typeof value === "object" && "bytes" in value) {
      const { blob: _blob, path: _path, ...rest } = value as GeneratedMedia;
      return { ...rest, omitted: true };
    }
    return value;
  };
  return JSON.stringify(convs, omitMedia, 2);
}

/**
 * Imports the chats of llmman's built-in web UI on this origin onto
 * `server`, once (the web only); returns how many were added.
 */
export function importWebUiChats(server: string): Promise<number> {
  // In the index queue: a chat saved meanwhile must not drop an import.
  const next = indexQueue.then(async () => {
    const added = await importFromWebUi(server);
    if (added) for (const l of indexListeners) l();
    return added;
  });
  indexQueue = next.catch(() => {});
  return next;
}

/** Removes every conversation on this device; returns their ids. */
export async function deleteAllConversations(): Promise<string[]> {
  await indexQueue;
  const ids = (await readIndex()).map((s) => s.id);
  await storage.removeChats(ids);
  await updateIndex(() => []);
  await removeConversationMedia(ids);
  return ids;
}
