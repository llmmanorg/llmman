// llmman's earlier built-in web UI kept its chats in the browser's
// IndexedDB (database `llmman`, store `conversations`):
//   { id, title, model, createdAt, updatedAt, systemPrompt, temperature,
//     maxTokens, media?, messages: [{ role, content, reasoning?, model?, at,
//     generate?, prompt?, request?, media?: { kind, blob, ... }, error?,
//     stopped?, finishReason? }] }
// Unset values are `null` there. This maps one record to the app's shape;
// the web storage (chatStorage.web.ts) reads them and imports them once.

import type { ChatSettings } from "@/lib/chatSettings";
import type { Conversation, Message } from "@/lib/conversations";
import { MEDIA_KINDS, MIME_TYPES, normalizeOptions, type GeneratedMedia, type MediaKind } from "@/lib/media";

const str = (v: unknown): string | undefined => (typeof v === "string" ? v : undefined);
const num = (v: unknown): number | undefined => (typeof v === "number" && Number.isFinite(v) ? v : undefined);

function mediaFrom(raw: unknown): GeneratedMedia | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const r = raw as Record<string, unknown>;
  const blob = typeof Blob !== "undefined" && r.blob instanceof Blob ? r.blob : undefined;
  if (!blob || !MEDIA_KINDS.includes(r.kind as MediaKind)) return undefined;
  const kind = r.kind as MediaKind;
  const out: GeneratedMedia = { kind, mime: blob.type || MIME_TYPES[kind], bytes: blob.size, blob };
  for (const k of ["width", "height", "seconds", "fps"] as const) {
    const v = num(r[k]);
    if (v) out[k] = v;
  }
  if (typeof r.hasAudio === "boolean") out.hasAudio = r.hasAudio;
  if (str(r.revisedPrompt)) out.revisedPrompt = r.revisedPrompt as string;
  return out;
}

function messageFrom(raw: unknown): Message | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  if (r.role !== "user" && r.role !== "assistant") return null;
  const m: Message = { role: r.role, content: str(r.content) ?? "", at: num(r.at) ?? 0 };
  if (Array.isArray(r.images)) {
    const images = r.images.filter((u): u is string => typeof u === "string" && u.startsWith("data:"));
    if (images.length) m.images = images;
  }
  if (str(r.reasoning)) m.reasoning = r.reasoning as string;
  if (str(r.model)) m.model = r.model as string;
  if (str(r.finishReason)) m.finishReason = r.finishReason as string;
  if (str(r.error)) m.error = r.error as string;
  if (r.stopped === true) m.stopped = true;
  if (r.generate === true) m.generate = true;
  if (str(r.prompt) !== undefined) m.prompt = r.prompt as string;
  const request = normalizeOptions(r.request);
  if (request && m.role === "assistant") m.request = request;
  const media = mediaFrom(r.media);
  if (media) m.media = media;
  return m;
}

/** One web UI conversation as the app keeps it, on `server`; null for a record it cannot use. */
export function fromWebUi(raw: unknown, server: string): Conversation | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  const id = str(r.id);
  if (!id || !Array.isArray(r.messages)) return null;
  const messages = r.messages.map(messageFrom).filter((m): m is Message => m !== null);
  if (!messages.length) return null;
  const settings: ChatSettings = {};
  if (str(r.systemPrompt)?.trim()) settings.systemPrompt = r.systemPrompt as string;
  const temperature = num(r.temperature);
  if (temperature !== undefined) settings.temperature = temperature;
  const maxTokens = num(r.maxTokens);
  if (maxTokens !== undefined && maxTokens > 0) settings.maxTokens = maxTokens;
  const createdAt = num(r.createdAt) ?? messages[0]!.at;
  const media = normalizeOptions(r.media);
  return {
    id,
    server,
    title: str(r.title) ?? "",
    model: str(r.model) ?? "",
    ...(Object.keys(settings).length ? { settings } : {}),
    ...(media ? { media } : {}),
    createdAt,
    updatedAt: num(r.updatedAt) ?? createdAt,
    messages,
  };
}
