// Media generation, as `llmman run` does it: a diffusion
// model's capabilities say what it makes, a conversation keeps the options
// of its runs, and each reply is one picture, clip or sound. Pure helpers;
// the requests are in api/media.ts and the files in lib/mediaFiles.ts.

/** What a media generation model can make, in the composer's order. */
export const MEDIA_KINDS = ["image", "video", "audio"] as const;
export type MediaKind = (typeof MEDIA_KINDS)[number];

/** The media kinds in `capabilities` (from /api/show), in the composer's order. */
export function mediaKindsOf(capabilities: readonly string[] | null | undefined): MediaKind[] {
  return MEDIA_KINDS.filter((k) => capabilities?.includes(k));
}

/**
 * A run's options, as the web UI's settings popover has them. Unset is the
 * model's default. A conversation keeps them under `media`.
 */
export interface MediaOptions {
  kind: MediaKind;
  width?: number;
  height?: number;
  seconds?: number;
  steps?: number;
  seed?: number;
  /** Guidance (`cfg_scale`). */
  cfgScale?: number;
  negativePrompt?: string;
}

/** A generated picture, clip or sound, kept with its conversation. */
export interface GeneratedMedia {
  kind: MediaKind;
  mime: string;
  /** Size in bytes. */
  bytes: number;
  width?: number;
  height?: number;
  seconds?: number;
  fps?: number;
  hasAudio?: boolean;
  /** The prompt as the model rewrote it, when it says. */
  revisedPrompt?: string;
  /** Native: the file, relative to the app's document directory. */
  path?: string;
  /** Web: the bytes, stored in IndexedDB as they are. */
  blob?: Blob;
}

export const PLACEHOLDERS: Record<MediaKind, string> = {
  image: "Describe the image to generate",
  video: "Describe the video to generate",
  audio: "Describe the sound to generate",
};

export const KIND_LABELS: Record<MediaKind, string> = { image: "Image", video: "Video", audio: "Audio" };

const EXTENSIONS: Record<MediaKind, string> = { image: "png", video: "mp4", audio: "wav" };
export const MIME_TYPES: Record<MediaKind, string> = {
  image: "image/png",
  video: "video/mp4",
  audio: "audio/wav",
};

export function extensionFor(kind: MediaKind): string {
  return EXTENSIONS[kind];
}

/**
 * `opts` with a kind the model offers: the conversation's choice when the
 * model has it, else the model's first kind (the web UI's rule).
 */
export function optionsFor(opts: MediaOptions | undefined, kinds: readonly MediaKind[]): MediaOptions {
  const base: MediaOptions = opts ? { ...opts } : { kind: "image" };
  if (kinds.length && !kinds.includes(base.kind)) base.kind = kinds[0]!;
  return base;
}

/**
 * Options read from storage or from llmman's web UI (which keeps unset
 * values as `null`): only the fields that make sense, with their types.
 */
export function normalizeOptions(raw: unknown): MediaOptions | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const r = raw as Record<string, unknown>;
  const kind = MEDIA_KINDS.includes(r.kind as MediaKind) ? (r.kind as MediaKind) : "image";
  const out: MediaOptions = { kind };
  const num = (k: keyof MediaOptions) => {
    const v = r[k];
    if (typeof v === "number" && Number.isFinite(v)) (out as unknown as Record<string, number>)[k] = v;
  };
  for (const k of ["width", "height", "seconds", "steps", "seed", "cfgScale"] as const) num(k);
  if (typeof r.negativePrompt === "string" && r.negativePrompt) out.negativePrompt = r.negativePrompt;
  return out;
}

// ---- The settings form -------------------------------------------------

export interface MediaOptionsInput {
  width: string;
  height: string;
  seconds: string;
  steps: string;
  seed: string;
  cfgScale: string;
  negativePrompt: string;
}

export type MediaErrors = Partial<Record<keyof MediaOptionsInput, string>>;

export function toMediaInput(o: MediaOptions | undefined): MediaOptionsInput {
  const s = (v: number | undefined) => (v === undefined ? "" : String(v));
  return {
    width: s(o?.width),
    height: s(o?.height),
    seconds: s(o?.seconds),
    steps: s(o?.steps),
    seed: s(o?.seed),
    cfgScale: s(o?.cfgScale),
    negativePrompt: o?.negativePrompt ?? "",
  };
}

/**
 * The limits of the web UI's inputs (min, max and step), so a value that
 * page would refuse is refused here too and never reaches a request.
 */
export const LIMITS = {
  width: { min: 32, max: 4096, step: 32, int: true },
  height: { min: 32, max: 4096, step: 32, int: true },
  seconds: { min: 0.5, max: 42, step: 0.5, int: false },
  steps: { min: 1, max: 200, step: 1, int: true },
  seed: { min: 0, max: 4294967295, step: 1, int: true },
  cfgScale: { min: 0, max: 30, step: 0.5, int: false },
} as const;

type NumberField = keyof typeof LIMITS;

function limitText(k: NumberField): string {
  const { min, max, step, int } = LIMITS[k];
  if (k === "width" || k === "height") return `A multiple of ${step} from ${min} to ${max}.`;
  if (int) return `A whole number from ${min} to ${max}.`;
  return `${min} to ${max}, in steps of ${step}.`;
}

/** Checks typed values; blank is the model's default. Size takes both or neither. */
export function parseMediaOptions(
  input: MediaOptionsInput,
  kind: MediaKind,
): { ok: true; value: MediaOptions } | { ok: false; errors: MediaErrors } {
  const errors: MediaErrors = {};
  const value: MediaOptions = { kind };
  for (const k of Object.keys(LIMITS) as NumberField[]) {
    const text = input[k].trim().replace(",", ".");
    if (!text) continue;
    const { min, max, step } = LIMITS[k];
    const n = Number(text);
    // On the step grid from `min`, as an <input type=number> checks it.
    const onGrid = Math.abs((n - min) / step - Math.round((n - min) / step)) < 1e-9;
    if (!/^[+]?\d*\.?\d+$/.test(text) || !Number.isFinite(n) || n < min || n > max || !onGrid) {
      errors[k] = limitText(k);
    } else {
      value[k] = n;
    }
  }
  // Some backends need both dimensions or neither.
  if (!errors.width && !errors.height && !input.width.trim() !== !input.height.trim()) {
    errors.height = "Set both width and height, or neither.";
  }
  if (input.negativePrompt.trim()) value.negativePrompt = input.negativePrompt;
  return Object.keys(errors).length ? { ok: false, errors } : { ok: true, value };
}

/** A one-line summary for a settings row, e.g. "768×512 · 20 steps · seed 7". */
export function describeOptions(o: MediaOptions | undefined): string {
  if (!o) return "Model defaults";
  const parts = [
    o.kind !== "audio" && o.width && o.height ? `${o.width}×${o.height}` : "",
    o.kind !== "image" && o.seconds ? `${trimNumber(o.seconds)}s` : "",
    o.steps ? `${o.steps} steps` : "",
    o.seed !== undefined ? `seed ${o.seed}` : "",
    o.cfgScale !== undefined ? `guidance ${o.cfgScale}` : "",
    o.negativePrompt?.trim() ? "negative prompt" : "",
  ].filter(Boolean);
  return parts.length ? parts.join(" · ") : "Model defaults";
}

// ---- Results -------------------------------------------------------------

function trimNumber(n: number): string {
  return Number(n).toFixed(1).replace(/\.0$/, "");
}

/** `1.4 MB`, as the web UI's util.js formats sizes. */
export function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n < 10 && i > 0 ? n.toFixed(1) : Math.round(n)} ${units[i]}`;
}

/** The caption under a result: "768×512 · 4s · 24 fps · with audio · 20 steps · seed 7 · 1.4 MB". */
export function describeMedia(media: GeneratedMedia, request?: MediaOptions): string {
  return [
    media.width && media.height ? `${media.width}×${media.height}` : "",
    media.seconds ? `${trimNumber(media.seconds)}s` : "",
    media.fps ? `${media.fps} fps` : "",
    media.hasAudio ? "with audio" : "",
    request?.steps ? `${request.steps} steps` : "",
    Number.isInteger(request?.seed) ? `seed ${request!.seed}` : "",
    formatBytes(media.bytes),
  ]
    .filter(Boolean)
    .join(" · ");
}

/**
 * The download name, `<prompt slug>-<YYYYMMDD-HHMMSS>.<ext>`, as the web UI
 * (and `llmman run`) names it, in local time.
 */
export function mediaFileName(prompt: string | undefined, at: number, kind: MediaKind): string {
  const slug = (prompt || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 50);
  const d = new Date(at);
  const p = (n: number) => String(n).padStart(2, "0");
  const stamp = `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  return `${slug || "image"}-${stamp}.${EXTENSIONS[kind]}`;
}

/** The elapsed time as `m:ss`. */
export function clock(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** What a turn in progress says, word for word as the web UI's status line. */
export function generationStatus({
  kind,
  model,
  loading,
  elapsedMs,
  step,
  total,
}: {
  kind: MediaKind;
  model: string;
  loading: boolean;
  elapsedMs: number;
  step?: number;
  total?: number;
}): string {
  if (total) return `Generating ${kind}… step ${step ?? 0}/${total}`;
  const verb = loading ? `Loading ${model}, then generating` : "Generating";
  return `${verb} ${kind}… ${clock(elapsedMs)}`;
}

/** A PNG's size from its IHDR chunk, or null for anything else. */
export function pngSize(bytes: Uint8Array): { width: number; height: number } | null {
  const sig = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (bytes.length < 24 || sig.some((b, i) => bytes[i] !== b)) return null;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const width = view.getUint32(16);
  const height = view.getUint32(20);
  return width && height ? { width, height } : null;
}

/** A WAV file's length in seconds, from its `fmt ` and `data` chunks, or null. */
export function wavSeconds(bytes: Uint8Array): number | null {
  if (bytes.length < 12) return null;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const tag = (at: number) => String.fromCharCode(...bytes.subarray(at, at + 4));
  if (tag(0) !== "RIFF" || tag(8) !== "WAVE") return null;
  let byteRate = 0;
  let at = 12;
  while (at + 8 <= bytes.length) {
    const id = tag(at);
    const size = view.getUint32(at + 4, true);
    if (id === "fmt " && at + 16 <= bytes.length) byteRate = view.getUint32(at + 16, true);
    if (id === "data") {
      // A streamed wav may say 0 or 0xFFFFFFFF: then it runs to the end.
      const len = size && size !== 0xffffffff ? Math.min(size, bytes.length - at - 8) : bytes.length - at - 8;
      return byteRate ? len / byteRate : null;
    }
    at += 8 + size + (size % 2);
  }
  return null;
}
