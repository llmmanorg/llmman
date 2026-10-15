// Media generation on the daemon:
// a diffusion model answers on `/v1/images/generations` (streamed, with a
// step counter), `/v1/videos` (one synchronous job, then its `content_url`)
// and `/v1/audio/speech`. Unset knobs are left out (the model's default).

import { ApiError, errorFrom, lines, parseJson, type ApiClient, type ResponseLike } from "./client";

export interface MediaKnobs {
  width?: number;
  height?: number;
  seconds?: number;
  steps?: number;
  seed?: number;
  cfgScale?: number;
  negativePrompt?: string;
}

/** The bytes of a result and their type. */
export interface MediaPayload {
  bytes: Uint8Array;
  mime: string;
}

export interface MediaRequest extends MediaKnobs {
  model: string;
  prompt: string;
  signal?: AbortSignal;
}

/** The knobs shared by the three routes, only those set; `omit` names the ones a route lacks. */
export function mediaFields(
  { width, height, steps, seed, cfgScale, negativePrompt, seconds }: MediaKnobs,
  omit: ("width" | "height" | "seconds")[] = [],
): Record<string, number | string> {
  const body: Record<string, number | string> = {};
  if (width && width > 0) body.width = Math.round(width);
  if (height && height > 0) body.height = Math.round(height);
  if (steps && steps > 0) body.steps = Math.round(steps);
  if (Number.isInteger(seed) && seed! >= 0) body.seed = seed!;
  if (cfgScale && cfgScale > 0) body.cfg_scale = cfgScale;
  if (negativePrompt?.trim()) body.negative_prompt = negativePrompt.trim();
  if (seconds && seconds > 0) body.seconds = seconds;
  for (const k of omit) delete body[k];
  return body;
}

/** Base64 to bytes, without a quadratic string walk. */
export function bytesFromBase64(b64: string): Uint8Array {
  const bin = atob(b64.replace(/\s+/g, ""));
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** A body read to its end; `signal` stops the read. */
async function readBytes(r: ResponseLike, signal?: AbortSignal): Promise<Uint8Array> {
  if (!r.body) return new Uint8Array(0);
  const reader = r.body.getReader();
  const onAbort = () => void reader.cancel().catch(() => {});
  signal?.addEventListener("abort", onAbort, { once: true });
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    for (;;) {
      let chunk: ReadableStreamReadResult<Uint8Array>;
      try {
        chunk = await reader.read();
      } catch (e) {
        if (signal?.aborted) throw abort();
        const detail = e instanceof Error && e.message ? `: ${e.message}` : "";
        throw new ApiError(`the connection dropped${detail}`, 0, "network");
      }
      if (signal?.aborted) throw abort();
      if (chunk.done) break;
      chunks.push(chunk.value);
      size += chunk.value.length;
    }
  } finally {
    signal?.removeEventListener("abort", onAbort);
  }
  const out = new Uint8Array(size);
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}

function abort(): Error {
  const e = new Error("aborted");
  e.name = "AbortError";
  return e;
}

/** The response's media type, or `fallback` when it sent none (or a generic one). */
function typeOf(r: ResponseLike, fallback: string): string {
  const t = (r.headers.get("content-type") ?? "").split(";")[0]!.trim();
  return t && t !== "application/octet-stream" ? t : fallback;
}

// ---- Images ----------------------------------------------------------------

export interface ImageProgress {
  step: number;
  total: number;
}

export interface ImageResult extends MediaPayload {
  revisedPrompt: string;
}

/** One SSE event of the image stream, as the daemon sends them (src/mediagen/server.rs). */
export type ImageEvent =
  | { type: "progress"; step: number; total: number }
  | { type: "completed"; b64: string; revisedPrompt: string }
  | { type: "error"; message: string };

/** An SSE line of `/v1/images/generations`, or null for anything else (comments, keep-alives). */
export function parseImageEvent(line: string): ImageEvent | null {
  if (!line.startsWith("data:")) return null;
  let ev: {
    type?: string;
    step?: number;
    total?: number;
    b64_json?: string;
    revised_prompt?: string;
    error?: { message?: string } | string;
  };
  try {
    ev = JSON.parse(line.slice(5).trim());
  } catch {
    return null;
  }
  if (ev?.type === "image_generation.progress") {
    return { type: "progress", step: ev.step || 0, total: ev.total || 0 };
  }
  if (ev?.type === "image_generation.completed") {
    if (!ev.b64_json) return { type: "error", message: "the reply carried no image" };
    return { type: "completed", b64: ev.b64_json, revisedPrompt: ev.revised_prompt || "" };
  }
  if (ev?.type === "error") {
    const e = ev.error;
    const message = (typeof e === "string" ? e : e?.message) || "image generation failed";
    return { type: "error", message };
  }
  return null;
}

function imageResult(item: { b64_json?: string; revised_prompt?: string } | undefined): ImageResult {
  if (!item?.b64_json) throw new ApiError("the reply carried no image", 200, "protocol");
  return {
    bytes: bytesFromBase64(item.b64_json),
    mime: "image/png",
    revisedPrompt: item.revised_prompt || "",
  };
}

/**
 * `POST /v1/images/generations`, streamed: `onProgress` gets `{step,
 * total}` per denoising step. A backend that does not stream answers with
 * plain JSON, which is taken as is.
 */
export async function generateImage(
  client: ApiClient,
  { model, prompt, signal, onProgress, ...knobs }: MediaRequest & { onProgress?: (p: ImageProgress) => void },
): Promise<ImageResult> {
  const body = {
    model,
    prompt,
    stream: true,
    response_format: "b64_json",
    ...mediaFields(knobs, ["seconds"]),
  };
  const r = await client.postJson("v1/images/generations", body, { signal });
  if (!(r.headers.get("content-type") || "").includes("text/event-stream")) {
    const json = parseJson<{ data?: { b64_json?: string; revised_prompt?: string }[] }>(await r.text());
    return imageResult(json.data?.[0]);
  }
  for await (const line of lines(r.body, signal)) {
    const ev = parseImageEvent(line);
    if (!ev) continue;
    if (ev.type === "progress") onProgress?.({ step: ev.step, total: ev.total });
    else if (ev.type === "completed")
      return imageResult({ b64_json: ev.b64, revised_prompt: ev.revisedPrompt });
    else throw new ApiError(ev.message, 200, "protocol");
  }
  throw new ApiError("the stream ended without an image", 200, "protocol");
}

// ---- Video -----------------------------------------------------------------

/** The completed job `/v1/videos` answers with. */
export interface VideoJob {
  id?: string;
  size?: string;
  seconds?: string | number;
  fps?: number;
  n_frames?: number;
  has_audio?: boolean;
  revised_prompt?: string;
  content_url?: string;
  frames?: unknown[];
}

export interface VideoResult extends MediaPayload {
  job: VideoJob;
  width?: number;
  height?: number;
  seconds?: number;
  fps?: number;
  hasAudio?: boolean;
  revisedPrompt: string;
}

/**
 * `POST /v1/videos` (synchronous), then `GET` its `content_url` with the
 * key. Stopping abandons the wait; the daemon finishes the clip anyway.
 */
export async function generateVideo(
  client: ApiClient,
  { model, prompt, signal, fps, ...knobs }: MediaRequest & { fps?: number },
): Promise<VideoResult> {
  const body: Record<string, unknown> = { model, prompt, stream: false, ...mediaFields(knobs) };
  if (fps && fps > 0) body.fps = fps;
  const r = await client.postJson("v1/videos", body, { signal });
  const job = parseJson<VideoJob>(await r.text());
  if (!job.content_url) {
    const frames = job.n_frames || (job.frames || []).length;
    throw new ApiError(
      `the server generated ${frames} frames but has no ffmpeg to mux an mp4`,
      200,
      "protocol",
    );
  }
  // Relative to the server's base URL, so a gateway prefix works.
  const content = await client.request(String(job.content_url).replace(/^\/+/, ""), {}, { signal });
  if (!content.ok) throw await errorFrom(content);
  const bytes = await readBytes(content, signal);
  const [w, h] = String(job.size || "")
    .split("x")
    .map(Number);
  const seconds = Number(job.seconds);
  return {
    bytes,
    mime: typeOf(content, "video/mp4"),
    job,
    ...(w && h ? { width: w, height: h } : {}),
    ...(seconds > 0 ? { seconds } : {}),
    ...(job.fps ? { fps: job.fps } : {}),
    ...(job.has_audio !== undefined ? { hasAudio: job.has_audio } : {}),
    revisedPrompt: job.revised_prompt || "",
  };
}

// ---- Audio -----------------------------------------------------------------

/** `POST /v1/audio/speech` (OpenAI's field is `input`); the answer is a wav. */
export async function generateAudio(
  client: ApiClient,
  { model, prompt, signal, ...knobs }: MediaRequest,
): Promise<MediaPayload> {
  const body = { model, input: prompt, response_format: "wav", ...mediaFields(knobs, ["width", "height"]) };
  const r = await client.postJson("v1/audio/speech", body, { signal });
  return { bytes: await readBytes(r, signal), mime: typeOf(r, "audio/wav") };
}
