/** @jest-environment node */

import { ApiClient, ApiError, isAbort, type ResponseLike } from "../client";
import {
  bytesFromBase64,
  generateAudio,
  generateImage,
  generateVideo,
  mediaFields,
  parseImageEvent,
  type ImageProgress,
} from "../media";
import { fakeFetch, fixture, llmman401, response } from "@/test/fakeFetch";

const URL = "http://10.0.0.5:17434";
const PNG_SIG = [0x89, 0x50, 0x4e, 0x47];

/** A response whose body is `bytes`, in chunks. */
function binary(bytes: Uint8Array, contentType: string | null, chunk = 1000): ResponseLike {
  let at = 0;
  const headers: Record<string, string> = contentType ? { "content-type": contentType } : {};
  return {
    ok: true,
    status: 200,
    statusText: "OK",
    headers: { get: (n) => headers[n.toLowerCase()] ?? null },
    body: new ReadableStream<Uint8Array>({
      pull(c) {
        if (at >= bytes.length) return c.close();
        c.enqueue(bytes.slice(at, at + chunk));
        at += chunk;
      },
    }),
    text: async () => "",
  };
}

const bodyOf = (init: { body?: string }) => JSON.parse(init.body ?? "{}");

describe("parseImageEvent", () => {
  it("reads progress, completed and error events as the daemon sends them", () => {
    expect(parseImageEvent('data: {"type":"image_generation.progress","step":3,"total":20}')).toEqual({
      type: "progress",
      step: 3,
      total: 20,
    });
    expect(
      parseImageEvent('data: {"type":"image_generation.completed","b64_json":"QUJD","revised_prompt":"x"}'),
    ).toEqual({ type: "completed", b64: "QUJD", revisedPrompt: "x" });
    expect(parseImageEvent('data: {"type":"error","error":{"message":"too big"}}')).toEqual({
      type: "error",
      message: "too big",
    });
  });

  it("ignores what is not an event: comments, other types, bad JSON", () => {
    expect(parseImageEvent(": keep-alive")).toBeNull();
    expect(parseImageEvent("event: ping")).toBeNull();
    expect(parseImageEvent('data: {"type":"something.else"}')).toBeNull();
    expect(parseImageEvent("data: {not json")).toBeNull();
  });

  it("turns a completed event without an image, or an error without a message, into an error", () => {
    expect(parseImageEvent('data: {"type":"image_generation.completed"}')).toEqual({
      type: "error",
      message: "the reply carried no image",
    });
    expect(parseImageEvent('data: {"type":"error"}')).toEqual({
      type: "error",
      message: "image generation failed",
    });
  });
});

describe("mediaFields", () => {
  it("sends only what is set, with the daemon's names", () => {
    expect(mediaFields({})).toEqual({});
    expect(
      mediaFields({
        width: 768,
        height: 512,
        steps: 20,
        seed: 0,
        cfgScale: 7.5,
        negativePrompt: "  blurry ",
        seconds: 4,
      }),
    ).toEqual({
      width: 768,
      height: 512,
      steps: 20,
      seed: 0,
      cfg_scale: 7.5,
      negative_prompt: "blurry",
      seconds: 4,
    });
    expect(mediaFields({ width: 768, height: 512, seconds: 4 }, ["width", "height"])).toEqual({ seconds: 4 });
    expect(mediaFields({ seed: -1, steps: 0, cfgScale: 0, negativePrompt: "  " })).toEqual({});
  });
});

describe("generateImage", () => {
  it("streams progress and resolves with the PNG and the revised prompt", async () => {
    const { fetch, calls } = fakeFetch(() =>
      response(fixture("images-stream.sse"), { contentType: "text/event-stream", chunk: 97 }),
    );
    const client = new ApiClient({ url: URL, key: "k" }, fetch);
    const progress: ImageProgress[] = [];
    const r = await generateImage(client, {
      model: "sd-turbo",
      prompt: "a fox",
      width: 64,
      height: 48,
      seconds: 3,
      onProgress: (p) => progress.push(p),
    });
    expect(calls[0]!.url).toBe(`${URL}/v1/images/generations`);
    expect(calls[0]!.init.headers?.authorization).toBe("Bearer k");
    expect(bodyOf(calls[0]!.init)).toEqual({
      model: "sd-turbo",
      prompt: "a fox",
      stream: true,
      response_format: "b64_json",
      width: 64,
      height: 48,
    });
    expect(progress).toEqual([1, 2, 3, 4].map((step) => ({ step, total: 4 })));
    expect([...r.bytes.slice(0, 4)]).toEqual(PNG_SIG);
    expect(r.mime).toBe("image/png");
    expect(r.revisedPrompt).toBe("a red fox in fresh snow, morning light");
  });

  it("takes a backend's plain JSON answer when it does not stream", async () => {
    const { fetch } = fakeFetch(() => response(fixture("images-plain.json")));
    const r = await generateImage(new ApiClient({ url: URL }, fetch), { model: "m", prompt: "p" });
    expect([...r.bytes.slice(0, 4)]).toEqual(PNG_SIG);
    expect(r.revisedPrompt).toBe("");
  });

  it("rejects with the daemon's error event", async () => {
    const { fetch } = fakeFetch(() =>
      response(fixture("images-error.sse"), { contentType: "text/event-stream" }),
    );
    const err = await generateImage(new ApiClient({ url: URL }, fetch), { model: "m", prompt: "p" }).catch(
      (e) => e,
    );
    expect(err).toBeInstanceOf(ApiError);
    expect(err.message).toBe("the requested size exceeds the model's pixel budget");
  });

  it("rejects a stream that ends without an image", async () => {
    const truncated = fixture("images-stream.sse").split("\n\n").slice(0, 2).join("\n\n");
    const { fetch } = fakeFetch(() => response(truncated, { contentType: "text/event-stream" }));
    await expect(
      generateImage(new ApiClient({ url: URL }, fetch), { model: "m", prompt: "p" }),
    ).rejects.toThrow("the stream ended without an image");
  });

  it("stops on abort with an AbortError", async () => {
    const controller = new AbortController();
    const { fetch } = fakeFetch(() =>
      response(fixture("images-stream.sse"), { contentType: "text/event-stream", chunk: 40 }),
    );
    const err = await generateImage(new ApiClient({ url: URL }, fetch), {
      model: "m",
      prompt: "p",
      signal: controller.signal,
      onProgress: () => controller.abort(),
    }).catch((e) => e);
    expect(isAbort(err)).toBe(true);
  });

  it("passes the daemon's 401 through as the key problem it is", async () => {
    const { fetch } = fakeFetch(() => llmman401());
    const err = await generateImage(new ApiClient({ url: URL }, fetch), { model: "m", prompt: "p" }).catch(
      (e) => e,
    );
    expect(err.unauthorized).toBe(true);
  });
});

describe("generateVideo", () => {
  const mp4 = Uint8Array.from({ length: 5000 }, (_, i) => i % 251);

  it("posts the job, then fetches its relative content_url with the key", async () => {
    const { fetch, calls } = fakeFetch((url) =>
      url.endsWith("/v1/videos") ? response(fixture("video-job.json")) : binary(mp4, "video/mp4"),
    );
    const client = new ApiClient({ url: `${URL}/gw`, key: "k" }, fetch);
    const r = await generateVideo(client, { model: "ltx", prompt: "waves", seconds: 4, steps: 8 });
    expect(bodyOf(calls[0]!.init)).toEqual({
      model: "ltx",
      prompt: "waves",
      stream: false,
      seconds: 4,
      steps: 8,
    });
    // Under the base URL's prefix, as the web UI keeps it relative.
    expect(calls[1]!.url).toBe(`${URL}/gw/v1/videos/video_1760500000_0a1b2c3d_0/content`);
    expect(calls[1]!.init.headers?.authorization).toBe("Bearer k");
    expect(r.bytes).toEqual(mp4);
    expect(r).toMatchObject({
      mime: "video/mp4",
      width: 768,
      height: 512,
      seconds: 4,
      fps: 24,
      hasAudio: false,
    });
  });

  it("says why when the daemon has no ffmpeg to make an mp4", async () => {
    const job = { ...JSON.parse(fixture("video-job.json")), content_url: undefined, frames: ["a", "b"] };
    delete job.n_frames;
    const { fetch } = fakeFetch(() => response(JSON.stringify(job)));
    await expect(
      generateVideo(new ApiClient({ url: URL }, fetch), { model: "m", prompt: "p" }),
    ).rejects.toThrow("the server generated 2 frames but has no ffmpeg to mux an mp4");
  });

  it("types an untyped body as mp4", async () => {
    const { fetch } = fakeFetch((url) =>
      url.endsWith("/v1/videos") ? response(fixture("video-job.json")) : binary(mp4, null),
    );
    const r = await generateVideo(new ApiClient({ url: URL }, fetch), { model: "m", prompt: "p" });
    expect(r.mime).toBe("video/mp4");
  });
});

describe("generateAudio", () => {
  it("sends the prompt as OpenAI's `input`, asks for a wav, and leaves out the size", async () => {
    const wav = Uint8Array.from([82, 73, 70, 70, 1, 2, 3]);
    const { fetch, calls } = fakeFetch(() => binary(wav, "audio/wav", 3));
    const r = await generateAudio(new ApiClient({ url: URL }, fetch), {
      model: "audio-model",
      prompt: "rain",
      width: 512,
      height: 512,
      seconds: 5,
    });
    expect(calls[0]!.url).toBe(`${URL}/v1/audio/speech`);
    expect(bodyOf(calls[0]!.init)).toEqual({
      model: "audio-model",
      input: "rain",
      response_format: "wav",
      seconds: 5,
    });
    expect(r).toEqual({ bytes: wav, mime: "audio/wav" });
  });
});

describe("bytesFromBase64", () => {
  it("decodes, ignoring line breaks", () => {
    expect([...bytesFromBase64("QU\nJD")]).toEqual([65, 66, 67]);
  });
});
