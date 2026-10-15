import { readFileSync } from "fs";
import { join } from "path";

import {
  clock,
  describeMedia,
  describeOptions,
  formatBytes,
  generationStatus,
  mediaFileName,
  mediaKindsOf,
  normalizeOptions,
  optionsFor,
  parseMediaOptions,
  pngSize,
  toMediaInput,
  wavSeconds,
  type MediaOptionsInput,
} from "../media";

const blank: MediaOptionsInput = toMediaInput(undefined);

describe("mediaKindsOf", () => {
  it("keeps the media capabilities in the composer's order", () => {
    expect(mediaKindsOf(["audio", "video", "image", "vision"])).toEqual(["image", "video", "audio"]);
    expect(mediaKindsOf(["completion"])).toEqual([]);
    expect(mediaKindsOf(null)).toEqual([]);
  });
});

describe("optionsFor", () => {
  it("keeps the chat's kind when the model makes it, else takes the model's first", () => {
    expect(optionsFor({ kind: "video", steps: 8 }, ["image", "video"])).toEqual({ kind: "video", steps: 8 });
    expect(optionsFor({ kind: "audio", steps: 8 }, ["image", "video"])).toEqual({ kind: "image", steps: 8 });
    expect(optionsFor(undefined, ["audio"])).toEqual({ kind: "audio" });
  });
});

describe("parseMediaOptions", () => {
  it("leaves blank fields to the model", () => {
    expect(parseMediaOptions(blank, "image")).toEqual({ ok: true, value: { kind: "image" } });
  });

  it("takes values inside the web UI's limits", () => {
    const r = parseMediaOptions(
      {
        width: "768",
        height: "512",
        seconds: "2.5",
        steps: "30",
        seed: "4294967295",
        cfgScale: "7,5",
        negativePrompt: "blurry",
      },
      "video",
    );
    expect(r).toEqual({
      ok: true,
      value: {
        kind: "video",
        width: 768,
        height: 512,
        seconds: 2.5,
        steps: 30,
        seed: 4294967295,
        cfgScale: 7.5,
        negativePrompt: "blurry",
      },
    });
  });

  it.each([
    ["width", "500", "A multiple of 32 from 32 to 4096."],
    ["width", "8192", "A multiple of 32 from 32 to 4096."],
    ["seconds", "0.3", "0.5 to 42, in steps of 0.5."],
    ["seconds", "1.25", "0.5 to 42, in steps of 0.5."],
    ["steps", "0", "A whole number from 1 to 200."],
    ["steps", "2.5", "A whole number from 1 to 200."],
    ["seed", "-1", "A whole number from 0 to 4294967295."],
    ["seed", "4294967296", "A whole number from 0 to 4294967295."],
    ["cfgScale", "31", "0 to 30, in steps of 0.5."],
    ["cfgScale", "abc", "0 to 30, in steps of 0.5."],
  ] as const)("refuses %s = %j", (field, value, message) => {
    const input = { ...blank, [field]: value };
    if (field === "width") input.height = "512";
    const r = parseMediaOptions(input, "image");
    expect(r.ok).toBe(false);
    expect(!r.ok && r.errors[field]).toBe(message);
  });

  it("wants both width and height, or neither", () => {
    const r = parseMediaOptions({ ...blank, width: "768" }, "image");
    expect(!r.ok && r.errors).toEqual({ height: "Set both width and height, or neither." });
    expect(parseMediaOptions({ ...blank, height: "512" }, "image").ok).toBe(false);
  });

  it("round-trips through the form", () => {
    const opts = { kind: "image" as const, width: 64, height: 32, seed: 0, cfgScale: 0 };
    expect(parseMediaOptions(toMediaInput(opts), "image")).toEqual({ ok: true, value: opts });
  });
});

describe("normalizeOptions", () => {
  it("reads the web UI's options, where unset is null", () => {
    expect(
      normalizeOptions({
        kind: "video",
        width: null,
        height: null,
        seconds: 4,
        steps: null,
        seed: 7,
        cfgScale: null,
        negativePrompt: "",
      }),
    ).toEqual({ kind: "video", seconds: 4, seed: 7 });
    expect(normalizeOptions({ kind: "hologram", steps: "9" })).toEqual({ kind: "image" });
    expect(normalizeOptions(null)).toBeUndefined();
  });
});

describe("mediaFileName", () => {
  // 2026-10-15 09:05:03 local time, as the web UI stamps it.
  const at = new Date(2026, 9, 15, 9, 5, 3).getTime();

  it("is <prompt slug>-<YYYYMMDD-HHMMSS>.<ext>, as the web UI names it", () => {
    expect(mediaFileName("A red fox, in fresh snow!", at, "image")).toBe(
      "a-red-fox-in-fresh-snow-20261015-090503.png",
    );
    expect(mediaFileName("Waves", at, "video")).toBe("waves-20261015-090503.mp4");
    expect(mediaFileName("Rain on a roof", at, "audio")).toBe("rain-on-a-roof-20261015-090503.wav");
  });

  it("cuts the slug at 50 characters and falls back to `image` without one", () => {
    const name = mediaFileName("word ".repeat(30), at, "image");
    expect(name.length).toBe(50 + "-20261015-090503.png".length);
    expect(mediaFileName("日本語 ✨", at, "video")).toBe("image-20261015-090503.mp4");
    expect(mediaFileName(undefined, at, "audio")).toBe("image-20261015-090503.wav");
  });
});

describe("captions and status", () => {
  it("describes a result as the web UI does", () => {
    expect(
      describeMedia(
        {
          kind: "video",
          mime: "video/mp4",
          bytes: 1_468_006,
          width: 768,
          height: 512,
          seconds: 4,
          fps: 24,
          hasAudio: true,
        },
        { kind: "video", steps: 20, seed: 7 },
      ),
    ).toBe("768×512 · 4s · 24 fps · with audio · 20 steps · seed 7 · 1.4 MB");
    expect(describeMedia({ kind: "audio", mime: "audio/wav", bytes: 512, seconds: 2.25 })).toBe(
      "2.3s · 512 B",
    );
    expect(describeMedia({ kind: "image", mime: "image/png", bytes: 2048 }, { kind: "image", seed: 0 })).toBe(
      "seed 0 · 2.0 KB",
    );
  });

  it("summarizes settings for the kind", () => {
    expect(describeOptions(undefined)).toBe("Model defaults");
    expect(describeOptions({ kind: "audio", width: 64, height: 64, seconds: 3 })).toBe("3s");
    expect(describeOptions({ kind: "image", width: 64, height: 64, seconds: 3, seed: 1 })).toBe(
      "64×64 · seed 1",
    );
  });

  it("words the status line like the web UI", () => {
    const base = { kind: "image" as const, model: "sd-turbo", loading: true, elapsedMs: 75_400 };
    expect(generationStatus(base)).toBe("Loading sd-turbo, then generating image… 1:15");
    expect(generationStatus({ ...base, loading: false, kind: "video" })).toBe("Generating video… 1:15");
    expect(generationStatus({ ...base, step: 12, total: 30 })).toBe("Generating image… step 12/30");
    expect(clock(9_999)).toBe("0:09");
  });

  it("formats bytes like the web UI's util.js", () => {
    expect([0, 999, 1536, 10 * 1024 * 1024, 5 * 1024 ** 3].map(formatBytes)).toEqual([
      "0 B",
      "999 B",
      "1.5 KB",
      "10 MB",
      "5.0 GB",
    ]);
  });
});

describe("pngSize", () => {
  it("reads the IHDR size", () => {
    const sse = readFileSync(join(__dirname, "..", "..", "api", "__fixtures__", "images-plain.json"), "utf8");
    const b64 = JSON.parse(sse).data[0].b64_json as string;
    expect(pngSize(Uint8Array.from(Buffer.from(b64, "base64")))).toEqual({ width: 64, height: 48 });
    expect(pngSize(new Uint8Array([1, 2, 3]))).toBeNull();
  });
});

describe("wavSeconds", () => {
  function wav(seconds: number, rate = 16000, channels = 1, dataSize?: number): Uint8Array {
    const n = Math.round(seconds * rate) * channels * 2;
    const buf = new ArrayBuffer(44 + n);
    const v = new DataView(buf);
    const tag = (at: number, s: string) => [...s].forEach((ch, i) => v.setUint8(at + i, ch.charCodeAt(0)));
    tag(0, "RIFF");
    v.setUint32(4, 36 + n, true);
    tag(8, "WAVE");
    tag(12, "fmt ");
    v.setUint32(16, 16, true);
    v.setUint16(20, 1, true);
    v.setUint16(22, channels, true);
    v.setUint32(24, rate, true);
    v.setUint32(28, rate * channels * 2, true);
    v.setUint16(32, channels * 2, true);
    v.setUint16(34, 16, true);
    tag(36, "data");
    v.setUint32(40, dataSize ?? n, true);
    return new Uint8Array(buf);
  }

  it("is the data size over the byte rate", () => {
    expect(wavSeconds(wav(2.5))).toBeCloseTo(2.5);
    expect(wavSeconds(wav(1, 44100, 2))).toBeCloseTo(1);
  });

  it("runs to the end for a streamed wav that does not know its size", () => {
    expect(wavSeconds(wav(0.5, 16000, 1, 0xffffffff))).toBeCloseTo(0.5);
  });

  it("is null for anything else", () => {
    expect(wavSeconds(new Uint8Array(8))).toBeNull();
  });
});
