/** @jest-environment node */

import { ApiClient, ApiError } from "../client";
import { avatarUrl, modelCard, popular, search, SEARCH_LIMIT } from "../search";
import { fakeFetch, fixture, response } from "@/test/fakeFetch";

const URL = "http://10.0.0.5:17434";

function redirect(location: string | null) {
  return response("", { status: 302, headers: location === null ? {} : { location } });
}

describe("search", () => {
  it("asks for the query and a page per registry, with the key", async () => {
    const { fetch, calls } = fakeFetch(() => response(fixture("search-qwen.json")));
    const hits = await search(new ApiClient({ url: URL, key: "sk-1" }, fetch), "qwen 3/coder");
    expect(calls[0]!.url).toBe(`${URL}/llmman/search?q=qwen%203%2Fcoder&limit=${SEARCH_LIMIT}`);
    expect(calls[0]!.init.headers).toEqual({ authorization: "Bearer sk-1" });
    expect(hits.length).toBeGreaterThan(0);
    expect(hits.every((h) => typeof h.name === "string")).toBe(true);
  });

  it("reads an empty result", async () => {
    const { fetch } = fakeFetch(() => response(fixture("search-empty.json")));
    expect(await search(new ApiClient({ url: URL }, fetch), "zzz")).toEqual([]);
  });

  it("passes the abort signal on", async () => {
    const { fetch, calls } = fakeFetch(() => response(fixture("search-empty.json")));
    const abort = new AbortController();
    await search(new ApiClient({ url: URL }, fetch), "q", { signal: abort.signal, limit: 5 });
    expect(calls[0]!.url).toBe(`${URL}/llmman/search?q=q&limit=5`);
    expect(calls[0]!.init.signal).toBe(abort.signal);
  });
});

describe("popular", () => {
  it("lists Docker Hub's rows before Hugging Face's", async () => {
    const { fetch, calls } = fakeFetch(() => response(fixture("search-popular.json")));
    const hits = await popular(new ApiClient({ url: URL }, fetch));
    expect(calls[0]!.url).toBe(`${URL}/llmman/search/popular?limit=${SEARCH_LIMIT}`);
    const firstHf = hits.findIndex((h) => h.name.startsWith("hf.co/"));
    expect(firstHf).toBeGreaterThan(0);
    expect(hits.slice(firstHf).every((h) => h.name.startsWith("hf.co/"))).toBe(true);
  });
});

describe("modelCard", () => {
  it("encodes the repo name and reads its variants", async () => {
    const { fetch, calls } = fakeFetch(() => response(fixture("model-hf-qwen3-coder-30b.json")));
    const card = await modelCard(
      new ApiClient({ url: URL }, fetch),
      "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
    );
    expect(calls[0]!.url).toBe(
      `${URL}/llmman/search/model?name=hf.co%2Funsloth%2FQwen3-Coder-30B-A3B-Instruct-GGUF`,
    );
    expect(card.variants).toHaveLength(27);
    expect(card.variants.filter((v) => v.default).map((v) => v.tag)).toEqual(["Q4_K_M"]);
  });

  it("reads a gated repo with no README", async () => {
    const { fetch } = fakeFetch(() => response(fixture("model-gated.json")));
    const card = await modelCard(
      new ApiClient({ url: URL }, fetch),
      "hf.co/meta-llama/Llama-3.2-1B-Instruct",
    );
    expect(card.gated).toBe(true);
    expect(card.readme).toBeNull();
  });

  it("turns a registry failure into the daemon's 502 message", async () => {
    const { fetch } = fakeFetch(() => response(fixture("model-notfound.json"), { status: 502 }));
    const err = await modelCard(
      new ApiClient({ url: URL }, fetch),
      "hf.co/nobody-xyz/nothing-here-GGUF",
    ).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(502);
    expect(err.message).toMatch(/^HF model info: GET https:\/\/huggingface\.co\/api\/models\//);
  });
});

describe("avatarUrl", () => {
  const picture =
    "https://cdn-avatars.huggingface.co/v1/production/uploads/62ecdc18b72a69615d6bd857/E4lkPz1TZNLzIFr_dR273.png";

  it("reads the redirect itself, so the key stays with the daemon", async () => {
    const { fetch, calls } = fakeFetch(() => redirect(picture));
    const url = await avatarUrl(new ApiClient({ url: URL, key: "sk-1" }, fetch), "hf.co/unsloth/Qwen3-GGUF");
    expect(url).toBe(picture);
    expect(calls).toHaveLength(1);
    expect(calls[0]!.url).toBe(`${URL}/llmman/search/avatar?name=hf.co%2Funsloth%2FQwen3-GGUF`);
    expect(calls[0]!.init.redirect).toBe("manual");
    expect(calls[0]!.init.headers).toEqual({ authorization: "Bearer sk-1" });
  });

  it("is null when the owner has no picture", async () => {
    const { fetch } = fakeFetch(() => response("", { status: 404 }));
    expect(await avatarUrl(new ApiClient({ url: URL }, fetch), "docker.io/ai/qwen3")).toBeNull();
  });

  it.each([
    null,
    "",
    "/relative/path.png",
    "http://plain.example/a.png",
    "javascript:alert(1)",
    "https:///x",
  ])("ignores a location of %p", async (location) => {
    const { fetch } = fakeFetch(() => redirect(location));
    expect(await avatarUrl(new ApiClient({ url: URL }, fetch), "hf.co/o/r")).toBeNull();
  });

  it("ignores a 200, which means a redirect was followed", async () => {
    const { fetch } = fakeFetch(() => response("png bytes", { status: 200, contentType: "image/png" }));
    expect(await avatarUrl(new ApiClient({ url: URL }, fetch), "hf.co/o/r")).toBeNull();
  });
});
