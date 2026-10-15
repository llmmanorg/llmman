/** @jest-environment node */

import { ApiClient, ApiError, errorFrom, isAbort, lines, remoteRef, splitRemoteRef } from "../client";
import { chunked, fakeFetch, fixture, llmman401, response } from "@/test/fakeFetch";

const URL = "http://10.0.0.5:17434";

async function collect(gen: AsyncGenerator<string>): Promise<string[]> {
  const out: string[] = [];
  for await (const l of gen) out.push(l);
  return out;
}

describe("remote refs", () => {
  it("round-trips provider refs and leaves local ones alone", () => {
    const ref = remoteRef("openrouter", "meta/llama-3:free");
    expect(ref).toBe("llmman.provider/openrouter/meta/llama-3:free");
    expect(splitRemoteRef(ref)).toEqual({ provider: "openrouter", model: "meta/llama-3:free" });
    expect(splitRemoteRef("docker.io/ai/qwen3.5:0.8b")).toBeNull();
    expect(splitRemoteRef("llmman.provider/noslash")).toBeNull();
    expect(splitRemoteRef("")).toBeNull();
  });
});

describe("lines", () => {
  it("splits on newlines across chunk boundaries, drops CR and blanks, keeps a final unterminated line", async () => {
    const out = await collect(lines(chunked("a\r\n\nbé€\nc", 1)));
    expect(out).toEqual(["a", "bé€", "c"]);
  });

  it("yields nothing for a null body", async () => {
    expect(await collect(lines(null))).toEqual([]);
  });
});

describe("errorFrom", () => {
  it.each([
    ['{"error":"bad key"}', "bad key"],
    ['{"error":{"message":"nested"}}', "nested"],
    ['{"message":"plain"}', "plain"],
    ["not json at all\n", "not json at all"],
    ['{"error":{"code":5}}', '{"code":5}'],
  ])("reads %s", async (body, message) => {
    const e = await errorFrom(response(body, { status: 400 }));
    expect(e).toBeInstanceOf(ApiError);
    expect(e.message).toBe(message);
    expect(e.status).toBe(400);
    expect(e.kind).toBe("http");
  });

  it("falls back to the status line for an empty body", async () => {
    const e = await errorFrom({ ...response("", { status: 502 }), statusText: "Bad Gateway" });
    expect(e.message).toBe("502 Bad Gateway");
  });
});

describe("ApiClient.request", () => {
  it("sends the key as a bearer and joins paths without double slashes", async () => {
    const { fetch, calls } = fakeFetch(() => response('{"version":"0.1.0"}'));
    await new ApiClient({ url: `${URL}/`, key: "sk-1" }, fetch).version();
    expect(calls[0]!.url).toBe(`${URL}/api/version`);
    expect(calls[0]!.init.headers).toEqual({ authorization: "Bearer sk-1" });
  });

  it("sends no authorization header without a key", async () => {
    const { fetch, calls } = fakeFetch(() => response('{"version":"0.1.0"}'));
    await new ApiClient({ url: URL }, fetch).version();
    expect(calls[0]!.init.headers).toEqual({});
  });

  it("keeps a gateway path prefix", async () => {
    const { fetch, calls } = fakeFetch(() => response('{"data":[]}'));
    await new ApiClient({ url: "https://gw.example/llmman" }, fetch).listLocal();
    expect(calls[0]!.url).toBe("https://gw.example/llmman/v1/models");
  });

  it("turns a 401 into an unauthorized ApiError with the daemon's message", async () => {
    const { fetch } = fakeFetch(() => llmman401());
    const err = await new ApiClient({ url: URL, key: "wrong" }, fetch).listLocal().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.unauthorized).toBe(true);
    expect(err.message).toMatch(/requires an API key/);
  });

  it("does not take a relayed provider 401 (no llmman realm) for a bad llmman key", async () => {
    const body = '{"error":{"message":"Incorrect API key provided","type":"invalid_request_error"}}';
    const { fetch } = fakeFetch(() =>
      response(body, { status: 401, headers: { "www-authenticate": 'Bearer realm="OpenAI"' } }),
    );
    const err = await new ApiClient({ url: URL, key: "k" }, fetch)
      .chat({ model: "llmman.provider/openai/gpt", messages: [{ role: "user", content: "x" }] })
      .catch((e) => e);
    expect(err.status).toBe(401);
    expect(err.unauthorized).toBe(false);
    expect(err.message).toBe("Incorrect API key provided");
  });

  it("turns a transport failure into a network error", async () => {
    const fetch = async () => {
      throw new TypeError("Network request failed");
    };
    const err = await new ApiClient({ url: URL }, fetch).version().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.kind).toBe("network");
    expect(err.status).toBe(0);
    expect(err.message).toContain(URL);
  });

  it("times out a request that never answers", async () => {
    const fetch = (_url: string, init?: { signal?: AbortSignal }) =>
      new Promise<never>((_, reject) => {
        init?.signal?.addEventListener("abort", () => {
          const e = new Error("aborted");
          e.name = "AbortError";
          reject(e);
        });
      });
    const err = await new ApiClient({ url: URL }, fetch).version({ timeoutMs: 20 }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.kind).toBe("timeout");
  });

  it("rethrows the caller's own abort as an AbortError", async () => {
    const controller = new AbortController();
    const fetch = (_url: string, init?: { signal?: AbortSignal }) =>
      new Promise<never>((_, reject) => {
        init?.signal?.addEventListener("abort", () => reject(new TypeError("cancelled")));
      });
    const p = new ApiClient({ url: URL }, fetch).version({ signal: controller.signal });
    controller.abort();
    const err = await p.catch((e) => e);
    expect(isAbort(err)).toBe(true);
  });

  it("rejects a 200 that is not llmman as a protocol error", async () => {
    const { fetch } = fakeFetch(() => response("<html>router login</html>", { contentType: "text/html" }));
    const err = await new ApiClient({ url: URL }, fetch).version().catch((e) => e);
    expect(err.kind).toBe("protocol");
    const { fetch: f2 } = fakeFetch(() => response('{"ok":true}'));
    expect((await new ApiClient({ url: URL }, f2).version().catch((e) => e)).kind).toBe("protocol");
  });
});

describe("models and providers", () => {
  it("maps /v1/models with the loaded flag", async () => {
    const body = JSON.stringify({
      data: [
        { id: "a", status: { value: "loaded" } },
        { id: "b", status: { value: "unloaded" } },
        { id: "c" },
      ],
    });
    const { fetch } = fakeFetch(() => response(body));
    expect(await new ApiClient({ url: URL }, fetch).listLocal()).toEqual([
      { id: "a", loaded: true },
      { id: "b", loaded: false },
      { id: "c", loaded: false },
    ]);
  });

  it("lists providers and encodes a provider id in its path", async () => {
    const { fetch, calls } = fakeFetch((url) =>
      url.endsWith("/llmman/providers")
        ? response('{"providers":[{"id":"x","name":"X","key_usable":true}]}')
        : response('{"models":[{"id":"m1"}]}'),
    );
    const client = new ApiClient({ url: URL }, fetch);
    expect(await client.listProviders()).toEqual([{ id: "x", name: "X", key_usable: true }]);
    expect(await client.providerModels("a/b")).toEqual([{ id: "m1" }]);
    expect(calls[1]!.url).toBe(`${URL}/llmman/providers/a%2Fb`);
  });

  it("sets and removes a provider key, sending it only in the body", async () => {
    const answer = '{"id":"anthropic","key_usable":true,"key_env":"ANTHROPIC_API_KEY","env_override":false}';
    const { fetch, calls } = fakeFetch(() => response(answer));
    const client = new ApiClient({ url: URL, key: "k" }, fetch);
    expect(await client.setProviderKey("anthropic", "sk-ant")).toEqual(JSON.parse(answer));
    await client.deleteProviderKey("a/b");
    expect(calls.map((c) => [c.init.method, c.url.slice(URL.length), c.init.body ?? null])).toEqual([
      ["PUT", "/llmman/providers/anthropic/key", '{"api_key":"sk-ant"}'],
      ["DELETE", "/llmman/providers/a%2Fb/key", null],
    ]);
  });

  it("posts load, unload and delete the way the web UI does", async () => {
    const { fetch, calls } = fakeFetch(() => response("{}"));
    const client = new ApiClient({ url: URL }, fetch);
    await client.loadModel("m");
    await client.unloadModel("m");
    await client.deleteModel("m");
    expect(calls.map((c) => [c.init.method, c.url.slice(URL.length), JSON.parse(c.init.body!)])).toEqual([
      ["POST", "/api/generate", { model: "m", prompt: "", stream: false }],
      ["POST", "/api/generate", { model: "m", prompt: "", stream: false, keep_alive: 0 }],
      ["DELETE", "/api/delete", { model: "m" }],
    ]);
  });
});

describe("pull", () => {
  it("reports each NDJSON line and resolves on success", async () => {
    const { fetch } = fakeFetch(() =>
      response(fixture("pull-existing.ndjson"), { contentType: "application/x-ndjson" }),
    );
    const seen: (string | undefined)[] = [];
    await new ApiClient({ url: URL }, fetch).pull("m", (p) => seen.push(p.status));
    expect(seen).toEqual(["pulling manifest", "success"]);
  });

  it("passes progress fields through", async () => {
    const body = [
      '{"status":"pulling manifest"}',
      '{"status":"pulling sha256:ab","digest":"sha256:ab","total":100,"completed":40}',
      '{"status":"success"}',
    ].join("\n");
    const { fetch } = fakeFetch(() => response(body, { chunk: 3 }));
    const seen: unknown[] = [];
    await new ApiClient({ url: URL }, fetch).pull("m", (p) => seen.push(p));
    expect(seen[1]).toEqual({ status: "pulling sha256:ab", digest: "sha256:ab", total: 100, completed: 40 });
  });

  it("rejects on the daemon's error line", async () => {
    const { fetch } = fakeFetch(() => response(fixture("pull-error.ndjson")));
    const err = await new ApiClient({ url: URL }, fetch).pull("m").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.message).toMatch(/HF model info/);
  });

  it("rejects a stream that ends before success", async () => {
    const { fetch } = fakeFetch(() => response('{"status":"pulling manifest"}\n'));
    const err = await new ApiClient({ url: URL }, fetch).pull("m").catch((e) => e);
    expect(err.message).toMatch(/ended before/);
  });
});

describe("chat", () => {
  const messages = [{ role: "user" as const, content: "hi" }];

  it("streams content from a recorded reply and resolves with the finish reason", async () => {
    const { fetch, calls } = fakeFetch(() =>
      response(fixture("chat-smollm.sse"), { contentType: "text/event-stream" }),
    );
    let content = "";
    const result = await new ApiClient({ url: URL, key: "k" }, fetch).chat({
      model: "m",
      messages,
      temperature: 0.2,
      maxTokens: 64,
      reasoningEffort: "xhigh",
      onDelta: (d) => {
        content += d.content;
        expect(d.reasoning).toBe("");
      },
    });
    expect(content).toBe("Hello!");
    expect(result.finishReason).toBe("stop");
    expect(JSON.parse(calls[0]!.init.body!)).toEqual({
      model: "m",
      messages,
      stream: true,
      temperature: 0.2,
      max_tokens: 64,
      reasoning_effort: "xhigh",
    });
  });

  it("separates reasoning_content from content", async () => {
    const { fetch } = fakeFetch(() => response(fixture("chat-qwen-reasoning.sse"), { chunk: 13 }));
    let content = "";
    let reasoning = "";
    const result = await new ApiClient({ url: URL }, fetch).chat({
      model: "m",
      messages,
      onDelta: (d) => {
        content += d.content;
        reasoning += d.reasoning;
      },
    });
    expect(reasoning.startsWith("Thinking Process:")).toBe(true);
    expect(content).toBe("");
    expect(result.finishReason).toBe("length");
  });

  it.each(["reasoning", "thinking"])("reads reasoning spelled `%s`", async (field) => {
    const body = [
      `data: {"choices":[{"delta":{"${field}":"hmm"}}]}`,
      'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}',
      "data: [DONE]",
      "",
    ].join("\n\n");
    const { fetch } = fakeFetch(() => response(body));
    const deltas: unknown[] = [];
    await new ApiClient({ url: URL }, fetch).chat({ model: "m", messages, onDelta: (d) => deltas.push(d) });
    expect(deltas).toEqual([
      { content: "", reasoning: "hmm" },
      { content: "ok", reasoning: "" },
    ]);
  });

  it("rejects a truncated stream (no [DONE], no finish reason)", async () => {
    const full = fixture("chat-smollm.sse");
    const cut = full.slice(0, full.indexOf('"finish_reason":"stop"'));
    const truncated = cut.slice(0, cut.lastIndexOf("data:"));
    const { fetch } = fakeFetch(() => response(truncated));
    const err = await new ApiClient({ url: URL }, fetch).chat({ model: "m", messages }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.message).toBe("the reply ended before the model finished");
  });

  it("accepts a finish reason without [DONE]", async () => {
    const body = 'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}]}\n\n';
    const { fetch } = fakeFetch(() => response(body));
    expect(await new ApiClient({ url: URL }, fetch).chat({ model: "m", messages })).toEqual({
      finishReason: "stop",
    });
  });

  it("shows an in-stream error without a message as JSON", async () => {
    const body = 'data: {"error":{"code":"overloaded"}}\n\n';
    const { fetch } = fakeFetch(() => response(body));
    const err = await new ApiClient({ url: URL }, fetch).chat({ model: "m", messages }).catch((e) => e);
    expect(err.message).toBe('{"code":"overloaded"}');
  });

  it("rejects an error event inside the stream", async () => {
    const body = 'data: {"error":{"message":"out of memory"}}\n\ndata: [DONE]\n\n';
    const { fetch } = fakeFetch(() => response(body));
    const err = await new ApiClient({ url: URL }, fetch).chat({ model: "m", messages }).catch((e) => e);
    expect(err.message).toBe("out of memory");
  });

  it("rejects a non-2xx answer with the daemon's message", async () => {
    const { fetch } = fakeFetch(() => response(fixture("chat-error-500.json"), { status: 500 }));
    const err = await new ApiClient({ url: URL }, fetch).chat({ model: "m", messages }).catch((e) => e);
    expect(err.status).toBe(500);
    expect(err.message).toMatch(/^pull failed/);
  });

  it("skips comments and malformed data lines", async () => {
    const body =
      ': keep-alive\n\ndata: {oops\n\ndata: {"choices":[{"delta":{"content":"y"},"finish_reason":"stop"}]}\n\n';
    const { fetch } = fakeFetch(() => response(body));
    let content = "";
    await new ApiClient({ url: URL }, fetch).chat({
      model: "m",
      messages,
      onDelta: (d) => (content += d.content),
    });
    expect(content).toBe("y");
  });

  it("stops a live stream when the caller aborts after the headers arrived", async () => {
    const controller = new AbortController();
    let cancelled = false;
    // Never closes on its own: only the abort can end it.
    const body = new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"a"}}]}\n\n'));
      },
      cancel() {
        cancelled = true;
      },
    });
    const fetch = async () => ({ ...response(""), body });
    const deltas: string[] = [];
    const err = await new ApiClient({ url: URL }, fetch)
      .chat({
        model: "m",
        messages,
        signal: controller.signal,
        onDelta: (d) => {
          deltas.push(d.content);
          controller.abort();
        },
      })
      .catch((e) => e);
    expect(deltas).toEqual(["a"]);
    expect(isAbort(err)).toBe(true);
    expect(cancelled).toBe(true);
  });

  it("reports a connection dropped mid-stream as a network error", async () => {
    let n = 0;
    const body = new ReadableStream<Uint8Array>({
      pull(c) {
        if (n++ === 0)
          c.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"a"}}]}\n\n'));
        else c.error(new TypeError("socket closed"));
      },
    });
    const fetch = async () => ({ ...response(""), body });
    const err = await new ApiClient({ url: URL }, fetch).chat({ model: "m", messages }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.kind).toBe("network");
    expect(err.message).toMatch(/connection dropped/);
  });

  it("does not send a request whose signal is already aborted", async () => {
    const { fetch, calls } = fakeFetch(() => response("{}"));
    const controller = new AbortController();
    controller.abort();
    const err = await new ApiClient({ url: URL }, fetch)
      .chat({ model: "m", messages, signal: controller.signal })
      .catch((e) => e);
    expect(isAbort(err)).toBe(true);
    expect(calls).toHaveLength(0);
  });
});

it("sends a pasted key without its surrounding whitespace", async () => {
  const { fetch, calls } = fakeFetch(() => response('{"version":"0.1.0"}'));
  await new ApiClient({ url: URL, key: "  sk-pasted\n" }, fetch).version();
  expect(calls[0]!.init.headers?.authorization).toBe("Bearer sk-pasted");
});
