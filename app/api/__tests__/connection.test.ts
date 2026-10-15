/** @jest-environment node */

import { testConnection } from "../connection";
import { fakeFetch, llmman401, response } from "@/test/fakeFetch";

const URL = "http://10.0.0.5:17434";

describe("testConnection", () => {
  it("returns the version on success", async () => {
    const { fetch, calls } = fakeFetch(() => response('{"version":"0.1.0","pid":1}'));
    expect(await testConnection(URL, "k", { fetchImpl: fetch })).toEqual({
      ok: true,
      version: { version: "0.1.0", pid: 1 },
    });
    expect(calls[0]!.url).toBe(`${URL}/api/version`);
    expect(calls[0]!.init.headers).toEqual({ authorization: "Bearer k" });
  });

  it("asks for a key on a 401 without one", async () => {
    const { fetch } = fakeFetch(() => llmman401());
    const r = await testConnection(URL, undefined, { fetchImpl: fetch });
    expect(r).toMatchObject({
      ok: false,
      problem: "unauthorized",
      message: expect.stringMatching(/needs an API key/),
    });
  });

  it("on the web, a realm-less 401 counts as the key only from another origin", async () => {
    const { Platform } = jest.requireActual<typeof import("react-native")>("react-native");
    const g = globalThis as { window?: unknown };
    const was = { os: Platform.OS, window: g.window };
    Platform.OS = "web";
    g.window = { location: { origin: "http://10.0.0.5:17452" } };
    try {
      const hidden = fakeFetch(() => response('{"error":"needs a key"}', { status: 401 }));
      expect(await testConnection("http://10.0.0.5:17434", "k", { fetchImpl: hidden.fetch })).toMatchObject({
        ok: false,
        problem: "unauthorized",
      });
      // The page's own origin: the realm would be readable, so its absence means not llmman's key check.
      const same = fakeFetch(() => response('{"error":"proxy login"}', { status: 401 }));
      expect(await testConnection("http://10.0.0.5:17452", "k", { fetchImpl: same.fetch })).toMatchObject({
        ok: false,
        problem: "server-error",
      });
    } finally {
      Platform.OS = was.os;
      g.window = was.window;
    }
  });

  it("says the key is wrong on a 401 with one", async () => {
    const { fetch } = fakeFetch(() => llmman401());
    const r = await testConnection(URL, "bad", { fetchImpl: fetch });
    expect(r).toMatchObject({
      ok: false,
      problem: "unauthorized",
      message: expect.stringMatching(/rejected this API key/),
    });
  });

  it("explains an unreachable address", async () => {
    const fetch = async () => {
      throw new TypeError("Could not connect to the server.");
    };
    const r = await testConnection(URL, undefined, { fetchImpl: fetch });
    expect(r).toMatchObject({
      ok: false,
      problem: "unreachable",
      message: expect.stringMatching(/LLMMAN_HOST=0\.0\.0\.0/),
    });
  });

  it("does not call a 401 from something other than llmman a key problem", async () => {
    const { fetch } = fakeFetch(() => response("Unauthorized", { status: 401, contentType: "text/plain" }));
    const r = await testConnection(URL, "k", { fetchImpl: fetch });
    expect(r).toMatchObject({ ok: false, problem: "server-error" });
  });

  it("suggests port 17434 when an explicit http:// address without a port is unreachable", async () => {
    const fetch = async () => {
      throw new TypeError("Could not connect to the server.");
    };
    const r = await testConnection("http://10.0.0.77", undefined, { fetchImpl: fetch });
    expect(r.ok === false && r.message).toMatch(/did you mean http:\/\/10\.0\.0\.77:17434\?/);
    const r2 = await testConnection("http://10.0.0.77:9", undefined, { fetchImpl: fetch });
    expect(r2.ok === false && r2.message).not.toMatch(/did you mean/);
  });

  it("times out", async () => {
    const fetch = (_u: string, init?: { signal?: AbortSignal }) =>
      new Promise<never>((_, reject) =>
        init?.signal?.addEventListener("abort", () => reject(new Error("x"))),
      );
    const r = await testConnection(URL, undefined, { fetchImpl: fetch, timeoutMs: 10 });
    expect(r).toMatchObject({ ok: false, problem: "timeout" });
  });

  it.each([
    ["a 404", () => response("404 page not found", { status: 404 })],
    ["an HTML page", () => response("<html></html>", { contentType: "text/html" })],
  ])("calls %s not llmman", async (_label, reply) => {
    const { fetch } = fakeFetch(reply);
    const r = await testConnection(URL, undefined, { fetchImpl: fetch });
    expect(r).toMatchObject({ ok: false, problem: "not-llmman" });
  });

  it("passes other errors through", async () => {
    const { fetch } = fakeFetch(() => response('{"error":"boom"}', { status: 500 }));
    const r = await testConnection(URL, undefined, { fetchImpl: fetch });
    expect(r).toMatchObject({
      ok: false,
      problem: "server-error",
      message: expect.stringMatching(/500.*boom/),
    });
  });

  it("rethrows the caller's abort", async () => {
    const controller = new AbortController();
    controller.abort();
    const { fetch } = fakeFetch(() => response("{}"));
    await expect(
      testConnection(URL, undefined, { fetchImpl: fetch, signal: controller.signal }),
    ).rejects.toMatchObject({
      name: "AbortError",
    });
  });
});
