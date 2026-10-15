import { basePath, keyStorageName, LEGACY_KEY_STORAGE, pageServerUrl, readScopedKey } from "@/lib/pageBase";

test("the base path is the page's directory, without the app's own routes", () => {
  expect(basePath("http://127.0.0.1:17434/")).toBe("/");
  expect(basePath("http://127.0.0.1:17434/#/models")).toBe("/");
  expect(basePath("http://127.0.0.1:17434/models")).toBe("/");
  expect(basePath("http://127.0.0.1:17434/models/variants")).toBe("/");
  expect(basePath("http://127.0.0.1:17434/c/abc123")).toBe("/");
  expect(basePath("https://gw.example/llmman/")).toBe("/llmman/");
  expect(basePath("https://gw.example/llmman/index.html")).toBe("/llmman/");
  expect(basePath("https://gw.example/team/a/settings")).toBe("/team/a/");
  expect(basePath("not a url")).toBe("/");
});

test("the page's server URL keeps a gateway's path prefix", () => {
  expect(pageServerUrl("http://127.0.0.1:17434/")).toBe("http://127.0.0.1:17434");
  expect(pageServerUrl("https://gw.example/llmman/")).toBe("https://gw.example/llmman");
  expect(pageServerUrl("file:///Users/x/index.html")).toBeNull();
});

test("keys are scoped by base path, as the earlier web UI named them", () => {
  expect(keyStorageName("/")).toBe(LEGACY_KEY_STORAGE);
  expect(keyStorageName("/a/")).toBe("llmman.apiKey:/a/");
});

function memory(init: Record<string, string>) {
  const m = new Map(Object.entries(init));
  return {
    getItem: (k: string) => m.get(k) ?? null,
    setItem: (k: string, v: string) => void m.set(k, v),
    m,
  };
}

test("the key at / still works, and is taken over by a prefixed base", () => {
  expect(readScopedKey(memory({ "llmman.apiKey:/": "root" }), "/")).toBe("root");
  const s = memory({ "llmman.apiKey:/": "root" });
  expect(readScopedKey(s, "/a/")).toBe("root");
  expect(s.m.get("llmman.apiKey:/a/")).toBe("root");
  // Its own key wins once there is one.
  expect(readScopedKey(memory({ "llmman.apiKey:/": "root", "llmman.apiKey:/a/": "own" }), "/a/")).toBe("own");
  expect(readScopedKey(memory({}), "/a/")).toBeUndefined();
});
