import { basePath, keyStorageName, LEGACY_KEY_STORAGE, pageServerUrl, readScopedKey } from "@/lib/pageBase";

test("the base path is the directory of the page's <base>", () => {
  expect(basePath("http://127.0.0.1:17434/")).toBe("/");
  expect(basePath("http://127.0.0.1:17434/#/models")).toBe("/");
  expect(basePath("https://gw.example/llmman/")).toBe("/llmman/");
  expect(basePath("https://gw.example/llmman/index.html")).toBe("/llmman/");
  // A prefix named like one of the app's routes is still the prefix.
  expect(basePath("https://gw.example/servers/x/")).toBe("/servers/x/");
  expect(basePath("https://gw.example/team/models/")).toBe("/team/models/");
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

test("each base path reads its own key only", () => {
  expect(readScopedKey(memory({ "llmman.apiKey:/": "root" }), "/")).toBe("root");
  // The key at / is another daemon's: a prefixed base never gets it.
  const s = memory({ "llmman.apiKey:/": "root" });
  expect(readScopedKey(s, "/a/")).toBeUndefined();
  expect(s.m.has("llmman.apiKey:/a/")).toBe(false);
  expect(readScopedKey(memory({ "llmman.apiKey:/": "root", "llmman.apiKey:/a/": "own" }), "/a/")).toBe("own");
});
