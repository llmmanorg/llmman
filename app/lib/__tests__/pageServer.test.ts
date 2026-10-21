// The page's own llmman serve on the web, when storage will not keep it.

import AsyncStorage from "@react-native-async-storage/async-storage";

import { adoptPageServer } from "../pageServer";

const URL = "http://127.0.0.1:17434";

jest.mock("@/lib/page", () => ({
  pageUrl: () => "http://127.0.0.1:17434",
  pageKey: () => undefined,
  askPageKey: async () => null,
  setPageKey: () => {},
}));
jest.mock("@/api/connection", () => ({
  testConnection: async () => ({ ok: true, version: { version: "1.0.0" } }),
}));
jest.mock("@/lib/conversations", () => ({ importWebUiChats: async () => 0 }));

test("a daemon that answered is used even when the list cannot be stored", async () => {
  jest.mocked(AsyncStorage.setItem).mockRejectedValueOnce(new Error("QuotaExceededError"));
  const next = await adoptPageServer({ servers: [], active: null });
  expect(next.active).toBe(URL);
  expect(next.servers.map((s) => s.url)).toEqual([URL]);
});
