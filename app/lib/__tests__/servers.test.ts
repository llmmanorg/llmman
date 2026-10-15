import AsyncStorage from "@react-native-async-storage/async-storage";
import * as SecureStore from "expo-secure-store";

import {
  getServerKey,
  loadServers,
  removeServer,
  renameServer,
  saveServer,
  setActiveServer,
  setServerDefaults,
} from "../servers";

jest.mock("expo-crypto", () => {
  let n = 0;
  return { randomUUID: () => `id-${++n}` };
});

const empty = { servers: [], active: null };

beforeEach(async () => {
  await AsyncStorage.clear();
  (SecureStore as unknown as { __reset(): void }).__reset();
});

it("starts empty", async () => {
  expect(await loadServers()).toEqual(empty);
});

it("saves a server, keeps its key out of AsyncStorage, and makes it active", async () => {
  const list = await saveServer(empty, { url: "http://a:1", name: "A", key: "sk-secret" });
  expect(list.active).toBe("http://a:1");
  expect(list.servers).toHaveLength(1);
  const stored = JSON.stringify(await AsyncStorage.multiGet(await AsyncStorage.getAllKeys()));
  expect(stored).not.toContain("sk-secret");
  expect(await getServerKey(list.servers[0]!)).toBe("sk-secret");
  expect(await loadServers()).toEqual(list);
});

it("updates the server with the same URL instead of adding one; no key removes the stored key", async () => {
  let list = await saveServer(empty, { url: "http://a:1", key: "k1" });
  const id = list.servers[0]!.id;
  list = await saveServer(list, { url: "http://a:1", name: "Renamed" });
  expect(list.servers).toEqual([expect.objectContaining({ id, name: "Renamed" })]);
  expect(await getServerKey({ id })).toBeUndefined();
});

it("can save without making the server active", async () => {
  let list = await saveServer(empty, { url: "http://a:1" });
  list = await saveServer(list, { url: "http://b:2" }, { activate: false });
  expect(list.active).toBe("http://a:1");
  list = await saveServer(list, { url: "http://b:2", name: "B" }, { activate: false });
  expect(list.active).toBe("http://a:1");
});

it("removes a server with its key and picks another active one", async () => {
  let list = await saveServer(empty, { url: "http://a:1", key: "ka" });
  list = await saveServer(list, { url: "http://b:2", key: "kb" });
  const a = list.servers[0]!;
  list = await setActiveServer(list, "http://a:1");
  list = await removeServer(list, "http://a:1");
  expect(list).toEqual({ servers: [expect.objectContaining({ url: "http://b:2" })], active: "http://b:2" });
  expect(await getServerKey(a)).toBeUndefined();
  await removeServer(list, "http://b:2");
  expect(await loadServers()).toEqual(empty);
});

it("ignores activating an unknown server, and renames (blank clears the name)", async () => {
  let list = await saveServer(empty, { url: "http://a:1", name: "A" });
  expect(await setActiveServer(list, "http://zzz")).toBe(list);
  list = await renameServer(list, "http://a:1", "  New  ");
  expect(list.servers[0]!.name).toBe("New");
  list = await renameServer(list, "http://a:1", " ");
  expect(list.servers[0]).not.toHaveProperty("name");
});

it("drops a corrupt list and a dangling active URL", async () => {
  await AsyncStorage.setItem("llmman.servers", "{nope");
  expect(await loadServers()).toEqual(empty);
  await AsyncStorage.setItem(
    "llmman.servers",
    JSON.stringify([{ id: "x", url: "http://a:1", addedAt: 1 }, { bad: 1 }]),
  );
  await AsyncStorage.setItem("llmman.activeServer", "http://gone");
  expect(await loadServers()).toEqual({
    servers: [{ id: "x", url: "http://a:1", addedAt: 1 }],
    active: "http://a:1",
  });
});

it("stores chat defaults per server and clears them when empty", async () => {
  let list = await saveServer(empty, { url: "http://a:1" });
  list = await setServerDefaults(list, "http://a:1", { temperature: 0.5 });
  expect((await loadServers()).servers[0]!.chatDefaults).toEqual({ temperature: 0.5 });
  list = await saveServer(list, { url: "http://a:1", name: "A" });
  expect(list.servers[0]!.chatDefaults).toEqual({ temperature: 0.5 });
  list = await setServerDefaults(list, "http://a:1", {});
  expect(list.servers[0]).not.toHaveProperty("chatDefaults");
});
