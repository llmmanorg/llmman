import AsyncStorage from "@react-native-async-storage/async-storage";

import {
  deleteConversation,
  deleteServerConversations,
  listConversations,
  loadConversation,
  newConversation,
  saveConversation,
  titleFrom,
} from "../conversations";

jest.mock("expo-crypto", () => {
  let n = 0;
  return { randomUUID: () => `c-${++n}` };
});

beforeEach(() => AsyncStorage.clear());

describe("titleFrom", () => {
  it.each([
    ["Hello there", "Hello there"],
    ["  first line\nsecond", "first line"],
    ["Write **bold** and `code`", "Write bold and code"],
    ["# Heading", "Heading"],
    ["- item one", "item one"],
    ["word ".repeat(20), "word word word word word word word word word…"],
  ])("%j → %j", (input, title) => {
    expect(titleFrom(input)).toBe(title);
  });
});

it("keeps conversations per server, newest first, and does not save empty ones", async () => {
  const a = newConversation("http://a:1", "m");
  await saveConversation(a);
  expect(await listConversations("http://a:1")).toEqual([]);

  a.messages.push({ role: "user", content: "hi", at: 1 });
  a.updatedAt = 10;
  await saveConversation(a);
  const b = { ...newConversation("http://a:1", "m"), updatedAt: 20 };
  b.messages.push({ role: "user", content: "second", at: 2 });
  await saveConversation(b);
  const other = newConversation("http://b:2", "m");
  other.messages.push({ role: "user", content: "x", at: 3 });
  await saveConversation(other);

  const list = await listConversations("http://a:1");
  expect(list.map((s) => s.id)).toEqual([b.id, a.id]);
  expect(list[0]).toMatchObject({ preview: "second" });
  expect(list[0]).not.toHaveProperty("messages");
  expect(await loadConversation(a.id)).toEqual(a);
});

it("saves concurrently without losing an index entry", async () => {
  const convs = Array.from({ length: 5 }, (_, i) => {
    const c = newConversation("http://a:1", "m");
    c.messages.push({ role: "user", content: String(i), at: i });
    return c;
  });
  await Promise.all(convs.map(saveConversation));
  expect(await listConversations("http://a:1")).toHaveLength(5);
});

it("deletes a conversation and its messages", async () => {
  const a = newConversation("http://a:1", "m");
  a.messages.push({ role: "user", content: "hi", at: 1 });
  await saveConversation(a);
  await deleteConversation(a.id);
  expect(await listConversations("http://a:1")).toEqual([]);
  expect(await loadConversation(a.id)).toBeNull();
});

it("starts a chat from the server's defaults, as a copy", () => {
  const defaults = { systemPrompt: "x", maxTokens: 5 };
  const c = newConversation("http://a:1", "m", defaults);
  expect(c.settings).toEqual(defaults);
  expect(c.settings).not.toBe(defaults);
  expect(newConversation("http://a:1", "m", {})).not.toHaveProperty("settings");
});

it("deletes only the removed server's conversations", async () => {
  const mk = async (server: string) => {
    const c = newConversation(server, "m");
    c.messages.push({ role: "user", content: "hi", at: 1 });
    await saveConversation(c);
    return c;
  };
  const a1 = await mk("http://a:1");
  const a2 = await mk("http://a:1");
  const b = await mk("http://b:2");
  expect((await deleteServerConversations("http://a:1")).sort()).toEqual([a1.id, a2.id].sort());
  expect(await listConversations("http://a:1")).toEqual([]);
  expect(await loadConversation(a1.id)).toBeNull();
  expect((await listConversations("http://b:2")).map((s) => s.id)).toEqual([b.id]);
});

describe("export and delete all", () => {
  it("exports every server's chats newest first, leaving media out, then deletes them all", async () => {
    const {
      allConversations,
      deleteAllConversations,
      exportJson,
      listConversations,
      newConversation,
      saveConversation,
    } = jest.requireActual<typeof import("../conversations")>("../conversations");
    const a = {
      ...newConversation("http://a", "m"),
      updatedAt: 1,
      messages: [{ role: "user" as const, content: "hi", at: 1 }],
    };
    const b = {
      ...newConversation("http://b", "m"),
      updatedAt: 2,
      messages: [{ role: "user" as const, content: "look", at: 2, images: ["data:image/jpeg;base64,AAA"] }],
    };
    await saveConversation(a);
    await saveConversation(b);
    const all = await allConversations();
    expect(all.map((c) => c.server)).toEqual(["http://b", "http://a"]);
    const json = exportJson(all);
    expect(json).not.toContain("base64");
    expect(JSON.parse(json)[0].messages[0].images).toEqual({ omitted: true });

    await deleteAllConversations();
    expect(await allConversations()).toEqual([]);
    expect(await listConversations("http://a")).toEqual([]);
  });
});
