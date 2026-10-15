/** @jest-environment node */

// The conversation API on the web's storage: the same calls as on a phone,
// over IndexedDB, and the one-time import of llmman's web UI chats.

import "fake-indexeddb/auto";

import {
  allConversations,
  deleteAllConversations,
  deleteConversation,
  importWebUiChats,
  listConversations,
  loadConversation,
  newConversation,
  onConversationsChanged,
  saveConversation,
} from "../conversations";
import { fromWebUi } from "../webuiImport";

// What Metro resolves on the web.
jest.mock("@/lib/chatStorage", () => require("@/lib/chatStorage.web"));

jest.mock("expo-crypto", () => {
  let n = 0;
  return { randomUUID: () => `w-${++n}` };
});

const SERVER = "http://127.0.0.1:17434";

it("saves, lists, loads and deletes as on a phone", async () => {
  const a = newConversation(SERVER, "m");
  a.messages.push({ role: "user", content: "first", at: 1 });
  a.updatedAt = 10;
  await saveConversation(a);
  const b = { ...newConversation(SERVER, "m"), updatedAt: 20 };
  b.messages.push({ role: "user", content: "second", at: 2 });
  await saveConversation(b);

  expect((await listConversations(SERVER)).map((s) => s.id)).toEqual([b.id, a.id]);
  expect(await loadConversation(a.id)).toEqual(a);
  expect((await allConversations()).map((c) => c.id)).toEqual([b.id, a.id]);

  await deleteConversation(a.id);
  expect(await loadConversation(a.id)).toBeNull();
  expect(await deleteAllConversations()).toEqual([b.id]);
  expect(await listConversations(SERVER)).toEqual([]);
});

it("imports the web UI's chats through the list, telling the drawer", async () => {
  const db = await new Promise<IDBDatabase>((resolve) => {
    const req = indexedDB.open("llmman", 1);
    req.onupgradeneeded = () => req.result.createObjectStore("conversations", { keyPath: "id" });
    req.onsuccess = () => resolve(req.result);
  });
  const tx = db.transaction("conversations", "readwrite");
  tx.objectStore("conversations").put({
    id: "web-1",
    title: "From the web UI",
    model: "m",
    createdAt: 5,
    updatedAt: 6,
    systemPrompt: "Be brief.",
    temperature: 0.3,
    maxTokens: null,
    messages: [{ role: "user", content: "hi", at: 5 }],
  });
  await new Promise((r) => (tx.oncomplete = r));
  db.close();

  const changed = jest.fn();
  const off = onConversationsChanged(changed);
  expect(await importWebUiChats(SERVER)).toBe(1);
  expect(await importWebUiChats(SERVER)).toBe(0);
  off();
  expect(changed).toHaveBeenCalledTimes(1);
  expect((await listConversations(SERVER)).map((s) => s.title)).toEqual(["From the web UI"]);
  expect((await loadConversation("web-1"))!.settings).toEqual({
    systemPrompt: "Be brief.",
    temperature: 0.3,
  });
});

describe("fromWebUi", () => {
  it("keeps errors, stops and reasoning, and skips what it cannot read", () => {
    const c = fromWebUi(
      {
        id: "x",
        title: "t",
        model: "m",
        createdAt: 1,
        updatedAt: 2,
        messages: [
          { role: "user", content: "q", at: 1 },
          { role: "assistant", content: "", reasoning: "hmm", at: 2, stopped: true },
          { role: "assistant", content: "", at: 3, error: "boom" },
          { role: "system", content: "?", at: 4 },
          null,
        ],
      },
      SERVER,
    )!;
    expect(c.server).toBe(SERVER);
    expect(c.messages).toEqual([
      { role: "user", content: "q", at: 1 },
      { role: "assistant", content: "", reasoning: "hmm", at: 2, stopped: true },
      { role: "assistant", content: "", at: 3, error: "boom" },
    ]);
    expect(fromWebUi({ id: "empty", messages: [] }, SERVER)).toBeNull();
    expect(fromWebUi("nope", SERVER)).toBeNull();
  });
});
