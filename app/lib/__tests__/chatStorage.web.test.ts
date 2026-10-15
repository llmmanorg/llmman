/** @jest-environment node */

// The web's conversation storage on IndexedDB (a faithful fake): the same
// interface as AsyncStorage's, Blobs kept as they are, the chats an earlier
// build left in localStorage taken over, and llmman's web UI's imported once.

import { IDBFactory } from "fake-indexeddb";

import { createWebStorage, WEBUI_DB } from "../chatStorage.web";
import type { Conversation } from "../conversations";

const SERVER = "http://127.0.0.1:17434";

// fake-indexeddb clones with structuredClone, which in Jest's sandbox comes
// from Node's own realm and hands back Blobs that are not the sandbox's
// Blob. A browser has one realm; Blobs are immutable, so keeping them as
// they are is what a browser's clone amounts to here.
const nativeClone = globalThis.structuredClone;
function cloneKeepingBlobs(v: unknown): unknown {
  if (v instanceof Blob) return v;
  if (Array.isArray(v)) return v.map(cloneKeepingBlobs);
  if (v && typeof v === "object") {
    return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, cloneKeepingBlobs(x)]));
  }
  return nativeClone(v);
}
beforeAll(() => {
  globalThis.structuredClone = cloneKeepingBlobs as typeof structuredClone;
});
afterAll(() => {
  globalThis.structuredClone = nativeClone;
});

/** localStorage, in memory. */
function memoryStorage(entries: Record<string, string> = {}): Storage {
  const m = new Map(Object.entries(entries));
  return {
    get length() {
      return m.size;
    },
    key: (i) => [...m.keys()][i] ?? null,
    getItem: (k) => m.get(k) ?? null,
    setItem: (k, v) => void m.set(k, String(v)),
    removeItem: (k) => void m.delete(k),
    clear: () => m.clear(),
  };
}

function conv(id: string, extra: Partial<Conversation> = {}): Conversation {
  return {
    id,
    server: SERVER,
    title: id,
    model: "m",
    createdAt: 1,
    updatedAt: 2,
    messages: [{ role: "user", content: `hello ${id}`, at: 1 }],
    ...extra,
  };
}

/** The earlier built-in web UI's database, as it created it. */
async function seedWebUi(idb: IDBFactory, records: unknown[]) {
  const db = await new Promise<IDBDatabase>((resolve, reject) => {
    const req = idb.open(WEBUI_DB, 1);
    req.onupgradeneeded = () => {
      const store = req.result.createObjectStore("conversations", { keyPath: "id" });
      store.createIndex("updatedAt", "updatedAt");
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  const tx = db.transaction("conversations", "readwrite");
  for (const r of records) tx.objectStore("conversations").put(r);
  await new Promise((resolve) => (tx.oncomplete = resolve));
  db.close();
}

async function webUiCount(idb: IDBFactory): Promise<number> {
  const db = await new Promise<IDBDatabase>((resolve) => {
    const req = idb.open(WEBUI_DB);
    req.onsuccess = () => resolve(req.result);
  });
  const n = await new Promise<number>((resolve) => {
    const req = db.transaction("conversations").objectStore("conversations").count();
    req.onsuccess = () => resolve(req.result);
  });
  db.close();
  return n;
}

const webUiRecord = (id: string, extra: Record<string, unknown> = {}) => ({
  id,
  title: `Web ${id}`,
  model: "docker.io/ai/smollm2",
  createdAt: 100,
  updatedAt: 200,
  systemPrompt: "",
  temperature: null,
  maxTokens: null,
  messages: [
    { role: "user", content: "hi", at: 100 },
    {
      role: "assistant",
      content: "Hello!",
      reasoning: "",
      model: "docker.io/ai/smollm2",
      at: 101,
      finishReason: "stop",
    },
  ],
  ...extra,
});

describe("the adapter", () => {
  it("stores, reads and removes chats and the index", async () => {
    const s = createWebStorage(new IDBFactory(), null);
    expect(await s.getIndex()).toEqual([]);
    expect(await s.getChat("a")).toBeNull();
    await s.setChat(conv("a"));
    await s.setChat(conv("b"));
    await s.setIndex([{ ...conv("a"), preview: "x" } as never]);
    expect(await s.getChat("a")).toEqual(conv("a"));
    expect(await s.getIndex()).toEqual([expect.objectContaining({ id: "a" })]);
    await s.removeChats(["a", "missing"]);
    expect(await s.getChat("a")).toBeNull();
    expect(await s.getChat("b")).toEqual(conv("b"));
  });

  it("keeps a generated result's Blob as it is", async () => {
    const s = createWebStorage(new IDBFactory(), null);
    const blob = new Blob([new Uint8Array([137, 80, 78, 71])], { type: "image/png" });
    await s.setChat(
      conv("m", {
        messages: [
          { role: "user", content: "a fox", generate: true, at: 1 },
          {
            role: "assistant",
            content: "",
            at: 2,
            media: { kind: "image", mime: "image/png", bytes: 4, blob },
          },
        ],
      }),
    );
    const back = (await s.getChat("m"))!.messages[1]!.media!.blob!;
    expect(back).toBeInstanceOf(Blob);
    expect(back.type).toBe("image/png");
    expect([...new Uint8Array(await back.arrayBuffer())]).toEqual([137, 80, 78, 71]);
  });

  it("takes over the chats an earlier build kept in localStorage, once, and frees them", async () => {
    const idb = new IDBFactory();
    const legacy = memoryStorage({
      "llmman.chats": JSON.stringify([{ ...conv("old"), preview: "hello old" }]),
      "llmman.chat.old": JSON.stringify(conv("old")),
      "llmman.servers": "[]",
    });
    const s = createWebStorage(idb, legacy);
    expect(await s.getIndex()).toEqual([expect.objectContaining({ id: "old", preview: "hello old" })]);
    expect(await s.getChat("old")).toEqual(conv("old"));
    expect(legacy.getItem("llmman.chats")).toBeNull();
    expect(legacy.getItem("llmman.chat.old")).toBeNull();
    // Other keys are not its business.
    expect(legacy.getItem("llmman.servers")).toBe("[]");

    // A later run does not take anything over again.
    legacy.setItem("llmman.chats", JSON.stringify([{ ...conv("late"), preview: "" }]));
    legacy.setItem("llmman.chat.late", JSON.stringify(conv("late")));
    const again = createWebStorage(idb, legacy);
    expect((await again.getIndex()) as unknown[]).toHaveLength(1);
    expect(await again.getChat("late")).toBeNull();
  });
});

describe("importing the built-in web UI's chats", () => {
  it("imports them once onto the server, leaving the web UI's database as it was", async () => {
    const idb = new IDBFactory();
    const png = new Blob([new Uint8Array([1, 2, 3])], { type: "image/png" });
    await seedWebUi(idb, [
      webUiRecord("w1"),
      webUiRecord("w2", {
        model: "sd-turbo",
        media: {
          kind: "image",
          width: 512,
          height: 512,
          seconds: null,
          steps: 4,
          seed: null,
          cfgScale: null,
          negativePrompt: "",
        },
        messages: [
          { role: "user", content: "a fox", at: 300, generate: true },
          {
            role: "assistant",
            content: "",
            model: "sd-turbo",
            at: 301,
            prompt: "a fox",
            request: {
              kind: "image",
              width: 512,
              height: 512,
              steps: 4,
              seed: null,
              cfgScale: null,
              negativePrompt: "",
            },
            media: { kind: "image", blob: png, revisedPrompt: "a red fox" },
          },
        ],
      }),
      { id: "broken" },
    ]);
    const s = createWebStorage(idb, null);
    expect(await s.importWebUiChats(SERVER)).toBe(2);

    const index = (await s.getIndex()) as { id: string; server: string; preview: string }[];
    expect(index.map((i) => [i.id, i.server, i.preview])).toEqual([
      ["w1", SERVER, "Hello!"],
      ["w2", SERVER, "a fox"],
    ]);
    const w2 = (await s.getChat("w2"))!;
    expect(w2).toMatchObject({
      title: "Web w2",
      model: "sd-turbo",
      media: { kind: "image", width: 512, height: 512, steps: 4 },
    });
    expect(w2.messages[0]).toEqual({ role: "user", content: "a fox", at: 300, generate: true });
    expect(w2.messages[1]).toMatchObject({
      prompt: "a fox",
      request: { kind: "image", width: 512, height: 512, steps: 4 },
      media: { kind: "image", mime: "image/png", bytes: 3, revisedPrompt: "a red fox" },
    });
    expect(w2.messages[1]!.media!.blob).toBeInstanceOf(Blob);

    // The originals stay, and nothing is imported twice.
    expect(await webUiCount(idb)).toBe(3);
    expect(await s.importWebUiChats(SERVER)).toBe(0);
    expect(await createWebStorage(idb, null).importWebUiChats(SERVER)).toBe(0);
    expect((await s.getIndex()) as unknown[]).toHaveLength(2);
  });

  it("leaves a chat that is already here as it is", async () => {
    const idb = new IDBFactory();
    await seedWebUi(idb, [webUiRecord("same"), webUiRecord("new")]);
    const s = createWebStorage(idb, null);
    const mine = conv("same", { title: "Mine" });
    await s.setChat(mine);
    await s.setIndex([{ ...mine, preview: "" } as never]);
    expect(await s.importWebUiChats(SERVER)).toBe(1);
    expect((await s.getChat("same"))!.title).toBe("Mine");
  });

  it("does not create the web UI's database when there is none, and looks again later", async () => {
    const idb = new IDBFactory();
    const s = createWebStorage(idb, null);
    expect(await s.importWebUiChats(SERVER)).toBe(0);
    expect((await idb.databases()).map((d) => d.name)).not.toContain(WEBUI_DB);

    await seedWebUi(idb, [webUiRecord("later")]);
    expect(await s.importWebUiChats(SERVER)).toBe(1);
  });
});
