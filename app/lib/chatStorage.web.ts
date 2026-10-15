// Conversations in the browser: IndexedDB, whose quota is a share of the
// disk rather than localStorage's ~5 MB, and which stores the generated
// media as Blobs inside each conversation, as llmman's web UI does.
//
// Database `llmman-app` (not `llmman`: that one is the built-in web UI's,
// on the same origin when llmman serve hosts this page):
//   chats  conversations, keyed by id
//   meta   `index` (the summaries the conversation list reads) and flags
// On first use it takes over the chats an earlier build kept in
// localStorage, and it imports the built-in web UI's chats once.

import type { Conversation, ConversationSummary } from "@/lib/conversations";
import type { ChatStorage } from "@/lib/chatStorage";
import { fromWebUi } from "@/lib/webuiImport";

const DB_NAME = "llmman-app";
const DB_VERSION = 1;
const CHATS = "chats";
const META = "meta";

const LEGACY_INDEX = "llmman.chats";
const legacyChat = (id: string) => `llmman.chat.${id}`;
const LEGACY_DONE = "legacy-localstorage";

/** llmman's earlier web UI's database. */
export const WEBUI_DB = "llmman";
const WEBUI_STORE = "conversations";
const WEBUI_DONE = "webui-import";

function done<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function finished(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error ?? new Error("the transaction was aborted"));
  });
}

async function readIndex(db: IDBDatabase): Promise<ConversationSummary[]> {
  const v = await done(db.transaction(META).objectStore(META).get("index"));
  return Array.isArray(v) ? (v as ConversationSummary[]) : [];
}

export interface WebChatStorage extends ChatStorage {
  /** Imports llmman's web UI's chats onto `server`, once; returns how many were added. */
  importWebUiChats(server: string): Promise<number>;
}

/** The storage over `idb` (the browser's indexedDB; a fake one in tests). */
export function createWebStorage(idb: IDBFactory, legacy: Storage | null): WebChatStorage {
  let opening: Promise<IDBDatabase> | null = null;

  const open = () =>
    (opening ??= new Promise<IDBDatabase>((resolve, reject) => {
      const req = idb.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(CHATS)) db.createObjectStore(CHATS, { keyPath: "id" });
        if (!db.objectStoreNames.contains(META)) db.createObjectStore(META);
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    }).then(async (db) => {
      await adoptLegacy(db).catch(() => {});
      return db;
    })).catch((e) => {
      opening = null;
      throw e;
    });

  /** Moves what an earlier build kept in localStorage into the database, once. */
  async function adoptLegacy(db: IDBDatabase) {
    if (!legacy) return;
    if (await done(db.transaction(META).objectStore(META).get(LEGACY_DONE))) return;
    let index: ConversationSummary[] = [];
    try {
      const parsed = JSON.parse(legacy.getItem(LEGACY_INDEX) ?? "[]");
      if (Array.isArray(parsed)) index = parsed.filter((s) => s && typeof s.id === "string");
    } catch {
      // unreadable: nothing to take over
    }
    const kept: { summary: ConversationSummary; conv: unknown }[] = [];
    for (const s of index) {
      try {
        const raw = legacy.getItem(legacyChat(s.id));
        if (raw) kept.push({ summary: s, conv: JSON.parse(raw) });
      } catch {
        // one bad chat is skipped, not the rest
      }
    }
    const have = await readIndex(db);
    // Requests only from here on: awaiting inside it would let it commit.
    const tx = db.transaction([CHATS, META], "readwrite");
    const chats = tx.objectStore(CHATS);
    const meta = tx.objectStore(META);
    for (const k of kept) chats.put(k.conv);
    const added = kept.map((k) => k.summary).filter((s) => !have.some((h) => h.id === s.id));
    meta.put([...have, ...added], "index");
    meta.put(true, LEGACY_DONE);
    await finished(tx);
    // Only now that the database has them: free localStorage's small quota.
    for (const s of index) legacy.removeItem(legacyChat(s.id));
    legacy.removeItem(LEGACY_INDEX);
  }

  /** The web UI's records, without creating its database when there is none. */
  function readWebUi(): Promise<unknown[] | null> {
    return new Promise((resolve) => {
      let req: IDBOpenDBRequest;
      try {
        req = idb.open(WEBUI_DB);
      } catch {
        resolve(null);
        return;
      }
      // An upgrade means it did not exist: abort, so it is not created.
      req.onupgradeneeded = () => req.transaction?.abort();
      req.onerror = () => resolve(null);
      req.onblocked = () => resolve(null);
      req.onsuccess = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(WEBUI_STORE)) {
          db.close();
          resolve(null);
          return;
        }
        const get = db.transaction(WEBUI_STORE).objectStore(WEBUI_STORE).getAll();
        get.onsuccess = () => {
          db.close();
          resolve(get.result as unknown[]);
        };
        get.onerror = () => {
          db.close();
          resolve(null);
        };
      };
    });
  }

  const storage: WebChatStorage = {
    async getIndex() {
      const db = await open();
      return (await done(db.transaction(META).objectStore(META).get("index"))) ?? [];
    },
    async setIndex(index) {
      const db = await open();
      const tx = db.transaction(META, "readwrite");
      tx.objectStore(META).put(index, "index");
      await finished(tx);
    },
    async getChat(id) {
      const db = await open();
      return ((await done(db.transaction(CHATS).objectStore(CHATS).get(id))) as Conversation) ?? null;
    },
    async setChat(conv) {
      const db = await open();
      const tx = db.transaction(CHATS, "readwrite");
      tx.objectStore(CHATS).put(conv);
      await finished(tx);
    },
    async removeChats(ids) {
      if (!ids.length) return;
      const db = await open();
      const tx = db.transaction(CHATS, "readwrite");
      const store = tx.objectStore(CHATS);
      for (const id of ids) store.delete(id);
      await finished(tx);
    },
    async importWebUiChats(server) {
      const db = await open();
      if (await done(db.transaction(META).objectStore(META).get(WEBUI_DONE))) return 0;
      const records = await readWebUi();
      // No web UI database (yet): look again next time.
      if (!records) return 0;
      const convs = records.map((r) => fromWebUi(r, server)).filter((c): c is Conversation => !!c);
      const index = await readIndex(db);
      const tx = db.transaction([CHATS, META], "readwrite");
      const chats = tx.objectStore(CHATS);
      const meta = tx.objectStore(META);
      let added = 0;
      for (const conv of convs) {
        // Already here (imported before, or the same id): left as it is.
        if (index.some((s) => s.id === conv.id)) continue;
        chats.put(conv);
        index.push(summaryOf(conv));
        added++;
      }
      meta.put(index, "index");
      meta.put(true, WEBUI_DONE);
      await finished(tx);
      return added;
    },
  };
  return storage;
}

/** As conversations.ts summarizes (kept here to avoid an import cycle at load). */
function summaryOf(conv: Conversation): ConversationSummary {
  const { messages, ...rest } = conv;
  const last = [...messages].reverse().find((m) => m.content.trim());
  return { ...rest, preview: last ? last.content.trim().slice(0, 120) : "" };
}

function unavailable(): never {
  throw new Error("This browser has no IndexedDB, so chats cannot be saved.");
}

const browser =
  typeof indexedDB !== "undefined"
    ? createWebStorage(indexedDB, typeof localStorage !== "undefined" ? localStorage : null)
    : null;

export const storage: ChatStorage = browser ?? {
  getIndex: async () => [],
  setIndex: async () => unavailable(),
  getChat: async () => null,
  setChat: async () => unavailable(),
  removeChats: async () => {},
};

export async function importWebUiChats(server: string): Promise<number> {
  return browser ? browser.importWebUiChats(server) : 0;
}
