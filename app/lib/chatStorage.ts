// Where conversations are kept. On a phone: AsyncStorage, an index of
// summaries under one key and each conversation's JSON under its own; the
// media a conversation generated is in files (lib/mediaFiles.ts), so only
// their paths are in the JSON. The web keeps them in IndexedDB instead
// (chatStorage.web.ts), with the same interface.

import AsyncStorage from "@react-native-async-storage/async-storage";

import type { Conversation, ConversationSummary } from "@/lib/conversations";

export interface ChatStorage {
  /** The summaries, as stored (the caller checks their shape). */
  getIndex(): Promise<unknown>;
  setIndex(index: ConversationSummary[]): Promise<void>;
  getChat(id: string): Promise<Conversation | null>;
  setChat(conv: Conversation): Promise<void>;
  removeChats(ids: string[]): Promise<void>;
}

const INDEX_KEY = "llmman.chats";
const chatKey = (id: string) => `llmman.chat.${id}`;

export const storage: ChatStorage = {
  async getIndex() {
    const raw = await AsyncStorage.getItem(INDEX_KEY);
    return raw ? JSON.parse(raw) : [];
  },
  async setIndex(index) {
    await AsyncStorage.setItem(INDEX_KEY, JSON.stringify(index));
  },
  async getChat(id) {
    const raw = await AsyncStorage.getItem(chatKey(id));
    return raw ? (JSON.parse(raw) as Conversation) : null;
  },
  async setChat(conv) {
    await AsyncStorage.setItem(chatKey(conv.id), JSON.stringify(conv));
  },
  async removeChats(ids) {
    if (ids.length) await AsyncStorage.multiRemove(ids.map(chatKey));
  },
};

/** The web imports llmman's built-in web UI's chats (chatStorage.web.ts); a phone has none. */
export async function importWebUiChats(_server: string): Promise<number> {
  return 0;
}
