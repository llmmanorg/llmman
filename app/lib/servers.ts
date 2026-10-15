// The servers the phone knows. A server is identified by its URL; the list
// (URL, name, when it was added) is in AsyncStorage, and each server's API
// key is in the platform secure store under the server's random id, never
// in AsyncStorage.

import AsyncStorage from "@react-native-async-storage/async-storage";
import * as SecureStore from "@/lib/secretStore";

import type { ChatSettings } from "@/lib/chatSettings";
import { newId } from "@/lib/id";

export interface Server {
  /** Random, stable; names the secure-store entry for the key. */
  id: string;
  /** Normalized base URL (see api/serverUrl.ts); the server's identity. */
  url: string;
  /** Display name, when the user gave one. */
  name?: string;
  /** What new chats on this server start with. */
  chatDefaults?: ChatSettings;
  addedAt: number;
}

export interface ServerList {
  servers: Server[];
  /** URL of the server the app talks to, or null before onboarding. */
  active: string | null;
}

const LIST_KEY = "llmman.servers";
const ACTIVE_KEY = "llmman.activeServer";
// SecureStore keys allow only [A-Za-z0-9._-].
const secretKey = (id: string) => `llmman.apikey.${id}`;

export async function loadServers(): Promise<ServerList> {
  const [rawList, active] = await Promise.all([
    AsyncStorage.getItem(LIST_KEY),
    AsyncStorage.getItem(ACTIVE_KEY),
  ]);
  let servers: Server[] = [];
  try {
    const parsed = rawList ? JSON.parse(rawList) : [];
    if (Array.isArray(parsed)) servers = parsed.filter(isServer);
  } catch {
    // A corrupt list is dropped rather than crashing the app on launch.
  }
  const known = servers.some((s) => s.url === active);
  return { servers, active: known ? active : (servers[0]?.url ?? null) };
}

function isServer(v: unknown): v is Server {
  const s = v as Server;
  return !!s && typeof s.id === "string" && typeof s.url === "string" && typeof s.addedAt === "number";
}

async function saveList(list: ServerList): Promise<void> {
  await AsyncStorage.setItem(LIST_KEY, JSON.stringify(list.servers));
  if (list.active) await AsyncStorage.setItem(ACTIVE_KEY, list.active);
  else await AsyncStorage.removeItem(ACTIVE_KEY);
}

export async function getServerKey(server: Pick<Server, "id">): Promise<string | undefined> {
  return (await SecureStore.getItemAsync(secretKey(server.id)))?.trim() || undefined;
}

/**
 * Adds a server, or updates the one with the same URL (its name, and its
 * key: an absent key removes the stored one). The saved server becomes
 * active unless `activate` is false.
 */
export async function saveServer(
  list: ServerList,
  entry: { url: string; name?: string; key?: string },
  { activate = true }: { activate?: boolean } = {},
): Promise<ServerList> {
  const existing = list.servers.find((s) => s.url === entry.url);
  const server: Server = existing
    ? { ...existing, ...(entry.name ? { name: entry.name } : {}) }
    : { id: newId(), url: entry.url, addedAt: Date.now(), ...(entry.name ? { name: entry.name } : {}) };

  const key = entry.key?.trim();
  if (key) {
    await SecureStore.setItemAsync(secretKey(server.id), key);
  } else {
    await SecureStore.deleteItemAsync(secretKey(server.id));
  }
  const servers = existing
    ? list.servers.map((s) => (s.url === entry.url ? server : s))
    : [...list.servers, server];
  const next = { servers, active: activate || !list.active ? server.url : list.active };
  await saveList(next);
  return next;
}

export async function removeServer(list: ServerList, url: string): Promise<ServerList> {
  const server = list.servers.find((s) => s.url === url);
  if (server) await SecureStore.deleteItemAsync(secretKey(server.id));
  const servers = list.servers.filter((s) => s.url !== url);
  const active = list.active === url ? (servers[0]?.url ?? null) : list.active;
  const next = { servers, active };
  await saveList(next);
  return next;
}

export async function setActiveServer(list: ServerList, url: string): Promise<ServerList> {
  if (!list.servers.some((s) => s.url === url)) return list;
  const next = { ...list, active: url };
  await saveList(next);
  return next;
}

export async function renameServer(list: ServerList, url: string, name: string): Promise<ServerList> {
  const trimmed = name.trim();
  const servers = list.servers.map((s) => {
    if (s.url !== url) return s;
    const { name: _old, ...rest } = s;
    return trimmed ? { ...rest, name: trimmed } : rest;
  });
  const next = { ...list, servers };
  await saveList(next);
  return next;
}

export async function setServerDefaults(
  list: ServerList,
  url: string,
  defaults: ChatSettings,
): Promise<ServerList> {
  const servers = list.servers.map((s) => {
    if (s.url !== url) return s;
    const { chatDefaults: _old, ...rest } = s;
    return Object.keys(defaults).length ? { ...rest, chatDefaults: defaults } : rest;
  });
  const next = { ...list, servers };
  await saveList(next);
  return next;
}
