// App-wide preferences: theme, your name for the greeting, (web) what
// sends a message, and whether the wide-screen sidebar is collapsed.
// Stored under `llmman.settings` in the shape llmman's earlier web UI used,
// so its settings carry over; keys this app does not know are kept as they
// are.

import AsyncStorage from "@react-native-async-storage/async-storage";
import { useSyncExternalStore } from "react";
import { Appearance, Platform } from "react-native";

export type ThemePref = "system" | "light" | "dark";
export type SendWith = "enter" | "mod-enter";

export interface Prefs {
  theme: ThemePref;
  name: string;
  sendWith: SendWith;
  /** The wide-screen sidebar is hidden (the web UI's `sidebarCollapsed`). */
  sidebarCollapsed: boolean;
}

export const DEFAULT_PREFS: Prefs = { theme: "system", name: "", sendWith: "enter", sidebarCollapsed: false };
const KEY = "llmman.settings";

let current: Prefs = DEFAULT_PREFS;
/** Everything stored, including the web UI's own keys. */
let stored: Record<string, unknown> = {};
const listeners = new Set<() => void>();

/** The stored value, made safe: unknown or malformed fields fall back to the defaults. */
export function parsePrefs(raw: string | null): { prefs: Prefs; stored: Record<string, unknown> } {
  let obj: Record<string, unknown> = {};
  try {
    const parsed = raw ? JSON.parse(raw) : {};
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) obj = parsed;
  } catch {
    // A corrupt value is replaced on the next change.
  }
  const theme = ["system", "light", "dark"].includes(obj.theme as string)
    ? (obj.theme as ThemePref)
    : "system";
  const sendWith = obj.sendWith === "mod-enter" ? "mod-enter" : "enter";
  const name = typeof obj.name === "string" ? obj.name : "";
  const sidebarCollapsed = obj.sidebarCollapsed === true;
  return { prefs: { theme, name, sendWith, sidebarCollapsed }, stored: obj };
}

function apply() {
  // Native controls (alerts, pickers, the keyboard) follow the app's choice too.
  if (Platform.OS !== "web")
    Appearance.setColorScheme(current.theme === "system" ? "unspecified" : current.theme);
  for (const l of listeners) l();
}

/** Reads the stored preferences; call once at launch. */
export async function loadPrefs(): Promise<void> {
  const parsed = parsePrefs(await AsyncStorage.getItem(KEY).catch(() => null));
  current = parsed.prefs;
  stored = parsed.stored;
  apply();
}

export async function setPrefs(patch: Partial<Prefs>): Promise<void> {
  current = { ...current, ...patch };
  stored = { ...stored, ...patch };
  apply();
  await AsyncStorage.setItem(KEY, JSON.stringify(stored)).catch(() => {});
}

export function getPrefs(): Prefs {
  return current;
}

export function usePrefs(): Prefs {
  return useSyncExternalStore(
    (fn) => {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    () => current,
  );
}

/** "Good morning, Ricky", as the web UI greets; `name` empty: no name. */
export function greeting(name: string, hour = new Date().getHours()): string {
  const part =
    hour < 5 ? "Good evening" : hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";
  return name ? `${part}, ${name}` : part;
}
