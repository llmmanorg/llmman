// On the web the app is usually served by llmman serve itself, so it talks
// to the page's own daemon, with the key stored under
// `llmman.apiKey:<base path>`, where llmman's earlier web UI kept it too
// (see pageBase.ts). llmman's 401 on any request asks for it (one prompt,
// shared) and the request is retried once.

import type { KeySource } from "@/api/client";
import { showDialog } from "@/lib/dialog";
import { createKeyPrompt } from "@/lib/keyPrompt";
import { basePath, keyStorageName, pageServerUrl, readScopedKey } from "@/lib/pageBase";

// Fixed at load: the app's own navigation changes the page's address.
const BASE_URI = typeof document === "undefined" ? "" : document.baseURI;
const BASE = basePath(BASE_URI);
const KEY_STORAGE = keyStorageName(BASE);

const listeners = new Set<(key: string) => void>();

/** The key last set when storage would not take it (blocked, or full): it holds for this page. */
let unsaved: { key: string | undefined } | null = null;

/** The page's daemon as a server URL (origin and base path), when the page is on http(s). */
export function pageUrl(): string | null {
  return BASE_URI ? pageServerUrl(BASE_URI) : null;
}

export function pageKey(): string | undefined {
  if (unsaved) return unsaved.key;
  try {
    return readScopedKey(localStorage, BASE);
  } catch {
    return undefined;
  }
}

function storeKey(key: string | undefined): void {
  try {
    if (key) localStorage.setItem(KEY_STORAGE, key);
    else localStorage.removeItem(KEY_STORAGE);
    unsaved = null;
  } catch {
    // Storage blocked or full: the key lasts as long as the page.
    unsaved = { key };
  }
}

const prompt = createKeyPrompt(
  async () => {
    // The app's own dialog: window.prompt is a no-op in some web views.
    const { button, text } = await showDialog({
      title: "Enter your API key",
      message: "This llmman serve asks for an API key.",
      input: { label: "API key", placeholder: "API key", secure: true },
      hint: "It's one of the keys in LLMMAN_API_KEYS on the computer running llmman serve.",
      brand: true,
      buttons: [{ text: "Cancel", style: "cancel" }, { text: "Continue" }],
    });
    return button === 1 ? text : null;
  },
  (key) => {
    storeKey(key);
    for (const l of listeners) l(key);
  },
);

/** Asks for the daemon's key (the one shared prompt); null when the user cancels. */
export async function askPageKey(): Promise<string | null> {
  return (await prompt.ask()) ? (pageKey() ?? null) : null;
}

/** Stores the key for the page's daemon (Settings); a declined prompt may show again. */
export function setPageKey(key: string | undefined): void {
  storeKey(key);
  prompt.reset();
}

/** The page's daemon key as a client's key source: asked for on llmman's 401. */
export const pageKeySource: KeySource | undefined = {
  get: pageKey,
  ask: () => prompt.ask(),
};

/** Calls `fn` with a key entered at the prompt. */
export function onPageKeyEntered(fn: (key: string) => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
