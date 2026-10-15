// The page the app is served from, on the web; nothing on a phone. See
// page.web.ts.

import type { KeySource } from "@/api/client";

/** The page's daemon as a server URL, when the page may be a llmman serve to connect to. */
export function pageUrl(): string | null {
  return null;
}

/** The key llmman's web UI stored for this page's daemon, if any. */
export function pageKey(): string | undefined {
  return undefined;
}

/** Asks for the daemon's key; null when the user cancels. */
export async function askPageKey(): Promise<string | null> {
  return null;
}

/** Stores the key llmman's web UI uses for this page's daemon; nothing on a phone. */
export function setPageKey(_key: string | undefined): void {}

/** None on a phone: a server's key is the one it was added with. */
export const pageKeySource: KeySource | undefined = undefined;

export function onPageKeyEntered(_fn: (key: string) => void): () => void {
  return () => {};
}
