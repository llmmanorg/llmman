// Where the web app's own llmman serve is, from the page's address. llmman's
// web UI scopes its key by the page's base path (`llmman.apiKey:<path>`), so
// two daemons behind one gateway (`/a/`, `/b/`) never share a key; the app
// does the same, and its server URL keeps the path too.
//
// The app's routes are paths (`/models`, `/c/<id>`), so the base is not the
// address's directory: index.html's first script works it out and sets it
// as the page's <base> (scripts/export-web.mjs), and `document.baseURI` is
// that base.

/** Where the web UI keeps the key of a daemon at `/`. */
export const LEGACY_KEY_STORAGE = "llmman.apiKey:/";

/** The page's base path, with leading and trailing slashes: `/`, or `/gateway/llmman/`. */
export function basePath(baseURI: string): string {
  try {
    return new URL(".", baseURI).pathname;
  } catch {
    return "/";
  }
}

/** The page's daemon as a server URL: origin and base path, no trailing slash. */
export function pageServerUrl(baseURI: string): string | null {
  try {
    const origin = new URL(baseURI).origin;
    if (!/^https?:/i.test(origin)) return null;
    return `${origin}${basePath(baseURI).replace(/\/+$/, "")}`;
  } catch {
    return null;
  }
}

/** Where the key of the daemon at `path` is kept, as the earlier web UI named it. */
export function keyStorageName(path: string): string {
  return `llmman.apiKey:${path}`;
}

/** The key stored for the page's base path, and no other: a daemon at `/` never lends its key to another. */
export function readScopedKey(storage: Pick<Storage, "getItem">, path: string): string | undefined {
  return storage.getItem(keyStorageName(path))?.trim() || undefined;
}
