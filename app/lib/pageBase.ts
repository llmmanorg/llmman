// Where the web app's own llmman serve is, from the page's address. llmman's
// web UI scopes its key by the page's base path (`llmman.apiKey:<path>`), so
// two daemons behind one gateway (`/a/`, `/b/`) never share a key; the app
// does the same, and its server URL keeps the path too.
//
// Unlike llmman's web UI, which routes in the fragment, this app's routes
// are paths (`/models`, `/c/<id>`), so the base is the page's directory
// without the app's own route directories.

/** Where the web UI and earlier versions of this app kept the key of a daemon at `/`. */
export const LEGACY_KEY_STORAGE = "llmman.apiKey:/";

/** The app's route directories (app/), which are never part of the base. */
const ROUTE_DIRS = new Set(["c", "models", "providers", "settings", "servers", "server", "connect"]);

/** The page's base path, with leading and trailing slashes: `/`, or `/gateway/llmman/`. */
export function basePath(baseURI: string): string {
  let path: string;
  try {
    path = new URL(".", baseURI).pathname;
  } catch {
    return "/";
  }
  const parts = path.split("/").filter(Boolean);
  if (parts.length && ROUTE_DIRS.has(parts[parts.length - 1]!)) parts.pop();
  return parts.length ? `/${parts.join("/")}/` : "/";
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

/** The key stored for the page's base path, taking over the one at `/` from earlier versions. */
export function readScopedKey(
  storage: Pick<Storage, "getItem" | "setItem">,
  path: string,
): string | undefined {
  const name = keyStorageName(path);
  const own = storage.getItem(name)?.trim();
  if (own) return own;
  if (name === LEGACY_KEY_STORAGE) return undefined;
  const legacy = storage.getItem(LEGACY_KEY_STORAGE)?.trim();
  if (!legacy) return undefined;
  storage.setItem(name, legacy);
  return legacy;
}
