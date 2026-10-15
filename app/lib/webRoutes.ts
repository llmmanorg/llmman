// The web app's addresses as llmman's web UI had them, and what the page
// is called on each screen.

/**
 * The app route for one of llmman's web UI fragment links (`#/models`,
 * `#/c/<id>`, `#/new`, `#/shell`; its docs link `#/models`), or null when
 * the fragment is none of them (`#/`, empty, anything else).
 */
export function hashRoute(hash: string): string | null {
  const path = hash.replace(/^#/, "");
  if (path === "/models" || path === "/models/") return "/models";
  if (path === "/shell") return "/shell";
  if (path === "/new") return "/";
  const m = /^\/c\/([^/?#]+)$/.exec(path);
  if (m) return `/c/${m[1]}`;
  return null;
}

const TITLES: [prefix: string, title: string][] = [
  ["/shell", "Shell"],
  ["/models", "Models"],
  ["/providers", "Providers"],
  ["/settings", "Settings"],
  ["/servers", "Servers"],
  ["/server", "Server"],
  ["/connect", "Add a server"],
  ["/welcome", "Connect"],
];

/** The page's title for `pathname`: "Shell · llmman", as the web UI names its views; chats are "llmman". */
export function documentTitle(pathname: string): string {
  const hit = TITLES.find(([p]) => pathname === p || pathname.startsWith(`${p}/`));
  return hit ? `${hit[1]} · llmman` : "llmman";
}

/** Screens shown as a sheet or popover over the page: Escape closes them. */
const SHEETS = new Set(["/model-picker", "/chat-settings", "/chat-actions", "/provider-key", "/connect"]);

export function isSheet(pathname: string): boolean {
  return SHEETS.has(pathname.replace(/\/+$/, ""));
}
