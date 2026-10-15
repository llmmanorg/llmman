// Server addresses: the URL a user types by hand, read the way the daemon
// reads `LLMMAN_HOST`.
//
// No `URL` here: React Native's implementation is partial, so the parsing
// is explicit and tested.

export type ParseResult<T> = { ok: true; value: T } | { ok: false; error: string };

const URL_SHAPE = /^(https?):\/\/(\[[0-9a-f:.]+\]|[^\s/?#:@[\]]+)(?::(\d{1,5}))?(\/[^\s?#]*)?$/i;

/** llmman serve's port when `LLMMAN_HOST` names none (llmman src/daemon.rs). */
export const DEFAULT_PORT = 17434;

/**
 * A typed server address as the base URL the client uses. Read
 * the way the daemon reads `LLMMAN_HOST`: no scheme means `http://` on
 * port 17434; an explicit `http://`/`https://` without a port means 80/443.
 * The scheme and host are lowercased, no query or fragment, no trailing slash.
 */
export function normalizeServerUrl(input: string): ParseResult<string> {
  let text = input.trim();
  if (!text) return { ok: false, error: "Enter the server's address." };
  const bare = !/^[a-z][a-z0-9+.-]*:\/\//i.test(text);
  if (bare) {
    if (/^[a-z][a-z0-9+.-]*:(?!\d)/i.test(text)) {
      return { ok: false, error: "Use an http:// or https:// address." };
    }
    text = `http://${text}`;
  }
  const m = URL_SHAPE.exec(text);
  if (!m) {
    if (/^[a-z][a-z0-9+.-]*:\/\//i.test(text) && !/^https?:\/\//i.test(text)) {
      return { ok: false, error: "Use an http:// or https:// address." };
    }
    if (/[?#]/.test(text)) return { ok: false, error: "Leave out the ? and # parts of the address." };
    return { ok: false, error: "That doesn't look like an address, e.g. 192.168.1.20:17434." };
  }
  const [, scheme = "", host = "", port, path = ""] = m;
  if (port !== undefined) {
    const n = Number(port);
    if (n < 1 || n > 65535) return { ok: false, error: "The port must be between 1 and 65535." };
  }
  const portPart = port !== undefined ? `:${Number(port)}` : bare ? `:${DEFAULT_PORT}` : "";
  const base = `${scheme.toLowerCase()}://${host.toLowerCase()}${portPart}`;
  return { ok: true, value: base + path.replace(/\/+$/, "") };
}

/** A short label for a server without a name: its host and port. */
export function hostLabel(url: string): string {
  return url.replace(/^https?:\/\//i, "");
}
