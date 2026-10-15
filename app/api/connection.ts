// "Test connection": one `GET /api/version` with the key, and a message
// the user can act on for each way it can fail.

import { Platform } from "react-native";

import { ApiClient, ApiError, type FetchLike, type VersionInfo } from "./client";
import { DEFAULT_PORT } from "./serverUrl";

export type ConnectionProblem =
  | "unauthorized" // 401: key missing or wrong
  | "unreachable" // nothing answered at that address
  | "timeout" // something may be there, but it did not answer in time
  | "not-llmman" // something answered, but it is not llmman serve
  | "server-error"; // llmman answered with another error

export type ConnectionResult =
  | { ok: true; version: VersionInfo }
  | { ok: false; problem: ConnectionProblem; message: string; detail?: string };

export const CONNECT_TIMEOUT_MS = 8000;

export async function testConnection(
  url: string,
  key: string | undefined,
  opts: { signal?: AbortSignal; fetchImpl?: FetchLike; timeoutMs?: number } = {},
): Promise<ConnectionResult> {
  const client = new ApiClient({ url, key }, opts.fetchImpl);
  try {
    const version = await client.version({
      signal: opts.signal,
      timeoutMs: opts.timeoutMs ?? CONNECT_TIMEOUT_MS,
    });
    return { ok: true, version };
  } catch (e) {
    if (!(e instanceof ApiError)) throw e; // an abort, or a bug
    return describe(e, url, Boolean(key));
  }
}

/**
 * For `http://host` with no port (port 80, as the daemon reads it too), the
 * likely intent: llmman serve's default port. Empty otherwise.
 */
function portHint(url: string): string {
  const m = /^(https?:\/\/(?:\[[^\]]+\]|[^/:]+))(\/.*)?$/i.exec(url);
  return m
    ? ` llmman serve listens on ${DEFAULT_PORT} by default; did you mean ${m[1]}:${DEFAULT_PORT}${m[2] ?? ""}?`
    : "";
}

function describe(e: ApiError, url: string, hadKey: boolean): ConnectionResult & { ok: false } {
  const detail = e.message;
  switch (e.kind) {
    case "network":
      return {
        ok: false,
        problem: "unreachable",
        message:
          `Nothing answered at ${url}. Check the address and port, that the phone is on the same network, ` +
          "and that llmman serve listens beyond loopback (LLMMAN_HOST=0.0.0.0)." +
          portHint(url),
        detail,
      };
    case "timeout":
      return {
        ok: false,
        problem: "timeout",
        message:
          `${url} did not answer in time. Check the address and that the phone is on the same network.` +
          portHint(url),
        detail,
      };
    case "protocol":
      return {
        ok: false,
        problem: "not-llmman",
        message: `Something answered at ${url}, but it is not llmman serve. Check the address and port.`,
        detail,
      };
    case "http":
      // A browser on another origin cannot read WWW-Authenticate unless the
      // daemon exposes it, so on the web a realm-less 401 from /api/version
      // (llmman's own route, never relayed) counts as llmman's too. Elsewhere
      // one is a proxy's or the like, not llmman's key check.
      if (e.unauthorized || (e.status === 401 && hiddenRealm(url))) {
        return {
          ok: false,
          problem: "unauthorized",
          message: hadKey
            ? "The server rejected this API key. Check it against LLMMAN_API_KEYS on the computer."
            : "This server needs an API key (LLMMAN_API_KEYS). Enter it to connect.",
          detail,
        };
      }
      if (e.status === 404 || e.status === 405) {
        return {
          ok: false,
          problem: "not-llmman",
          message: `${url} has no llmman API (${e.status}). Check the address, port and any path prefix.`,
          detail,
        };
      }
      return {
        ok: false,
        problem: "server-error",
        message: `The server answered with an error (${e.status}): ${detail}`,
        detail,
      };
  }
}

/**
 * Whether a browser may have hidden the 401's realm: only on the web, for
 * another origin (CORS exposes WWW-Authenticate only when the daemon lists
 * it). On the page's own origin the realm is always readable, so "key
 * rejected" there comes from llmman's own 401 alone.
 */
function hiddenRealm(url: string): boolean {
  if (Platform.OS !== "web" || typeof window === "undefined") return false;
  return !url.startsWith(window.location.origin);
}
