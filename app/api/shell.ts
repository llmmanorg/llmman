// `/llmman/shell`: a terminal on the computer (protocol in llmman's
// src/cmd/serve/shell.rs and docs/webui.md). A plain
// GET says whether a shell may open; the upgrade opens one. Binary frames
// carry terminal bytes both ways, the client's text frames are
// `{"resize":{"cols","rows"}}`, and the daemon's one text frame is
// `{"exit":CODE}` before it closes. The daemon decides; this side only asks.

import { ApiError, type ApiClient, type RequestOptions } from "./client";
import { openKeyedSocket, socketUrl } from "./socket";

export interface ShellStatus {
  enabled: boolean;
  /** Why the shell is off (e.g. llmman serve listens beyond loopback). */
  reason?: string;
  /** False for a daemon without the route at all (404). */
  endpoint: boolean;
  /** True when the upgrade needs the daemon's shell token; unset from a daemon older than the token. */
  tokenRequired?: boolean;
}

export const NO_SHELL_ENDPOINT = "This llmman serve has no shell endpoint.";

/** `GET /llmman/shell` → `{enabled, reason?, token_required?}`; a 404 is a daemon with no shell. */
export async function shellStatus(client: ApiClient, opts?: RequestOptions): Promise<ShellStatus> {
  try {
    const body = await client.getJson<{ enabled?: unknown; reason?: unknown; token_required?: unknown }>(
      "llmman/shell",
      opts,
    );
    const reason = typeof body?.reason === "string" && body.reason.trim() ? body.reason.trim() : undefined;
    return {
      enabled: body?.enabled === true,
      endpoint: true,
      ...(reason ? { reason } : {}),
      ...(body?.token_required === true ? { tokenRequired: true } : {}),
    };
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) {
      return { enabled: false, endpoint: false, reason: NO_SHELL_ENDPOINT };
    }
    throw e;
  }
}

/** `ws(s)://…/llmman/shell` for the server at `url`. */
export function shellSocketUrl(url: string): string {
  return socketUrl(url, "llmman/shell");
}

export function shellTokenProtocol(token: string): string {
  return `llmman.shell.${token}`;
}

/** A shell WebSocket, carrying the key when there is one (a subprotocol in a browser) and the shell token. */
export function openShellSocket(url: string, key: string | undefined, token: string | undefined): WebSocket {
  return openKeyedSocket(shellSocketUrl(url), key, token ? [shellTokenProtocol(token)] : []);
}
