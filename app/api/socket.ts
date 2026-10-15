// The daemon's WebSocket (the shell) and how it carries the key: a phone
// sends it as a header; a browser cannot set one on an upgrade, so it
// offers the key as a subprotocol the daemon echoes.

import { Platform } from "react-native";

/** The key as a WebSocket subprotocol, the way a browser presents it. */
export function keyProtocol(key: string): string {
  const bytes = new TextEncoder().encode(key);
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return `llmman.bearer.${btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "")}`;
}

/** `ws(s)://…/<path>` on the server at `url` (a base URL, no trailing slash). */
export function socketUrl(url: string, path: string): string {
  return `${url.replace(/^http/i, "ws")}/${path.replace(/^\/+/, "")}`;
}

/** A binary WebSocket to `target`, carrying `key` the way this platform can. */
export function openKeyedSocket(target: string, key: string | undefined): WebSocket {
  const ws =
    Platform.OS === "web"
      ? new WebSocket(target, key ? [keyProtocol(key)] : undefined)
      : new (WebSocket as unknown as new (u: string, p: string[] | undefined, o: object) => WebSocket)(
          target,
          undefined,
          key ? { headers: { Authorization: `Bearer ${key}` } } : {},
        );
  ws.binaryType = "arraybuffer";
  return ws;
}
