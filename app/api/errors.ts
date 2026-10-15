// What went wrong with a request to a server that is already set up, in
// words a person can act on: one wording for the Chat and Models tabs. The
// raw message is kept as `detail` (shown small); the connection layer's
// internals (Swift file names, exception classes) and the daemon's CLI
// advice stay out of the headline. Testing a new address is a different
// job: see connection.ts.

import { ApiError } from "./client";
import { hostLabel } from "./serverUrl";

export interface Explained {
  /** One or two plain sentences. */
  title: string;
  /** The raw message, for whoever wants it; absent when the title already is it. */
  detail?: string;
  /** llmman itself refused the key: retrying cannot help, the key must change. */
  keyRejected: boolean;
}

export function explain(e: unknown, server: { name?: string; url: string }): Explained {
  const name = server.name || hostLabel(server.url);
  const message = e instanceof Error ? e.message : String(e);
  if (e instanceof ApiError) {
    if (e.unauthorized) {
      return {
        title: `${name} rejected this phone's API key. Check it against LLMMAN_API_KEYS on the computer.`,
        detail: message,
        keyRejected: true,
      };
    }
    if (e.kind === "network") {
      return {
        title: `${name} isn't answering. Is llmman serve running there, and is this phone on the same network?`,
        detail: message,
        keyRejected: false,
      };
    }
    if (e.kind === "timeout") {
      return {
        title: `${name} didn't answer in time. Is this phone on the same network?`,
        detail: message,
        keyRejected: false,
      };
    }
  }
  // The daemon's own words (a provider's 401, a registry's 502, a pull's {error}).
  return { title: message, keyRejected: false };
}
