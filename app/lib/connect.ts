// Connect flow: check the address and test it with the key. Nothing is
// saved here.

import type { FetchLike } from "@/api/client";
import { testConnection, type ConnectionProblem } from "@/api/connection";
import { normalizeServerUrl } from "@/api/serverUrl";

export type ConnectOutcome =
  | { ok: true; url: string; key?: string; version: string }
  | { ok: false; field: "address" | "key" | null; problem: ConnectionProblem | "invalid"; message: string };

export async function checkServer(
  input: { address: string; key?: string },
  opts: { signal?: AbortSignal; fetchImpl?: FetchLike } = {},
): Promise<ConnectOutcome> {
  const url = normalizeServerUrl(input.address);
  if (!url.ok) return { ok: false, field: "address", problem: "invalid", message: url.error };
  const key = input.key?.trim() || undefined;

  const result = await testConnection(url.value, key, opts);
  if (!result.ok) {
    const field =
      result.problem === "unauthorized" ? "key" : result.problem === "server-error" ? null : "address";
    return { ok: false, field, problem: result.problem, message: result.message };
  }
  return { ok: true, url: url.value, ...(key ? { key } : {}), version: result.version.version };
}
