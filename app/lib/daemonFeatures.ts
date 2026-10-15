// What the active llmman serve offers beyond the core API, asked once per
// server: for now, the shell. A probe that fails means "not here".

import type { ApiClient, RequestOptions } from "@/api/client";
import { shellStatus, type ShellStatus } from "@/api/shell";

export interface DaemonFeatures {
  /** `GET /llmman/shell`; `endpoint: false` from a daemon without one. */
  shell: ShellStatus;
}

export async function detectFeatures(client: ApiClient, opts?: RequestOptions): Promise<DaemonFeatures> {
  const shell = await shellStatus(client, opts).catch((e): ShellStatus => ({
    enabled: false,
    endpoint: true,
    reason: e instanceof Error ? e.message : String(e),
  }));
  return { shell };
}
