// The active llmman serve as the frame shows it: whether it answers (a
// `/api/version` poll every 15 s, as llmman's web UI drives its status dot)
// and which optional features it has (daemonFeatures.ts). Screens read it
// with `useDaemon()`.

import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { AppState } from "react-native";

import type { VersionInfo } from "@/api/client";
import { detectFeatures, type DaemonFeatures } from "@/lib/daemonFeatures";
import { useServers } from "@/lib/ServersContext";

export const POLL_MS = 15_000;

export interface DaemonState {
  /** Null until the first answer (or failure) for this server. */
  reachable: boolean | null;
  version: VersionInfo | null;
  /** Null until probed: treat as none. */
  features: DaemonFeatures | null;
  /** Asks again now, e.g. after the key changed. */
  refresh(): void;
}

interface Snapshot {
  client: unknown;
  reachable: boolean | null;
  version: VersionInfo | null;
  features: DaemonFeatures | null;
}

const DaemonContext = createContext<DaemonState | null>(null);

export function DaemonProvider({ children }: { children: ReactNode }) {
  const { client } = useServers();
  const [state, setState] = useState<Snapshot>({
    client: null,
    reachable: null,
    version: null,
    features: null,
  });
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (!client) return;
    let cancelled = false;
    // This client's snapshot, started afresh when the server or key changed.
    const update = (patch: Partial<Snapshot>) =>
      setState((s) => ({
        ...(s.client === client ? s : { client, reachable: null, version: null, features: null }),
        ...patch,
      }));
    // Features: once per client, and again when the daemon comes back after
    // being unreachable (it may have been restarted with other settings).
    const probe = () =>
      void detectFeatures(client, { timeoutMs: 8000 }).then((features) => !cancelled && update({ features }));
    let wasReachable: boolean | null = null;
    const poll = async () => {
      // Not while the app is in the background on a phone.
      if (AppState.currentState === "background") return;
      const version = await client.version({ timeoutMs: 8000 }).catch(() => null);
      if (cancelled) return;
      const reachable = version !== null;
      if (reachable && wasReachable === false) probe();
      wasReachable = reachable;
      update(version ? { reachable, version } : { reachable });
    };
    probe();
    void poll();
    const timer = setInterval(() => void poll(), POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [client, tick]);

  const current = state.client === client ? state : null;
  const value = useMemo<DaemonState>(
    () => ({
      reachable: current?.reachable ?? null,
      version: current?.version ?? null,
      features: current?.features ?? null,
      refresh: () => setTick((t) => t + 1),
    }),
    [current],
  );
  return <DaemonContext.Provider value={value}>{children}</DaemonContext.Provider>;
}

export function useDaemon(): DaemonState {
  const ctx = useContext(DaemonContext);
  if (!ctx) throw new Error("useDaemon outside DaemonProvider");
  return ctx;
}
