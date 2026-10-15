// App-wide server state: the known servers, the active one, and an
// `ApiClient` for it (with its key read from the secure store). Screens
// get the client from `useServers()`; nothing else builds one.

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

import { ApiClient } from "@/api/client";
import type { ChatSettings } from "@/lib/chatSettings";
import { onPageKeyEntered, pageKeySource, pageUrl } from "@/lib/page";
import { adoptPageServer } from "@/lib/pageServer";
import {
  getServerKey,
  loadServers,
  removeServer,
  renameServer,
  saveServer,
  setActiveServer,
  setServerDefaults,
  type Server,
  type ServerList,
} from "./servers";

export interface ServersState {
  /** False until the list has been read from storage. */
  ready: boolean;
  servers: Server[];
  active: Server | null;
  /** A client for the active server, or null when there is none (or its key is still loading). */
  client: ApiClient | null;
  /** Adds or updates a server; it becomes active unless `activate` is false. */
  save(entry: { url: string; name?: string; key?: string }, opts?: { activate?: boolean }): Promise<void>;
  remove(url: string): Promise<void>;
  activate(url: string): Promise<void>;
  rename(url: string, name: string): Promise<void>;
  setDefaults(url: string, defaults: ChatSettings): Promise<void>;
  keyFor(server: Server): Promise<string | undefined>;
}

const ServersContext = createContext<ServersState | null>(null);

export function ServersProvider({ children }: { children: ReactNode }) {
  const [list, setList] = useState<ServerList | null>(null);
  // Serializes writes so two quick taps cannot interleave storage updates;
  // `latest` is the list as of the last write, which the next one builds on.
  const queue = useRef<Promise<unknown>>(Promise.resolve());
  const latest = useRef<ServerList>({ servers: [], active: null });

  useEffect(() => {
    let cancelled = false;
    loadServers()
      .catch((): ServerList => ({ servers: [], active: null }))
      // On the web, the llmman serve the page came from.
      .then((l) => adoptPageServer(l).catch(() => l))
      .then((l) => {
        if (cancelled) return;
        latest.current = l;
        setList(l);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const active = useMemo(() => list?.servers.find((s) => s.url === list.active) ?? null, [list]);

  // A new client when the active server or its key changes. `keyVersion`
  // bumps on every save so a changed key is picked up for the same URL.
  const [keyVersion, setKeyVersion] = useState(0);
  const [loaded, setLoaded] = useState<{ server: Server; keyVersion: number; client: ApiClient } | null>(
    null,
  );
  useEffect(() => {
    let cancelled = false;
    if (!active) return;
    getServerKey(active)
      .catch(() => undefined)
      .then((key) => {
        if (cancelled) return;
        // The page's own daemon (web): its key is the page's, asked for on llmman's 401.
        const keySource = active.url === pageUrl() ? pageKeySource : undefined;
        setLoaded({
          server: active,
          keyVersion,
          client: new ApiClient({ url: active.url, key }, undefined, { keySource }),
        });
      });
    return () => {
      cancelled = true;
    };
  }, [active, keyVersion]);
  // A client built for another server, or before the last key change, is not handed out.
  const client =
    loaded && loaded.server === active && loaded.keyVersion === keyVersion ? loaded.client : null;

  const run = useCallback((update: (l: ServerList) => Promise<ServerList>) => {
    const next = queue.current.then(async () => {
      const updated = await update(latest.current);
      latest.current = updated;
      setList(updated);
    });
    queue.current = next.catch(() => {});
    return next;
  }, []);

  // A key entered at the page's prompt is this server's key from now on
  // (sockets and Settings read it from the store).
  useEffect(
    () =>
      onPageKeyEntered((key) => {
        const url = pageUrl();
        const server = url ? latest.current.servers.find((s) => s.url === url) : undefined;
        if (!server) return;
        void run((l) =>
          saveServer(
            l,
            { url: server.url, key, ...(server.name ? { name: server.name } : {}) },
            { activate: false },
          ),
        ).then(() => setKeyVersion((v) => v + 1));
      }),
    [run],
  );

  const value = useMemo<ServersState>(
    () => ({
      ready: list !== null,
      servers: list?.servers ?? [],
      active,
      client,
      save: async (entry, opts) => {
        await run((l) => saveServer(l, entry, opts));
        setKeyVersion((v) => v + 1);
      },
      remove: (url) => run((l) => removeServer(l, url)),
      activate: (url) => run((l) => setActiveServer(l, url)),
      rename: (url, name) => run((l) => renameServer(l, url, name)),
      setDefaults: (url, defaults) => run((l) => setServerDefaults(l, url, defaults)),
      keyFor: (server) => getServerKey(server),
    }),
    [list, active, client, run],
  );

  return <ServersContext.Provider value={value}>{children}</ServersContext.Provider>;
}

export function useServers(): ServersState {
  const ctx = useContext(ServersContext);
  if (!ctx) throw new Error("useServers outside ServersProvider");
  return ctx;
}
