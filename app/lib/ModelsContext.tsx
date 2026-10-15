// The models the chat can use on the active server, as the web UI's picker
// offers them: local models that take a chat turn (from /v1/models, with
// /api/show capabilities), local media generation models (capabilities
// `image`/`video`/`audio` rather than `completion`), and hosted models of
// providers whose key the daemon can use. The chosen model is remembered
// per server.

import AsyncStorage from "@react-native-async-storage/async-storage";
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

import { splitRemoteRef, type ApiClient, type ProviderModel } from "@/api/client";
import { explain } from "@/api/errors";
import { mediaKindsOf, type MediaKind } from "@/lib/media";
import { modelName } from "@/lib/modelName";
import { useServers } from "@/lib/ServersContext";

export interface LocalChatModel {
  id: string;
  loaded: boolean;
  /** From /api/show; null when unknown (counted as chattable, as the web UI does). */
  capabilities: string[] | null;
  /** What a media generation model makes; empty for a chat model. */
  kinds: MediaKind[];
  /** Bytes on disk, from /api/tags; absent when unknown. */
  size?: number;
}

export interface HostedProvider {
  id: string;
  name: string;
  count: number;
  /** Null until fetched (on demand: there can be hundreds). */
  models: ProviderModel[] | null;
}

export interface ModelsState {
  local: LocalChatModel[];
  providers: HostedProvider[];
  loading: boolean;
  error: string | null;
  /** When the lists were last fetched (ms), 0 before the first answer. */
  loadedAt: number;
  selected: string;
  select(ref: string): void;
  refresh(): Promise<void>;
  loadProvider(id: string): Promise<void>;
  isLoaded(ref: string): boolean;
  acceptsImages(ref: string): boolean;
  /** The media `ref` generates; empty for a chat model, a hosted one or an unknown ref. */
  mediaKinds(ref: string): MediaKind[];
  markLoaded(ref: string, loaded?: boolean): void;
  /** Unloads a local model from memory (`keep_alive: 0`, as the web UI's picker does). */
  unload(ref: string): Promise<void>;
  /** The friendly name (lib/modelName.ts), with twins told apart. */
  displayName(ref: string): string;
}

/**
 * Whether a model takes images: a local one whose capabilities say
 * `vision` (unknown counts as yes, as for chat); a hosted one always, since
 * the catalog does not say, and a refusal comes back as the reply's error.
 */
export function acceptsImages(ref: string, local: Pick<LocalChatModel, "id" | "capabilities">[]): boolean {
  if (!ref) return false;
  if (splitRemoteRef(ref)) return true;
  const m = local.find((l) => l.id === ref);
  return !m?.capabilities || m.capabilities.includes("vision");
}

/** Whether a local model can take a chat turn. */
export function chattable(m: Pick<LocalChatModel, "capabilities">): boolean {
  return !m.capabilities || m.capabilities.includes("completion");
}

/** Whether a local model generates media rather than text. */
export function generative(m: Pick<LocalChatModel, "capabilities">): boolean {
  return !chattable(m) && mediaKindsOf(m.capabilities).length > 0;
}

/** The friendly name of `ref` among this server's models (lib/modelName.ts). */
export function displayNameOf(
  ref: string,
  providers: HostedProvider[] | { id: string; name: string }[],
  local: readonly { id: string }[] = [],
): string {
  return modelName(ref, { providers, local: local.map((m) => m.id) });
}

/** Local and provider lists for `client`; capabilities cached by `cache`. */
export async function fetchModels(
  client: ApiClient,
  cache: Map<string, string[] | null>,
): Promise<{ local: LocalChatModel[]; providers: Omit<HostedProvider, "models">[] }> {
  const [local, providers, tags] = await Promise.all([
    client.listLocal(),
    client.listProviders().catch(() => []),
    // Sizes only; a daemon that cannot list them still lists the models.
    client.listLocalDetailed().catch(() => []),
  ]);
  const sizes = new Map(tags.map((t) => [t.name, t.size]));
  const withCaps = await Promise.all(
    local.map(async (m) => {
      if (!cache.has(m.id)) {
        try {
          const info = await client.showModel(m.id);
          cache.set(m.id, Array.isArray(info.capabilities) ? info.capabilities : null);
        } catch {
          cache.set(m.id, null);
        }
      }
      const capabilities = cache.get(m.id) ?? null;
      const kinds = generative({ capabilities }) ? mediaKindsOf(capabilities) : [];
      const size = sizes.get(m.id);
      return { ...m, capabilities, kinds, ...(size ? { size } : {}) };
    }),
  );
  return {
    // What the picker offers: chat models and media generation models.
    local: withCaps.filter((m) => chattable(m) || m.kinds.length).sort((a, b) => a.id.localeCompare(b.id)),
    providers: providers
      .filter((p) => p.key_usable)
      .map((p) => ({ id: p.id, name: p.name, count: p.models ?? 0 }))
      .sort((a, b) => a.name.localeCompare(b.name)),
  };
}

const ModelsContext = createContext<ModelsState | null>(null);
const selectedKey = (server: string) => `llmman.model:${server}`;

export function ModelsProvider({ children }: { children: ReactNode }) {
  const { client, active } = useServers();
  const [data, setData] = useState<{
    server: string;
    local: LocalChatModel[];
    providers: HostedProvider[];
    error: string | null;
    at: number;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<{ server: string; ref: string } | null>(null);
  const caps = useRef(new Map<string, Map<string, string[] | null>>());

  const server = active?.url ?? "";
  const current = data?.server === server ? data : null;
  const selectedRef = selected?.server === server ? selected.ref : "";

  const refresh = useCallback(async () => {
    if (!client) return;
    const url = client.url;
    if (!caps.current.has(url)) caps.current.set(url, new Map());
    setLoading(true);
    try {
      const { local, providers } = await fetchModels(client, caps.current.get(url)!);
      setData((prev) => ({
        server: url,
        local,
        // Keep model lists already fetched for providers still present.
        providers: providers.map((p) => ({
          ...p,
          models: (prev?.server === url && prev.providers.find((q) => q.id === p.id)?.models) || null,
        })),
        error: null,
        at: Date.now(),
      }));
    } catch (e) {
      setData((prev) => ({
        server: url,
        local: prev?.server === url ? prev.local : [],
        providers: prev?.server === url ? prev.providers : [],
        error: explain(e, { name: active?.name, url }).title,
        at: Date.now(),
      }));
    } finally {
      setLoading(false);
    }
  }, [client, active?.name]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // The remembered choice for this server.
  useEffect(() => {
    if (!server) return;
    let cancelled = false;
    AsyncStorage.getItem(selectedKey(server)).then((ref) => {
      if (!cancelled) setSelected((s) => (s?.server === server ? s : { server, ref: ref ?? "" }));
    });
    return () => {
      cancelled = true;
    };
  }, [server]);

  // Nothing chosen yet (or the choice is gone): a loaded chat model, else
  // the first chat model, else a media model, so the composer is usable.
  const fallback = useMemo(() => {
    if (!current) return "";
    const known = (ref: string) => !!splitRemoteRef(ref) || current.local.some((m) => m.id === ref);
    if (selectedRef && known(selectedRef)) return selectedRef;
    const chat = current.local.filter((m) => !m.kinds.length);
    return (chat.find((m) => m.loaded) ?? chat[0] ?? current.local[0])?.id ?? "";
  }, [current, selectedRef]);

  const value = useMemo<ModelsState>(
    () => ({
      local: current?.local ?? [],
      providers: current?.providers ?? [],
      loading,
      error: current?.error ?? null,
      loadedAt: current?.at ?? 0,
      selected: fallback || selectedRef,
      select: (ref) => {
        if (!server) return;
        setSelected({ server, ref });
        void AsyncStorage.setItem(selectedKey(server), ref);
      },
      refresh,
      loadProvider: async (id) => {
        if (!client) return;
        const url = client.url;
        let models: ProviderModel[];
        try {
          models = await client.providerModels(id);
        } catch {
          models = [];
        }
        setData((prev) =>
          prev?.server === url
            ? { ...prev, providers: prev.providers.map((p) => (p.id === id ? { ...p, models } : p)) }
            : prev,
        );
      },
      acceptsImages: (ref) => acceptsImages(ref, current?.local ?? []),
      mediaKinds: (ref) => current?.local.find((m) => m.id === ref)?.kinds ?? [],
      isLoaded: (ref) => !!current?.local.some((m) => m.id === ref && m.loaded),
      markLoaded: (ref, loaded = true) =>
        setData((prev) =>
          prev ? { ...prev, local: prev.local.map((m) => (m.id === ref ? { ...m, loaded } : m)) } : prev,
        ),
      unload: async (ref) => {
        if (!client) return;
        await client.unloadModel(ref);
        setData((prev) =>
          prev?.server === client.url
            ? { ...prev, local: prev.local.map((m) => (m.id === ref ? { ...m, loaded: false } : m)) }
            : prev,
        );
      },
      displayName: (ref) => displayNameOf(ref, current?.providers ?? [], current?.local ?? []),
    }),
    [current, loading, fallback, selectedRef, server, refresh, client],
  );

  return <ModelsContext.Provider value={value}>{children}</ModelsContext.Provider>;
}

export function useModels(): ModelsState {
  const ctx = useContext(ModelsContext);
  if (!ctx) throw new Error("useModels outside ModelsProvider");
  return ctx;
}
