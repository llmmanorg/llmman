// The Models tab's state, shared by its screens, one provider per server:
// this computer's models and memory, the cards fetched so far, the variant
// picked on each, and the one pull at a time (as in the web UI). A pull
// outlives the screen it started on; after the app comes back from the
// background it watches the same pull again, since the daemon keeps pulling
// whether or not anyone watches.

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { AppState } from "react-native";

import { ApiError, isAbort, type LocalModel, type TagModel } from "@/api/client";
import { modelCard, type ModelCard, type NodeInfo } from "@/api/search";
import { useModels as useChatModels } from "@/lib/ModelsContext";
import { useServers } from "@/lib/ServersContext";
import { applyPullEvent, repoOf, startPull, type PullState } from "./logic";

export type Pull = PullState & {
  /** Why the pull failed (see errors.ts explain), or null. */
  error: Error | null;
  /** Set once it succeeded; the banner says so for a moment. */
  done: boolean;
};

export type CardEntry = ModelCard | ApiError | Error | "loading";

export interface ModelsTabState {
  local: LocalModel[];
  /** `/api/tags` rows by lowercase name. */
  details: Map<string, TagModel>;
  /** Bytes of model memory, 0 while unknown. */
  memory: number;
  /** False until the first read of the local store finished (or failed). */
  localReady: boolean;
  localError: Error | null;
  reloadLocal(): Promise<void>;
  /** The store changed (pull, load, unload, delete): re-read it here and in the chat's model list. */
  storeChanged(): Promise<void>;

  card(name: string): CardEntry | undefined;
  loadCard(name: string): void;
  picked(name: string): string | undefined;
  pick(name: string, variant: string): void;

  pull: Pull | null;
  /** Starts pulling `ref`; false when another pull is running. */
  startPull(ref: string): boolean;
  /** Stops following the pull. The computer keeps downloading. */
  stopWatching(): void;
  /** Forgets a failed or finished pull. */
  dismissPull(): void;
  /** The daemon's notices (signature verification) from this session's finished pulls of repo `name`. */
  noticesFor(name: string): { ref: string; notices: string[] }[];
}

const ModelsTabContext = createContext<ModelsTabState | null>(null);

/** How long the banner says "Pulled …" after a pull succeeds. */
const DONE_FOR_MS = 4000;

export function ModelsTabProvider({ children }: { children: ReactNode }) {
  const { client } = useServers();
  const [local, setLocal] = useState<LocalModel[]>([]);
  const [details, setDetails] = useState<Map<string, TagModel>>(new Map());
  const [memory, setMemory] = useState(0);
  const [localReady, setLocalReady] = useState(false);
  const [localError, setLocalError] = useState<Error | null>(null);
  const [cards, setCards] = useState<Map<string, CardEntry>>(new Map());
  const [choices, setChoices] = useState<Map<string, string>>(new Map());
  const [pull, setPull] = useState<Pull | null>(null);
  const [notices, setNotices] = useState<Map<string, string[]>>(new Map());
  const pullAbort = useRef<AbortController | null>(null);
  const pullRunning = useRef(false);
  const doneTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // The layout remounts this per server, so all of it belongs to one; a
  // pull still followed when it goes is stopped being followed.
  useEffect(
    () => () => {
      pullAbort.current?.abort();
      if (doneTimer.current) clearTimeout(doneTimer.current);
    },
    [],
  );

  const reloadLocal = useCallback(async () => {
    if (!client) return;
    const [models, tags, node] = await Promise.allSettled([
      client.listLocal(),
      client.listLocalDetailed(),
      client.nodeInfo<NodeInfo>(),
    ]);
    if (models.status === "fulfilled") {
      setLocal(models.value);
      setLocalError(null);
    } else {
      setLocalError(asError(models.reason));
    }
    if (tags.status === "fulfilled") setDetails(new Map(tags.value.map((t) => [t.name.toLowerCase(), t])));
    if (node.status === "fulfilled") setMemory(node.value.memory || 0);
    setLocalReady(true);
  }, [client]);

  useEffect(() => {
    reloadLocal();
  }, [reloadLocal]);

  const { refresh: refreshChat } = useChatModels();
  const storeChanged = useCallback(async () => {
    await Promise.all([reloadLocal(), refreshChat()]);
  }, [reloadLocal, refreshChat]);

  const cardsRef = useRef(cards);
  cardsRef.current = cards;
  const loadCard = useCallback(
    (name: string) => {
      const have = cardsRef.current.get(name);
      if (!client || have === "loading" || (have && !(have instanceof Error))) return;
      const set = (entry: CardEntry) => setCards((m) => new Map(m).set(name, entry));
      set("loading");
      // An error is kept only until the card is opened again, which retries.
      modelCard(client, name).then(set, (e) => set(e instanceof Error ? e : new Error(String(e))));
    },
    [client],
  );

  const startPullFn = useCallback(
    (ref: string) => {
      if (!client || pullRunning.current) return false;
      pullRunning.current = true;
      const abort = new AbortController();
      pullAbort.current = abort;
      setPull({ ...startPull(ref), error: null, done: false });

      // Whether the app went to the background during this attempt: a
      // dropped connection then means the OS closed it, not a failure.
      let backgrounded = false;
      const sub = AppState.addEventListener("change", (s) => {
        if (s !== "active") backgrounded = true;
      });

      let progress = startPull(ref);
      (async () => {
        try {
          for (;;) {
            try {
              await client.pull(
                ref,
                (ev) => {
                  progress = applyPullEvent(progress, ev);
                  const now = progress;
                  setPull((p) => p && { ...p, ...now });
                },
                abort.signal,
              );
              // Notices outlive the banner: they are about the model now on this computer.
              const said = progress.notices;
              if (said.length) setNotices((m) => new Map(m).set(ref, said));
              setPull((p) => p && { ...p, done: true });
              if (doneTimer.current) clearTimeout(doneTimer.current);
              doneTimer.current = setTimeout(
                () => setPull((p) => (p?.done && p.ref === ref ? null : p)),
                DONE_FOR_MS,
              );
              return;
            } catch (e) {
              if (isAbort(e)) {
                setPull(null);
                return;
              }
              const dropped = e instanceof ApiError && e.kind === "network";
              if (!dropped || !(backgrounded || AppState.currentState !== "active")) {
                setPull((p) => p && { ...p, error: asError(e) });
                return;
              }
              // Watch the same pull again once the app is in front.
              await untilActive(abort.signal);
              backgrounded = false;
            }
          }
        } catch (e) {
          if (isAbort(e)) setPull(null);
        } finally {
          sub.remove();
          pullRunning.current = false;
          if (pullAbort.current === abort) pullAbort.current = null;
          storeChanged();
        }
      })();
      return true;
    },
    [client, storeChanged],
  );

  const value = useMemo<ModelsTabState>(
    () => ({
      local,
      details,
      memory,
      localReady,
      localError,
      reloadLocal,
      storeChanged,
      card: (name) => cards.get(name),
      loadCard,
      picked: (name) => choices.get(name),
      pick: (name, variant) => setChoices((m) => new Map(m).set(name, variant)),
      pull,
      startPull: startPullFn,
      stopWatching: () => pullAbort.current?.abort(),
      dismissPull: () => {
        if (!pullRunning.current) setPull(null);
      },
      noticesFor: (name) =>
        [...notices]
          .filter(([ref]) => repoOf(ref).toLowerCase() === name.toLowerCase())
          .map(([ref, n]) => ({ ref, notices: n })),
    }),
    [
      local,
      details,
      memory,
      localReady,
      localError,
      reloadLocal,
      storeChanged,
      cards,
      loadCard,
      choices,
      pull,
      startPullFn,
      notices,
    ],
  );

  return <ModelsTabContext.Provider value={value}>{children}</ModelsTabContext.Provider>;
}

export function useModelsTab(): ModelsTabState {
  const ctx = useContext(ModelsTabContext);
  if (!ctx) throw new Error("useModelsTab outside ModelsTabProvider");
  return ctx;
}

/** Whether `pull` is of repository `name`. */
export function pullOf(pull: Pull | null, name: string): boolean {
  return !!pull && repoOf(pull.ref).toLowerCase() === name.toLowerCase();
}

export function asError(e: unknown): Error {
  return e instanceof Error ? e : new Error(String(e));
}

/** Resolves when the app is in front, or rejects with an AbortError on `signal`. */
function untilActive(signal: AbortSignal): Promise<void> {
  if (AppState.currentState === "active") return Promise.resolve();
  return new Promise((resolve, reject) => {
    const sub = AppState.addEventListener("change", (s) => {
      if (s !== "active") return;
      done();
      resolve();
    });
    const onAbort = () => {
      done();
      const e = new Error("aborted");
      e.name = "AbortError";
      reject(e);
    };
    const done = () => {
      sub.remove();
      signal.removeEventListener("abort", onAbort);
    };
    signal.addEventListener("abort", onAbort, { once: true });
  });
}
