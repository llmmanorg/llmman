// The Models screen: popular models before a search, search results after
// one, and what this computer already has, as the web UI's Models page lists
// them. A full reference typed in the search bar is offered as a pull. On a
// wide screen the card opens beside the list, as on the web UI.

import { Stack, useFocusEffect } from "expo-router";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Platform,
  Pressable,
  RefreshControl,
  SectionList,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";

import type { LocalModel } from "@/api/client";
import { popular as fetchPopular, search, type SearchHit } from "@/api/search";
import { Icon } from "@/components/Icon";
import { Avatar } from "@/components/models/Avatar";
import { Muted, Segments, useLargeText, useTextSizeKey } from "@/components/models/bits";
import { HitRow } from "@/components/models/HitRow";
import { LocalTagRow } from "@/components/models/LocalTagRow";
import {
  holdsRef,
  isPulled,
  isReference,
  localRepos,
  referenceOpensCard,
  registryOf,
  repoOf,
  shortName,
} from "@/components/models/logic";
import { useModelsTab } from "@/components/models/ModelsTabContext";
import { ModelCardView } from "@/components/models/ModelCard";
import { MODELS_MAX_WIDTH, openModel, setCardPane } from "@/components/models/navigation";
import { ErrorBox } from "@/components/models/ErrorBox";
import { PullBanner } from "@/components/models/PullViews";
import { useWide } from "@/components/shell/Header";
import { repoName } from "@/lib/modelName";
import { useServers } from "@/lib/ServersContext";
import { font, radius, spacing, useTheme } from "@/lib/theme";
import { alert } from "@/lib/alert";

type Filter = "all" | "docker" | "hf" | "local";
// Short enough to fit a phone's width in one row.
const FILTERS = [
  { value: "all", label: "All" },
  { value: "local", label: "Installed" },
  { value: "docker", label: "Docker" },
  { value: "hf", label: "Hugging Face" },
] as const;

/** A list from the daemon: not asked yet, on its way, failed, or here. */
type Rows = SearchHit[] | "loading" | Error | null;

type Row =
  | { kind: "hit"; hit: SearchHit }
  | { kind: "local"; model: LocalModel }
  | { kind: "message"; text: string }
  | { kind: "error"; error: Error }
  | { kind: "reference"; ref: string };

type Section = { key: string; title: string; repo?: string; data: Row[] };

const asError = (e: unknown) => (e instanceof Error ? e : new Error(String(e)));

export default function Models() {
  const c = useTheme();
  const { client, active } = useServers();
  const { local, localError, reloadLocal, storeChanged, pull, startPull, pick } = useModelsTab();
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  // Answers carry what they answer, so a stale one is never shown for a new
  // server or query, while a refresh keeps the rows it is refreshing.
  const [popularAnswer, setPopularAnswer] = useState<{ server: string; rows: SearchHit[] | Error } | null>(
    null,
  );
  const [hitsAnswer, setHitsAnswer] = useState<{ text: string; rows: SearchHit[] | Error } | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [popularVersion, setPopularVersion] = useState(0);
  const [searchVersion, setSearchVersion] = useState(0);

  const textSizeKey = useTextSizeKey();
  // Wide: the list on the left, the open card on the right.
  const wide = useWide();
  const [paneName, setPaneName] = useState<string | null>(null);
  useEffect(() => (wide ? setCardPane(setPaneName) : undefined), [wide]);
  const text = query.trim();
  const reference = isReference(text) ? text : null;
  const searching = text.length >= 2 && !reference;
  const popular: Rows =
    popularAnswer && popularAnswer.server === client?.url ? popularAnswer.rows : "loading";
  const hits: Rows = !searching ? null : hitsAnswer?.text === text ? hitsAnswer.rows : "loading";

  // Models pulled or loaded from elsewhere (the CLI, the web UI) show up on return, here and in the chat's picker.
  useFocusEffect(
    useCallback(() => {
      storeChanged();
    }, [storeChanged]),
  );

  // The registries' popular models change slowly: once per server, and on pull-to-refresh.
  useEffect(() => {
    if (!client) return;
    const abort = new AbortController();
    const answer = (rows: SearchHit[] | Error) => setPopularAnswer({ server: client.url, rows });
    fetchPopular(client, { signal: abort.signal }).then(answer, (e) => {
      if (!abort.signal.aborted) answer(asError(e));
    });
    return () => abort.abort();
  }, [client, popularVersion]);

  // Search 300 ms after typing stops; a newer query aborts the older one.
  useEffect(() => {
    if (!client || !searching) return;
    const abort = new AbortController();
    const answer = (rows: SearchHit[] | Error) => setHitsAnswer({ text, rows });
    const timer = setTimeout(() => {
      search(client, text, { signal: abort.signal }).then(answer, (e) => {
        if (!abort.signal.aborted) answer(asError(e));
      });
    }, 300);
    return () => {
      clearTimeout(timer);
      abort.abort();
    };
  }, [client, searching, text, searchVersion]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    if (searching) setSearchVersion((v) => v + 1);
    else setPopularVersion((v) => v + 1);
    await reloadLocal();
    setRefreshing(false);
  }, [searching, reloadLocal]);

  // A reference to a registry the daemon searches opens its card, its tag
  // picked, so size and fit show before anything is pulled. Any other is
  // pulled as typed, after asking. One already here just opens.
  const openReference = useCallback(
    (ref: string) => {
      const repo = repoOf(ref);
      if (holdsRef(ref, local) || referenceOpensCard(ref)) {
        if (repo !== ref) pick(repo, ref);
        return openModel(repo);
      }
      alert(`Pull ${ref} to ${active?.name || active?.url || "this server"}?`, undefined, [
        { text: "Cancel", style: "cancel" },
        {
          text: "Pull",
          onPress: () => {
            startPull(ref);
            openModel(repo);
          },
        },
      ]);
    },
    [local, pick, startPull, active],
  );

  const sections = useMemo(
    () => listSections({ filter, local, localError, text, reference, searching, hits, popular }),
    [filter, local, localError, text, reference, searching, hits, popular],
  );

  const pulling = !!pull && !pull.error && !pull.done;
  const renderItem = useCallback(
    ({ item }: { item: Row }) => {
      switch (item.kind) {
        case "hit":
          return <HitRow hit={item.hit} pulled={isPulled(item.hit.name, local)} onPress={openModel} />;
        case "local":
          return (
            <View style={styles.localRow}>
              <LocalTagRow model={item.model} />
            </View>
          );
        case "error":
          return (
            <View style={styles.message}>
              <ErrorBox error={item.error} onRetry={onRefresh} />
            </View>
          );
        case "reference":
          return (
            <ReferenceRow
              reference={item.ref}
              have={holdsRef(item.ref, local)}
              busy={pulling}
              onPress={openReference}
            />
          );
        case "message":
          return (
            <View style={styles.message}>
              <Muted>{item.text}</Muted>
            </View>
          );
      }
    },
    [local, pulling, openReference, onRefresh],
  );

  if (!active) {
    return (
      <View style={[styles.center, { backgroundColor: c.background }]}>
        <Muted>Add a computer running llmman serve under Servers to browse and pull models.</Muted>
      </View>
    );
  }

  const list = (
    <>
      <Stack.Screen
        options={{
          // The web gets the field in the list instead (no native search bar there).
          ...(Platform.OS !== "web" && {
            headerSearchBarOptions: {
              placeholder: "Search, or paste a reference",
              autoCapitalize: "none",
              hideWhenScrolling: false,
              onChangeText: (e) => setQuery(e.nativeEvent.text),
              onCancelButtonPress: () => setQuery(""),
              onSearchButtonPress: (e) => {
                const t = e.nativeEvent.text.trim();
                if (isReference(t)) openReference(t);
              },
            },
          }),
        }}
      />
      <SectionList
        key={textSizeKey}
        sections={sections}
        keyExtractor={(row, i) =>
          row.kind === "hit" ? row.hit.name : row.kind === "local" ? row.model.id : `${row.kind}-${i}`
        }
        renderItem={renderItem}
        renderSectionHeader={({ section }) => <SectionHeader section={section} />}
        stickySectionHeadersEnabled={false}
        contentInsetAdjustmentBehavior="automatic"
        keyboardDismissMode="on-drag"
        keyboardShouldPersistTaps="handled"
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
        ListHeaderComponent={
          <>
            {/* The web has no native header search bar. */}
            {Platform.OS === "web" ? (
              <TextInput
                value={query}
                onChangeText={setQuery}
                onSubmitEditing={() => isReference(text) && openReference(text)}
                placeholder="Search, or paste a reference"
                placeholderTextColor={c.textMuted}
                autoCapitalize="none"
                autoCorrect={false}
                accessibilityLabel="Search models"
                style={[styles.search, { color: c.text, backgroundColor: c.surface }]}
              />
            ) : null}
            <Segments options={FILTERS} value={filter} onChange={setFilter} label="Show models from" />
            {pull ? <PullBanner pull={pull} onPress={() => openModel(repoOf(pull.ref))} /> : null}
          </>
        }
        style={{ backgroundColor: c.background }}
      />
    </>
  );
  if (!wide) return list;
  // Until one is chosen, the card shows the first installed model, else the first listed.
  const firstHit = sections.flatMap((s) => s.data).find((r) => r.kind === "hit");
  const shownName =
    paneName ?? (local[0] ? repoOf(local[0].id) : firstHit?.kind === "hit" ? firstHit.hit.name : null);
  return (
    <View style={[styles.page, { backgroundColor: c.background }]}>
      <View style={[styles.panes, { borderColor: c.border }]}>
        <View style={[styles.listPane, { borderRightColor: c.border }]}>{list}</View>
        <View style={styles.cardPane}>
          {shownName ? (
            <ModelCardView key={shownName} name={shownName} />
          ) : (
            <View style={styles.center}>
              <Muted>Choose a model to see its versions, how it fits this computer, and its README.</Muted>
            </View>
          )}
        </View>
      </View>
    </View>
  );
}

/** What the list shows for the current filter, query and answers. */
function listSections({
  filter,
  local,
  localError,
  text,
  reference,
  searching,
  hits,
  popular,
}: {
  filter: Filter;
  local: LocalModel[];
  localError: Error | null;
  text: string;
  reference: string | null;
  searching: boolean;
  hits: Rows;
  popular: Rows;
}): Section[] {
  const message = (key: string, title: string, msg: string): Section => ({
    key,
    title,
    data: [{ kind: "message", text: msg }],
  });
  const failure = (key: string, title: string, error: Error): Section => ({
    key,
    title,
    data: [{ kind: "error", error }],
  });
  const hitSection = (key: string, title: string, rows: SearchHit[]): Section => ({
    key,
    title,
    data: rows.map((hit) => ({ kind: "hit", hit })),
  });
  const out: Section[] = [];
  if (reference) {
    out.push({ key: "reference", title: "Reference", data: [{ kind: "reference", ref: reference }] });
  }

  if (filter === "local") {
    const repos = localRepos(local);
    const needle = reference ? "" : text.toLowerCase();
    const shown = repos.filter((r) => r.name.toLowerCase().includes(needle));
    if (!shown.length) {
      const empty = repos.length
        ? "Nothing pulled matches the search."
        : "Nothing pulled yet. Search to find a model.";
      out.push(
        localError
          ? failure("local-error", "On this computer", localError)
          : message("local-empty", "On this computer", empty),
      );
    }
    for (const r of shown) {
      out.push({
        key: r.name,
        title: shortName(r.name),
        repo: r.name,
        data: r.local.map((model) => ({ kind: "local", model })),
      });
    }
    return out;
  }

  const shown = (rows: SearchHit[]) => rows.filter((h) => filter === "all" || registryOf(h.name) === filter);

  // What this computer has comes first, before what it could pull.
  if (filter === "all" && !searching && !reference && local.length) {
    out.push({
      key: "installed",
      title: "On this computer",
      data: localRepos(local).map((r) => ({
        kind: "hit" as const,
        hit: { name: r.name },
      })),
    });
  }

  if (searching) {
    const title = `Results for “${text}”`;
    if (hits === "loading" || hits === null) return [...out, message("searching", title, "Searching…")];
    if (hits instanceof Error) return [...out, failure("search-error", title, hits)];
    const rows = shown(hits);
    return [
      ...out,
      rows.length ? hitSection("results", title, rows) : message("none", title, "No models found."),
    ];
  }
  if (reference) return out;
  if (text.length === 1) out.push(message("too-short", "Search", "Type at least 2 characters to search."));

  if (popular === "loading" || popular === null) {
    return [message("popular-loading", "Popular", "Loading popular models…")];
  }
  if (popular instanceof Error) return [failure("popular-error", "Popular", popular)];
  const rows = shown(popular);
  if (!rows.length) {
    return [
      message("popular-none", "Popular", "No popular models from this registry right now. Search above."),
    ];
  }
  for (const [registry, title] of [
    ["docker", "Featured on Docker Hub"],
    ["hf", "Popular GGUF on Hugging Face"],
  ] as const) {
    const group = rows.filter((h) => registryOf(h.name) === registry);
    if (group.length) out.push(hitSection(registry, title, group));
  }
  return out;
}

function SectionHeader({ section }: { section: Section }) {
  const c = useTheme();
  const large = useLargeText();
  const repo = section.repo;
  if (!repo) {
    return (
      <Text style={[styles.header, { color: c.textMuted }]} accessibilityRole="header">
        {section.title}
      </Text>
    );
  }
  return (
    <Pressable
      onPress={() => openModel(repo)}
      accessibilityRole="button"
      accessibilityLabel={`${repo}, open its card`}
      style={[styles.repoHeader, { borderTopColor: c.border }]}
    >
      <Avatar name={repo} size={28} />
      <View style={styles.repoText}>
        <Text style={[styles.repoName, { color: c.text }]} numberOfLines={large ? undefined : 1}>
          {repoName(repo)}
        </Text>
        <Text style={[styles.repoSub, { color: c.textMuted }]} numberOfLines={large ? undefined : 1}>
          {repo}
        </Text>
      </View>
      <Icon name={{ ios: "chevron.right", android: "chevron_right" }} size={14} color={c.textMuted} />
    </Pressable>
  );
}

/** A typed reference: open its card, or (another registry) pull it after asking. */
function ReferenceRow({
  reference,
  have,
  busy,
  onPress,
}: {
  reference: string;
  have: boolean;
  busy: boolean;
  onPress: (ref: string) => void;
}) {
  const c = useTheme();
  const opens = have || referenceOpensCard(reference);
  const blocked = !opens && busy;
  const title = `${opens ? "Open" : "Pull"} ${shortName(reference)}`;
  const sub = have ? "On this computer" : blocked ? "Another pull is running." : reference;
  return (
    <Pressable
      onPress={() => onPress(reference)}
      disabled={blocked}
      accessibilityRole="button"
      accessibilityLabel={`${opens ? "Open" : "Pull"} ${reference}${have ? ", on this computer" : ""}`}
      accessibilityState={{ disabled: blocked }}
      style={[styles.reference, { backgroundColor: c.surface, opacity: blocked ? 0.5 : 1 }]}
    >
      <Icon
        name={
          opens
            ? { ios: "arrow.right.circle.fill", android: "arrow_circle_right" }
            : { ios: "arrow.down.circle.fill", android: "download" }
        }
        size={22}
        color={c.accent}
      />
      <View style={styles.repoText}>
        <Text style={[styles.repoName, { color: c.text }]}>{title}</Text>
        <Text style={[styles.repoSub, { color: have ? c.success : c.textMuted }]} numberOfLines={2}>
          {sub}
        </Text>
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  page: { flex: 1, alignItems: "center" },
  panes: { flex: 1, flexDirection: "row", width: "100%", maxWidth: MODELS_MAX_WIDTH },
  listPane: { width: 420, borderRightWidth: StyleSheet.hairlineWidth },
  cardPane: { flex: 1 },
  search: {
    minHeight: 40,
    borderRadius: radius.md,
    paddingHorizontal: spacing.md,
    marginHorizontal: spacing.lg,
    marginTop: spacing.sm,
    fontSize: font.body,
    outlineWidth: 0,
  },
  center: { flex: 1, alignItems: "center", justifyContent: "center", padding: spacing.xl },
  header: {
    fontSize: font.small,
    fontWeight: "600",
    textTransform: "uppercase",
    letterSpacing: 0.5,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.lg,
    paddingBottom: spacing.xs,
  },
  repoHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.xs,
    marginTop: spacing.sm,
    borderTopWidth: StyleSheet.hairlineWidth,
  },
  repoText: { flex: 1, gap: 2 },
  repoName: { fontSize: font.body, fontWeight: "600" },
  repoSub: { fontSize: font.small },
  localRow: { paddingLeft: spacing.lg + 28 + spacing.md, paddingRight: spacing.lg },
  message: { paddingHorizontal: spacing.lg, paddingVertical: spacing.md },
  reference: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    marginHorizontal: spacing.lg,
    padding: spacing.md,
    borderRadius: radius.md,
  },
});
