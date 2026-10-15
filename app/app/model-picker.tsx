// Choose a model: local ones on the computer first, then its media
// generation models ("Generate"), then hosted providers with a usable key
// (each provider's models fetched when it is opened). On a wide screen it
// is a popover under the header's title, driven by the keyboard too
// (arrows, Enter; Enter in the filter takes the first match; Escape
// closes, WebChrome); on a phone, a sheet.

import { router, Stack, useLocalSearchParams } from "expo-router";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  ActivityIndicator,
  Platform,
  Pressable,
  SectionList,
  StyleSheet,
  Text,
  TextInput,
  useWindowDimensions,
  View,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { remoteRef, splitRemoteRef } from "@/api/client";
import { explain } from "@/api/errors";
import { Icon } from "@/components/Icon";
import { formatBytes } from "@/components/models/logic";
import { useWide } from "@/components/shell/Header";
import { Notice } from "@/components/ui";
import { alert } from "@/lib/alert";
import { getConversationState, setEffort, setModel, useConversation } from "@/lib/chatStore";
import { KIND_LABELS, type MediaKind } from "@/lib/media";
import { effortLabel, hostedName } from "@/lib/modelName";
import { pickerAnchor, popoverFrame } from "@/lib/pickerAnchor";
import { choosable, selectedLocation, step } from "@/lib/pickerLocation";
import { useModels } from "@/lib/ModelsContext";
import { useServers } from "@/lib/ServersContext";
import { font, radius, readable, spacing, useTheme, useTextSizeKey } from "@/lib/theme";

type Row =
  | { kind: "local"; id: string; loaded: boolean; size?: number }
  | { kind: "media"; id: string; loaded: boolean; kinds: MediaKind[]; size?: number }
  | { kind: "provider"; id: string; name: string; count: number; open: boolean; loading: boolean }
  | { kind: "hosted"; ref: string; model: string; name?: string };

/** A list older than this is fetched again when the picker opens, as the web UI's is. */
const STALE_MS = 5000;

export default function ModelPicker() {
  const c = useTheme();
  const sizeKey = useTextSizeKey();
  const wide = useWide();
  const insets = useSafeAreaInsets();
  const window = useWindowDimensions();
  const { active } = useServers();
  const { conversation } = useLocalSearchParams<{ conversation?: string }>();
  const m = useModels();
  const [query, setQuery] = useState("");
  // The chosen hosted model's provider starts open, so its effort levels show under it.
  const [open, setOpen] = useState<Set<string>>(() => {
    const chosen = splitRemoteRef(conversation ? getConversationState(conversation).conv?.model : undefined);
    return new Set(chosen ? [chosen.provider] : []);
  });
  const [loadingProvider, setLoadingProvider] = useState<string | null>(null);
  const [unloading, setUnloading] = useState<string | null>(null);
  // The row the arrow keys are on (web).
  const [focus, setFocus] = useState<string | null>(null);
  const q = query.trim().toLowerCase();

  // A stale list is fetched again on open; the web has no pull-to-refresh.
  const { refresh, loadedAt } = m;
  useEffect(() => {
    if (Date.now() - loadedAt > STALE_MS) void refresh();
    // Once, on open.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const matches = (ref: string, ...more: string[]) =>
    !q || [ref, m.displayName(ref), ...more].some((s) => s.toLowerCase().includes(q));

  const sections = useMemo(() => {
    const local: Row[] = m.local
      .filter((l) => !l.kinds.length && matches(l.id))
      .map((l) => ({ kind: "local", id: l.id, loaded: l.loaded, size: l.size }));
    // A media model also matches what it makes ("video").
    const media: Row[] = m.local
      .filter((l) => l.kinds.length && matches(l.id, ...l.kinds))
      .map((l) => ({ kind: "media", id: l.id, loaded: l.loaded, kinds: l.kinds, size: l.size }));
    const hosted: Row[] = [];
    for (const p of m.providers) {
      const models = (p.models ?? []).filter((x) => matches(remoteRef(p.id, x.id), x.id));
      const isOpen = open.has(p.id) || (!!q && models.length > 0);
      if (q && !models.length && !p.name.toLowerCase().includes(q)) continue;
      hosted.push({
        kind: "provider",
        id: p.id,
        name: p.name,
        count: p.count,
        open: isOpen,
        loading: loadingProvider === p.id,
      });
      if (isOpen)
        for (const x of models)
          hosted.push({
            kind: "hosted",
            ref: remoteRef(p.id, x.id),
            model: x.id,
            ...(typeof x.name === "string" ? { name: x.name } : {}),
          });
    }
    const out = [{ title: "On this computer", data: local }];
    if (media.length) out.push({ title: "Generate", data: media });
    if (m.providers.length) out.push({ title: "Hosted", data: hosted });
    return out;
    // `matches` reads `q` and the names, which follow these.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [m.local, m.providers, q, open, loadingProvider, m.displayName]);

  const choose = (ref: string) => {
    m.select(ref);
    if (conversation) setModel(conversation, ref);
    router.back();
  };

  const unload = async (ref: string) => {
    setUnloading(ref);
    try {
      await m.unload(ref);
    } catch (e) {
      alert(`Couldn't unload ${m.displayName(ref)}`, active ? explain(e, active).title : String(e));
    } finally {
      setUnloading(null);
    }
  };

  const toggle = async (id: string) => {
    const next = new Set(open);
    if (next.has(id)) next.delete(id);
    else {
      next.add(id);
      if (m.providers.find((p) => p.id === id)?.models === null) {
        setLoadingProvider(id);
        await m.loadProvider(id);
        setLoadingProvider(null);
      }
    }
    setOpen(next);
  };

  const conv = useConversation(conversation).conv;
  const current = conversation ? conv?.model : m.selected;
  const generating = m.mediaKinds(current ?? "").length > 0;
  const hasMedia = m.local.some((l) => l.kinds.length);

  // A hosted model's thinking levels, from its provider's model list
  // (fetched here if the picker has not opened that provider yet).
  const remote = splitRemoteRef(current);
  const provider = remote ? m.providers.find((p) => p.id === remote.provider) : undefined;
  const levels = provider?.models?.find((x) => x.id === remote?.model)?.thinking ?? [];
  const { loadProvider } = m;
  useEffect(() => {
    if (provider && provider.models === null) void loadProvider(provider.id);
  }, [provider, loadProvider]);

  // The chosen hosted model's thinking levels, right under its row: they are its own setting.
  const effortRow =
    conversation && levels.length ? (
      <View style={[styles.levels, styles.indent]} accessibilityRole="radiogroup" accessibilityLabel="Effort">
        <Text style={{ color: c.textMuted, fontSize: font.small }}>Effort</Text>
        {[undefined, ...levels].map((level) => {
          const on = conv?.effort === level;
          return (
            <Pressable
              key={level ?? "default"}
              onPress={() => setEffort(conversation, level)}
              accessibilityRole="radio"
              accessibilityState={{ checked: on }}
              style={[
                styles.level,
                { borderColor: on ? c.accent : c.border, backgroundColor: on ? c.accent : "transparent" },
              ]}
            >
              <Text style={{ color: on ? c.onAccent : c.text, fontSize: font.small + 1, fontWeight: "500" }}>
                {effortLabel(level)}
              </Text>
            </Pressable>
          );
        })}
      </View>
    ) : null;

  // Open scrolled to the chosen model, with its effort levels under it in
  // view: once, when its row is first listed (a provider's models load late).
  const list = useRef<SectionList<Row>>(null);
  const scrolled = useRef(false);
  const target = selectedLocation(sections, current);
  const sectionIndex = target?.sectionIndex;
  const itemIndex = target?.itemIndex;
  useEffect(() => {
    if (sectionIndex === undefined || itemIndex === undefined || scrolled.current) return;
    scrolled.current = true;
    const t = setTimeout(
      () => list.current?.scrollToLocation({ sectionIndex, itemIndex, viewPosition: 0.3, animated: false }),
      50,
    );
    return () => clearTimeout(t);
  }, [sectionIndex, itemIndex]);

  // The keyboard (web): arrows move, Enter picks the row they are on.
  const rows = choosable(sections);
  const latest = useRef({ rows, focus });
  useEffect(() => {
    latest.current = { rows, focus };
  });
  useEffect(() => {
    if (Platform.OS !== "web") return;
    const onKey = (e: KeyboardEvent) => {
      const { rows: all, focus: at } = latest.current;
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        const next = step(
          all.map((r) => r.ref),
          at,
          e.key === "ArrowDown" ? 1 : -1,
        );
        setFocus(next);
        const where = all.find((r) => r.ref === next);
        if (where)
          list.current?.scrollToLocation({
            sectionIndex: where.sectionIndex,
            itemIndex: where.itemIndex,
            viewPosition: 0.5,
            animated: false,
          });
      } else if (e.key === "Enter" && at) {
        // Not the filter's own Enter as well: one choice, one close.
        e.preventDefault();
        e.stopPropagation();
        choose(at);
      }
    };
    // Capture: the filter field would otherwise keep the arrows to itself.
    document.addEventListener("keydown", onKey, true);
    return () => document.removeEventListener("keydown", onKey, true);
    // `choose` only reads what does not change while the picker is open.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const goTo = (path: "/providers" | "/models") => {
    router.back();
    router.navigate(path);
  };

  const linkRow = (
    icon: Parameters<typeof Icon>[0]["name"],
    label: string,
    onPress: () => void,
    hint?: string,
  ) => (
    <Pressable
      onPress={onPress}
      accessibilityRole="button"
      accessibilityHint={hint}
      style={({ pressed }) => [styles.row, styles.link, { opacity: pressed ? 0.6 : 1 }]}
    >
      <Icon name={icon} color={c.textMuted} size={18} />
      <Text style={[styles.name, { color: c.text }]}>{label}</Text>
    </Pressable>
  );

  const header = (
    <View style={[styles.head, { borderBottomColor: c.border }]}>
      <View style={[styles.searchRow, readable]}>
        <TextInput
          value={query}
          onChangeText={(t) => {
            setQuery(t);
            setFocus(null);
          }}
          onSubmitEditing={() => {
            const first = latest.current.focus ?? latest.current.rows[0]?.ref;
            if (first) choose(first);
          }}
          autoFocus={Platform.OS === "web"}
          placeholder="Filter models"
          placeholderTextColor={c.textMuted}
          autoCapitalize="none"
          autoCorrect={false}
          clearButtonMode="while-editing"
          returnKeyType="go"
          accessibilityLabel="Filter models"
          style={[styles.search, { color: c.text, backgroundColor: c.surface }]}
        />
        <Pressable
          onPress={() => void m.refresh()}
          disabled={m.loading}
          accessibilityRole="button"
          accessibilityLabel="Refresh models"
          hitSlop={6}
          style={({ pressed }) => [styles.refresh, { opacity: pressed ? 0.6 : 1 }]}
        >
          {m.loading ? (
            <ActivityIndicator size="small" />
          ) : (
            <Icon name={{ ios: "arrow.clockwise", android: "refresh" }} color={c.textMuted} size={18} />
          )}
        </Pressable>
      </View>
    </View>
  );

  const sheet = (
    <View
      key={sizeKey}
      style={
        wide
          ? [styles.card, { backgroundColor: c.background, borderColor: c.border }]
          : { flex: 1, backgroundColor: c.background }
      }
    >
      <Stack.Screen options={{ title: "Model" }} />
      {header}
      {m.error ? (
        <View style={{ paddingHorizontal: spacing.lg, paddingTop: spacing.sm }}>
          <Notice tone="error" title="Couldn't list models">
            {m.error}
          </Notice>
        </View>
      ) : null}
      <SectionList
        ref={list}
        sections={sections}
        style={wide ? styles.cardList : undefined}
        // Rows below the rendered window have no layout yet: retry once they do.
        onScrollToIndexFailed={() => {
          if (sectionIndex === undefined || itemIndex === undefined) return;
          setTimeout(
            () =>
              list.current?.scrollToLocation({ sectionIndex, itemIndex, viewPosition: 0.3, animated: false }),
            100,
          );
        }}
        keyExtractor={(r) => (r.kind === "hosted" ? `h:${r.ref}` : `${r.kind}:${r.id}`)}
        keyboardShouldPersistTaps="handled"
        stickySectionHeadersEnabled={false}
        contentContainerStyle={[
          readable,
          // Clear of the home indicator on a phone.
          { paddingBottom: wide ? spacing.sm : insets.bottom + spacing.lg },
        ]}
        refreshing={Platform.OS === "web" ? false : m.loading}
        onRefresh={Platform.OS === "web" ? undefined : m.refresh}
        renderSectionHeader={({ section }) => (
          <Text style={[styles.section, { color: c.textMuted }]} accessibilityRole="header">
            {section.title}
          </Text>
        )}
        renderSectionFooter={({ section }) =>
          section.data.length === 0 && section.title === "On this computer" && !m.loading ? (
            <Text style={[styles.hint, { color: c.textMuted }]}>
              {q
                ? "No local model matches."
                : hasMedia
                  ? "No chat models on this computer, only media generation models."
                  : "No chat models on this computer yet."}
            </Text>
          ) : null
        }
        ListFooterComponent={
          <View style={[styles.footer, { borderTopColor: c.border }]}>
            {linkRow(
              { ios: "arrow.down.circle", android: "download" },
              "Pull a model…",
              () => goTo("/models"),
              "Find and download models on the Models screen",
            )}
            {linkRow(
              { ios: "key", android: "key" },
              m.providers.length ? "Providers…" : "Use a hosted model (Anthropic, OpenAI…)",
              () => goTo("/providers"),
              "Add an API key for Anthropic, OpenAI and others on the computer",
            )}
            {conversation
              ? // The chat's own settings live here, so the header stays menu · model · new chat.
                linkRow(
                  { ios: "slider.horizontal.3", android: "tune" },
                  generating ? "Generation settings…" : "Chat settings…",
                  () =>
                    router.replace({
                      pathname: generating ? "/generate-settings" : "/chat-settings",
                      params: { conversation },
                    }),
                  generating
                    ? "Size, seconds, steps, seed, guidance and negative prompt"
                    : "System prompt, temperature and max tokens",
                )
              : null}
          </View>
        }
        renderItem={({ item }) => {
          if (item.kind === "provider") {
            return (
              <Pressable
                onPress={() => toggle(item.id)}
                accessibilityRole="button"
                accessibilityState={{ expanded: item.open }}
                accessibilityLabel={`${item.name}, ${item.count} models`}
                style={styles.row}
              >
                <Text style={[styles.name, { color: c.text }]}>{item.name}</Text>
                <Text style={{ color: c.textMuted, fontSize: font.small }}>{item.count}</Text>
                {item.loading ? (
                  <ActivityIndicator size="small" />
                ) : (
                  <Icon
                    name={
                      item.open
                        ? { ios: "chevron.up", android: "expand_less" }
                        : { ios: "chevron.down", android: "expand_more" }
                    }
                    color={c.textMuted}
                    size={14}
                  />
                )}
              </Pressable>
            );
          }
          const onComputer = item.kind === "local" || item.kind === "media";
          const ref = onComputer ? item.id : item.ref;
          const title = onComputer ? m.displayName(ref) : (item.name ?? hostedName(item.model));
          const selected = ref === (current ?? "");
          const focused = ref === focus;
          const makes = item.kind === "media" ? item.kinds.map((k) => KIND_LABELS[k]).join(" · ") : "";
          const sub = onComputer
            ? [makes, item.loaded && "Loaded", item.size ? formatBytes(item.size) : ""]
                .filter(Boolean)
                .join(" · ")
            : // A hosted model's raw id, when the name is not just it.
              title.toLowerCase() !== item.model.toLowerCase()
              ? item.model
              : "";
          return (
            <>
              <View style={styles.rowWrap}>
                <Pressable
                  onPress={() => choose(ref)}
                  onHoverIn={() => setFocus(ref)}
                  accessibilityRole="button"
                  accessibilityState={{ selected }}
                  accessibilityLabel={[
                    title,
                    makes && `generates ${makes.replace(/ · /g, ", ")}`,
                    onComputer && item.loaded && "loaded",
                  ]
                    .filter(Boolean)
                    .join(", ")}
                  accessibilityHint={ref}
                  // The full reference, on hover (web).
                  {...(Platform.OS === "web" ? ({ title: ref } as object) : {})}
                  style={({ pressed }) => [
                    styles.row,
                    styles.grow,
                    item.kind === "hosted" && styles.indent,
                    {
                      opacity: pressed ? 0.6 : 1,
                      backgroundColor: focused && Platform.OS === "web" ? c.surface : "transparent",
                    },
                  ]}
                >
                  <View style={styles.names}>
                    <Text style={[styles.title, { color: c.text }]} numberOfLines={1}>
                      {title}
                    </Text>
                    {sub ? (
                      <View style={styles.sub}>
                        {onComputer && item.loaded ? (
                          <View style={[styles.dot, { backgroundColor: c.success }]} />
                        ) : null}
                        <Text
                          style={{ color: c.textMuted, fontSize: font.small, flexShrink: 1 }}
                          numberOfLines={1}
                        >
                          {sub}
                        </Text>
                      </View>
                    ) : null}
                  </View>
                  {selected ? (
                    <Icon name={{ ios: "checkmark", android: "check" }} color={c.accent} size={18} />
                  ) : null}
                </Pressable>
                {onComputer && item.loaded ? (
                  <Pressable
                    onPress={() => void unload(ref)}
                    disabled={unloading === ref}
                    accessibilityRole="button"
                    accessibilityLabel={`Unload ${title}`}
                    accessibilityHint="Frees its memory on the computer; it loads again when used"
                    hitSlop={6}
                    style={({ pressed }) => [
                      styles.unload,
                      { borderColor: c.border, opacity: pressed || unloading === ref ? 0.5 : 1 },
                    ]}
                  >
                    <Text style={{ color: c.textMuted, fontSize: font.small, fontWeight: "600" }}>
                      Unload
                    </Text>
                  </Pressable>
                ) : null}
              </View>
              {selected && item.kind === "hosted" ? effortRow : null}
            </>
          );
        }}
      />
    </View>
  );
  if (!wide) return sheet;
  // A popover under the header's title, as on ChatGPT; a click outside closes it.
  const frame = popoverFrame(pickerAnchor(), window, 420);
  return (
    <Pressable
      style={styles.backdrop}
      onPress={() => router.back()}
      accessibilityLabel="Close"
      accessibilityRole="button"
    >
      <Pressable
        style={[
          styles.cardWrap,
          { left: frame.left, top: frame.top, width: frame.width, maxHeight: Math.min(frame.maxHeight, 620) },
        ]}
        onPress={(e) => e.stopPropagation()}
        accessible={false}
      >
        {sheet}
      </Pressable>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  backdrop: { flex: 1, backgroundColor: "transparent" },
  cardWrap: { position: "absolute" },
  card: {
    borderRadius: 14,
    borderWidth: StyleSheet.hairlineWidth,
    overflow: "hidden",
    maxHeight: "100%",
    shadowColor: "#000",
    shadowOpacity: 0.16,
    shadowRadius: 24,
    shadowOffset: { width: 0, height: 8 },
  },
  cardList: { flexGrow: 0 },
  head: { borderBottomWidth: StyleSheet.hairlineWidth },
  searchRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm + 2,
  },
  search: {
    flex: 1,
    minHeight: 38,
    borderRadius: radius.md,
    paddingHorizontal: spacing.md,
    fontSize: font.body,
    ...(Platform.OS === "web" ? ({ outlineStyle: "none" } as object) : {}),
  },
  refresh: { width: 34, height: 34, alignItems: "center", justifyContent: "center" },
  section: {
    fontSize: font.small,
    fontWeight: "600",
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.xs,
  },
  hint: { paddingHorizontal: spacing.lg, paddingVertical: spacing.sm },
  rowWrap: { flexDirection: "row", alignItems: "center", paddingRight: spacing.md },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    paddingHorizontal: spacing.lg,
    minHeight: 48,
    paddingVertical: spacing.sm,
  },
  grow: { flex: 1, borderRadius: radius.md },
  link: { gap: spacing.md, minHeight: 44 },
  footer: { borderTopWidth: StyleSheet.hairlineWidth, marginTop: spacing.md, paddingTop: spacing.xs },
  levels: {
    flexDirection: "row",
    flexWrap: "wrap",
    alignItems: "center",
    gap: spacing.sm,
    paddingRight: spacing.lg,
    paddingBottom: spacing.md,
  },
  level: {
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: radius.pill,
    paddingHorizontal: spacing.md,
    paddingVertical: 6,
  },
  indent: { paddingLeft: spacing.xl + spacing.md },
  names: { flex: 1, gap: 3 },
  title: { fontSize: font.body },
  name: { flex: 1, fontSize: font.body },
  sub: { flexDirection: "row", alignItems: "center", gap: 6 },
  dot: { width: 6, height: 6, borderRadius: 3 },
  unload: {
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: radius.pill,
    paddingHorizontal: spacing.md,
    paddingVertical: 4,
  },
});
