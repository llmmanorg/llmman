// Hosted providers (Anthropic, OpenAI, OpenRouter…): which ones the
// computer has a key for, and a way to add one. Keys are stored on the
// computer running llmman, never on this device.

import { router, useFocusEffect } from "expo-router";
import { useCallback, useMemo, useState } from "react";
import { Platform, Pressable, SectionList, StyleSheet, Text, TextInput, View } from "react-native";

import type { Provider } from "@/api/client";
import { explain } from "@/api/errors";
import { hostLabel } from "@/api/serverUrl";
import { Icon } from "@/components/Icon";
import { Notice } from "@/components/ui";
import { providerSections } from "@/lib/providerGroups";
import { useServers } from "@/lib/ServersContext";
import { font, radius, readable, spacing, useTheme, useTextSizeKey } from "@/lib/theme";

export default function Providers() {
  const c = useTheme();
  const sizeKey = useTextSizeKey();
  const { client, active } = useServers();
  const [list, setList] = useState<Provider[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  useFocusEffect(
    useCallback(() => {
      if (!client || !active) return;
      const abort = new AbortController();
      client
        .listProviders({ signal: abort.signal })
        .then((l) => {
          setList(l);
          setError(null);
        })
        .catch((e) => !abort.signal.aborted && setError(explain(e, active).title));
      return () => abort.abort();
    }, [client, active]),
  );

  // Connected ones (a key is set), then the popular ones, then all the rest by name.
  const sections = useMemo(() => providerSections(list ?? [], query), [list, query]);

  const computer = active?.name || (active ? hostLabel(active.url) : "the computer");

  return (
    <SectionList
      key={sizeKey}
      sections={sections}
      stickySectionHeadersEnabled={false}
      keyExtractor={(p) => p.id}
      renderSectionHeader={({ section }) => (
        <Text style={[styles.section, { color: c.textMuted }]} accessibilityRole="header">
          {section.title}
        </Text>
      )}
      contentInsetAdjustmentBehavior="automatic"
      keyboardShouldPersistTaps="handled"
      contentContainerStyle={[styles.content, readable]}
      style={{ backgroundColor: c.background }}
      ListHeaderComponent={
        <View style={styles.head}>
          <Text style={[styles.lead, { color: c.textMuted }]}>
            Chat with hosted models through {computer}. An API key you add is stored on {computer} (in its
            llmman.conf), not on this device, and requests go from there.
          </Text>
          <TextInput
            value={query}
            onChangeText={setQuery}
            placeholder="Search providers"
            placeholderTextColor={c.textMuted}
            autoCapitalize="none"
            autoCorrect={false}
            clearButtonMode="while-editing"
            accessibilityLabel="Search providers"
            style={[styles.search, { color: c.text, backgroundColor: c.surface }]}
          />
          {error ? (
            <Notice tone="error" title="Couldn't list providers">
              {error}
            </Notice>
          ) : null}
        </View>
      }
      ItemSeparatorComponent={() => <View style={[styles.sep, { backgroundColor: c.border }]} />}
      renderItem={({ item }) => {
        const status = item.key_usable
          ? "Connected"
          : item.key_set
            ? "Key set, not usable from here"
            : "No key";
        return (
          <Pressable
            onPress={() =>
              router.push({
                pathname: "/provider-key",
                params: {
                  id: item.id,
                  name: item.name,
                  env: item.key_env ?? "",
                  set: item.key_set ? "1" : "",
                },
              })
            }
            accessibilityRole="button"
            accessibilityLabel={`${item.name}, ${status}`}
            accessibilityHint={item.key_set ? "Replace or remove the key" : "Add a key"}
            style={({ pressed }) => [styles.row, { opacity: pressed ? 0.6 : 1 }]}
          >
            <View style={{ flex: 1, gap: 2 }}>
              <Text style={{ color: c.text, fontSize: font.body }} numberOfLines={1}>
                {item.name}
              </Text>
              <View style={styles.status}>
                {/* A ring for no key: a grey dot is lost on a dark background. */}
                <View
                  style={[
                    styles.dot,
                    item.key_usable || item.key_set
                      ? { backgroundColor: item.key_usable ? c.success : c.warning }
                      : { borderWidth: 1, borderColor: c.textMuted },
                  ]}
                />
                <Text style={{ color: c.textMuted, fontSize: font.small }}>
                  {status}
                  {item.models ? ` · ${item.models} models` : ""}
                </Text>
              </View>
            </View>
            <Icon name={{ ios: "chevron.right", android: "chevron_right" }} color={c.textMuted} size={14} />
          </Pressable>
        );
      }}
      ListEmptyComponent={
        list && !sections.length ? (
          <Text style={[styles.lead, { color: c.textMuted, paddingHorizontal: spacing.lg }]}>
            No provider matches.
          </Text>
        ) : null
      }
    />
  );
}

const styles = StyleSheet.create({
  section: {
    fontSize: font.small + 1,
    fontWeight: "600",
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.lg,
    paddingBottom: spacing.xs,
  },
  content: { paddingBottom: spacing.xl },
  head: { padding: spacing.lg, gap: spacing.md },
  lead: { fontSize: font.small + 1, lineHeight: 20 },
  search: {
    minHeight: 40,
    borderRadius: radius.md,
    paddingHorizontal: spacing.md,
    fontSize: font.body,
    ...(Platform.OS === "web" ? { outlineWidth: 0 } : {}),
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    paddingHorizontal: spacing.lg,
    minHeight: 56,
  },
  status: { flexDirection: "row", alignItems: "center", gap: 6 },
  dot: { width: 8, height: 8, borderRadius: 4 },
  sep: { height: StyleSheet.hairlineWidth, marginLeft: spacing.lg },
});
