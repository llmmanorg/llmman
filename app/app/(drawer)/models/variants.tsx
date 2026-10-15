// The version picker, as a sheet over a model's card: every tag or
// quantization smallest first, with its size and fit. Repos with many
// versions get size chips (`0.8b`, `27b`…) and a filter.

import { router, useLocalSearchParams } from "expo-router";
import { useMemo, useState } from "react";
import { FlatList, Pressable, StyleSheet, TextInput, View } from "react-native";

import type { Variant } from "@/api/search";
import { Icon } from "@/components/Icon";
import { Muted, Segments, useTextSizeKey } from "@/components/models/bits";
import {
  filterVariants,
  initialVariant,
  MANY_VARIANTS,
  sizeGroups,
  sortBySize,
  variantLabel,
} from "@/components/models/logic";
import { useModelsTab } from "@/components/models/ModelsTabContext";
import { VariantLine } from "@/components/models/VariantLine";
import { font, radius, spacing, useTheme } from "@/lib/theme";

const ALL = "all";
const CHECK_WIDTH = 20;

export default function Variants() {
  const c = useTheme();
  const { name = "" } = useLocalSearchParams<{ name: string }>();
  const { card, picked, pick, local, memory } = useModelsTab();
  const entry = card(name);
  const variants = useMemo(
    () => (entry && typeof entry === "object" && !(entry instanceof Error) ? entry.variants : []),
    [entry],
  );
  const chosen = initialVariant(variants, memory, picked(name));
  const many = variants.length > MANY_VARIANTS;
  const groups = useMemo(() => (many ? sizeGroups(variants) : []), [many, variants]);
  const [group, setGroup] = useState<string>(ALL);
  const [filter, setFilter] = useState("");
  const shown = useMemo(
    () => sortBySize(filterVariants(variants, group === ALL ? null : group, filter)),
    [variants, group, filter],
  );

  const choose = (v: Variant) => {
    pick(name, v.name);
    router.back();
  };

  return (
    <View style={[styles.screen, { backgroundColor: c.background }]}>
      {many ? (
        <View style={styles.tools}>
          {groups.length ? (
            <Segments
              options={[{ value: ALL, label: "All sizes" }, ...groups.map((g) => ({ value: g, label: g }))]}
              value={group}
              onChange={setGroup}
              label="Model size"
            />
          ) : null}
          <TextInput
            value={filter}
            onChangeText={setFilter}
            placeholder="Filter, e.g. q4"
            placeholderTextColor={c.textMuted}
            autoCapitalize="none"
            autoCorrect={false}
            clearButtonMode="while-editing"
            accessibilityLabel="Filter versions"
            style={[styles.filter, { color: c.text, backgroundColor: c.surface, borderColor: c.border }]}
          />
        </View>
      ) : null}
      <FlatList
        key={useTextSizeKey()}
        data={shown}
        keyExtractor={(v) => v.name}
        keyboardShouldPersistTaps="handled"
        ListEmptyComponent={
          <View style={styles.empty}>
            <Muted>No version matches.</Muted>
          </View>
        }
        renderItem={({ item: v }) => {
          const on = v.name === chosen?.name;
          return (
            <Pressable
              onPress={() => choose(v)}
              accessibilityRole="radio"
              accessibilityState={{ checked: on }}
              accessibilityLabel={variantLabel(v, local, memory)}
              style={({ pressed }) => [styles.row, { backgroundColor: pressed ? c.surface : "transparent" }]}
            >
              <VariantLine
                variant={v}
                local={local}
                memory={memory}
                strong={on}
                leadWidth={CHECK_WIDTH}
                lead={
                  <View style={styles.check}>
                    {on ? (
                      <Icon name={{ ios: "checkmark", android: "check" }} size={18} color={c.accent} />
                    ) : null}
                  </View>
                }
              />
            </Pressable>
          );
        }}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1 },
  tools: { gap: spacing.xs, paddingBottom: spacing.sm },
  filter: {
    marginHorizontal: spacing.lg,
    minHeight: 40,
    paddingHorizontal: spacing.md,
    borderRadius: radius.md,
    borderWidth: StyleSheet.hairlineWidth,
    fontSize: font.body,
  },
  empty: { padding: spacing.lg },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    minHeight: 48,
    paddingHorizontal: spacing.lg,
  },
  check: { width: CHECK_WIDTH, alignItems: "center" },
});
