// A model in the popular list or the search results.

import { memo } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";

import type { SearchHit } from "@/api/search";
import { Icon } from "@/components/Icon";
import { font, spacing, useTheme } from "@/lib/theme";
import { Avatar } from "./Avatar";
import { Chip, RegistryBadge } from "./bits";
import { repoName } from "@/lib/modelName";
import { counted, formatCount, ownerOf, registryName } from "./logic";

export const HitRow = memo(function HitRow({
  hit,
  pulled,
  onPress,
}: {
  hit: SearchHit;
  pulled: boolean;
  onPress: (name: string) => void;
}) {
  const c = useTheme();
  const owner = ownerOf(hit.name);
  const registry = registryName(hit.name);
  const label = [
    repoName(hit.name),
    owner && `by ${owner}`,
    registry && `on ${registry}`,
    hit.pulls != null && counted(hit.pulls, "pull"),
    hit.likes != null && counted(hit.likes, "like"),
    pulled && "installed",
  ]
    .filter(Boolean)
    .join(", ");
  return (
    <Pressable
      onPress={() => onPress(hit.name)}
      accessibilityRole="button"
      accessibilityLabel={label}
      style={({ pressed }) => [styles.row, { backgroundColor: pressed ? c.surface : c.background }]}
    >
      <Avatar name={hit.name} />
      <View style={styles.text}>
        <View style={styles.title}>
          <Text style={[styles.name, { color: c.text }]} numberOfLines={2}>
            {repoName(hit.name)}
          </Text>
          {pulled ? <Chip tone="accent">Installed</Chip> : null}
        </View>
        <View style={styles.sub}>
          <RegistryBadge name={hit.name} />
          <Text style={[styles.subText, { color: c.textMuted }]} numberOfLines={1}>
            {owner}
          </Text>
          {hit.pulls != null ? <Stat icon="pulls" text={formatCount(hit.pulls)} /> : null}
          {hit.likes != null ? <Stat icon="likes" text={formatCount(hit.likes)} /> : null}
        </View>
      </View>
      <Icon name={{ ios: "chevron.right", android: "chevron_right" }} size={14} color={c.textMuted} />
    </Pressable>
  );
});

export function Stat({ icon, text }: { icon: "pulls" | "likes"; text: string }) {
  const c = useTheme();
  return (
    <View style={styles.stat}>
      <Icon
        name={
          icon === "pulls"
            ? { ios: "arrow.down.circle", android: "download" }
            : { ios: "heart", android: "favorite" }
        }
        size={12}
        color={c.textMuted}
      />
      <Text style={[styles.subText, { color: c.textMuted }]}>{text}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm + 2,
    minHeight: 60,
  },
  text: { flex: 1, gap: 3 },
  title: { flexDirection: "row", alignItems: "center", flexWrap: "wrap", gap: spacing.sm },
  name: { fontSize: font.body, fontWeight: "600", flexShrink: 1 },
  // Wraps rather than squeezing the owner away at large text sizes.
  sub: { flexDirection: "row", alignItems: "center", columnGap: spacing.sm, rowGap: 2, flexWrap: "wrap" },
  subText: { fontSize: font.small },
  stat: { flexDirection: "row", alignItems: "center", gap: 3 },
});
