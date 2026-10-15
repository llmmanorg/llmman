// A version as the card and the picker show it: its tag, `default` and
// `pulled`, then its size and fit. On one line, or on two when the text is
// large enough that the tag would be squeezed out.

import type { ReactNode } from "react";
import { StyleSheet, Text, View } from "react-native";

import type { LocalModel } from "@/api/client";
import type { Variant } from "@/api/search";
import { font, spacing, useTheme } from "@/lib/theme";
import { Chip, FitDot, useLargeText } from "./bits";
import { formatBytes, holds } from "./logic";

export function VariantLine({
  variant: v,
  local,
  memory,
  strong = false,
  lead,
  leadWidth = 0,
  trail,
}: {
  variant: Variant;
  local: readonly LocalModel[];
  memory: number;
  strong?: boolean;
  /** Before the tag (the picker's checkmark), `leadWidth` wide: stacked, the second line starts after it. */
  lead?: ReactNode;
  leadWidth?: number;
  /** After the fit dot (the card's chevron). */
  trail?: ReactNode;
}) {
  const c = useTheme();
  const stacked = useLargeText();
  return (
    <View style={stacked ? styles.stacked : styles.line}>
      <View style={styles.name}>
        {lead}
        <Text
          style={[styles.tag, { color: c.text, fontWeight: strong ? "700" : "500" }]}
          numberOfLines={stacked ? undefined : 1}
        >
          {v.tag}
        </Text>
        {v.default ? <Chip>default</Chip> : null}
        {holds(v, local) ? <Chip tone="success">pulled</Chip> : null}
      </View>
      <View style={[styles.facts, stacked && leadWidth ? { paddingLeft: leadWidth + spacing.sm } : null]}>
        <Text style={[styles.size, { color: c.textMuted }]}>{v.size ? formatBytes(v.size) : "?"}</Text>
        <FitDot size={v.size} memory={memory} />
        {trail}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  line: { flex: 1, flexDirection: "row", alignItems: "center", gap: spacing.sm },
  stacked: { flex: 1, gap: spacing.xs, paddingVertical: spacing.sm },
  name: { flex: 1, flexDirection: "row", flexWrap: "wrap", alignItems: "center", gap: spacing.sm },
  tag: { fontSize: font.body, flexShrink: 1 },
  facts: { flexDirection: "row", alignItems: "center", gap: spacing.sm },
  size: { fontSize: font.small + 1, fontVariant: ["tabular-nums"] },
});
