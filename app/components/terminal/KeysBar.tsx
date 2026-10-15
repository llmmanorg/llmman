// The keys a phone keyboard lacks (esc, tab, arrows, ^C), under a terminal.

import { Pressable, ScrollView, StyleSheet, Text } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { font, radius, spacing, useTheme } from "@/lib/theme";

import { KEYS } from "./types";

export function KeysBar({ onKey }: { onKey: (seq: string) => void }) {
  const c = useTheme();
  const insets = useSafeAreaInsets();
  return (
    <ScrollView
      horizontal
      keyboardShouldPersistTaps="always"
      style={{ flexGrow: 0, backgroundColor: c.surface }}
      contentContainerStyle={[styles.keys, { paddingBottom: spacing.sm + insets.bottom }]}
    >
      {KEYS.map((k) => (
        <Pressable
          key={k.label}
          onPress={() => onKey(k.seq)}
          accessibilityRole="button"
          accessibilityLabel={k.label}
          style={({ pressed }) => [styles.key, { backgroundColor: pressed ? c.surfaceAlt : c.background }]}
        >
          <Text style={{ color: c.text, fontFamily: font.mono, fontSize: font.small + 1 }}>{k.label}</Text>
        </Pressable>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  keys: { gap: spacing.xs, paddingHorizontal: spacing.sm, paddingTop: spacing.sm },
  key: {
    minWidth: 44,
    height: 36,
    borderRadius: radius.sm,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 10,
  },
});
