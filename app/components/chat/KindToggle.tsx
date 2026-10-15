// With a media generation model, what the prompt becomes: Image, Video or
// Audio, only the kinds the model makes (the web UI's toggle).

import { Pressable, StyleSheet, Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import { KIND_LABELS, type MediaKind } from "@/lib/media";
import { font, radius, spacing, useTheme } from "@/lib/theme";

export const KIND_ICONS = {
  image: { ios: "photo", android: "image" },
  video: { ios: "video", android: "videocam" },
  audio: { ios: "waveform", android: "graphic_eq" },
} as const;

export function KindToggle({
  kinds,
  value,
  onChange,
}: {
  kinds: readonly MediaKind[];
  value: MediaKind;
  onChange: (kind: MediaKind) => void;
}) {
  const c = useTheme();
  return (
    <View
      style={[styles.group, { backgroundColor: c.surfaceAlt }]}
      accessibilityRole="radiogroup"
      accessibilityLabel="Output"
    >
      {kinds.map((k) => {
        const on = k === value;
        return (
          <Pressable
            key={k}
            onPress={() => onChange(k)}
            accessibilityRole="radio"
            accessibilityState={{ checked: on }}
            accessibilityLabel={`Generate ${k === "audio" ? "audio" : `a${k === "image" ? "n" : ""} ${k}`}`}
            hitSlop={4}
            style={({ pressed }) => [
              styles.option,
              on && [styles.on, { backgroundColor: c.background, shadowColor: "#000" }],
              { opacity: pressed && !on ? 0.6 : 1 },
            ]}
          >
            <Icon name={KIND_ICONS[k]} color={on ? c.text : c.textMuted} size={15} />
            <Text style={[styles.label, { color: on ? c.text : c.textMuted }]}>{KIND_LABELS[k]}</Text>
          </Pressable>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  group: { flexDirection: "row", borderRadius: radius.pill, padding: 3, gap: 2 },
  option: {
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
    borderRadius: radius.pill,
    paddingHorizontal: spacing.md - 2,
    minHeight: 30,
  },
  on: { shadowOpacity: 0.08, shadowRadius: 3, shadowOffset: { width: 0, height: 1 }, elevation: 1 },
  label: { fontSize: font.small, fontWeight: "600" },
});
