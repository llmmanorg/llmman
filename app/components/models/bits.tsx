// Small pieces the Models screens share.

import type { ReactNode } from "react";
import { Platform, Pressable, ScrollView, StyleSheet, Text, useWindowDimensions, View } from "react-native";

import { font, radius, spacing, useTheme, type Palette } from "@/lib/theme";
import { FIT_WORDS, fitOf, registryOf, type Fit } from "./logic";

export function Chip({
  children,
  tone = "plain",
}: {
  children: ReactNode;
  tone?: "plain" | "accent" | "success";
}) {
  const c = useTheme();
  const color = { plain: c.textMuted, accent: c.accent, success: c.success }[tone];
  return (
    <View style={[styles.chip, { borderColor: tone === "plain" ? c.border : color }]}>
      <Text style={[styles.chipText, { color }]}>{children}</Text>
    </View>
  );
}

/**
 * A key that changes with the text size. React Native keeps text measured at
 * the old size when the size changes while the app runs (iOS Dynamic Type
 * from Control Center), so the lists are remounted under a new key.
 */
export function useTextSizeKey(): string {
  return String(useWindowDimensions().fontScale);
}

/** Whether text is set so large (accessibility sizes) that rows should stack instead of squeezing. */
export function useLargeText(): boolean {
  return useWindowDimensions().fontScale >= 1.6;
}

export function RegistryBadge({ name }: { name: string }) {
  const r = registryOf(name);
  if (r === "other") return null;
  return <Chip tone="accent">{r === "hf" ? "HF" : "Hub"}</Chip>;
}

export function fitColor(c: Palette, fit: Fit | null): string {
  return fit === "ok" ? c.success : fit === "warn" ? c.warning : fit === "bad" ? c.danger : c.border;
}

/** A dot for how a size fits this computer; a screen reader hears the words. */
export function FitDot({ size, memory }: { size: number; memory: number }) {
  const c = useTheme();
  const fit = fitOf(size, memory);
  // A mark as well as a colour, for anyone who cannot tell the colours apart.
  return (
    <View
      style={[styles.dot, { backgroundColor: fitColor(c, fit) }]}
      accessible={!!fit}
      accessibilityLabel={fit ? FIT_WORDS[fit] : undefined}
    >
      {fit ? (
        <Text style={[styles.mark, { color: c.background }]} allowFontScaling={false}>
          {FIT_MARK[fit]}
        </Text>
      ) : null}
    </View>
  );
}

const FIT_MARK: Record<Fit, string> = { ok: "✓", warn: "!", bad: "✕" };

export function SectionTitle({ children }: { children: ReactNode }) {
  const c = useTheme();
  return (
    <Text style={[styles.section, { color: c.textMuted }]} accessibilityRole="header">
      {children}
    </Text>
  );
}

export function Muted({ children }: { children: ReactNode }) {
  const c = useTheme();
  return <Text style={[styles.muted, { color: c.textMuted }]}>{children}</Text>;
}

/** A row of mutually exclusive choices, scrolling sideways when they do not fit. */
export function Segments<T extends string>({
  options,
  value,
  onChange,
  label,
  inset = spacing.lg,
}: {
  options: readonly { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
  label: string;
  /** Room at the row's ends: none inside a page that has its own margin. */
  inset?: number;
}) {
  const c = useTheme();
  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      contentContainerStyle={[styles.segments, { paddingHorizontal: inset }]}
      accessibilityLabel={label}
      accessibilityRole={Platform.OS === "android" ? "tablist" : undefined}
    >
      {options.map((o) => {
        const on = o.value === value;
        return (
          <Pressable
            key={o.value}
            onPress={() => onChange(o.value)}
            // UIKit has no tab trait: on iOS a selected button says which is on.
            accessibilityRole={Platform.OS === "android" ? "tab" : "button"}
            accessibilityState={{ selected: on }}
            style={[
              styles.segment,
              { backgroundColor: on ? c.accent : c.surface, borderColor: on ? c.accent : c.border },
            ]}
          >
            <Text style={[styles.segmentText, { color: on ? c.onAccent : c.text }]}>{o.label}</Text>
          </Pressable>
        );
      })}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  chip: {
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: radius.pill,
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
  },
  chipText: { fontSize: font.small - 1, fontWeight: "600" },
  dot: { width: 14, height: 14, borderRadius: 7, alignItems: "center", justifyContent: "center" },
  mark: { fontSize: 10, lineHeight: 12, fontWeight: "800", includeFontPadding: false },
  section: {
    fontSize: font.small,
    fontWeight: "600",
    textTransform: "uppercase",
    letterSpacing: 0.5,
    marginTop: spacing.lg,
    marginBottom: spacing.sm,
  },
  muted: { fontSize: font.small + 1, lineHeight: 20 },
  segments: { gap: spacing.sm, paddingHorizontal: spacing.lg, paddingVertical: spacing.sm },
  segment: {
    minHeight: 36,
    justifyContent: "center",
    paddingHorizontal: spacing.md,
    borderRadius: radius.pill,
    borderWidth: StyleSheet.hairlineWidth,
  },
  segmentText: { fontSize: font.small + 1, fontWeight: "600" },
});
