// The pull, shown where it matters: in full on its model's card, and as a
// slim banner on the list. Stopping only stops this phone following it;
// the daemon has no way to cancel a pull, and other clients may be
// watching the same one.

import { Pressable, StyleSheet, Text, View } from "react-native";

import { Button } from "@/components/ui";
import { ErrorBox } from "./ErrorBox";
import { font, radius, spacing, useTheme } from "@/lib/theme";
import { formatBytes, pullPercent, shortName, tagOf } from "./logic";
import type { Pull } from "./ModelsTabContext";

export function ProgressBar({ percent }: { percent: number | null }) {
  const c = useTheme();
  return (
    <View
      style={[styles.bar, { backgroundColor: c.surfaceAlt }]}
      accessibilityRole="progressbar"
      accessibilityValue={percent == null ? undefined : { min: 0, max: 100, now: percent }}
    >
      <View
        style={[
          styles.fill,
          { backgroundColor: c.accent },
          percent == null ? styles.unknown : { width: `${percent}%` },
        ]}
      />
    </View>
  );
}

export function PullPanel({
  pull,
  onStop,
  onRetry,
  onDismiss,
}: {
  pull: Pull;
  onStop: () => void;
  onRetry: () => void;
  onDismiss: () => void;
}) {
  const c = useTheme();
  const percent = pullPercent(pull);
  if (pull.error) {
    return (
      <View style={styles.panel}>
        <ErrorBox
          error={pull.error}
          title={`Pulling ${tagOf(pull.ref)} failed`}
          onRetry={onRetry}
          onDismiss={onDismiss}
        />
      </View>
    );
  }
  return (
    <View
      style={[styles.panel, { backgroundColor: c.surface, borderRadius: radius.md, padding: spacing.md }]}
    >
      <Text style={[styles.status, { color: c.text }]} accessibilityLiveRegion="polite">
        {pull.done ? `Pulled ${tagOf(pull.ref)}` : pull.status || `Pulling ${tagOf(pull.ref)}`}
      </Text>
      <ProgressBar percent={pull.done ? 100 : percent} />
      {pull.total ? (
        <Text style={[styles.detail, { color: c.textMuted }]}>
          {formatBytes(pull.completed)} of {formatBytes(pull.total)}
          {percent != null ? ` · ${percent}%` : ""}
        </Text>
      ) : null}
      {pull.notices.map((n, i) => (
        <Text key={i} style={[styles.detail, { color: c.textMuted }]}>
          {n}
        </Text>
      ))}
      {pull.done ? null : (
        <>
          <Button title="Stop watching" kind="secondary" onPress={onStop} />
          <Text style={[styles.detail, { color: c.textMuted }]}>
            The computer keeps downloading after you stop watching. Pull it again to follow it.
          </Text>
        </>
      )}
    </View>
  );
}

export function PullBanner({ pull, onPress }: { pull: Pull; onPress: () => void }) {
  const c = useTheme();
  const percent = pullPercent(pull);
  const text = pull.error
    ? `Pull of ${shortName(pull.ref)} failed`
    : pull.done
      ? `Pulled ${shortName(pull.ref)}`
      : `Pulling ${shortName(pull.ref)}${percent != null ? ` · ${percent}%` : ""}`;
  return (
    <Pressable
      onPress={onPress}
      accessibilityRole="button"
      accessibilityLabel={text}
      accessibilityHint="Opens the model"
      style={[styles.banner, { backgroundColor: c.surface, borderColor: pull.error ? c.danger : c.border }]}
    >
      <Text style={[styles.bannerText, { color: pull.error ? c.danger : c.text }]} numberOfLines={1}>
        {text}
      </Text>
      {pull.error || pull.done ? null : <ProgressBar percent={percent} />}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  panel: { gap: spacing.sm, marginTop: spacing.md },
  buttons: { flexDirection: "row", gap: spacing.sm },
  grow: { flex: 1 },
  status: { fontSize: font.body, fontWeight: "600" },
  detail: { fontSize: font.small, lineHeight: 18 },
  bar: { height: 6, borderRadius: 3, overflow: "hidden" },
  fill: { height: 6, borderRadius: 3 },
  unknown: { width: "30%", opacity: 0.5 },
  banner: {
    marginHorizontal: spacing.lg,
    marginVertical: spacing.sm,
    padding: spacing.md,
    borderRadius: radius.md,
    borderWidth: StyleSheet.hairlineWidth,
    gap: spacing.sm,
  },
  bannerText: { fontSize: font.small + 1, fontWeight: "600" },
});
