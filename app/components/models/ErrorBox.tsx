// An error on the Models tab, with the one thing that can fix it: a new
// key for a rejected one, otherwise trying again.

import { router } from "expo-router";
import { StyleSheet, Text, View } from "react-native";

import { Button, Notice } from "@/components/ui";
import { useServers } from "@/lib/ServersContext";
import { font, spacing, useTheme } from "@/lib/theme";
import { explain } from "@/api/errors";

export function ErrorBox({
  error,
  title,
  onRetry,
  onDismiss,
}: {
  error: unknown;
  /** What failed, e.g. "Could not load this model". */
  title?: string;
  onRetry?: () => void;
  onDismiss?: () => void;
}) {
  const c = useTheme();
  const { active } = useServers();
  const { title: what, detail, keyRejected } = explain(error, active ?? { url: "The server" });
  return (
    <View style={styles.box}>
      <Notice tone="error" title={title}>
        {what}
      </Notice>
      {/* A rejected key's detail is the daemon's advice for CLI users: not for a phone. */}
      {detail && !keyRejected ? (
        <Text style={[styles.detail, { color: c.textMuted }]} selectable>
          {detail}
        </Text>
      ) : null}
      <View style={styles.buttons}>
        {keyRejected && active ? (
          <Button
            title="Update the key"
            onPress={() => router.push({ pathname: "/server/[id]", params: { id: active.id } })}
            style={styles.grow}
          />
        ) : onRetry ? (
          <Button
            title="Try again"
            kind={onDismiss ? "primary" : "secondary"}
            onPress={onRetry}
            style={styles.grow}
          />
        ) : null}
        {onDismiss ? (
          <Button title="Dismiss" kind="secondary" onPress={onDismiss} style={styles.grow} />
        ) : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  box: { gap: spacing.sm },
  detail: { fontSize: font.small, lineHeight: 18 },
  buttons: { flexDirection: "row", flexWrap: "wrap", gap: spacing.sm },
  grow: { flexGrow: 1 },
});
