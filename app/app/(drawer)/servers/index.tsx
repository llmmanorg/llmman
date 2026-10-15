// The servers this phone knows: which one is active, whether each answers
// right now, and a way to add, switch and edit them.

import { router, Stack, useFocusEffect } from "expo-router";
import { useCallback, useState } from "react";
import { FlatList, Pressable, StyleSheet, Text, View } from "react-native";

import { testConnection, type ConnectionResult } from "@/api/connection";
import { hostLabel } from "@/api/serverUrl";
import { Icon } from "@/components/Icon";
import type { Server } from "@/lib/servers";
import { serverName, shortVersion } from "@/lib/serverName";
import { useServers } from "@/lib/ServersContext";
import { font, readable, spacing, useTheme, useTextSizeKey } from "@/lib/theme";

type Status = ConnectionResult;

export default function Servers() {
  const c = useTheme();
  const sizeKey = useTextSizeKey();
  const { servers, active, activate, keyFor } = useServers();
  const [status, setStatus] = useState<Record<string, Status>>({});

  // Re-check every server each time the tab comes into view; the last
  // result stays shown until the new one arrives.
  useFocusEffect(
    useCallback(() => {
      const controller = new AbortController();
      for (const s of servers) {
        keyFor(s)
          .then((key) => testConnection(s.url, key, { signal: controller.signal }))
          .then((r) => setStatus((prev) => ({ ...prev, [s.url]: r })))
          .catch(() => {});
      }
      return () => controller.abort();
    }, [servers, keyFor]),
  );

  return (
    <View key={sizeKey} style={{ flex: 1, backgroundColor: c.background }}>
      <Stack.Screen
        options={{
          title: "Servers",
          headerRight: () => (
            <Pressable
              onPress={() => router.push("/connect")}
              accessibilityRole="button"
              accessibilityLabel="Add a server"
              hitSlop={12}
            >
              <Icon name={{ ios: "plus", android: "add" }} color={c.text} />
            </Pressable>
          ),
        }}
      />
      <FlatList
        data={servers}
        keyExtractor={(s) => s.id}
        contentInsetAdjustmentBehavior="automatic"
        contentContainerStyle={readable}
        ItemSeparatorComponent={() => <View style={[styles.sep, { backgroundColor: c.border }]} />}
        ListFooterComponent={
          <Text style={[styles.footer, { color: c.textMuted }]}>
            Tap a server to use it. Chats and models come from the active server.
          </Text>
        }
        renderItem={({ item }) => (
          <ServerRow
            server={item}
            active={item.url === active?.url}
            status={status[item.url]}
            onPress={() => activate(item.url)}
            onInfo={() => router.push({ pathname: "/server/[id]", params: { id: item.id } })}
          />
        )}
      />
    </View>
  );
}

function ServerRow({
  server,
  active,
  status,
  onPress,
  onInfo,
}: {
  server: Server;
  active: boolean;
  status: Status | undefined;
  onPress: () => void;
  onInfo: () => void;
}) {
  const c = useTheme();
  const [dot, line] =
    status === undefined
      ? [c.textMuted, "Checking…"]
      : status.ok
        ? [c.success, `Connected · llmman ${shortVersion(status.version.version)}`]
        : [c.danger, problemLabel(status.problem)];
  // The host as the name; the address under it.
  const name = serverName(server);
  const showUrl = name !== hostLabel(server.url);
  return (
    <View style={styles.row}>
      <Pressable
        onPress={onPress}
        style={({ pressed }) => [styles.main, { opacity: pressed ? 0.6 : 1 }]}
        accessibilityRole="button"
        accessibilityLabel={`${name}${showUrl ? `, ${hostLabel(server.url)}` : ""}, ${line}${active ? ", active" : ""}`}
        accessibilityHint={active ? undefined : "Makes this the active server"}
        accessibilityState={{ selected: active }}
      >
        <View style={styles.check}>
          {active ? <Icon name={{ ios: "checkmark", android: "check" }} color={c.accent} size={20} /> : null}
        </View>
        <View style={{ flex: 1 }}>
          <Text style={[styles.name, { color: c.text }]} numberOfLines={1}>
            {name}
          </Text>
          {showUrl ? (
            <Text style={[styles.url, { color: c.textMuted }]} numberOfLines={1}>
              {hostLabel(server.url)}
            </Text>
          ) : null}
          <View style={styles.statusRow}>
            <View style={[styles.dot, { backgroundColor: dot }]} />
            <Text style={[styles.status, { color: c.textMuted }]}>{line}</Text>
          </View>
        </View>
      </Pressable>
      <Pressable
        onPress={onInfo}
        accessibilityRole="button"
        accessibilityLabel={`Edit ${name}`}
        hitSlop={8}
        style={styles.info}
      >
        <Icon name={{ ios: "info.circle", android: "info" }} color={c.textMuted} />
      </Pressable>
    </View>
  );
}

function problemLabel(p: Exclude<ConnectionResult, { ok: true }>["problem"]): string {
  switch (p) {
    case "unauthorized":
      return "API key rejected";
    case "unreachable":
      return "Not reachable";
    case "timeout":
      return "No answer";
    case "not-llmman":
      return "Not llmman";
    case "server-error":
      return "Server error";
  }
}

const styles = StyleSheet.create({
  row: { flexDirection: "row", alignItems: "center", paddingRight: spacing.md },
  main: { flex: 1, flexDirection: "row", alignItems: "center", paddingVertical: spacing.md },
  check: { width: 44, alignItems: "center" },
  name: { fontSize: font.body + 1, fontWeight: "600" },
  url: { fontSize: font.small, marginTop: 2 },
  statusRow: { flexDirection: "row", alignItems: "center", gap: 6, marginTop: 4 },
  dot: { width: 8, height: 8, borderRadius: 4 },
  status: { fontSize: font.small },
  info: { padding: spacing.sm },
  sep: { height: StyleSheet.hairlineWidth, marginLeft: 44 },
  footer: { fontSize: font.small, padding: spacing.lg, paddingLeft: 44 },
});
