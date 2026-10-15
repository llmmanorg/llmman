// The drawer (phones) and sidebar (wide screens): New chat, the
// conversations by date, then Models and Servers, and the server in use.

import type { DrawerContentComponentProps } from "expo-router/drawer";
import { router } from "expo-router";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Image, Platform, Pressable, SectionList, StyleSheet, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { Icon } from "@/components/Icon";
import { onDrawerToggleRequest, useSidebar } from "@/components/shell/Header";
import { groupChats } from "@/lib/chatGroups";
import { useDaemon } from "@/lib/DaemonContext";
import { listConversations, onConversationsChanged, type ConversationSummary } from "@/lib/conversations";
import { useModels } from "@/lib/ModelsContext";
import { serverName, shortVersion } from "@/lib/serverName";
import { useServers } from "@/lib/ServersContext";
import { font, radius, spacing, useTheme } from "@/lib/theme";
import { useOpenChat } from "@/lib/useOpenChat";

export function DrawerContent({ state, navigation }: DrawerContentComponentProps) {
  const c = useTheme();
  const insets = useSafeAreaInsets();
  const { docked, setCollapsed } = useSidebar();
  const { active } = useServers();
  const models = useModels();
  const openChat = useOpenChat();
  const [chats, setChats] = useState<ConversationSummary[]>([]);
  const [hovered, setHovered] = useState<string | null>(null);
  const daemon = useDaemon();
  const shell = daemon.features?.shell ?? null;

  const reload = useCallback(() => {
    if (active) void listConversations(active.url).then(setChats);
  }, [active]);
  useEffect(() => {
    reload();
    return onConversationsChanged(reload);
  }, [reload]);
  const sections = useMemo(() => groupChats(chats), [chats]);

  const route = state.routes[state.index];
  const current = route?.name === "c/[id]" ? (route.params as { id?: string } | undefined)?.id : undefined;
  const section = route?.name;

  const close = () => {
    if (!docked) navigation.closeDrawer();
  };

  // The web's keys: Cmd/Ctrl+\ on a narrow page, and Escape, for the drawer over the page.
  useEffect(() => {
    if (docked) return;
    const offToggle = onDrawerToggleRequest(() => navigation.toggleDrawer());
    if (Platform.OS !== "web" || typeof document === "undefined") return offToggle;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") navigation.closeDrawer();
    };
    document.addEventListener("keydown", onKey);
    return () => {
      offToggle();
      document.removeEventListener("keydown", onKey);
    };
  }, [docked, navigation]);
  const go = (fn: () => void) => {
    fn();
    close();
  };

  // Rename or delete: a small sheet, the same on every platform.
  const openActions = (chat: ConversationSummary) =>
    router.push({ pathname: "/chat-actions", params: { id: chat.id, title: chat.title || "New chat" } });

  // The 15 s /api/version poll, as llmman's web UI drives its dot; the
  // model list's answer until the first poll is in.
  const reachable = daemon.reachable ?? (models.error ? false : models.loading ? null : true);
  const status =
    reachable === false
      ? "Not reachable"
      : reachable === null
        ? "Connecting…"
        : `Connected${daemon.version?.version ? ` · ${shortVersion(daemon.version.version)}` : ""}`;
  const dot = reachable === false ? c.danger : reachable === null ? c.textMuted : c.success;

  return (
    <View style={[styles.root, { paddingTop: insets.top + spacing.sm, paddingBottom: insets.bottom }]}>
      <View style={styles.brand}>
        <Image
          source={require("@/assets/llmman-mark.png")}
          style={styles.mark}
          resizeMode="contain"
          accessible={false}
        />
        <Text style={[styles.brandText, { color: c.text, flex: 1 }]}>llmman</Text>
        {docked ? (
          <Pressable
            onPress={() => setCollapsed(true)}
            accessibilityRole="button"
            accessibilityLabel="Hide the sidebar"
            hitSlop={8}
            style={({ pressed, hovered }: { pressed: boolean; hovered?: boolean }) => [
              styles.collapse,
              { backgroundColor: pressed || hovered ? c.surfaceAlt : "transparent" },
            ]}
          >
            <Icon name={{ ios: "sidebar.left", android: "left_panel_close" }} color={c.textMuted} size={20} />
          </Pressable>
        ) : null}
      </View>
      <Row
        icon={{ ios: "square.and.pencil", android: "edit_square" }}
        label="New chat"
        selected={section === "index"}
        onPress={() => go(() => openChat())}
      />
      <SectionList
        style={styles.list}
        sections={sections}
        keyExtractor={(s) => s.id}
        stickySectionHeadersEnabled={false}
        renderSectionHeader={({ section: s }) => (
          <Text style={[styles.group, { color: c.textMuted }]} accessibilityRole="header">
            {s.title}
          </Text>
        )}
        renderItem={({ item }) => (
          <Pressable
            onPress={() => go(() => router.navigate({ pathname: "/c/[id]", params: { id: item.id } }))}
            onLongPress={() => openActions(item)}
            onHoverIn={() => setHovered(item.id)}
            onHoverOut={() => setHovered((h) => (h === item.id ? null : h))}
            accessibilityRole="button"
            accessibilityLabel={item.title || "New chat"}
            accessibilityState={{ selected: current === item.id }}
            accessibilityHint="Long press to rename or delete"
            accessibilityActions={[{ name: "actions", label: "Rename or delete" }]}
            onAccessibilityAction={(e) => e.nativeEvent.actionName === "actions" && openActions(item)}
            style={({ pressed }) => [
              styles.chat,
              { backgroundColor: current === item.id ? c.surfaceAlt : pressed ? c.surface : "transparent" },
            ]}
          >
            <Text style={{ color: c.text, fontSize: font.body - 1, flex: 1 }} numberOfLines={1}>
              {item.title || "New chat"}
            </Text>
            {/* The web has no long press to speak of: a ⋯ on the row under the pointer. */}
            {Platform.OS === "web" && (hovered === item.id || current === item.id) ? (
              <Pressable
                onPress={() => openActions(item)}
                accessibilityRole="button"
                accessibilityLabel={`Rename or delete ${item.title || "New chat"}`}
                hitSlop={6}
              >
                <Icon name={{ ios: "ellipsis", android: "more_horiz" }} color={c.textMuted} size={18} />
              </Pressable>
            ) : null}
          </Pressable>
        )}
        ListEmptyComponent={
          <Text style={[styles.none, { color: c.textMuted }]}>Your chats will show up here.</Text>
        }
      />
      <View style={[styles.footer, { borderTopColor: c.border }]}>
        <Row
          icon={{ ios: "square.stack.3d.up", android: "deployed_code" }}
          label="Models"
          selected={section === "models"}
          onPress={() => go(() => router.navigate("/models"))}
        />
        {shell ? (
          <Row
            icon={{ ios: "apple.terminal", android: "terminal" }}
            label="Shell"
            selected={section === "shell"}
            muted={!shell.enabled}
            hint={shell.enabled ? "A terminal on the computer" : `Unavailable: ${shell.reason ?? ""}`}
            onPress={() => go(() => router.navigate("/shell"))}
          />
        ) : null}
        <Row
          icon={{ ios: "key", android: "key" }}
          label="Providers"
          selected={section === "providers"}
          onPress={() => go(() => router.navigate("/providers"))}
        />
        <Row
          icon={{ ios: "gearshape", android: "settings" }}
          label="Settings"
          selected={section === "settings"}
          onPress={() => go(() => router.navigate("/settings"))}
        />
        <Pressable
          onPress={() => go(() => router.navigate("/servers"))}
          accessibilityRole="button"
          accessibilityLabel={`Servers. Using ${active ? serverName(active) : "none"}, ${status}`}
          accessibilityState={{ selected: section === "servers" }}
          style={({ pressed }) => [
            styles.row,
            {
              backgroundColor: section === "servers" ? c.surfaceAlt : pressed ? c.surface : "transparent",
            },
          ]}
        >
          <Icon name={{ ios: "desktopcomputer", android: "computer" }} color={c.text} size={20} />
          <View style={{ flex: 1 }}>
            <Text style={{ color: c.text, fontSize: font.body - 1 }} numberOfLines={1}>
              {active ? serverName(active) : "Servers"}
            </Text>
            <View style={styles.status}>
              <View style={[styles.dot, { backgroundColor: dot }]} />
              <Text style={{ color: c.textMuted, fontSize: font.small }}>{status}</Text>
            </View>
          </View>
          <Icon name={{ ios: "chevron.right", android: "chevron_right" }} color={c.textMuted} size={14} />
        </Pressable>
      </View>
    </View>
  );
}

function Row({
  icon,
  label,
  selected,
  onPress,
  muted = false,
  hint,
}: {
  icon: Parameters<typeof Icon>[0]["name"];
  label: string;
  selected: boolean;
  onPress: () => void;
  /** Shown but unavailable here: greyed, and the screen says why. */
  muted?: boolean;
  hint?: string;
}) {
  const c = useTheme();
  const color = muted ? c.textMuted : c.text;
  return (
    <Pressable
      onPress={onPress}
      accessibilityRole="button"
      accessibilityLabel={muted ? `${label}, unavailable` : label}
      accessibilityHint={hint}
      accessibilityState={{ selected }}
      style={({ pressed }) => [
        styles.row,
        { backgroundColor: selected ? c.surfaceAlt : pressed ? c.surface : "transparent" },
      ]}
    >
      <Icon name={icon} color={color} size={20} />
      <Text style={{ color, fontSize: font.body - 1, fontWeight: "500", flex: 1 }}>{label}</Text>
      {muted ? <Text style={{ color: c.textMuted, fontSize: font.small }}>Off</Text> : null}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, paddingHorizontal: spacing.sm },
  brand: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    paddingHorizontal: spacing.sm,
    paddingBottom: spacing.md,
  },
  mark: { width: 30, height: 22 },
  collapse: { padding: 6, borderRadius: radius.sm },
  brandText: { fontSize: font.body, fontWeight: "700" },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    paddingHorizontal: spacing.sm + 2,
    paddingVertical: spacing.sm + 2,
    borderRadius: radius.md,
    minHeight: 44,
  },
  list: { flex: 1, marginTop: spacing.sm },
  group: {
    fontSize: font.small - 1,
    fontWeight: "600",
    paddingHorizontal: spacing.sm + 2,
    paddingTop: spacing.lg,
    paddingBottom: spacing.xs,
  },
  chat: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    paddingHorizontal: spacing.sm + 2,
    paddingVertical: spacing.sm + 2,
    borderRadius: radius.md,
    minHeight: 40,
    justifyContent: "center",
  },
  none: { paddingHorizontal: spacing.sm + 2, paddingTop: spacing.lg, fontSize: font.small },
  footer: { borderTopWidth: StyleSheet.hairlineWidth, paddingTop: spacing.sm, gap: 2 },
  status: { flexDirection: "row", alignItems: "center", gap: 6, marginTop: 1 },
  dot: { width: 7, height: 7, borderRadius: 4 },
});
