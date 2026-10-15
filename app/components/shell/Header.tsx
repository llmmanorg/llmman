// The header's pieces: the menu button (phones), the model as a dropdown
// title (its sheet also holds the chat's settings), and new chat.

import { router, useNavigation } from "expo-router";
import { DrawerActions } from "expo-router/react-navigation";
import { useRef } from "react";
import { Pressable, StyleSheet, Text, useWindowDimensions, View } from "react-native";

import { Icon } from "@/components/Icon";
import { useConversation } from "@/lib/chatStore";
import { effortLabel } from "@/lib/modelName";
import { useModels } from "@/lib/ModelsContext";
import { setPickerAnchor } from "@/lib/pickerAnchor";
import { setPrefs, usePrefs } from "@/lib/prefs";
import { font, SIDEBAR_MIN_WIDTH, spacing, useTheme } from "@/lib/theme";
import { useOpenChat } from "@/lib/useOpenChat";

/** Whether the conversations are a persistent sidebar rather than a drawer. */
export function useWide(): boolean {
  return useWindowDimensions().width >= SIDEBAR_MIN_WIDTH;
}

/**
 * The sidebar: `docked` beside the page on a wide screen unless collapsed
 * there (remembered, as the web UI's `sidebarCollapsed`); otherwise a drawer
 * over the page.
 */
export function useSidebar(): { wide: boolean; docked: boolean; setCollapsed: (collapsed: boolean) => void } {
  const wide = useWide();
  const { sidebarCollapsed } = usePrefs();
  return {
    wide,
    docked: wide && !sidebarCollapsed,
    setCollapsed: (collapsed) => void setPrefs({ sidebarCollapsed: collapsed }),
  };
}

// Cmd/Ctrl+\ on a narrow page opens or closes the drawer, which only the
// drawer's own content can reach.
const drawerToggles = new Set<() => void>();
export function requestDrawerToggle(): void {
  for (const fn of drawerToggles) fn();
}
export function onDrawerToggleRequest(fn: () => void): () => void {
  drawerToggles.add(fn);
  return () => drawerToggles.delete(fn);
}

/** Opens the drawer, or brings back a collapsed sidebar; nothing while the sidebar is shown. */
export function MenuButton() {
  const c = useTheme();
  const navigation = useNavigation();
  const { wide, docked, setCollapsed } = useSidebar();
  if (docked) return null;
  return (
    <Pressable
      onPress={() => (wide ? setCollapsed(false) : navigation.dispatch(DrawerActions.toggleDrawer()))}
      accessibilityRole="button"
      accessibilityLabel={wide ? "Show the sidebar" : "Conversations and settings"}
      hitSlop={12}
      style={styles.icon}
    >
      <Icon name={{ ios: "line.3.horizontal", android: "menu" }} color={c.text} />
    </Pressable>
  );
}

export function ModelTitle({ conversation }: { conversation: string }) {
  const c = useTheme();
  const models = useModels();
  const box = useRef<View>(null);
  const { conv } = useConversation(conversation);
  const model = conv?.model || "";
  // The friendly name (lib/modelName.ts), then the chat's thinking level.
  const base = models.displayName(model);
  const name = conv?.effort ? `${base} · ${effortLabel(conv.effort)}` : base;
  const open = () => {
    // A wide screen shows the menu under the title: where the title is, now.
    box.current?.measureInWindow((x, y, width, height) => {
      setPickerAnchor({ x, y, width, height });
      router.push({ pathname: "/model-picker", params: { conversation } });
    });
  };
  return (
    <Pressable
      ref={box}
      onPress={open}
      accessibilityRole="button"
      accessibilityLabel={`Model: ${model ? name : "none"}`}
      accessibilityHint="Choose the model for this chat"
      hitSlop={8}
      style={({ pressed, hovered }: { pressed: boolean; hovered?: boolean }) => [
        styles.title,
        { opacity: pressed ? 0.6 : 1, backgroundColor: hovered ? c.surface : "transparent" },
      ]}
    >
      <Text style={[styles.titleText, { color: c.text }]} numberOfLines={1}>
        {name}
      </Text>
      <Icon name={{ ios: "chevron.down", android: "expand_more" }} color={c.textMuted} size={13} />
    </Pressable>
  );
}

export function HeaderActions() {
  const c = useTheme();
  const openChat = useOpenChat();
  // The sidebar has its own New chat.
  const { docked } = useSidebar();
  if (docked) return null;
  return (
    <View style={styles.actions}>
      <Pressable
        onPress={() => openChat()}
        accessibilityRole="button"
        accessibilityLabel="New chat"
        hitSlop={8}
        style={styles.icon}
      >
        <Icon name={{ ios: "square.and.pencil", android: "edit_square" }} color={c.text} size={21} />
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  icon: { padding: spacing.xs + 2 },
  // At least 36 high: easy to hit with a mouse as with a finger.
  title: {
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
    maxWidth: 320,
    minHeight: 36,
    paddingHorizontal: 10,
    borderRadius: 10,
  },
  titleText: { fontSize: font.body, fontWeight: "600", flexShrink: 1 },
  actions: { flexDirection: "row", alignItems: "center", gap: spacing.xs },
});
