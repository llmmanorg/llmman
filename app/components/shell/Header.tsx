// The header's pieces: the menu button (phones), the model as a dropdown
// title (its sheet also holds the chat's settings), and new chat.

import { router, useNavigation } from "expo-router";
import { DrawerActions } from "expo-router/react-navigation";
import type { NativeStackHeaderItem, NativeStackNavigationOptions } from "expo-router/native-stack";
import { type ReactNode, useRef } from "react";
import { Platform, Pressable, StyleSheet, Text, useWindowDimensions, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

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

/**
 * A page's padding at its sides for the safe area there: a notch in
 * landscape, or the vertical bar a foldable iPhone keeps beside the page.
 * Beside the sidebar only the right edge can be the screen's.
 */
export function usePageSides(): { paddingLeft: number; paddingRight: number } {
  const insets = useSafeAreaInsets();
  const { docked } = useSidebar();
  return { paddingLeft: docked ? 0 : insets.left, paddingRight: insets.right };
}

/** A native stack's header buttons: React views, or on iOS the system's own. */
type HeaderButtons = {
  headerLeft?: () => ReactNode;
  headerRight?: () => ReactNode;
  unstable_headerLeftItems?: () => NativeStackHeaderItem[];
  unstable_headerRightItems?: () => NativeStackHeaderItem[];
};

/**
 * A page header's left item: the menu button, or none while the sidebar is
 * shown, so iOS draws no empty button where it would be. On iOS it is a
 * system bar button, which a foldable iPhone held upright puts in the
 * vertical bar beside the page, under the status bar.
 */
export function useMenuHeaderLeft(): HeaderButtons {
  const { wide, docked, setCollapsed } = useSidebar();
  if (docked) return {};
  if (Platform.OS !== "ios") return { headerLeft: () => <MenuButton /> };
  const item: NativeStackHeaderItem = {
    type: "button",
    label: wide ? "Show the sidebar" : "Conversations and settings",
    icon: { type: "sfSymbol", name: "line.3.horizontal" },
    onPress: () => (wide ? setCollapsed(false) : requestDrawerToggle()),
  };
  return { unstable_headerLeftItems: () => [item] };
}

/** A chat's header buttons: the menu, and New chat but beside the sidebar, which has its own. */
export function useChatHeaderButtons(): HeaderButtons {
  const menu = useMenuHeaderLeft();
  const openChat = useOpenChat();
  const { docked } = useSidebar();
  if (docked) return menu;
  if (Platform.OS === "web") {
    // Inset as the drawer's header had them.
    return {
      headerLeft: () => (
        <View style={{ marginLeft: spacing.sm }}>
          <MenuButton />
        </View>
      ),
      headerRight: () => (
        <View style={{ marginRight: spacing.sm }}>
          <HeaderActions />
        </View>
      ),
    };
  }
  if (Platform.OS !== "ios") return { ...menu, headerRight: () => <HeaderActions /> };
  const item: NativeStackHeaderItem = {
    type: "button",
    label: "New chat",
    icon: { type: "sfSymbol", name: "square.and.pencil" },
    onPress: () => openChat(),
  };
  return { ...menu, unstable_headerRightItems: () => [item] };
}

/** The chat stacks' header, as the drawer's own looks on the other pages. */
export function useChatStackOptions(): NativeStackNavigationOptions {
  const c = useTheme();
  return {
    headerStyle: { backgroundColor: c.background },
    headerShadowVisible: false,
    headerTitleAlign: "center",
    headerTintColor: c.text,
    headerBackButtonDisplayMode: "minimal",
    contentStyle: { backgroundColor: c.background },
    ...useChatHeaderButtons(),
  };
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
