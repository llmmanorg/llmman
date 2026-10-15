// The app's shell: conversations in a drawer on phones and a persistent
// sidebar on wide screens, with chat as home.

import { Redirect } from "expo-router";
import { Drawer } from "expo-router/drawer";
import { ActivityIndicator, StyleSheet, useWindowDimensions, View } from "react-native";

import { DrawerContent } from "@/components/shell/DrawerContent";
import { MenuButton, useSidebar } from "@/components/shell/Header";
import { usePageHeader } from "@/components/shell/PageHeader";
import { useServers } from "@/lib/ServersContext";
import { spacing, useTheme } from "@/lib/theme";

export default function DrawerLayout() {
  const c = useTheme();
  const { docked } = useSidebar();
  const { width } = useWindowDimensions();
  const { ready, active } = useServers();
  // The terminal is as wide as the page; its title is at the page's edge.
  const page = usePageHeader(Number.MAX_SAFE_INTEGER);
  if (!ready) {
    return (
      <View
        style={{ flex: 1, alignItems: "center", justifyContent: "center", backgroundColor: c.background }}
      >
        <ActivityIndicator accessibilityLabel="Loading" />
      </View>
    );
  }
  if (!active) return <Redirect href="/welcome" />;
  return (
    <Drawer
      drawerContent={(props) => <DrawerContent {...props} />}
      screenOptions={{
        drawerType: docked ? "permanent" : "front",
        drawerStyle: {
          backgroundColor: c.sidebar,
          width: docked ? 280 : Math.min(320, width * 0.84),
          borderRightColor: c.border,
          borderRightWidth: docked ? StyleSheet.hairlineWidth : 0,
        },
        overlayColor: "rgba(0,0,0,0.3)",
        swipeEdgeWidth: 48,
        headerStyle: { backgroundColor: c.background },
        headerShadowVisible: false,
        headerTitleAlign: "center",
        headerTintColor: c.text,
        headerLeft: () => (
          <View style={{ marginLeft: spacing.sm }}>
            <MenuButton />
          </View>
        ),
        headerRightContainerStyle: { paddingRight: spacing.sm },
        // No back button here; "minimal" stops the header reserving room for
        // one's title, which cut the model title ("claude-opus-5-5 · xhigh").
        headerBackButtonDisplayMode: "minimal",
        sceneStyle: { backgroundColor: c.background },
      }}
    >
      <Drawer.Screen name="index" options={{ title: "New chat" }} />
      <Drawer.Screen name="c/[id]" options={{ title: "Chat" }} />
      <Drawer.Screen name="models" options={{ title: "Models", headerShown: false }} />
      <Drawer.Screen name="providers" options={{ title: "Providers", headerShown: false }} />
      <Drawer.Screen name="shell" options={{ title: "Shell", ...page }} />
      <Drawer.Screen name="settings" options={{ title: "Settings", headerShown: false }} />
      <Drawer.Screen name="servers" options={{ title: "Servers", headerShown: false }} />
    </Drawer>
  );
}
