import { DarkTheme, DefaultTheme, Stack, ThemeProvider } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { LogBox, Platform, useWindowDimensions } from "react-native";
import { GestureHandlerRootView } from "react-native-gesture-handler";
import { KeyboardProvider } from "react-native-keyboard-controller";
import { SafeAreaProvider } from "react-native-safe-area-context";

import { useEffect, useMemo } from "react";

import { DialogHost } from "@/components/DialogHost";
import { WebChrome } from "@/components/shell/WebChrome";
import { DaemonProvider } from "@/lib/DaemonContext";
import { ModelsProvider } from "@/lib/ModelsContext";
import { loadPrefs } from "@/lib/prefs";
import { ServersProvider } from "@/lib/ServersContext";
import { dark, SIDEBAR_MIN_WIDTH, useTheme } from "@/lib/theme";

// react-native-keyboard-controller 1.21 (the SDK 57 version) passes
// dependency arrays to Reanimated 4 hooks, which ignore them on native and
// warn in development only. Ours has none; drop the line when the library
// is updated.
LogBox.ignoreLogs(["[Reanimated] Dependencies should only be used on the web"]);

export const unstable_settings = { initialRouteName: "(drawer)" };

export default function RootLayout() {
  const c = useTheme();
  useEffect(() => {
    void loadPrefs();
  }, []);
  // Wide screens show the model menu as a popover over the chat (model-picker.tsx).
  const wide = useWindowDimensions().width >= SIDEBAR_MIN_WIDTH;
  // Navigation's own theme follows ours: native headers take their light or
  // dark look from it (iOS draws the header buttons' glass by it), not from
  // the system's.
  const navTheme = useMemo(() => {
    const base = c === dark ? DarkTheme : DefaultTheme;
    return {
      ...base,
      colors: {
        ...base.colors,
        primary: c.accent,
        background: c.background,
        card: c.background,
        text: c.text,
        border: c.border,
      },
    };
  }, [c]);
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <SafeAreaProvider>
        {/* Keyboard frames that are right under Android's edge-to-edge layout too. */}
        <KeyboardProvider>
          <ServersProvider>
            <DaemonProvider>
              <ModelsProvider>
                <ThemeProvider value={navTheme}>
                  <StatusBar style="auto" />
                  {Platform.OS === "web" ? <DialogHost /> : null}
                  {Platform.OS === "web" ? <WebChrome /> : null}
                  <Stack
                    screenOptions={{
                      headerStyle: { backgroundColor: c.background },
                      headerTintColor: c.text,
                      headerShadowVisible: false,
                      headerTitleStyle: { color: c.text },
                      contentStyle: { backgroundColor: c.background },
                    }}
                  >
                    {/* Its title is what screens pushed over the shell show on their back button. */}
                    <Stack.Screen name="(drawer)" options={{ headerShown: false, title: "Back" }} />
                    <Stack.Screen
                      name="connect/index"
                      options={{ title: "Add a server", presentation: "modal" }}
                    />
                    <Stack.Screen name="welcome" options={{ headerShown: false, gestureEnabled: false }} />
                    <Stack.Screen
                      name="server/[id]"
                      options={{ title: "Server", headerBackTitle: "Servers" }}
                    />
                    <Stack.Screen
                      name="model-picker"
                      options={
                        wide
                          ? {
                              presentation: "transparentModal",
                              headerShown: false,
                              animation: "fade",
                              contentStyle: { backgroundColor: "transparent" },
                            }
                          : { title: "Model", presentation: "modal" }
                      }
                    />
                    <Stack.Screen
                      name="chat-settings"
                      options={{ title: "Chat settings", presentation: "modal" }}
                    />
                    <Stack.Screen
                      name="generate-settings"
                      options={{ title: "Generation settings", presentation: "modal" }}
                    />
                  </Stack>
                </ThemeProvider>
              </ModelsProvider>
            </DaemonProvider>
          </ServersProvider>
        </KeyboardProvider>
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}
