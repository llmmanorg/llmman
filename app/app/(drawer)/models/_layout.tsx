import { Stack } from "expo-router";
import { Platform } from "react-native";

import { ModelsTabProvider } from "@/components/models/ModelsTabContext";
import { useServers } from "@/lib/ServersContext";
import { useMenuHeaderLeft, usePageSides } from "@/components/shell/Header";
import { usePageHeader } from "@/components/shell/PageHeader";
import { MODELS_MAX_WIDTH } from "@/components/models/navigation";
import { useTheme } from "@/lib/theme";

export default function Layout() {
  const menu = useMenuHeaderLeft();
  const sides = usePageSides();
  const c = useTheme();
  const { active } = useServers();
  // The desktop header, over the list and card.
  const page = usePageHeader(MODELS_MAX_WIDTH);
  return (
    // Keyed by server: another computer's models, cards and pull start from nothing.
    <ModelsTabProvider key={active?.url ?? ""}>
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: c.background },
          headerTintColor: c.text,
          headerShadowVisible: false,
          headerTitleStyle: { color: c.text },
          contentStyle: { backgroundColor: c.background, ...sides },
          ...page,
        }}
      >
        <Stack.Screen name="index" options={{ title: "Models", ...menu }} />
        <Stack.Screen name="[name]" options={{ title: "" }} />
        <Stack.Screen
          name="variants"
          options={{
            title: "Pick a version",
            contentStyle: { backgroundColor: c.background },
            // A sheet on iOS; on Android a sheet sits under the native tab
            // bar, so the picker is a screen of its own there.
            ...(Platform.OS === "ios"
              ? { presentation: "formSheet", sheetAllowedDetents: [0.6, 1], sheetGrabberVisible: true }
              : {}),
          }}
        />
      </Stack>
    </ModelsTabProvider>
  );
}
