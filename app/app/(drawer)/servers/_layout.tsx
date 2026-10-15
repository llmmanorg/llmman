import { Stack } from "expo-router";

import { MenuButton } from "@/components/shell/Header";
import { usePageHeader } from "@/components/shell/PageHeader";
import { useTheme } from "@/lib/theme";

export default function Layout() {
  const c = useTheme();
  // The desktop header: the title over the page's column.
  const page = usePageHeader();
  return (
    <Stack
      screenOptions={{
        headerStyle: { backgroundColor: c.background },
        headerTintColor: c.text,
        headerShadowVisible: false,
        headerTitleStyle: { color: c.text },
        contentStyle: { backgroundColor: c.background },
        ...page,
      }}
    >
      <Stack.Screen name="index" options={{ title: "Servers", headerLeft: () => <MenuButton /> }} />
    </Stack>
  );
}
