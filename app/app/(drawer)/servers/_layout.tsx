import { Stack } from "expo-router";

import { useMenuHeaderLeft, useStackOptions } from "@/components/shell/Header";
import { usePageHeader } from "@/components/shell/PageHeader";

export default function Layout() {
  const menu = useMenuHeaderLeft();
  const stack = useStackOptions({ sides: true });
  // The desktop header: the title over the page's column.
  const page = usePageHeader();
  return (
    <Stack screenOptions={{ ...stack, ...page }}>
      <Stack.Screen name="index" options={{ title: "Servers", ...menu }} />
    </Stack>
  );
}
