import { Stack } from "expo-router";

import { useChatStackOptions } from "@/components/shell/Header";

// A stack of one, for the native header: on iOS its buttons are the
// system's, which a foldable iPhone moves into the bar beside the page.
export default function Layout() {
  return <Stack screenOptions={useChatStackOptions()} />;
}
