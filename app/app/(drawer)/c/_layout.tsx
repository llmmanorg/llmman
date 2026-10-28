import { Stack } from "expo-router";

import { useChatStackOptions } from "@/components/shell/Header";

// As home's (app/(drawer)/(home)/_layout.tsx). Opening another chat replaces
// the one shown: one id for every screen keeps the stack at one.
export default function Layout() {
  return (
    <Stack screenOptions={useChatStackOptions()}>
      <Stack.Screen name="[id]" getId={() => "chat"} />
    </Stack>
  );
}
