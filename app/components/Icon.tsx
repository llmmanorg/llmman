// One icon component for both platforms: SF Symbols on iOS, Material
// Symbols on Android and the web (Icon.web.tsx draws those inline). Icons are decoration: the control around one carries
// the label. iOS still exposes a bare SymbolView as an image named after
// the symbol, so the wrapper hides it from screen readers.

import { SymbolView } from "expo-symbols";
import type { ComponentProps } from "react";
import { View } from "react-native";

type SymbolName = ComponentProps<typeof SymbolView>["name"];

export function Icon({ name, size = 22, color }: { name: SymbolName; size?: number; color: string }) {
  return (
    <View accessible={false} accessibilityElementsHidden importantForAccessibility="no-hide-descendants">
      <SymbolView name={withWeb(name)} size={size} tintColor={color} accessible={false} />
    </View>
  );
}

/** The web draws Material Symbols too: the Android name unless one is given. */
function withWeb(name: SymbolName): SymbolName {
  return typeof name === "object" && !name.web && name.android ? { ...name, web: name.android } : name;
}
