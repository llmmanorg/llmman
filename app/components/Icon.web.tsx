// The web's Icon (see Icon.tsx): the same Material Symbols glyphs Android
// draws from expo-symbols' font, as inline SVG, so the page needs no font
// file. Only the glyphs the app uses are kept (iconPaths.generated.ts,
// from `npm run make-web-icons`).

import type { SymbolViewProps } from "expo-symbols";
import { createElement } from "react";
import { View } from "react-native";

import { ICON_PATHS } from "./iconPaths.generated";

type SymbolName = SymbolViewProps["name"];

export function Icon({ name, size = 22, color }: { name: SymbolName; size?: number; color: string }) {
  const glyph = typeof name === "object" ? (name.web ?? name.android) : undefined;
  const d = glyph ? ICON_PATHS[glyph] : undefined;
  if (__DEV__ && glyph && !d) console.warn(`Icon: no path for "${glyph}"; run npm run make-web-icons`);
  // The same views as expo-symbols draws in, so layouts don't move; a name
  // for iOS alone draws nothing, as there.
  return (
    <View accessible={false} aria-hidden importantForAccessibility="no-hide-descendants">
      {glyph ? (
        <View style={{ width: size, height: size }}>
          {d
            ? createElement(
                "svg",
                {
                  width: size,
                  height: size,
                  viewBox: "0 -960 960 960",
                  fill: color,
                  "aria-hidden": true,
                  focusable: false,
                  style: { display: "block" },
                },
                createElement("path", { d }),
              )
            : null}
        </View>
      ) : null}
    </View>
  );
}
