// The box a generated picture, clip or sound is shown in (MediaView.tsx,
// MediaPlayers.tsx).

import type { ReactNode } from "react";
import { StyleSheet, View } from "react-native";

import type { GeneratedMedia } from "@/lib/media";
import { radius } from "@/lib/theme";

/** The longest side a result is shown at, in points. */
export const MAX_SIDE = 512;

/** The shape a result fits in: its own aspect ratio, at most `side` on its long side. */
export function frame(ratio: number, side = MAX_SIDE) {
  return { ratio, maxWidth: ratio >= 1 ? side : side * ratio };
}

export const ratioOf = (media: GeneratedMedia, fallback: number) =>
  media.width && media.height ? media.width / media.height : fallback;

/**
 * A box of `ratio`, as wide as the column up to `maxWidth`. The cap is on an
 * outer box: with both on one view, iOS takes the height from the uncapped
 * width.
 */
export function Framed({
  ratio,
  maxWidth,
  style,
  children,
}: {
  ratio: number;
  maxWidth: number;
  style?: object;
  children?: ReactNode;
}) {
  return (
    <View style={{ width: "100%", maxWidth }}>
      <View style={[{ width: "100%", aspectRatio: ratio }, style]}>{children}</View>
    </View>
  );
}

export const { visual } = StyleSheet.create({ visual: { borderRadius: radius.lg, overflow: "hidden" } });
