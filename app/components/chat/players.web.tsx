// The web's media players: expo-video and expo-audio are fetched with the
// first generated clip or sound shown, not with the page. Until then, the
// box the player will fill.

import { lazy, Suspense, type ComponentProps } from "react";
import { StyleSheet, View } from "react-native";

import { radius, useTheme } from "@/lib/theme";

import { Framed, frame, MAX_SIDE, ratioOf, visual } from "./mediaFrame";
import type { Clip as ClipView, Sound as SoundView } from "./MediaPlayers";

const LazyClip = lazy(() => import("./MediaPlayers").then((m) => ({ default: m.Clip })));
const LazySound = lazy(() => import("./MediaPlayers").then((m) => ({ default: m.Sound })));

export function Clip(props: ComponentProps<typeof ClipView>) {
  const c = useTheme();
  const box = (
    <Framed {...frame(ratioOf(props.media, 16 / 9))} style={[visual, { backgroundColor: c.surface }]} />
  );
  return (
    <Suspense fallback={box}>
      <LazyClip {...props} />
    </Suspense>
  );
}

export function Sound(props: ComponentProps<typeof SoundView>) {
  const c = useTheme();
  return (
    <Suspense fallback={<View style={[styles.sound, { backgroundColor: c.surface }]} />}>
      <LazySound {...props} />
    </Suspense>
  );
}

// MediaPlayers.tsx's sound bar, without its contents.
const styles = StyleSheet.create({
  sound: { borderRadius: radius.pill, height: 52, maxWidth: MAX_SIDE },
});
