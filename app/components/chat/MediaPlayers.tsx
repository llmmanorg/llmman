// The players for a generated clip or sound. The web loads them with the
// first one shown (players.web.tsx).

import { useAudioPlayer, useAudioPlayerStatus } from "expo-audio";
import { useVideoPlayer, VideoView } from "expo-video";
import { useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import { clock, type GeneratedMedia } from "@/lib/media";
import { font, radius, spacing, useTheme } from "@/lib/theme";

import { Framed, frame, MAX_SIDE, ratioOf, visual } from "./mediaFrame";

export function Clip({ uri, media }: { uri: string; media: GeneratedMedia }) {
  const c = useTheme();
  const player = useVideoPlayer(uri, (p) => {
    p.loop = false;
  });
  return (
    <Framed {...frame(ratioOf(media, 16 / 9))} style={[visual, { backgroundColor: c.surface }]}>
      <VideoView
        player={player}
        style={StyleSheet.absoluteFill}
        nativeControls
        contentFit="contain"
        fullscreenOptions={{ enable: true }}
        allowsPictureInPicture={false}
        accessibilityLabel="Generated video"
      />
    </Framed>
  );
}

export function Sound({ uri, media }: { uri: string; media: GeneratedMedia }) {
  const c = useTheme();
  const player = useAudioPlayer(uri);
  const status = useAudioPlayerStatus(player);
  const duration = status.duration || media.seconds || 0;
  const at = Math.min(status.currentTime || 0, duration || Infinity);
  const toggle = () => {
    if (status.playing) player.pause();
    else {
      // From the start again once it has played through.
      if (duration && at >= duration - 0.05) void player.seekTo(0);
      player.play();
    }
  };
  const [width, setWidth] = useState(0);
  return (
    <View style={[styles.sound, { backgroundColor: c.surface }]}>
      <Pressable
        onPress={toggle}
        accessibilityRole="button"
        accessibilityLabel={status.playing ? "Pause" : "Play"}
        style={[styles.play, { backgroundColor: c.text }]}
      >
        <Icon
          name={
            status.playing
              ? { ios: "pause.fill", android: "pause" }
              : { ios: "play.fill", android: "play_arrow" }
          }
          color={c.background}
          size={16}
        />
      </Pressable>
      <Pressable
        style={styles.track}
        onLayout={(e) => setWidth(e.nativeEvent.layout.width)}
        onPress={(e) => {
          if (duration && width) void player.seekTo((e.nativeEvent.locationX / width) * duration);
        }}
        accessibilityRole="adjustable"
        accessibilityLabel="Position"
        accessibilityValue={{ text: `${clock(at * 1000)} of ${clock(duration * 1000)}` }}
      >
        <View style={[styles.rail, { backgroundColor: c.border }]}>
          <View
            style={[
              styles.fill,
              { backgroundColor: c.accent, width: duration ? `${Math.min(100, (100 * at) / duration)}%` : 0 },
            ]}
          />
        </View>
      </Pressable>
      <Text style={[styles.time, { color: c.textMuted }]}>
        {clock(at * 1000)} / {clock(duration * 1000)}
      </Text>
    </View>
  );
}

export const styles = StyleSheet.create({
  sound: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    borderRadius: radius.pill,
    paddingLeft: 6,
    paddingRight: spacing.lg,
    height: 52,
    maxWidth: MAX_SIDE,
  },
  play: { width: 40, height: 40, borderRadius: 20, alignItems: "center", justifyContent: "center" },
  track: { flex: 1, height: 32, justifyContent: "center" },
  rail: { height: 4, borderRadius: 2, overflow: "hidden" },
  fill: { height: 4 },
  time: { fontSize: font.small, fontVariant: ["tabular-nums"] },
});
