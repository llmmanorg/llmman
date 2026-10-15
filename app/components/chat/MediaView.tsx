// A generated reply: the picture, clip or sound with its caption, and the
// status line while it is being made ("Generating image… step 12/30").

import { useEffect, useState, type ReactNode } from "react";
import { Animated, Image, Platform, Pressable, StyleSheet, Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import type { Generation } from "@/lib/chatStore";
import type { Message } from "@/lib/conversations";
import { describeMedia, generationStatus, type GeneratedMedia, type MediaOptions } from "@/lib/media";
import { useMediaUri } from "@/lib/mediaFiles";
import { font, radius, spacing, useTheme } from "@/lib/theme";

import { Framed, frame, MAX_SIDE, ratioOf, visual } from "./mediaFrame";
import { Clip, Sound } from "./players";

export function MediaResult({ message }: { message: Message }) {
  const c = useTheme();
  const media = message.media!;
  const uri = useMediaUri(media);
  const [expanded, setExpanded] = useState(false);
  const revised = media.revisedPrompt?.trim();
  return (
    <View style={styles.result}>
      {!uri ? (
        <View style={[styles.missing, { backgroundColor: c.surface }]}>
          <Text style={{ color: c.textMuted }}>The file is no longer on this device.</Text>
        </View>
      ) : media.kind === "image" ? (
        <Picture uri={uri} media={media} label={message.prompt || "Generated image"} />
      ) : media.kind === "video" ? (
        <Clip uri={uri} media={media} />
      ) : (
        <Sound uri={uri} media={media} />
      )}
      <Text style={[styles.caption, { color: c.textMuted }]} selectable>
        {describeMedia(media, message.request)}
      </Text>
      {revised && revised !== message.prompt?.trim() ? (
        <Pressable
          onPress={() => setExpanded((e) => !e)}
          accessibilityRole="button"
          accessibilityState={{ expanded }}
          accessibilityLabel={`Prompt as enhanced by the model: ${revised}`}
        >
          <Text
            style={[styles.revised, { color: c.textMuted }]}
            numberOfLines={expanded ? undefined : 2}
            selectable={expanded}
          >
            <Text style={{ fontWeight: "600" }}>Enhanced prompt </Text>
            {revised}
          </Text>
        </Pressable>
      ) : null}
    </View>
  );
}

function Picture({ uri, media, label }: { uri: string; media: GeneratedMedia; label: string }) {
  const c = useTheme();
  return (
    <Framed {...frame(ratioOf(media, 1))} style={[visual, { backgroundColor: c.surface }]}>
      <Image
        source={{ uri }}
        style={StyleSheet.absoluteFill}
        resizeMode="contain"
        accessibilityLabel={label}
      />
    </Framed>
  );
}

/** "Loading <model>, then generating image… 0:12", or the step counter with a bar. */
export function GenerationStatus({
  generation,
  request,
}: {
  generation: Generation;
  request?: MediaOptions;
}) {
  const c = useTheme();
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  const text = generationStatus({ ...generation, elapsedMs: now - generation.startedAt });
  const fraction = generation.total ? Math.min(1, (generation.step ?? 0) / generation.total) : null;
  return (
    <View style={styles.status} accessibilityLiveRegion="polite" accessibilityLabel={text}>
      <Placeholder generation={generation} request={request}>
        <Icon
          name={
            generation.kind === "image"
              ? { ios: "photo", android: "image" }
              : generation.kind === "video"
                ? { ios: "video", android: "videocam" }
                : { ios: "waveform", android: "graphic_eq" }
          }
          color={c.border}
          size={32}
        />
      </Placeholder>
      <Text style={[styles.statusText, { color: c.textMuted }]}>{text}</Text>
      <Bar fraction={fraction} />
    </View>
  );
}

/** Where the result will appear: its requested shape when the size is set. */
function Placeholder({
  generation: g,
  request,
  children,
}: {
  generation: Generation;
  request?: MediaOptions;
  children: ReactNode;
}) {
  const c = useTheme();
  const style = [styles.placeholder, { backgroundColor: c.surface }];
  if (g.kind === "audio") {
    return <View style={[style, { height: 52, width: "100%", maxWidth: MAX_SIDE }]}>{children}</View>;
  }
  const ratio =
    request?.width && request.height ? request.width / request.height : g.kind === "video" ? 16 / 9 : 1;
  return (
    <Framed {...frame(ratio, g.kind === "video" ? MAX_SIDE : 320)} style={style}>
      {children}
    </Framed>
  );
}

/** A thin progress bar: filled to `fraction`, or sweeping while it is unknown. */
function Bar({ fraction }: { fraction: number | null }) {
  const c = useTheme();
  const [sweep] = useState(() => new Animated.Value(0));
  const indeterminate = fraction === null;
  useEffect(() => {
    if (!indeterminate) return;
    const loop = Animated.loop(
      Animated.timing(sweep, { toValue: 1, duration: 1400, useNativeDriver: Platform.OS !== "web" }),
    );
    loop.start();
    return () => loop.stop();
  }, [indeterminate, sweep]);
  const [width, setWidth] = useState(0);
  return (
    <View
      style={[styles.bar, { backgroundColor: c.surfaceAlt }]}
      onLayout={(e) => setWidth(e.nativeEvent.layout.width)}
      accessibilityRole="progressbar"
      accessibilityValue={
        fraction === null ? undefined : { min: 0, max: 100, now: Math.round(fraction * 100) }
      }
    >
      {indeterminate ? (
        <Animated.View
          style={[
            styles.barFill,
            {
              backgroundColor: c.accent,
              width: width * 0.3,
              transform: [
                { translateX: sweep.interpolate({ inputRange: [0, 1], outputRange: [-width * 0.3, width] }) },
              ],
            },
          ]}
        />
      ) : (
        <View
          style={[
            styles.barFill,
            { backgroundColor: c.accent, width: `${Math.round(fraction * 1000) / 10}%` },
          ]}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  result: { gap: spacing.xs + 2 },
  missing: { borderRadius: radius.lg, padding: spacing.lg },
  caption: { fontSize: font.small },
  revised: { fontSize: font.small, lineHeight: 18 },
  status: { gap: spacing.sm },
  placeholder: { borderRadius: radius.lg, alignItems: "center", justifyContent: "center" },
  statusText: { fontSize: font.small + 1, fontVariant: ["tabular-nums"] },
  bar: { height: 3, borderRadius: 2, overflow: "hidden", maxWidth: MAX_SIDE },
  barFill: { height: 3, borderRadius: 2 },
});
