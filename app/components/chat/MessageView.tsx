// One message. The user's is a soft bubble on the right; the model's is
// full width, in markdown, with its reasoning as one line: "Thinking…"
// while it reasons, then "Thought for 12s", which opens it.

import * as Clipboard from "expo-clipboard";
import { memo, useEffect, useState } from "react";
import {
  ActivityIndicator,
  Animated,
  Image,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";

import { GenerationStatus, MediaResult } from "@/components/chat/MediaView";
import { Icon } from "@/components/Icon";
import { Markdown } from "@/components/Markdown";
import { alert } from "@/lib/alert";
import type { Generation, Phase } from "@/lib/chatStore";
import type { Message } from "@/lib/conversations";
import { mediaFileName } from "@/lib/media";
import { canSaveToPhotos, downloadMedia, saveToPhotos, shareMedia } from "@/lib/mediaFiles";
import { font, radius, spacing, useTheme } from "@/lib/theme";

export const MessageView = memo(function MessageView({
  message,
  live,
  last = false,
  phase,
  generation,
  modelName,
  onRetry,
  onEdit,
  onFixKey,
}: {
  message: Message;
  /** This is the reply being written now. */
  live: boolean;
  /** The newest message: its actions are always shown (others' on hover on the web, by long press on a phone). */
  last?: boolean;
  phase: Phase;
  /** The media generation in progress, for a live generated reply. */
  generation?: Generation;
  /** The friendly name of the model that wrote this reply. */
  modelName?: string;
  /** Writes this reply again, cutting the conversation after it. */
  onRetry?: () => void;
  /** Puts this user message back in the composer, cutting the conversation there. */
  onEdit?: () => void;
  /** Opens the server's settings, for a reply llmman refused over the key. */
  onFixKey?: () => void;
}) {
  const c = useTheme();
  const [copied, setCopied] = useState(false);
  const [saved, setSaved] = useState<string | null>(null);
  const [hover, setHover] = useState(false);
  const [details, setDetails] = useState(false);
  const web = Platform.OS === "web";
  const copy = async () => {
    await Clipboard.setStringAsync(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  // The web shows a message's actions on hover.
  const hoverProps = web
    ? { onPointerEnter: () => setHover(true), onPointerLeave: () => setHover(false) }
    : {};

  if (message.role === "user") {
    // A phone offers Copy and Edit on a long press.
    const menu = () =>
      alert("Your message", undefined, [
        { text: "Copy", onPress: () => void copy() },
        ...(onEdit ? [{ text: "Edit", onPress: onEdit }] : []),
        { text: "Cancel", style: "cancel" as const },
      ]);
    return (
      <View style={styles.userRow} {...hoverProps}>
        {message.images?.length ? (
          <View style={styles.userImages}>
            {message.images.map((uri, i) => (
              <Image
                key={i}
                source={{ uri }}
                style={[styles.userImage, message.images!.length > 1 && styles.userImageSmall]}
                accessibilityLabel={`Your image ${i + 1}`}
              />
            ))}
          </View>
        ) : null}
        {message.content ? (
          <Pressable
            onLongPress={web ? undefined : menu}
            accessibilityLabel={`You: ${message.content}`}
            accessibilityHint={web ? undefined : "Long press to copy or edit"}
            accessibilityActions={[
              { name: "copy", label: "Copy" },
              ...(onEdit ? [{ name: "edit", label: "Edit" }] : []),
            ]}
            onAccessibilityAction={(e) =>
              e.nativeEvent.actionName === "copy"
                ? void copy()
                : e.nativeEvent.actionName === "edit" && onEdit?.()
            }
            style={[styles.bubble, { backgroundColor: c.userBubble }]}
          >
            <Text
              selectable
              style={{ color: c.onUserBubble, fontSize: font.body, lineHeight: Math.round(font.body * 1.4) }}
            >
              {message.content}
            </Text>
          </Pressable>
        ) : null}
        {web ? (
          // Kept in the layout, shown on hover, so the list does not jump.
          <View style={[styles.userActions, { opacity: hover || copied ? 1 : 0 }]}>
            <Action
              label={copied ? "Copied" : "Copy"}
              icon={{ ios: "doc.on.doc", android: "content_copy" }}
              onPress={() => void copy()}
              done={copied}
            />
            {onEdit ? (
              <Action
                label="Edit"
                icon={{ ios: "square.and.pencil", android: "edit_square" }}
                onPress={onEdit}
              />
            ) : null}
          </View>
        ) : copied ? (
          <Text style={[styles.note, { color: c.textMuted }]}>Copied</Text>
        ) : null}
      </View>
    );
  }

  const media = message.media;
  const save = () => {
    if (!media) return;
    const name = mediaFileName(message.prompt, message.at, media.kind);
    const run = (how: () => Promise<void>, done?: string) =>
      how()
        .then(() => {
          if (!done) return;
          setSaved(done);
          setTimeout(() => setSaved(null), 1800);
        })
        .catch((e) => alert("Could not save", e instanceof Error ? e.message : String(e)));
    if (web) return downloadMedia(media, name);
    if (!canSaveToPhotos(media.kind)) return void run(() => shareMedia(media, name));
    alert(name, undefined, [
      {
        text: media.kind === "video" ? "Save video to Photos" : "Save image to Photos",
        onPress: () => void run(() => saveToPhotos(media, name), "Saved to Photos"),
      },
      { text: "Share…", onPress: () => void run(() => shareMedia(media, name)) },
      { text: "Cancel", style: "cancel" },
    ]);
  };

  const status = live && phase === "loading-model" ? `Loading ${modelName || "the model"}…` : null;
  const thinking = live && phase === "thinking" && !message.reasoning;
  const shown = !live && (last || hover);
  // An earlier reply's actions on a phone: a long press.
  const menu =
    !web && !live && !last
      ? () =>
          alert("Reply", undefined, [
            ...(message.content ? [{ text: "Copy", onPress: () => void copy() }] : []),
            ...(onRetry ? [{ text: "Regenerate", onPress: onRetry }] : []),
            { text: "Cancel", style: "cancel" as const },
          ])
      : undefined;
  const timing =
    message.durationMs !== undefined && !message.error && !message.stopped
      ? `${(message.durationMs / 1000).toFixed(1)}s`
      : "";
  const label = [modelName, timing].filter(Boolean).join(" · ");

  return (
    <Pressable
      style={styles.assistant}
      onLongPress={menu}
      delayLongPress={400}
      accessible={false}
      // The web shows earlier replies' actions on hover.
      {...hoverProps}
    >
      {message.reasoning || thinking ? (
        <Reasoning
          text={message.reasoning ?? ""}
          thinkingNow={live && !message.content}
          thoughtMs={message.thoughtMs}
        />
      ) : null}
      {status ? (
        <View style={styles.statusRow} accessibilityLiveRegion="polite">
          <ActivityIndicator size="small" />
          <Text style={{ color: c.textMuted }}>{status}</Text>
        </View>
      ) : null}
      {live && phase === "generating" && generation ? (
        <GenerationStatus generation={generation} request={message.request} />
      ) : null}
      {media ? <MediaResult message={message} /> : null}
      {message.content ? <Markdown source={message.content} /> : null}
      {message.error ? (
        <View style={styles.failure} accessibilityRole="alert">
          <View style={styles.failLine}>
            <Icon name={{ ios: "xmark.circle", android: "cancel" }} color={c.textMuted} size={15} />
            <Text style={[styles.failText, { color: c.textMuted }]}>
              {message.request
                ? `Couldn't make the ${message.request.kind === "audio" ? "sound" : message.request.kind}.`
                : "Couldn't finish the reply."}
            </Text>
            {message.keyRejected && onFixKey ? (
              <TextButton label="Update the key" onPress={onFixKey} />
            ) : onRetry ? (
              <TextButton label="Retry" onPress={onRetry} hint="Writes this reply again" />
            ) : null}
            <TextButton
              label={details ? "Hide details" : "Details"}
              onPress={() => setDetails(!details)}
              muted
              expanded={details}
            />
          </View>
          {details ? (
            <Text selectable style={[styles.details, { color: c.textMuted, borderLeftColor: c.border }]}>
              {[message.error, message.errorDetail].filter(Boolean).join("\n")}
            </Text>
          ) : null}
        </View>
      ) : null}
      {(shown && !message.error) || (!live && (message.stopped || message.finishReason === "length")) ? (
        <View style={styles.footer}>
          {message.stopped ? (
            <Text style={[styles.note, styles.shrink, { color: c.textMuted }]}>
              {message.request && message.request.kind !== "image"
                ? `Stopped. The computer finishes this ${message.request.kind === "video" ? "video" : "sound"} anyway; it won't appear here.`
                : "Stopped"}
            </Text>
          ) : null}
          {message.finishReason === "length" ? (
            <Text style={[styles.note, { color: c.textMuted }]}>Cut off at the length limit</Text>
          ) : null}
          {message.content && shown ? (
            <Action
              label={copied ? "Copied" : "Copy reply"}
              icon={{ ios: "doc.on.doc", android: "content_copy" }}
              onPress={() => void copy()}
              done={copied}
            />
          ) : null}
          {media && shown ? (
            saved ? (
              <Text style={[styles.note, { color: c.textMuted }]}>{saved}</Text>
            ) : (
              <Action
                label={web ? "Download" : "Save"}
                icon={
                  web
                    ? { ios: "arrow.down.to.line", android: "download" }
                    : { ios: "square.and.arrow.down", android: "download" }
                }
                onPress={save}
              />
            )
          ) : null}
          {onRetry && shown ? (
            <Action
              label="Regenerate"
              icon={{ ios: "arrow.clockwise", android: "refresh" }}
              onPress={onRetry}
            />
          ) : null}
          {label && shown ? (
            <Text style={[styles.note, styles.shrink, { color: c.textMuted }]} numberOfLines={1}>
              {label}
            </Text>
          ) : null}
        </View>
      ) : null}
    </Pressable>
  );
});

/** A small icon action under a message, labelled for screen readers (and as a tooltip on the web). */
function Action({
  label,
  icon,
  onPress,
  done = false,
}: {
  label: string;
  icon: Parameters<typeof Icon>[0]["name"];
  onPress: () => void;
  done?: boolean;
}) {
  const c = useTheme();
  return (
    <Pressable
      onPress={onPress}
      accessibilityRole="button"
      accessibilityLabel={label}
      hitSlop={8}
      {...(Platform.OS === "web" ? ({ title: label } as object) : {})}
      style={({ pressed, hovered }: { pressed: boolean; hovered?: boolean }) => [
        styles.action,
        { opacity: pressed ? 0.5 : 1, backgroundColor: hovered ? c.surface : "transparent" },
      ]}
    >
      <Icon name={done ? { ios: "checkmark", android: "check" } : icon} color={c.textMuted} size={16} />
    </Pressable>
  );
}

/** A word that acts, inline in a line of text: "Retry", "Details". */
function TextButton({
  label,
  onPress,
  hint,
  muted = false,
  expanded,
}: {
  label: string;
  onPress: () => void;
  hint?: string;
  muted?: boolean;
  expanded?: boolean;
}) {
  const c = useTheme();
  return (
    <Pressable
      onPress={onPress}
      accessibilityRole="button"
      accessibilityHint={hint}
      accessibilityState={expanded === undefined ? undefined : { expanded }}
      hitSlop={8}
      style={({ pressed }) => [
        styles.textButton,
        muted ? null : { borderColor: c.border },
        { opacity: pressed ? 0.5 : 1 },
      ]}
    >
      <Text style={{ color: muted ? c.textMuted : c.text, fontSize: font.small, fontWeight: "600" }}>
        {label}
      </Text>
    </Pressable>
  );
}

function Reasoning({
  text,
  thinkingNow,
  thoughtMs,
}: {
  text: string;
  thinkingNow: boolean;
  thoughtMs?: number;
}) {
  const c = useTheme();
  const [open, setOpen] = useState(false);
  const label = thinkingNow
    ? "Thinking…"
    : thoughtMs !== undefined && thoughtMs >= 1000
      ? `Thought for ${Math.round(thoughtMs / 1000)}s`
      : "Thought briefly";
  return (
    <View style={styles.reasoning}>
      <Pressable
        onPress={() => text && setOpen(!open)}
        accessibilityRole="button"
        accessibilityState={{ expanded: open }}
        accessibilityLabel={label}
        style={styles.reasoningHead}
      >
        {thinkingNow ? (
          <Shimmer text={label} color={c.textMuted} />
        ) : (
          <Text style={[styles.reasoningLabel, { color: c.textMuted }]}>{label}</Text>
        )}
        {text ? (
          <Icon
            name={
              open
                ? { ios: "chevron.down", android: "expand_more" }
                : { ios: "chevron.right", android: "chevron_right" }
            }
            color={c.textMuted}
            size={12}
          />
        ) : null}
      </Pressable>
      {open ? (
        <Text selectable style={[styles.reasoningText, { color: c.textMuted, borderLeftColor: c.border }]}>
          {/* While it streams, only the latest lines, so the answer is not pushed far below. */}
          {thinkingNow ? tail(text.trim(), LIVE_LINES) : text.trim()}
        </Text>
      ) : null}
    </View>
  );
}

/** A label that pulses softly while the model is working. */
function Shimmer({ text, color }: { text: string; color: string }) {
  const [pulse] = useState(() => new Animated.Value(0.45));
  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, { toValue: 1, duration: 700, useNativeDriver: Platform.OS !== "web" }),
        Animated.timing(pulse, { toValue: 0.45, duration: 700, useNativeDriver: Platform.OS !== "web" }),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, [pulse]);
  return (
    <Animated.Text
      style={[styles.reasoningLabel, { color, opacity: pulse }]}
      accessibilityLiveRegion="polite"
    >
      {text}
    </Animated.Text>
  );
}

const LIVE_LINES = 6;

/** The last `n` lines of `text` (long lines count by width, roughly), with an ellipsis when cut. */
export function tail(text: string, n: number, width = 60): string {
  const lines = text.split("\n");
  const out: string[] = [];
  let used = 0;
  for (let i = lines.length - 1; i >= 0 && used < n; i--) {
    const line = lines[i]!;
    const rows = Math.max(1, Math.ceil(line.length / width));
    if (used + rows > n) {
      out.unshift(`…${line.slice(-(n - used) * width)}`);
      used = n;
      break;
    }
    out.unshift(line);
    used += rows;
  }
  const cut = out.length < lines.length || out[0]?.startsWith("…");
  return cut && !out[0]?.startsWith("…") ? `…\n${out.join("\n")}` : out.join("\n");
}

const styles = StyleSheet.create({
  userImages: { flexDirection: "row", flexWrap: "wrap", justifyContent: "flex-end", gap: spacing.xs },
  userImage: { width: 180, height: 180, borderRadius: radius.lg },
  userImageSmall: { width: 96, height: 96, borderRadius: radius.md },
  userRow: { alignItems: "flex-end", paddingLeft: 56, gap: 2 },
  bubble: { borderRadius: 20, paddingHorizontal: spacing.lg, paddingVertical: spacing.sm + 2 },
  assistant: { gap: spacing.sm },
  statusRow: { flexDirection: "row", alignItems: "center", gap: spacing.sm },
  failure: { gap: spacing.sm },
  failLine: { flexDirection: "row", alignItems: "center", flexWrap: "wrap", gap: spacing.sm },
  failText: { fontSize: font.small + 1 },
  details: {
    fontSize: font.small,
    lineHeight: 18,
    borderLeftWidth: 2,
    paddingLeft: spacing.md,
    fontFamily: font.mono,
  },
  textButton: {
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "transparent",
    borderRadius: radius.pill,
    paddingHorizontal: spacing.sm + 2,
    paddingVertical: 3,
  },
  footer: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.xs,
    minHeight: 28,
    marginTop: -4,
    marginLeft: -6,
  },
  action: { width: 30, height: 30, borderRadius: 8, alignItems: "center", justifyContent: "center" },
  userActions: { flexDirection: "row", gap: spacing.xs, marginRight: -6 },
  note: { fontSize: font.small },
  shrink: { flexShrink: 1 },
  reasoning: { gap: spacing.xs },
  reasoningHead: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    minHeight: 28,
    alignSelf: "flex-start",
  },
  reasoningLabel: { fontSize: font.small + 1, fontWeight: "500" },
  reasoningText: {
    fontSize: font.small + 1,
    lineHeight: 20,
    borderLeftWidth: 2,
    paddingLeft: spacing.md,
  },
});
