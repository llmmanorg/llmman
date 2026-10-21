// A conversation: messages, the floating composer, Stop, and the header's
// model dropdown. The home screen shows a fresh one; `onSent` lets it move
// to the conversation's own route after the first message.

import * as Haptics from "expo-haptics";
import { router } from "expo-router";
import { useIsFocused } from "expo-router/react-navigation";
import { Drawer } from "expo-router/drawer";
import { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  FlatList,
  Image,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
  type NativeSyntheticEvent,
  type TextInputKeyPressEventData,
} from "react-native";
import { KeyboardAvoidingView } from "react-native-keyboard-controller";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { hostLabel } from "@/api/serverUrl";
import { KindToggle } from "@/components/chat/KindToggle";
import { MessageView } from "@/components/chat/MessageView";
import { Icon } from "@/components/Icon";
import { HeaderActions, ModelTitle, useWide } from "@/components/shell/Header";
import {
  editFrom,
  openConversation,
  retry,
  send,
  setMediaOptions,
  stop,
  useConversation,
} from "@/lib/chatStore";
import { alert } from "@/lib/alert";
import { MAX_IMAGES, pickImages, prepareImage } from "@/lib/attachments";
import { optionsFor, PLACEHOLDERS, type MediaKind } from "@/lib/media";
import { isAvailable } from "@/lib/modelName";
import { useModels } from "@/lib/ModelsContext";
import { greeting, usePrefs } from "@/lib/prefs";
import { useServers } from "@/lib/ServersContext";
import { useKeyboardShown } from "@/lib/useKeyboardShown";
import { font, radius, readable, spacing, useTheme, useTextSizeKey } from "@/lib/theme";

const SUGGESTIONS = [
  "Explain a concept simply",
  "Write a short poem about the sea",
  "Summarize a paragraph",
  "Help me debug some code",
];

/** With a media generation model: prompts that show what it does. */
const MEDIA_SUGGESTIONS: Record<MediaKind, string[]> = {
  image: [
    "A lighthouse on a cliff at dusk, oil painting",
    "A red fox in fresh snow, soft morning light",
    "An isometric cozy reading nook, pastel colors",
  ],
  video: [
    "Waves rolling onto a sandy beach at sunset",
    "A paper boat drifting down a rainy street",
    "Time-lapse of clouds over a mountain lake",
  ],
  audio: [
    "Rain on a tin roof with distant thunder",
    "A crackling campfire at night",
    "Birdsong in a quiet forest",
  ],
};

const MEDIA_GREETING: Record<MediaKind, string> = {
  image: "What should I draw?",
  video: "What should the video show?",
  audio: "What should it sound like?",
};

export function ChatScreen({ id, onSent }: { id: string; onSent?: () => void }) {
  const c = useTheme();
  const sizeKey = useTextSizeKey();
  const insets = useSafeAreaInsets();
  const shell = useRef<View>(null);
  const [top, setTop] = useState(0);
  const keyboardShown = useKeyboardShown();
  // The composer (as wide as the message column at most) clears the notch in landscape.
  const side = Math.max(spacing.md, insets.left, insets.right);
  // The web's textarea does not grow by itself: it follows its content's height.
  const [inputHeight, setInputHeight] = useState(0);
  const { client: activeClient, active, servers, activate } = useServers();
  const server = active ? { name: active.name, url: active.url } : undefined;
  const models = useModels();
  const prefs = usePrefs();
  const { conv, phase, generation } = useConversation(id);
  const focused = useIsFocused();
  // A chat goes only to the server it was had with. Opened from a link
  // while another server is active, it switches to its own (when it is the
  // screen in front: a chat left open behind must not undo a switch made
  // elsewhere); one whose server is not added here can be read, not continued.
  const elsewhere = !!conv && !!active && conv.server !== active.url;
  const known = elsewhere && servers.some((s) => s.url === conv.server);
  const client = elsewhere ? null : activeClient;
  useEffect(() => {
    if (known && focused && conv) void activate(conv.server);
  }, [known, focused, conv, activate]);
  const [draft, setDraft] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const [attaching, setAttaching] = useState(false);
  const [missing, setMissing] = useState(false);
  const composer = useRef<TextInput>(null);
  const wide = useWide();

  useEffect(() => {
    let cancelled = false;
    openConversation(id).then((found) => {
      if (!cancelled && !found) setMissing(true);
    });
    return () => {
      cancelled = true;
    };
  }, [id]);

  const busy = phase !== "idle";
  // The greeting follows the time of day while the page stays open.
  const [hour, setHour] = useState(() => new Date().getHours());
  useEffect(() => {
    const timer = setInterval(() => setHour(new Date().getHours()), 60_000);
    return () => clearInterval(timer);
  }, []);
  const model = conv?.model || "";
  // A media generation model turns the composer into a prompt for it.
  const mediaKinds = models.mediaKinds(model);
  const generating = mediaKinds.length > 0;
  const media = optionsFor(conv?.media, mediaKinds);
  const canAttach = !generating && models.acceptsImages(model);
  const canSend =
    !!client && !!model && (!!draft.trim() || images.length > 0) && !busy && (canAttach || !images.length);

  const submit = (text = draft, withImages = images) => {
    if (!client || !model || (!text.trim() && !withImages.length) || busy || !conv) return;
    if (withImages.length && !canAttach) return;
    setDraft("");
    setImages([]);
    if (Platform.OS === "ios") void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    const loaded = models.isLoaded(model);
    void send(client, conv.id, text, { loaded, server, mediaKinds }, withImages).then(() =>
      models.markLoaded(model),
    );
    onSent?.();
  };

  const turn = () => ({ loaded: models.isLoaded(model), server, mediaKinds });

  const edit = async (index: number) => {
    if (!conv) return;
    const back = await editFrom(conv.id, index);
    if (!back) return;
    setDraft(back.text);
    setImages(back.images);
    composer.current?.focus();
  };

  // A chat whose model the server no longer offers says so, as the web UI does.
  const gone =
    !elsewhere &&
    !!model &&
    !!conv?.messages.length &&
    models.loadedAt > 0 &&
    !models.error &&
    !models.loading &&
    !isAvailable(model, models.local, models.providers);

  const addImages = async (load: () => Promise<string[]>) => {
    setAttaching(true);
    try {
      const added = await load();
      setImages((list) => [...list, ...added].slice(0, MAX_IMAGES));
    } catch (e) {
      alert("Could not attach the image", e instanceof Error ? e.message : String(e));
    } finally {
      setAttaching(false);
    }
  };

  const attach = () => {
    const room = MAX_IMAGES - images.length;
    if (Platform.OS === "web") {
      void addImages(() => pickImages("library", room));
      return;
    }
    alert("Attach images", undefined, [
      { text: "Photo library", onPress: () => void addImages(() => pickImages("library", room)) },
      { text: "Camera", onPress: () => void addImages(() => pickImages("camera", 1)) },
      { text: "Cancel", style: "cancel" },
    ]);
  };

  // The web also takes a pasted or dropped image, as long as this chat is on screen.
  useEffect(() => {
    if (Platform.OS !== "web" || !focused || !canAttach) return;
    const take = (files: FileList | null | undefined) => {
      const list = [...(files ?? [])].filter((f) => f.type.startsWith("image/"));
      if (!list.length) return false;
      void addImages(() =>
        Promise.all(
          list.slice(0, MAX_IMAGES).map(async (f) => {
            const url = URL.createObjectURL(f);
            try {
              return await prepareImage(url, f.type);
            } finally {
              URL.revokeObjectURL(url);
            }
          }),
        ),
      );
      return true;
    };
    const onPaste = (e: ClipboardEvent) => {
      if (take(e.clipboardData?.files)) e.preventDefault();
    };
    const onDragOver = (e: DragEvent) => e.preventDefault();
    const onDrop = (e: DragEvent) => {
      e.preventDefault();
      take(e.dataTransfer?.files);
    };
    document.addEventListener("paste", onPaste);
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("drop", onDrop);
    return () => {
      document.removeEventListener("paste", onPaste);
      window.removeEventListener("dragover", onDragOver);
      window.removeEventListener("drop", onDrop);
    };
  }, [focused, canAttach]);

  // On the web, Enter sends (Shift+Enter is a newline), or Cmd/Ctrl+Enter
  // does when Settings says so, as in the web UI.
  const onKeyPress = (
    e: NativeSyntheticEvent<
      TextInputKeyPressEventData & { shiftKey?: boolean; metaKey?: boolean; ctrlKey?: boolean }
    >,
  ) => {
    const k = e.nativeEvent;
    if (Platform.OS !== "web" || k.key !== "Enter") return;
    const mod = !!(k.metaKey || k.ctrlKey);
    if (prefs.sendWith === "mod-enter" ? !mod : k.shiftKey) return;
    e.preventDefault();
    submit();
  };

  const header = (
    <Drawer.Screen
      options={{
        headerTitle: () => <ModelTitle conversation={id} />,
        headerRight: () => <HeaderActions />,
      }}
    />
  );

  if (missing) {
    return (
      <View style={[styles.center, { backgroundColor: c.background }]}>
        {header}
        <Text style={{ color: c.textMuted }}>This conversation no longer exists.</Text>
      </View>
    );
  }

  const input = (
    <TextInput
      ref={composer}
      // A desktop page is for typing: the composer has the focus from the start.
      autoFocus={Platform.OS === "web" && wide}
      value={draft}
      onChangeText={setDraft}
      onKeyPress={onKeyPress}
      {...(Platform.OS === "web"
        ? {
            numberOfLines: 1,
            onContentSizeChange: (e: { nativeEvent: { contentSize: { height: number } } }) =>
              setInputHeight(e.nativeEvent.contentSize.height),
          }
        : {})}
      placeholder={!model ? "Choose a model first" : generating ? PLACEHOLDERS[media.kind] : "Ask anything"}
      placeholderTextColor={c.textMuted}
      multiline
      // Android's full-screen keyboard in landscape would hide the chat.
      disableFullscreenUI
      // VoiceOver and TalkBack read the placeholder as the field's name; a
      // label too would be read twice ("Message, Ask anything").
      accessibilityLabel={Platform.OS === "web" ? (generating ? "Prompt" : "Message") : undefined}
      style={[
        styles.input,
        generating && styles.inputWide,
        { color: c.text },
        Platform.OS === "web" && inputHeight ? { height: Math.min(160, Math.max(40, inputHeight)) } : null,
      ]}
    />
  );

  const sendButton = busy ? (
    <Pressable
      onPress={() => conv && stop(conv.id)}
      accessibilityRole="button"
      accessibilityLabel="Stop"
      style={[styles.send, { backgroundColor: c.text }]}
    >
      <Icon name={{ ios: "stop.fill", android: "stop" }} color={c.background} size={14} />
    </Pressable>
  ) : (
    <Pressable
      onPress={() => submit()}
      disabled={!canSend}
      accessibilityRole="button"
      accessibilityLabel={generating ? "Generate" : "Send"}
      accessibilityState={{ disabled: !canSend }}
      style={[styles.send, { backgroundColor: canSend ? c.accent : c.surfaceAlt }]}
    >
      <Icon
        name={{ ios: "arrow.up", android: "arrow_upward" }}
        color={canSend ? c.onAccent : c.textMuted}
        size={18}
      />
    </Pressable>
  );

  const messages = conv?.messages ?? [];
  const lastIndex = messages.length - 1;
  // Newest first: the list is inverted so it stays pinned to the bottom.
  const reversed = [...messages].reverse();

  return (
    <View
      key={sizeKey}
      ref={shell}
      style={{ flex: 1, backgroundColor: c.background }}
      onLayout={() => shell.current?.measureInWindow((_x, y) => setTop(y))}
    >
      {header}
      {/* The keyboard offset is where this screen starts on screen, measured:
        the header height is short of the drawn header in Android landscape. */}
      <KeyboardAvoidingView style={{ flex: 1 }} behavior="padding" keyboardVerticalOffset={top}>
        {messages.length === 0 ? (
          // Not the inverted list's empty view: Android inverts that on both axes.
          <View style={styles.empty}>
            <Image
              source={require("@/assets/llmman-mark.png")}
              style={styles.mark}
              resizeMode="contain"
              accessibilityIgnoresInvertColors
              accessible={false}
            />
            <Text style={[styles.greeting, { color: c.text }]} accessibilityRole="header">
              {!model
                ? "Choose a model to start"
                : generating
                  ? MEDIA_GREETING[media.kind]
                  : // "Good afternoon, Ricky", by the time of day, as the web UI greets.
                    greeting(prefs.name, hour)}
            </Text>
            {model ? (
              <View style={styles.chips}>
                {(generating ? MEDIA_SUGGESTIONS[media.kind] : SUGGESTIONS).map((s) => (
                  <Pressable
                    key={s}
                    onPress={() => submit(s)}
                    accessibilityRole="button"
                    style={({ pressed }) => [
                      styles.chip,
                      { borderColor: c.border, backgroundColor: pressed ? c.surface : c.background },
                    ]}
                  >
                    <Text style={{ color: c.text, fontSize: font.small + 1 }}>{s}</Text>
                  </Pressable>
                ))}
              </View>
            ) : null}
          </View>
        ) : (
          <FlatList
            inverted
            data={reversed}
            keyExtractor={(_, i) => String(messages.length - 1 - i)}
            contentContainerStyle={[styles.list, readable]}
            keyboardDismissMode="interactive"
            keyboardShouldPersistTaps="handled"
            renderItem={({ item, index }) => {
              const k = messages.length - 1 - index;
              const isLast = k === lastIndex;
              return (
                <MessageView
                  message={item}
                  live={busy && isLast && item.role === "assistant"}
                  last={isLast}
                  phase={phase}
                  generation={isLast ? generation : undefined}
                  modelName={item.model ? models.displayName(item.model) : undefined}
                  onRetry={
                    !busy && item.role === "assistant" && client
                      ? () => void retry(client, conv!.id, turn(), k)
                      : undefined
                  }
                  onEdit={!busy && item.role === "user" ? () => void edit(k) : undefined}
                  onFixKey={
                    active
                      ? () => router.push({ pathname: "/server/[id]", params: { id: active.id } })
                      : undefined
                  }
                />
              );
            }}
          />
        )}

        {elsewhere && !known ? (
          <View style={[styles.gone, { paddingHorizontal: side }]} accessibilityLiveRegion="polite">
            <Text style={{ color: c.textMuted, fontSize: font.small + 1, flexShrink: 1 }}>
              This chat was with {hostLabel(conv!.server)}, which is not one of your servers here
            </Text>
          </View>
        ) : gone ? (
          <View style={[styles.gone, { paddingHorizontal: side }]} accessibilityLiveRegion="polite">
            <Text style={{ color: c.textMuted, fontSize: font.small + 1, flexShrink: 1 }}>
              {models.displayName(model)} is no longer available; pick a model
            </Text>
            <Pressable
              onPress={() => router.push({ pathname: "/model-picker", params: { conversation: id } })}
              accessibilityRole="button"
              hitSlop={8}
              style={[styles.goneButton, { borderColor: c.border }]}
            >
              <Text style={{ color: c.text, fontSize: font.small, fontWeight: "600" }}>Pick a model</Text>
            </Pressable>
          </View>
        ) : null}
        <View
          style={{
            paddingHorizontal: side,
            paddingTop: spacing.xs,
            // The keyboard covers the home indicator area while it is up.
            paddingBottom: keyboardShown ? spacing.sm : Math.max(insets.bottom, spacing.md),
          }}
        >
          <View style={[styles.pill, { backgroundColor: c.surface, borderColor: c.border }]}>
            {images.length ? (
              <ScrollView
                horizontal
                contentContainerStyle={styles.thumbs}
                keyboardShouldPersistTaps="handled"
              >
                {images.map((uri, i) => (
                  <View key={i}>
                    <Image
                      source={{ uri }}
                      style={styles.thumb}
                      accessibilityLabel={`Attached image ${i + 1}`}
                    />
                    <Pressable
                      onPress={() => setImages((list) => list.filter((_, k) => k !== i))}
                      accessibilityRole="button"
                      accessibilityLabel={`Remove image ${i + 1}`}
                      hitSlop={8}
                      style={[styles.unattach, { backgroundColor: c.text }]}
                    >
                      <Icon name={{ ios: "xmark", android: "close" }} color={c.background} size={10} />
                    </Pressable>
                  </View>
                ))}
              </ScrollView>
            ) : null}
            {generating ? (
              <>
                {input}
                <View style={styles.toolRow}>
                  <Pressable
                    onPress={() =>
                      conv &&
                      router.push({ pathname: "/generate-settings", params: { conversation: conv.id } })
                    }
                    accessibilityRole="button"
                    accessibilityLabel="Generation settings"
                    hitSlop={6}
                    style={styles.plus}
                  >
                    <Icon name={{ ios: "slider.horizontal.3", android: "tune" }} color={c.text} size={19} />
                  </Pressable>
                  <KindToggle
                    kinds={mediaKinds}
                    value={media.kind}
                    onChange={(kind) => conv && setMediaOptions(conv.id, { ...media, kind })}
                  />
                  <View style={{ flex: 1 }} />
                  {sendButton}
                </View>
              </>
            ) : (
              <View style={styles.pillRow}>
                <Pressable
                  onPress={attach}
                  disabled={!canAttach || attaching || images.length >= MAX_IMAGES}
                  accessibilityRole="button"
                  accessibilityLabel="Attach images"
                  accessibilityHint={canAttach ? undefined : "This model does not take images"}
                  accessibilityState={{ disabled: !canAttach || attaching }}
                  style={[styles.plus, { opacity: canAttach ? 1 : 0.35 }]}
                >
                  {attaching ? (
                    <ActivityIndicator size="small" />
                  ) : (
                    <Icon name={{ ios: "plus", android: "add" }} color={c.text} size={20} />
                  )}
                </Pressable>
                {input}
                {sendButton}
              </View>
            )}
          </View>
        </View>
      </KeyboardAvoidingView>
    </View>
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, alignItems: "center", justifyContent: "center" },
  gone: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    flexWrap: "wrap",
    gap: spacing.sm,
    paddingVertical: spacing.xs,
  },
  goneButton: {
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: radius.pill,
    paddingHorizontal: spacing.md,
    paddingVertical: 4,
  },
  list: { paddingHorizontal: spacing.lg, paddingVertical: spacing.lg, gap: spacing.xl, flexGrow: 1 },
  // Only horizontal padding: in landscape with the keyboard up it must shrink to nothing.
  empty: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: spacing.xl,
    gap: spacing.lg,
    overflow: "hidden",
  },
  mark: { width: 72, height: 52 },
  greeting: { fontSize: 22, fontWeight: "600", textAlign: "center" },
  chips: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "center",
    gap: spacing.sm,
    maxWidth: 520,
  },
  chip: {
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: radius.pill,
    paddingHorizontal: spacing.md + 2,
    paddingVertical: spacing.sm,
  },
  pill: {
    width: "100%",
    maxWidth: readable.maxWidth,
    alignSelf: "center",
    borderRadius: 26,
    borderWidth: StyleSheet.hairlineWidth,
    paddingLeft: 6,
    paddingRight: 6,
    paddingVertical: 6,
    gap: spacing.xs,
  },
  pillRow: { flexDirection: "row", alignItems: "flex-end", gap: spacing.xs },
  toolRow: { flexDirection: "row", alignItems: "center", gap: spacing.xs },
  plus: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 2,
  },
  thumbs: { gap: spacing.sm, padding: spacing.xs },
  thumb: { width: 56, height: 56, borderRadius: radius.md },
  unattach: {
    position: "absolute",
    top: -4,
    right: -4,
    width: 18,
    height: 18,
    borderRadius: 9,
    alignItems: "center",
    justifyContent: "center",
  },
  input: {
    flex: 1,
    minHeight: 40,
    maxHeight: 160,
    paddingTop: 10,
    paddingBottom: 10,
    fontSize: font.body,
    lineHeight: font.body + 5,
    // The pill is the focus frame; no browser outline inside it.
    outlineWidth: 0,
    ...(Platform.OS === "web" ? ({ outlineStyle: "none" } as object) : {}),
  },
  inputWide: { flex: 0, paddingHorizontal: spacing.sm + 2 },
  send: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 2,
  },
});
