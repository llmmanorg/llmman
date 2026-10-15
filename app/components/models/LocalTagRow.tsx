// A model on this computer: whether it is loaded, what it is, and
// load / unload / delete.

import { useState } from "react";
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from "react-native";

import type { LocalModel, TagModel } from "@/api/client";
import { Icon } from "@/components/Icon";
import { useModels as useChatModels } from "@/lib/ModelsContext";
import { useServers } from "@/lib/ServersContext";
import { useOpenChat } from "@/lib/useOpenChat";
import { font, radius, spacing, useTheme } from "@/lib/theme";
import { useLargeText } from "./bits";
import { formatBytes, tagOf } from "./logic";
import { explain } from "@/api/errors";
import { alert } from "@/lib/alert";
import { useModelsTab } from "./ModelsTabContext";

/** `gguf · 0.8B · Q4_K_M · 505 MB`, from what `/api/tags` knows. */
export function tagMeta(d: TagModel | undefined): string {
  if (!d) return "";
  const x = (d.details ?? {}) as Record<string, unknown>;
  return [x.format, x.parameter_size, x.quantization_level, d.size ? formatBytes(d.size) : ""]
    .filter((v) => typeof v === "string" && v)
    .join(" · ");
}

export function LocalTagRow({ model, fullName = false }: { model: LocalModel; fullName?: boolean }) {
  const c = useTheme();
  const { client, active } = useServers();
  const { details, storeChanged } = useModelsTab();
  const large = useLargeText();
  const chat = useChatModels();
  const openChat = useOpenChat();
  // The chat's list holds the models that take a chat turn and the media
  // generation models; the action says which (the web UI's rule).
  const usable = chat.local.find((m) => m.id === model.id);
  const generates = !!usable?.kinds.length;
  const [busy, setBusy] = useState<"load" | "unload" | "delete" | null>(null);
  // The friendly name, then the reference it stands for.
  const name = chat.displayName(model.id);
  const meta = [fullName ? model.id : tagOf(model.id), tagMeta(details.get(model.id.toLowerCase()))]
    .filter(Boolean)
    .join(" · ");

  const act = async (kind: "load" | "unload" | "delete") => {
    if (!client) return;
    setBusy(kind);
    try {
      if (kind === "load") await client.loadModel(model.id);
      else if (kind === "unload") await client.unloadModel(model.id);
      else await client.deleteModel(model.id);
    } catch (e) {
      const what = { load: "load", unload: "unload", delete: "delete" }[kind];
      const { title, detail } = explain(e, active ?? { url: "The server" });
      alert(`Could not ${what} ${model.id}`, detail ? `${title}\n\n${detail}` : title);
    } finally {
      setBusy(null);
      storeChanged();
    }
  };

  const confirmDelete = () =>
    alert(`Delete ${model.id}?`, "It is removed from this computer's store. You can pull it again later.", [
      { text: "Cancel", style: "cancel" },
      { text: "Delete", style: "destructive", onPress: () => act("delete") },
    ]);

  return (
    // Large text puts the actions on a line of their own rather than squeezing the name.
    <View style={[large ? styles.stacked : styles.row, { borderColor: c.border }]}>
      {/* What it is, read as one element before its actions. */}
      <View
        style={[styles.info, !large && styles.grow]}
        accessible
        accessibilityLabel={[name, model.loaded ? "loaded" : "not loaded", meta].filter(Boolean).join(", ")}
      >
        <View style={[styles.dot, { backgroundColor: model.loaded ? c.success : c.border }]} />
        <View style={styles.text}>
          <Text style={[styles.name, { color: c.text }]} numberOfLines={large ? undefined : 2}>
            {name}
          </Text>
          {meta ? <Text style={[styles.meta, { color: c.textMuted }]}>{meta}</Text> : null}
        </View>
      </View>
      <View style={styles.actions}>
        {usable ? (
          <RowAction
            label={generates ? `Generate with ${model.id}` : `Chat with ${model.id}`}
            icon={generates ? { ios: "photo", android: "image" } : { ios: "bubble.left", android: "chat" }}
            busy={false}
            disabled={!!busy}
            onPress={() => openChat(model.id)}
          />
        ) : null}
        <RowAction
          label={model.loaded ? `Unload ${model.id}` : `Load ${model.id}`}
          icon={model.loaded ? { ios: "eject", android: "eject" } : { ios: "play", android: "play_arrow" }}
          busy={busy === "load" || busy === "unload"}
          disabled={!!busy}
          onPress={() => act(model.loaded ? "unload" : "load")}
        />
        <RowAction
          label={`Delete ${model.id}`}
          icon={{ ios: "trash", android: "delete" }}
          busy={busy === "delete"}
          disabled={!!busy}
          danger
          onPress={confirmDelete}
        />
      </View>
    </View>
  );
}

function RowAction({
  label,
  icon,
  busy,
  disabled,
  danger = false,
  onPress,
}: {
  label: string;
  icon: {
    ios: "eject" | "play" | "trash" | "bubble.left" | "photo";
    android: "eject" | "play_arrow" | "delete" | "chat" | "image";
  };
  busy: boolean;
  disabled: boolean;
  danger?: boolean;
  onPress: () => void;
}) {
  const c = useTheme();
  const color = danger ? c.danger : c.accent;
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      accessibilityRole="button"
      accessibilityLabel={label}
      accessibilityState={{ disabled, busy }}
      hitSlop={6}
      style={({ pressed }) => [
        styles.action,
        { backgroundColor: c.surface, opacity: disabled && !busy ? 0.4 : pressed ? 0.6 : 1 },
      ]}
    >
      {busy ? <ActivityIndicator color={color} /> : <Icon name={icon} size={18} color={color} />}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    paddingVertical: spacing.sm,
    minHeight: 56,
  },
  stacked: { gap: spacing.sm, paddingVertical: spacing.sm },
  actions: { flexDirection: "row", gap: spacing.md },
  info: { flexDirection: "row", alignItems: "center", gap: spacing.md },
  grow: { flex: 1 },
  dot: { width: 10, height: 10, borderRadius: 5 },
  text: { flex: 1, gap: 2 },
  name: { fontSize: font.body, fontWeight: "500" },
  meta: { fontSize: font.small },
  action: { width: 40, height: 40, borderRadius: radius.md, alignItems: "center", justifyContent: "center" },
});
