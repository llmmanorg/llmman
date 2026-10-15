// Draws lib/dialog's dialogs in the page (the web app's alerts, confirmations
// and key prompt), so they work in any browser, web view or iframe.

import { useEffect, useState, useSyncExternalStore } from "react";
import { Image, Modal, Platform, Pressable, StyleSheet, Text, TextInput, View } from "react-native";

import {
  cancelIndex,
  currentDialog,
  primaryIndex,
  resolveDialog,
  subscribeDialogs,
  type OpenDialog,
} from "@/lib/dialog";
import { font, radius, spacing, useTheme } from "@/lib/theme";

export function DialogHost() {
  const dialog = useSyncExternalStore(subscribeDialogs, currentDialog, currentDialog);
  // Keyed by id: each dialog starts with an empty field.
  return dialog ? <Dialog key={dialog.id} dialog={dialog} /> : null;
}

function Dialog({ dialog }: { dialog: OpenDialog }) {
  const c = useTheme();
  const [text, setText] = useState("");
  const choose = (button: number) => resolveDialog(dialog.id, { button, text: text.trim() });
  const cancel = () => choose(cancelIndex(dialog));
  const primary = primaryIndex(dialog);

  // Escape cancels, as in any dialog; Enter is handled by the field below.
  useEffect(() => {
    if (typeof document === "undefined") return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        resolveDialog(dialog.id, { button: cancelIndex(dialog), text: "" });
      } else if (e.key === "Enter" && !dialog.input && primary >= 0) {
        e.preventDefault();
        resolveDialog(dialog.id, { button: primary, text: "" });
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [dialog, primary]);

  return (
    <Modal transparent visible animationType="fade" onRequestClose={cancel}>
      <Pressable
        style={styles.backdrop}
        onPress={cancel}
        accessibilityLabel="Close"
        accessibilityRole="button"
      >
        <Pressable
          style={[styles.card, { backgroundColor: c.background, borderColor: c.border }]}
          onPress={(e) => e.stopPropagation()}
          accessibilityRole="alert"
          accessible={false}
        >
          {dialog.brand ? (
            <View style={styles.brand}>
              <Image
                source={require("@/assets/llmman-mark.png")}
                style={styles.mark}
                resizeMode="contain"
                accessible={false}
              />
              <Text style={[styles.brandName, { color: c.text }]}>llmman</Text>
            </View>
          ) : null}
          <Text style={[styles.title, { color: c.text }]} accessibilityRole="header">
            {dialog.title}
          </Text>
          {dialog.message ? (
            <Text style={[styles.message, { color: c.textMuted }]}>{dialog.message}</Text>
          ) : null}
          {dialog.input ? (
            <TextInput
              value={text}
              onChangeText={setText}
              autoFocus
              secureTextEntry={dialog.input.secure}
              placeholder={dialog.input.placeholder}
              placeholderTextColor={c.textMuted}
              accessibilityLabel={dialog.input.label ?? dialog.input.placeholder ?? dialog.title}
              autoCapitalize="none"
              autoCorrect={false}
              onSubmitEditing={() => primary >= 0 && text.trim() && choose(primary)}
              style={[styles.input, { color: c.text, backgroundColor: c.surface, borderColor: c.border }]}
            />
          ) : null}
          {dialog.hint ? <Text style={[styles.hint, { color: c.textMuted }]}>{dialog.hint}</Text> : null}
          <View style={styles.buttons}>
            {dialog.buttons.map((b, i) => {
              const main = i === primary;
              const danger = b.style === "destructive";
              return (
                <Pressable
                  key={i}
                  onPress={() => choose(i)}
                  accessibilityRole="button"
                  disabled={main && !!dialog.input && !text.trim()}
                  style={({ pressed }) => [
                    styles.button,
                    {
                      backgroundColor: main ? (danger ? c.danger : c.accent) : "transparent",
                      borderColor: main ? "transparent" : c.border,
                      opacity: pressed || (main && !!dialog.input && !text.trim()) ? 0.6 : 1,
                    },
                  ]}
                >
                  <Text
                    style={{
                      color: main ? (danger ? "#fff" : c.onAccent) : danger ? c.danger : c.text,
                      fontWeight: "600",
                    }}
                  >
                    {b.text}
                  </Text>
                </Pressable>
              );
            })}
          </View>
        </Pressable>
      </Pressable>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.4)",
    alignItems: "center",
    justifyContent: "center",
    padding: spacing.lg,
  },
  card: {
    width: "100%",
    maxWidth: 420,
    borderRadius: radius.lg,
    borderWidth: StyleSheet.hairlineWidth,
    padding: spacing.xl,
    gap: spacing.md,
    shadowColor: "#000",
    shadowOpacity: 0.2,
    shadowRadius: 24,
    shadowOffset: { width: 0, height: 8 },
  },
  title: { fontSize: font.body + 1, fontWeight: "700" },
  brand: { flexDirection: "row", alignItems: "center", gap: spacing.sm, marginBottom: spacing.xs },
  mark: { width: 34, height: 25 },
  brandName: { fontSize: font.body, fontWeight: "700" },
  hint: { fontSize: font.small, lineHeight: 18, marginTop: -spacing.xs },
  message: { fontSize: font.small + 1, lineHeight: 20 },
  input: {
    minHeight: 44,
    borderRadius: radius.md,
    borderWidth: StyleSheet.hairlineWidth,
    paddingHorizontal: spacing.md,
    fontSize: font.body,
    ...(Platform.OS === "web" ? ({ outlineStyle: "none" } as object) : {}),
  },
  buttons: { flexDirection: "row", justifyContent: "flex-end", gap: spacing.sm, marginTop: spacing.xs },
  button: {
    borderWidth: 1,
    borderRadius: radius.md,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm + 2,
  },
});
