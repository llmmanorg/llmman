// Small shared controls, themed and labelled for screen readers.

import { useState } from "react";
import type { ReactNode, Ref } from "react";
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
  type StyleProp,
  type TextInputProps,
  type ViewStyle,
} from "react-native";

import { Icon } from "@/components/Icon";
import { font, radius, spacing, useTheme } from "@/lib/theme";

type ButtonKind = "primary" | "secondary" | "danger" | "plain";

export function Button({
  title,
  onPress,
  kind = "primary",
  busy = false,
  disabled = false,
  icon,
  accessibilityHint,
  style,
}: {
  title: string;
  onPress: () => void;
  kind?: ButtonKind;
  busy?: boolean;
  disabled?: boolean;
  icon?: ReactNode;
  accessibilityHint?: string;
  style?: StyleProp<ViewStyle>;
}) {
  const c = useTheme();
  const bg = { primary: c.accent, secondary: c.surfaceAlt, danger: c.surfaceAlt, plain: "transparent" }[kind];
  const fg = { primary: c.onAccent, secondary: c.text, danger: c.danger, plain: c.accent }[kind];
  const off = disabled || busy;
  return (
    <Pressable
      onPress={onPress}
      disabled={off}
      accessibilityRole="button"
      accessibilityLabel={title}
      accessibilityHint={accessibilityHint}
      accessibilityState={{ disabled: off, busy }}
      style={({ pressed }) => [
        styles.button,
        { backgroundColor: bg, opacity: disabled ? 0.45 : pressed ? 0.75 : 1 },
        style,
      ]}
    >
      {busy ? <ActivityIndicator color={fg} /> : icon}
      <Text style={[styles.buttonText, { color: fg }]}>{title}</Text>
    </Pressable>
  );
}

export function Field({
  label,
  error,
  secure = false,
  hint,
  ref,
  style,
  labelHidden = false,
  ...input
}: TextInputProps & {
  label: string;
  error?: string | null;
  secure?: boolean;
  hint?: string;
  ref?: Ref<TextInput>;
  /** The field's heading already names it: the label is for screen readers only. */
  labelHidden?: boolean;
}) {
  const c = useTheme();
  const [shown, setShown] = useState(false);
  return (
    <View style={styles.field}>
      {labelHidden ? null : <Text style={[styles.label, { color: c.textMuted }]}>{label}</Text>}
      <View
        style={[styles.inputRow, { backgroundColor: c.surface, borderColor: error ? c.danger : c.border }]}
      >
        <TextInput
          ref={ref}
          accessibilityLabel={label}
          accessibilityHint={hint}
          placeholderTextColor={c.textMuted}
          autoCapitalize="none"
          autoCorrect={false}
          spellCheck={false}
          secureTextEntry={secure && !shown}
          {...input}
          style={[styles.input, { color: c.text }, style]}
        />
        {secure ? (
          <Pressable
            onPress={() => setShown((s) => !s)}
            accessibilityRole="button"
            accessibilityLabel={shown ? `Hide ${label}` : `Show ${label}`}
            hitSlop={8}
            style={styles.reveal}
          >
            <Icon
              name={
                shown
                  ? { ios: "eye.slash", android: "visibility_off" }
                  : { ios: "eye", android: "visibility" }
              }
              color={c.textMuted}
              size={20}
            />
          </Pressable>
        ) : null}
      </View>
      {error ? (
        <Text style={[styles.error, { color: c.danger }]} accessibilityLiveRegion="polite">
          {error}
        </Text>
      ) : hint ? (
        <Text style={[styles.hint, { color: c.textMuted }]}>{hint}</Text>
      ) : null}
    </View>
  );
}

/** A boxed message: an error with what to do about it, or a note. */
export function Notice({
  tone,
  title,
  children,
}: {
  tone: "error" | "info";
  title?: string;
  children: ReactNode;
}) {
  const c = useTheme();
  const color = tone === "error" ? c.danger : c.textMuted;
  return (
    <View
      style={[
        styles.notice,
        { backgroundColor: c.surface, borderColor: tone === "error" ? c.danger : c.border },
      ]}
      accessibilityLiveRegion="polite"
      accessibilityRole={tone === "error" ? "alert" : undefined}
    >
      {title ? <Text style={[styles.noticeTitle, { color }]}>{title}</Text> : null}
      <Text style={{ color: c.text, fontSize: font.small + 1, lineHeight: 20 }}>{children}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  button: {
    minHeight: 48,
    borderRadius: radius.md,
    paddingHorizontal: spacing.lg,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.sm,
  },
  buttonText: { fontSize: font.body, fontWeight: "600" },
  field: { gap: spacing.xs },
  label: { fontSize: font.small, fontWeight: "600", textTransform: "uppercase", letterSpacing: 0.5 },
  inputRow: {
    flexDirection: "row",
    alignItems: "center",
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: radius.md,
  },
  input: { flex: 1, minHeight: 48, paddingHorizontal: spacing.md, fontSize: font.body },
  reveal: { paddingHorizontal: spacing.md, minHeight: 48, justifyContent: "center" },
  error: { fontSize: font.small },
  hint: { fontSize: font.small },
  notice: {
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: radius.md,
    padding: spacing.md,
    gap: spacing.xs,
  },
  noticeTitle: { fontWeight: "600", fontSize: font.small + 1 },
});
