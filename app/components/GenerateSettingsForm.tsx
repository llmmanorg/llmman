// A media generation run's options, as the web UI's settings popover has
// them with a diffusion model: size, seconds, steps, seed, guidance and a
// negative prompt. Blank is the model's default. Size is not asked for
// sound, nor seconds for a picture.

import { useState } from "react";
import { StyleSheet, Text, TextInput, View } from "react-native";

import { Button, Field } from "@/components/ui";
import {
  parseMediaOptions,
  toMediaInput,
  type MediaErrors,
  type MediaKind,
  type MediaOptions,
  type MediaOptionsInput,
} from "@/lib/media";
import { font, radius, spacing, useTheme } from "@/lib/theme";

export function GenerateSettingsForm({
  initial,
  kind,
  onSave,
}: {
  initial: MediaOptions | undefined;
  kind: MediaKind;
  onSave: (options: MediaOptions) => void;
}) {
  const c = useTheme();
  const [input, setInput] = useState(() => toMediaInput(initial));
  const [errors, setErrors] = useState<MediaErrors>({});

  const save = () => {
    const parsed = parseMediaOptions(input, kind);
    if (!parsed.ok) {
      setErrors(parsed.errors);
      return;
    }
    onSave(parsed.value);
  };

  const change = (k: keyof MediaOptionsInput) => (text: string) => {
    setInput((v) => ({ ...v, [k]: text }));
    // Width and height are checked together.
    setErrors((e) => {
      const next = { ...e };
      delete next[k];
      if (k === "width" || k === "height") delete next.height;
      return next;
    });
  };

  const visual = kind !== "audio";
  const timed = kind !== "image";
  const sizeError = errors.width ?? errors.height;

  return (
    <View style={styles.root}>
      {visual ? (
        <View style={styles.group}>
          <Text style={[styles.label, { color: c.textMuted }]}>Size</Text>
          <View style={styles.pair}>
            <TextInput
              value={input.width}
              onChangeText={change("width")}
              placeholder="768"
              placeholderTextColor={c.textMuted}
              keyboardType="number-pad"
              accessibilityLabel="Width"
              style={[
                styles.dim,
                {
                  color: c.text,
                  backgroundColor: c.surface,
                  borderColor: errors.width ? c.danger : c.border,
                },
              ]}
            />
            <Text style={[styles.times, { color: c.textMuted }]} accessible={false}>
              ×
            </Text>
            <TextInput
              value={input.height}
              onChangeText={change("height")}
              placeholder="512"
              placeholderTextColor={c.textMuted}
              keyboardType="number-pad"
              accessibilityLabel="Height"
              style={[
                styles.dim,
                {
                  color: c.text,
                  backgroundColor: c.surface,
                  borderColor: errors.height ? c.danger : c.border,
                },
              ]}
            />
          </View>
          <Text
            style={[styles.hint, { color: sizeError ? c.danger : c.textMuted }]}
            accessibilityLiveRegion="polite"
          >
            {sizeError ?? "In pixels, multiples of 32. Blank: the model's default."}
          </Text>
        </View>
      ) : null}
      {timed ? (
        <Field
          label="Seconds"
          value={input.seconds}
          onChangeText={change("seconds")}
          placeholder="Model default"
          keyboardType="decimal-pad"
          error={errors.seconds}
          hint="How long the clip runs, 0.5 to 42."
        />
      ) : null}
      <Field
        label="Steps"
        value={input.steps}
        onChangeText={change("steps")}
        placeholder="Model default"
        keyboardType="number-pad"
        error={errors.steps}
        hint="Denoising steps, 1 to 200. More is slower, often finer."
      />
      <Field
        label="Seed"
        value={input.seed}
        onChangeText={change("seed")}
        placeholder="Random"
        keyboardType="number-pad"
        error={errors.seed}
        hint="The same seed and settings give the same result."
      />
      <Field
        label="Guidance"
        value={input.cfgScale}
        onChangeText={change("cfgScale")}
        placeholder="Model default"
        keyboardType="decimal-pad"
        error={errors.cfgScale}
        hint="How closely to follow the prompt, 0 to 30 (cfg_scale)."
      />
      <Field
        label="Negative prompt"
        value={input.negativePrompt}
        onChangeText={change("negativePrompt")}
        placeholder="What to keep out"
        multiline
        autoCapitalize="sentences"
        autoCorrect
        spellCheck
        hint="Applies when guidance is above 1."
        style={styles.negative}
      />
      <View style={styles.buttons}>
        <Button title="Save" onPress={save} />
        <Button
          title="Use model defaults"
          kind="plain"
          onPress={() => {
            setInput(toMediaInput(undefined));
            setErrors({});
          }}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { gap: spacing.lg },
  group: { gap: spacing.xs },
  label: { fontSize: font.small, fontWeight: "600", textTransform: "uppercase", letterSpacing: 0.5 },
  pair: { flexDirection: "row", alignItems: "center", gap: spacing.sm },
  dim: {
    flex: 1,
    minHeight: 48,
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: radius.md,
    paddingHorizontal: spacing.md,
    fontSize: font.body,
  },
  times: { fontSize: font.body },
  hint: { fontSize: font.small },
  negative: { minHeight: 72, textAlignVertical: "top", paddingTop: 12 },
  buttons: { gap: spacing.xs },
});
