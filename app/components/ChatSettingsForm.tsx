// System prompt, temperature and max tokens, used for a chat's own
// settings and for a server's defaults for new chats.

import { useState } from "react";
import { StyleSheet, View } from "react-native";

import { Button, Field } from "@/components/ui";
import { parseSettings, toInput, type ChatSettings, type SettingsErrors } from "@/lib/chatSettings";
import { spacing } from "@/lib/theme";

export function ChatSettingsForm({
  initial,
  saveTitle,
  onSave,
}: {
  initial: ChatSettings | undefined;
  saveTitle: string;
  onSave: (settings: ChatSettings) => void | Promise<void>;
}) {
  const [input, setInput] = useState(() => toInput(initial));
  const [errors, setErrors] = useState<SettingsErrors>({});
  const [busy, setBusy] = useState(false);

  const save = async () => {
    const parsed = parseSettings(input);
    if (!parsed.ok) {
      setErrors(parsed.errors);
      return;
    }
    setBusy(true);
    try {
      await onSave(parsed.value);
    } finally {
      setBusy(false);
    }
  };

  const change = (k: keyof typeof input) => (text: string) => {
    setInput((v) => ({ ...v, [k]: text }));
    if (k in errors) setErrors(({ [k as keyof SettingsErrors]: _gone, ...rest }) => rest);
  };

  return (
    <View style={styles.root}>
      <Field
        label="System prompt"
        value={input.systemPrompt}
        onChangeText={change("systemPrompt")}
        placeholder="Optional instructions sent with every message"
        multiline
        autoCapitalize="sentences"
        autoCorrect
        spellCheck
        style={styles.prompt}
      />
      <Field
        label="Temperature"
        value={input.temperature}
        onChangeText={change("temperature")}
        placeholder="Model default"
        keyboardType="decimal-pad"
        error={errors.temperature}
        hint="0 to 2. Lower is more focused, higher more varied."
      />
      <Field
        label="Max tokens"
        value={input.maxTokens}
        onChangeText={change("maxTokens")}
        placeholder="No limit"
        keyboardType="number-pad"
        error={errors.maxTokens}
        hint="The longest reply, in tokens. Stops a model that runs on."
      />
      <Button title={saveTitle} onPress={save} busy={busy} />
    </View>
  );
}

const styles = StyleSheet.create({
  root: { gap: spacing.lg },
  prompt: { minHeight: 96, textAlignVertical: "top", paddingTop: 12 },
});
