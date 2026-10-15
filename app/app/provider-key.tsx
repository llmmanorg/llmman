// A provider's API key: added or replaced on the computer running llmman
// (sent once over the server connection, never kept on this device), or
// removed from it.

import { router, Stack, useLocalSearchParams } from "expo-router";
import { useState } from "react";
import { ScrollView, Text } from "react-native";
import { KeyboardAvoidingView } from "react-native-keyboard-controller";

import { explain } from "@/api/errors";
import { hostLabel } from "@/api/serverUrl";
import { Button, Field, Notice } from "@/components/ui";
import { alert } from "@/lib/alert";
import { useModels } from "@/lib/ModelsContext";
import { useServers } from "@/lib/ServersContext";
import { font, readable, spacing, useTheme, useTextSizeKey } from "@/lib/theme";

export default function ProviderKey() {
  const c = useTheme();
  const sizeKey = useTextSizeKey();
  const {
    id = "",
    name = "",
    env = "",
    set,
  } = useLocalSearchParams<{
    id: string;
    name: string;
    env?: string;
    set?: string;
  }>();
  const { client, active } = useServers();
  const models = useModels();
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState<"save" | "remove" | null>(null);
  const [error, setError] = useState<{ title: string; detail?: string } | null>(null);
  const computer = active?.name || (active ? hostLabel(active.url) : "the computer");

  const run = async (what: "save" | "remove") => {
    if (!client || !active) return;
    setBusy(what);
    setError(null);
    try {
      const r =
        what === "save" ? await client.setProviderKey(id, key.trim()) : await client.deleteProviderKey(id);
      // The chat's model list picks up (or drops) the provider's models.
      void models.refresh();
      setKey("");
      if (r.env_override) {
        alert(
          what === "save" ? "Saved, but not in use" : "Removed",
          `${r.key_env} is set on ${computer}, and it is used instead of the key stored here.`,
        );
      }
      router.back();
    } catch (e) {
      const { title, detail } = explain(e, active);
      setError({ title, ...(detail ? { detail } : {}) });
    } finally {
      setBusy(null);
    }
  };

  const confirmRemove = () =>
    alert(`Remove the ${name} key?`, `It is deleted from ${computer}'s llmman.conf.`, [
      { text: "Cancel", style: "cancel" },
      { text: "Remove", style: "destructive", onPress: () => void run("remove") },
    ]);

  return (
    <KeyboardAvoidingView key={sizeKey} style={{ flex: 1, backgroundColor: c.background }} behavior="padding">
      <Stack.Screen options={{ title: name || id }} />
      <ScrollView
        contentContainerStyle={[{ padding: spacing.xl, gap: spacing.lg }, readable]}
        keyboardShouldPersistTaps="handled"
      >
        <Text style={{ color: c.textMuted, fontSize: font.small + 1, lineHeight: 20 }}>
          The key is sent once to {computer} and stored there, in llmman.conf (readable only by its owner).
          This device does not keep it.
          {env ? ` If ${env} is set on ${computer}, that one is used instead.` : ""}
        </Text>
        {set ? (
          <Notice tone="info" title="A key is set">
            Enter a new one to replace it.
          </Notice>
        ) : null}
        <Field
          label="API key"
          value={key}
          onChangeText={setKey}
          secure
          autoCapitalize="none"
          autoCorrect={false}
          autoComplete="off"
          placeholder={env || "API key"}
          onSubmitEditing={() => key.trim() && void run("save")}
        />
        {error ? (
          <Notice tone="error" title={error.title}>
            {error.detail ?? ""}
          </Notice>
        ) : null}
        <Button
          title={set ? "Replace key" : "Save key"}
          onPress={() => void run("save")}
          busy={busy === "save"}
          disabled={!key.trim() || !!busy}
        />
        {set ? (
          <Button
            title="Remove key"
            kind="danger"
            onPress={confirmRemove}
            busy={busy === "remove"}
            disabled={!!busy}
          />
        ) : null}
      </ScrollView>
    </KeyboardAvoidingView>
  );
}
