// Add a server: type its address and an API key. The connection is tested
// before anything is saved.

import { useEffect, useRef, useState } from "react";
import { StyleSheet, View, type TextInput } from "react-native";

import { isAbort } from "@/api/client";
import { normalizeServerUrl } from "@/api/serverUrl";
import { Button, Field, Notice } from "@/components/ui";
import { checkServer, type ConnectOutcome } from "@/lib/connect";
import { useServers } from "@/lib/ServersContext";
import { spacing } from "@/lib/theme";

/** A failed attempt, and whether it sent a key ("Wrong API key" vs "API key needed"). */
type Failure = Extract<ConnectOutcome, { ok: false }> & { triedKey: boolean };

export function ConnectForm({ onConnected }: { onConnected: () => void }) {
  const { save, servers } = useServers();
  const [address, setAddress] = useState("");
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<Failure | null>(null);
  const abort = useRef<AbortController | null>(null);
  const keyInput = useRef<TextInput>(null);

  useEffect(() => () => abort.current?.abort(), []);

  const submit = async () => {
    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;
    setBusy(true);
    setFailure(null);
    try {
      const outcome = await checkServer({ address, key }, { signal: controller.signal });
      if (!outcome.ok) {
        setFailure({ ...outcome, triedKey: Boolean(key.trim()) });
        if (outcome.field === "key") keyInput.current?.focus();
        return;
      }
      await save({ url: outcome.url, key: outcome.key });
      onConnected();
    } catch (e) {
      if (!isAbort(e)) {
        setFailure({ ok: false, field: null, problem: "server-error", message: String(e), triedKey: false });
      }
    } finally {
      if (abort.current === controller) {
        abort.current = null;
        setBusy(false);
      }
    }
  };

  const cancel = () => {
    abort.current?.abort();
    abort.current = null;
    setBusy(false);
  };

  const typed = normalizeServerUrl(address);
  const known = typed.ok && servers.some((s) => s.url === typed.value);

  return (
    <View style={styles.root}>
      <Field
        label="Address"
        value={address}
        onChangeText={(t) => {
          setAddress(t);
          if (failure?.field === "address") setFailure(null);
        }}
        placeholder="192.168.1.20:17434"
        keyboardType="url"
        textContentType="URL"
        returnKeyType="next"
        onSubmitEditing={() => keyInput.current?.focus()}
        editable={!busy}
        error={failure?.field === "address" && failure.problem === "invalid" ? failure.message : null}
        hint="The computer running llmman serve, as host:port"
      />
      <Field
        ref={keyInput}
        label="API key"
        value={key}
        onChangeText={(t) => {
          setKey(t);
          if (failure?.field === "key") setFailure(null);
        }}
        placeholder="From LLMMAN_API_KEYS"
        secure
        textContentType="password"
        returnKeyType="go"
        onSubmitEditing={() => void submit()}
        editable={!busy}
        hint="Leave empty if the server has no keys"
      />

      {failure && failure.problem !== "invalid" ? (
        <Notice tone="error" title={titleFor(failure, failure.triedKey)}>
          {failure.message}
        </Notice>
      ) : null}

      {busy ? (
        <View style={styles.row}>
          <Button title="Connecting…" busy onPress={() => {}} style={styles.flex} />
          <Button title="Cancel" kind="secondary" onPress={cancel} />
        </View>
      ) : (
        <Button
          title={known ? "Update server" : "Connect"}
          onPress={() => void submit()}
          disabled={!address.trim()}
        />
      )}
    </View>
  );
}

function titleFor(f: Failure, triedKey: boolean): string {
  switch (f.problem) {
    case "unauthorized":
      return triedKey ? "Wrong API key" : "API key needed";
    case "unreachable":
      return "Can't reach the server";
    case "timeout":
      return "No answer";
    case "not-llmman":
      return "Not llmman";
    default:
      return "Couldn't connect";
  }
}

const styles = StyleSheet.create({
  root: { gap: spacing.lg },
  row: { flexDirection: "row", gap: spacing.sm },
  flex: { flex: 1 },
});
