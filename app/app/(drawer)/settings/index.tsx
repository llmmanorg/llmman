// Settings, as the web UI's Settings dialog has them: theme, your name for
// the greeting, what sends a message (web), the daemon's key (web; a phone
// keeps one per server), and the conversations stored here.

import { useEffect, useState } from "react";
import { Platform, ScrollView, StyleSheet, Text, View } from "react-native";

import { hostLabel } from "@/api/serverUrl";
import { Segments } from "@/components/models/bits";
import { Button, Field } from "@/components/ui";
import { alert } from "@/lib/alert";
import { dropConversations } from "@/lib/chatStore";
import { useDaemon } from "@/lib/DaemonContext";
import { allConversations, deleteAllConversations, exportJson } from "@/lib/conversations";
import { setPageKey } from "@/lib/page";
import { setPrefs, usePrefs, type SendWith, type ThemePref } from "@/lib/prefs";
import { saveFile } from "@/lib/saveFile";
import { useServers } from "@/lib/ServersContext";
import { font, readable, spacing, useTheme, useTextSizeKey } from "@/lib/theme";

const THEMES = [
  { value: "system", label: "System" },
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
] as const;

const SEND_WITH = [
  { value: "enter", label: "Enter" },
  {
    value: "mod-enter",
    label: Platform.OS === "web" && /Mac/.test(navigator.platform) ? "⌘ Enter" : "Ctrl+Enter",
  },
] as const;

export default function Settings() {
  const c = useTheme();
  const sizeKey = useTextSizeKey();
  const prefs = usePrefs();
  const { active, save, keyFor } = useServers();
  const daemon = useDaemon();
  const [name, setName] = useState(prefs.name);
  const [key, setKey] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // The web shows the daemon key it holds for this page's server.
  useEffect(() => {
    if (Platform.OS !== "web" || !active) return;
    let cancelled = false;
    void keyFor(active).then((k) => !cancelled && setKey(k ?? ""));
    return () => {
      cancelled = true;
    };
  }, [active, keyFor]);

  const saveKey = async () => {
    if (!active || key === null) return;
    const trimmed = key.trim();
    await save({
      url: active.url,
      ...(active.name ? { name: active.name } : {}),
      ...(trimmed ? { key: trimmed } : {}),
    });
    setPageKey(trimmed || undefined);
  };

  const exportAll = async () => {
    setBusy(true);
    try {
      const convs = await allConversations();
      const date = new Date().toISOString().slice(0, 10);
      await saveFile(`llmman-chats-${date}.json`, exportJson(convs), "application/json");
    } catch (e) {
      alert("Could not export", e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const deleteAll = () =>
    alert("Delete every conversation?", "All chats stored on this device, for every server, are deleted.", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Delete all",
        style: "destructive",
        onPress: async () => dropConversations(await deleteAllConversations()),
      },
    ]);

  return (
    <ScrollView
      key={sizeKey}
      style={{ backgroundColor: c.background }}
      contentContainerStyle={[styles.content, readable]}
      contentInsetAdjustmentBehavior="automatic"
      keyboardShouldPersistTaps="handled"
    >
      <Section title="Appearance">
        <Segments
          options={THEMES}
          value={prefs.theme}
          onChange={(theme: ThemePref) => void setPrefs({ theme })}
          label="Theme"
          inset={0}
        />
      </Section>

      <Section title="Your name">
        <Field
          label="Your name"
          labelHidden
          value={name}
          onChangeText={setName}
          onBlur={() => void setPrefs({ name: name.trim() })}
          onSubmitEditing={() => void setPrefs({ name: name.trim() })}
          placeholder="For the greeting"
          autoCorrect={false}
          hint="New chats greet you by it."
        />
      </Section>

      {Platform.OS === "web" ? (
        <Section title="Send a message with">
          <Segments
            options={SEND_WITH}
            value={prefs.sendWith}
            onChange={(sendWith: SendWith) => void setPrefs({ sendWith })}
            label="Send with"
            inset={0}
          />
          <Text style={[styles.hint, { color: c.textMuted }]}>
            {prefs.sendWith === "enter" ? "Shift+Enter starts a new line." : "Enter starts a new line."}
          </Text>
        </Section>
      ) : null}

      {Platform.OS === "web" && active && key !== null ? (
        <Section title="API key">
          <Field
            label="This server's API key"
            labelHidden
            value={key}
            onChangeText={setKey}
            onBlur={() => void saveKey()}
            onSubmitEditing={() => void saveKey()}
            secure
            autoCapitalize="none"
            autoCorrect={false}
            hint="The LLMMAN_API_KEYS key, if llmman serve requires one. Kept in this browser."
          />
        </Section>
      ) : null}

      <Section title="Conversations">
        <Text style={[styles.hint, { color: c.textMuted }]}>
          Stored on this device. The export is JSON, without generated media.
        </Text>
        <Button title="Export all chats" kind="secondary" onPress={() => void exportAll()} busy={busy} />
        <Button title="Delete all chats" kind="danger" onPress={deleteAll} />
      </Section>

      {active ? (
        <Section title="About">
          {/* As llmman's web UI words it: llmman serve <version> at <host> · <exe>. */}
          <Text selectable style={[styles.hint, { color: c.textMuted }]}>
            {daemon.version
              ? `llmman serve ${daemon.version.version} at ${hostLabel(active.url)}${
                  daemon.version.exe ? ` · ${daemon.version.exe}` : ""
                }`
              : daemon.reachable === false
                ? `llmman serve at ${hostLabel(active.url)} is unreachable`
                : `llmman serve at ${hostLabel(active.url)}`}
          </Text>
        </Section>
      ) : null}
    </ScrollView>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  const c = useTheme();
  return (
    <View style={styles.section}>
      <Text style={[styles.title, { color: c.textMuted }]} accessibilityRole="header">
        {title}
      </Text>
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  content: { padding: spacing.lg, gap: spacing.xl, paddingBottom: spacing.xl * 2 },
  section: { gap: spacing.sm },
  title: { fontSize: font.small + 1, fontWeight: "600" },
  hint: { fontSize: font.small, lineHeight: 18 },
});
