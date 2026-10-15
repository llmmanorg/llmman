// One server: rename it, replace its API key, test it, set what new chats
// start with, or remove it. The address is the server's identity; a new
// address is a new server.

import { router, Stack, useLocalSearchParams } from "expo-router";
import { useEffect, useState } from "react";
import { Platform, Text, View } from "react-native";
import { KeyboardAwareScrollView } from "react-native-keyboard-controller";

import { testConnection, type ConnectionResult } from "@/api/connection";
import { hostLabel } from "@/api/serverUrl";
import { ChatSettingsForm } from "@/components/ChatSettingsForm";
import { Button, Field, Notice } from "@/components/ui";
import { dropConversations } from "@/lib/chatStore";
import { deleteServerConversations } from "@/lib/conversations";
import { useServers } from "@/lib/ServersContext";
import { font, readable, spacing, useTheme, useTextSizeKey } from "@/lib/theme";
import { alert } from "@/lib/alert";

export default function ServerDetail() {
  const c = useTheme();
  const sizeKey = useTextSizeKey();
  const { id } = useLocalSearchParams<{ id: string }>();
  const { servers, save, remove, keyFor, setDefaults } = useServers();
  const server = servers.find((s) => s.id === id);

  const [name, setName] = useState(server?.name ?? "");
  const [key, setKey] = useState("");
  const [hadKey, setHadKey] = useState<boolean | null>(null);
  const [testing, setTesting] = useState(false);
  const [result, setResult] = useState<ConnectionResult | null>(null);
  const [defaultsSaved, setDefaultsSaved] = useState(false);

  useEffect(() => {
    if (!server) return;
    let cancelled = false;
    keyFor(server).then((k) => {
      if (cancelled) return;
      setHadKey(Boolean(k));
      setKey(k ?? "");
    });
    return () => {
      cancelled = true;
    };
  }, [server, keyFor]);

  if (!server) {
    return (
      <View style={{ flex: 1, backgroundColor: c.background, padding: spacing.xl }}>
        <Stack.Screen options={{ title: "Server" }} />
        <Text style={{ color: c.textMuted }}>This server was removed.</Text>
      </View>
    );
  }

  const test = async () => {
    setTesting(true);
    setResult(await testConnection(server.url, key.trim() || undefined));
    setTesting(false);
  };

  const saveChanges = async () => {
    await save(
      { url: server.url, name: name.trim() || undefined, key: key.trim() || undefined },
      { activate: false },
    );
    router.back();
  };

  const here = Platform.OS === "web" ? "browser" : "device";
  const confirmRemove = () => {
    const cancel = { text: "Cancel", style: "cancel" as const };
    const keepChats = {
      text: "Remove server",
      isPreferred: true,
      onPress: async () => {
        await remove(server.url);
        router.back();
      },
    };
    const withChats = {
      text: "Remove server and its chats",
      style: "destructive" as const,
      onPress: async () => {
        dropConversations(await deleteServerConversations(server.url));
        await remove(server.url);
        router.back();
      },
    };
    alert(
      `Remove ${server.name ?? hostLabel(server.url)}?`,
      `Its API key is deleted from this ${here}. Its chats can stay here, and come back if you add the server again.`,
      // Android has no destructive style and makes the last button the
      // primary one, so there the safe action goes last and the destructive one first.
      Platform.OS === "android" ? [withChats, cancel, keepChats] : [cancel, keepChats, withChats],
    );
  };

  return (
    <KeyboardAwareScrollView
      key={sizeKey}
      style={{ backgroundColor: c.background }}
      contentContainerStyle={[{ padding: spacing.xl, gap: spacing.lg }, readable]}
      keyboardShouldPersistTaps="handled"
    >
      <Stack.Screen options={{ title: server.name ?? hostLabel(server.url) }} />
      <View style={{ gap: spacing.xs }}>
        <Text style={{ color: c.textMuted, fontSize: font.small, fontWeight: "600" }}>ADDRESS</Text>
        <Text style={{ color: c.text, fontSize: font.body }} selectable>
          {server.url}
        </Text>
      </View>
      <Field label="Name" value={name} onChangeText={setName} placeholder={hostLabel(server.url)} />
      <Field
        label="API key"
        value={key}
        onChangeText={setKey}
        secure
        placeholder={hadKey === false ? "None" : ""}
        hint={Platform.OS === "web" ? "Kept in this browser" : "Stored in the phone's secure storage"}
      />
      {result ? (
        result.ok ? (
          <Notice tone="info" title="Connected">
            llmman {result.version.version}
          </Notice>
        ) : (
          <Notice tone="error" title="Couldn't connect">
            {result.message}
          </Notice>
        )
      ) : null}
      <Button title="Test connection" kind="secondary" busy={testing} onPress={test} />
      {/* Saving before the stored key has loaded would erase it. */}
      <Button title="Save" onPress={saveChanges} disabled={hadKey === null} />

      <View style={{ gap: spacing.xs, marginTop: spacing.lg }}>
        <Text style={{ color: c.text, fontSize: font.title, fontWeight: "700" }} accessibilityRole="header">
          New chats
        </Text>
        <Text style={{ color: c.textMuted, fontSize: font.small + 1, lineHeight: 20 }}>
          What new chats on this server start with. Each chat can change its own in its ⋯ menu.
        </Text>
      </View>
      <ChatSettingsForm
        initial={server.chatDefaults}
        saveTitle="Save chat defaults"
        onSave={async (d) => {
          await setDefaults(server.url, d);
          setDefaultsSaved(true);
        }}
      />
      {defaultsSaved ? <Notice tone="info">Saved. New chats start with these.</Notice> : null}

      <Button title="Remove server" kind="danger" onPress={confirmRemove} style={{ marginTop: spacing.xl }} />
    </KeyboardAwareScrollView>
  );
}
