// A chat's own settings (the ⋯ button in a conversation).

import { router, Stack, useLocalSearchParams } from "expo-router";
import { ScrollView, Text } from "react-native";
import { KeyboardAvoidingView } from "react-native-keyboard-controller";

import { ChatSettingsForm } from "@/components/ChatSettingsForm";
import { setSettings, useConversation } from "@/lib/chatStore";
import { font, readable, spacing, useTheme, useTextSizeKey } from "@/lib/theme";

export default function ChatSettingsScreen() {
  const c = useTheme();
  const sizeKey = useTextSizeKey();
  const { conversation } = useLocalSearchParams<{ conversation: string }>();
  const { conv } = useConversation(conversation);
  if (!conv) return null;
  return (
    <KeyboardAvoidingView key={sizeKey} style={{ flex: 1, backgroundColor: c.background }} behavior="padding">
      <Stack.Screen options={{ title: "Chat settings" }} />
      <ScrollView
        contentContainerStyle={[{ padding: spacing.xl, gap: spacing.lg }, readable]}
        keyboardShouldPersistTaps="handled"
      >
        <Text style={{ color: c.textMuted, fontSize: font.small + 1, lineHeight: 20 }}>
          For this chat only; they apply from the next message. New chats start from the server&apos;s
          defaults (Servers → this server).
        </Text>
        <ChatSettingsForm
          initial={conv.settings}
          saveTitle="Save"
          onSave={(s) => {
            setSettings(conv.id, s);
            router.back();
          }}
        />
      </ScrollView>
    </KeyboardAvoidingView>
  );
}
