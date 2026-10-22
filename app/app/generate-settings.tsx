// A chat's media generation settings, with a diffusion model (the sliders
// button in the composer, or the model picker's "Generation settings…").

import { router, Stack, useLocalSearchParams } from "expo-router";
import { ScrollView, Text, View } from "react-native";
import { KeyboardAvoidingView } from "react-native-keyboard-controller";

import { GenerateSettingsForm } from "@/components/GenerateSettingsForm";
import { setMediaOptions, useStoredConversation } from "@/lib/chatStore";
import { KIND_LABELS, optionsFor } from "@/lib/media";
import { useModels } from "@/lib/ModelsContext";
import { font, readable, spacing, useTheme, useTextSizeKey } from "@/lib/theme";

export default function GenerateSettingsScreen() {
  const c = useTheme();
  const sizeKey = useTextSizeKey();
  const models = useModels();
  const { conversation } = useLocalSearchParams<{ conversation: string }>();
  // Opened by its address (a reload), the chat is read from storage first.
  const { conv, missing } = useStoredConversation(conversation);
  if (missing) {
    return (
      <View style={{ flex: 1, backgroundColor: c.background, padding: spacing.xl }}>
        <Stack.Screen options={{ title: "Generation settings" }} />
        <Text style={{ color: c.textMuted }}>This conversation no longer exists.</Text>
      </View>
    );
  }
  if (!conv) return null;
  const options = optionsFor(conv.media, models.mediaKinds(conv.model));
  return (
    <KeyboardAvoidingView key={sizeKey} style={{ flex: 1, backgroundColor: c.background }} behavior="padding">
      <Stack.Screen options={{ title: `${KIND_LABELS[options.kind]} settings` }} />
      <ScrollView
        contentContainerStyle={[{ padding: spacing.xl, gap: spacing.lg }, readable]}
        keyboardShouldPersistTaps="handled"
      >
        <Text style={{ color: c.textMuted, fontSize: font.small + 1, lineHeight: 20 }}>
          For this chat only; they apply from the next prompt. Blank fields use the model&apos;s defaults.
        </Text>
        <GenerateSettingsForm
          initial={options}
          kind={options.kind}
          onSave={(value) => {
            setMediaOptions(conv.id, value);
            router.back();
          }}
        />
      </ScrollView>
    </KeyboardAvoidingView>
  );
}
