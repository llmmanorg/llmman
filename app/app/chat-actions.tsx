// A chat's actions: rename it, or delete it (long press in the list, or
// its ⋯ on the web).

import { router, Stack, useLocalSearchParams } from "expo-router";
import { useState } from "react";
import { ScrollView } from "react-native";
import { KeyboardAvoidingView } from "react-native-keyboard-controller";

import { Button, Field } from "@/components/ui";
import { alert } from "@/lib/alert";
import { dropConversation, getConversationState, renameConversation } from "@/lib/chatStore";
import { deleteConversation } from "@/lib/conversations";
import { readable, spacing, useTheme, useTextSizeKey } from "@/lib/theme";

export default function ChatActions() {
  const c = useTheme();
  const sizeKey = useTextSizeKey();
  const { id = "", title = "" } = useLocalSearchParams<{ id: string; title: string }>();
  const [name, setName] = useState(title);

  const rename = async () => {
    await renameConversation(id, name);
    router.back();
  };

  const remove = () =>
    alert(`Delete "${title}"?`, "It is removed from this device.", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Delete",
        style: "destructive",
        onPress: async () => {
          const open = getConversationState(id).conv;
          dropConversation(id);
          await deleteConversation(id);
          router.back();
          // Leaving a chat that no longer exists for a new one.
          if (open) router.navigate("/");
        },
      },
    ]);

  return (
    <KeyboardAvoidingView key={sizeKey} style={{ flex: 1, backgroundColor: c.background }} behavior="padding">
      <Stack.Screen options={{ title: "Chat" }} />
      <ScrollView
        contentContainerStyle={[{ padding: spacing.xl, gap: spacing.lg }, readable]}
        keyboardShouldPersistTaps="handled"
      >
        <Field
          label="Title"
          value={name}
          onChangeText={setName}
          onSubmitEditing={() => void rename()}
          autoFocus
          selectTextOnFocus
          returnKeyType="done"
        />
        <Button title="Rename" onPress={() => void rename()} disabled={name.trim() === title.trim()} />
        <Button title="Delete chat" kind="danger" onPress={remove} />
      </ScrollView>
    </KeyboardAvoidingView>
  );
}
