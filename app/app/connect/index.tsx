// Add a server (from the Servers tab).

import { router } from "expo-router";
import { ScrollView } from "react-native";
import { KeyboardAvoidingView } from "react-native-keyboard-controller";

import { ConnectForm } from "@/components/ConnectForm";
import { readable, spacing, useTheme, useTextSizeKey } from "@/lib/theme";

export default function Connect() {
  const c = useTheme();
  const sizeKey = useTextSizeKey();

  return (
    <KeyboardAvoidingView key={sizeKey} style={{ flex: 1, backgroundColor: c.background }} behavior="padding">
      <ScrollView
        contentContainerStyle={[{ padding: spacing.xl }, readable]}
        keyboardShouldPersistTaps="handled"
      >
        <ConnectForm
          onConnected={() => {
            router.dismissTo("/");
          }}
        />
      </ScrollView>
    </KeyboardAvoidingView>
  );
}
