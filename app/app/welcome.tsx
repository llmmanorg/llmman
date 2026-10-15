// First run: connect to the computer running llmman serve.

import { router } from "expo-router";
import { ScrollView, StyleSheet, Text } from "react-native";
import { KeyboardAvoidingView } from "react-native-keyboard-controller";
import { SafeAreaView } from "react-native-safe-area-context";

import { ConnectForm } from "@/components/ConnectForm";
import { font, readable, spacing, useTheme, useTextSizeKey } from "@/lib/theme";

export default function Welcome() {
  const c = useTheme();
  const sizeKey = useTextSizeKey();
  return (
    <SafeAreaView key={sizeKey} style={{ flex: 1, backgroundColor: c.background }} edges={["top", "bottom"]}>
      <KeyboardAvoidingView style={{ flex: 1 }} behavior="padding">
        <ScrollView contentContainerStyle={[styles.content, readable]} keyboardShouldPersistTaps="handled">
          <Text style={[styles.title, { color: c.text }]} accessibilityRole="header">
            Connect to llmman
          </Text>
          <Text style={[styles.lead, { color: c.textMuted }]}>
            This app is a remote for a computer running llmman serve. Enter the address it listens on and, if
            it has one, its API key (LLMMAN_API_KEYS).
          </Text>
          <ConnectForm onConnected={() => router.replace("/")} />
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  content: { padding: spacing.xl, gap: spacing.lg },
  title: { fontSize: 32, fontWeight: "800", marginTop: spacing.xl },
  lead: { fontSize: font.body, lineHeight: 23, marginBottom: spacing.sm },
});
