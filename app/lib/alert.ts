// Alert.alert, which React Native Web leaves out: on the web, the app's own
// dialog (lib/dialog.ts), since window.alert/confirm are no-ops in some web
// views (llmman-desktop's). Native keeps the system Alert.

import { Alert, Platform, type AlertButton } from "react-native";

import { showDialog } from "@/lib/dialog";

export function alert(title: string, message?: string, buttons?: AlertButton[]): void {
  if (Platform.OS !== "web") {
    Alert.alert(title, message, buttons);
    return;
  }
  const list = buttons?.length ? buttons : [{ text: "OK" }];
  void showDialog({
    title,
    ...(message ? { message } : {}),
    buttons: list.map((b) => ({ text: b.text ?? "OK", ...(b.style ? { style: b.style } : {}) })),
  }).then(({ button }) => {
    const chosen = button >= 0 ? list[button] : list.find((b) => b.style === "cancel");
    chosen?.onPress?.();
  });
}
