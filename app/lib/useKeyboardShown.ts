// Whether the on-screen keyboard is up, for layouts that make room for it.

import { useEffect, useState } from "react";
import { Keyboard, Platform } from "react-native";

export function useKeyboardShown(): boolean {
  const [shown, setShown] = useState(false);
  useEffect(() => {
    // iOS announces the keyboard before it moves; Android only after.
    const show = Platform.OS === "ios" ? "keyboardWillShow" : "keyboardDidShow";
    const hide = Platform.OS === "ios" ? "keyboardWillHide" : "keyboardDidHide";
    const subs = [
      Keyboard.addListener(show, () => setShown(true)),
      Keyboard.addListener(hide, () => setShown(false)),
    ];
    return () => subs.forEach((s) => s.remove());
  }, []);
  return shown;
}
