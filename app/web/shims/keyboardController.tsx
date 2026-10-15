// react-native-keyboard-controller on the web page (see metro.config.js).
// A browser reports no keyboard frames, so the library's components stay
// at rest there: the provider is a full-size View, a keyboard-avoiding
// view a View, and a keyboard-aware scroll view a ScrollView. These are
// those, without the Reanimated machinery behind them.

import type { ReactNode } from "react";
import { ScrollView, View, type ScrollViewProps, type ViewProps } from "react-native";

export function KeyboardProvider({ children }: { children?: ReactNode }) {
  return <View style={{ flex: 1 }}>{children}</View>;
}

export function KeyboardAvoidingView({
  behavior: _behavior,
  keyboardVerticalOffset: _offset,
  contentContainerStyle: _content,
  enabled: _enabled,
  ...props
}: ViewProps & {
  behavior?: string;
  keyboardVerticalOffset?: number;
  contentContainerStyle?: unknown;
  enabled?: boolean;
}) {
  return <View {...props} />;
}

export function KeyboardAwareScrollView({
  bottomOffset: _bottom,
  disableScrollOnKeyboardHide: _disable,
  enabled: _enabled,
  extraKeyboardSpace: _extra,
  ...props
}: ScrollViewProps & {
  bottomOffset?: number;
  disableScrollOnKeyboardHide?: boolean;
  enabled?: boolean;
  extraKeyboardSpace?: number;
}) {
  return <ScrollView {...props} />;
}
