// Colors and spacing, light and dark from Settings (the system's by default).

import { Platform, useColorScheme, useWindowDimensions } from "react-native";

import { usePrefs } from "@/lib/prefs";

export interface Palette {
  background: string;
  surface: string;
  surfaceAlt: string;
  border: string;
  text: string;
  textMuted: string;
  accent: string;
  onAccent: string;
  danger: string;
  warning: string;
  success: string;
  userBubble: string;
  onUserBubble: string;
  codeBackground: string;
  /** The drawer and the wide-screen sidebar. */
  sidebar: string;
}

// Neutral greys with one accent, llmman's teal from the manatee mark:
// its outline in light mode, its body in dark mode.
export const light: Palette = {
  background: "#FFFFFF",
  surface: "#F4F4F4",
  surfaceAlt: "#EAEAEA",
  border: "#E3E3E3",
  text: "#0D0D0D",
  textMuted: "#6B6B6B",
  accent: "#1F7A72",
  onAccent: "#FFFFFF",
  danger: "#C62828",
  warning: "#B26A00",
  success: "#2E7D32",
  userBubble: "#F2F2F2",
  onUserBubble: "#0D0D0D",
  codeBackground: "#F6F6F6",
  sidebar: "#F9F9F9",
};

export const dark: Palette = {
  background: "#171717",
  surface: "#242424",
  surfaceAlt: "#2F2F2F",
  border: "#303030",
  text: "#ECECEC",
  textMuted: "#9A9A9A",
  accent: "#7FD1BF",
  onAccent: "#0A2420",
  danger: "#FF8A80",
  warning: "#FFB74D",
  success: "#81C784",
  userBubble: "#2C2C2C",
  onUserBubble: "#ECECEC",
  codeBackground: "#1F1F1F",
  sidebar: "#0F0F0F",
};

export const spacing = { xs: 4, sm: 8, md: 12, lg: 16, xl: 24 } as const;
export const radius = { sm: 6, md: 10, lg: 16, pill: 999 } as const;
export const font = {
  small: 13,
  // 17 pt is the iOS body size; 16 px on the web.
  body: Platform.OS === "web" ? 16 : 17,
  title: 20,
  mono: Platform.select({ ios: "Menlo", default: "monospace" }),
} as const;

/**
 * Keeps text and forms at a readable width on an iPad or a phone in
 * landscape, centred; on a landscape phone the margins also clear the notch.
 */
export const readable = { width: "100%", maxWidth: 720, alignSelf: "center" } as const;

/** From this window width the conversations are a sidebar, not a drawer. */
export const SIDEBAR_MIN_WIDTH = 900;

/** The palette for the theme chosen in Settings (the system's by default). */
export function useTheme(): Palette {
  const system = useColorScheme();
  const { theme } = usePrefs();
  return (theme === "system" ? system : theme) === "dark" ? dark : light;
}

/**
 * A key that changes with the text size. React Native keeps text measured at
 * the old size when it changes while the app runs (Dynamic Type, Android
 * font size), clipping it, so a screen's content is remounted under a new key.
 */
export function useTextSizeKey(): string {
  return String(useWindowDimensions().fontScale);
}
