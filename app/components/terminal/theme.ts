// Terminal colors in the app's palette: its background and text, with ANSI
// colors that read on it. Dark is the shell palette of llmman's earlier
// web UI (the site's colors); light is the same hues, darkened.

import { dark as darkPalette, type Palette } from "@/lib/theme";

import type { TerminalColors } from "./types";

const DARK_ANSI = {
  black: "#151a25",
  red: "#f07178",
  green: "#6ee7c7",
  yellow: "#e5c07b",
  blue: "#7aa2f7",
  magenta: "#c3a6ff",
  cyan: "#7fdbff",
  white: "#d7dce6",
  brightBlack: "#8a94a8",
  brightRed: "#ff8b92",
  brightGreen: "#8ff0d6",
  brightYellow: "#f0d398",
  brightBlue: "#9bbcff",
  brightMagenta: "#d6c2ff",
  brightCyan: "#a6e8ff",
  brightWhite: "#f2f5fa",
};

const LIGHT_ANSI = {
  black: "#1f2430",
  red: "#c2313b",
  green: "#17795f",
  yellow: "#8a6100",
  blue: "#2f5fcf",
  magenta: "#7a46c9",
  cyan: "#0a7896",
  white: "#7d8494",
  brightBlack: "#5b6273",
  brightRed: "#d8434d",
  brightGreen: "#1e8f71",
  brightYellow: "#9c6f00",
  brightBlue: "#3f6fe0",
  brightMagenta: "#8b57da",
  brightCyan: "#108aa8",
  brightWhite: "#1f2430",
};

export function terminalColors(c: Palette): TerminalColors {
  const dark = c === darkPalette;
  return {
    background: c.codeBackground,
    foreground: c.text,
    cursor: c.accent,
    cursorAccent: c.codeBackground,
    selectionBackground: dark ? "rgba(127, 209, 191, 0.3)" : "rgba(31, 122, 114, 0.25)",
    ...(dark ? DARK_ANSI : LIGHT_ANSI),
  };
}
