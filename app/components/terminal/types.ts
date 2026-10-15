/** xterm's theme: the three the app always sets, and optionally the 16 ANSI colors. */
export type TerminalColors = { background: string; foreground: string; cursor: string } & Partial<
  Record<
    | "cursorAccent"
    | "selectionBackground"
    | "black"
    | "red"
    | "green"
    | "yellow"
    | "blue"
    | "magenta"
    | "cyan"
    | "white"
    | "brightBlack"
    | "brightRed"
    | "brightGreen"
    | "brightYellow"
    | "brightBlue"
    | "brightMagenta"
    | "brightCyan"
    | "brightWhite",
    string
  >
>;

export interface TerminalProps {
  /**
   * Opens the socket the terminal talks over: the shell's. Called once per mount; to reconnect, remount (a new `key`).
   * Binary terminal bytes both ways, `{"resize"}` out, `{"exit"}` in.
   */
  connect: () => WebSocket;
  colors: TerminalColors;
  /** The program exited (the daemon's `{"exit":N}` frame). */
  onExit: (code: number) => void;
  /** The socket is open, or closed (with the server's reason, if any). */
  onState: (state: "attached" | "closed", reason?: string) => void;
  /** Hands over a way to type into the terminal, for the extra keys bar. */
  onSender?: (send: (text: string) => void) => void;
  /** On screen (default true): a terminal behind another screen is not refitted, and is focused on return. */
  visible?: boolean;
}

/** Escape sequences for the keys a phone keyboard lacks. */
export const KEYS = [
  { label: "esc", seq: "\x1b" },
  { label: "tab", seq: "\t" },
  { label: "↑", seq: "\x1b[A" },
  { label: "↓", seq: "\x1b[B" },
  { label: "←", seq: "\x1b[D" },
  { label: "→", seq: "\x1b[C" },
  { label: "⏎", seq: "\r" },
  { label: "^C", seq: "\x03" },
] as const;

/**
 * Whether a click on a URL in the output opens it: shift- or cmd/ctrl-click,
 * as in alacritty, iTerm2 and llmman's web UI. A plain click stays a click
 * (focus, selection, or input to a program in mouse mode).
 */
export function opensLink(e: { shiftKey?: boolean; metaKey?: boolean; ctrlKey?: boolean }): boolean {
  return !!(e.shiftKey || e.metaKey || e.ctrlKey);
}

/** Only web addresses leave the terminal. */
export function isWebUrl(uri: string): boolean {
  return /^https?:\/\//i.test(uri);
}
