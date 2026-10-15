// A terminal on the web: xterm.js in the page, over the socket `connect`
// opens (the shell's; the key goes as its subprotocol). URLs in the output open on shift- or cmd/ctrl-click.
// Loaded on first use by Terminal.web.tsx.

import "@xterm/xterm/css/xterm.css";

import { FitAddon } from "@xterm/addon-fit";
import { Unicode11Addon } from "@xterm/addon-unicode11";
import { WebLinksAddon } from "@xterm/addon-web-links";
import { Terminal as XTerm } from "@xterm/xterm";
import { useEffect, useRef } from "react";
import { View } from "react-native";

import { isWebUrl, opensLink, type TerminalProps } from "./types";

export default function XTermView({
  connect,
  colors,
  onExit,
  onState,
  onSender,
  visible = true,
}: TerminalProps) {
  const host = useRef<View>(null);
  // The socket's handlers call the latest callbacks without reattaching.
  const latest = useRef({ connect, onExit, onState, onSender });
  useEffect(() => {
    latest.current = { connect, onExit, onState, onSender };
  });
  const refit = useRef<(() => void) | null>(null);
  const focus = useRef<(() => void) | null>(null);
  const termRef = useRef<XTerm | null>(null);
  const initialColors = useRef(colors);

  useEffect(() => {
    const el = host.current as unknown as HTMLElement;
    const term = new XTerm({
      fontSize: 13,
      // Box-drawing-complete fonts first, as llmman's web UI has them.
      fontFamily:
        'Menlo, "SF Mono", Consolas, "DejaVu Sans Mono", "Liberation Mono", ui-monospace, monospace',
      lineHeight: 1.0,
      cursorBlink: true,
      theme: initialColors.current,
      scrollback: 5000,
      allowProposedApi: true,
    });
    const fit = new FitAddon();
    term.loadAddon(fit);
    // Unicode 11 widths: emoji and other wide glyphs take two cells.
    term.loadAddon(new Unicode11Addon());
    term.unicode.activeVersion = "11";
    term.loadAddon(
      new WebLinksAddon((e, uri) => {
        if (opensLink(e) && isWebUrl(uri)) window.open(uri, "_blank", "noopener,noreferrer");
      }),
    );
    term.open(el);
    termRef.current = term;

    const ws = latest.current.connect();
    const encoder = new TextEncoder();
    const open = () => ws.readyState === WebSocket.OPEN;
    const send = (text: string) => open() && ws.send(encoder.encode(text));
    latest.current.onSender?.(send);
    const resize = () => open() && ws.send(JSON.stringify({ resize: { cols: term.cols, rows: term.rows } }));
    // Not while hidden behind another screen: a zero-sized host would
    // shrink the program's terminal to nothing.
    const fitNow = () => {
      if (!el.clientWidth || !el.clientHeight) return;
      try {
        fit.fit();
      } catch {
        // The host can be zero-sized for a frame while screens swap.
      }
    };
    refit.current = fitNow;
    focus.current = () => term.focus();
    fitNow();
    ws.onopen = () => {
      resize();
      latest.current.onState("attached");
      term.focus();
    };
    ws.onmessage = (e) => {
      if (typeof e.data === "string") {
        try {
          const m = JSON.parse(e.data) as { exit?: number };
          if (typeof m.exit === "number") latest.current.onExit(m.exit);
        } catch {
          // Not a control frame.
        }
      } else term.write(new Uint8Array(e.data as ArrayBuffer));
    };
    ws.onclose = (e) => latest.current.onState("closed", e.reason || undefined);
    const typing = term.onData(send);
    // Bytes xterm reports outside UTF-8 (some mouse modes), as raw bytes.
    const binary = term.onBinary((data) => {
      if (!open()) return;
      const bytes = new Uint8Array(data.length);
      for (let i = 0; i < data.length; i++) bytes[i] = data.charCodeAt(i) & 0xff;
      ws.send(bytes);
    });
    const resizing = term.onResize(resize);
    const observer = new ResizeObserver(fitNow);
    observer.observe(el);
    return () => {
      refit.current = null;
      focus.current = null;
      termRef.current = null;
      observer.disconnect();
      typing.dispose();
      binary.dispose();
      resizing.dispose();
      ws.onclose = null;
      ws.close();
      term.dispose();
    };
    // Once per mount: a new socket means a new shell.
  }, []);

  // A theme change recolors the terminal in place; the session goes on.
  useEffect(() => {
    if (termRef.current) termRef.current.options.theme = colors;
  }, [colors]);

  // Back on screen: fit to the (maybe changed) size and take the keyboard.
  useEffect(() => {
    if (!visible) return;
    refit.current?.();
    focus.current?.();
  }, [visible]);

  return <View ref={host} style={{ flex: 1, backgroundColor: colors.background, padding: 6 }} />;
}
