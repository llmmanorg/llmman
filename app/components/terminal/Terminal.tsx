// A terminal on a phone: xterm.js in a WebView, with the socket `connect`
// opens (the shell's; the key goes as a header) here and the bytes passed
// over postMessage.

import { useEffect, useRef, useState } from "react";
import { Linking } from "react-native";
import { WebView, type WebViewMessageEvent } from "react-native-webview";

import { terminalHtml, toBase64 } from "./page";
import { isWebUrl, type TerminalProps } from "./types";

export function Terminal({ connect, colors, onExit, onState, onSender }: TerminalProps) {
  const web = useRef<WebView>(null);
  const socket = useRef<WebSocket | null>(null);
  // Built once: a new page would mean a new socket, and a new shell.
  const [html] = useState(() => terminalHtml(colors));
  const latest = useRef({ connect, onExit, onState });
  useEffect(() => {
    latest.current = { connect, onExit, onState };
  });

  const sendText = (text: string) => {
    const ws = socket.current;
    if (ws?.readyState === WebSocket.OPEN) ws.send(new TextEncoder().encode(text));
  };

  useEffect(() => {
    onSender?.(sendText);
    return () => {
      const ws = socket.current;
      socket.current = null;
      ws?.close();
    };
    // Once: the sender reads the current socket.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // A theme change recolors the page in place.
  useEffect(() => {
    web.current?.injectJavaScript(`window.__theme && window.__theme(${JSON.stringify(colors)});true;`);
  }, [colors]);

  const attach = (cols: number, rows: number) => {
    // Once per mount; the page reports ready again only if the WebView reloads.
    if (socket.current) return;
    const ws = latest.current.connect();
    socket.current = ws;
    ws.onopen = () => {
      ws.send(JSON.stringify({ resize: { cols, rows } }));
      latest.current.onState("attached");
    };
    ws.onmessage = (e) => {
      if (typeof e.data === "string") {
        try {
          const m = JSON.parse(e.data) as { exit?: number };
          if (typeof m.exit === "number") latest.current.onExit(m.exit);
        } catch {
          // Not a control frame.
        }
        return;
      }
      const b64 = toBase64(new Uint8Array(e.data as ArrayBuffer));
      web.current?.injectJavaScript(`window.__write(${JSON.stringify(b64)});true;`);
    };
    ws.onclose = (e) => {
      if (socket.current === ws) latest.current.onState("closed", e.reason || undefined);
    };
  };

  const onMessage = (e: WebViewMessageEvent) => {
    let m: { t: string; d?: string; cols?: number; rows?: number; uri?: string; mod?: boolean };
    try {
      m = JSON.parse(e.nativeEvent.data);
    } catch {
      return;
    }
    if (m.t === "ready") attach(m.cols ?? 80, m.rows ?? 24);
    else if (m.t === "in" && m.d) sendText(m.d);
    // A tap on a link opens it: a phone has no Shift or Cmd to hold (the web keeps them, Terminal.web.tsx).
    else if (m.t === "link" && m.uri && isWebUrl(m.uri)) void Linking.openURL(m.uri);
    else if (m.t === "resize" && socket.current?.readyState === WebSocket.OPEN) {
      socket.current.send(JSON.stringify({ resize: { cols: m.cols, rows: m.rows } }));
    }
  };

  return (
    <WebView
      ref={web}
      source={{ html }}
      originWhitelist={["*"]}
      onMessage={onMessage}
      style={{ flex: 1, backgroundColor: colors.background }}
      keyboardDisplayRequiresUserAction={false}
      hideKeyboardAccessoryView
      bounces={false}
      overScrollMode="never"
      // Text in the terminal is the program's; nothing here opens links by itself.
      onShouldStartLoadWithRequest={(r) => r.url === "about:blank" || r.url.startsWith("data:")}
    />
  );
}
