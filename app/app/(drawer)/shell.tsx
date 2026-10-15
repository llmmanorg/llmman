// Shell: a terminal on the computer running llmman serve, over
// `/llmman/shell`. The daemon decides
// whether one may open (only on loopback, with its key; never under
// LLMMAN_SHELL=off); this screen only asks and explains. One session at a
// time: it lives as long as this screen, which the drawer keeps while you
// go elsewhere, so coming back finds the same shell.

import { useIsFocused } from "expo-router";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Platform, StyleSheet, Text, View } from "react-native";

import { openShellSocket } from "@/api/shell";
import { Icon } from "@/components/Icon";
import { KeysBar } from "@/components/terminal/KeysBar";
import { Terminal } from "@/components/terminal/Terminal";
import { terminalColors } from "@/components/terminal/theme";
import { Button } from "@/components/ui";
import { useDaemon } from "@/lib/DaemonContext";
import { useServers } from "@/lib/ServersContext";
import { font, radius, spacing, useTheme } from "@/lib/theme";

type Session =
  | { state: "connecting" }
  | { state: "attached" }
  | { state: "exited"; code: number }
  | { state: "closed"; reason?: string };

export default function Shell() {
  const c = useTheme();
  const focused = useIsFocused();
  const { active, keyFor } = useServers();
  const { features } = useDaemon();
  const shell = features?.shell ?? null;
  const colors = useMemo(() => terminalColors(c), [c]);

  // The key as of each connect: a key changed in Settings applies to the next shell.
  const key = useRef<string | undefined | null>(null);
  // Which server the key was read for: no shell opens with another's.
  const [keyFrom, setKeyFrom] = useState<string | null>(null);
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    void keyFor(active).then((k) => {
      if (cancelled) return;
      key.current = k;
      setKeyFrom(active.url);
    });
    return () => {
      cancelled = true;
    };
  }, [active, keyFor]);

  const [generation, setGeneration] = useState(0);
  // Per server: another server's shell starts from a clean slate.
  const [sessionOf, setSessionOf] = useState<{ url: string; session: Session }>({
    url: "",
    session: { state: "connecting" },
  });
  const send = useRef<((text: string) => void) | null>(null);
  const url = active?.url ?? "";
  const session: Session = sessionOf.url === url ? sessionOf.session : { state: "connecting" };
  const setSession = (next: (s: Session) => Session) =>
    setSessionOf((o) => ({ url, session: next(o.url === url ? o.session : { state: "connecting" }) }));
  const connect = useCallback(() => openShellSocket(url, key.current ?? undefined), [url]);

  const restart = () => {
    setSession(() => ({ state: "connecting" }));
    setGeneration((g) => g + 1);
  };

  if (!shell) return <View style={{ flex: 1, backgroundColor: colors.background }} />;
  if (!shell.enabled) {
    return (
      <View style={[styles.fill, { backgroundColor: colors.background }]}>
        <Overlay
          title="The shell is not available"
          text={shell.reason || "llmman serve has the shell turned off."}
        />
      </View>
    );
  }

  const overlay =
    session.state === "exited"
      ? { title: `The shell exited with status ${session.code}.` }
      : session.state === "closed"
        ? { title: "Disconnected from llmman serve.", text: session.reason }
        : null;

  return (
    <View style={[styles.fill, { backgroundColor: colors.background }]}>
      <View style={styles.fill}>
        {active && keyFrom === url ? (
          <Terminal
            key={`${url}|${generation}`}
            connect={connect}
            colors={colors}
            visible={focused}
            onExit={(code) => setSession(() => ({ state: "exited", code }))}
            onState={(st, reason) =>
              setSession((s) =>
                st === "attached"
                  ? { state: "attached" }
                  : // The exit frame comes first; its close adds nothing.
                    s.state === "exited"
                    ? s
                    : { state: "closed", ...(reason ? { reason } : {}) },
              )
            }
            onSender={(fn) => {
              send.current = fn;
            }}
          />
        ) : null}
        {overlay ? (
          <View style={[StyleSheet.absoluteFill, styles.scrim]}>
            <Overlay {...overlay} onRestart={restart} />
          </View>
        ) : null}
      </View>
      {Platform.OS !== "web" && !overlay ? <KeysBar onKey={(seq) => send.current?.(seq)} /> : null}
    </View>
  );
}

function Overlay({ title, text, onRestart }: { title: string; text?: string; onRestart?: () => void }) {
  const c = useTheme();
  return (
    <View style={styles.center}>
      <View
        style={[styles.card, { backgroundColor: c.background, borderColor: c.border }]}
        accessibilityRole="alert"
      >
        <Icon name={{ ios: "terminal", android: "terminal" }} color={c.textMuted} size={28} />
        <Text style={[styles.title, { color: c.text }]}>{title}</Text>
        {text ? <Text style={[styles.text, { color: c.textMuted }]}>{text}</Text> : null}
        {onRestart ? <Button title="Open a new shell" onPress={onRestart} style={styles.button} /> : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
  scrim: { backgroundColor: "rgba(0,0,0,0.25)" },
  center: { flex: 1, alignItems: "center", justifyContent: "center", padding: spacing.lg },
  card: {
    width: "100%",
    maxWidth: 420,
    alignItems: "center",
    gap: spacing.sm,
    padding: spacing.xl,
    borderRadius: radius.lg,
    borderWidth: StyleSheet.hairlineWidth,
  },
  title: { fontSize: font.body, fontWeight: "600", textAlign: "center" },
  text: { fontSize: font.small, lineHeight: 19, textAlign: "center" },
  button: { alignSelf: "stretch", marginTop: spacing.sm },
});
