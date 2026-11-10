// Shell: a terminal on the computer running llmman serve, over
// `/llmman/shell`. The daemon decides
// whether one may open (only on loopback, with its key and the token of
// its current start; never under LLMMAN_SHELL=off); this screen only asks
// and explains. One session at a
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
import { onPageShellToken, pageShellToken, pageUrl } from "@/lib/page";
import { useServers } from "@/lib/ServersContext";
import { font, radius, spacing, useTheme } from "@/lib/theme";

const NO_TOKEN_HELP =
  Platform.OS === "web"
    ? "llmman serve makes a new shell token each time it starts. Open the link it prints in its terminal, " +
      "or add #shell-token= and the token from shell-<port>.token in llmman's data directory to this page's address."
    : "The shell needs the token llmman serve makes each time it starts, which only a browser on the same " +
      "computer can take from the link llmman serve prints.";

const STALE_TOKEN_HELP =
  "If llmman serve has restarted since its shell link was opened, open the new link: each start makes a new token.";

type Session =
  | { state: "connecting" }
  | { state: "attached" }
  | { state: "exited"; code: number }
  | { state: "closed"; reason?: string };

export default function Shell() {
  const c = useTheme();
  const focused = useIsFocused();
  const { active, keyFor } = useServers();
  const { features, reachable, checkReachable } = useDaemon();
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

  // Only the page's own daemon has a token: its shell link opened this page.
  const [pageToken, setPageToken] = useState(pageShellToken);
  useEffect(
    () =>
      onPageShellToken((token) => {
        setPageToken(token);
        setSessionOf({ url: "", session: { state: "connecting" } });
        setGeneration((g) => g + 1);
      }),
    [],
  );
  const token = url && url === pageUrl() ? pageToken : undefined;
  const connect = useCallback(() => openShellSocket(url, key.current ?? undefined, token), [url, token]);

  const restart = () => {
    // Another tab may have opened a newer shell link.
    setPageToken(pageShellToken());
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

  if (shell.tokenRequired && !token) {
    return (
      <View style={[styles.fill, { backgroundColor: colors.background }]}>
        <Overlay title="Open the shell link from llmman serve" text={NO_TOKEN_HELP} />
      </View>
    );
  }

  const overlay =
    session.state === "exited"
      ? { title: `The shell exited with status ${session.code}.` }
      : session.state === "closed"
        ? { title: "Disconnected from llmman serve.", text: session.reason }
        : null;
  const serverUnreachable = reachable === false;

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
            onState={(st, reason) => {
              // Otherwise the drawer waits for the next 15 s poll.
              if (st === "closed") checkReachable();
              setSession((s) => {
                if (st === "attached") return { state: "attached" };
                // The exit frame comes first; its close adds nothing.
                if (s.state === "exited") return s;
                // A browser is not told why an upgrade was refused; with a
                // shell link opened, a stale token is the likely reason.
                const why =
                  reason ?? (shell.tokenRequired && s.state === "connecting" ? STALE_TOKEN_HELP : undefined);
                return { state: "closed", ...(why ? { reason: why } : {}) };
              });
            }}
            onSender={(fn) => {
              send.current = fn;
            }}
          />
        ) : null}
        {overlay ? (
          <View style={[StyleSheet.absoluteFill, styles.scrim]}>
            <Overlay
              {...overlay}
              {...(serverUnreachable ? { text: "Start llmman serve, then open a new shell." } : {})}
              onRestart={restart}
              restartDisabled={serverUnreachable}
            />
          </View>
        ) : null}
      </View>
      {Platform.OS !== "web" && !overlay ? <KeysBar onKey={(seq) => send.current?.(seq)} /> : null}
    </View>
  );
}

function Overlay({
  title,
  text,
  onRestart,
  restartDisabled,
}: {
  title: string;
  text?: string;
  onRestart?: () => void;
  restartDisabled?: boolean;
}) {
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
        {onRestart ? (
          <Button
            title="Open a new shell"
            onPress={onRestart}
            disabled={restartDisabled}
            style={styles.button}
          />
        ) : null}
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
