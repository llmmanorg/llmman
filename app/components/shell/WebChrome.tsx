// The web page's chrome, as llmman's earlier web UI had it: its
// fragment links (`#/models`…), the page title per screen, the keys
// (Cmd/Ctrl+Shift+O new chat, Cmd/Ctrl+\ the sidebar, Escape closes a
// sheet), and a warning before leaving while a reply streams. Renders
// nothing; web only.

import { router, usePathname } from "expo-router";
import { useEffect, useRef } from "react";

import { requestDrawerToggle, useSidebar } from "@/components/shell/Header";
import { anyRunning } from "@/lib/chatStore";
import { currentDialog } from "@/lib/dialog";
import { useServers } from "@/lib/ServersContext";
import { useOpenChat } from "@/lib/useOpenChat";
import { documentTitle, hashRoute, isSheet } from "@/lib/webRoutes";

export function WebChrome() {
  const pathname = usePathname();
  const { ready, active } = useServers();
  const openChat = useOpenChat();
  const { wide, docked, setCollapsed } = useSidebar();
  const latest = useRef({ pathname, openChat, wide, docked, setCollapsed });
  useEffect(() => {
    latest.current = { pathname, openChat, wide, docked, setCollapsed };
  });

  // A fragment link from the web UI's days, once there is a server to show it.
  const routed = useRef(false);
  useEffect(() => {
    if (routed.current || !ready || !active) return;
    routed.current = true;
    const to = hashRoute(window.location.hash);
    if (!to) return;
    history.replaceState(history.state, "", window.location.pathname + window.location.search);
    router.replace(to as never);
  }, [ready, active]);

  useEffect(() => {
    document.title = documentTitle(pathname);
  }, [pathname]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // In the terminal, Ctrl combinations are the program's (Ctrl+\ is
      // SIGQUIT); Cmd ones stay the page's.
      const inTerminal = e.target instanceof Element && !!e.target.closest(".xterm");
      const mod = e.metaKey || (e.ctrlKey && !inTerminal);
      const {
        pathname: path,
        openChat: newChat,
        wide: isWide,
        docked: isDocked,
        setCollapsed: collapse,
      } = latest.current;
      if (mod && e.shiftKey && e.key.toLowerCase() === "o") {
        e.preventDefault();
        newChat();
      } else if (mod && e.key === "\\") {
        e.preventDefault();
        if (isWide) collapse(isDocked);
        else requestDrawerToggle();
      } else if (e.key === "Escape" && !currentDialog() && isSheet(path) && router.canGoBack()) {
        // An open dialog takes its own Escape (DialogHost).
        e.preventDefault();
        router.back();
      }
    };
    const onLeave = (e: BeforeUnloadEvent) => {
      if (anyRunning()) e.preventDefault();
    };
    // Capture: before xterm, which stops the keys it takes.
    document.addEventListener("keydown", onKey, true);
    window.addEventListener("beforeunload", onLeave);
    return () => {
      document.removeEventListener("keydown", onKey, true);
      window.removeEventListener("beforeunload", onLeave);
    };
  }, []);

  return null;
}
