// The web's Terminal: xterm.js (XTermView.tsx), a third of the page's
// script, is fetched the first time a terminal opens rather than with the
// page.

import { lazy, Suspense } from "react";
import { View } from "react-native";

import type { TerminalProps } from "./types";

const XTermView = lazy(() => import("./XTermView"));

export function Terminal(props: TerminalProps) {
  return (
    <Suspense fallback={<View style={{ flex: 1, backgroundColor: props.colors.background, padding: 6 }} />}>
      <XTermView {...props} />
    </Suspense>
  );
}
