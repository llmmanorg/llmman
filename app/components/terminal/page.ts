// The terminal page the native app shows in a WebView: xterm.js, fitted to
// the view, talking to the app over postMessage (the app owns the socket, so
// the page needs no key and no origin).

import { FIT_JS, UNICODE11_JS, WEB_LINKS_JS, XTERM_CSS, XTERM_JS } from "./xtermBundle.generated";
import type { TerminalColors } from "./types";

export function terminalHtml(colors: TerminalColors): string {
  const theme = JSON.stringify(colors);
  return `<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<style>${XTERM_CSS}
html,body{margin:0;height:100%;background:${colors.background};overflow:hidden}
#t{position:absolute;top:4px;left:6px;right:2px;bottom:2px}</style></head>
<body><div id="t"></div>
<script>${XTERM_JS}</script><script>${FIT_JS}</script>
<script>${UNICODE11_JS}</script><script>${WEB_LINKS_JS}</script>
<script>
var term = new Terminal({ fontSize: 13, cursorBlink: true, theme: ${theme}, scrollback: 5000, lineHeight: 1.0, allowProposedApi: true });
var fit = new FitAddon.FitAddon();
term.loadAddon(fit);
term.loadAddon(new Unicode11Addon.Unicode11Addon());
term.unicode.activeVersion = "11";
function post(m) { window.ReactNativeWebView.postMessage(JSON.stringify(m)); }
// The app decides what a tap on a URL does (Terminal.tsx opens it).
term.loadAddon(new WebLinksAddon.WebLinksAddon(function (e, uri) {
  post({ t: "link", uri: uri, mod: !!(e.shiftKey || e.metaKey || e.ctrlKey) });
}));
term.open(document.getElementById("t"));
fit.fit();
term.onData(function (d) { post({ t: "in", d: d }); });
var last = "";
function resized() {
  fit.fit();
  var now = term.cols + "x" + term.rows;
  if (now !== last) { last = now; post({ t: "resize", cols: term.cols, rows: term.rows }); }
}
window.addEventListener("resize", resized);
window.__write = function (b64) {
  var s = atob(b64), u = new Uint8Array(s.length);
  for (var i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  term.write(u);
};
window.__theme = function (t) { term.options.theme = t; document.body.style.background = t.background; };
window.__focus = function () { term.focus(); };
last = term.cols + "x" + term.rows;
post({ t: "ready", cols: term.cols, rows: term.rows });
</script></body></html>`;
}

/** Bytes as base64, in chunks (String.fromCharCode cannot take a megabyte at once). */
export function toBase64(bytes: Uint8Array): string {
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    bin += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + 0x8000)));
  }
  return btoa(bin);
}
