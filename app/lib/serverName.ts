// How a server reads in lists: the name it was given, else its host
// (without the port; "This computer" for loopback), and llmman's version
// without the build it came from.

import { hostLabel } from "@/api/serverUrl";

/** "Studio", "192.168.1.20", or "This computer" for 127.0.0.1 / localhost / ::1. */
export function serverName(server: { name?: string; url: string }): string {
  if (server.name?.trim()) return server.name.trim();
  let host = "";
  try {
    host = new URL(server.url).hostname;
  } catch {
    host = hostLabel(server.url).replace(/:\d+$/, "");
  }
  host = host.replace(/^\[|\]$/g, "");
  if (/^(localhost|127(\.\d+){3}|::1)$/i.test(host)) return "This computer";
  return host || hostLabel(server.url);
}

/** `0.1.0` for `0.1.0 (5379f22-dirty)`: the build is for bug reports, not the list. */
export function shortVersion(version: string): string {
  return version.replace(/\s*\(.*\)\s*$/, "").trim();
}
