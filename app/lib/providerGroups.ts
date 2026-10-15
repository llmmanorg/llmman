// Hosted providers as the Providers screen groups them.

import type { Provider } from "@/api/client";

// The ones most people look for, before the long tail in name order.
const WELL_KNOWN = ["anthropic", "openai", "google", "openrouter", "mistral", "deepseek", "xai", "groq"];
const rank = (p: Provider) => {
  const i = WELL_KNOWN.indexOf(p.id);
  return i < 0 ? WELL_KNOWN.length : i;
};

/** The providers in groups: connected, popular, then the rest; a query searches them all. */
export function providerSections(list: Provider[], query: string): { title: string; data: Provider[] }[] {
  const q = query.trim().toLowerCase();
  const shown = list
    .filter((p) => !p.key_optional || p.key_set)
    .filter((p) => !q || p.name.toLowerCase().includes(q) || p.id.includes(q));
  const connected = shown
    .filter((p) => p.key_set)
    .sort((a, b) => rank(a) - rank(b) || a.name.localeCompare(b.name));
  const rest = shown.filter((p) => !p.key_set);
  const popular = rest.filter((p) => rank(p) < WELL_KNOWN.length).sort((a, b) => rank(a) - rank(b));
  const others = rest
    .filter((p) => rank(p) >= WELL_KNOWN.length)
    .sort((a, b) => a.name.localeCompare(b.name));
  return [
    { title: "Connected", data: connected },
    { title: "Popular", data: popular },
    { title: connected.length || popular.length ? "All providers" : "Providers", data: others },
  ].filter((s) => s.data.length);
}
