import type { Provider } from "@/api/client";

import { providerSections } from "../providerGroups";

const p = (id: string, name: string, extra: Partial<Provider> = {}): Provider => ({
  id,
  name,
  key_usable: false,
  ...extra,
});

test("connected, then popular, then the rest by name", () => {
  const list = [
    p("zeta", "Zeta"),
    p("openai", "OpenAI"),
    p("anthropic", "Anthropic"),
    p("groq", "Groq", { key_set: true, key_usable: true }),
    p("abacus", "Abacus"),
    p("local", "Local", { key_optional: true }),
  ];
  expect(providerSections(list, "").map((s) => [s.title, s.data.map((x) => x.id)])).toEqual([
    ["Connected", ["groq"]],
    ["Popular", ["anthropic", "openai"]],
    ["All providers", ["abacus", "zeta"]],
  ]);
  expect(providerSections(list, "ab").map((s) => s.title)).toEqual(["Providers"]);
});
