import { describeSettings, parseSettings, toInput } from "../chatSettings";

const input = (o: Partial<{ systemPrompt: string; temperature: string; maxTokens: string }>) => ({
  systemPrompt: "",
  temperature: "",
  maxTokens: "",
  ...o,
});

it("leaves blank fields unset", () => {
  expect(parseSettings(input({ systemPrompt: "  " }))).toEqual({ ok: true, value: {} });
});

it("reads valid values, a decimal comma included", () => {
  expect(parseSettings(input({ systemPrompt: "Be brief.", temperature: "0,7", maxTokens: "512" }))).toEqual({
    ok: true,
    value: { systemPrompt: "Be brief.", temperature: 0.7, maxTokens: 512 },
  });
  expect(parseSettings(input({ temperature: "0" }))).toEqual({ ok: true, value: { temperature: 0 } });
  expect(parseSettings(input({ temperature: "2" }))).toEqual({ ok: true, value: { temperature: 2 } });
});

it.each([
  [{ temperature: "2.1" }, "temperature"],
  [{ temperature: "-0.1" }, "temperature"],
  [{ temperature: "warm" }, "temperature"],
  [{ maxTokens: "0" }, "maxTokens"],
  [{ maxTokens: "1.5" }, "maxTokens"],
  [{ maxTokens: "-3" }, "maxTokens"],
  [{ maxTokens: "1e3" }, "maxTokens"],
])("rejects %j", (o, field) => {
  const r = parseSettings(input(o));
  expect(r.ok).toBe(false);
  if (!r.ok) expect(Object.keys(r.errors)).toEqual([field]);
});

it("round-trips through the form and describes itself", () => {
  const s = { systemPrompt: "x", temperature: 0.3, maxTokens: 64 };
  expect(parseSettings(toInput(s))).toEqual({ ok: true, value: s });
  expect(describeSettings(s)).toBe("System prompt · Temperature 0.3 · 64 tokens max");
  expect(describeSettings(undefined)).toBe("Model defaults");
});
