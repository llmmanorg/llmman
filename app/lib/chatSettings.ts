// Per-chat settings, as the web UI's chat settings popover has them: a
// system prompt, temperature and max tokens. Unset means the model's own
// default. A server has defaults that new chats start from; a chat's own
// values then override them.

export interface ChatSettings {
  systemPrompt?: string;
  /** 0–2; unset: the model's default. */
  temperature?: number;
  /** A positive integer; unset: no limit. */
  maxTokens?: number;
}

export interface ChatSettingsInput {
  systemPrompt: string;
  temperature: string;
  maxTokens: string;
}

export type SettingsErrors = Partial<Record<"temperature" | "maxTokens", string>>;

export function toInput(s: ChatSettings | undefined): ChatSettingsInput {
  return {
    systemPrompt: s?.systemPrompt ?? "",
    temperature: s?.temperature !== undefined ? String(s.temperature) : "",
    maxTokens: s?.maxTokens !== undefined ? String(s.maxTokens) : "",
  };
}

/** Checks typed values (same limits as the web UI's inputs); blank fields are unset. */
export function parseSettings(
  input: ChatSettingsInput,
): { ok: true; value: ChatSettings } | { ok: false; errors: SettingsErrors } {
  const errors: SettingsErrors = {};
  const value: ChatSettings = {};
  if (input.systemPrompt.trim()) value.systemPrompt = input.systemPrompt;

  const t = input.temperature.trim().replace(",", ".");
  if (t) {
    const n = Number(t);
    if (!Number.isFinite(n) || n < 0 || n > 2) errors.temperature = "A number from 0 to 2.";
    else value.temperature = n;
  }
  const m = input.maxTokens.trim();
  if (m) {
    const n = Number(m);
    if (!/^\d+$/.test(m) || !Number.isSafeInteger(n) || n < 1)
      errors.maxTokens = "A whole number of 1 or more.";
    else value.maxTokens = n;
  }
  return Object.keys(errors).length ? { ok: false, errors } : { ok: true, value };
}

/** A one-line summary, e.g. for a settings row: "System prompt · 0.7 · 512 tokens". */
export function describeSettings(s: ChatSettings | undefined): string {
  const parts = [
    s?.systemPrompt?.trim() ? "System prompt" : "",
    s?.temperature !== undefined ? `Temperature ${s.temperature}` : "",
    s?.maxTokens !== undefined ? `${s.maxTokens} tokens max` : "",
  ].filter(Boolean);
  return parts.length ? parts.join(" · ") : "Model defaults";
}
