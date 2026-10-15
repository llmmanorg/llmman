// One prompt for the daemon's key, shared by every request that got
// llmman's 401 at the same time: concurrent 401s
// wait on the same question, and each retries once with the answer.

export interface KeyPrompt {
  /** Asks once; resolves true when a key was entered (and stored). */
  ask(): Promise<boolean>;
  /** Forgets a declined prompt, e.g. when the key is changed in Settings. */
  reset(): void;
}

/**
 * `ask` shows the question and resolves with the text, or null when
 * cancelled; `store` keeps an entered key. A cancelled prompt is not shown
 * again until `reset`, so a page that keeps polling does not nag; after an
 * entered key, a later refusal asks again.
 */
export function createKeyPrompt(ask: () => Promise<string | null>, store: (key: string) => void): KeyPrompt {
  let pending: Promise<boolean> | null = null;
  return {
    ask() {
      pending ??= ask()
        .catch(() => null)
        .then((text) => {
          const key = text?.trim() ?? "";
          if (!key) return false;
          store(key);
          pending = null;
          return true;
        });
      return pending;
    },
    reset() {
      pending = null;
    },
  };
}
