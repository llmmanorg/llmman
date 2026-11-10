/** @jest-environment node */

// The page's key on the web, when the browser's storage will not keep it.

import { adoptShellToken, onPageShellToken, pageKey, pageShellToken, setPageKey } from "../page.web";

test("a key storage will not take still holds for the page", () => {
  const stored = new Map<string, string>();
  let full = false;
  Object.assign(globalThis, {
    localStorage: {
      getItem: (k: string) => stored.get(k) ?? null,
      setItem: (k: string, v: string) => {
        if (full) throw new Error("QuotaExceededError");
        stored.set(k, v);
      },
      removeItem: (k: string) => void stored.delete(k),
    },
  });

  setPageKey("saved");
  expect(pageKey()).toBe("saved");

  full = true;
  setPageKey("unsaved");
  expect(pageKey()).toBe("unsaved");

  // Once storage takes a key again, it is read from there.
  full = false;
  setPageKey("again");
  expect(pageKey()).toBe("again");
  stored.set([...stored.keys()][0]!, "other tab");
  expect(pageKey()).toBe("other tab");
});

test("a shell link's token moves from the address into storage", () => {
  const stored = new Map<string, string>();
  const address = { pathname: "/shell", search: "?x=1", hash: "#shell-token=tok_1" };
  const replaced: string[] = [];
  Object.assign(globalThis, {
    localStorage: {
      getItem: (k: string) => stored.get(k) ?? null,
      setItem: (k: string, v: string) => void stored.set(k, v),
      removeItem: (k: string) => void stored.delete(k),
    },
    location: address,
    history: { state: null, replaceState: (_s: unknown, _t: string, u: string) => void replaced.push(u) },
  });
  const seen: string[] = [];
  const stop = onPageShellToken((t) => seen.push(t));

  adoptShellToken();
  expect(pageShellToken()).toBe("tok_1");
  expect(replaced).toEqual(["/shell?x=1"]);
  expect(seen).toEqual(["tok_1"]);

  // Any other fragment is the page's own: left alone.
  address.hash = "#section";
  adoptShellToken();
  expect(replaced).toHaveLength(1);
  expect(pageShellToken()).toBe("tok_1");
  stop();
});
