/** @jest-environment node */

// The page's key on the web, when the browser's storage will not keep it.

import { pageKey, setPageKey } from "../page.web";

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
