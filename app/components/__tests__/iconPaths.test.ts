import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { ICON_PATHS } from "@/components/iconPaths.generated";

const root = join(__dirname, "../..");

function sources(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const path = join(dir, e.name);
    if (e.isDirectory()) return e.name === "__tests__" ? [] : sources(path);
    return /\.tsx?$/.test(e.name) && !e.name.endsWith(".generated.ts") ? [path] : [];
  });
}

// The same scan as scripts/make-web-icons.mjs.
const used = new Set(
  ["app", "components", "lib"]
    .flatMap((d) => sources(join(root, d)))
    .flatMap((f) =>
      [...readFileSync(f, "utf8").matchAll(/\b(?:android|web):\s*"([a-z0-9_]+)"/g)].map((m) => m[1]!),
    ),
);

test("the web has a path for every icon the app uses (npm run make-web-icons)", () => {
  expect([...used].filter((name) => !ICON_PATHS[name])).toEqual([]);
});

test("the web keeps no icon the app no longer uses", () => {
  expect(Object.keys(ICON_PATHS).filter((name) => !used.has(name))).toEqual([]);
});
