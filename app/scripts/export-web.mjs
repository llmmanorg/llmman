#!/usr/bin/env node
// Builds the web app for llmman serve: `npm run export-web [-- <out dir>]` (default dist/).
// The output replaces llmman's webui/, which build.rs gzips into the binary.
//
// llmman serve answers `/` and the app's routes (`/c/<id>`, `/models`) with
// index.html, and `/ui/*` with the files next to it. A gateway may put all
// of it under a prefix (`/team/llmman/`), so nothing in the export names an
// absolute path, as in llmman's own web UI:
//
// - index.html refers to `ui/…`, resolved against a <base> its first
//   script sets from the page's address: the directory the page is in,
//   without one of the app's route directories (`/team/llmman/c/x` →
//   `/team/llmman/`).
// - Scripts load chunks and images from `ui/` relative to their own URL
//   (`globalThis.__llmmanUi`), and the router takes its base path from the
//   same place. Expo bakes one base path into the build
//   (`experiments.baseUrl`), so the export uses a placeholder and swaps it
//   for those runtime values here.
//
// The export holds no binary files: the manatee mark is llmman's own
// `ui/manatee.png` (build.rs fetches it by SHA-256), and the few small
// images the router's own screens carry are inlined as data: URIs. Files
// the daemon never serves (metadata.json, the favicon.ico Expo makes) are
// left out.

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import sharp from "sharp";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const out = resolve(process.argv[2] ?? "dist");

/** What Expo is told the base path is; replaced in every file below. */
const PLACEHOLDER = "/__llmman_base__";
/** The mark, byte for byte the one llmman's build.rs fetches as `manatee.png` (FETCHED_ASSETS). */
const MARK = {
  sha256: "c6036711a349dc8f8b2b4405be90b342b477dbcf4a7d95eb7d05b42702a241d5",
  served: "manatee.png",
};
/** Images up to this size are inlined; a larger one fails the export. */
const MAX_INLINE = 16 * 1024;
/** The JavaScript `ui/` URL: the export's root, from the running script's own URL (`ui/_expo/static/js/web/`). */
const UI = "globalThis.__llmmanUi";
const UI_PRELUDE = `${UI}||(${UI}=new URL("../../../../",document.currentScript.src).href);`;

execFileSync("npx", ["expo", "export", "-p", "web", "--clear", "--output-dir", out], {
  stdio: "inherit",
  cwd: root,
  env: { ...process.env, LLMMAN_EXPORT_BASE: PLACEHOLDER },
});

async function files(dir) {
  const all = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) all.push(...(await files(path)));
    else all.push(path);
  }
  return all.sort();
}

/** Route directories under app/ (groups like `(drawer)` looked into, not counted). */
async function routeDirs(dir = join(root, "app")) {
  const dirs = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    if (/^\(.*\)$/.test(entry.name)) dirs.push(...(await routeDirs(join(dir, entry.name))));
    else if (!/^[[+_]/.test(entry.name)) dirs.push(entry.name);
  }
  return [...new Set(dirs)].sort();
}

// What each exported image becomes: llmman's mark, a data: URI, or a 404 for
// an unused one (Expo writes every scale of an image; the web asks for 1x).
const images = new Map();
for (const path of await files(join(out, "assets"))) {
  const rel = relative(out, path);
  const bytes = await readFile(path);
  const url = `${PLACEHOLDER}/${rel}`;
  if (createHash("sha256").update(bytes).digest("hex") === MARK.sha256) {
    images.set(url, `${UI}+${JSON.stringify(MARK.served)}`);
  } else if (/\.png$/.test(rel) && bytes.length <= MAX_INLINE) {
    // Re-encoded when that is smaller: the same pixels, without the metadata
    // some carry (expo-router's 48 px arrow is 9 KB of it).
    const packed = await sharp(bytes).png({ compressionLevel: 9, effort: 10 }).toBuffer();
    const png = packed.length < bytes.length ? packed : bytes;
    images.set(url, JSON.stringify(`data:image/png;base64,${png.toString("base64")}`));
  } else {
    images.set(url, null);
  }
}
await rm(join(out, "assets"), { recursive: true, force: true });
await rm(join(out, "metadata.json"), { force: true });
await rm(join(out, "favicon.ico"), { force: true });

const escape = (s) => s.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&");
const quoted = new RegExp(`"${escape(PLACEHOLDER)}(/[^"]*)?"`, "g");

for (const path of await files(out)) {
  const rel = relative(out, path);
  if (rel.endsWith(".js")) {
    let text = await readFile(path, "utf8");
    // The app config Expo embeds, as JSON in a string: no base path there.
    text = text.replaceAll(`\\"baseUrl\\":\\"${PLACEHOLDER}\\"`, `\\"baseUrl\\":\\"\\"`);
    text = text.replace(quoted, (whole, rest) => {
      // The router's base path: the directory `ui/` is in, "" at the root.
      if (!rest) return `new URL("..",${UI}).pathname.replace(/\\/$/,"")`;
      if (rest.startsWith("/assets/")) {
        const image = images.get(whole.slice(1, -1));
        if (!image) throw new Error(`${rel} refers to ${rest}, which the web cannot do without`);
        return image;
      }
      if (rest.startsWith("/_expo/")) return `${UI}+${JSON.stringify(rest.slice(1))}`;
      throw new Error(`${rel}: unexpected ${whole}`);
    });
    if (text.includes(PLACEHOLDER)) throw new Error(`${rel} still names ${PLACEHOLDER}`);
    await writeFile(path, UI_PRELUDE + text);
  } else if (rel.endsWith(".css")) {
    const text = await readFile(path, "utf8");
    if (text.includes(PLACEHOLDER)) throw new Error(`${rel} names ${PLACEHOLDER}`);
  } else if (rel !== "index.html") {
    throw new Error(`unexpected file in the export: ${rel}`);
  }
}

const ROUTE_DIRS = await routeDirs();
let html = await readFile(join(out, "index.html"), "utf8");
// The page's stylesheets and scripts, taken out of the markup: a browser's
// preload scanner fetches tags it sees before any script has set the base,
// so at `/c/<id>` it would ask for `/c/ui/…` first. They are added by script
// once the base is in place instead.
const styles = [];
const scripts = [];
html = html
  .replace(/<link rel="preload" href="[^"]*" as="style">\s*/g, "")
  .replace(/<link rel="stylesheet" href="([^"]*)">\s*/g, (_, href) => (styles.push(href), ""))
  .replace(/<script src="([^"]*)" defer><\/script>\s*/g, (_, src) => (scripts.push(src), ""));
const ui = (url) => {
  if (!url.startsWith(`${PLACEHOLDER}/`)) throw new Error(`index.html: unexpected URL ${url}`);
  return `ui/${url.slice(PLACEHOLDER.length + 1)}`;
};
if (!scripts.length) throw new Error("index.html: no scripts found");
// First thing in <head>: the page's base, as described above, then its stylesheets.
const headScript = `<script>(function(){var d=location.pathname.split("/").slice(1,-1),r=${JSON.stringify(ROUTE_DIRS)};for(var i=0;i<d.length;i++)if(r.indexOf(d[i])>=0){d=d.slice(0,i);break}var h=document.head,b=document.createElement("base"),p="/"+d.map(function(s){return s+"/"}).join("");b.href=p;h.appendChild(b);${JSON.stringify(styles.map(ui))}.forEach(function(s){var l=document.createElement("link");l.rel="stylesheet";l.href=s;h.appendChild(l)})})()</script>`;
// Where the scripts were, in order, as the deferred scripts they were.
const bodyScript = `<script>${JSON.stringify(scripts.map(ui))}.forEach(function(s){var e=document.createElement("script");e.src=s;e.async=false;document.body.appendChild(e)})</script>`;
html = html
  .replace(/^<!DOCTYPE html>/i, "<!doctype html>")
  .replace(/<meta charset="utf-8" \/>/, (m) => `${m}\n    ${headScript}`)
  .replace(/<\/body>/, `${bodyScript}\n</body>`)
  .replace(
    new RegExp(`<link rel="icon" href="${escape(PLACEHOLDER)}/favicon\\.ico"\\s*/?>`),
    `<link rel="icon" href="ui/${MARK.served}" type="image/png">`,
  );
if (!html.includes(headScript) || !html.includes(bodyScript))
  throw new Error("index.html: no place for the loader");
if (html.includes(PLACEHOLDER) || /(href|src)="\//.test(html))
  throw new Error(`index.html keeps an absolute path:\n${html}`);
await writeFile(join(out, "index.html"), html);

const listing = await files(out);
console.log(`web app in ${out}: ${listing.length} files, relative to the page's base`);
