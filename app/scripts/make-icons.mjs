#!/usr/bin/env node
// Generates the app's icons and splash image from llmman's manatee mark,
// assets/llmman-mark.png. None of them are committed (llmman keeps no
// binaries in git): the mark is downloaded from llmman's docs-assets
// release, pinned by SHA-256 (the file llmman's build.rs serves as
// ui/manatee.png), and everything else derives from it. `npm ci` runs this
// with --if-missing (the prepare script); `npm run make-icons` remakes all.
//
// The mark is 294×213. Home-screen icons are drawn at about 180 px and
// scale it down, but the 1024 px masters and the splash scale it up, so
// it is kept moderate on the canvas rather than stretched to fill it.

import { Buffer } from "node:buffer";
import { createHash } from "node:crypto";
import { access, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const asset = (name) => join(root, "assets", name);
const MARK = asset("llmman-mark.png");
const MARK_URL = "https://github.com/llmmanorg/llmman/releases/download/docs-assets/llmman-mark.png";
const MARK_SHA256 = "c6036711a349dc8f8b2b4405be90b342b477dbcf4a7d95eb7d05b42702a241d5";
const OUTPUTS = [
  "icon.png",
  "icon-dark.png",
  "icon-tinted.png",
  "android-icon-foreground.png",
  "android-icon-monochrome.png",
  "favicon.png",
];

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const exists = (path) =>
  access(path).then(
    () => true,
    () => false,
  );

if (process.argv.includes("--if-missing")) {
  const all = await Promise.all(["llmman-mark.png", ...OUTPUTS].map((name) => exists(asset(name))));
  if (all.every(Boolean)) process.exit(0);
}

const have = await readFile(MARK).catch(() => null);
if (!have || sha256(have) !== MARK_SHA256) {
  const r = await fetch(MARK_URL);
  if (!r.ok) throw new Error(`${MARK_URL}: ${r.status}`);
  const bytes = Buffer.from(await r.arrayBuffer());
  if (sha256(bytes) !== MARK_SHA256) throw new Error(`${MARK_URL} does not match the pinned SHA-256`);
  await writeFile(MARK, bytes);
}

// The background behind the mark, shared with app.config.ts: a pale tint of
// its teal, so its dark outline reads; dark is the website's background.
const { background: BACKGROUND } = JSON.parse(await readFile(asset("brand.json"), "utf8"));

const SIZE = 1024;

/** The mark `width` px wide, resized with Lanczos, as a PNG buffer. */
async function mark(width, transform = (img) => img) {
  return transform(sharp(MARK).resize({ width, kernel: "lanczos3" }))
    .png()
    .toBuffer();
}

/** `layer` centred on a SIZE×SIZE canvas of `background`; transparent when null, and fully opaque otherwise (iOS rejects an icon with an alpha channel). */
async function canvas(layer, background, out) {
  const meta = await sharp(layer).metadata();
  const image = sharp({
    create: {
      width: SIZE,
      height: SIZE,
      channels: 4,
      background: background ?? { r: 0, g: 0, b: 0, alpha: 0 },
    },
  }).composite([
    { input: layer, left: Math.round((SIZE - meta.width) / 2), top: Math.round((SIZE - meta.height) / 2) },
  ]);
  const flat = background
    ? sharp(await image.png().toBuffer())
        .flatten({ background })
        .removeAlpha()
    : image;
  await flat.png().toFile(out);
}

/**
 * The mark as one colour, opacity carrying its drawing: the dark outline
 * solid, the light body faint. For icons the system colours itself (Android
 * themed icons, iOS tinted icons).
 */
async function silhouette(width, rgb) {
  const { data, info } = await sharp(MARK)
    .resize({ width, kernel: "lanczos3" })
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const out = Buffer.alloc(data.length);
  for (let i = 0; i < data.length; i += 4) {
    const lum = (0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2]) / 255;
    out[i] = rgb[0];
    out[i + 1] = rgb[1];
    out[i + 2] = rgb[2];
    out[i + 3] = Math.round(data[i + 3] * (lum < 0.5 ? 1 : 0.45));
  }
  return sharp(out, { raw: { width: info.width, height: info.height, channels: 4 } })
    .png()
    .toBuffer();
}

// iOS: opaque on the light background; dark is the mark alone (the system
// draws the dark backdrop); tinted is a grayscale mark the system colours.
const IOS_WIDTH = 600;
// Android adaptive icons: the launcher masks the 108 dp layer to as little
// as a 66 dp circle, so the whole mark (its diagonal) stays inside 66%.
const ANDROID_WIDTH = 540;

await canvas(await mark(IOS_WIDTH), BACKGROUND, asset("icon.png"));
await canvas(await mark(IOS_WIDTH), null, asset("icon-dark.png"));
await canvas(await mark(IOS_WIDTH, (img) => img.grayscale()), null, asset("icon-tinted.png"));
await canvas(await mark(ANDROID_WIDTH), null, asset("android-icon-foreground.png"));
await canvas(await silhouette(ANDROID_WIDTH, [255, 255, 255]), null, asset("android-icon-monochrome.png"));
// The splash shows assets/llmman-mark.png itself (see app.config.ts).
await sharp(MARK)
  .resize(48, 48, { fit: "contain", kernel: "lanczos3", background: { r: 0, g: 0, b: 0, alpha: 0 } })
  .png()
  .toFile(asset("favicon.png"));

console.log("icons written to assets/");
