#!/usr/bin/env node
// Writes components/iconPaths.generated.ts: the Material Symbols glyphs the
// app uses, as SVG paths, for the web's Icon (components/Icon.web.tsx).
// Android draws them from expo-symbols' font; the web page draws the same
// outlines inline instead of downloading the whole font (~1 MB).
//
// The names are found in the source (`android: "name"` or `web: "name"` in
// an Icon's name), and their outlines read from the font expo-symbols
// ships, so they match Android's exactly. Run `npm run make-web-icons`
// after using a new icon; a test fails until then.

import { execFileSync } from "node:child_process";
import { readdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const modules = join(root, "node_modules");
const FONT = join(modules, "@expo-google-fonts/material-symbols/400Regular/MaterialSymbols_400Regular.ttf");
const CODEPOINTS = JSON.parse(
  await readFile(join(modules, "expo-symbols/build/android/symbols.json"), "utf8"),
);
const OUT = join(root, "components/iconPaths.generated.ts");

/** Every Material Symbols name the source gives an Icon. */
export async function usedIconNames() {
  const names = new Set();
  async function walk(dir) {
    for (const entry of await readdir(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) {
        if (entry.name !== "__tests__" && entry.name !== "node_modules") await walk(path);
      } else if (/\.tsx?$/.test(entry.name) && !entry.name.endsWith(".generated.ts")) {
        const text = await readFile(path, "utf8");
        for (const m of text.matchAll(/\b(?:android|web):\s*"([a-z0-9_]+)"/g)) names.add(m[1]);
      }
    }
  }
  for (const dir of ["app", "components", "lib"]) await walk(join(root, dir));
  return [...names].sort();
}

// A minimal TrueType reader: cmap (formats 4 and 12), loca and glyf, which
// is all it takes to turn a code point into its outline.
function readFont(buf) {
  const tables = {};
  for (let i = 0; i < buf.readUInt16BE(4); i++) {
    const o = 12 + 16 * i;
    tables[buf.toString("latin1", o, o + 4)] = buf.readUInt32BE(o + 8);
  }
  const longLoca = buf.readInt16BE(tables.head + 50) === 1;
  const unitsPerEm = buf.readUInt16BE(tables.head + 18);

  const cmap = new Map();
  const c = tables.cmap;
  for (let i = 0; i < buf.readUInt16BE(c + 2); i++) {
    const sub = c + buf.readUInt32BE(c + 8 + 8 * i);
    const format = buf.readUInt16BE(sub);
    if (format === 12) {
      const groups = buf.readUInt32BE(sub + 12);
      for (let g = 0; g < groups; g++) {
        const p = sub + 16 + 12 * g;
        const [start, end, glyph] = [buf.readUInt32BE(p), buf.readUInt32BE(p + 4), buf.readUInt32BE(p + 8)];
        for (let cp = start; cp <= end; cp++) if (!cmap.has(cp)) cmap.set(cp, glyph + cp - start);
      }
    } else if (format === 4) {
      const segs = buf.readUInt16BE(sub + 6) / 2;
      const ends = sub + 14;
      const starts = ends + 2 * segs + 2;
      const deltas = starts + 2 * segs;
      const offsets = deltas + 2 * segs;
      for (let s = 0; s < segs; s++) {
        const [end, start] = [buf.readUInt16BE(ends + 2 * s), buf.readUInt16BE(starts + 2 * s)];
        const [delta, ro] = [buf.readInt16BE(deltas + 2 * s), buf.readUInt16BE(offsets + 2 * s)];
        for (let cp = start; cp <= end && cp !== 0xffff; cp++) {
          if (cmap.has(cp)) continue;
          let glyph;
          if (ro === 0) glyph = (cp + delta) & 0xffff;
          else {
            glyph = buf.readUInt16BE(offsets + 2 * s + ro + 2 * (cp - start));
            if (glyph) glyph = (glyph + delta) & 0xffff;
          }
          cmap.set(cp, glyph);
        }
      }
    }
  }

  const loca = (g) =>
    longLoca ? buf.readUInt32BE(tables.loca + 4 * g) : buf.readUInt16BE(tables.loca + 2 * g) * 2;

  /** The glyph's contours: arrays of {x, y, on}. */
  function contours(g, transform = (x, y) => [x, y]) {
    const start = tables.glyf + loca(g);
    if (loca(g + 1) === loca(g)) return [];
    const n = buf.readInt16BE(start);
    if (n < 0) {
      const out = [];
      let p = start + 10;
      for (;;) {
        const flags = buf.readUInt16BE(p);
        const glyph = buf.readUInt16BE(p + 2);
        p += 4;
        let dx, dy;
        if (flags & 1) [dx, dy, p] = [buf.readInt16BE(p), buf.readInt16BE(p + 2), p + 4];
        else [dx, dy, p] = [buf.readInt8(p), buf.readInt8(p + 1), p + 2];
        if (!(flags & 2)) throw new Error(`glyph ${g}: point-matched components are not supported`);
        let [a, b, cc, d] = [1, 0, 0, 1];
        const f2 = (o) => buf.readInt16BE(o) / 16384;
        if (flags & 8) [a, d, p] = [f2(p), f2(p), p + 2];
        else if (flags & 0x40) [a, d, p] = [f2(p), f2(p + 2), p + 4];
        else if (flags & 0x80) [a, b, cc, d, p] = [f2(p), f2(p + 2), f2(p + 4), f2(p + 6), p + 8];
        out.push(...contours(glyph, (x, y) => transform(a * x + cc * y + dx, b * x + d * y + dy)));
        if (!(flags & 0x20)) return out;
      }
    }
    const endPts = Array.from({ length: n }, (_, i) => buf.readUInt16BE(start + 10 + 2 * i));
    const count = n ? endPts[n - 1] + 1 : 0;
    let p = start + 10 + 2 * n;
    p += 2 + buf.readUInt16BE(p);
    const flags = [];
    while (flags.length < count) {
      const f = buf[p++];
      flags.push(f);
      if (f & 8) for (let r = buf[p++]; r > 0; r--) flags.push(f);
    }
    const coords = (short, same) => {
      let v = 0;
      return flags.map((f) => {
        if (f & short) {
          const d = buf[p++];
          v += f & same ? d : -d;
        } else if (!(f & same)) {
          v += buf.readInt16BE(p);
          p += 2;
        }
        return v;
      });
    };
    const xs = coords(2, 0x10);
    const ys = coords(4, 0x20);
    const out = [];
    let first = 0;
    for (const end of endPts) {
      const pts = [];
      for (let i = first; i <= end; i++) {
        const [x, y] = transform(xs[i], ys[i]);
        pts.push({ x, y, on: !!(flags[i] & 1) });
      }
      out.push(pts);
      first = end + 1;
    }
    return out;
  }

  return { unitsPerEm, glyphOf: (cp) => cmap.get(cp), contours };
}

const num = (v) => String(Math.round(v * 10) / 10);

/** Contours as SVG path data, y flipped so the em box is `0 -em em em`. */
function pathData(contourList) {
  const pt = ({ x, y }) => `${num(x)} ${num(-y)}`.replace(" -", "-");
  const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, on: true });
  let d = "";
  for (const pts of contourList) {
    // Variable fonts keep empty contours collapsed to a point: nothing to draw.
    if (pts.every((q) => q.x === pts[0].x && q.y === pts[0].y)) continue;
    const startIdx = pts.findIndex((q) => q.on);
    let start;
    let seq;
    if (startIdx < 0) {
      start = mid(pts[0], pts[1]);
      seq = [...pts.slice(1), pts[0]];
    } else {
      start = pts[startIdx];
      seq = [...pts.slice(startIdx + 1), ...pts.slice(0, startIdx)];
    }
    seq.push(start);
    d += `M${pt(start)}`;
    for (let i = 0; i < seq.length; i++) {
      const q = seq[i];
      if (q.on) {
        d += `L${pt(q)}`;
        continue;
      }
      const next = seq[i + 1];
      const end = next.on ? next : mid(q, next);
      d += `Q${pt(q)} ${pt(end)}`;
      if (next.on) i++;
    }
    // The closing line back to the start is implied by Z.
    const close = `L${pt(start)}`;
    if (d.endsWith(close)) d = d.slice(0, -close.length);
    d += "Z";
  }
  return d;
}

async function main() {
  const font = readFont(await readFile(FONT));
  if (font.unitsPerEm !== 960) throw new Error(`unexpected unitsPerEm ${font.unitsPerEm}`);
  const names = await usedIconNames();
  const entries = names.map((name) => {
    const cp = CODEPOINTS[name];
    if (cp === undefined) throw new Error(`${name} is not a Material Symbols name`);
    const glyph = font.glyphOf(cp);
    if (!glyph) throw new Error(`${name}: no glyph for U+${cp.toString(16)}`);
    return `  ${/^[a-z_][a-z0-9_]*$/.test(name) ? name : JSON.stringify(name)}: ${JSON.stringify(pathData(font.contours(glyph)))},`;
  });
  const text = `// Generated by scripts/make-web-icons.mjs from Material Symbols (Apache-2.0),
// as @expo-google-fonts/material-symbols ships it; do not edit.
// Paths are in a 960-unit em box: viewBox="0 -960 960 960".

export const ICON_PATHS: Record<string, string> = {
${entries.join("\n")}
};
`;
  await writeFile(OUT, text);
  execFileSync(join(root, "node_modules/.bin/prettier"), ["--write", "--log-level", "warn", OUT]);
  console.log(`wrote ${names.length} icons to components/iconPaths.generated.ts`);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) await main();
