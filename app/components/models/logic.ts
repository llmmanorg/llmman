// The Models tab's pure parts, as the web UI before this app had them, so
// the daemon's answers read the same way they always have. Nothing here knows a
// model name; everything comes from the daemon.

import type { PullProgress } from "@/api/client";
import type { Variant } from "@/api/search";

/** What the Models tab needs of a local model (`/v1/models`). */
export type LocalModel = { id: string; loaded?: boolean };

export type Registry = "hf" | "docker" | "other";

/** `docker.io/ai/qwen3.5:0.8b` → `docker.io/ai/qwen3.5`. */
export function repoOf(ref: string): string {
  const slash = ref.lastIndexOf("/");
  const colon = ref.lastIndexOf(":");
  return colon > slash ? ref.slice(0, colon) : ref;
}

/** `docker.io/ai/qwen3.5:0.8b` → `0.8b`; a ref without a tag is itself. */
export function tagOf(ref: string): string {
  const colon = ref.lastIndexOf(":");
  return colon > ref.lastIndexOf("/") ? ref.slice(colon + 1) : ref;
}

export function registryOf(name: string): Registry {
  if (name.startsWith("hf.co/")) return "hf";
  if (name.startsWith("docker.io/")) return "docker";
  return "other";
}

/** `hf.co/unsloth/Qwen3-GGUF` → `unsloth`. */
export function ownerOf(name: string): string {
  return name.split("/").at(-2) ?? "";
}

/** `hf.co/unsloth/Qwen3-GGUF` → `Qwen3-GGUF`. */
export function shortName(name: string): string {
  return name.split("/").at(-1) ?? name;
}

/** A full reference (registry host or scheme) to pull as typed, rather than a search. */
export function isReference(text: string): boolean {
  return text.includes("://") || /^[\w-]+(\.[\w-]+)+(:\d+)?\//.test(text);
}

/** Whether any tag of repository `name` is already in the local store. */
export function isPulled(name: string, local: readonly LocalModel[]): boolean {
  const repo = name.toLowerCase();
  return local.some((m) => {
    const id = m.id.toLowerCase();
    return id === repo || id.startsWith(`${repo}:`);
  });
}

/** Whether the store holds variant `v`: exactly, or as the `:latest` a tagless pull is stored under. */
export function holds(v: Variant, local: readonly LocalModel[]): boolean {
  const want = v.name.toLowerCase();
  const latest = `${repoOf(want)}:latest`;
  return local.some((m) => {
    const id = m.id.toLowerCase();
    return id === want || (v.default && id === latest);
  });
}

/** Local models, one row per repo, with the tags this computer holds. */
export function localRepos<M extends LocalModel>(local: readonly M[]): { name: string; local: M[] }[] {
  const byRepo = new Map<string, { name: string; local: M[] }>();
  for (const m of local) {
    const repo = repoOf(m.id);
    let row = byRepo.get(repo);
    if (!row) byRepo.set(repo, (row = { name: repo, local: [] }));
    row.local.push(m);
  }
  return [...byRepo.values()];
}

/** The local models that are tags of repository `name`. */
export function localOf<M extends LocalModel>(name: string, local: readonly M[]): M[] {
  const repo = name.toLowerCase();
  return local.filter((m) => repoOf(m.id).toLowerCase() === repo);
}

// ---- Fit ----------------------------------------------------------------

export type Fit = "ok" | "warn" | "bad";

export const FIT_WORDS: Record<Fit, string> = {
  ok: "Fits this computer",
  warn: "Tight on this computer",
  bad: "Too big for this computer",
};

/** ok / tight / too big, against this computer's model memory; null when either is unknown. */
export function fitOf(size: number, memory: number): Fit | null {
  if (!size || !memory) return null;
  const share = size / memory;
  return share <= 0.6 ? "ok" : share <= 0.9 ? "warn" : "bad";
}

/** The line under the variant picker. */
export function fitSentence(v: Variant, memory: number): string {
  const fit = fitOf(v.size, memory);
  if (!fit) return v.size ? "" : "Size unknown until pulled.";
  return `${FIT_WORDS[fit]}: ${formatBytes(v.size)} of weights, ${formatBytes(memory)} of model memory (context and runtime need more).`;
}

// ---- Variants -----------------------------------------------------------

/** Above this many variants the picker offers size chips and a filter. */
export const MANY_VARIANTS = 12;

/** Smallest first, sizes the registry did not report last; otherwise in the daemon's order. */
export function sortBySize(variants: readonly Variant[]): Variant[] {
  return [...variants].sort((a, b) => (a.size || Infinity) - (b.size || Infinity));
}

/** A tag's parameter-count prefix: `35b-a3b-int4` → `35b`, `Q4_K_M` → null. */
export function sizePrefix(tag: string): string | null {
  return /^(\d+(?:\.\d+)?[mb])(?:-|$)/i.exec(tag)?.[1]?.toLowerCase() ?? null;
}

function prefixValue(prefix: string): number {
  return parseFloat(prefix) * (prefix.endsWith("b") ? 1e9 : 1e6);
}

/** The size chips for a picker: each prefix once, smallest model first; none unless there are two. */
export function sizeGroups(variants: readonly Variant[]): string[] {
  const groups = new Set<string>();
  for (const v of variants) {
    const p = sizePrefix(v.tag);
    if (p) groups.add(p);
  }
  if (groups.size < 2) return [];
  return [...groups].sort((a, b) => prefixValue(a) - prefixValue(b));
}

/** The variants a chip (`group`, null for all) and the filter text leave. */
export function filterVariants(variants: readonly Variant[], group: string | null, text: string): Variant[] {
  const needle = text.trim().toLowerCase();
  return variants.filter(
    (v) => (!group || sizePrefix(v.tag) === group) && (!needle || v.tag.toLowerCase().includes(needle)),
  );
}

/**
 * The variant a card opens on: the one picked earlier, else the one a
 * tagless pull takes, else the largest that fits this computer comfortably,
 * else the smallest.
 */
export function initialVariant(
  variants: readonly Variant[],
  memory: number,
  picked?: string | null,
): Variant | undefined {
  const bySize = sortBySize(variants);
  return (
    (picked ? variants.find((v) => sameRef(v.name, picked)) : undefined) ??
    variants.find((v) => v.default) ??
    bySize.filter((v) => fitOf(v.size, memory) === "ok").at(-1) ??
    bySize[0]
  );
}

/**
 * The version a card shows: as initialVariant, except that a picked
 * reference the registry does not list (a tag typed in the search bar) is
 * shown as itself, its size unknown until pulled.
 */
export function chosenVariant(
  variants: readonly Variant[],
  memory: number,
  picked?: string | null,
): Variant | undefined {
  if (picked && tagOf(picked) !== picked && !variants.some((v) => sameRef(v.name, picked))) {
    return { name: picked, tag: tagOf(picked), size: 0, default: false };
  }
  return initialVariant(variants, memory, picked);
}

function sameRef(a: string, b: string): boolean {
  return a.toLowerCase() === b.toLowerCase();
}

// ---- References typed in the search bar ---------------------------------

/**
 * Whether a typed reference opens its card (a registry the daemon can
 * search, so size and fit can be shown before pulling) rather than asking
 * to pull it as typed (ms://, s3://, another OCI registry…).
 */
export function referenceOpensCard(ref: string): boolean {
  return registryOf(ref) !== "other";
}

/** Whether a typed reference is already here: that tag, or any tag of a tagless one. */
export function holdsRef(ref: string, local: readonly LocalModel[]): boolean {
  if (tagOf(ref) === ref) return isPulled(ref, local);
  return local.some((m) => sameRef(m.id, ref));
}

/** What a screen reader says for a version: `Q4_K_M, default, 17 GB, Fits this computer, on this computer`. */
export function variantLabel(v: Variant, local: readonly LocalModel[], memory: number): string {
  const fit = fitOf(v.size, memory);
  return [
    v.tag,
    v.default && "default",
    v.size ? formatBytes(v.size) : "size unknown",
    fit && FIT_WORDS[fit],
    holds(v, local) && "on this computer",
  ]
    .filter(Boolean)
    .join(", ");
}

// ---- Card text ----------------------------------------------------------

/** The card's tag chips: registry tags without the long `base_model:`/`dataset:` ones. */
export function cardTags(tags: readonly string[] | null | undefined): string[] {
  return (tags ?? []).filter((t) => !t.startsWith("base_model:") && !t.startsWith("dataset:")).slice(0, 14);
}

/**
 * A registry README as Markdown for the app's renderer: front matter and
 * comments dropped, and outside code blocks the HTML model cards are full
 * of reduced to what it says (links, headings, line breaks, text). Images
 * go: most are badges, and the app fetches nothing a README asks for.
 * The renderer still treats all of it as untrusted text. Relative links
 * point into the repo at `page`, when given.
 *
 * Text left from an HTML block (a line starting with a tag, to the next
 * blank line) loses the HTML's indentation, which a browser would collapse
 * but the app's renderer keeps; Markdown outside such blocks keeps its own
 * (nested lists), and fenced code stays byte for byte.
 */
export function readmeMarkdown(text: string, page: string): string {
  // Front matter, after any blank lines (Hugging Face reads it there too).
  const body = text.replace(/^\uFEFF?\s*---[ \t]*\r?\n[\s\S]*?\r?\n---[ \t]*\r?\n/, "");
  // Split on fenced code blocks; odd pieces are code and stay as written.
  const pieces = body.split(/(^```[\s\S]*?^```[^\n]*$)/m);
  const out = pieces.map((piece, i) => {
    if (i % 2) return piece;
    const lines = piece.replace(/<!--[\s\S]*?-->/g, "").split("\n");
    const parts: string[] = [];
    for (let k = 0; k < lines.length;) {
      const start = k;
      if (HTML_BLOCK.test(lines[k]!)) {
        while (k < lines.length && lines[k]!.trim()) k++;
        const reduced = reduceHtml(lines.slice(start, k).join("\n"), page);
        parts.push(
          reduced
            .split("\n")
            .map((l) => l.trim())
            .join("\n"),
        );
      } else {
        while (k < lines.length && !HTML_BLOCK.test(lines[k]!)) k++;
        parts.push(reduceHtml(lines.slice(start, k).join("\n"), page));
      }
    }
    return parts
      .join("\n")
      .replace(/^[ \t]+$/gm, "")
      .replace(/\n{3,}/g, "\n\n");
  });
  return out.join("").trim();
}

/** A line that opens an HTML block (CommonMark: a tag after at most three spaces). */
const HTML_BLOCK = /^ {0,3}<\/?[a-z]/i;

const ENTITIES: Record<string, string> = { amp: "&", lt: "<", gt: ">", quot: '"', "#39": "'", nbsp: " " };

/** HTML in `s` reduced to the Markdown it says; see readmeMarkdown. */
function reduceHtml(s: string, page: string): string {
  const flat = (t: string) =>
    t
      .replace(/<[^>]+>/g, "")
      .replace(/\s+/g, " ")
      .trim();
  s = s.replace(/<img\b[^>]*>/gi, "");
  s = s.replace(/<br\s*\/?>/gi, "\n");
  s = s.replace(
    /<h([1-6])\b[^>]*>([\s\S]*?)<\/h\1>/gi,
    (_, n: string, t: string) => `\n\n${"#".repeat(+n)} ${flat(t)}\n\n`,
  );
  s = s.replace(
    /<a\b[^>]*?href\s*=\s*["']([^"']+)["'][^>]*>([\s\S]*?)<\/a>/gi,
    (_, href: string, t: string) => (flat(t) ? `[${flat(t)}](${href})` : ""),
  );
  s = s.replace(/<\/?(p|div|center|table|thead|tbody|tr|details|summary|ul|ol|li)\b[^>]*>/gi, "\n");
  s = s.replace(/<\/?[a-z][^>\n]*>/gi, "");
  s = s.replace(/&(amp|lt|gt|quot|#39|nbsp);/g, (_, e: string) => ENTITIES[e] ?? "");
  if (page) {
    s = s.replace(
      /\]\((?!https?:|mailto:|#)([^)\s]+)\)/g,
      (_, path: string) => `](${page}/blob/main/${path.replace(/^\.?\//, "")})`,
    );
  }
  return s;
}

// ---- Pull progress ------------------------------------------------------

export type PullState = {
  ref: string;
  status: string;
  /** Bytes for the whole model; 0 until the daemon knows. */
  total: number;
  completed: number;
  /** The daemon's `{notice}` lines (verification), in order. */
  notices: string[];
};

export function startPull(ref: string): PullState {
  return { ref, status: "", total: 0, completed: 0, notices: [] };
}

/**
 * Folds one `/api/pull` line into the state. llmman reports one total for
 * the whole model (no per-layer `digest`), so the latest counts replace the
 * last. `{error}` is the api layer's to throw.
 */
export function applyPullEvent(state: PullState, ev: PullProgress): PullState {
  const next = { ...state };
  if (ev.status) next.status = ev.status;
  if (ev.total) {
    next.total = ev.total;
    next.completed = Math.min(ev.completed ?? 0, ev.total);
  }
  if (ev.notice) next.notices = [...state.notices, ev.notice];
  return next;
}

/** Whole percent done, or null while the size is unknown. */
export function pullPercent(state: PullState): number | null {
  return state.total ? Math.floor((100 * state.completed) / state.total) : null;
}

// ---- Formatting -----------------------------------------------------------

/** 1024-based, as the web UI and the CLI print sizes. */
export function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n < 10 && i > 0 ? n.toFixed(1) : Math.round(n)} ${units[i]}`;
}

/**
 * `1415492` → `1.4M`. By hand rather than Intl's compact notation, which
 * Hermes does not implement on every platform.
 */
export function formatCount(n: number): string {
  if (n < 1000) return String(n);
  const steps: [number, string][] = [
    [1e3, "K"],
    [1e6, "M"],
    [1e9, "B"],
    [1e12, "T"],
  ];
  for (const [size, suffix] of steps) {
    // Rounded first, so 999,960 moves on to 1M rather than reading 1000K.
    const scaled = Math.round((n / size) * 10) / 10;
    if (scaled < 1000 || suffix === "T") return `${scaled}${suffix}`;
  }
  return String(n);
}

/** `1 pull`, `1.4M pulls`: a count with its noun, for labels a screen reader speaks. */
export function counted(n: number, noun: string): string {
  return `${formatCount(n)} ${n === 1 ? noun : `${noun}s`}`;
}

/** The registry's name for people: `Hugging Face`, `Docker Hub`, or empty. */
export function registryName(name: string): string {
  return { hf: "Hugging Face", docker: "Docker Hub", other: "" }[registryOf(name)];
}

/** "3 months ago"; by hand for the same reason (Intl.RelativeTimeFormat). */
export function relativeTime(iso: string, now = Date.now()): string {
  const days = Math.round((Date.parse(iso) - now) / 86_400_000);
  if (!Number.isFinite(days)) return "";
  const say = (n: number, unit: string, last: string, next: string) => {
    if (n === -1) return last;
    if (n === 1) return next;
    const plural = `${Math.abs(n)} ${unit}${Math.abs(n) === 1 ? "" : "s"}`;
    return n < 0 ? `${plural} ago` : `in ${plural}`;
  };
  if (days === 0) return "today";
  if (Math.abs(days) < 31) return say(days, "day", "yesterday", "tomorrow");
  if (Math.abs(days) < 365) return say(Math.round(days / 30), "month", "last month", "next month");
  return say(Math.round(days / 365), "year", "last year", "next year");
}
