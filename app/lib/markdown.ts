// Markdown → a plain tree, ported from llmman's earlier web UI so a reply
// reads the same as it did there. Model output is untrusted: raw HTML
// stays text, links are http(s)/mailto only, and images become links (a
// model must not make the phone fetch a URL unasked). GitHub-flavoured:
// headings, lists nested by indent, fenced code, quotes, tables, rules,
// emphasis, links, autolinks; a single newline is a line break.
//
// components/Markdown.tsx renders the tree; this file has no React.

export type Inline =
  | { type: "text"; text: string }
  | { type: "br" }
  | { type: "code"; text: string }
  | { type: "strong" | "em" | "del"; children: Inline[] }
  | { type: "link"; url: string; title: string; children: Inline[]; image: boolean };

export type Align = "left" | "right" | "center" | "";

export type Block =
  | { type: "paragraph"; text: string; raw: string }
  | { type: "heading"; level: number; text: string; raw: string }
  | { type: "hr"; raw: string }
  | { type: "code"; lang: string; text: string; raw: string }
  | { type: "quote"; children: Block[]; raw: string }
  | { type: "list"; ordered: boolean; start: number; items: Block[][]; raw: string }
  | { type: "table"; header: string[]; aligns: Align[]; rows: string[][]; raw: string };

type Unraw<T> = T extends unknown ? Omit<T, "raw"> : never;
type BlockNoRaw = Unraw<Block>;

const splitLines = (source: string) => source.replace(/\r\n?/g, "\n").split("\n");

/** The blocks of `source`. Each carries its `raw` text, so a re-render can keep unchanged blocks. */
export function parseMarkdown(source: string): Block[] {
  return parseBlocks(splitLines(source));
}

// ---- Block parsing ----------------------------------------------------

const FENCE = /^( {0,3})(`{3,}|~{3,})\s*([^`\s]*)/;
const HEADING = /^ {0,3}(#{1,6})\s+(.*?)(?:\s+#+)?\s*$/;
const HR = /^ {0,3}([-*_])(?:\s*\1){2,}\s*$/;
const UL = /^(\s*)([-*+])\s+(.*)$/;
const OL = /^(\s*)(\d{1,9})[.)]\s+(.*)$/;
const QUOTE = /^ {0,3}>\s?(.*)$/;
const TABLE_SEP = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;

function parseBlocks(lines: string[]): Block[] {
  const blocks: Block[] = [];
  let i = 0;
  while (i < lines.length) {
    const start = i;
    const [block, next] = nextBlock(lines, i);
    i = next;
    if (block) blocks.push({ ...block, raw: lines.slice(start, i).join("\n") } as Block);
  }
  return blocks;
}

/** The block starting at line `i` (null for a blank line) and the index after it. */
function nextBlock(lines: string[], i: number): [BlockNoRaw | null, number] {
  const line = lines[i]!;
  if (!line.trim()) return [null, i + 1];

  let m: RegExpExecArray | null;
  if ((m = FENCE.exec(line))) {
    const fence = m[2]!;
    const lang = m[3] ?? "";
    const body: string[] = [];
    i++;
    while (i < lines.length && !closesFence(lines[i]!, fence)) body.push(lines[i++]!);
    // Past the closing fence, or past the end while streaming.
    return [{ type: "code", lang, text: body.join("\n") }, i + 1];
  }

  if ((m = HEADING.exec(line))) {
    return [{ type: "heading", level: m[1]!.length, text: m[2]! }, i + 1];
  }

  if (HR.test(line)) return [{ type: "hr" }, i + 1];

  if (QUOTE.test(line)) {
    const inner: string[] = [];
    while (i < lines.length && (m = QUOTE.exec(lines[i]!))) {
      inner.push(m[1]!);
      i++;
    }
    // Lazy continuation: a non-blank line right after keeps the quote.
    while (i < lines.length && lines[i]!.trim() && !startsBlock(lines[i]!)) inner.push(lines[i++]!);
    return [{ type: "quote", children: parseBlocks(inner) }, i];
  }

  if (UL.test(line) || OL.test(line)) return parseList(lines, i);

  if (i + 1 < lines.length && line.includes("|") && TABLE_SEP.test(lines[i + 1]!)) {
    const header = splitRow(line);
    const aligns = splitRow(lines[i + 1]!).map((c): Align => {
      const l = c.startsWith(":");
      const r = c.endsWith(":");
      return l && r ? "center" : r ? "right" : l ? "left" : "";
    });
    i += 2;
    const rows: string[][] = [];
    while (i < lines.length && lines[i]!.trim() && lines[i]!.includes("|")) rows.push(splitRow(lines[i++]!));
    return [{ type: "table", header, aligns, rows }, i];
  }

  // Paragraph: until a blank line or the start of another block.
  const para = [line];
  i++;
  while (i < lines.length && lines[i]!.trim() && !startsBlock(lines[i]!)) para.push(lines[i++]!);
  // Leading spaces are layout, not content: the web UI's <p> collapses them.
  return [{ type: "paragraph", text: para.map((l) => l.trimStart()).join("\n") }, i];
}

function closesFence(line: string, fence: string): boolean {
  const m = /^ {0,3}(`{3,}|~{3,})\s*$/.exec(line);
  return !!m && m[1]![0] === fence[0] && m[1]!.length >= fence.length;
}

function startsBlock(line: string): boolean {
  return (
    FENCE.test(line) ||
    HEADING.test(line) ||
    HR.test(line) ||
    QUOTE.test(line) ||
    UL.test(line) ||
    OL.test(line)
  );
}

function splitRow(line: string): string[] {
  let s = line.trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|") && !s.endsWith("\\|")) s = s.slice(0, -1);
  const cells: string[] = [];
  let cur = "";
  let inCode = false;
  for (let k = 0; k < s.length; k++) {
    const c = s[k];
    if (c === "`") inCode = !inCode;
    if (c === "\\" && s[k + 1] === "|") {
      cur += "|";
      k++;
    } else if (c === "|" && !inCode) {
      cells.push(cur.trim());
      cur = "";
    } else {
      cur += c;
    }
  }
  cells.push(cur.trim());
  return cells;
}

/** Parse a list starting at `start`; returns `[block, nextIndex]`. */
function parseList(lines: string[], start: number): [BlockNoRaw, number] {
  const first = (UL.exec(lines[start]!) || OL.exec(lines[start]!))!;
  const ordered = /\d/.test(first[2]!);
  const indent = first[1]!.length;
  const items: Block[][] = [];
  let i = start;
  const startNumber = ordered ? parseInt(first[2]!, 10) : 1;

  while (i < lines.length) {
    const line = lines[i]!;
    const m = ordered ? OL.exec(line) : UL.exec(line);
    // A differently-indented or differently-typed item ends this list
    // unless it's deeper (handled as a child below).
    if (!m || m[1]!.length !== indent) break;
    const contentIndent = m[0].length - m[3]!.length;
    const body = [m[3]!];
    i++;
    // Continuation lines: blank lines followed by indented content, or
    // indented lines, belong to this item.
    while (i < lines.length) {
      const next = lines[i]!;
      if (!next.trim()) {
        // Look ahead: blank then indented → still this item (loose list).
        let j = i;
        while (j < lines.length && !lines[j]!.trim()) j++;
        if (j < lines.length && leading(lines[j]!) >= contentIndent) {
          body.push("");
          i = j;
          continue;
        }
        break;
      }
      if (leading(next) >= contentIndent) {
        body.push(next.slice(contentIndent));
        i++;
        continue;
      }
      // A nested list marker with more indent than ours, but less than
      // the content column (common: 2-space nesting under "- ").
      const nm = UL.exec(next) || OL.exec(next);
      if (nm && nm[1]!.length > indent) {
        body.push(next.slice(Math.min(nm[1]!.length, contentIndent)));
        i++;
        continue;
      }
      // Lazy paragraph continuation.
      if (!startsBlock(next) && body.length && body[body.length - 1]!.trim()) {
        body.push(next.trim());
        i++;
        continue;
      }
      break;
    }
    items.push(parseBlocks(body));
  }
  return [{ type: "list", ordered, start: startNumber, items }, i];
}

function leading(line: string): number {
  return line.length - line.trimStart().length;
}

// ---- Inline -----------------------------------------------------------

const SAFE_URL = /^(https?:|mailto:)/i;
const AUTOLINK = /^https?:\/\/[^\s<>()]+[^\s<>().,;:!?'"]/i;

/** Whether the app may open `url` (http, https, mailto). */
export function isSafeUrl(url: string): boolean {
  return SAFE_URL.test(url);
}

/** The inline nodes of `text`. */
export function parseInline(text: string): Inline[] {
  const out: Inline[] = [];
  let i = 0;
  let buf = "";
  // No `]` past here means no link can start here: skips the quadratic
  // rescans a reply full of stray `[` would otherwise cause.
  const lastClose = text.lastIndexOf("]");
  const flush = () => {
    if (buf) out.push({ type: "text", text: buf });
    buf = "";
  };

  while (i < text.length) {
    const c = text[i]!;

    // Backslash escapes.
    if (c === "\\" && i + 1 < text.length) {
      const n = text[i + 1]!;
      if (n === "\n") {
        flush();
        out.push({ type: "br" });
        i += 2;
        continue;
      }
      if (/[\\`*_{}[\]()#+\-.!|~<>]/.test(n)) {
        buf += n;
        i += 2;
        continue;
      }
    }

    if (c === "\n") {
      flush();
      out.push({ type: "br" });
      i++;
      continue;
    }

    // Inline code: matching run of backticks.
    if (c === "`") {
      let run = 0;
      while (text[i + run] === "`") run++;
      const close = text.indexOf("`".repeat(run), i + run);
      if (close > 0 && text[close + run] !== "`") {
        flush();
        let inner = text.slice(i + run, close);
        if (inner.startsWith(" ") && inner.endsWith(" ") && inner.trim()) inner = inner.slice(1, -1);
        out.push({ type: "code", text: inner });
        i = close + run;
        continue;
      }
      buf += "`".repeat(run);
      i += run;
      continue;
    }

    // Images render as links (no unasked fetches); links are http(s)/mailto.
    if (c === "!" && text[i + 1] === "[" && i < lastClose) {
      const link = matchLink(text, i + 1);
      if (link) {
        flush();
        pushLink(out, link, `Image: ${link.label || link.url}`);
        i = link.end;
        continue;
      }
    }
    if (c === "[" && i < lastClose) {
      const link = matchLink(text, i);
      if (link) {
        flush();
        pushLink(out, link, null);
        i = link.end;
        continue;
      }
    }

    // Autolinks.
    if ((c === "h" || c === "H") && (i === 0 || /[\s(<]/.test(text[i - 1]!))) {
      const m = AUTOLINK.exec(text.slice(i));
      if (m) {
        flush();
        out.push({
          type: "link",
          url: m[0],
          title: "",
          children: [{ type: "text", text: m[0] }],
          image: false,
        });
        i += m[0].length;
        continue;
      }
    }

    // Emphasis: ** / __ strong, * / _ em, ~~ strike. A run that opens
    // nothing is taken as text whole, so it is scanned once.
    if (c === "*" || c === "_" || c === "~") {
      const span = matchEmphasis(text, i);
      if (span) {
        flush();
        out.push({ type: span.tag, children: parseInline(span.inner) });
        i = span.end;
        continue;
      }
      let run = i;
      while (text[run] === c) run++;
      buf += text.slice(i, run);
      i = run;
      continue;
    }

    buf += c;
    i++;
  }
  flush();
  return out;
}

interface LinkMatch {
  label: string;
  url: string;
  title: string;
  end: number;
}

/** A link, or its label as text when the URL scheme is not allowed. */
function pushLink(out: Inline[], link: LinkMatch, plainLabel: string | null) {
  if (!SAFE_URL.test(link.url)) {
    out.push(...parseInline(link.label));
    return;
  }
  out.push({
    type: "link",
    url: link.url,
    title: link.title,
    children: plainLabel !== null ? [{ type: "text", text: plainLabel }] : parseInline(link.label),
    image: plainLabel !== null,
  });
}

/** Longest link, label or emphasis span considered; bounds what a stray marker costs. */
const MAX_SPAN = 2000;

/** `[label](url "title")` at `start`, or null. */
function matchLink(text: string, start: number): LinkMatch | null {
  if (text[start] !== "[") return null;
  let depth = 0;
  let i = start;
  const limit = Math.min(text.length, start + MAX_SPAN);
  for (; i < limit; i++) {
    if (text[i] === "\\") {
      i++;
      continue;
    }
    if (text[i] === "[") depth++;
    else if (text[i] === "]") {
      depth--;
      if (depth === 0) break;
    }
  }
  if (i >= limit || text[i + 1] !== "(") return null;
  const label = text.slice(start + 1, i);
  let j = i + 2;
  let paren = 1;
  let url = "";
  const urlLimit = Math.min(text.length, j + MAX_SPAN);
  for (; j < urlLimit; j++) {
    const ch = text[j]!;
    if (ch === "\\" && j + 1 < text.length) {
      url += text[j + 1];
      j++;
      continue;
    }
    if (ch === "(") paren++;
    if (ch === ")") {
      paren--;
      if (paren === 0) break;
    }
    url += ch;
  }
  if (j >= urlLimit) return null;
  let title = "";
  const tm = /^(\S*)\s+["'(](.*)["')]$/.exec(url.trim());
  if (tm) {
    url = tm[1]!;
    title = tm[2]!;
  }
  url = url.trim();
  if (url.startsWith("<") && url.endsWith(">")) url = url.slice(1, -1);
  return { label, url, title, end: j + 1 };
}

/** Emphasis at `start`, or null. */
function matchEmphasis(
  text: string,
  start: number,
): { tag: "strong" | "em" | "del"; inner: string; end: number } | null {
  const c = text[start]!;
  let run = 0;
  while (text[start + run] === c) run++;
  if (c === "~" && run < 2) return null;
  const marker = c.repeat(Math.min(run, 2));
  // Left-flanking: not followed by whitespace.
  const after = text[start + marker.length];
  if (!after || /\s/.test(after)) return null;
  // `_` inside a word is literal (snake_case).
  if (c === "_" && start > 0 && /\w/.test(text[start - 1]!)) return null;

  const close = findClose(text, start + marker.length, marker);
  if (close < 0) return null;
  const inner = text.slice(start + marker.length, close);
  if (!inner.trim()) return null;
  const tag = c === "~" ? "del" : marker.length >= 2 ? "strong" : "em";
  return { tag, inner, end: close + marker.length };
}

function findClose(text: string, from: number, marker: string): number {
  const c = marker[0]!;
  const limit = Math.min(text.length, from + MAX_SPAN);
  for (let i = from; i < limit; i++) {
    if (text[i] === "\\") {
      i++;
      continue;
    }
    if (text[i] === "`") {
      // Skip inline code.
      const end = text.indexOf("`", i + 1);
      if (end > 0) {
        i = end;
        continue;
      }
    }
    if (text.startsWith(marker, i) && text[i + marker.length] !== c && !/\s/.test(text[i - 1] || "")) {
      if (c === "_" && /\w/.test(text[i + marker.length] || "")) continue;
      return i;
    }
  }
  return -1;
}
