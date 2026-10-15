// One friendly name for a model, wherever it shows (the picker, the chat
// header, a reply's footer, the Models list): the family and size, then
// the quantization, "SmolLM2 135M · Q4_K_M" for
// `hf.co/unsloth/SmolLM2-135M-Instruct-GGUF:Q4_K_M`. The publisher is
// added only to tell apart two refs that would read the same. A hosted
// model reads as its catalog name when the catalog has one, else its id
// made readable ("Claude Opus 4.5" for `claude-opus-4-5-20251101`). The
// raw ref stays available to screens as a secondary line.

import { splitRemoteRef } from "@/api/client";
import { ownerOf, repoOf, shortName, tagOf } from "@/components/models/logic";

/** Words a family name spells its own way; anything else all-lowercase gets a capital. */
const WORDS: Record<string, string> = {
  smollm: "SmolLM",
  qwen: "Qwen",
  qwq: "QwQ",
  llama: "Llama",
  gemma: "Gemma",
  mistral: "Mistral",
  mixtral: "Mixtral",
  deepseek: "DeepSeek",
  phi: "Phi",
  gpt: "GPT",
  glm: "GLM",
  oss: "OSS",
  olmo: "OLMo",
  vl: "VL",
  moe: "MoE",
  lfm: "LFM",
  claude: "Claude",
  gemini: "Gemini",
  grok: "Grok",
  kimi: "Kimi",
  minimax: "MiniMax",
};

/** Repo or tag words that say how the weights are stored, not what the model is. */
const FORMAT = /^(mlx|safetensors|awq|gptq|exl2|onnx)$/i;
/** Words that say nothing to tell models apart: chat tuning, which every chat model has, and GGUF, the usual format. */
const TUNING = /^(instruct|chat|it|hf|gguf)$/i;
const QUANT = /^(i?q\d[\w]*|f16|f32|bf16|fp16|fp8|fp4|int4|int8|nvfp4|mxfp4|\d+bit)$/i;
const SIZE = /^(\d+(?:\.\d+)?[mbkt]|\d+x\d+(?:\.\d+)?b|a\d+(?:\.\d+)?b|e\d+(?:\.\d+)?b)$/i;

function word(w: string): string {
  const m = /^([a-z]+)(.*)$/i.exec(w);
  if (!m) return w;
  const [, letters, rest] = m as unknown as [string, string, string];
  const known = WORDS[letters.toLowerCase()];
  if (known) return known + rest;
  return w === w.toLowerCase() ? w[0]!.toUpperCase() + w.slice(1) : w;
}

function size(w: string): string {
  // `135M`, `0.8B`, `8x7B`, `A3B` (active parameters), `E4B` (effective).
  return w.replace(/[mbkt]$/i, (u) => u.toUpperCase()).replace(/^[ae]/i, (p) => p.toUpperCase());
}

function quant(w: string): string {
  if (/^\d+bit$/i.test(w)) return `${parseInt(w, 10)}-bit`;
  if (/^mlx$/i.test(w)) return "MLX";
  if (/^safetensors$/i.test(w)) return "safetensors";
  return w.toUpperCase();
}

/** The parts of a local ref's name: family, size, and how it is stored. */
export function localNameParts(ref: string): { family: string; size: string; quant: string } {
  const repo = repoOf(ref);
  const tag = tagOf(ref);
  const families: string[] = [];
  const sizes: string[] = [];
  const quants: string[] = [];
  for (const w of shortName(repo).split("-").filter(Boolean)) {
    if (SIZE.test(w)) sizes.push(size(w));
    else if (QUANT.test(w) || FORMAT.test(w)) quants.push(quant(w));
    else if (!TUNING.test(w)) families.push(word(w));
  }
  const extra: string[] = [];
  if (tag !== ref && tag.toLowerCase() !== "latest") {
    // A tag like `0.8b-safetensors`, `9b-mtp-q8_0` or `Q4_K_M`.
    for (const w of tag.split("-").filter(Boolean)) {
      if (SIZE.test(w)) {
        if (!sizes.some((s) => s.toLowerCase() === w.toLowerCase())) sizes.push(size(w));
      } else if (QUANT.test(w) || FORMAT.test(w)) quants.push(quant(w));
      else if (!TUNING.test(w) && w.toLowerCase() !== "latest")
        extra.push(w.length <= 4 ? w.toUpperCase() : w);
    }
  }
  const family = families.join(" ") || shortName(repo);
  return {
    family,
    size: [...sizes, ...extra].join(" "),
    quant: [...new Set(quants)].join(" "),
  };
}

/** "Qwen3.5 0.8B", "SmolLM2 135M · Q4_K_M". */
export function localName(ref: string): string {
  const p = localNameParts(ref);
  const head = p.size ? `${p.family} ${p.size}` : p.family;
  return p.quant ? `${head} · ${p.quant}` : head;
}

/** A repository's name in a list ("SmolLM2 135M" for `hf.co/unsloth/SmolLM2-135M-Instruct-GGUF`). */
export function repoName(repo: string): string {
  const p = localNameParts(repoOf(repo));
  return p.size ? `${p.family} ${p.size}` : p.family;
}

/**
 * A hosted model's id made readable: no release date, versions as
 * numbers ("claude-opus-4-5-20251101" → "Claude Opus 4.5"), a free tier
 * after a dot ("llama-3.1-8b:free" → "Llama 3.1 8B · free").
 */
export function hostedName(id: string): string {
  const colon = id.lastIndexOf(":");
  const variant = colon > id.lastIndexOf("/") ? id.slice(colon + 1) : "";
  let base = (variant ? id.slice(0, colon) : id).split("/").at(-1) ?? id;
  base = base.replace(/[-_@](\d{8}|\d{4}-\d{2}-\d{2}|\d{4})$/, "").replace(/-latest$/, "");
  const parts = base.split(/[-_ ]+/).filter(Boolean);
  const out: string[] = [];
  for (let i = 0; i < parts.length; i++) {
    const w = parts[i]!;
    const prev = out[out.length - 1];
    // `4-5` is version 4.5.
    if (/^\d{1,2}$/.test(w) && prev !== undefined && /^\d{1,2}(\.\d+)?$/.test(prev) && !prev.includes(".")) {
      out[out.length - 1] = `${prev}.${w}`;
    } else if (/^gpt$/i.test(w) && parts[i + 1]) {
      // GPT keeps its hyphen: GPT-4o, GPT-5.
      out.push(`GPT-${parts[++i]}`);
    } else if (SIZE.test(w)) out.push(size(w));
    else if (/^o\d/i.test(w)) out.push(w.toLowerCase());
    else out.push(word(w));
  }
  const name = out.join(" ") || id;
  return variant ? `${name} · ${variant}` : name;
}

export interface NameSources {
  /** Hosted providers and, when fetched, their models (a catalog `name` wins). */
  providers?: { id: string; name: string; models?: { id: string; name?: unknown }[] | null }[];
  /** Every local ref the server has, to tell same-named ones apart. */
  local?: readonly string[];
}

/** The friendly name of any ref; "Choose a model" for none. */
export function modelName(ref: string, sources: NameSources = {}): string {
  if (!ref) return "Choose a model";
  const remote = splitRemoteRef(ref);
  if (remote) {
    const provider = sources.providers?.find((p) => p.id === remote.provider);
    const named = provider?.models?.find((m) => m.id === remote.model)?.name;
    return typeof named === "string" && named.trim() ? named.trim() : hostedName(remote.model);
  }
  const name = localName(ref);
  const owner = ownerOf(repoOf(ref));
  const twins = (sources.local ?? []).filter((r) => r !== ref && localName(r) === name);
  if (!twins.length || !owner) return name;
  // Only the publisher differs: say whose it is.
  return `${name} · ${owner}`;
}

/** The provider's name for a hosted ref ("Anthropic"); empty for a local one. */
export function providerName(ref: string, providers: { id: string; name: string }[] = []): string {
  const remote = splitRemoteRef(ref);
  if (!remote) return "";
  return providers.find((p) => p.id === remote.provider)?.name ?? remote.provider;
}

const EFFORTS: Record<string, string> = { xhigh: "Extra high", xlow: "Extra low" };

/** A thinking level as people read it: "Default" for none, "Extra high" for `xhigh`. */
export function effortLabel(level: string | undefined): string {
  if (!level) return "Default";
  return EFFORTS[level] ?? level[0]!.toUpperCase() + level.slice(1).replace(/[-_]/g, " ");
}

/**
 * Whether the server still offers `ref`: a local one in its list, a hosted
 * one while its provider has a usable key (the provider's models are
 * fetched lazily, so the provider being there is enough).
 */
export function isAvailable(
  ref: string,
  local: readonly { id: string }[],
  providers: readonly { id: string }[],
): boolean {
  if (!ref) return false;
  const remote = splitRemoteRef(ref);
  if (remote) return providers.some((p) => p.id === remote.provider);
  return local.some((m) => m.id === ref);
}
