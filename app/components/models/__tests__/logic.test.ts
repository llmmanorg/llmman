import type { PullProgress } from "@/api/client";
import type { ModelCard, SearchHit, Variant } from "@/api/search";
import { fixture } from "@/test/fakeFetch";

import {
  applyPullEvent,
  cardTags,
  chosenVariant,
  counted,
  filterVariants,
  fitOf,
  fitSentence,
  formatBytes,
  formatCount,
  holds,
  holdsRef,
  initialVariant,
  isPulled,
  isReference,
  localOf,
  localRepos,
  ownerOf,
  pullPercent,
  readmeMarkdown,
  referenceOpensCard,
  registryOf,
  relativeTime,
  repoOf,
  shortName,
  sizeGroups,
  sizePrefix,
  sortBySize,
  startPull,
  tagOf,
  variantLabel,
  type LocalModel,
} from "../logic";

const json = <T>(name: string): T => JSON.parse(fixture(name)) as T;
const ndjson = (name: string) =>
  fixture(name)
    .split("\n")
    .filter(Boolean)
    .flatMap((line) => {
      try {
        return [JSON.parse(line) as PullProgress];
      } catch {
        return []; // a truncated last line
      }
    });

const docker = json<ModelCard>("model-docker-qwen3.5.json");
const gguf = json<ModelCard>("model-hf-qwen3-coder-30b.json");
const memory = json<{ memory: number }>("node.json").memory;
const local: LocalModel[] = json<{ data: { id: string; status: { value: string } }[] }>(
  "v1-models.json",
).data.map((m) => ({ id: m.id, loaded: m.status.value === "loaded" }));

const variant = (tag: string, size: number, isDefault = false): Variant => ({
  name: `hf.co/o/r:${tag}`,
  tag,
  size,
  default: isDefault,
});

describe("names", () => {
  it("splits a ref into repo and tag", () => {
    expect(repoOf("docker.io/ai/qwen3.5:0.8b")).toBe("docker.io/ai/qwen3.5");
    expect(tagOf("docker.io/ai/qwen3.5:0.8b")).toBe("0.8b");
    // A port is not a tag.
    expect(repoOf("localhost:5000/team/model")).toBe("localhost:5000/team/model");
    expect(tagOf("localhost:5000/team/model")).toBe("localhost:5000/team/model");
  });

  it("knows the registries the daemon searches", () => {
    const names = json<{ models: SearchHit[] }>("search-popular.json").models.map((h) => h.name);
    expect(new Set(names.map(registryOf))).toEqual(new Set(["docker", "hf"]));
    expect(registryOf("ms://Qwen/Qwen3")).toBe("other");
    expect(ownerOf("hf.co/unsloth/Qwen3-GGUF")).toBe("unsloth");
    expect(shortName("hf.co/unsloth/Qwen3-GGUF")).toBe("Qwen3-GGUF");
  });

  it("tells a full reference from a search", () => {
    for (const ref of [
      "hf.co/unsloth/Qwen3-GGUF",
      "docker.io/ai/qwen3",
      "ms://Qwen/Qwen3",
      "registry.lan:5000/m",
    ]) {
      expect(isReference(ref)).toBe(true);
    }
    for (const query of ["qwen", "qwen3.5", "unsloth/Qwen3-GGUF", "ai/gemma3", "localhost:5000/m"]) {
      expect(isReference(query)).toBe(false);
    }
  });
});

describe("the local store", () => {
  it("finds what this computer holds of a repo", () => {
    expect(isPulled("docker.io/ai/qwen3.5", local)).toBe(true);
    expect(isPulled("DOCKER.IO/AI/QWEN3.5", local)).toBe(true);
    expect(isPulled("docker.io/ai/qwen3", local)).toBe(false);
    expect(localOf("docker.io/ai/qwen3.5", local).map((m) => m.id)).toEqual([
      "docker.io/ai/qwen3.5:0.8b-safetensors",
      "docker.io/ai/qwen3.5:0.8b",
    ]);
  });

  it("holds a variant exactly, or a default as :latest", () => {
    const tag = docker.variants.find((v) => v.tag === "0.8b")!;
    expect(holds(tag, local)).toBe(true);
    const latest = docker.variants.find((v) => v.default)!;
    expect(holds(latest, local)).toBe(false);
    expect(holds({ ...variant("Q4_K_M", 1, true) }, [{ id: "hf.co/o/r:latest" }])).toBe(true);
    expect(holds({ ...variant("Q4_K_M", 1, false) }, [{ id: "hf.co/o/r:latest" }])).toBe(false);
  });

  it("groups local models by repo", () => {
    const repos = localRepos(local);
    expect(repos.map((r) => [r.name, r.local.length])).toEqual([
      ["docker.io/ai/qwen3.5", 2],
      ["hf.co/bartowski/SmolLM2-135M-Instruct-GGUF", 1],
      ["hf.co/unsloth/SmolLM2-135M-Instruct-GGUF", 1],
    ]);
  });
});

describe("fit", () => {
  it("uses the web UI's thresholds", () => {
    expect(fitOf(60, 100)).toBe("ok");
    expect(fitOf(61, 100)).toBe("warn");
    expect(fitOf(90, 100)).toBe("warn");
    expect(fitOf(91, 100)).toBe("bad");
    expect(fitOf(0, 100)).toBeNull();
    expect(fitOf(50, 0)).toBeNull();
  });

  it("says it against this computer's memory", () => {
    const latest = docker.variants.find((v) => v.default)!;
    expect(fitSentence(latest, memory)).toBe(
      "Fits this computer: 21 GB of weights, 64 GB of model memory (context and runtime need more).",
    );
    expect(fitSentence(variant("x", 0), memory)).toBe("Size unknown until pulled.");
    expect(fitSentence(variant("x", 5), 0)).toBe("");
  });
});

describe("variants", () => {
  it("sorts by size with unknown sizes last", () => {
    const sorted = sortBySize([variant("b", 0), variant("c", 30), variant("a", 10)]);
    expect(sorted.map((v) => v.tag)).toEqual(["a", "c", "b"]);
    const all = sortBySize(docker.variants);
    expect(all).toHaveLength(128);
    expect(all.every((v, i) => i === 0 || all[i - 1]!.size <= v.size)).toBe(true);
  });

  it("reads a tag's size prefix", () => {
    expect(sizePrefix("35b-a3b-int4")).toBe("35b");
    expect(sizePrefix("0.8b")).toBe("0.8b");
    expect(sizePrefix("135M-q4")).toBe("135m");
    expect(sizePrefix("Q4_K_M")).toBeNull();
    expect(sizePrefix("latest")).toBeNull();
    expect(sizePrefix("bf16")).toBeNull();
  });

  it("offers size chips only when there are two sizes", () => {
    expect(sizeGroups(docker.variants)).toEqual(["0.8b", "2b", "4b", "9b", "27b", "35b", "122b", "397b"]);
    expect(sizeGroups(gguf.variants)).toEqual([]);
    expect(sizeGroups([variant("135m-a", 1), variant("1b-a", 1)])).toEqual(["135m", "1b"]);
  });

  it("filters by chip and text", () => {
    const tags = (g: string | null, t: string) => filterVariants(docker.variants, g, t).map((v) => v.tag);
    expect(tags("27b", "q4").sort()).toEqual(["27b-iq4_XS", "27b-mtp-q4_K_M", "27b-q4_K_M", "27b-q4_K_XL"]);
    expect(tags(null, "  Q8_0 ")).toContain("27b-q8_0");
    expect(tags(null, "")).toHaveLength(128);
  });

  describe("the variant a card opens on", () => {
    // No default, sizes out of order: 20 and 50 fit 100 comfortably, 70 is tight, 80 too.
    const undecided = [variant("big", 80), variant("small", 20), variant("mid", 50), variant("tight", 70)];

    it("is the one picked earlier", () => {
      expect(initialVariant(gguf.variants, memory, gguf.variants[0]!.name)?.tag).toBe(gguf.variants[0]!.tag);
      expect(initialVariant(undecided, 100, "hf.co/o/r:big")?.tag).toBe("big");
    });

    it("else the registry's default", () => {
      expect(initialVariant(gguf.variants, memory)?.tag).toBe("Q4_K_M");
      expect(initialVariant(docker.variants, memory)?.tag).toBe("latest");
      expect(initialVariant(gguf.variants, memory, "a pick no longer listed")?.tag).toBe("Q4_K_M");
    });

    it("else the largest that fits comfortably", () => {
      expect(initialVariant(undecided, 100)?.tag).toBe("mid");
    });

    it("else the smallest", () => {
      expect(initialVariant(undecided, 10)?.tag).toBe("small"); // nothing fits
      expect(initialVariant(undecided, 0)?.tag).toBe("small"); // memory unknown
      expect(initialVariant([variant("unknown", 0), variant("small", 20)], 100)?.tag).toBe("small");
      expect(initialVariant([], 100)).toBeUndefined();
    });
  });
});

describe("references typed in the search bar", () => {
  it("opens the card for registries the daemon searches, and asks for the rest", () => {
    expect(referenceOpensCard("hf.co/unsloth/Qwen3-GGUF:Q4_K_M")).toBe(true);
    expect(referenceOpensCard("docker.io/ai/qwen3:0.6b")).toBe(true);
    for (const ref of ["ms://Qwen/Qwen3", "s3://bucket/model", "ghcr.io/org/model:1", "ngc://nvidia/x"]) {
      expect(referenceOpensCard(ref)).toBe(false);
    }
  });

  it("knows a reference already pulled, by tag or by repo", () => {
    expect(holdsRef("hf.co/unsloth/SmolLM2-135M-Instruct-GGUF:Q4_K_M", local)).toBe(true);
    expect(holdsRef("HF.CO/UNSLOTH/SmolLM2-135M-Instruct-GGUF:q4_k_m", local)).toBe(true);
    expect(holdsRef("hf.co/unsloth/SmolLM2-135M-Instruct-GGUF:Q8_0", local)).toBe(false);
    expect(holdsRef("docker.io/ai/qwen3.5", local)).toBe(true); // tagless: any tag
    expect(holdsRef("docker.io/ai/qwen3", local)).toBe(false);
  });

  it("shows a typed tag on the card, listed or not", () => {
    const listed = chosenVariant(
      gguf.variants,
      memory,
      "HF.CO/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:q8_0",
    );
    expect(listed?.tag).toBe("Q8_0");
    expect(listed?.size).toBeGreaterThan(0);
    const unlisted = chosenVariant(
      gguf.variants,
      memory,
      "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q9_X",
    );
    expect(unlisted).toEqual({
      name: "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q9_X",
      tag: "Q9_X",
      size: 0,
      default: false,
    });
    expect(fitSentence(unlisted!, memory)).toBe("Size unknown until pulled.");
    // A tagless pick is the repo itself: the usual choice.
    expect(chosenVariant(gguf.variants, memory, "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF")?.tag).toBe(
      "Q4_K_M",
    );
    expect(chosenVariant(gguf.variants, memory, null)?.tag).toBe("Q4_K_M");
  });
});

describe("screen reader labels", () => {
  it("counts with the right noun", () => {
    expect(counted(1, "pull")).toBe("1 pull");
    expect(counted(0, "like")).toBe("0 likes");
    expect(counted(1415492, "pull")).toBe("1.4M pulls");
  });

  it("says a version's size and fit in words", () => {
    const latest = docker.variants.find((v) => v.default)!;
    expect(variantLabel(latest, local, memory)).toBe("latest, default, 21 GB, Fits this computer");
    const held = docker.variants.find((v) => v.tag === "0.8b")!;
    expect(variantLabel(held, local, memory)).toBe("0.8b, 703 MB, Fits this computer, on this computer");
    expect(variantLabel(variant("x", 0), [], memory)).toBe("x, size unknown");
    expect(variantLabel(variant("big", 95), [], 100)).toBe("big, 95 B, Too big for this computer");
  });
});

describe("card text", () => {
  it("drops the long registry tags and keeps 14", () => {
    expect(cardTags(["gguf", "base_model:x", "dataset:y", "license:mit"])).toEqual(["gguf", "license:mit"]);
    expect(cardTags(Array.from({ length: 20 }, (_, i) => `t${i}`))).toHaveLength(14);
    expect(cardTags(null)).toEqual([]);
  });

  it("reduces a README's HTML to Markdown", () => {
    const readme = [
      "---",
      "license: mit",
      "---",
      "<!-- hidden -->",
      '<div align="center"><img src="https://x/badge.svg"></div>',
      '<h2 class="t">Hello <b>there</b></h2>',
      'See <a href="docs/USAGE.md">the <i>usage</i></a> and [abs](https://e.com) and [rel](./a.md).',
      'Line<br/>break &amp; &lt;tag&gt; <a href="x"><img src="y"></a>',
      "```html",
      "<b>kept</b> &amp;",
      "```",
    ].join("\n");
    expect(readmeMarkdown(readme, "https://huggingface.co/o/r")).toBe(
      [
        "## Hello there",
        "",
        "See [the usage](https://huggingface.co/o/r/blob/main/docs/USAGE.md) and [abs](https://e.com) and [rel](https://huggingface.co/o/r/blob/main/a.md).",
        "Line",
        "break & <tag>",
        "```html",
        "<b>kept</b> &amp;",
        "```",
      ].join("\n"),
    );
    expect(readmeMarkdown("[rel](a.md)", "")).toBe("[rel](a.md)");
  });

  it("handles the fixtures' READMEs", () => {
    const md = readmeMarkdown(docker.readme ?? "", "");
    expect(md.length).toBeGreaterThan(1000);
    expect(md).not.toMatch(/<img/i);
  });

  it("drops front matter that follows blank lines", () => {
    const card = json<ModelCard>("model-hf-quantfactory-smollm2.json");
    expect(card.readme?.startsWith("\n---\n")).toBe(true);
    const md = readmeMarkdown(card.readme ?? "", card.page);
    expect(md).not.toMatch(/library_name:|license: apache/);
    expect(md.startsWith("[![QuantFactory Banner](")).toBe(true);
    // A rule later in a README is not front matter.
    expect(readmeMarkdown("Intro\n\n---\n\nmore\n\n---\n", "")).toBe("Intro\n\n---\n\nmore\n\n---");
  });

  describe("indentation left from HTML", () => {
    const qwen3 = json<ModelCard>("model-docker-qwen3.json");
    /** Lines outside fenced code. */
    const prose = (md: string) =>
      md.split(/^```[\s\S]*?^```.*$/m).flatMap((p, i) => (i % 2 ? [] : p.split("\n")));

    it("starts the qwen3 and qwen3.5 cards' paragraphs at the margin", () => {
      const md3 = readmeMarkdown(qwen3.readme ?? "", "");
      expect(md3).toMatch(/^See \[our collection\]\(https:\/\/huggingface\.co\/collections\/unsloth\//m);
      expect(md3).toMatch(/^Learn to run Qwen3 correctly - \[Read our Guide\]\(/m);
      expect(md3).toMatch(
        /^\[Unsloth Dynamic 2\.0\]\(https:\/\/docs\.unsloth\.ai\/basics\/unsloth-dynamic-v2\.0-gguf\) achieves/m,
      );
      const md35 = readmeMarkdown(docker.readme ?? "", "");
      expect(md35.startsWith("# To run Qwen3.5 locally - Read our Guide!\n\n[Unsloth Dynamic 2.0](")).toBe(
        true,
      );
      for (const md of [md3, md35]) {
        const lines = prose(md);
        // No blank-looking lines made of spaces, and only list items indented.
        expect(lines.filter((l) => /^[ \t]+$/.test(l))).toEqual([]);
        expect(lines.filter((l) => /^\s/.test(l) && !/^\s+([-*+]|\d+[.)])\s/.test(l))).toEqual([]);
      }
    });

    it("keeps Markdown's own indentation and fenced code as written", () => {
      const readme = [
        "- a list",
        "  - nested under it",
        "",
        "```html",
        "    <div>indented &amp; kept</div>",
        "```",
        "",
        "<div>",
        "    <p>",
        "      text in HTML",
        "    </p>",
        "</div>",
      ].join("\n");
      expect(readmeMarkdown(readme, "")).toBe(
        [
          "- a list",
          "  - nested under it",
          "",
          "```html",
          "    <div>indented &amp; kept</div>",
          "```",
          "",
          "text in HTML",
        ].join("\n"),
      );
    });
  });
});

describe("pull progress", () => {
  it("follows a recorded pull to success", () => {
    let state = startPull("hf.co/QuantFactory/SmolLM2-135M-Instruct-GGUF:Q2_K");
    const seen: (number | null)[] = [];
    for (const ev of ndjson("pull-hf-q2k.ndjson")) {
      state = applyPullEvent(state, ev);
      seen.push(pullPercent(state));
    }
    // "pulling manifest" and "pulling" carry no counts; then one total for the whole model.
    expect(seen.slice(0, 4)).toEqual([null, null, null, null]);
    const known = seen.filter((p): p is number => p !== null);
    expect(known.every((p, i) => i === 0 || known[i - 1]! <= p)).toBe(true);
    expect(known.at(-1)).toBe(100);
    expect([state.status, state.total, state.completed]).toEqual(["success", 88201792, 88201792]);
    expect(state.notices).toEqual([]);
  });

  it("keeps the daemon's notices", () => {
    const notice = "warning: cannot verify hf.co/o/r:Q2_K: no signature found";
    let state = applyPullEvent(startPull("r"), { status: "pulling", total: 10, completed: 10 });
    state = applyPullEvent(state, { notice });
    state = applyPullEvent(state, { status: "success" });
    expect(state.notices).toEqual([notice]);
    expect(state.status).toBe("success");
  });

  it("clamps completed to total and keeps counts across status-only lines", () => {
    let state = applyPullEvent(startPull("r"), { status: "pulling", total: 100, completed: 150 });
    expect(state.completed).toBe(100);
    state = applyPullEvent(state, { status: "verifying" });
    expect([state.status, state.total, state.completed]).toEqual(["verifying", 100, 100]);
  });

  it("leaves a stream cut mid-line unfinished", () => {
    const text = fixture("pull-hf-q2k.ndjson");
    const cut = text.slice(0, text.indexOf('"status":"success"') - 10);
    const state = cut
      .split("\n")
      .flatMap((line) => {
        try {
          return [JSON.parse(line) as PullProgress];
        } catch {
          return [];
        }
      })
      .reduce(applyPullEvent, startPull("r"));
    expect(pullPercent(state)).toBe(100);
    expect(state.status).not.toBe("success");
  });
});

describe("formatting", () => {
  it("formats bytes like the web UI", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(88202080)).toBe("84 MB");
    expect(formatBytes(1769978895)).toBe("1.6 GB");
    expect(formatBytes(memory)).toBe("64 GB");
  });

  // Expected strings are what the web UI's Intl formatters print for the same values.
  const now = Date.parse("2026-10-07T12:00:00Z");
  const popular = json<{ models: SearchHit[] }>("search-popular.json").models;
  const row = (name: string) => popular.find((h) => h.name === name)!;

  it("formats the fixtures' counts as the web UI does", () => {
    expect([docker.pulls, docker.likes].map((n) => formatCount(n!))).toEqual(["639.5K", "10"]);
    expect([gguf.pulls, gguf.likes].map((n) => formatCount(n!))).toEqual(["6.9M", "1.1K"]);
    expect(formatCount(row("docker.io/ai/gemma4").pulls!)).toBe("1.4M");
    expect(formatCount(row("docker.io/ai/gemma3").pulls!)).toBe("1M");
    expect(formatCount(row("docker.io/ai/qwen3").pulls!)).toBe("842.1K");
    expect(formatCount(row("hf.co/prism-ml/Ternary-Bonsai-2-27B-gguf").likes!)).toBe("2.5K");
  });

  it("formats counts at the unit edges as the web UI does", () => {
    expect(formatCount(0)).toBe("0");
    expect(formatCount(999)).toBe("999");
    expect(formatCount(1000)).toBe("1K");
    expect(formatCount(1049)).toBe("1K");
    expect(formatCount(1050)).toBe("1.1K");
    expect(formatCount(999400)).toBe("999.4K");
    expect(formatCount(999960)).toBe("1M");
    expect(formatCount(2.5e12)).toBe("2.5T");
  });

  it("dates the fixtures as the web UI does", () => {
    expect(relativeTime(docker.updated!, now)).toBe("last month");
    expect(relativeTime(gguf.updated!, now)).toBe("8 months ago");
    expect(relativeTime(row("docker.io/ai/gemma3").updated!, now)).toBe("2 months ago");
    expect(relativeTime(row("hf.co/prism-ml/Ternary-Bonsai-2-27B-gguf").updated!, now)).toBe("12 days ago");
  });

  it("says how long ago at the unit edges", () => {
    const ago = (days: number) => relativeTime(new Date(now - days * 86_400_000).toISOString(), now);
    expect(ago(0)).toBe("today");
    expect(ago(1)).toBe("yesterday");
    expect(ago(4)).toBe("4 days ago");
    expect(ago(-1)).toBe("tomorrow");
    expect(ago(35)).toBe("last month");
    expect(ago(100)).toBe("3 months ago");
    expect(ago(400)).toBe("last year");
    expect(ago(1000)).toBe("3 years ago");
    expect(relativeTime("not a date", now)).toBe("");
  });
});
