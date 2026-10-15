import { effortLabel, hostedName, isAvailable, localName, modelName, repoName } from "../modelName";

it.each([
  ["docker.io/ai/qwen3.5:0.8b", "Qwen3.5 0.8B"],
  ["docker.io/ai/qwen3.5:0.8b-safetensors", "Qwen3.5 0.8B · safetensors"],
  ["docker.io/ai/qwen3.5:4b-mlx", "Qwen3.5 4B · MLX"],
  ["docker.io/ai/qwen3.5:9b-mtp-q8_0", "Qwen3.5 9B MTP · Q8_0"],
  ["docker.io/ai/smollm2:135M-Q4_K_M", "SmolLM2 135M · Q4_K_M"],
  ["hf.co/unsloth/SmolLM2-135M-Instruct-GGUF:Q4_K_M", "SmolLM2 135M · Q4_K_M"],
  ["hf.co/lmstudio-community/Qwen3.5-9B-MLX-4bit:latest", "Qwen3.5 9B · MLX 4-bit"],
  ["hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M", "Qwen3 30B A3B · Q4_K_M"],
  ["docker.io/ai/gpt-oss:20b", "GPT OSS 20B"],
  ["docker.io/ai/gemma3", "Gemma3"],
  ["plain", "Plain"],
])("local %s", (ref, name) => {
  expect(localName(ref)).toBe(name);
});

it.each([
  ["claude-opus-4-5-20251101", "Claude Opus 4.5"],
  ["claude-sonnet-4-20250514", "Claude Sonnet 4"],
  ["gpt-4o", "GPT-4o"],
  ["gpt-4o-mini", "GPT-4o Mini"],
  ["gemini-2.5-pro", "Gemini 2.5 Pro"],
  ["o3-mini", "o3 Mini"],
  ["meta-llama/llama-3.1-8b:free", "Llama 3.1 8B · free"],
])("hosted %s", (id, name) => {
  expect(hostedName(id)).toBe(name);
});

test("the publisher is shown only to tell twins apart", () => {
  const local = [
    "hf.co/unsloth/SmolLM2-135M-Instruct-GGUF:Q4_K_M",
    "hf.co/bartowski/SmolLM2-135M-Instruct-GGUF:Q4_K_M",
    "docker.io/ai/qwen3.5:0.8b",
  ];
  expect(modelName(local[0]!, { local })).toBe("SmolLM2 135M · Q4_K_M · unsloth");
  expect(modelName(local[1]!, { local })).toBe("SmolLM2 135M · Q4_K_M · bartowski");
  expect(modelName(local[2]!, { local })).toBe("Qwen3.5 0.8B");
});

test("a hosted model prefers the catalog's name", () => {
  const providers = [
    {
      id: "anthropic",
      name: "Anthropic",
      models: [{ id: "claude-x-1", name: "Claude X" }, { id: "claude-y-2" }],
    },
  ];
  expect(modelName("llmman.provider/anthropic/claude-x-1", { providers })).toBe("Claude X");
  expect(modelName("llmman.provider/anthropic/claude-y-2", { providers })).toBe("Claude Y 2");
  expect(modelName("", {})).toBe("Choose a model");
});

test("a repository reads without its format", () => {
  expect(repoName("hf.co/unsloth/SmolLM2-135M-Instruct-GGUF")).toBe("SmolLM2 135M");
  expect(repoName("docker.io/ai/qwen3-coder")).toBe("Qwen3 Coder");
});

test("effort levels read in sentence case", () => {
  expect(["", "low", "medium", "high", "xhigh", "max"].map((l) => effortLabel(l || undefined))).toEqual([
    "Default",
    "Low",
    "Medium",
    "High",
    "Extra high",
    "Max",
  ]);
});

describe("isAvailable", () => {
  const local = [{ id: "docker.io/ai/qwen3.5:0.8b" }];
  const providers = [{ id: "anthropic" }];
  it.each([
    ["docker.io/ai/qwen3.5:0.8b", true],
    ["docker.io/ai/gone:1b", false],
    ["llmman.provider/anthropic/claude-x", true],
    ["llmman.provider/openai/gpt-4o", false],
    ["", false],
  ])("%s", (ref, ok) => {
    expect(isAvailable(ref, local, providers)).toBe(ok);
  });
});
