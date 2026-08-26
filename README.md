# llmman

A command-line tool for managing and serving LLM models using OCI registries.
Models are packaged as standard OCI artifacts and stored in any compatible registry (Docker Hub, GHCR, quay, self-hosted, etc.).
`llmman serve` exposes Ollama-, OpenAI-, and Anthropic-compatible HTTP APIs.

## Commands

| Command | Description |
|---------|-------------|
| `serve`   | Start an inference server (Ollama / OpenAI / Anthropic APIs) |
| `launch`  | Launch an integration (Claude Code, OpenCode, …) |
| `run`     | Run a model interactively or with a one-shot prompt |
| `pull`    | Pull a model from a registry or HuggingFace |
| `list`    | List locally stored models |
| `ps`      | List models currently loaded |
| `stop`    | Stop (unload) a running model |
| `build`   | Package model files into a local OCI image |
| `push`    | Push a local image to a registry |
| `transfer` | Transfer an image directly from one location to another (e.g. HuggingFace to an OCI registry) |
| `cp`      | Copy a local image to a new reference |
| `rm`      | Remove a local image |
| `show`    | Show a local model's architecture, parameters, license, and template |
| `login`   | Log in to a container registry |
| `logout`  | Log out from a container registry |

## Install

**Linux, macOS:**

```
curl -fsSL https://raw.githubusercontent.com/llmmanorg/llmman/main/install.sh | sh
```

**Windows (PowerShell):**

```
irm https://raw.githubusercontent.com/llmmanorg/llmman/main/install.ps1 | iex
```

## Quick start

### Pull a model

```
llmman pull gemma4
```

### Transfer a model between locations

Transfer an image directly from a source to a destination without storing
it locally first — e.g. HuggingFace straight to an OCI registry:

```
llmman transfer hf.co/unsloth/Qwen3.5-0.8B-GGUF docker.io/owner/model:latest
```

Any source `llmman pull` understands (an OCI registry, `hf://`, `ms://`, ...) can be paired with any OCI registry destination.

### Serve

Start the inference server. Uses `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp) if it's already on `PATH`; otherwise `llmman` downloads and caches a prebuilt release matching your OS/arch/GPU automatically (see `--llama-cpp-version` to pin a specific release).

```
llmman serve
```

The server listens on `127.0.0.1:17434` by default, overridable via `LLMMAN_HOST`, and exposes:

| API | Endpoints |
|-----|-----------|
| Ollama | `/api/generate`, `/api/chat`, `/api/tags`, `/api/show`, `/api/pull`, `/api/ps`, `/api/delete` |
| OpenAI | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/v1/responses`, `/v1/responses/input_tokens` |
| Anthropic | `/v1/messages` |

`/v1/responses` implements the OpenAI Responses API (the dialect [OpenAI
Codex](https://github.com/openai/codex) requires), including streaming SSE
and function-tool-call re-mapping. This is a plain pass-through to
`llama-server`'s own native `/v1/responses` support, so a recent enough
`llama-server` build is required for it to work.

Use it as an Ollama-compatible server:

```
OLLAMA_HOST=127.0.0.1:17434 ollama run unsloth/Qwen3.5-0.8B-GGUF
```

Or with any Ollama, Anthropic or OpenAI-compatible client.

Models are loaded on demand. Each model gets its own `llama-server` subprocess on a random loopback port; subsequent requests reuse the running process.

`/api/chat` also supports Ollama's `tools` (function calling, streamed back
as `message.tool_calls`), `images` (vision, base64 — same as Ollama's own
wire format), and `format` (`"json"` or a JSON Schema object, for
constrained structured output).

An idle, unused model is automatically unloaded after `keep_alive`
(default 5 minutes, matching Ollama — set per-request, or daemon-wide via
`LLMMAN_KEEP_ALIVE`), and `llmman ps`/`/api/ps` reports each model's
`expires_at`.

Daemon-wide settings, set before `llmman serve` starts:

| Variable | Effect |
|----------|--------|
| `LLMMAN_HOST` | `[host][:port]` `llmman serve` binds to. Every `llmman` client in the same environment connects to it too, rewriting a wildcard host to loopback first. Defaults to `127.0.0.1:17434`. |
| `LLMMAN_CONTEXT_LENGTH` | Context size (`--ctx-size`) for every model this daemon loads. Defaults to a VRAM-tiered value when unset. |
| `LLMMAN_FLASH_ATTENTION` | Flash Attention mode (`--flash-attn`): `on`, `off`, or `auto` (llama-server's own default). Also accepts `1`/`0`/`true`/`false`, matching Ollama's `OLLAMA_FLASH_ATTENTION`. |
| `LLMMAN_KV_CACHE_TYPE` | KV-cache quantization (`--cache-type-k`/`--cache-type-v`), e.g. `f16` (default), `q8_0`, `q4_0` — trades output quality for memory at long context lengths, matching Ollama's `OLLAMA_KV_CACHE_TYPE`. |
| `LLMMAN_MODELS` | Local store directory, overriding the default below — matching Ollama's `OLLAMA_MODELS`. `pull`/`push`/`run`/etc. go through the daemon and always use whichever store it was started with. |
| `LLMMAN_TMPDIR` | Staging directory for `llama-server` release downloads, overriding the default `tmp` subdirectory of the install root — matching Ollama's `OLLAMA_TMPDIR`. |

### Launch an integration

Point an integration at a model in one step. `llmman launch` starts `serve` in the background if it isn't already running (preloading the requested model), then sets the right environment variables and execs the integration:

```
llmman launch claude --model gemma4
```

Run `llmman launch` with no arguments to list the supported integrations (Claude Code, OpenCode) and whether each is installed. Any extra arguments after `--` are forwarded to the integration's own CLI.

Short names work wherever a model reference is accepted.

### Use with vLLM directly

`llmman serve` already spawns `vllm` itself as a backend for safetensors
models. The [`vllm-llmman`](https://pypi.org/project/vllm-llmman/) plugin
is the inverse: install it alongside `vllm` and `vllm serve
oci://<reference>` pulls a CNCF ModelPack image directly, instead of a
HuggingFace repo.

## Store location

Default locations:

| OS | Path |
|----|------|
| Linux, macOS | `~/.local/share/llmman/store` |
| Windows | `%LOCALAPPDATA%\llmman\store` |

Set `LLMMAN_MODELS` to change this (matching Ollama's `OLLAMA_MODELS`).
Commands that read or write the local store directly (`list`, `rm`,
`build`, `serve`) all honor it. Commands that go through the background
daemon instead (`pull`, `push`, `run`, `launch`, `ps`) always use whichever
store the daemon was started with — set `LLMMAN_MODELS` before
`llmman serve` to change it for all of them. `transfer`, `login`, and
`logout` never touch a local store at all.

The store uses [OCI Image Layout](https://github.com/opencontainers/image-spec/blob/main/image-layout.md), readable by `docker` and `podman`.

## Transport backends

The registry transport is a compiled-in Go shim. Two backends are available via Cargo feature flags.

### Docker (default)

Uses [`github.com/containerd/containerd`](https://github.com/containerd/containerd) — the same OCI resolver used by Docker.

```
cargo build --release
```

### Podman

Uses [`github.com/podman-container-tools/container-libs`](https://github.com/podman-container-tools/container-libs) — the same library Podman uses internally.

```
cargo build --release --no-default-features --features podman
```

