<p align="center">
  <img src="https://avatars.githubusercontent.com/u/316802122?s=400&v=4" alt="llmman" width="180">
</p>

<h1 align="center">llmman</h1>

<p align="center"><b>Run any coding agent on any model.</b></p>

<p align="center">
Claude Code, Codex, OpenCode and friends, pointed at a model
running on your own machine, or at any hosted provider, in one command.
No subscription, no rewiring, no vendor's idea of which model you should use.
</p>

```
llmman launch claude --model gemma4
```

That starts a local inference server, downloads a `llama.cpp` build matching your
GPU, loads the model, and execs Claude Code against it. Every token is generated
on your machine. No Anthropic API key, no subscription, nothing leaves the box.

<!-- TODO: 20s asciinema/GIF here: launch claude -> boots -> wifi off -> still coding -->

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

Three commands cover most of it:

```sh
llmman launch claude --model gemma4   # a coding agent on a local model
llmman run gemma4                     # just chat with a model
llmman serve                          # an Ollama/OpenAI/Anthropic-compatible endpoint
```

Run `llmman launch` with no arguments to see the supported agents and whether
each is installed. Want a hosted model instead of a local one? Every command
above takes `--provider`; see [Hosted providers](#hosted-providers).

## Commands

| Command | Description |
|---------|-------------|
| `serve`   | Start an inference server (Ollama / OpenAI / Anthropic APIs) |
| `launch`  | Launch an integration (Claude Code, OpenCode, …) |
| `run`     | Run a model interactively or with a one-shot prompt |
| `pull`    | Pull a model from a registry or HuggingFace |
| `list`    | List locally stored models, or a hosted provider's (`--provider`) models |
| `ps`      | List models currently loaded |
| `providers` | List the hosted providers `--provider` can route to |
| `stop`    | Stop (unload) a running model |
| `build`   | Package model files into a local OCI image |
| `push`    | Push a local image to a registry |
| `transfer` | Transfer an image directly from one location to another (e.g. HuggingFace to an OCI registry) |
| `cp`      | Copy a local image to a new reference |
| `rm`      | Remove a local image |
| `show`    | Show a local model's architecture, parameters, license, and template |
| `login`   | Log in to a container registry |
| `logout`  | Log out from a container registry |

See [docs/configuration.md](docs/configuration.md) for environment variables
and the model store layout.

## Models are OCI artifacts

Models are packaged as standard OCI artifacts and stored in any compatible
registry: Docker Hub, GHCR, quay, self-hosted. There is no curated library and
no gatekeeper: push a model anywhere you can push a container image, and anyone
can `llmman run` it straight from there.

### Pull a model

```
llmman pull gemma4
```

### Transfer a model between locations

Transfer an image directly from a source to a destination without storing
it locally first, e.g. HuggingFace straight to an OCI registry:

```
llmman transfer hf.co/unsloth/Qwen3.5-0.8B-GGUF docker.io/owner/model:latest
```

Any source `llmman pull` understands (an OCI registry, `hf://`, `ms://`, ...) can be paired with any OCI registry destination.

## Serve

Start the inference server. GGUF models are served by `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp), used from `PATH` if it's already there; otherwise `llmman` downloads and caches a prebuilt release matching your OS/arch/GPU automatically (see `--llama-cpp-version` to pin a specific release). Safetensors models are served by [`vllm`](https://github.com/vllm-project/vllm) (plain `vllm` is CPU-only on macOS, unless you separately install [vllm-metal](https://github.com/vllm-project/vllm-metal) for Metal GPU support), or, on Apple Silicon macOS, by [`mlx-lm`](https://github.com/ml-explore/mlx-lm)'s `mlx_lm.server` instead when it's on `PATH`: Metal-accelerated, with no vLLM dependency at all, and it supports more model families than vllm-metal does.

```
llmman serve
```

The server listens on `127.0.0.1:17434` by default, overridable via `LLMMAN_HOST`, and exposes:

| API | Endpoints |
|-----|-----------|
| Ollama | `/api/generate`, `/api/chat`, `/api/tags`, `/api/show`, `/api/pull`, `/api/ps`, `/api/delete` |
| OpenAI | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/v1/responses`, `/v1/responses/input_tokens` |
| Anthropic | `/v1/messages` |
| llmman | `/llmman/providers`, `/llmman/providers/{id}` |

`/llmman/...` is llmman's own API, not a compatibility surface: no
upstream API has a notion of a [models.dev](https://models.dev) provider
(see [Hosted providers](#hosted-providers)). `/llmman/providers` lists
the ones this daemon can route to, each with its API-key variable,
whether the daemon has that key, and how many models it serves;
`/llmman/providers/{id}` adds those models and what each costs in US
dollars per million tokens (absent, not zero, where models.dev publishes
no price). `llmman providers`, `list --provider`, `run --provider` and
`launch --provider` are all clients of it, so the catalog is fetched and
cached in one process: the one that forwards the request upstream.

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
as `message.tool_calls`), `images` (vision, base64, same as Ollama's own
wire format), and `format` (`"json"` or a JSON Schema object, for
constrained structured output).

An idle, unused model is automatically unloaded after `keep_alive`
(default 5 minutes, matching Ollama; set per-request, or daemon-wide via
`LLMMAN_KEEP_ALIVE`), and `llmman ps`/`/api/ps` reports each model's
`expires_at`.

Daemon-wide settings (bind address, context length, keep-alive, GPU backend
selection and the rest) are environment variables, set before `llmman serve`
starts. They're documented in [docs/configuration.md](docs/configuration.md).

## Launch an integration

Point an integration at a model in one step. `llmman launch` starts `serve` in the background if it isn't already running (preloading the requested model), then sets the right environment variables and execs the integration:

```
llmman launch claude --model gemma4
```

Run `llmman launch` with no arguments to list the supported integrations (Claude Code, OpenCode) and whether each is installed. Any extra arguments after `--` are forwarded to the integration's own CLI.

Short names work wherever a model reference is accepted.

### Hosted providers

`--provider` points llmman at a model it doesn't serve itself, from
`launch`, from `run`, and from `list`:

```sh
export OPENROUTER_API_KEY=...
llmman providers                                    # which providers, and is the key set
llmman list --provider openrouter                   # its models, and $/Mtok in and out
llmman run --provider openrouter qwen/qwen3-coder   # chat with one directly
llmman launch opencode --provider openrouter --model qwen/qwen3-coder
```

The provider list is fetched at runtime from
[models.dev](https://models.dev), the same catalog `opencode` resolves
its own providers from, so a newly added provider works without an
llmman release. It's cached for 24 hours, and a stale copy is used if the
fetch fails, so being offline means an out-of-date list rather than a
broken command.

All four commands ask the daemon over [`/llmman/providers`](#serve)
rather than fetching models.dev themselves, so the cache outlives any
single command and the key status reported is that of the environment
whose key actually gets spent.

Requests still go through `llmman serve`; `--provider` changes where the
daemon forwards them, not who the client talks to. So one endpoint and
one place integrations are configured, whether a model is local or
hosted, and both usable from the same session.

The API key is read from the variable models.dev names for that provider
and travels per request, never to disk. `hermes` and `talos` are the
exceptions: llmman configures them through a file on disk, so they can't
carry a key and `llmman serve` needs the variable in its own environment
instead. That fallback is only used for a daemon bound to loopback, and
never for a browser request from another site. It bounds the blast radius
rather than authenticating anyone, so on a shared machine prefer an integration
that sends its own key. `cline`, `kimi`, `copilot` and `openclaw` can't be
used with `--provider` at all: the first two pick their own model rather
than taking llmman's, `copilot` has no way to send a key, and `openclaw`
only takes a model during first-run onboarding.

`--provider` needs a local `llmman serve`. The daemon talks plain HTTP
and has no authentication, so neither `run` nor `launch` will send a real
key to a remote `LLMMAN_HOST`, and a daemon bound to anything but
loopback will not spend its own environment's key on behalf of a caller
that didn't present one. (`llmman providers` and `llmman list
--provider` read the catalog only, no key involved, and work against any
daemon.)

## Use with vLLM directly

`llmman serve` already spawns `vllm` itself as a backend for safetensors
models. The [`vllm-llmman`](https://pypi.org/project/vllm-llmman/) plugin
is the inverse: install it alongside `vllm` and `vllm serve
oci://<reference>` pulls a CNCF ModelPack image directly, instead of a
HuggingFace repo.

## MLX (Apple Silicon)

On Apple Silicon macOS, `llmman serve` uses
[`mlx_lm.server`](https://github.com/ml-explore/mlx-lm) instead of `vllm`
for safetensors models whenever it's on `PATH`, Metal-accelerated, with
no vLLM dependency at all (unlike getting the same acceleration out of
`vllm serve` itself via [vllm-metal](https://github.com/vllm-project/vllm-metal)).
Falls back to `vllm` otherwise. Doesn't support `/v1/embeddings`.

## Transport backends

The registry transport is a compiled-in Go shim. Two backends are available via Cargo feature flags.

### Docker (default)

Uses [`github.com/containerd/containerd`](https://github.com/containerd/containerd), the same OCI resolver used by Docker.

```
cargo build --release
```

### Podman

Uses [`github.com/podman-container-tools/container-libs`](https://github.com/podman-container-tools/container-libs), the same library Podman uses internally.

```
cargo build --release --no-default-features --features podman
```

