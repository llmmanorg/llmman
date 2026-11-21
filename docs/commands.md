# Commands

`llmman --help` and `llmman <command> --help` are the authoritative
reference; this is the one-line summary of each subcommand.

| Command | Description |
|---------|-------------|
| `serve`   | Start an inference server (Ollama / OpenAI / Anthropic APIs) |
| `launch`  | Launch an integration (Claude Code, OpenCode, …); `--optimizer tk` compacts the output of the agent's shell commands `--optimizer cvmn` asks for terse replies and `--optimizer hdrm` compresses what the agent sends the model (see [optimizers.md](optimizers.md)) |
| `run`     | Run a model interactively or with a one-shot prompt |
| `pull`    | Pull a model from a registry or HuggingFace |
| `search`  | Search for models on Docker Hub and Hugging Face (Docker Hub results first), with a `FIT` column: how much of this machine's model memory each default download takes |
| `list` (`ls`) | List locally stored models, or a hosted provider's (`--provider`) models |
| `ps`      | List models currently loaded |
| `log`     | Show the prompts `serve` has seen, newest first, like `git log` |
| `usage`   | Show the tokens and cost of what `serve` has served, per model, provider, client, day or route |
| `providers` | List the hosted providers `--provider` can route to |
| `stop`    | Stop (unload) a running model |
| `build`   | Package model files into a local OCI image; `--raw` stores each file without a tar wrapper, so the layer digest is the file's SHA-256 |
| `push`    | Push a local image to a registry |
| `transfer` | Transfer an image directly from one location to another (e.g. HuggingFace to an OCI registry) |
| `cp`      | Copy a local image to a new reference |
| `rm`      | Remove a local image |
| `show`    | Show a local model's architecture, parameters, license, and template |
| `verify`  | Check a registry model's signatures against trusted public keys |
| `login`   | Log in to a container registry, HuggingFace, or (`--provider`) a [hosted provider](providers.md#logging-in); `--param KEY=VALUE` passes a method its settings |
| `logout`  | Log out from a container registry, HuggingFace, or (`--provider`) a hosted provider |
| `config`  | Read and write `llmman.conf` settings (aliases, API keys, trust policy, aggregation peers) |

`resolve` is not listed above because `--help` hides it: it is the hook other
programs call. `llmman resolve REFERENCE` pulls the reference if needed (under
the `[verify]` policy, as `pull`) and prints one line of JSON with the path to
serve (`path`, `format`, and for a GGUF with a projector `mmproj`); the
`vllm-llmman` plugin uses it. `llmman resolve --dir REFERENCE` prints
`{"reference", "dir", "digest", "format"}` instead, where `dir` is the whole
model directory, every layer of the artifact at its packaged path: what
[`llmman-csi`](kubernetes.md) mounts into pods. Both take `--cache DIR` and
`--no-pull`.
