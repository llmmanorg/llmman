# Commands

`llmman --help` and `llmman <command> --help` are the authoritative
reference; this is the one-line summary of each subcommand.

| Command | Description |
|---------|-------------|
| `serve`   | Start an inference server (Ollama / OpenAI / Anthropic APIs) |
| `launch`  | Launch an integration (Claude Code, OpenCode, …) |
| `run`     | Run a model interactively or with a one-shot prompt |
| `pull`    | Pull a model from a registry or HuggingFace |
| `search`  | Search for models on Docker Hub and Hugging Face (Docker Hub results first) |
| `list` (`ls`) | List locally stored models, or a hosted provider's (`--provider`) models |
| `ps`      | List models currently loaded |
| `log`     | Show the prompts `serve` has seen, newest first, like `git log` |
| `usage`   | Show the tokens and cost of what `serve` has served, per model, provider, client, day or route |
| `providers` | List the hosted providers `--provider` can route to |
| `stop`    | Stop (unload) a running model |
| `build`   | Package model files into a local OCI image |
| `push`    | Push a local image to a registry |
| `transfer` | Transfer an image directly from one location to another (e.g. HuggingFace to an OCI registry) |
| `cp`      | Copy a local image to a new reference |
| `rm`      | Remove a local image |
| `show`    | Show a local model's architecture, parameters, license, and template |
| `verify`  | Check a registry model's signatures against trusted public keys |
| `login`   | Log in to a container registry or HuggingFace |
| `logout`  | Log out from a container registry or HuggingFace |
| `config`  | Read and write `llmman.conf` settings (aliases, API keys, trust policy, aggregation peers) |
