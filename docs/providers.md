# Hosted providers

`--provider` points llmman at a model it doesn't serve itself, from
`launch`, from `run`, and from `list`:

```sh
export OPENROUTER_API_KEY=...
llmman providers                                    # which providers, and is the key set
llmman list --provider openrouter                   # its models, and $/Mtok in and out
llmman run --provider openrouter qwen/qwen3-coder   # chat with one directly
llmman launch opencode --provider openrouter --model qwen/qwen3-coder
```

Requests still go through `llmman serve`; `--provider` changes where the
daemon forwards them, not who the client talks to, so local and hosted
models share one endpoint and one integration config.

## The catalog

The provider list comes from [models.dev](https://models.dev), the
catalog `opencode` uses, so a new provider needs no llmman release. It
is cached for 24 hours and a stale copy is used when the fetch fails.

All four commands read it from the daemon over
[`/llmman/providers`](api.md#llmmans-own-api), so the cache outlives any
one command and the key status reported is the daemon's.

## API keys

The API key comes from the variable models.dev names for that provider,
or, when that is unset, from `~/.config/llmman/llmman.conf`, keyed by
provider id:

```toml
[providers.openrouter]
api_key = "sk-or-..."
```

or, equivalently, `llmman config set providers.openrouter.api_key sk-or-...`.

Either way it travels per request; it is never written into an
integration's config.
A file carrying one must be `chmod 600` or its keys are ignored with a
warning; an `export` overrides it. See
[configuration.md](configuration.md#provider-api-keys).

`--provider` needs a local `llmman serve`. The daemon is plain HTTP with
no authentication, so `run` and `launch` never send a key to a remote
`LLMMAN_HOST`, and a daemon bound off loopback never spends its own key
for a caller that presented none. (`providers` and `list --provider`
read the catalog only and work against any daemon.)

## Integrations

`llmman launch` with no arguments lists these and whether each is
installed:

| Name | Integration | `--provider` |
|------|-------------|--------------|
| `claude` | Claude Code | yes |
| `opencode` | OpenCode | yes |
| `codex` | OpenAI Codex CLI | yes |
| `aider` | Aider | yes |
| `qwen` | Qwen Code | yes |
| `hermes` | Hermes Agent | yes, but the daemon holds the key (below) |
| `gemini` | Gemini CLI | no: llmman cannot confirm the key would come here rather than go to Google |
| `cline` | Cline | no: it picks its own model rather than taking llmman's |
| `kimi` | Kimi Code CLI | no: it picks its own model rather than taking llmman's |
| `copilot` | GitHub Copilot CLI (`gh`) | no: it has no way to send a key |
| `openclaw` | OpenClaw | no: it only takes a model during first-run onboarding |

Any extra arguments after `--` are forwarded to the integration's own CLI.

`hermes` is configured through a file on disk, so it can't carry a key
per request; `llmman serve` needs one of its own, spent only for a
loopback daemon and never for a cross-site browser request. On a shared
machine prefer an integration that sends its own key.
