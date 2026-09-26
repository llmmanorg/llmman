# Hosted providers

`--provider` points llmman at a model it doesn't serve itself, from
`launch`, from `run`, and from `list`:

```sh
export OPENROUTER_API_KEY=...
llmman providers                                    # which providers, and is the key set
llmman list --provider openrouter                   # its models, and $/Mtok in and out
llmman run --provider openrouter qwen/qwen3-coder   # chat with one directly
llmman launch opencode --provider openrouter --model qwen/qwen3-coder
llmman usage --since yesterday                      # what that session cost, per model
```

Requests still go through `llmman serve`; `--provider` changes where the
daemon forwards them, not who the client talks to, so local and hosted
models share one endpoint and one integration config.

The daemon also records each reply's tokens, priced at the catalog's
rates (cache, reasoning and long-context included); `llmman usage` sums them. See
`LLMMAN_NOUSAGE` in [configuration.md](configuration.md).

## The catalog

The provider list comes from [models.dev](https://models.dev), the
catalog `opencode` uses, so a new provider needs no llmman release. It
is cached for 24 hours and a stale copy is used when the fetch fails.

All four commands read it from the daemon over
[`/llmman/providers`](api.md#llmmans-own-api), so the cache outlives any
one command and the key status reported is the daemon's.

## API keys

The API key comes from the variable models.dev names for that provider,
or — when that is unset — from `~/.config/llmman/llmman.conf`, keyed by
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

`--provider` needs a local `llmman serve`, or one reached over TLS
(`LLMMAN_HOST=https://...`): `run` and `launch` never send a key over
plain http to a remote `LLMMAN_HOST`. A daemon bound off loopback spends
its own key only for a caller that authenticated with the daemon's API
key ([api.md](api.md#authentication)) — and since that key takes the
`Authorization` header, `launch`'s integrations then rely on the
daemon's provider key rather than carrying one; `run --provider` sends
its own as `x-api-key`. (`providers` and `list --provider` read the
catalog only and work against any daemon.)

## Your own endpoints

A provider models.dev has never heard of — vLLM or llama-server on a
box down the hall, LM Studio on a laptop, a proxy in front of OpenAI —
is defined in `llmman.conf` by giving a `[providers.<id>]` a
`base_url`:

```toml
[providers.gpubox]
base_url = "http://gpubox:8000/v1"
```

```console
$ llmman config set providers.gpubox.base_url http://gpubox:8000/v1
$ llmman providers | grep gpubox
gpubox      gpubox    -          none needed    -
$ llmman list --provider gpubox                # asks the box's own /models
$ llmman launch opencode --provider gpubox --model qwen3-coder
```

From there it is a provider like any other: `run`, `list`, `launch` and
`--overflow-provider` all take the id, and requests still go through
`llmman serve`, which forwards to the URL.

| Field | Meaning |
|-------|---------|
| `base_url` | Required to define one. An absolute `http://` or `https://` URL the wire's route is appended to — `/chat/completions` for `openai`, `/messages` for `anthropic` — so it usually ends in `/v1`. |
| `wire` | `openai` (default) or `anthropic`. See [Wire formats](#wire-formats). |
| `api_key` | Sent as the wire's credential when set. Most local servers take none, and none is sent. |
| `api_key_env` | An environment variable to read the key from instead; it wins over `api_key`, as for a catalog provider. `""` clears one an earlier file named. |
| `name` | Display name for listings. The id when absent. |

The rules the catalog is filtered by do not apply. They vet a list
fetched from the network at runtime; a URL you wrote into your own
owner-only file needs no vetting beyond parsing. So a defined provider
may be plain `http` — that is the point on a LAN — and may take no key.
If it has a key *and* a plain-http URL, whichever process is about to
send the key warns that it crosses the network in cleartext, and sends
it.

A defined provider with a catalog id (`[providers.openai]` with a
`base_url`) replaces the catalog entry, which is how a proxy or regional
endpoint gets used without renaming the provider in every integration's
config. The catalog's model list goes with it.

Models are not listed in the file. For an `openai`-wire provider,
`list --provider <id>` and the `--model` check ask the endpoint's own
`GET /models` and take what it says; a box that is down or lacks the
route lists nothing, and the request still goes to it. An `anthropic`
provider has no such route and lists nothing. `llmman providers` shows
`-` in the models column for the same reason.

`llmman serve` reads `llmman.conf` once, at startup, so a provider added
while it runs needs a restart to appear. A machine that cannot reach
models.dev at all still has its defined providers.

The `base_url` is reported by the daemon's API and printed in warnings,
so it may not carry a `user:password@`; `api_key` is where a credential
goes. The id may not contain `/`.

## Hybrid model pairs

`--overflow-provider` and `--overflow-model` pair the local `--model`
with a hosted one under a single name, and `llmman serve` picks a side
per request:

```sh
llmman launch opencode --model gemma4 --overflow-provider anthropic --overflow-model claude-sonnet-5
llmman run gemma4 --overflow-provider anthropic --overflow-model claude-sonnet-5
```

Both halves travel as one reference,
`llmman.hybrid/gemma4,anthropic/claude-sonnet-5`, in the same `"model"`
field an ordinary name uses, so a pair works from any client on every
inference endpoint (`/api/show`, `/api/pull` and the other store
operations take a plain model name). The local half is resolved and pulled as `--model` always
is; the hosted half is validated and authenticates exactly as a bare
`--provider` model does, so the same integration rules apply. The two
cannot be combined with `--provider`, since the local half has to be
local.

Which side serves a request:

1. **`x-llmman-route: local` or `cloud`** on the request wins. Any other
   value, or the header given twice, is a `400`, never a guess; a blank
   value counts as absent.
2. **Otherwise, size.** A request larger than the local context can hold
   goes to the provider. The budget is four bytes per token of the
   daemon's context size (`LLMMAN_CONTEXT_LENGTH`); `LLMMAN_HYBRID_LOCAL_BYTES`
   sets it directly, `0` turns the rule off. A request that declares no
   `Content-Length` stays local.
3. **Otherwise, local.**

Local is the default because the two mistakes are not equal: a worse
local answer is recoverable, a request sent to someone else's servers is
not. Every request logs which way it went and why; the log, not the
response's `model` field, is the record of the side.

The byte budget is an estimate. If a chat, completion, Responses or
Messages request it kept local is then refused by the local backend as
larger than its context, the daemon sends it to the hosted half instead,
before anything has reached the client. Without that an agent would see the local model's context error,
compact its history and stay local. A `local` pin is never overridden
this way, and `LLMMAN_HYBRID_LOCAL_BYTES=0` disables only the size rule,
not this retry.

`/v1/audio/transcriptions` cannot forward to a provider, so a pair takes
its local half there whatever the body size. An unload (`keep_alive: 0`)
or a startup preload of a pair acts on its local half, the only one that
loads.

## Integrations

`llmman launch` with no arguments lists these and whether each is
installed:

| Name | Integration | `--provider` |
|------|-------------|--------------|
| `claude` | Claude Code | yes |
| `opencode` | OpenCode | yes |
| `codex` | OpenAI Codex CLI | yes (below) |
| `pi` | Pi coding agent | yes |
| `aider` | Aider | yes |
| `qwen` | Qwen Code | yes |
| `dsh` | DeepSeek Harness | yes |
| `goose` | Block goose | yes |
| `grok` | Grok Build (requires `--model`) | no: its fetched model catalog cannot represent llmman's hosted-provider routing reference |
| `docker-agent` | Docker Agent (requires `--model`) | yes |
| `hermes` | Hermes Agent | yes, but the daemon holds the key (below) |
| `agy` | Antigravity CLI (requires `--model`) | yes |
| `gemini` | Gemini CLI | no: llmman cannot confirm the key would come here rather than go to Google |
| `cline` | Cline (requires `--model`) | yes |
| `kimi` | Kimi Code CLI | no: it picks its own model rather than taking llmman's |
| `copilot` | GitHub Copilot CLI (`gh`) | no: it has no way to send a key |
| `openclaw` | OpenClaw | no: it only takes a model during first-run onboarding |

Any extra arguments after `--` are forwarded to the integration's own CLI.

`hermes` is configured through a file on disk, so it can't carry a key
per request; `llmman serve` needs one of its own, spent only for a
loopback daemon and never for a cross-site browser request. On a shared
machine prefer an integration that sends its own key.

`codex` speaks only OpenAI's Responses API, which most providers lack
(`mistral` 404s it, `opencode` 500s it for non-OpenAI models). The
daemon tries the provider first and, on a 404/405/501 or 5xx, translates
the request to a chat completion and the reply back, tool calls included.
Providers that have the API (`openai`, `groq`, `openrouter`) are used
natively; any other 4xx is relayed as-is.

### Thinking

Thinking depth is set from inside the integration and reaches the model
as `reasoning_effort`: llama-server reads it natively (`none` turns
thinking off; a level goes to the chat template), a provider gets it in
its own form (see [wire formats](#wire-formats)). Nothing selected leaves
the model's default.

- `opencode`: variants read off the model's chat template (what `llmman
  show` lists as `thinking`), cycled with `variant_cycle` (ctrl+t) or
  `/variants`: `none`, each `reasoning_effort` level the template takes
  (Qwen3.8: `low`, `medium`, `xhigh`), or `thinking` for a template with
  only an `enable_thinking` switch (Gemma 4, Qwen3.5). A provider's model
  gets the levels models.dev's `reasoning_options` list, as opencode
  does (Claude Opus 5.5: `low` to `max`; an Anthropic `budget_tokens`
  model: `high`, `max`); none where models.dev says it does not reason;
  else `none`, `low`, `medium`, `high`.
- `claude`: Claude Code's `/effort <low|medium|high|xhigh|max>`, sent as
  spelled; a level the template rejects is a 400.
- `codex`: `model_reasoning_effort`, e.g. `-- -c
  model_reasoning_effort=high`; its `/model` picker lists only OpenAI's
  catalog.

## Wire formats

Each provider is spoken to in one of two wire formats, reported as
`wire` by `/llmman/providers`:

- `openai`: OpenAI Chat Completions with `Authorization: Bearer <key>`.
  Every `@ai-sdk/openai-compatible` provider, plus the hand-checked
  endpoints for `openai`, `google`, `groq`, `mistral` and the rest, and
  the default for a provider [defined in `llmman.conf`](#your-own-endpoints).
- `anthropic`: the Anthropic Messages API with `x-api-key: <key>`.
  `anthropic` itself. Other Messages-compatible endpoints are not
  offered from the catalog, since their auth scheme varies and has not
  been checked; one you know takes `x-api-key` can be defined with
  `wire = "anthropic"`.

A provider that takes no key gets no credential header at all, not an
empty one.

Anthropic is never reached through its OpenAI-compatibility shim. What a
request becomes on the way to a `wire: anthropic` provider depends on
the surface it arrived on:

| Arrived on | Sent as |
|------------|---------|
| `/v1/messages` (Claude Code) | The same request, relayed intact: cache breakpoints, thinking, tools and `anthropic-beta` headers included. Only `model` is rewritten. |
| `/v1/chat/completions` (OpenCode, Aider, Qwen Code, Hermes), `/api/chat`, `/api/generate` | A Messages request, and the reply back as chat-completion chunks: system turns to `system`, tool calls to `tool_use`/`tool_result`, `reasoning_effort` to thinking (see below), thinking back as `reasoning_content`. |
| `/v1/responses` (Codex) | The Responses bridge above, then the same translation. The provider is not probed for `/v1/responses`. |

`max_tokens` is required by the Messages API; a translated request
without one gets the model's `limit.output` from the catalog, or 4096
(a relayed `/v1/messages` request is the client's own to complete). `/v1/completions`,
`/v1/embeddings`, `/api/embed`, `/api/embeddings` and
`/v1/responses/input_tokens` have no Messages equivalent and are refused
with a 501.

`reasoning_effort` takes the form the model accepts, by the version in
a Claude's name. From Claude 4.6 (Sonnet 5, Opus 4.7, an unversioned
preview) it is adaptive thinking with the level as `output_config.effort`
(`minimal` as `low`); `none` turns thinking off, or on Fable and Mythos,
which always think, is `low`. Through Claude 4.5, and on another
vendor's Messages endpoint, it is a budget spent from `max_tokens`, left
off for the continuation of a tool call or a forced tool, which want a
signed thinking block no OpenAI client can hand back.

The translation also does what the API needs that an OpenAI client would
not know to: prompt caching is on (breakpoints on the last tool, system
block and user block), a `response_format` JSON schema becomes a forced
tool whose arguments are returned as the reply, tools used earlier in
the history are declared back when the client offers none, and an
unanswered tool call gets a placeholder result.

## OAuth credential forwarding

Enable `[managed]` in [configuration.md](configuration.md#managed-oauth-forwarding)
to forward native Codex and Claude requests on the daemon's existing TLS
listener. The caller owns login, refresh, account selection, authorization,
and network policy. llmman never persists these request-local OAuth tokens or
falls back to stored provider keys, environment credentials, or peers.

Every forwarding request must come from a loopback connection and supply all of:

```text
X-Api-Key: <configured daemon API key>
Authorization: Bearer <current provider OAuth access token>
X-LLMMan-Upstream-IP: <policy-authorized public IP for the provider>
```

The daemon key is checked separately and removed before forwarding. An OAuth
bearer alone cannot authenticate to the daemon. Peer admission uses the actual
connection address, never `Forwarded` or `X-Forwarded-For`. Neither disabling
daemon authentication nor configuring an HTTP listener is allowed in managed mode.

| Managed operation | Fixed upstream | Required model prefix |
|---|---|---|
| `POST /api/codex/responses` | `https://chatgpt.com/backend-api/codex/responses` | `llmman.provider/openai/` |
| `POST /api/codex/responses/compact` | `https://chatgpt.com/backend-api/codex/responses/compact` | `llmman.provider/openai/` |
| `POST /api/anthropic/messages` | `https://api.anthropic.com/v1/messages` | `llmman.provider/anthropic/` |
| `POST /api/anthropic/messages/count_tokens` | `https://api.anthropic.com/v1/messages/count_tokens` | `llmman.provider/anthropic/` |

The route selects the provider; no auth-profile header or account-header
heuristic is needed. `ChatGPT-Account-ID` is optional on Codex requests and is
never sent to Anthropic. The exact nonempty model suffix is forwarded after
removing the table's prefix. Other native payload fields are preserved; invalid
credentials, mismatched model prefixes, and duplicate top-level JSON fields
are rejected before contacting a provider. Request bodies are bounded to 32 MiB.
Encoded request bodies are rejected with 415; absent or case-insensitive
`identity` content coding is accepted and removed before JSON is reserialized.

The supervisor resolves the fixed provider hostname, authorizes a particular
IP against its network policy, and supplies that numeric address in
`X-LLMMan-Upstream-IP`. llmman pins the connection to it while verifying the
provider's TLS hostname. There is no DNS fallback or caller-controlled URL.
Special-use addresses, including private networks, loopback, and IPv6 6to4,
are rejected. Redirects and environment proxy/CA overrides are not used.

Provider extension headers pass through in both directions, including repeated
values. Hop-by-hop headers, fields named by `Connection`, `Host`, internal
`X-LLMMan-*` headers, daemon/alternate credentials, and cookies are removed.
`Content-Length` is regenerated after body changes. The provider bearer and
Codex account ID are set explicitly. Codex `originator` and `openai-beta` are
forwarded only when supplied; llmman does not invent a client identity.
For Claude, llmman merges `oauth-2025-04-20` into `anthropic-beta` and defaults
`anthropic-version` to `2023-06-01` when absent. The caller supplies any required
native system content or metadata.

Successful response bodies stream with backpressure and cancellation on
disconnect. Upstream connection establishment has a 30-second timeout; reads
have no deadline unless `managed.read_timeout_seconds` is configured. That
optional timeout applies to response-header waits and individual stalled reads,
not total generation duration. A continuously active stream can outlive this
limit; a quiet generation is interrupted when the limit expires. Choose a
value appropriate for the workload.

Provider error statuses are preserved with bearer/account values redacted from
error bodies and response headers. JSON error bodies are decoded before
redacting string values and keys, so escaped credentials are removed too.
Error bodies above 1 MiB, interrupted reads,
redirects, and encoded error bodies that cannot be safely redacted produce 502.
llmman requests uncompressed upstream responses. Transport diagnostics are
redacted and available with debug logging. A failure after streaming headers
have been sent terminates the stream instead of changing its status.

`GET /api/version` confirms daemon liveness. Before supplying provider tokens,
call `GET /api/managed/capabilities` over a trusted TLS connection with the
same daemon API key, and require the corresponding capability:

```json
{"capabilities":["codex-oauth-forwarding-v2","claude-oauth-forwarding-v2"]}
```

These routes bypass prompt-history recording and ordinary provider routing.
They are disabled by default. The earlier draft's second listener, JSON config,
ready file, and managed-key/auth-profile headers have been removed; callers of
that draft must migrate configuration, paths, authentication, and capability
checks together. A supervisor must bind readiness to the child it launched and
invalidate it on exit; a version response alone is not a forwarding handshake.
