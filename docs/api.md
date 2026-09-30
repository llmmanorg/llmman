# HTTP API

`llmman serve` listens on `127.0.0.1:17434` by default (`LLMMAN_HOST`
changes it; see [configuration.md](configuration.md)) and speaks the
Ollama, OpenAI, Anthropic and System One wire formats, plus a small API of its own.

| API | Endpoints |
|-----|-----------|
| Ollama | `/api/generate`, `/api/chat`, `/api/embed`, `/api/embeddings`, `/api/tags`, `/api/show`, `/api/pull`, `/api/push`, `/api/copy`, `/api/create`, `/api/blobs/{digest}`, `/api/ps`, `/api/delete`, `/api/version` |
| OpenAI | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/v1/responses`, `/v1/responses/input_tokens`, `/v1/audio/transcriptions` (also `/audio/transcriptions`) |
| Anthropic | `/v1/messages` |
| System One | `/v1/systemone` |
| llmman | `/llmman/providers`, `/llmman/providers/{id}`, `/llmman/node`, `/llmman/search`, `/llmman/search/model`, `/llmman/search/popular`, `/llmman/search/avatar`, `/llmman/shell` |
| Web UI | `/` and `/ui/*` — see [webui.md](webui.md) |
| llama.cpp | `/props` |
| Prometheus | `/metrics` (off unless `LLMMAN_METRICS` is `1`, `true`, `yes` or `on`) |

Every route but the web UI's own files requires an API key when the
daemon has any configured; see [Authentication](#authentication).

Use it as a drop-in Ollama server:

```
OLLAMA_HOST=127.0.0.1:17434 ollama run unsloth/Qwen3.5-0.8B-GGUF
```

Or with any Ollama, OpenAI or Anthropic client. `http://127.0.0.1:17434/`
in a browser is llmman's own web UI ([webui.md](webui.md)).

## Model lifecycle

Models load on demand, each in its own backend subprocess
(`llama-server`, `vllm`, `sglang` or `mlx_lm.server`; see [backends.md](backends.md))
on a random loopback port, reused by later requests.

An idle model unloads after `keep_alive` (default 5 minutes, as in
Ollama; per-request, or daemon-wide via `LLMMAN_KEEP_ALIVE`);
`llmman ps`/`/api/ps` report each model's `expires_at`.
`LLMMAN_MAX_LOADED_MODELS` caps how many stay loaded, evicting the
least-recently-used idle one. `llmman stop <model>` (or `keep_alive: 0`
on `/api/generate`) unloads immediately.

## Ollama API notes

`/api/chat` supports Ollama's `tools` (function calling, streamed back as
`message.tool_calls` with their `id`, and matched to `tool_call_id` on
the way back in), `images` (vision, base64, same as Ollama's own wire
format), `think`, and `format` (`"json"` or a JSON Schema object, for
constrained structured output). `/api/generate` takes `system` and
`images` too; `raw`, `suffix` and `template` are refused with a 400,
since llmman leaves templating to the backend.

Of `options`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`,
`num_predict`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`
and `num_thread` apply; `num_ctx` is `LLMMAN_CONTEXT_LENGTH`. The `done`
chunk carries Ollama's counts and durations (`prompt_eval_count`,
`eval_count`, `total_duration`, ...) and `done_reason` is `stop` or
`length` as in Ollama, which reports a tool-calling turn as `stop`.

`/api/embed` and `/api/embeddings` work with any GGUF embedding model
(one with a pooling type, e.g. `embeddinggemma`, `nomic-embed-text`):
`llama-server` is started with `--embeddings` for it, so `/v1/embeddings`
works too.

`think` is `true`/`false` or a level (`minimal`, `low`, `medium`,
`high`, `xhigh`, `max`), forwarded as llama-server's
`chat_template_kwargs`. `/api/show` returns `capabilities` and, for a
local model with one, `template`.

`/api/create` supports `from` (alias a model) and `files` (GGUFs uploaded
via `/api/blobs/{digest}`, as `ollama create` does). Modelfile fields such
as `system` or `quantize` are refused with a 400: the GGUF's own chat
template applies.

## OpenAI API notes

`/v1/responses` implements the OpenAI Responses API (the dialect
[OpenAI Codex](https://github.com/openai/codex) requires), including
streaming SSE and function-tool-call re-mapping. For a local model this
is a plain pass-through to `llama-server`'s own native `/v1/responses`
support, so a recent enough `llama-server` build is required for it to
work. For a [provider](providers.md) without the route, the daemon
translates to and from `/v1/chat/completions` itself, and for a provider
that speaks the Anthropic Messages API from there to `/v1/messages` (see
[wire formats](providers.md#wire-formats)).

A request bound for a provider loses llama-server's own fields
(`repeat_penalty`, `top_k`, `min_p`, `chat_template_kwargs`, ...), which
a strict provider would reject the whole request over; OpenAI's reasoning
models get `max_completion_tokens` and lose the sampling overrides they
refuse. `previous_response_id` is refused when `/v1/responses` has to be
bridged through chat completions: nothing is stored to resolve it against.

A local `/v1/chat/completions` with `reasoning_effort` also gets the
`chat_template_kwargs` Ollama's `think` would (`none` →
`enable_thinking: false`; a level → `enable_thinking: true` plus
`reasoning_effort`), so it works on llama-server builds that do not read
`reasoning_effort` themselves. The caller's own kwargs are kept.

A streamed `/v1/chat/completions` or `/v1/completions` to an `openai`-wire
provider (Cohere aside) or a non-llama.cpp backend goes upstream with
`stream_options.include_usage` for [`llmman usage`](commands.md); a
client that did not ask gets the usage chunk stripped back out.
llama-server's `timings` need no asking.

`/v1/audio/transcriptions` is likewise a pass-through. The model needs
audio support (an `--mmproj` projector, supplied when the model image
carries one). Bodies up to 200 MiB are accepted.

## Anthropic API notes

`/v1/messages` implements the Anthropic Messages API (the dialect
[Claude Code](https://github.com/anthropics/claude-code) requires). For
a local model or a provider on the `openai` wire the daemon translates
it to a chat completion and the reply back: system-role turns fold into
one leading system message, `tool_use`/`tool_result` become
`tool_calls`/`role: "tool"`, tools and `tool_choice` become functions
(names over 64 characters shortened and restored), `thinking` and
`output_config.effort` (Claude Code's `/effort`) become
`reasoning_effort` and llama-server's `chat_template_kwargs`,
`output_format` becomes `response_format`. Text,
reasoning (as `thinking`) and tool input stream back as they arrive,
with the real `stop_reason` and usage. Anthropic's server tools, cache
breakpoints and `anthropic-beta` headers have no chat-completion form
and are dropped; a `tool_choice` naming a server tool is a 400. A
provider on the `anthropic` wire gets the request relayed as sent (see
[wire formats](providers.md#wire-formats)).

## System One API notes

`/v1/systemone` is the System One decision API: a `state` and up to 64 typed
questions in, one answer each out, with a probability per candidate. Clients
written for it (the TypeSafe SDKs) work by pointing their base URL here, with
`model` naming a model llmman serves. Wording, labels, refusals and formulas
follow sglang's `/v1/systemone`.

```sh
curl localhost:17434/v1/systemone -d '{
  "state": "I have tried to connect Stripe for 3 days and the integration keeps failing.",
  "model": "qwen3.5:0.8b",
  "questions": {
    "team":   { "type": "choice", "instructions": "Which team should handle this?",
                "criteria": { "billing": null, "technical": "Bugs or integration problems", "sales": null } },
    "urgent": { "type": "noul", "instructions": "The customer needs an answer today." },
    "mood":   { "type": "score", "instructions": "How upset is the customer?",
                "criteria": ["Calm", "Annoyed", "Furious"] }
  }
}'
```

A `choice` answer has the most probable option, its `confidence` and
`probabilities` in the order the options were sent (the first is labelled
`A`, so order is part of the question); `noul` is the probability of yes;
`score` the probability-weighted mean level, with the levels as `legend`.
`usage` counts every question's prompt, so the state is counted once per
question, with cache and reasoning counts when a provider reports them;
hosted calls appear in `llmman usage`, including those answered beside a
question that failed.

Each answer's `x_source` says where its probabilities come from:

- **A local model is read** (`logprobs`): one user turn per question, rendered
  with the model's chat template, thinking off, and the model's own
  next-token probability of each label (`A`..`Z`, a level digit, `yes`/`no`),
  renormalised over the labels. Nothing is generated. `x_label_mass` is the
  probability the model put on the labels at all; low means it wanted to say
  something else. This needs `llama-server` (read through `/apply-template`,
  `/tokenize` and `/completion`); other engines get a 501. A label that is not
  one token at the answer position, a template that leaves a `<think>` block
  open, a model putting no probability on the labels, and a prompt past the
  context are each a 400. `chat_template_kwargs`, if given, is an object,
  accepted over `enable_thinking: false`.
- **A hosted model is asked** (`elicited`), for any provider and wire
  (`llmman.provider/<provider>/<model>`): hosted APIs show no token
  probabilities (Anthropic's has no logprobs), so the model states its
  probabilities as JSON, one request per question, four at a time. No
  `x_label_mass`.

A hosted model's variant is the `reasoning_effort` it thinks at: `none`,
`minimal`, `low`, `medium`, `high`, `xhigh` or `max`, in each provider's own
form. Name it as a `reasoning_effort` field or as a last segment of `model`,
which is all an SDK can set; these are one request:

```
"model": "llmman.provider/anthropic/claude-sonnet-5-5/xhigh"
"model": "anthropic/claude-sonnet-5-5/xhigh"
"model": "llmman.provider/anthropic/claude-sonnet-5-5", "reasoning_effort": "xhigh"
```

The bare form needs a known provider and an effort level; a bare
`<org>/<repo>` stays a Hugging Face repository, as on every other route. With
no variant the model's own default applies. A local model is read before it
thinks, so a variant on one is a 400, as is a `model` and `reasoning_effort`
that disagree.

Stated probabilities are the model's account of its own uncertainty, not a
measurement of it, and a small model's can be far worse than its token
probabilities for the same question. Neither source is a calibrated
probability of being right, and a threshold tuned on one does not carry to the
other; validate it on your own labelled data.

An invalid request is a 422 with FastAPI's `detail` list, one entry per
problem with its `loc`; any other refusal is a 400 in OpenAI's error shape,
and a provider's refusal of the caller (a bad key, a rate limit) keeps its
status. A choice takes 1 to 26 options (the schema allows 255; sglang labels
the rest with two-letter tokens, llmman does not) and a score 1 to 10 levels.
Unknown top-level fields are ignored, except `temperature`,
`prompt_format_version` and `return_prompt_token_ids`, which are refused unless
`null`, as sglang refuses them; unknown keys in a question are refused.

## llmman's own API

`/llmman/...` is llmman's own API, not a compatibility surface: no
upstream API has a notion of a [models.dev](https://models.dev) provider
(see [providers.md](providers.md)). `/llmman/providers` lists the ones
this daemon can route to, each with its API-key variable, whether the
daemon has that key, and how many models it serves;
`/llmman/providers/{id}` adds those models and what each costs in US
dollars per million tokens (absent, not zero, where models.dev publishes
no price), including `cache_read`/`cache_write`/`reasoning` where it publishes those.
`?model=<id>` filters nothing; a catalog that lacks it is re-fetched first
(at most every five minutes). `llmman providers`, `list --provider`, `run --provider` and
`launch --provider` are all clients of it, so the catalog is fetched and
cached in one process: the one that forwards the request upstream.

A provider [defined in `llmman.conf`](providers.md#your-own-endpoints)
appears alongside the catalog ones with `key_optional: true`, no
`key_env` unless the file names one, and — on `/llmman/providers/{id}`
— for the `openai` wire, whatever model ids its own `GET /models`
reports, unpriced.

`/llmman/node` reports this node's memory and loaded/stored models; it
is what aggregation peers ask each other. See [aggregation.md](aggregation.md).

`/llmman/search?q=<query>` is `llmman search` over HTTP: the same Docker
Hub and Hugging Face rows in the same order, as `{"models": [{name,
pulls, likes, updated}]}`, each `name` ready for `/api/pull`. `limit`
caps the rows per registry as `--limit` does (default 25, at most 64).
`/llmman/search/popular` has the same shape and needs no query: Docker
Hub's most pulled models, then Hugging Face's most downloaded GGUF
text-generation repos.
`/llmman/search/model?name=<a row's name>` expands one row: every tag
`pull` can take for it with its size (`variants`, the one a tagless
`pull` takes marked `default`), plus the repo's pulls, likes, license,
tags and README (Hugging Face's `README.md`, Docker Hub's overview; the
first 64 KiB). For a GGUF repo on Hugging Face the variants are its
quantizations, each one that resolves to its own file the way `pull`
picks one. `/llmman/search/avatar?name=` redirects to the repo owner's
picture on Hugging Face or Gravatar, or is a `404` when it has none.
These ask the registries live; a registry that cannot be reached is a
`502`.

## Authentication

With `LLMMAN_API_KEYS` (or `[auth] api_keys` in `llmman.conf`; see
[configuration.md](configuration.md#authentication)) set, every request
must present one of the keys, in either spelling the compatible surfaces
already use:

```http
Authorization: Bearer <key>      # OpenAI clients
x-api-key: <key>                 # Anthropic clients
```

Anything else is a `401` with `WWW-Authenticate: Bearer realm="llmman"`
and the daemon's usual `{"error": ...}` body. Only `GET /` and `/ui/*`
— the web UI's own files — are exempt, so the page can load and ask its
user for the key; it then sends the key on every call, and, since a
browser cannot set a header on a WebSocket, opens `/llmman/shell` with
the subprotocol `llmman.bearer.<base64url(key)>`, which the daemon
echoes back.

The `llmman` CLI sends `LLMMAN_API_KEY`, defaulting to the first
configured server key; `llmman launch` hands the same key to the
integration it starts.

A key that opens the daemon is not a provider key: the header it
arrived in is removed before routing, so it is never relayed upstream.
An authenticated caller *is* the operator, though, so the daemon spends
its own provider keys for it whatever `LLMMAN_HOST` it is bound to —
where an open daemon spends them only on a loopback bind. A caller that
wants to use its own provider key sends it in the other header
(`llmman run --provider` does this on its own).

Without keys the daemon is open, and only allowed to be so on loopback:
bound anywhere else it refuses to start unless `LLMMAN_AUTH=off`.

TLS is terminated by the daemon itself with `LLMMAN_TLS_CERT` and
`LLMMAN_TLS_KEY`, together with an `https://` `LLMMAN_HOST` so clients
in the same environment connect the same way (the daemon refuses one
without the other). `LLMMAN_TLS_CA` trusts a private CA — for every
connection the process makes, providers included.

## Metrics

`/metrics` is a Prometheus scrape target, off by default because an
open daemon has no authentication; with keys configured it requires one
like every other route. `LLMMAN_METRICS=1 llmman serve` turns it on;
the fifteen metric families and how to read them are in
[metrics.md](metrics.md).

## CORS

Browser clients on `localhost`, `127.0.0.1`, `0.0.0.0` and `[::1]` (any
scheme, any port) are always allowed. `LLMMAN_ORIGINS` adds more; see
[configuration.md](configuration.md#environment-variables).
