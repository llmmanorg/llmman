# Configuration

## Store location

Default locations:

| OS | Path |
|----|------|
| Linux, macOS | `~/.local/share/llmman/store` |
| Windows | `%LOCALAPPDATA%\llmman\store` |

Set `LLMMAN_MODELS` to change this (matching Ollama's `OLLAMA_MODELS`).
Commands that read or write the local store directly (`list`, `rm`, `cp`,
`show`, `build`, `serve`, `log`) all honor it. Commands that go through the
background daemon instead (`pull`, `push`, `run`, `launch`, `ps`, `stop`)
always use whichever store the daemon was started with; set `LLMMAN_MODELS`
before `llmman serve` to change it for all of them. `transfer`, `login`, and
`logout` never touch a local store at all.

If `LLMMAN_MODELS` puts the store on a different filesystem from its sibling cache (`<store>/../cache`), llmman copies cached model files instead of hardlinking them.

The store uses [OCI Image Layout](https://github.com/opencontainers/image-spec/blob/main/image-layout.md), readable by `docker` and `podman`.

When listing OCI artifact referrers, the Docker-backed build reads the registry's
native referrers API when available and otherwise uses the digest-derived
fallback tag. The Podman-backed build reads and writes only the fallback tag,
even if the registry supports the native API, because its backend does not
expose an authenticated request for that endpoint. Consequently, Podman
listing does not include artifacts published only through the native API.

## llmman.conf

Everything that needs a file rather than an environment variable lives in
one place: short-name aliases, provider API keys and endpoints, the
signature trust policy, the peers of an aggregation, the daemon's own
API keys, registry mirrors, and the defaults for `llmman launch`.

```toml
# ~/.config/llmman/llmman.conf

[aliases]
gemma4 = "docker.io/ai/gemma4"

[providers.openrouter]
api_key = "sk-or-..."

[providers.inferencebox]           # an endpoint models.dev does not list
base_url = "http://inferencebox:8000/v1"

[verify]
default      = "off"
layers       = "off"               # hash stored layers on a cold load
trusted_root = "sigstore/trusted_root.json"   # for keyless rules, relative to this file

[[verify.trust]]
pattern = "docker.io/myorg/**"
keys    = ["keys/myorg.pub"]   # relative to this file's directory
mode    = "enforce"

[[verify.trust]]
pattern = "docker.io/myorg/ci/**"  # keyless: an identity instead of keys
issuer  = "https://token.actions.githubusercontent.com"
subject = "https://github.com/myorg/models/.github/workflows/sign.yml@refs/heads/main"
mode    = "enforce"

[[verify.trust]]
pattern = "mirror.corp/**"         # pulled reference selects the rule
claims  = "docker.io/myorg/**"     # repository the signature may name
keys    = ["keys/myorg.pub"]
mode    = "enforce"

[aggregation]
peers = "asahi, spark:17434"

[auth]
api_keys = "k1, k2"                # what a request to this daemon must present

[registries."docker.io"]
mirrors = "https://mirror.gcr.io, registry-mirror.corp:5000"

[launch.opencode]                  # what `llmman launch opencode` means with no flags
provider = "anthropic"
model    = "claude-sonnet-5-5"
variant  = "xhigh"
```

Read from two locations, later overriding earlier:

| Path | Purpose |
|------|---------|
| `/etc/llmman/llmman.conf` | system-wide, admin-managed |
| `~/.config/llmman/llmman.conf` | per-user |

The same two paths on every platform, including macOS and Windows, with no
llmman-specific variable to move them (`~` is `$HOME` as usual): one
documented answer to "where does this go" beats a per-OS convention.

Unknown sections and keys are rejected rather than ignored: a misspelled
`[verify]` or `api_kye` that parsed happily would be a policy or a
credential that silently never takes effect, and the symptom would show up
somewhere else entirely.

### Launch defaults

A `[launch.<integration>]` table holds the flags `llmman launch
<integration>` uses when you leave them out:

```console
$ llmman config set launch.opencode.provider anthropic
$ llmman config set launch.opencode.model claude-sonnet-5-5
$ llmman config set launch.opencode.variant xhigh
$ llmman launch opencode
[llmman] opencode: llmman.conf supplies --provider anthropic --model claude-sonnet-5-5 --variant xhigh
```

`provider`, `model` and `variant` are `--provider`, `--model` and
`--variant`, all optional and checked as if typed. The table is named for
an integration as `llmman launch` lists it (any case, and `copilot-cli`
for `copilot`).

Flags you type win. `provider`, `model` and `variant` choose a model
together, so typing `--provider`, `--model`, `--overflow-provider` or
`--overflow-model` turns all three off: a configured provider would send
your model to the wrong host, and the variant belongs to the old model.
`--variant` alone replaces only the variant. A `--model` after `--` wins
over the configured model and drops the configured variant. With the table
above:

| Command | Runs |
|---------|------|
| `llmman launch opencode` | `anthropic`, `claude-sonnet-5-5`, `xhigh` |
| `llmman launch opencode --variant high` | `anthropic`, `claude-sonnet-5-5`, `high` |
| `llmman launch opencode --model qwen3.8` | local `qwen3.8`, no variant |

`--optimizer` is not one of these keys: type it (see [optimizers.md](optimizers.md)).

The two files merge one key at a time, later over earlier, and a blank
value (`variant = ""`) clears one that the system file set. A misspelled
integration, key or variant is rejected. If a file does not parse, a
launch that needs the defaults fails; naming a model with `--model` does
not read them.

### Managed OAuth forwarding

Native Codex/Claude OAuth forwarding is explicitly enabled in the same file:

```toml
[auth]
api_keys = "a-strong-daemon-key"

[managed]
enabled = "true"
# Optional upstream idle-read timeout, in seconds; absent or "0" is unbounded.
# read_timeout_seconds = "600"
```

Use `llmman config set managed.enabled true` and, if desired,
`llmman config set managed.read_timeout_seconds 600`. Values are strings,
matching the other sections. Invalid booleans or timeout values are rejected
before `config set` saves the file, and also when the daemon reads it.
Later files override individual fields. No additional configuration file is used.

Managed forwarding requires the existing `LLMMAN_TLS_CERT` and `LLMMAN_TLS_KEY`
settings and a matching HTTPS `LLMMAN_HOST`. The TLS private key must pass the
existing owner-only file permission check, as must configuration files supplying
daemon API keys. Missing keys, plaintext HTTP,
or `LLMMAN_AUTH=off` prevent startup when managed forwarding is enabled.
Managed forwarding is currently supported only on Unix platforms. On Windows
and other non-Unix platforms, enabling it fails startup because owner-only key
ACLs cannot yet be verified. Ordinary daemon operation remains available.

Only loopback peers can use the forwarding routes, even when the daemon also
serves ordinary routes on a network interface. The client sends its daemon key
as `X-Api-Key`, leaving `Authorization` for its provider OAuth bearer. See
[providers.md](providers.md#oauth-credential-forwarding) for routes, capability
checks, network-policy IP pinning, and header handling.

### llmman config

`llmman config` edits that file the way `git config` edits `.gitconfig`,
so nothing above has to be typed into an editor:

```console
$ llmman config set aliases.gemma4 docker.io/ai/gemma4
$ llmman config set providers.openrouter.api_key sk-or-...
$ llmman config get aliases.gemma4
docker.io/ai/gemma4
$ llmman config list
aliases.gemma4=docker.io/ai/gemma4
providers.openrouter.api_key=<redacted>
$ llmman config unset providers.openrouter.api_key
```

| Command | Effect |
|---------|--------|
| `llmman config list` | Print every key that is set, one `key=value` per line. `--show-origin` prefixes each with the file it came from. |
| `llmman config get <key>` | Print one value, or exit non-zero if it is not set. |
| `llmman config set <key> <value>` | Set one key. |
| `llmman config unset <key>` | Remove one key, or exit non-zero if it was not set. |
| `llmman config edit` | Open the file in `$VISUAL`/`$EDITOR`, then check that it still parses. |

A key is a TOML dotted key, so an id that is not a bare key is quoted
exactly as it is in the file: `llmman config get
'providers."wafer.ai".api_key'`. Array entries are addressed by index for
reading — `verify.trust[0].pattern` — but not for writing; use
`llmman config edit` for a trust policy. Values are always written as
strings, which is every field the format has.

`list` and `get` read both locations, later overriding earlier, so what
`get` prints is what llmman would actually use. Array indices are the
exception: trust rules are appended across files rather than overridden,
so `verify.trust[0]` names one entry per file and `--show-origin` is what
tells them apart. `set`, `unset` and `edit`
write the per-user file; pass `--system` for `/etc/llmman/llmman.conf`
(which generally needs root) or `--file <path>` for any other one.
`--user` restricts `list` and `get` to the per-user file alone.

Two things it does that an editor cannot. Every write is checked against
the format before the file is replaced, so `api_kye` is an error on the
spot rather than a key that silently never takes effect, and the file on
disk is left as it was. And a file that ends up holding an `api_key` is
made owner-only, since llmman would otherwise ignore the key that was
just set (see below); one holding no key keeps the mode it had. Comments
and layout survive a write.

`list` prints `<redacted>` for an `api_key`, since it is the command whose
output ends up in a bug report. `get` prints it in full.

### Provider API keys

`--provider` reaches a hosted model with a key llmman never generates
and never writes into an integration's config. It takes one from the variable models.dev names for that
provider, or — when that is unset — from `[providers.<id>]`. Entries are
keyed by the provider id `llmman providers` prints, not by the variable,
which is models.dev's naming rather than llmman's.

The environment wins where both have a key, as it does for `aws` and
`gh`: the file is the standing answer and an `export` is the deliberate,
this-session-only override. A login stored by `llmman login --provider`
(see [providers.md](providers.md#logging-in)) sits between them: it
outranks the file, so `llmman logout --provider <id>` is what makes the
file's key apply again. An id that is not a bare TOML key must be
quoted — `[providers."wafer.ai"]` — since `[providers.wafer.ai]` is two
nested tables. Setting `api_key = ""` blanks out a key `/etc` supplied.

On Unix a file carrying a key must not be readable by group or other —
`chmod 600` it — or llmman ignores the keys in it with a warning, the way
`ssh` refuses a loose private key. Only the keys: a world-readable
`/etc/llmman/llmman.conf` still supplies its aliases and trust policy,
which are not secrets. This is unchecked on Windows, which has no mode
bits.

Which process reads it matters. A key found by `llmman run`/`launch`
travels per request in an `Authorization` header. A key found by
`llmman serve` is the fallback for a request that presents none, and is
only spent for a daemon bound to loopback. `llmman providers` reports
which of the two has a usable key.

### Provider auth, headers and per-account URLs

Any `[providers.<id>]` — a catalog provider's or one you define — may say
how its credential is attached and where else it can come from:

```toml
[providers.azure]
auth_header = "api-key"                  # `auth` = bearer | x-api-key | header | query
vars        = { AZURE_RESOURCE_NAME = "my-resource" }
query       = { api-version = "v1" }
headers     = { X-Team = "ml" }

[providers.groq]
api_key_command = "pass show groq"       # or: api_key_file = "~/.secrets/groq"
```

| Field | Meaning |
|-------|---------|
| `auth` | `bearer`, `x-api-key`, `header` or `query`. Absent: the wire's own (`bearer` for `openai`, `x-api-key` for `anthropic`) — what llmman has always sent. |
| `auth_header` / `auth_query` | The header's or parameter's name. Either alone implies `auth`; with `bearer` or `x-api-key` they are a contradiction and rejected. |
| `headers` | Extra request headers. `Host`, `Content-Length` and other framing headers are rejected, and values are never printed. Held to the standard of a key: ignored in a group- or world-readable file, and sent only where the daemon's own key may go. |
| `query` | Extra query parameters, held to the same standard as `headers`. |
| `vars` | Values for `${VAR}` in a catalog provider's base URL (the environment wins). A value that does not fit where it lands is not used. |
| `api_key_command` | A shell command whose first line of output is the key, run when needed and cached for five minutes. |
| `api_key_file` | A file whose first line is the key; it must be owner-only. |

Tables merge field by field across files, later winning; `headers`,
`query` and `vars` merge by name. The credential-chain tables below merge as
described there. A file with an `api_key_command` or
`api_key_file` is held to the standard of a file with a key: on Unix a
group- or world-readable one is ignored for these with a warning, since a
command anyone can write is a command anyone can choose, and
`llmman config set` makes the file owner-only. The order of credentials
is: the provider's environment variable, a stored login, `api_key`, then
`api_key_command` / `api_key_file`.

Unlike `wire`, `api_key_env` and `name`, these may be set without a
`base_url`: they adjust the catalog provider, they do not define one.

### Cloud credential chains

A provider with no key can stand behind a chain: one of the tables below,
at most one per provider. They are how a provider models.dev does not list
gets one, and how a built-in one (`amazon-bedrock`, `google-vertex`,
`azure`, `databricks`, `watsonx`, `sap-ai-core`) is told which profile,
file, tenant or token endpoint to use. A chain is the *last* credential a
request looks for — after the provider's variable, a stored login, `api_key`
and `api_key_command` / `api_key_file` — and writing its table says "try
it" even where nothing on the machine hints that it will work. See
[providers.md](providers.md#cloud-credentials) for what each does.

```toml
[providers.amazon-bedrock]
aws = { profile = "work", region = "us-west-2" }     # service = "bedrock" is the default

[providers.google-vertex]
gcp = { project = "my-project", location = "us-central1", credentials_file = "~/keys/vertex.json" }

[providers.azure]
azure = { tenant_id = "...", client_id = "...", client_secret_env = "AZURE_CLIENT_SECRET" }

[providers.my-gateway]
base_url = "https://gateway.example/v1"
oauth2   = { token_url = "https://idp.example/oauth/token", client_id = "llmman",
             client_secret_file = "~/.secrets/gw", scope = "inference", client_auth = "basic" }
```

| Table | Fields |
|-------|--------|
| `aws` | `profile` (that profile alone, instead of the chain), `region`, `service` (the SigV4 signing name; `bedrock`, or `bedrock-mantle` for the Mantle endpoint). |
| `gcp` | `credentials_file` (a service-account or `authorized_user` file), `project`, `location`. |
| `azure` | `tenant_id`, `client_id`, one of `client_secret` / `client_secret_env` / `client_secret_file`, `federated_token_file`, `managed_identity` (bool), `cli` (bool), `scope` (default `https://cognitiveservices.azure.com/.default`), `authority_host`. |
| `oauth2` | OAuth 2.0 client credentials: `token_url` (https; may hold `${VAR}`), `client_id` / `client_id_env`, one of `client_secret` / `client_secret_env` / `client_secret_file`, `client_auth` (`body` or `basic`), `scope`, `audience`, `resource`, `params` (extra form fields). |
| `jwt_bearer` | RFC 7523: `token_url`, `issuer` (or `client_id`), `subject`, `audience` (the assertion's `aud`; the token URL if unset), `scope`, `key_file` (a PEM private key, owner-only), `key_id`, `algorithm` (`RS256` or `ES256`), `lifetime_secs`. |
| `ibm_iam` | IBM Cloud IAM (watsonx): `client_secret` / `_env` / `_file` for the IBM Cloud API key, `token_url` (default `https://iam.cloud.ibm.com/identity/token`). |

Across files a later table of the same kind overlays the earlier field by
field (the secret's three spellings replace each other as a group); a table
of another kind replaces it. A secret written in a table is held to the
standard of an `api_key`: the file must be owner-only.

### Stored logins

`llmman login --provider` keeps credentials in
`~/.config/llmman/credentials.json`, not in `llmman.conf` — it is rewritten
on every token refresh, which a file people edit by hand should not be. The
same rule applies as for a key: on Unix it must be `0600` or it is ignored
with a warning. A login for a cloud chain stores the selection (a profile, a
file's path, a tenant and client secret) and never a token. See
[providers.md](providers.md#where-logins-are-kept).

### Your own provider endpoints

A `[providers.<id>]` that sets `base_url` *defines* a provider rather
than keying a catalog one — an inference server at some host or URL
models.dev does not list, or a replacement URL for one it does:

```toml
[providers.inferencebox]
base_url    = "http://inferencebox:8000/v1"   # required; http or https
wire        = "openai"                        # default; or "anthropic"
api_key     = "..."                           # optional: most local servers take none
api_key_env = "INFERENCEBOX_API_KEY"          # optional: a variable to read it from
name        = "Inference box"                 # optional: for listings
```

`wire`, `api_key_env` and `name` only mean something on a definition, so
a table that sets one without `base_url` is rejected, as is a `base_url`
that is not an absolute `http`/`https` URL — by `llmman config set` on
the spot, or at load with the rest of the file. Later files override
earlier ones field by field. `llmman serve` reads the file at startup,
so a new definition needs a restart. See
[providers.md](providers.md#your-own-endpoints) for how one is used.

### Short-name aliases

`[aliases]` maps a short name to a full reference, so `llmman pull gemma4`
resolves to whatever you point it at. An alias for `name` answers `name:latest`
too, unless `name:latest` has an alias of its own.

Nothing is compiled into the binary but two Ollama names that Docker's `ai/`
namespace has no model for, used when `[aliases]` has nothing for them:

| Name | Resolves to |
|------|-------------|
| `nomic-embed-text` | `docker.io/ai/nomic-embed-text-v1.5` (768 dimensions) |
| `bge-m3` | `hf.co/ggml-org/bge-m3-Q8_0-GGUF` (1024 dimensions) |

### Signature trust policy

`[verify]` decides which references must carry a trusted signature before
`llmman pull` will accept them, and which keys count as trusted. Absent,
nothing is verified.

A file that is present but unreadable or malformed is a fatal error for
verification, rather than a silent downgrade to `off` — even though the
same failure only costs the other two sections their aliases and keys.
That asymmetry is deliberate: a trust policy llmman cannot read must not
be mistaken for one that does not exist.

A `[[verify.trust]]` rule trusts either `keys` or a keyless identity,
`issuer` and `subject`, which are compared exactly against the OIDC
issuer and a subject alternative name of the Fulcio certificate the
signature carries. Both are required: a rule setting one without the
other is rejected when the file is read. A rule that sets `keys` together
with `issuer` or `subject` is rejected when the file is read, with the
same fatal outcome as a malformed file. Keyless rules need
`verify.trusted_root`, a Sigstore TrustedRoot JSON holding the Fulcio
roots and the Rekor log keys; a relative path resolves against the
directory of the file that sets it, like `keys`, and a later file's
value replaces an earlier one. A keyless rule matching a pull with no
`trusted_root` configured is a broken configuration, never an unsigned
model: under `enforce` the pull is refused, under `warn` it proceeds
with a warning naming `trusted_root`, and the model is never reported
as "not signed".

`verify.layers` is a separate switch: `off` (the default), `warn`, or
`enforce`. It decides whether a cold load re-hashes the stored manifest,
its config blob and every stored layer blob and compares each SHA-256
with the digest the manifest records. `default` and the trust
rules do not turn it on, and it needs no keys: a signed model whose blob
was overwritten on disk is exactly what it catches. `warn` logs a notice
per mismatch and loads anyway; `enforce` fails the load. The manifest
is resolved once and that one manifest is checked and loaded, pinned
by digest (a model stored under an absolute path or a URI cannot carry
a digest and is loaded by name instead); each blob is opened without
following symlinks and must be a regular file, so a blob replaced by a
symlink, a directory or a FIFO is a mismatch. It runs once per
cold load, not per request, and the cost is one sequential read and
hash of every blob. Later config files override earlier ones, and
`LLMMAN_VERIFY_LAYERS` overrides them all.

See [verification.md](verification.md) for the format, what is checked,
and the limitations.

### Aggregation peers

`[aggregation]` names the other `llmman serve` daemons this one pools
hardware with, comma-separated and spelled like `LLMMAN_HOST`
(`[scheme://]host[:port]`, port defaulting to `17434`, or 80/443 with an
explicit scheme). `LLMMAN_PEERS`
overrides it for one daemon; a user file's value replaces `/etc`'s
rather than merging with it, and `peers = ""` opts out. See
[aggregation.md](aggregation.md).

### Authentication

```toml
[auth]
api_keys = "k1,k2"
```

Keys every request to `llmman serve` must present. `LLMMAN_API_KEYS`
overrides this for one daemon. The same rules as for provider keys: the
file must be owner-only (`chmod 600`) or the keys are ignored with a
warning, `llmman config set auth.api_keys ...` tightens the mode itself,
and `llmman config list` prints them redacted. A blank value in a user
file opts out of a system-wide set.

Without keys the daemon is open, as before — but only on a loopback
bind. `llmman serve` refuses to start on a host the network can reach
(`LLMMAN_HOST=0.0.0.0`, a LAN address) unless keys are configured or
`LLMMAN_AUTH=off` says the omission is deliberate. See
[api.md](api.md#authentication) for how a key is presented.

`[aggregation] api_key` is the key this daemon presents to its peers;
see [aggregation.md](aggregation.md#authentication).

### Registry mirrors

`[registries."<host>"] mirrors` names the mirrors a pull from that
registry tries first, in order, before the registry itself, as dockerd's
`registry-mirrors` does. The key is the host as a reference spells it
(`docker.io`, `ghcr.io`, `registry.corp:5000`; Hub's `index.docker.io`
and `registry-1.docker.io` count as `docker.io`); each mirror is
`[scheme://]host[:port][/path]`, `https` by default. A mirror without
the blob answers 404 and the next is tried, then the registry, so a
stale or unreachable mirror costs a request, never a pull. Pushes,
`transfer` destinations and signatures always go to the registry.

```toml
[registries."docker.io"]
mirrors = "https://mirror.gcr.io, registry-mirror.corp:5000"

[registries."ghcr.io"]
mirrors = "http://ghcr-cache.corp:5000"   # plain HTTP
```

A comma-separated string like `peers`, so `llmman config set
'registries."docker.io".mirrors' https://mirror.gcr.io` can write it. A
user file's value for a host replaces `/etc`'s rather than merging, and
`mirrors = ""` opts a host out. `LLMMAN_REGISTRY_MIRRORS` overrides
Docker Hub's list for one daemon, as dockerd's `--registry-mirror`
does; other registries are file-only.

A mirror is asked for the same repository path as the registry, with
containerd's `?ns=<registry>` hint (which a pull-through cache such as
`registry:2` with `proxy.remoteurl` ignores). A mirror's credentials are
its own: `llmman login mirror.corp:5000` (the `host[:port]`, no scheme or
path), and the registry's are never sent to it. Hugging Face has no
mirrors in this sense; `HF_ENDPOINT` moves the whole host instead.

### Hugging Face token

Gated and private Hugging Face repositories need an access token. llmman
checks `HF_TOKEN`, then the legacy `HUGGING_FACE_HUB_TOKEN`, then the
token file at `$HF_TOKEN_PATH` (default `$HF_HOME/token`; `HF_HOME`
defaults to `$XDG_CACHE_HOME/huggingface`, else
`~/.cache/huggingface`). Blank tokens (in the variables or the file) are
skipped; an empty `HF_TOKEN_PATH`, `HF_HOME` or `XDG_CACHE_HOME` counts
as unset.

Before llmman honoured `XDG_CACHE_HOME` here, its token file was always
`~/.cache/huggingface/token` unless `HF_HOME` or `HF_TOKEN_PATH` was set.
So that a token saved then keeps working, llmman still reads that file
when neither `HF_TOKEN_PATH` nor `HF_HOME` is set, `XDG_CACHE_HOME` moves
the default elsewhere, and the token file at the new default does not
exist; `huggingface_hub` does not read it in that case. In that same
case `llmman logout hf.co` removes both files, including a token that
`hf auth login` saved at `~/.cache/huggingface/token` from an
environment without `XDG_CACHE_HOME`.

`huggingface_hub` uses the same variables in the same fallback order for
its default token file, so a token saved by `hf auth login` or
`llmman login hf.co` is found by the other tool, provided both resolve
the same token file and nothing shadows it in the reading process: a
nonblank `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`, or, for
`huggingface_hub`, its further sources such as `HF_OIDC_RESOURCE`.
Cases where the files can still differ include: `huggingface_hub`
expands `~` and `$VAR` in these variables, does not treat an empty value
as unset, and uses a separate token file (`~/.cache/huggingface_staging/token`) when
`HUGGINGFACE_CO_STAGING` is set to a true value such as `1`.

```bash
llmman login hf.co              # prompts; or: llmman login hf.co --token hf_...
llmman pull hf.co/<org>/<gated-repo>
llmman logout hf.co
```

`llmman pull` runs in the daemon, so the variables that count are the
daemon's: export `HF_TOKEN` before `llmman serve`, or before the first
command that starts the daemon. Setting it for one `llmman pull` against
a daemon that is already running has no effect. An uncached pull reads
the token file again, so `llmman login hf.co` works without restarting
the daemon, provided neither `HF_TOKEN` nor `HUGGING_FACE_HUB_TOKEN` is
set to a nonblank value in the daemon's environment and both processes
resolve the same token file (`HF_TOKEN_PATH`, else `HF_HOME`, else
`XDG_CACHE_HOME`, else the home directory).

A new Hugging Face pull records the commit the Hub reports, when it
reports one, in the model config `descriptor.revision` field and in the
manifest annotation `org.llmman.revision`, which `llmman show` prints
under Labels. A model pulled before this was recorded is not backfilled.

## Environment variables

Daemon-wide settings, set before `llmman serve` starts. llmman is a very
different program underneath (no per-GPU memory estimator, no embedded
inference engine, no cloud/desktop-app features), so an equivalent
setting may not behave identically.

| Variable | Effect |
|----------|--------|
| `LLMMAN_DEBUG` | Enables verbose diagnostic logging (a spawned backend's full command line, per-GPU probe detail, etc). Accepts `1`/`true`/`yes`/`on`, or any other non-zero integer. |
| `LLMMAN_HOST` | `[scheme://][host][:port]` `llmman serve` binds to. Every `llmman` client in the same environment connects to it too, rewriting a wildcard host to loopback first, and over TLS when the scheme is `https://` (see `LLMMAN_TLS_CERT`). Defaults to `127.0.0.1:17434`. Pointing it at an Ollama server works for `llmman pull` and `llmman run`. llmman never stops or restarts a server it does not recognise as its own `llmman serve` daemon. |
| `LLMMAN_API_KEYS` | Comma-separated API keys every request to `llmman serve` must present, as `Authorization: Bearer <key>` or `x-api-key: <key>`. Overrides `[auth] api_keys` in `llmman.conf`; set but empty means none. Required whenever `LLMMAN_HOST` binds beyond loopback: the daemon refuses to start otherwise. See [Authentication](#authentication). |
| `LLMMAN_AUTH` | `off` (or `0`/`false`/`no`) serves without keys even on a bind the network can reach — for a daemon behind a gateway that authenticates for it. Configured keys are still recognized and stripped from requests, just not required. |
| `LLMMAN_API_KEY` | The key the `llmman` CLI (and `llmman launch`'s integrations) present to the daemon. Defaults to the first of `LLMMAN_API_KEYS`/`auth.api_keys` (unless `LLMMAN_AUTH=off`), so a CLI and the daemon it manages need only one setting. Sent to a remote `http://` daemon with a one-time warning: prefer TLS. |
| `LLMMAN_PEER_API_KEY` | The key `llmman serve` presents to its [aggregation](aggregation.md) peers, overriding `[aggregation] api_key`. Defaults to the first of its own keys, so a pool sharing one key set needs nothing more. |
| `LLMMAN_TLS_CERT` / `LLMMAN_TLS_KEY` | PEM certificate chain and private key; set both and `llmman serve` terminates TLS itself (rustls). Set `LLMMAN_HOST=https://...` too, so the CLI in the same environment connects accordingly. |
| `LLMMAN_TLS_CA` | A PEM bundle of extra roots to trust, for a private CA: the daemon uses it to reach peers, and the CLI to reach the daemon. |
| `LLMMAN_CONTEXT_LENGTH` | Context size for llama-server/vLLM/SGLang/ds4 when set. Defaults to `262144` (256k) for llama-server, capped to each model's trained context; backend-specific forwarding is below. A llama-server model can be given its own with `options.num_ctx` on `/api/chat` and `/api/generate`, which wins over this (see [Ollama API notes](api.md#ollama-api-notes)). |
| `LLMMAN_EMBEDDING_CONTEXT_LENGTH` | Context size for embedding models, in place of `LLMMAN_CONTEXT_LENGTH`. An embedding model reads its whole input in one batch as large as its context, so at the default it costs memory in proportion to its trained context (3.6 GiB for a 0.6B model trained to 32768); this lets one daemon serve a large-context chat model and a small embedder. Capped to the model's trained context, and `0` means that context. Unset, embedding models follow `LLMMAN_CONTEXT_LENGTH`. |
| `LLMMAN_HYBRID_LOCAL_BYTES` | Largest request body, in bytes, that a [hybrid model pair](providers.md#hybrid-model-pairs) serves locally; anything larger goes to the hosted half. `0` disables the size rule; a request the local half then refuses as over its context is still retried on the hosted half. Set, it is never overridden by a load; unset, the budget is four bytes per token of the window the local half loaded with, falling back to the daemon's context size until then. |
| `LLMMAN_KEEP_ALIVE` | The daemon-wide default `keep_alive` (how long an idle, unused model stays loaded before being unloaded). Defaults to 5 minutes. Overridden per-request by `/api/chat`/`/api/generate`'s own `keep_alive` field. |
| `LLMMAN_MAX_LOADED_MODELS` | Caps how many models this daemon keeps loaded at once, as one flat daemon-wide total (llmman has no per-model memory estimate to size an automatic per-GPU figure against). Once at the cap, the least-recently-used idle model is evicted to make room; if every loaded model is busy, the request gets a `503` instead. Defaults to `0` (unbounded, today's behavior, unchanged). |
| `LLMMAN_MAX_QUEUE` | Caps how many requests `llmman serve` admits into scheduling at once; anything beyond that gets an immediate `503` (`server busy, please try again.  maximum pending requests exceeded`, two spaces included). Defaults to `512`. |
| `LLMMAN_MAX_TRANSFER_STREAMS` | Maximum number of a HuggingFace safetensors repo's files downloaded concurrently during `pull`. Has no effect on GGUF transfers, and is not read by `transfer`'s own `docker`-feature registry-push path, which streams files sequentially. Defaults to `4`. |
| `LLMMAN_METRICS` | Serves the Prometheus scrape endpoint at `/metrics`. Accepts `1`/`true`/`yes`/`on`. Off by default: a daemon without keys has no authentication, so an upgrade should not start publishing this daemon's version, route mix, model names and model churn to whoever can reach the port. With keys, the scrape route requires one like every other. Unset, the route is absent and answers `404`. |
| `LLMMAN_MODELS` | Local store directory, overriding the default above. `pull`/`push`/`run`/etc. go through the daemon and always use whichever store it was started with. |
| `LLMMAN_NUM_PARALLEL` | Number of parallel request slots (`--parallel`) for GGUF models (llama-server only; no vllm/mlx equivalent). `--ctx-size` is scaled up by this value first, so each slot still gets the full configured/default context rather than an even split of it; ignored (with a warning) for a load with no explicit context size to scale. Unset leaves llama-server's own default (`auto`: four slots sharing one KV cache), except for an embedding model, which gets one slot because four cost it several times the memory (llmmanorg/llmman#686); set this to opt back in. |
| `LLMMAN_SAFETENSORS_ENGINE` | Which engine serves safetensors models: `vllm` or `sglang` (case-insensitive). Unset (or empty) leaves the choice to llmman: `mlx_lm.server` on macOS (llmman's own copy, installed when the daemon starts; see [backends.md](backends.md#mlx-macos)), `vllm` otherwise; `vllm` forces vLLM even there. With a container runtime, `sglang` selects the `lmsysorg/sglang` image (pinned by `--sglang-version`). Any other value is reported once and ignored. See [backends.md](backends.md#sglang). |
| `LLMMAN_VLLM_IMAGE` | Container image for `vllm serve` with `--runtime docker`/`podman`, in place of the host's `vllm/vllm-openai`, `rocm/vllm` or `vllm/vllm-openai-cpu` (e.g. `dhi.io/vllm-openai:0.32`). Used as written; same as `--vllm-image`; conflicts with `--vllm-version`. Not used for Diffusers models. See [backends](backends.md#in-a-container). |
| `LLMMAN_VLLM_ARGS` | Extra arguments appended to every `vllm serve` llmman starts (plain and `--omni`, local or in a container), split on whitespace. An argument that starts with `'` or `"` runs to the matching quote, so it can hold a space (`--chat-template 'a b'`, not `--chat-template='a b'`), and loses the quotes; any other quote or backslash passes as-is. An unclosed quote stops `llmman serve` at startup. For the engine's own knobs llmman has no flag for: `--dtype bfloat16 --enforce-eager`, `--tp 2`, `--quantization awq`. They come after llmman's own `--served-model-name`/`--max-model-len`, so on vLLM's argparse a repeated flag here wins, except `--port` and `--host`: llmman's own always come last (`--host` is `127.0.0.1` locally, `0.0.0.0` inside a container). |
| `LLMMAN_MLX_LM_VERSION` | Exact [`mlx-lm`](https://pypi.org/project/mlx-lm/) release llmman installs into its own environment on macOS (`0.31.3`), overriding the repo-root `MLX_LM_RELEASE` pin. |
| `LLMMAN_SGLANG_ARGS` | The same for every `sglang serve` / `sglang.launch_server` llmman starts: `--disable-cuda-graph`, `--tp 2`, `--quantization mlx_q4`, ... |
| `LLMMAN_GGUF_ENGINE` | Which engine serves GGUF models: `ds4` (case-insensitive) for the `ds4-server` on `PATH`, under every `--runtime`, or `llama-server`; embedding and decision models stay on llama-server. Unset (or empty) sends DeepSeek V4.1 GGUFs to `ds4-server` when it is on `PATH` and everything else to llama-server. Any other value is reported once and ignored. See [backends.md](backends.md#ds4). |
| `LLMMAN_DS4_ARGS` | The same as `LLMMAN_VLLM_ARGS`, for every `ds4-server` llmman starts: `--ssd-streaming`, `--batched-session 4`, ... |
| `LLMMAN_LLAMA_ARGS` | Extra arguments appended to every `llama-server` llmman starts, local or in a container, split like `LLMMAN_VLLM_ARGS`. For llama-server's own knobs llmman has no variable for: `--slot-prompt-similarity 0.5`, `--cache-ram 16384`. `--port` or `--host` here stops `llmman serve` at startup: llmman sets both. A flag llmman sets from its own variable belongs in that variable (`--ctx-size` in `LLMMAN_CONTEXT_LENGTH`, `--parallel` in `LLMMAN_NUM_PARALLEL`, ...): llama-server warns on a repeated flag. In a container a path here must exist inside it: only the model (`/models`) and multimodal projector (`/mmproj`) directories are mounted. |
| `LLMMAN_PEERS` | Comma-separated peer daemons (`[scheme://]host[:port]`) to pool hardware with, overriding `[aggregation]` in `llmman.conf`. Set but empty takes this daemon out of its aggregation. See [aggregation.md](aggregation.md). |
| `LLMMAN_REGISTRY_MIRRORS` | Comma-separated mirrors (`[scheme://]host[:port][/path]`) to try before Docker Hub for a `docker.io/...` pull, overriding `[registries."docker.io"]` in `llmman.conf`; set but empty removes Hub's mirrors. Hub only, like dockerd's `--registry-mirror`; other registries take theirs from `llmman.conf`. See [registry mirrors](#registry-mirrors). |
| `LLMMAN_ORIGINS` | A comma-separated list of extra allowed CORS origins for the HTTP API. A single `*` anywhere in an entry matches any substring (`http://host:*` for any port, `https://*.example.com` for any subdomain, a bare `*` for everything), same as Ollama. Always includes every scheme/port on `localhost`/`127.0.0.1`/`0.0.0.0`/`[::1]` regardless of this variable. |
| `LLMMAN_SHELL` | The web UI's Shell (`/llmman/shell`, a terminal on the daemon's machine as the daemon's user). `0`/`false`/`no`/`off` removes it; unset (or `1`/`true`/`yes`/`on`) runs the login shell; any other value is the command to run instead, split like `LLMMAN_VLLM_ARGS`, so a quoted argument can hold a space (`tmux new -A -s llmman`, `sh -c 'exec tmux new -A'`); an unclosed quote stops `llmman serve` at startup. Regardless of this variable the shell is off whenever `LLMMAN_HOST` binds beyond loopback, requires the daemon's API key when it has one, requires the token `llmman serve` keeps in `shell-<host>-<port>.token` in llmman's data directory, and a browser page may only open one from an origin `LLMMAN_ORIGINS` allows. See [webui.md](webui.md). |
| `LLMMAN_WEBUI_DIR` | Serves the web UI from this directory instead of the copy built into the binary, uncompressed and uncached, for working on it (`LLMMAN_WEBUI_DIR=webui llmman serve`). Development only. |
| `LLMMAN_TK` | `off` makes `llmman tk`, which `llmman launch --optimizer tk` routes the agent's shell commands through, run every command untouched. See [optimizers.md](optimizers.md). |
| `LLMMAN_SANDBOX_IMAGE` | The image `llmman launch --sandbox` runs the integration from, for `docker`, `podman`, `apple-container`, `microsandbox` and `openshell`. Defaults to Docker Sandboxes' image for the agent (`docker.io/docker/sandbox-templates:<tag>`) where one exists. Other integrations need this set. `seatbelt` runs the installed integration rather than an image. `sbx` ignores this variable and uses its own agent image. See [sandbox.md](sandbox.md#images). |
| `LLMMAN_SCHED_SPREAD` | Truthy forwards `--split-mode layer` (spread a model across every GPU, already llama-server's own default); falsey forwards `--split-mode none` (restrict to one GPU). |
| `LLMMAN_FLASH_ATTENTION` | Flash Attention mode (`--flash-attn`): `on`, `off`, or `auto` (llama-server's own default). Also accepts `1`/`0`/`true`/`false`. |
| `LLMMAN_KV_CACHE_TYPE` | KV-cache quantization (`--cache-type-k`/`--cache-type-v`), e.g. `f16` (default), `q8_0`, `q4_0`. Trades output quality for memory at long context lengths. |
| `LLMMAN_RUNTIME` | Where `llmman serve` gets its inference engine: `auto` (default), `docker`, `podman`, `bin` or `path` — the same setting as `serve --runtime`, as an environment variable so it also reaches the daemon `llmman run`/`launch` start for you. See [backends.md](backends.md#choosing-a-runtime). Engine pins such as `--llama-cpp-version` have no environment variable: `serve --detach` with a different one restarts a running daemon (see [commands.md](commands.md)). |
| `LLMMAN_NO_HARDENED` | When set (to anything other than `0`/`false`/`no`/`off`), `llmman serve` runs its engine containers without the hardened flags they get by default (`--read-only` except behind `--gpus`, `--cap-drop all`, `--security-opt no-new-privileges` and a tmpfs per engine scratch path); the same setting as `serve --no-hardened`. Read once at startup, like `LLMMAN_NOHISTORY`. No effect under `--runtime bin` or `path`. See [backends.md](backends.md#--no-hardened). |
| `LLMMAN_LLM_LIBRARY` | Forces which GPU backend `llmman serve`/`run` picks (`cpu`, `cuda`/`cuda12`/`cuda_v12`, `cuda13`/`cuda_v13`, `rocm`, `opencl`, `vulkan`, or macOS-only `metal`), bypassing autodetection. Has no effect under `--runtime path` (that binary's backend is fixed), or on macOS's local-binary download (one asset per architecture, no separate choice to make). |
| `LLMMAN_IGPU_ENABLE` | Counts integrated GPUs (Vulkan only) when probing for an accelerator. Defaults to disabled, since an integrated GPU is usually a worse choice than the discrete/CPU fallback it would otherwise be skipped in favor of. |
| `LLMMAN_LOAD_TIMEOUT` | How long to allow a model load to stall before giving up. Zero or negative means wait forever. Defaults to 20 minutes (`vllm` and `sglang` can take many minutes to load a large safetensors model: a cold start of a 27B NVFP4 model on a DGX Spark takes about ten). Also passed to vLLM-Omni as `--init-timeout` (a day when unbounded). |
| `LLMMAN_VLLM_OMNI_GUARDRAILS` | When set (`1`/`true`/`yes`/`on`), a Diffusers-layout model served by vLLM-Omni keeps its safety guardrails on; llmman otherwise passes `--no-guardrails`. See [backends.md](backends.md#vllm-omni-diffusers-pipelines). |
| `LLMMAN_TMPDIR` | Staging directory for `llama-server` release downloads, overriding the default `tmp` subdirectory of the install root. |
| `LLMMAN_VERIFY` | Overrides the signature-verification mode (`off`, `warn`, or `enforce`) for every reference, ignoring what `[verify]` selected. Does *not* supply trusted keys — those still come from `llmman.conf`, so `enforce` with no configured keys fails every check rather than passing them. Intended for CI, which can demand `enforce` without editing config files. See [verification.md](verification.md). |
| `LLMMAN_VERIFY_LAYERS` | Overrides `[verify] layers` (`off`, `warn`, or `enforce`): whether a cold load re-hashes the stored manifest and layer blobs against the digests the manifest records. Independent of `LLMMAN_VERIFY` and of the trust rules. Resolved once when the policy loads; an unparsable value is an error, not a downgrade to `off`. See [verification.md](verification.md#layer-verification-at-load). |
| `LLMMAN_SIGN_PASSWORD` | Passphrase for the `--sign-key` private key used by `push`/`transfer`, when it is an encrypted PEM. Falls back to `COSIGN_PASSWORD`. Read by the CLI process, which does the signing itself; neither key nor passphrase reaches the daemon. |
| `LLMMAN_NOHISTORY` | When set (to anything other than `0`/`false`/`no`/`off`), `llmman serve` stops recording prompts for `llmman log`. Otherwise each request to a generation route (`/api/chat`, `/api/generate`, `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages`) appends its time, route, model, `User-Agent` and the last user message's text — not the transcript or the reply — to `prompts.jsonl` beside the store (`~/.local/share/llmman/prompts.jsonl` by default), readable only by its owner. Delete the file to clear the history. |
| `LLMMAN_NOUSAGE` | When set (to anything other than `0`/`false`/`no`/`off`), `llmman serve` stops recording token usage for `llmman usage`. Otherwise each reply that reports usage appends its time, route, serving model and provider, `User-Agent`, token counts and, for a priced provider model, the rate and cost in US dollars to `usage.jsonl` beside the store, readable only by its owner. No prompt text, so independent of `LLMMAN_NOHISTORY`; the `id` matches the `prompts.jsonl` entry when there is one. |
| `LLMMAN_NOPRUNE` | When set (to anything other than `0`/`false`/`no`/`off`), skips the garbage-collection sweep that `llmman rm`, `llmman serve` startup, and a `build`/`cp`/`/api/copy`/`/api/create` that replaces a tag's content otherwise run to delete blobs and extracted-cache entries no longer referenced by any local model. Note this is broader than skipping the daemon-startup catch-all: it also stops `llmman rm` itself from ever freeing disk space, so a removed model's (possibly multi-GB) weights stay on disk until a later sweep runs without this set. Useful for a shared/read-mostly store, or scripts that `rm` in a loop and prune once at the end. |
| `LLAMA_ARG_FIT` / `LLAMA_ARG_FIT_TARGET` / `LLAMA_ARG_THREADS` | llama.cpp's own env-configurable `--fit`/`--fit-target`/`--threads` options. Not something llmman parses itself, just forwarded through to every `llama-server` (local or container) it spawns, same as `CUDA_VISIBLE_DEVICES`/etc. below. |

### Context length by backend

| Backend | When `LLMMAN_CONTEXT_LENGTH` is set | When unset or invalid |
|---------|-------------------------------------|-----------------------|
| `llama-server` | Passed as `--ctx-size` as-is for generation models (llama-server allocates the KV cache at that size and caps each request slot to the model's trained context). Embedding models are always capped to their trained context, and `0` means that context; `LLMMAN_EMBEDDING_CONTEXT_LENGTH` sets theirs on its own. A decision model reads its whole prompt in one batch of this size (up to the trained context), costing memory in proportion, so unset it is `8192` tokens at most. | `--ctx-size 262144` (256k), or the model's trained context if smaller. If the load then fails with an out-of-memory error, llmman retries with the context halved (down to a 16384 floor) before giving up. |
| `vLLM` | Positive values are passed as `--max-model-len`; oversized values are rejected by vLLM. `0` is not forwarded. | Uses vLLM's model-derived default. |
| `vllm serve --omni` | Not forwarded: a diffusion pipeline has no context window. | — |
| `sglang` | Positive values are passed as `--context-length`, the same rule as vLLM's; oversized values are rejected by SGLang. `0` is not forwarded. | Uses the model's `config.json`. |
| `mlx_lm.server` | Not currently forwarded. | Uses `mlx_lm.server` defaults. |
| `ds4-server` | Positive values are passed as `--ctx`, the same rule as vLLM's. `0` is not forwarded. | Uses ds4's own default. |

GPU device-selection variables `llmman serve` forwards to every
`llama-server` it spawns (local or container):
`CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`,
`ROCR_VISIBLE_DEVICES`, `GGML_VK_VISIBLE_DEVICES`, `GPU_DEVICE_ORDINAL`,
`HSA_OVERRIDE_GFX_VERSION`.

## Pager

`providers`, `log`, `list`, `search`, and `ps` use a pager when stdout is a terminal. Pipes and
redirected output are written directly. Use `--no-pager` to bypass it for
one invocation.

```sh
llmman config set core.pager 'less -S'
llmman config set pager.providers false
llmman config set pager.log 'less -+F'
```

`pager.<command>` accepts `true`, `yes`, `on`, or `1` to use paging, and
`false`, `no`, `off`, or `0` to disable it (case-insensitive). A shell command
selects that command's pager. Supported commands are `providers`, `log`,
`list`, `search`, and `ps`.
A command-specific false value disables paging even when `LLMMAN_PAGER` is set.
Otherwise, the pager command is chosen from `LLMMAN_PAGER`, the command-specific
setting, `core.pager`, `PAGER`, then `less`, in that order; one that isn't
installed is skipped for the next, and if none is, output goes straight to the
terminal. An empty command or `cat` disables paging. `LESS=FRX` and `LV=-c` are supplied only when those
variables are unset.

## Following prompts

`llmman log -f` (`--follow`) bypasses the pager. It prints the selected
history, then streams new complete prompts in arrival order until Ctrl-C.
`--max-count`, `--skip`, and `--reverse` affect only the initial history;
model, text, and date filters also apply to new prompts. Follow mode waits
if the log has not been created yet and continues across truncation. Like
`tail -f`, it follows the opened file rather than a replacement at that path.
