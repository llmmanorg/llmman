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

## Every model inside the agent

`llmman launch` gives the agent every pulled model and every chat model of
every provider (embedding, speech, transcription and image models are left
out), each with its variants, window and image input where known. The flags
choose the starting model only:

| Flags | The agent starts on |
|-------|---------------------|
| `--provider P --model M` | `M` of `P`; `P`'s models are listed first |
| `--provider P` | `P`'s default: the most recently released tool-calling chat model whose input price is at or below the median of P's tool-calling chat models (no price stays eligible, no release date counts as oldest, a tie goes to the greater id; a provider with no dates and no prices, such as one from `llmman.conf`, gets the first one it lists). One stderr line says which |
| `--model M` | the pulled model `M` (pulled and preloaded if needed) |
| neither | the model the agent last ran, else the pulled model that is loaded now, else the one pulled most recently, else the default of a provider you hold a key for (said on stderr, with the rule that picked it); with nothing usable it stops and says how to fix that |

`--variant` is checked against the starting model in every case.

Two agents take one model and ignore the list: GitHub Copilot CLI and
Gemini CLI (its `/model` does not change the model). And only claude, codex,
opencode, hermes and openclaw keep a model llmman can read, so only for them
does "the model the agent last ran" apply to a launch with neither flag; the
others go straight to the pulled model or provider. A bare `llmman launch kimi`
keeps Kimi Code's own model and gives it the list beside it in its
`config.toml`. With `--sandbox openshell`, which shares no file with the
agent, goose, vibe, kimi and aider get their one starting model with no list
file, claude and opencode get it without the picker, and Gemini CLI is
refused.

Rows an earlier launch wrote into a shared configuration (Droid, Pi, OMP) and
that are no longer offered are taken out again; rows you added stay. A Droid
entry's place is part of its id, so an old llmman entry that stands before one
of yours stays. With `OPENCODE_CONFIG` set, opencode gets your file and the
list as one config written for the launch (yours is never changed).

The list is thousands of rows, so it travels in a file in the agent's own
state directory (mode 0600, no secret in it, removed when the agent exits),
never in an environment variable or an argument. An agent with a real limit
gets the front of the list (start, its provider, pulled models, providers
with a key) and one line saying what was left out.

**Keys.** An agent that can switch provider cannot hold one provider's key:
the daemon would send it to whichever provider the next request names. So
`llmman launch` registers the keys this shell holds with the daemon, each
under its own provider (`POST /llmman/sessions`, in the daemon's memory
only, dropped when the agent exits or after 12 idle hours) and hands the
agent a session token instead. A request is served with the key registered
for the provider it names and no other; a provider with no key falls back
to the daemon's own, or fails with a message naming the variable to set.
The variable a key is read from is the one the launcher itself ties to the
provider (`llmman.conf`, a built-in endpoint, or the cached provider catalog),
never one the daemon names, and for a loopback daemon no more than a remote
one. The daemon keeps at most 64 sessions and never evicts one used within
the last hour: with 64 in use a new launch is refused with a message saying
so. Against a daemon on another machine (TLS) only the starting provider's key
is registered. A daemon that requires an API key, and the integrations
configured through a file (`hermes`, `cline`, `pi`, `openclaw`),
cannot carry a token: they use the daemon's own keys.

The daemon also records each reply's tokens, priced at the catalog's
rates (cache, reasoning and long-context included); `llmman usage` sums them. See
`LLMMAN_NOUSAGE` in [configuration.md](configuration.md).

## The catalog

The provider list comes from [models.dev](https://models.dev), the
catalog `opencode` uses, so a new provider needs no llmman release. It
is cached for 24 hours and a stale copy is used when the fetch fails.

When `--model` names a model the cached catalog lacks, the daemon re-fetches
once (at most every five minutes) before warning that the provider does not list
it. If that fetch is skipped or fails, the warning reflects the cached catalog.

All four commands read it from the daemon over
[`/llmman/providers`](api.md#llmmans-own-api), so the cache outlives any
one command and the key status reported is the daemon's.

## API keys

The credential `llmman serve` sends comes, in this order, from

1. the variable models.dev names for that provider (an `export` is the
   deliberate, this-session override, as it is for `aws` and `gh`);
2. a login stored by [`llmman login --provider`](#logging-in) — an API key
   you pasted, a subscription login, a key command or file;
3. `api_key` in `~/.config/llmman/llmman.conf`, keyed by provider id;
4. `api_key_command` or `api_key_file` there.

A stored login outranks the file because `llmman login` is the newer,
explicit act, and a leftover `api_key` should not silently win over it.
`llmman logout --provider <id>` removes the login, and the file's key
applies again. A request that carries a key of its own (what `launch` and
`run` send) uses that instead of any of these. The file:

```toml
[providers.openrouter]
api_key = "sk-or-..."
```

or, equivalently, `llmman config set providers.openrouter.api_key sk-or-...`,
or from a client over HTTP: `PUT /llmman/providers/openrouter/key`
(what a client such as a phone app uses; see
[api.md](api.md#llmmans-own-api)). A key set over HTTP is spent by the
running `llmman serve` from its next request; one set with `llmman
config set` or a hand edit reaches a running daemon when it restarts,
like every other setting.

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

## Logging in

A key is not the only way into a hosted provider. `llmman login --provider
<id>` stores a credential for it in `~/.config/llmman/credentials.json` (see
[Where logins are kept](#where-logins-are-kept)); `llmman serve` spends it
from the next request, refreshing it as it expires.

```sh
llmman login --provider openai-chatgpt                  # ChatGPT/Codex subscription, device code
llmman login --provider github-copilot                  # GitHub Copilot, device code
llmman login --provider xai-supergrok                   # SuperGrok subscription, device code
llmman login --provider openrouter                      # browser sign-in; stores the API key it returns
llmman login --provider anthropic --method setup-token  # paste what `claude setup-token` prints
llmman login --provider anthropic --method import-claude   # reuse Claude Code's login
llmman login --provider poe                             # browser sign-in; stores the API key Poe returns
llmman login --provider digitalocean                    # browser sign-in (implicit grant)
llmman login --provider chutes --param client_id=cid_.. # browser sign-in with your Chutes OAuth app
llmman login --provider gitlab                          # GitLab Duo, PKCE (--param instance_url=...)
llmman login --provider snowflake-cortex --param account=myorg-acct   # PKCE against your account
llmman login --provider minimax                         # MiniMax account, user code
llmman login --provider amazon-bedrock --method aws-profile --param profile=work
llmman login --provider google-vertex --method service-account --file key.json --param project=my-proj
llmman login --provider azure --method entra --param tenant_id=... --param client_id=...
llmman login --provider groq                            # paste an API key (the default method)
llmman login --provider groq --method command --command 'pass show groq'
llmman login --provider groq --method token-file --file ~/.secrets/groq
llmman login --provider anthropic --method ?            # what a provider takes
llmman logout --provider openai-chatgpt
```

`llmman login [SERVER]` (registries, Hugging Face) is unchanged;
`--provider` is a different mode, and cannot be combined with `SERVER`,
`--username` or `--password`. `llmman providers` shows each provider's
credential in the `KEY` column — `set (oauth)`, `set (command)`,
`expired (token)` — and never the secret; a key in the environment or
`llmman.conf` is just `set`, as before.

| Provider | Method | What it does |
|----------|--------|--------------|
| `openai-chatgpt` | `device` (default) | ChatGPT account via OpenAI's device-code flow. Refreshes; sends `ChatGPT-Account-Id`. |
| | `import-codex` | Reuses `~/.codex/auth.json` (`$CODEX_HOME`). |
| `github-copilot` | `device` (default) | GitHub device flow with `read:user`. The GitHub token is the credential (it does not expire); `copilot_internal/user` names the API host. Also read from `COPILOT_GITHUB_TOKEN`. |
| `xai-supergrok` | `device` (default) | SuperGrok subscription via xAI's device flow. Refreshes. |
| `openrouter` | `oauth` (default) | PKCE in the browser, returning to `http://localhost:3000/openrouter-oauth/callback`; OpenRouter answers with an API key, which is what is stored. |
| `anthropic` | `setup-token` (default) | Paste the `sk-ant-oat01-…` token from `claude setup-token`. |
| | `import-claude` | Reuses `~/.claude/.credentials.json` (`$CLAUDE_CONFIG_DIR`). Not the macOS Keychain. |
| `poe` | `browser` (default) | PKCE at `poe.com/oauth/authorize` (scope `apikey:create`), returning to a loopback port; `api.poe.com/token` answers with an API key, stored with its expiry (none when Poe states none). When it lapses the provider reports `expired`; log in again. |
| `digitalocean` | `browser` (default) | The implicit grant at `cloud.digitalocean.com/v1/oauth/authorize` (scopes `genai:read inference:query`), returning to `http://localhost:1456/auth/callback`. A browser never sends a URL's `#fragment`, so the page the listener serves forwards the token to it; the login stops if the port is busy, since another program could read the token. Stored with its expiry (30 days when none is stated). |
| `chutes` | `oauth` (default) | PKCE at `api.chutes.ai/idp/authorize` for **your own** OAuth app (`--param client_id=`, or `CHUTES_CLIENT_ID`; `CHUTES_CLIENT_SECRET` is sent too if set), redirect `http://127.0.0.1:1456/oauth-callback`. Refreshes. Sent only to `llm.chutes.ai`. |
| `gitlab` | `pkce` (default) | PKCE against `--param instance_url=` / `GITLAB_INSTANCE_URL` / gitlab.com, scope `api`, redirect `http://127.0.0.1:8080/callback`. gitlab.com uses opencode's registered application; another instance needs `GITLAB_OAUTH_CLIENT_ID` for yours. The instance is kept with the login; the refresh token rotates. |
| `snowflake-cortex` | `oauth` (default) | PKCE at `<account>.snowflakecomputing.com/oauth/authorize` as Snowflake's built-in `LOCAL_APPLICATION` client (`--param account=` or `SNOWFLAKE_ACCOUNT`; `--param role=` adds `session:role:<role>`). Refreshes. A programmatic access token is the provider's key: `SNOWFLAKE_CORTEX_PAT`, or `--method key`. The account of a login stands for `SNOWFLAKE_ACCOUNT` when the base URL is built. |
| `minimax`, `minimax-coding-plan`, `minimax-cn`, `minimax-cn-coding-plan` | `device` (default) | MiniMax's user-code flow at `account.minimax.io` (`account.minimaxi.com` for the `-cn` providers), polled until approved. Refreshes. Sent as a bearer to `api.minimax.io`, `api.minimax.cn` or `api.minimaxi.com`. |
| `amazon-bedrock`, `amazon-bedrock-mantle` | `aws-chain` (default) | Select the [AWS credential chain](#amazon-bedrock) (`--param region=`). |
| | `aws-profile` | A named AWS profile (`--param profile=`, `--param region=`). |
| `google-vertex` | `adc` (default) | Select [Application Default Credentials](#google-vertex-ai) (`--param project=`, `--param location=`). |
| | `service-account` | `--file key.json`: the path is kept, the key stays in its file. |
| `azure` | `entra` (default) | A client secret: `--param tenant_id=`, `--param client_id=`, the secret asked for. The token endpoint is asked once before it is stored. |
| | `managed-identity` | The VM's, App Service's or container's identity (`--param client_id=` for a user-assigned one). |
| | `federated` | `--file <projected token>`, `--param tenant_id=`, `--param client_id=`. |
| | `cli` | Whoever is signed in to the Azure CLI. |
| `databricks` | `oauth-m2m` (default) | A service principal: `--param client_id=`, the secret asked for, `--param host=` (or `DATABRICKS_HOST`) to try it; a personal access token is `DATABRICKS_TOKEN` or `--method key`. |
| `sap-ai-core` | `service-key` (default) | `--file service-key.json` or pasted: `clientid`, `clientsecret`, `url`. |
| `watsonx` | `apikey` (default) | An IBM Cloud API key, exchanged for IAM tokens. |
| any | `client-credentials` | OAuth 2.0 client credentials at `--param token_url=` (`client_id=`, `scope=`, `audience=`, `client_auth=basic\|body`); the secret is asked for. Tried once, then stored. Listed by `--method ?` apart from a provider's own methods. |
| any | `jwt-bearer` | RFC 7523: `--file key.pem`, `--param token_url=`, `--param issuer=` (also `subject=`, `audience=`, `scope=`, `key_id=`, `algorithm=`). The key's path is stored, not the key. |
| any | `key` | Paste an API key (not echoed; `--token` to pass it). The default for a provider with no login of its own. |
| any | `command` | `--command '<cmd>'` prints the key; run through `sh -c` (`cmd /C` on Windows) when needed, output cached for five minutes, stderr discarded, 20-second limit. |
| any | `token-file` | `--file <path>`; first line is the key; the file must be owner-only. |

### Subscriptions are their own providers

A ChatGPT or SuperGrok login is not an API key for the API behind the
same name: ChatGPT's goes to `chatgpt.com/backend-api/codex`, SuperGrok's
to `cli-chat-proxy.grok.com/v1`, both over the **Responses** route
(`/v1/responses` — what Codex-style clients speak), not Chat Completions.
Folding them into `openai` and `xai` would make `OPENAI_API_KEY` and a
ChatGPT login fight over one id and one base URL, so they are
`openai-chatgpt` and `xai-supergrok`: each keeps meaning one thing, and
`openai` is untouched. They list `openai`'s and `xai`'s models without
prices, since a subscription is not billed per token. They exist only in a
daemon that can use a login (any `llmman serve`). Bodies are forwarded as
sent: the ChatGPT backend is the Codex one and expects a Codex-style
Responses request (litellm fills in default instructions and streams for
clients that send neither), so use them from clients that speak that, such
as Codex.

`github-copilot` is in the catalog under its own id too, but only on such a
daemon: models.dev advertises a plain `GITHUB_TOKEN`, which Copilot's API
rejects. It sends the editor headers Copilot expects. There is no second,
Copilot-specific token: OpenClaw's `runtime-auth.ts` records that the old
`/v2/token` exchange was retired, and that the GitHub token is sent as is.

### Importing another CLI's login

Claude Code and Codex rotate their refresh tokens when they use them. A
copy in llmman's store and the original in the CLI race, and whichever
refreshes second finds its token spent and is logged out; OpenClaw calls
the single place that avoids this the "token sink". So an `import-*`
**reads the other CLI's file on every request and never writes it**
(`cli_login` in `llmman providers`). It works while the CLI keeps its
access token current; when it has lapsed, the request fails and says to run
that CLI, or to log in again. `--copy` moves the login into llmman's store
instead, with its refresh token, and llmman then renews it; it prints that
the other CLI may need to log in again, and that is on you. Nothing is
copied without the flag.

An Anthropic subscription token (`sk-ant-oat…`) is sent as
`Authorization: Bearer` with `anthropic-beta: oauth-2025-04-20`, not as
`x-api-key`. Whether Anthropic permits a subscription token outside Claude
Code is their policy to state and to change; an API key is the safe choice.

### Where logins are kept

`~/.config/llmman/credentials.json`, mode `0600` in a `0700` directory. A
file group or other can read is ignored with a warning (as a loose
`llmman.conf` key is) and never written over; one that does not parse is
left alone, too. Writes go through a temporary file and a rename. A
refresh takes a lock on the file, so the daemon and a CLI cannot both
spend one rotating refresh token, and concurrent requests make one refresh,
not many. An access token is renewed a minute before it expires. If the
server refuses the refresh token, requests fail with a 401 that says to run
`llmman login --provider <id>` again — a stored login is never quietly
replaced by a lesser credential. A temporary network failure keeps using
the current token while it is still valid.

Nothing prints a secret: the credentials' `Debug` is redacted, headers are
sensitive, and the daemon's API reports only kind and state
(`GET /llmman/providers`, [api.md](api.md#llmmans-own-api)).

### What a login may be sent to

A subscription login is only sent to its own https host: ChatGPT's to
`chatgpt.com`, SuperGrok's to `cli-chat-proxy.grok.com`, Copilot's to
`*.githubcopilot.com` (the host `copilot_internal/user` named at login,
checked again on every request), Anthropic's to `api.anthropic.com`,
Chutes' to `llm.chutes.ai`, GitLab's to the instance it was made on (or
`gitlab.com`'s hosts), Snowflake's to its account's host, MiniMax's to its
API hosts. (OpenRouter's and Poe's end in an ordinary API key, and
DigitalOcean's in a token for the API its catalog entry names; those go
wherever a pasted key would.) If
`llmman.conf` points the provider anywhere else the request is refused
rather than sent. Login and refresh requests themselves are https only and
follow no redirects, and the daemon follows a redirect from a provider only
within the same origin, so a key in a header like `x-api-key` cannot be
carried to another host. The placeholder key (`llmman`) integrations send
is never forwarded, and a daemon bound beyond loopback spends its own
credential — a login included — only for a caller that authenticated.

## Cloud credentials

Some providers have no key to paste: the credential is what an AWS, Google
or Azure account hands out, or a token minted from a client secret. They
stand behind a **chain** that is the last credential a request looks for,
after the provider's key variable, a stored login, `api_key` in
`llmman.conf` and `api_key_command` / `api_key_file`. Each request resolves
it (off the async runtime, once per expiry: tokens are cached until a
minute before they expire, temporary AWS credentials until five), sends it
over https only, follows no redirect to another origin, and prints none of
it. `llmman providers` shows the kind — `aws_sigv4`, `adc`, `entra`,
`client_credentials`, `jwt_bearer`, `ibm_iam` — and `ready`, which means
the machine says the chain is there (an environment variable, a profile, a
credentials file, a stored selection, or a table in `llmman.conf`); nothing
is asked of the cloud to report it.

A chain can be written in `llmman.conf`
([configuration.md](configuration.md#cloud-credential-chains)) or selected
by `llmman login --provider <id>` (the selection is stored, never a token).

### Amazon Bedrock

`amazon-bedrock` is offered (from models.dev's `amazon-bedrock` entry) on
Bedrock's OpenAI-compatible Chat Completions route,
`https://bedrock-runtime.<region>.amazonaws.com/openai/v1` (the AWS guide's
recommended endpoint; `.amazonaws.com.cn` in China and the other
partitions' suffixes as LiteLLM maps them). The region is `AWS_REGION`,
`AWS_DEFAULT_REGION`, the profile's `region`, else `us-east-1`, and it is
also the region requests are signed for. Its models are the ones that route
serves — `openai.*` and `xai.*` — since LiteLLM documents "Grok, gpt-oss and
GPT 5.6 and newer" as native there. Credentials, first match wins:

1. `AWS_BEARER_TOKEN_BEDROCK`, a Bedrock API key, as a bearer (also `--method
   key`). It wins over SigV4, as in the AWS SDK.
2. **SigV4**, each request signed over its exact body bytes, with the SDK's
   own provider chain: `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` /
   `AWS_SESSION_TOKEN`; the shared files (`AWS_PROFILE`,
   `AWS_SHARED_CREDENTIALS_FILE`, `AWS_CONFIG_FILE`) with static keys,
   `role_arn` + `source_profile`, `credential_process` and SSO where a
   cached `aws sso login` token exists; web identity
   (`AWS_WEB_IDENTITY_TOKEN_FILE` + `AWS_ROLE_ARN`,
   `AssumeRoleWithWebIdentity`); container credentials
   (`AWS_CONTAINER_CREDENTIALS_RELATIVE_URI` / `_FULL_URI`, with
   `AWS_CONTAINER_AUTHORIZATION_TOKEN[_FILE]`; a full URI must be https or
   loopback / `169.254.170.2` / `169.254.170.23` / `fd00:ec2::23`); EC2 IMDSv2
   (`AWS_EC2_METADATA_SERVICE_ENDPOINT`, `AWS_EC2_METADATA_DISABLED`). A
   named profile (`aws = { profile = ... }` or `aws-profile`) is used alone.

The SigV4 chain is tried when something hints at it — one of those
variables, a `[default]` profile in the shared files, an EC2 instance, a
stored selection or an `aws` table — so a machine with no AWS does not pay
an IMDS timeout per request. A signed request signs `host`, `content-type`,
`x-amz-date` and `x-amz-security-token`, with the payload hash of the bytes
sent; a streamed body is refused rather than sent unsigned. The Mantle
endpoint (`https://bedrock-mantle.<region>.api.aws/v1`, signing name
`bedrock-mantle` per opencode; LiteLLM signs it as `bedrock`, so
`aws = { service = ... }` sets either) is not in the catalog: define it with
`[providers.amazon-bedrock-mantle] base_url = "..."` and it takes the same
credentials. **Not supported:** the Converse API (an AWS-specific body and a
binary event stream, neither of llmman's two wires) and so the models that
are Converse-only.

### Google Vertex AI

`google-vertex` is offered, from models.dev's `google-vertex` entry, on the
OpenAI-compatible endpoint
`https://<host>/v1/projects/<project>/locations/<location>/endpoints/openapi`,
where `<host>` is `aiplatform.googleapis.com` for `global`,
`aiplatform.us.rep.googleapis.com` / `aiplatform.eu.rep.googleapis.com` for
`us` / `eu`, and `<region>-aiplatform.googleapis.com` otherwise. The
project is `GOOGLE_VERTEX_PROJECT`, `GOOGLE_CLOUD_PROJECT`, `GCP_PROJECT`,
`GCLOUD_PROJECT`, the active `gcloud` configuration's `project`, then the
credentials file's `project_id` / `quota_project_id`; without one the
provider is absent. The location is `GOOGLE_VERTEX_LOCATION`,
`GOOGLE_CLOUD_LOCATION`, `VERTEX_LOCATION`, else `global`. Google's own
models are sent as `google/<id>` (the endpoint's spelling); `publisher/model-maas`
ids go as listed.

Credentials are Application Default Credentials, as a bearer for the
`cloud-platform` scope: the file `GOOGLE_APPLICATION_CREDENTIALS` names — a
**service account** key (a signed RS256 JWT exchanged at its `token_uri`,
which must be a `googleapis.com` https URL) or an **authorized user** file
(its refresh token exchanged at `oauth2.googleapis.com/token`) — else
`application_default_credentials.json` in `CLOUDSDK_CONFIG` or
`~/.config/gcloud`, else the **metadata server** of a GCE VM or GKE pod
(`GCE_METADATA_HOST`, else `metadata.google.internal`, with
`Metadata-Flavor: Google`). Key files must be owner-only. **Not
supported:** `external_account` (workload identity federation) files, which
are refused with a message saying so; the **express-mode API key**
(`GOOGLE_VERTEX_API_KEY`), because Vertex's OpenAI-compatible endpoint
"only supports Google Cloud Auth" (Google's guide; opencode's Chat route
rejects an API key too) and the key works only on the native Gemini routes,
which are not one of llmman's wires; and **Claude and Mistral on Vertex**
(`google-vertex-anthropic`, `claude-*`, `mistral-*`), which are called by
`:rawPredict` on a model-specific URL with the model out of the body.

### Azure OpenAI and AI Foundry

The `azure` provider takes `AZURE_API_KEY` as `api-key` against
`https://<AZURE_RESOURCE_NAME>.openai.azure.com/openai/v1?api-version=v1`
(see [Auth schemes](#auth-schemes-and-per-account-urls); covered by the
provider tests). A Foundry or other per-resource endpoint is a
`[providers.<id>] base_url` with `auth_header = "api-key"` and the same
chain. Without a key, a Microsoft Entra ID token for
`https://cognitiveservices.azure.com/.default` (`AZURE_SCOPE`, or `scope`)
is sent as a bearer, from the first of:

1. a client secret: `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`,
   `AZURE_CLIENT_SECRET` (`client_credentials` at
   `<AZURE_AUTHORITY_HOST>/<tenant>/oauth2/v2.0/token`);
2. a federated / workload identity token: `AZURE_FEDERATED_TOKEN_FILE` with
   the same tenant and client (`client_assertion`);
3. a managed identity: App Service / Functions' `IDENTITY_ENDPOINT` +
   `IDENTITY_HEADER`, else the VM's IMDS at `169.254.169.254`
   (`AZURE_CLIENT_ID` picks a user-assigned identity);
4. the Azure CLI: `az account get-access-token --scope <scope> --output
   json`, run directly (not through a shell) with a 15-second limit.

It is tried when its inputs are set, `az login` has left a profile, or an
`azure` table says so. Metadata services are asked over plain http only at
loopback / link-local addresses, with no proxy.

### Token services: Databricks, watsonx, SAP AI Core and your own

One mechanism serves them all: a token endpoint, a grant and a secret.

| Provider | Credential |
|----------|------------|
| `databricks` (offered when `DATABRICKS_HOST` is set) | A personal access token, `DATABRICKS_TOKEN`; or OAuth M2M: `DATABRICKS_CLIENT_ID` + `DATABRICKS_CLIENT_SECRET`, `client_credentials` with Basic authentication and `scope=all-apis` at `https://<host>/oidc/v1/token` (LiteLLM). |
| `snowflake-cortex` (offered when `SNOWFLAKE_ACCOUNT`, or a login's account, is set) | A programmatic access token, `SNOWFLAKE_CORTEX_PAT`; or the OAuth login above. |
| `cloudflare-workers-ai` | `CLOUDFLARE_ACCOUNT_ID` fills the URL, `CLOUDFLARE_API_KEY` is the token: verified through the per-account URL tests. Cloudflare AI Gateway is not in the catalog (its Anthropic route uses ids that differ from models.dev's); define it with `[providers.cf-gateway] base_url = "https://gateway.ai.cloudflare.com/v1/<account>/<gateway>/anthropic"`, `wire = "anthropic"`. |
| `watsonx` | `WATSONX_API_KEY` exchanged at `https://iam.cloud.ibm.com/identity/token` (`grant_type=urn:ibm:params:oauth:grant-type:apikey`). |
| `sap-ai-core` | `AICORE_CLIENT_ID`, `AICORE_CLIENT_SECRET`, `AICORE_AUTH_URL` (the service key's `url`; `/oauth/token` is appended), or `--method service-key`. A service key that authenticates with a client certificate (`certurl`) is refused. |
| any `[providers.<id>]` | `oauth2 = {...}` (client credentials) or `jwt_bearer = {...}` (RS256 / ES256 assertion) in `llmman.conf`, or `--method client-credentials` / `jwt-bearer`. |

watsonx and SAP AI Core are **credentials only**: their inference APIs
(`/ml/v1/text/chat` with `model_id` and `project_id`; the orchestration
service's `v2/completion`) are neither OpenAI nor Anthropic wires, so they
are not offered from the catalog. The same credentials work for a provider
you define at an OpenAI-compatible endpoint in front of them
(`[providers.watsonx] base_url = "..."`). GitLab Duo's catalog entry
(`gitlab-ai-provider`) is likewise not a wire llmman speaks: `login --provider
gitlab` works for a provider you define, with `base_url` on the instance.

**OCI request signing is not supported.** OCI signs requests with an RSA
HTTP signature (litellm `llms/oci/common_utils.py`), but the routes that
take it (`/20231130/actions/chat`) use OCI's own body format; the
OpenAI-compatible `oci` provider takes an API key (`OCI_GENAI_API_KEY`) and
already works.

## Auth schemes and per-account URLs

By default a key goes as `Authorization: Bearer` (OpenAI wire) or
`x-api-key` (Anthropic wire). Other providers want it elsewhere, and
`[providers.<id>]` — on a catalog provider or one you define — says where:

```toml
[providers.azure]
auth_header = "api-key"                 # or: auth = "header" + auth_header
vars        = { AZURE_RESOURCE_NAME = "my-resource" }   # fills the base URL
query       = { api-version = "v1" }

[providers.openrouter]
headers = { HTTP-Referer = "https://example.com", X-Title = "my app" }

[providers.gemini-gateway]
base_url    = "https://gateway.example/v1"
auth_header = "x-goog-api-key"
```

| Field | Meaning |
|-------|---------|
| `auth` | `bearer`, `x-api-key`, `header` or `query`; absent means the wire's own. |
| `auth_header` | The header's name; alone it implies `auth = "header"`. |
| `auth_query` | The query parameter's name; alone it implies `auth = "query"`. A key in a URL is logged by whatever sits on the path, so prefer a header. The daemon removes it from error messages it returns. |
| `headers` | Extra headers on every request (`{ Name = "value" }`). Values are never printed. Framing headers (`Host`, `Content-Length`, …) are refused. |
| `query` | Extra query parameters on every request. |
| `vars` | Values for a catalog provider's `${VAR}` base URL; an environment variable of the same name wins. |
| `api_key_command` / `api_key_file` | Where to get the key when `api_key` and the login have none, as in `--method command` / `token-file`. Read only from an owner-only `llmman.conf`. |

Azure OpenAI is built in on those terms: `api-key`, `?api-version=v1`, and
`https://<AZURE_RESOURCE_NAME>.openai.azure.com/openai/v1` (opencode's
`azure.ts`), so `AZURE_RESOURCE_NAME` and `AZURE_API_KEY` in the
environment are enough.

models.dev lists providers whose URL is per account — Cloudflare Workers AI
(`CLOUDFLARE_ACCOUNT_ID`), Snowflake Cortex (`SNOWFLAKE_ACCOUNT`),
Databricks (`DATABRICKS_HOST`), Infomaniak, Neon, Azure — plus a variable
that is the key. They are offered when every `${VAR}` has a value, from the
environment or `vars`, and absent otherwise. A value is checked against
where it lands (a host label, one path segment, or a whole https URL) and a
provider whose value would change the host the key goes to is dropped. A
variable that stands for a host may carry its `https://` and trailing `/`.

## Your own endpoints

A provider models.dev has never heard of — vLLM or llama-server on a
box down the hall, LM Studio on a laptop, a proxy in front of OpenAI —
is defined in `llmman.conf` by giving a `[providers.<id>]` a
`base_url`:

```toml
[providers.inferencebox]
base_url = "http://inferencebox:8000/v1"
```

```console
$ llmman config set providers.inferencebox.base_url http://inferencebox:8000/v1
$ llmman providers | grep inferencebox
inferencebox    inferencebox    -          none needed    -
$ llmman list --provider inferencebox          # asks the box's own /models
$ llmman launch opencode --provider inferencebox --model qwen3-coder
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
| `auth`, `auth_header`, `auth_query`, `headers`, `query`, `api_key_command`, `api_key_file` | How the credential is attached, and where else it can come from. See [Auth schemes](#auth-schemes-and-per-account-urls). These also apply to a catalog provider. |

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
   goes to the provider. The budget is four bytes per token of the window
   the local half actually loaded with — the same window `llmman launch`
   declares to the integration, so what an agent is told it can send and
   what stays on this machine agree. Until that half is loaded, the
   daemon's context size (`LLMMAN_CONTEXT_LENGTH`) stands in.
   `LLMMAN_HYBRID_LOCAL_BYTES` sets the budget directly and is never
   overridden by a load, `0` turns the rule off. A request that declares
   no `Content-Length` stays local.
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
| `pi` | Pi coding agent | yes, but the daemon holds the key (below) |
| `aider` | Aider | yes |
| `qwen` | Qwen Code | yes |
| `dsh` | DeepSeek Harness | yes |
| `goose` | Block goose | yes |
| `goose-desktop` | Block goose Desktop (`OPENAI_TRANSCRIPTION_MODEL` alone starts it for dictation only) | yes |
| `grok` | Grok Build (offered the first 300 models: it sends its list with every request) | yes |
| `docker-agent` | Docker Agent | yes |
| `vibe` | Mistral Vibe CLI | yes |
| `muse` | Meta Muse Code (also `muse-code`; its session, `resume` and `exec`; run its other subcommands with `muse` directly) | yes |
| `hermes` | Hermes Agent | yes, but the daemon holds the key (below) |
| `agy` | Antigravity CLI | yes |
| `gemini` | Gemini CLI (one model: `/model` does not change it) | yes: its model calls go to `llmman serve`, and it holds a session token or the daemon's own key, never a provider key |
| `cline` | Cline | yes |
| `kimi` | Kimi Code CLI (the end-of-life Python kimi-cli takes pulled models only) | yes |
| `copilot` | GitHub Copilot CLI (`copilot`, BYOK; one model: its list is ignored) | yes: it holds a session token, never a provider key |
| `openclaw` | OpenClaw | yes, but the daemon holds the key (below) |
| `omp` | OMP coding agent (pulled models only) | no: it stops at startup on a model id that names two vendors, such as `llmman.provider/amazon-bedrock/openai.gpt-5.4` |

Any extra arguments after `--` are forwarded to the integration's own CLI.

`hermes`, `cline`, `pi` and `openclaw` are configured through a file
on disk, so they can't carry a key per request; `llmman serve` needs one of its own, spent only for a
loopback daemon and never for a cross-site browser request. On a shared
machine prefer an integration that sends its own key.

`codex` speaks only OpenAI's Responses API, which most providers lack
(`mistral` 404s it, `opencode` 500s it for non-OpenAI models). The
daemon tries the provider first and, on a 404/405/501 (or any 5xx from
`opencode`), translates the request to a chat completion and the reply
back, tool calls included. Providers that have the API (`openai`, `groq`,
`openrouter`) are used natively; any other error is relayed as-is.

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
  else `none`, `low`, `medium`, `high`, plus the `--variant` asked for.
- `claude`: Claude Code's `/effort <low|medium|high|xhigh|max>`, sent as
  spelled; a level the template rejects is a 400.
- `codex`: `model_reasoning_effort`, e.g. `-- -c
  model_reasoning_effort=high`; its `/model` picker lists only OpenAI's
  catalog.

`--variant`, as on opencode's `run`, picks one of those up front:
`llmman run qwen3.8 --variant xhigh` sends it as is; `llmman launch`
first checks the model has it, then hands it over in the integration's
own form, still changeable from inside. A model with no known levels,
such as one models.dev does not list yet, takes any effort level.

| Integration | As | Takes |
|---|---|---|
| `opencode` | the model's default options and its saved variant | every variant |
| `claude`, `copilot` | `--effort` | `low` to `max` |
| `codex` | `-c model_reasoning_effort=` | all |
| `pi`, `omp` | `--thinking` | `none`, `minimal` to `high` (omp: `xhigh`) |
| `cline` | `--thinking` | `none`, `low` to `high` |
| `aider` | `--reasoning-effort` | all |
| `hermes` | `--reasoning` (not with `--tui`) | all |
| `grok` | `--effort` | all |
| `qwen` | its model entry | all but `minimal` |
| `dsh` | its route | all |
| `muse` | `--reasoning-effort` | all but `none` |

`thinking` goes to all but opencode as `medium`, which llmman serves as
thinking on. `kimi`, `openclaw`, `gemini`, `agy`, `goose`,
`goose-desktop`, `docker-agent` and `vibe` cannot carry a variant here, so they
refuse one.

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

Requests to `opencode-go` also carry `x-opencode-session` (the caller's own
header value if present, otherwise a stable per-daemon id) and
`User-Agent: llmman/<version>`, because its gateway rejects requests that
omit the session header.

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
(`minimal` as `low`); `none` turns thinking off (`disabled`, or
`between_tools` on Sonnet 5 and later, which refuse `disabled`), or on
Fable and Mythos, which always think, is `low`. Through Claude 4.5, and on another
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

llmman resolves the fixed provider hostname with the system resolver and
verifies its TLS hostname; egress policy is the network layer's. A legacy
`X-LLMMan-Upstream-IP` header is ignored. There is no caller-controlled URL.
Redirects and environment proxy/CA overrides are not used.

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

## How this compares to OpenClaw, opencode and LiteLLM

What the three reference projects accept for a hosted provider, and where
llmman stands. The reference code each row was checked against is cited in
the source of the module that implements it ([`src/logins/`](../src/logins/mod.rs),
[`src/credentials.rs`](../src/credentials.rs),
[`src/authscheme.rs`](../src/authscheme.rs)).

| Login type | llmman |
|------------|--------|
| Static API key (env, `llmman.conf`, pasted) | Yes. |
| OAuth device code with refresh — ChatGPT/Codex, GitHub Copilot, xAI | Yes (`device`). |
| OAuth authorization code + PKCE with a loopback callback | Yes: OpenRouter, Poe, Chutes (your OAuth app), GitLab Duo, Snowflake Cortex. DigitalOcean uses the implicit grant (token in the fragment, forwarded by the listener's page). MiniMax uses its user-code device flow. OpenAI's own browser flow is not implemented (the ChatGPT device code is). |
| Pasted long-lived token (`claude setup-token`) | Yes. |
| Reuse of Claude Code's / Codex's login file | Yes, read-only by default; `--copy` to take it over. macOS Keychain storage is not read. |
| Refresh persisted atomically, `0600`, single-flight, 60 s skew, re-login error | Yes. |
| Credential from a command or a file | Yes (`api_key_command`, `api_key_file`, `--method command`/`token-file`). LiteLLM's secret managers and `os.environ/` references are not reproduced; a command covers `pass`, `op`, `vault` and the cloud CLIs. |
| Custom header / query scheme, extra headers, per-account URLs | Yes. |
| GitHub Enterprise (data-residency) Copilot | No. |
| AWS Bedrock: `AWS_BEARER_TOKEN_BEDROCK`; SigV4 over env, shared files (profile, `role_arn` + `source_profile`, SSO, `credential_process`), web identity, container and IMDSv2 | Yes ([Amazon Bedrock](#amazon-bedrock)), on the OpenAI-compatible routes. The Converse API is not supported. |
| Google Vertex: service-account JWT, `gcloud` authorized-user file (`CLOUDSDK_CONFIG`), metadata server, project / location variables | Yes ([Google Vertex AI](#google-vertex-ai)). Express-mode API keys, `external_account` files and Claude / Mistral on Vertex are not supported. |
| Azure: `api-key` + per-resource URL + `api-version` | Yes. |
| Azure Entra ID: client secret, federated token file, managed identity (IMDS, App Service), `az` CLI | Yes ([Azure](#azure-openai-and-ai-foundry)). Certificate and username/password credentials (LiteLLM) are not. |
| watsonx IAM API-key exchange; SAP AI Core client credentials | The credentials, yes; the providers' own APIs are not OpenAI or Anthropic wires, so they are reachable only through an endpoint you define. |
| Databricks PAT and OAuth M2M | Yes. |
| Snowflake Cortex PAT and OAuth | Yes. |
| Cloudflare Workers AI / AI Gateway | Workers AI yes (per-account URL variable); AI Gateway by defining its URL. |
| Generic OAuth 2.0 client credentials and JWT-bearer sources for providers not built in | Yes (`oauth2`, `jwt_bearer`, `--method client-credentials` / `jwt-bearer`). |
| OCI request signing | Not supported (see above). |
| LiteLLM's Azure username/password and certificate credentials; Google `external_account` (workload identity federation) files | Not supported. |
