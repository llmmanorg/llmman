# Docker Compose

[`examples/compose`](../examples/compose) runs `llmman serve` behind a Caddy
gateway. The same gateway exposes the built-in web UI and the Ollama, OpenAI,
and Anthropic-compatible APIs. A named volume keeps pulled models between
container replacements.

From the repository root:

```sh
docker compose -f examples/compose/compose.yaml up
```

Open <http://localhost:8080/> for the web UI. Its Shell tab is unavailable
here: the daemon binds `0.0.0.0` inside the container, and the shell is
only offered by a daemon bound to loopback (see [webui.md](webui.md)).
Clients can use the same address as their API base URL. For example:

```sh
curl http://localhost:8080/api/version
```

The daemon binds `0.0.0.0` inside the container, which it only does with
API keys or `LLMMAN_AUTH=off` ([configuration.md](configuration.md#authentication)).
The example sets `LLMMAN_AUTH=off`, leaving authentication to the
gateway, since the daemon's port is not published — only Caddy's is.
To have the daemon check keys itself, clear that and set the keys —
`LLMMAN_AUTH= LLMMAN_API_KEYS=<key> docker compose ... up`; the web UI
then asks for one.

The service runs the published `ai/llmman:server` image
([backends.md](backends.md#in-a-container)). Pin a release, or pick a GPU
variant, with `LLMMAN_TAG`:

```sh
LLMMAN_TAG=server-0.1.400 docker compose -f examples/compose/compose.yaml up
```

The `llmman-data` volume is mounted at `/root/.local/share/llmman`, the store
and cache's default location. Remove the deployment while retaining its models
with `docker compose -f examples/compose/compose.yaml down`. Add `--volumes`
only when the stored models should be deleted as well.

## CPU limits and container backends

`LLMMAN_CPUS` controls the Compose CPU limit and defaults to `4`. The example
uses llmman's default local backend, so the `llama-server` child shares the
service's cgroup and llmman can derive its thread count from that limit.

With `--runtime docker` or `--runtime podman` the backend runs in a separate
container, a sibling of the service. The service's limit is forwarded to it as
`--cpus`, with a matching `--threads` for `llama-server`; an unconstrained
daemon starts an unconstrained container, and `LLAMA_ARG_THREADS` still wins.

## Customizing the gateway

The example only publishes Caddy's port. Add authentication and TLS to the
[`Caddyfile`](../examples/compose/Caddyfile) before exposing it outside a trusted
network, or have the daemon do both itself (`LLMMAN_API_KEYS`,
`LLMMAN_TLS_CERT`/`LLMMAN_TLS_KEY`; see [api.md](api.md#authentication)).
