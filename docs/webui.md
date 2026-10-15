# Web UI

`llmman serve` serves a web UI at `/` (`http://127.0.0.1:17434/` by
default). It is built into the binary — no separate install, nothing
fetched from the network — and talks to the daemon over the same HTTP API
every other client uses, so anything it shows is something `llmman`'s
CLI or an Ollama/OpenAI client could ask for too.

A browser asking for any other path outside the daemon's APIs gets the
same page, so a UI route can be reloaded or linked to. The API prefixes
(`/api`, `/v1`, `/llmman`, `/ui`, `/gemini`, `/audio`, `/props`,
`/metrics`) never get it: an unknown path there is a `404`, as it is for
any client that does not ask for HTML.

The sidebar lists past chats and opens the other pages:

- **Chat** — a conversation with any model the daemon can reach. The
  model picker lists the local store (`/v1/models`, with which ones are
  loaded) and every hosted provider the daemon holds a usable key for
  (`/llmman/providers`; see [providers.md](providers.md)). Replies stream
  over `/v1/chat/completions`; a model's reasoning, when it emits any,
  is shown collapsed above the answer. Images can be attached to a
  message: chosen, pasted or dropped.

  Diffusion models (`/api/show` capabilities `image`, `video` or `audio`
  rather than `completion`) are listed under *Generate* in the same
  picker. With one selected the composer generates instead, as `llmman
  run` does: an Image / Video / Audio toggle picks what the prompt
  becomes, and the settings button holds `run`'s flags — size, seconds,
  steps, seed, guidance, negative prompt; blank is the model's default.
  Images stream over `/v1/images/generations` with a step counter; a
  video is one `/v1/videos` request and then its `content_url`; audio is
  `/v1/audio/speech`. The result is the reply, with a Download button
  that names the file as `run` does. Each prompt stands alone: a
  diffusion model has no conversation. *Stop* abandons the request; a
  video or audio generation already running finishes on the daemon.
- **Models** (`/models`) — opens on popular models
  (`/llmman/search/popular`: Docker Hub's most pulled, then Hugging
  Face's most downloaded GGUF), searches Docker Hub and Hugging Face
  (`/llmman/search`, the rows `llmman search` prints), and has a *Pulled*
  tab for what this machine already has. Selecting a model opens its card
  (`/llmman/search/model`): each tag or quantization with its size, a
  rough fit against this machine's model memory (`/llmman/node`; weights
  only, not context), *Pull*, which streams `/api/pull`'s progress, and
  the repo's README. *Stop watching* only stops following a pull: the
  daemon finishes the download. The README's HTML is reduced to its
  links, headings and text, and its images are left out, so it renders
  like a reply. A model already here can be chatted with, loaded,
  unloaded or deleted. The search box also takes a full reference, with
  its registry or scheme — an OCI image, `hf.co/…`, `ms://…`, `ngc://…`,
  `s3://…`, `gs://…` — and pulls it. Owners' pictures load from Hugging
  Face and Gravatar through `/llmman/search/avatar`, the one thing the
  page fetches from outside the daemon; without one, or with an API key
  the browser cannot attach to an image request, the row shows initials.
- **Shell** (`/shell`) — a terminal on the machine running `llmman
  serve`, as the user running it: the login shell in a pty, bridged over
  a WebSocket at `/llmman/shell`. `llmman` itself is on `PATH` there, so
  `llmman launch claude --model …` or `llmman ps` work as they would in
  any terminal. Shift- or cmd/ctrl-click a URL in the output to open it
  in a new tab.
- **Providers** — the hosted providers the daemon knows, which of them
  have a usable key, and a way to set or remove one. The key is sent to
  the daemon once (`PUT`/`DELETE /llmman/providers/:id/key`), which keeps
  it in the per-user `llmman.conf`; the browser does not keep it.
- **Settings** — theme (the system's unless set), the name the greeting
  uses, whether Enter or Cmd/Ctrl+Enter sends, this daemon's API key, and
  the conversations: export as JSON (without generated media) or delete.
  *About* names the daemon's version and executable.
- **Servers** — the daemon that served the page, and any other `llmman
  serve` added by address and key; the page talks to the one selected.
  That daemon's CORS has to allow the page's origin: localhost is allowed
  by default, anything else through `LLMMAN_ORIGINS`
  ([configuration.md](configuration.md)).

Every page is a path, so it can be reloaded or linked to; old `#/models`
and `#/c/<id>` links still land where they did. Cmd/Ctrl+Shift+O starts a
new chat, `Cmd/Ctrl+\` shows or hides the sidebar, and Escape closes a
sheet.

Conversations and generated media are stored in the browser (IndexedDB),
not by the daemon. Chats kept by the earlier built-in UI on the same
origin are imported once, and its theme and settings carry over.

A daemon that requires an API key ([api.md](api.md#authentication))
serves the page itself without one — it is only files — and the page
asks for the key on its first `401` from llmman, keeps it in the
browser's local storage (scoped to the page's path, so two daemons behind
one gateway never share one), and sends it on every call thereafter; a
`401` relayed from a hosted provider is shown as that provider's error,
not as a wrong key. *Settings* shows and changes it. The shell's WebSocket
carries it as a subprotocol, since a browser cannot set a header on an
upgrade.

## The shell's guard rails

A shell endpoint is only acceptable because the daemon is, by default,
reachable only from the machine it runs on. Three checks keep it that
way:

1. **Loopback only.** When `LLMMAN_HOST` binds beyond loopback
   (`0.0.0.0`, a LAN address), the shell is off: a WebSocket upgrade gets
   `403` and the UI greys out *Shell* with the reason. Inference over the
   network behind an API key is a choice an operator can make; a shell
   is not, even behind one. On loopback, a daemon with keys requires one
   for the shell as for every other route.
2. **Same-site browsers only.** Browsers do not apply CORS to WebSockets,
   so a page on any site could otherwise open
   `ws://127.0.0.1:17434/llmman/shell` from a visitor's browser. The
   route checks the `Origin` header itself against the same patterns as
   the CORS layer — every localhost spelling plus `LLMMAN_ORIGINS`
   ([configuration.md](configuration.md)) — except patterns whose `*`
   covers the host (`*`, `https://*.example.com`), which are fine for CORS
   and refused here. A request with no `Origin` (not a browser) is
   allowed: it already had a shell to run from.
3. **`LLMMAN_SHELL=off`** removes it entirely for an operator who wants
   the UI without it. Any other value is the program to run instead of
   the login shell (`LLMMAN_SHELL="tmux new -A -s llmman"`).

`GET /llmman/shell` without a WebSocket upgrade always answers `200` with
`{"enabled": true}` or `{"enabled": false, "reason": "…"}`; only the
upgrade itself is refused with `403`. That is how the UI knows before it
tries. The page is also served with `frame-ancestors 'none'`, so another
site cannot frame it and reach the shell through the trusted origin.

### Protocol

For anyone wiring up another terminal: the client sends binary frames of
keystrokes and text frames of `{"resize":{"cols":N,"rows":N}}`; the
daemon sends binary frames of pty output and, when the program exits, a
final text frame `{"exit":CODE}` before closing. Closing the socket kills
the program.

## Working on it

The page is a React Native app (Expo, with Expo Router), whose source is
under `app/`; the iOS and Android apps are built from the same source.
What `cargo build` embeds is its web export, committed under `webui/`, so
building llmman needs no Node.js: `build.rs` gzips every file there into
the binary (`src/webui/`). CI checks that `webui/` is the export of
`app/`, so a change to `app/` is committed with a fresh export:

```
cd app
npm ci
npm run check
npm run export-web -- ../webui
```

The export has no absolute paths, so the page works under a gateway's
path prefix ([compose.md](compose.md)), and no binaries: the manatee
mark is downloaded by `build.rs` from the `docs-assets` GitHub release,
pinned by SHA-256 (`FETCHED_ASSETS`), and served as `ui/manatee.png`; an
offline build warns and ships a blank image in its place. `npm ci`
downloads the same file for the app's icons.

To iterate without rebuilding llmman, run the export into any directory
and point the daemon at it:

```
LLMMAN_WEBUI_DIR=/tmp/webui cargo run -- serve
```

Files are then read from disk on every request, uncompressed and
uncached, so an export and a reload is the whole loop; `npm start` and
`w` also serve the page from Expo's development server, which reaches
the daemon as another server would. Markdown from models is rendered by
`app/lib/markdown.ts` into plain components, never as HTML — with a shell
on the same origin, an HTML injection in a model's reply would be a
remote shell.
