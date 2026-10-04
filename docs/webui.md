# Web UI

`llmman serve` serves a web UI at `/` (`http://127.0.0.1:17434/` by
default). It is built into the binary — no separate install, nothing
fetched from the network — and talks to the daemon over the same HTTP API
every other client uses, so anything it shows is something `llmman`'s
CLI or an Ollama/OpenAI client could ask for too.

The page has two modes, toggled at the top, and a Models page in the
sidebar:

- **Chat** — a conversation with any model the daemon can reach. The
  model picker lists the local store (`/v1/models`, with which ones are
  loaded) and every hosted provider the daemon holds a usable key for
  (`/llmman/providers`; see [providers.md](providers.md)). Replies stream
  over `/v1/chat/completions`; a model's reasoning, when it emits any,
  is shown collapsed above the answer. *Pull a model…* opens the
  Models page.

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

  A chat model's composer also has:

  - **Deep thinking.** The light bulb asks a reasoning model to think at
    length; *Chat settings* has Off, Low, Medium and High. It is sent as
    `reasoning_effort` ([api.md](api.md#openai-api-notes)), which a model
    that does not reason ignores.
  - **Pictures.** The paperclip attaches images (or paste them) and the
    camera button takes a photo, up to four to a message. Each is shrunk
    to a JPEG of at most 1280 pixels and sent as an `image_url` part, so
    a local model needs `vision` in its `/api/show` capabilities (the page
    says when it lacks it; a hosted model is assumed to). The daemon reads
    at most 2 MiB of a chat request, so the newest pictures that fit
    with the text go with it, and a message whose pictures were left out
    says so to the model. All stay in the conversation.
  - **Search the web.** The globe grounds each reply in a web search: the
    daemon asks [Exa](https://exa.ai) (needs a key, see
    [configuration.md](configuration.md#web-search)), the numbered results
    go ahead of the question for that turn only, and the reply cites them
    as `[1]`, `[2]` above a list of sources. If the search fails the
    reply goes ahead without it. The question you typed goes to Exa, the
    one thing a prompt sends off the machine. The results are untrusted
    pages and the prompt says so, which makes an injected instruction
    less likely to work, not impossible.
  - **Voice.** The microphone dictates into the composer; *Voice mode*
    (the headphones) is hands-free: what is heard is sent at the pause,
    the reply is read aloud, and the microphone opens again. Every reply
    has a *Read aloud* button. Listening uses the browser's speech
    recognition where it has one (Chrome's goes to its vendor). Without
    one, as in Firefox and the [Android app](android.md), the page
    records and has the daemon transcribe over `/v1/audio/transcriptions`,
    which needs the selected model to take audio input. The camera and
    microphone need a secure context (`127.0.0.1` or HTTPS); over plain
    HTTP on a LAN address the camera button opens the phone's camera app
    and the microphone is unavailable.
- **Models** (`#/models`) — opens on popular models
  (`/llmman/search/popular`: Docker Hub's most pulled, then Hugging
  Face's most downloaded GGUF), searches Docker Hub and Hugging Face
  (`/llmman/search`, the rows `llmman search` prints), and has a *Pulled*
  tab for what this machine already has. Selecting a model opens its card
  (`/llmman/search/model`): each tag or quantization with its size, a
  rough fit against this machine's model memory (`/llmman/node`; weights
  only, not context), *Pull*, which streams `/api/pull`'s progress, and
  the repo's README. The README's HTML is reduced to its links, headings
  and text, and its images are left out, so it renders like a reply.
  A model already here can be chatted with, unloaded or deleted from
  its card. The search box also takes a full reference, with its registry
  or scheme — an OCI image, `hf.co/…`, `ms://…`, `ngc://…`, `s3://…`,
  `gs://…` — and pulls it on Enter; anything shorter is a search. Owners' pictures load from Hugging Face and
  Gravatar through `/llmman/search/avatar`, the one thing the page
  fetches from outside the daemon; without one, or with an API key the
  browser cannot attach to an image request, the row shows initials.
- **Shell** — a terminal on the machine running `llmman serve`, as the
  user running it: the login shell in a pty, bridged over a WebSocket at
  `/llmman/shell`. `llmman` itself is on `PATH` there, so `llmman launch
  claude --model …` or `llmman ps` work as they would in any terminal.
  Shift- or cmd/ctrl-click a URL in the output to open it in a new tab.

Conversations, generated media and attached pictures are stored in the
browser (IndexedDB), not by the daemon; *Settings* can export
conversations as JSON (without media or pictures) or delete them. Theme
follows the system unless set.

A daemon that requires an API key ([api.md](api.md#authentication))
serves the page itself without one — it is only files — and the page
asks for the key on its first `401`, keeps it in the browser's local
storage, and sends it on every call thereafter; *Settings* shows and
changes it. The shell's WebSocket carries it as a subprotocol, since a
browser cannot set a header on an upgrade.

## The shell's guard rails

A shell endpoint is only acceptable because the daemon is, by default,
reachable only from the machine it runs on. Three checks keep it that
way:

1. **Loopback only.** When `LLMMAN_HOST` binds beyond loopback
   (`0.0.0.0`, a LAN address), the shell is off: a WebSocket upgrade gets
   `403` and the UI greys out the tab with the reason. Inference over the
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

The UI is plain ES modules and CSS under `webui/` with no build step —
`cargo build` gzips them into the binary (`build.rs`, `src/webui/`). The
only third-party code is xterm.js (with its fit, WebGL, Unicode 11 and
web-links addons), vendored under `webui/vendor/` with its versions recorded in
`webui/vendor/VERSIONS`. No binaries live in the repository: the manatee
mark is downloaded by `build.rs` from the `docs-assets` GitHub release,
pinned by SHA-256 (`FETCHED_ASSETS`); an offline build warns and ships a
blank image in its place.

To iterate without rebuilding, point the daemon at the directory:

```
LLMMAN_WEBUI_DIR=webui cargo run -- serve
```

Files are then read from disk on every request, uncompressed and
uncached, so a save and a reload is the whole loop. Markdown from models
is rendered by `webui/markdown.js`, which builds DOM nodes directly and
never sets `innerHTML` — with a shell on the same origin, an HTML
injection in a model's reply would be a remote shell.
