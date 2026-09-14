# llmmanorg/homebrew-tap

Homebrew tap for [llmman](https://github.com/llmmanorg/llmman) — run any
agent on any model, models stored as OCI images.

## Install

```sh
brew install llmmanorg/tap/llmman
```

The fully-qualified name does the `brew tap` for you and trusts just this
formula (Homebrew does not load formulae from third-party taps otherwise).

Supported platforms (the platforms llmman publishes builds for):

| Platform | Architecture |
|---|---|
| macOS | Apple Silicon (`arm64`) |
| Linux | `x86_64` |
| Linux | `aarch64` |

Intel macOS is not supported — llmman publishes no `x86_64-apple-darwin`
build. `cargo install llmman` builds from source there (needs Rust and Go).

## Run as a service

The CLI starts the daemon on demand; to keep it running across logins:

```sh
brew services start llmman
```

This runs `llmman serve` under launchd or systemd, restarts it if it exits,
and logs to `$(brew --prefix)/var/log/llmman.log`. It listens on
`http://127.0.0.1:17434`; to change that, put `LLMMAN_HOST=...` (plus
`LLMMAN_API_KEYS=...` for a non-loopback bind) in
`~/.homebrew/services/llmman.env` (`$XDG_CONFIG_HOME/homebrew/...` if
set), restart the service, and export the same variables in your shell so
the CLI talks to it rather than starting its own daemon.

Run `brew services restart llmman` after `brew upgrade llmman`. Homebrew
does not restart services on upgrade, and the CLI replaces a daemon whose
version differs from its own, which leaves the service unable to bind
until it is restarted.

## Versions

Every commit that passes CI on llmman's `main` is a release, versioned
`MAJOR.MINOR.<commit count>` (e.g. `0.1.324`), and this formula is updated
to it within minutes. `brew upgrade llmman` therefore tracks `main`; there
is no separate stable channel.

## This formula is generated

`Formula/llmman.rb` is rendered by
[`packaging/render.sh`](https://github.com/llmmanorg/llmman/blob/main/packaging/render.sh)
in the main repo and pushed here by its CI on every release. **Edits made
directly in this repo are overwritten by the next release** — change
`packaging/homebrew/llmman.rb.in` upstream instead.
