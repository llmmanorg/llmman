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

## Desktop app

A tray icon that opens the web UI and keeps `llmman serve` running (see
[desktop/README.md](https://github.com/llmmanorg/llmman/blob/main/desktop/README.md)).
It is a formula on Linux and a cask on macOS, with the same version as
`llmman`:

```sh
brew install llmmanorg/tap/llmman-desktop          # Linux
brew install --cask llmmanorg/tap/llmman-desktop   # macOS, Apple Silicon
```

Launch llmman from the application menu on Linux, from Launchpad or the
Applications folder on macOS. The app is not notarized, so the cask removes its
quarantine attribute; the .dmg download still needs right-click, Open. The Linux formula links its `.desktop` file into
`${XDG_DATA_HOME:-~/.local/share}/applications`. Newer Homebrews sandbox that
step; the install output then gives the one `ln` command to run. `brew
uninstall` leaves the link behind, so remove it too.

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

`brew upgrade llmman` does not touch the running service, which keeps the old
build until `brew services restart llmman` or a reboot.

## Versions

Every commit that passes CI on llmman's `main` is a release, versioned
`MAJOR.MINOR.<commit count>` (e.g. `0.1.324`), and the tap is updated
to it within minutes. `brew upgrade llmman` therefore tracks `main`; there
is no separate stable channel.

## These files are generated

`Formula/` and `Casks/` are rendered by
[`packaging/render.sh`](https://github.com/llmmanorg/llmman/blob/main/packaging/render.sh)
in the main repo and pushed here by its CI on every release. **Edits made
directly in this repo are overwritten by the next release** — change
the `.rb.in` templates under `packaging/homebrew/` upstream instead.
