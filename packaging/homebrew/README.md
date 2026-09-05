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
