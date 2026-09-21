# llmmanorg/scoop-bucket

Scoop bucket for [llmman](https://github.com/llmmanorg/llmman) — run any
agent on any model, models stored as OCI images.

## Install

```powershell
scoop bucket add llmman https://github.com/llmmanorg/scoop-bucket
scoop install llmman
```

Supported platforms (the platforms llmman publishes Windows builds for):

| Platform | Architecture |
|---|---|
| Windows | `x64` |
| Windows | `arm64` |

The binaries link against the Visual C++ runtime (`VCRUNTIME140.dll`). It
is present on almost every Windows machine; if `llmman` fails to start
because it is missing, `scoop install extras/vcredist2022` provides it
(after `scoop bucket add extras`).

## Versions

Every commit that passes CI on llmman's `main` is a release, versioned
`MAJOR.MINOR.<commit count>` (e.g. `0.1.324`), and this manifest is updated
to it within minutes. `scoop update llmman` therefore tracks `main`; there
is no separate stable channel.

## This manifest is generated

`bucket/llmman.json` is rendered by
[`packaging/render.sh`](https://github.com/llmmanorg/llmman/blob/main/packaging/render.sh)
in the main repo and pushed here by its CI on every release. **Edits made
directly in this repo are overwritten by the next release** — change
`packaging/scoop/bucket/llmman.json.in` upstream instead.
