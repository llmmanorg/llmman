# E2E environment

This directory holds the pre-baked environment inputs for the slow e2e job.

`.mise.toml` is the source of truth for base toolchain versions that Renovate can bump directly: Go, Rust, Node.js, Bun, and Python. `manifest.json` is the source of truth for e2e-specific tools, install sources, checksums, and platform coverage. `lock.json` records the concrete artifacts CI should consume after a successful environment build. For Linux, that artifact is an OCI manifest list with `linux/amd64` and `linux/arm64` entries. macOS and Windows stay as native bundles because GitHub hosted jobs for those platforms do not run inside Linux containers.

The expected update flow is:

1. Change `.mise.toml`, `manifest.json`, or one of the root release pin files.
2. Run `python3 ci/e2e/environment/validate.py`.
3. Run the environment workflow.
4. Copy the published OCI index digest and per-platform digests into `lock.json`.
5. Replace the zero digest in `.github/workflows/ci.yml` with the published OCI index digest, preserving the `image:tag@sha256:...` form.
6. Move the e2e job to the new lock entry only after smoke checks pass.

The e2e job should still build and install the current checkout. This environment pre-bakes third-party tools and engine prerequisites; it is not a replacement for testing the current `llmman` binary or installer.
