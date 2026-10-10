# E2E environment

This directory holds the pre-baked environment inputs for the slow e2e job.

`.mise.toml` is the source of truth for Go, Rust, Node.js, Bun, Python, and npm CLI versions that Renovate can bump directly. Mise installs the npm CLIs into separate tool directories and exposes their binaries through its shims; Bun remains available for CLIs that need it at runtime. `manifest.json` records platform coverage and a pinned inventory of deferred native agents and engines under `deferred`, with an explicit `not-installed` status and reason. The Dockerfile is the source of truth for Linux system packages. `lock.json` records the concrete artifacts CI should consume after a successful environment build. For Linux, that artifact is an OCI manifest list with `linux/amd64` and `linux/arm64` entries. macOS and Windows stay as native bundles because GitHub-hosted jobs for those platforms do not run inside Linux containers.

Linux builds run on native AMD64 and ARM64 runners, then merge the two image digests into the published manifest list. Each architecture uses a separate persistent GHCR build cache. New main-branch workflow runs cancel older publication runs; PR validations and other branches use separate concurrency groups. System packages, mise tool installation, and smoke checks occupy separate layers so a tool pin or smoke change can reuse earlier installation work. Download and npm caches are removed in the installation layer; installed tools and mise shims remain available. `smoke.sh` checks required binaries and runs the tool version checks during every image build.

The foundation image installs Go, Rust, Node.js, Bun, Python, the nine mise-managed npm CLIs (omp, claude, opencode, codex, cline, pi, qwen, openclaw, and dsh), and the Dockerfile's Linux system packages. Native agents (goose, grok, docker-agent, muse, agy, and hermes) and inference engines (llama.cpp, vLLM, vLLM Metal, SGLang, and MLX LM) are deferred rather than preinstalled. Existing CI installation steps still provide the supported agents and engines; the deferred inventory preserves their pins for future image integration.

The expected update flow is:

1. Change `.mise.toml`, `manifest.json`, or one of the root release pin files.
2. Run `python3 ci/e2e/environment/validate.py`.
3. Run the environment workflow.
4. Copy the published tag (shown in the workflow run summary), the OCI index digest, and the per-platform digests into `lock.json`. Until then, these fields stay null; validate.py rejects placeholder values such as all-zero digests.
5. After smoke checks pass, wire the e2e job in `.github/workflows/ci.yml` to the published image in a follow-up change, using the `image:tag@sha256:...` form.

The e2e job should still build and install the current checkout. This environment pre-bakes third-party tools and engine prerequisites; it is not a replacement for testing the current `llmman` binary or installer.
