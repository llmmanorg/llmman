#!/usr/bin/env bash
set -euo pipefail

required=(
  bash
  bun
  cargo
  clang
  claude
  cline
  cmake
  codex
  curl
  dsh
  free
  gcc
  git
  go
  node
  npm
  omp
  openclaw
  opencode
  pgrep
  pi
  ps
  python3
  qwen
  rustc
  sudo
)

missing=()
for bin in "${required[@]}"; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    missing+=("$bin")
  fi
done

if [ "${#missing[@]}" -gt 0 ]; then
  printf 'missing required e2e tools: %s\n' "${missing[*]}" >&2
  exit 1
fi

# go has no --version flag; it reports its version via the `version` subcommand.
go version
for bin in cargo rustc node npm bun python3 omp claude opencode codex cline pi qwen openclaw dsh; do
  "$bin" --version
done
