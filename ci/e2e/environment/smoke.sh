#!/usr/bin/env bash
set -euo pipefail

required=(
  bash
  bun
  cargo
  clang
  cmake
  codex
  curl
  gcc
  git
  node
  npm
  omp
  opencode
  python3
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

node --version
npm --version
bun --version
