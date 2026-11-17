#!/usr/bin/env bash
# Prints the llama.cpp b<N> release (the one with the binaries) behind the
# repo-root LLAMA_CPP_RELEASE pin, which is the tag of a ghcr server image:
#
#   v1.0.0-b12359-b1b770d  ->  b12359 (the tag names its build)
#   v1.1.0                 ->  what that release's nightly-tag.txt names
set -euo pipefail

pin="$(tr -d '[:space:]' <"$(dirname "$0")/../LLAMA_CPP_RELEASE")"
case "$pin" in
v*-b*-*)
	build="${pin#*-}"
	echo "${build%%-*}"
	;;
v*)
	curl -fsSL --retry 8 \
		"https://github.com/ggml-org/llama.cpp/releases/download/$pin/nightly-tag.txt" |
		tr -d '[:space:]'
	echo
	;;
*) echo "$pin" ;;
esac
