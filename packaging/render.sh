#!/usr/bin/env bash
# Render the Homebrew formula and scoop manifest for one llmman release.
# Used by the publish-packages CI job; standalone so it can be run locally:
#
#   packaging/render.sh --version 0.1.324 --checksums checksums.txt --out-dir /tmp/out
#
# packaging/<pkg>/ mirrors the repo it is pushed to (*.in are rendered,
# the rest copied), so <out>/homebrew -> llmmanorg/homebrew-tap and
# <out>/scoop -> llmmanorg/scoop-bucket.
#
# scoop notes (JSON has no comments): the `#/llmman.exe` URL fragment
# names the downloaded exe for `bin` to shim. VCRUNTIME140.dll is a
# `suggest` for extras/vcredist2022, not `depends`, which aborts the
# install without the extras bucket. checkver/autoupdate serve scoop's own
# tooling; CI pushes every release directly.

set -euo pipefail

die() {
	printf 'render.sh: %s\n' "$@" >&2
	exit 1
}

usage() {
	cat >&2 <<-EOF
		usage: render.sh --version <x.y.z> --checksums <file> --out-dir <dir>
		                 [--repo <owner/repo>] [--tag <tag>]

		  --version    release version, as printed by packaging/version.sh
		  --checksums  sha256sum-format file covering the release's assets
		  --out-dir    directory to write homebrew/ and scoop/ into
		  --repo       GitHub repo the download URLs point at (default: llmmanorg/llmman)
		  --tag        release tag the assets live under (default: v<version>;
		               only differs for releases cut before the tag scheme changed, e.g. b321)
	EOF
	exit 2
}

VERSION=""
CHECKSUMS=""
OUT_DIR=""
REPO="llmmanorg/llmman"
TAG=""

while [ $# -gt 0 ]; do
	case "$1" in
	--version) VERSION="${2:-}"; shift 2 ;;
	--checksums) CHECKSUMS="${2:-}"; shift 2 ;;
	--out-dir) OUT_DIR="${2:-}"; shift 2 ;;
	--repo) REPO="${2:-}"; shift 2 ;;
	--tag) TAG="${2:-}"; shift 2 ;;
	-h | --help) usage ;;
	*) die "unknown argument: $1" ;;
	esac
done

[ -n "$VERSION" ] || usage
[ -n "$CHECKSUMS" ] || usage
[ -n "$OUT_DIR" ] || usage
[ -f "$CHECKSUMS" ] || die "no such checksums file: $CHECKSUMS"

# Strictly MAJOR.MINOR.PATCH: it lands in Ruby, JSON and a URL, and is
# what Homebrew and scoop sort by.
case "$VERSION" in
*[!0-9.]* | . | *..* | .* | *.) die "version \"$VERSION\" is not MAJOR.MINOR.PATCH" ;;
esac
dots="${VERSION//[!.]/}"
[ "${#dots}" -eq 2 ] || die "version \"$VERSION\" is not MAJOR.MINOR.PATCH"

TEMPLATE_DIR="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(dirname -- "$TEMPLATE_DIR")"
[ -n "$TAG" ] || TAG="v$VERSION"
case "$TAG" in
*[!A-Za-z0-9._-]*) die "tag \"$TAG\" has characters that do not belong in a URL path segment" ;;
esac
BASE_URL="https://github.com/$REPO/releases/download/$TAG"

# One description for the crate, the formula and the scoop manifest.
DESCRIPTION="$(awk '
	/^\[package\]/ { in_pkg = 1; next }
	/^\[/          { in_pkg = 0 }
	in_pkg && /^[[:space:]]*description[[:space:]]*=/ {
		gsub(/^[^"]*"|".*$/, "")
		print
		exit
	}
' "$REPO_ROOT/Cargo.toml")"
[ -n "$DESCRIPTION" ] || die "could not read [package] description from $REPO_ROOT/Cargo.toml"
case "$DESCRIPTION" in
*[\"\\\|@]*) die "description contains a character render.sh cannot safely interpolate: $DESCRIPTION" ;;
esac

# Hashes come from the release's own checksums.txt, never recomputed, so
# the packages and the published binaries cannot disagree.
sha_for() {
	local asset="$1" hash
	# Tolerates sha256sum's "*" binary marker and a directory prefix.
	hash="$(awk -v want="$asset" '{ n = $NF; sub(/^\*/, "", n); sub(/.*\//, "", n); if (n == want) { print $1; exit } }' "$CHECKSUMS")"
	[ -n "$hash" ] || die "no sha256 for \"$asset\" in $CHECKSUMS"
	[ "${#hash}" -eq 64 ] || die "malformed sha256 for \"$asset\": \"$hash\""
	case "$hash" in
	*[!0-9a-f]*) die "malformed sha256 for \"$asset\": \"$hash\"" ;;
	esac
	printf '%s' "$hash"
}

SHA_MACOS_ARM64="$(sha_for llmman-aarch64-apple-darwin)"
SHA_LINUX_X86_64="$(sha_for llmman-x86_64-unknown-linux-gnu)"
SHA_LINUX_AARCH64="$(sha_for llmman-aarch64-unknown-linux-gnu)"
SHA_WINDOWS_X86_64="$(sha_for llmman-x86_64-pc-windows-msvc.exe)"
SHA_WINDOWS_AARCH64="$(sha_for llmman-aarch64-pc-windows-msvc.exe)"

# `|` delimiter since several values are URLs; no value can contain a
# placeholder (the description is checked above).
render() {
	sed \
		-e "s|@VERSION@|$VERSION|g" \
		-e "s|@TAG@|$TAG|g" \
		-e "s|@REPO@|$REPO|g" \
		-e "s|@DESCRIPTION@|$DESCRIPTION|g" \
		-e "s|@BASE_URL@|$BASE_URL|g" \
		-e "s|@SHA_MACOS_ARM64@|$SHA_MACOS_ARM64|g" \
		-e "s|@SHA_LINUX_X86_64@|$SHA_LINUX_X86_64|g" \
		-e "s|@SHA_LINUX_AARCH64@|$SHA_LINUX_AARCH64|g" \
		-e "s|@SHA_WINDOWS_X86_64@|$SHA_WINDOWS_X86_64|g" \
		-e "s|@SHA_WINDOWS_AARCH64@|$SHA_WINDOWS_AARCH64|g" \
		"$1"
}

printf 'rendered llmman %s (tag %s)\n' "$VERSION" "$TAG" >&2
for pkg in homebrew scoop; do
	while IFS= read -r src; do
		dest="$OUT_DIR/${src#"$TEMPLATE_DIR/"}"
		mkdir -p "$(dirname -- "$dest")"
		case "$src" in
		*.in)
			dest="${dest%.in}"
			render "$src" >"$dest"
			# A leftover @PLACEHOLDER@ is a template/script mismatch; never publish it.
			if grep -n '@[A-Z_]\{2,\}@' "$dest"; then
				die "unsubstituted placeholder(s) left in $dest (see above)"
			fi
			;;
		*) cp "$src" "$dest" ;;
		esac
		printf '  %s\n' "$dest" >&2
	done < <(find "$TEMPLATE_DIR/$pkg" -type f | sort)
done
