#!/usr/bin/env bash
# Render the Homebrew formula and winget manifests for one llmman release
# from the templates next to this script. Used by the publish-homebrew and
# publish-winget CI jobs; a standalone script so the output can be
# reproduced locally:
#
#   packaging/render.sh --version 0.1.324 --checksums checksums.txt --out-dir /tmp/out
#
# Output (mirrors where each file is pushed):
#
#   <out>/Formula/llmman.rb, <out>/README.md            -> llmmanorg/homebrew-tap
#   <out>/manifests/l/llmmanorg/llmman/<version>/*.yaml -> microsoft/winget-pkgs
#
# winget notes (kept out of the manifests, which winget-pkgs likes bare):
#   - InstallerType `portable`: the assets are bare .exe files. winget
#     symlinks the exe onto PATH under the first `Commands` entry, so
#     `Commands: [llmman]` is what makes the command `llmman`.
#   - Both binaries import VCRUNTIME140.dll, hence the per-architecture
#     Microsoft.VCRedist.2015+ dependency.
#   - InstallerSha256 is upper-cased, matching wingetcreate's output.

set -euo pipefail

die() {
	printf 'render.sh: %s\n' "$@" >&2
	exit 1
}

usage() {
	cat >&2 <<-EOF
		usage: render.sh --version <x.y.z> --checksums <file> --out-dir <dir>
		                 [--repo <owner/repo>] [--tag <tag>] [--date <YYYY-MM-DD>]

		  --version    release version, as printed by packaging/version.sh
		  --checksums  sha256sum-format file covering the release's assets
		  --out-dir    directory to write Formula/ and manifests/ into
		  --repo       GitHub repo the download URLs point at (default: llmmanorg/llmman)
		  --tag        release tag the assets live under (default: v<version>;
		               only differs for releases cut before the tag scheme changed, e.g. b321)
		  --date       winget ReleaseDate (default: today, UTC)
	EOF
	exit 2
}

VERSION=""
CHECKSUMS=""
OUT_DIR=""
REPO="llmmanorg/llmman"
TAG=""
RELEASE_DATE=""

while [ $# -gt 0 ]; do
	case "$1" in
	--version) VERSION="${2:-}"; shift 2 ;;
	--checksums) CHECKSUMS="${2:-}"; shift 2 ;;
	--out-dir) OUT_DIR="${2:-}"; shift 2 ;;
	--repo) REPO="${2:-}"; shift 2 ;;
	--tag) TAG="${2:-}"; shift 2 ;;
	--date) RELEASE_DATE="${2:-}"; shift 2 ;;
	-h | --help) usage ;;
	*) die "unknown argument: $1" ;;
	esac
done

[ -n "$VERSION" ] || usage
[ -n "$CHECKSUMS" ] || usage
[ -n "$OUT_DIR" ] || usage
[ -f "$CHECKSUMS" ] || die "no such checksums file: $CHECKSUMS"
[ -z "$RELEASE_DATE" ] && RELEASE_DATE="$(date -u +%Y-%m-%d)"

# Strictly MAJOR.MINOR.PATCH: it lands in Ruby, YAML and a URL, and is
# what Homebrew and winget sort by.
case "$VERSION" in
*[!0-9.]* | . | *..* | .* | *.) die "version \"$VERSION\" is not MAJOR.MINOR.PATCH" ;;
esac
dots="${VERSION//[!.]/}"
[ "${#dots}" -eq 2 ] || die "version \"$VERSION\" is not MAJOR.MINOR.PATCH"
case "$RELEASE_DATE" in
[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) ;;
*) die "date \"$RELEASE_DATE\" is not YYYY-MM-DD" ;;
esac

TEMPLATE_DIR="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(dirname -- "$TEMPLATE_DIR")"
[ -n "$TAG" ] || TAG="v$VERSION"
case "$TAG" in
*[!A-Za-z0-9._-]*) die "tag \"$TAG\" has characters that do not belong in a URL path segment" ;;
esac
BASE_URL="https://github.com/$REPO/releases/download/$TAG"

# One description for the crate, the formula and the winget listing.
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
SHA_WINDOWS_X86_64_UPPER="$(printf '%s' "$SHA_WINDOWS_X86_64" | tr '[:lower:]' '[:upper:]')"
SHA_WINDOWS_AARCH64_UPPER="$(printf '%s' "$SHA_WINDOWS_AARCH64" | tr '[:lower:]' '[:upper:]')"

# `|` delimiter since several values are URLs; no value can contain a
# placeholder (the description is checked above).
render() {
	sed \
		-e "s|@VERSION@|$VERSION|g" \
		-e "s|@TAG@|$TAG|g" \
		-e "s|@REPO@|$REPO|g" \
		-e "s|@DESCRIPTION@|$DESCRIPTION|g" \
		-e "s|@BASE_URL@|$BASE_URL|g" \
		-e "s|@RELEASE_DATE@|$RELEASE_DATE|g" \
		-e "s|@SHA_MACOS_ARM64@|$SHA_MACOS_ARM64|g" \
		-e "s|@SHA_LINUX_X86_64@|$SHA_LINUX_X86_64|g" \
		-e "s|@SHA_LINUX_AARCH64@|$SHA_LINUX_AARCH64|g" \
		-e "s|@SHA_WINDOWS_X86_64_UPPER@|$SHA_WINDOWS_X86_64_UPPER|g" \
		-e "s|@SHA_WINDOWS_AARCH64_UPPER@|$SHA_WINDOWS_AARCH64_UPPER|g" \
		"$1"
}

WINGET_DIR="$OUT_DIR/manifests/l/llmmanorg/llmman/$VERSION"
mkdir -p "$OUT_DIR/Formula" "$WINGET_DIR"

render "$TEMPLATE_DIR/homebrew/llmman.rb.in" >"$OUT_DIR/Formula/llmman.rb"
cp "$TEMPLATE_DIR/homebrew/README.md" "$OUT_DIR/README.md"
for name in llmmanorg.llmman.yaml llmmanorg.llmman.installer.yaml llmmanorg.llmman.locale.en-US.yaml; do
	render "$TEMPLATE_DIR/winget/$name.in" >"$WINGET_DIR/$name"
done

# A leftover @PLACEHOLDER@ is a template/script mismatch; never publish it.
for rendered in "$OUT_DIR/Formula/llmman.rb" "$WINGET_DIR"/*.yaml; do
	if grep -n '@[A-Z_]\{2,\}@' "$rendered"; then
		die "unsubstituted placeholder(s) left in $rendered (see above)"
	fi
done

printf 'rendered llmman %s (tag %s)\n' "$VERSION" "$TAG" >&2
printf '  %s\n' "$OUT_DIR/Formula/llmman.rb" "$OUT_DIR/README.md" "$WINGET_DIR"/*.yaml >&2
