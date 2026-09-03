#!/usr/bin/env bash
# Render the Homebrew formula and Scoop manifest for one llmman release
# from the templates next to this script.
#
# Used by the `publish-homebrew` and `publish-scoop` jobs in
# .github/workflows/ci.yml, but deliberately a standalone script rather
# than inline YAML so the exact bytes that land in the tap/bucket can be
# reproduced (and diffed) locally:
#
#   packaging/render.sh --tag b304 --checksums checksums.txt --out-dir /tmp/out
#
# Two release channels, mirroring the two the `release` job itself
# publishes (see its own comment on the b<N>/v* split):
#
#   v<semver>  "stable" -> Formula/llmman.rb, bucket/llmman.json
#   b<N>       "dev"    -> Formula/llmman-dev.rb, bucket/llmman-dev.json
#
# They are separate packages *on purpose*. A single formula cannot serve
# both: Homebrew and Scoop both order upgrades by comparing version
# strings, and the two channels' versions are not mutually comparable
# (b305's "305" sorts above a stable "0.2.0", so one merge to main would
# permanently shadow every future stable release for anyone tracking it).
# Keeping them as two names means `llmman` upgrades along semver and
# `llmman-dev` tracks main, and neither can ever shadow the other.

set -euo pipefail

die() {
	printf 'render.sh: %s\n' "$@" >&2
	exit 1
}

usage() {
	cat >&2 <<-EOF
		usage: render.sh --tag <tag> --checksums <file> --out-dir <dir> [--repo <owner/repo>]

		  --tag        release tag to render for: "v1.2.3" (stable) or "b304" (dev)
		  --checksums  sha256sum-format file covering the release's assets
		  --out-dir    directory to write Formula/ and bucket/ into
		  --repo       GitHub repo the download URLs point at
		               (default: llmmanorg/llmman)
	EOF
	exit 2
}

TAG=""
CHECKSUMS=""
OUT_DIR=""
REPO="llmmanorg/llmman"

while [ $# -gt 0 ]; do
	case "$1" in
	--tag) TAG="${2:-}"; shift 2 ;;
	--checksums) CHECKSUMS="${2:-}"; shift 2 ;;
	--out-dir) OUT_DIR="${2:-}"; shift 2 ;;
	--repo) REPO="${2:-}"; shift 2 ;;
	-h | --help) usage ;;
	*) die "unknown argument: $1" ;;
	esac
done

[ -n "$TAG" ] || usage
[ -n "$CHECKSUMS" ] || usage
[ -n "$OUT_DIR" ] || usage
[ -f "$CHECKSUMS" ] || die "no such checksums file: $CHECKSUMS"

TEMPLATE_DIR="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(dirname -- "$TEMPLATE_DIR")"

# The crate version, read from the same Cargo.toml the binaries were built
# from. Used as the string the Homebrew formula's `test do` block asserts
# `llmman --version` contains -- see VERSION_MATCH below. Taken from the
# first `version = "..."` after [package] so a dependency's own version
# line can never be picked up instead.
CARGO_VERSION="$(awk '
	/^\[package\]/ { in_pkg = 1; next }
	/^\[/          { in_pkg = 0 }
	in_pkg && /^[[:space:]]*version[[:space:]]*=/ {
		gsub(/^[^"]*"|".*$/, "")
		print
		exit
	}
' "$REPO_ROOT/Cargo.toml")"
[ -n "$CARGO_VERSION" ] || die "could not read [package] version from $REPO_ROOT/Cargo.toml"

# ── Channel, version and version-assertion string ─────────────────────────
#
# VERSION is what the package manager sorts on.
#
# VERSION_MATCH is a substring of `llmman --version`'s own output, which
# the Homebrew formula's `test do` block asserts on. It is the *crate*
# version for both channels, not the tag: build.rs's emit_version prints
# "llmman <crate version> (<git describe>)", and on the dev channel that
# `git describe` is a bare commit hash, not the b<N> tag -- the `build`
# job checks the commit out before the `release` job has created that tag,
# so there is no tag for describe to find. (Confirmed against release
# b304, whose binary reports "llmman 0.1.0 (6cdaadb)".) The crate version
# is the only part of that line both channels reliably share.
case "$TAG" in
v*)
	CHANNEL="stable"
	NAME="llmman"
	VERSION="${TAG#v}"
	# A stable tag must name the version actually compiled into the
	# binaries it points at, or the formula would advertise a version
	# `llmman --version` disagrees with. The publish-crate job enforces
	# the same equality before uploading to crates.io; this is the same
	# check for the Homebrew/Scoop side, which does not go through Cargo.
	[ "$VERSION" = "$CARGO_VERSION" ] || die \
		"tag \"$TAG\" does not match the Cargo.toml version \"$CARGO_VERSION\"" \
		"(bump the crate version before tagging a stable release)"
	;;
b*)
	CHANNEL="dev"
	NAME="llmman-dev"
	# Bare build number: the `release` job derives b<N> from `git
	# rev-list --count HEAD`, so this increases monotonically with every
	# commit and orders correctly within this channel.
	VERSION="${TAG#b}"
	;;
*)
	die "unrecognised tag \"$TAG\": expected v<semver> (stable) or b<N> (dev)"
	;;
esac

VERSION_MATCH="$CARGO_VERSION"

case "$VERSION" in
'' | *[!0-9.]*) die "tag \"$TAG\" yielded a non-numeric version \"$VERSION\"" ;;
esac

# Homebrew derives a formula's class name from its filename: llmman.rb ->
# Llmman, llmman-dev.rb -> LlmmanDev. Getting this wrong makes the tap
# fail to load rather than merely misbehave.
case "$CHANNEL" in
stable) CLASS="Llmman" ;;
dev) CLASS="LlmmanDev" ;;
esac

BASE_URL="https://github.com/$REPO/releases/download/$TAG"

# ── Asset checksums ───────────────────────────────────────────────────────
# Looked up by asset name from the release's own checksums.txt rather than
# recomputed here, so the formula/manifest and the published binaries can
# never disagree: both come from the same file the `release` job attached.
sha_for() {
	local asset="$1" hash
	# The `release` job runs sha256sum from inside dist/, so the names in
	# the file are already bare basenames. The two sub()s tolerate the
	# other forms a hand-made checksums file can have anyway: a leading
	# "*" (sha256sum's binary-mode marker) and a directory prefix.
	hash="$(awk -v want="$asset" '{ n = $NF; sub(/^\*/, "", n); sub(/.*\//, "", n); if (n == want) { print $1; exit } }' "$CHECKSUMS")"
	[ -n "$hash" ] || die "no sha256 for \"$asset\" in $CHECKSUMS"
	# 64 lowercase hex chars. A truncated or otherwise malformed hash here
	# would otherwise only surface as a checksum mismatch on a user's
	# machine, long after the release was published.
	case "$hash" in
	[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) ;;
	*) die "malformed sha256 for \"$asset\": \"$hash\"" ;;
	esac
	printf '%s' "$hash"
}

SHA_MACOS_ARM64="$(sha_for llmman-aarch64-apple-darwin)"
SHA_LINUX_X86_64="$(sha_for llmman-x86_64-unknown-linux-gnu)"
SHA_LINUX_AARCH64="$(sha_for llmman-aarch64-unknown-linux-gnu)"
SHA_WINDOWS_X86_64="$(sha_for llmman-x86_64-pc-windows-msvc.exe)"
SHA_WINDOWS_AARCH64="$(sha_for llmman-aarch64-pc-windows-msvc.exe)"

# ── Render ────────────────────────────────────────────────────────────────
# sed with a distinct delimiter (|) since several values are URLs. Every
# placeholder is a fixed @NAME@ token and none of the values can contain
# one, so ordering between the expressions does not matter.
render() {
	sed \
		-e "s|@CLASS@|$CLASS|g" \
		-e "s|@VERSION@|$VERSION|g" \
		-e "s|@VERSION_MATCH@|$VERSION_MATCH|g" \
		-e "s|@BASE_URL@|$BASE_URL|g" \
		-e "s|@SHA_MACOS_ARM64@|$SHA_MACOS_ARM64|g" \
		-e "s|@SHA_LINUX_X86_64@|$SHA_LINUX_X86_64|g" \
		-e "s|@SHA_LINUX_AARCH64@|$SHA_LINUX_AARCH64|g" \
		-e "s|@SHA_WINDOWS_X86_64@|$SHA_WINDOWS_X86_64|g" \
		-e "s|@SHA_WINDOWS_AARCH64@|$SHA_WINDOWS_AARCH64|g" \
		"$1"
}

mkdir -p "$OUT_DIR/Formula" "$OUT_DIR/bucket"
render "$TEMPLATE_DIR/homebrew/llmman.rb.in" >"$OUT_DIR/Formula/$NAME.rb"
render "$TEMPLATE_DIR/scoop/llmman.json.in" >"$OUT_DIR/bucket/$NAME.json"

# Nothing downstream should ever publish a half-substituted template: a
# leftover @PLACEHOLDER@ is a rename in a template that this script was
# not updated for, and would otherwise reach the tap as a formula that
# fails to load (or, worse, a manifest with a literal "@SHA...@" hash).
for rendered in "$OUT_DIR/Formula/$NAME.rb" "$OUT_DIR/bucket/$NAME.json"; do
	if grep -n '@[A-Z_]\{2,\}@' "$rendered"; then
		die "unsubstituted placeholder(s) left in $rendered (see above)"
	fi
done

printf 'rendered %s channel %s (version %s)\n' "$CHANNEL" "$NAME" "$VERSION" >&2
printf '  %s\n' "$OUT_DIR/Formula/$NAME.rb" "$OUT_DIR/bucket/$NAME.json" >&2
