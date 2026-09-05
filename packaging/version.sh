#!/usr/bin/env bash
# Decides llmman's release version:
#
#   packaging/version.sh            # print it, e.g. 0.1.324
#   packaging/version.sh --apply    # also write it into Cargo.toml + Cargo.lock
#
# Every commit on main is a release, published to GitHub Releases,
# crates.io, Homebrew and winget at once, so the version is
#
#     <MAJOR>.<MINOR>.<number of commits reachable from HEAD>
#
# MAJOR.MINOR come from Cargo.toml's [package] version (its PATCH is a
# placeholder; keep it 0). The commit count is the old b<N> build number,
# now in a shape that is valid semver for crates.io and sorts correctly
# for Homebrew and winget. Bump MAJOR or MINOR in Cargo.toml to start a
# new series; the count only grows, so versions stay strictly increasing.
#
# Needs a full clone (fetch-depth: 0): a shallow one miscounts.

set -euo pipefail

die() {
	printf 'version.sh: %s\n' "$@" >&2
	exit 1
}

APPLY=0
case "${1:-}" in
"") ;;
--apply) APPLY=1 ;;
-h | --help)
	sed -n '2,/^$/s/^# \{0,1\}//p' "$0" >&2
	exit 2
	;;
*) die "unknown argument: $1" ;;
esac

REPO_ROOT="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)"
MANIFEST="$REPO_ROOT/Cargo.toml"
LOCKFILE="$REPO_ROOT/Cargo.lock"

# First `version = "..."` inside [package] only.
read_manifest_version() {
	awk '
		/^\[package\]/ { in_pkg = 1; next }
		/^\[/          { in_pkg = 0 }
		in_pkg && /^[[:space:]]*version[[:space:]]*=/ {
			gsub(/^[^"]*"|".*$/, "")
			print
			exit
		}
	' "$MANIFEST"
}

# The `version` line right after `name = "llmman"` (exact, so llmman-fuzz
# or any future llmman-* crate does not match).
read_lock_version() {
	awk '
		hit && /^version = / { gsub(/^[^"]*"|".*$/, ""); print; exit }
		/^name = "llmman"$/ { hit = 1 }
	' "$LOCKFILE"
}

BASE="$(read_manifest_version)"
[ -n "$BASE" ] || die "could not read [package] version from $MANIFEST"
case "$BASE" in
[0-9]*.[0-9]*.[0-9]*) ;;
*) die "Cargo.toml version \"$BASE\" is not MAJOR.MINOR.PATCH" ;;
esac
MAJOR="${BASE%%.*}"
MINOR="${BASE#*.}"
MINOR="${MINOR%%.*}"
case "$MAJOR$MINOR" in
*[!0-9]*) die "Cargo.toml version \"$BASE\" has a non-numeric MAJOR or MINOR" ;;
esac

if git -C "$REPO_ROOT" rev-parse --is-shallow-repository 2>/dev/null | grep -qx true; then
	die "shallow clone: the commit count would be wrong (use fetch-depth: 0)"
fi
COUNT="$(git -C "$REPO_ROOT" rev-list --count HEAD)" || die "git rev-list failed (not a git checkout?)"
case "$COUNT" in
'' | *[!0-9]*) die "unexpected commit count \"$COUNT\"" ;;
esac

VERSION="$MAJOR.$MINOR.$COUNT"

if [ "$APPLY" -eq 1 ]; then
	awk -v v="$VERSION" '
		/^\[package\]/ { in_pkg = 1 }
		/^\[/ && !/^\[package\]/ { in_pkg = 0 }
		in_pkg && !done && /^[[:space:]]*version[[:space:]]*=/ {
			sub(/"[^"]*"/, "\"" v "\"")
			done = 1
		}
		{ print }
	' "$MANIFEST" >"$MANIFEST.tmp" && mv "$MANIFEST.tmp" "$MANIFEST"

	# Cargo.lock has its own entry for the root package, and `cargo publish
	# --locked` refuses to run if it disagrees. Edited directly: `cargo
	# update --workspace` would need the registry index.
	if [ -f "$LOCKFILE" ]; then
		awk -v v="$VERSION" '
			hit && /^version = / { sub(/"[^"]*"/, "\"" v "\""); hit = 0 }
			/^name = "llmman"$/ { hit = 1 }
			{ print }
		' "$LOCKFILE" >"$LOCKFILE.tmp" && mv "$LOCKFILE.tmp" "$LOCKFILE"
	fi

	# Read back: a Cargo.toml reshuffle that broke the edit above must not
	# silently publish the placeholder 0.x.0.
	check="$(read_manifest_version)"
	[ "$check" = "$VERSION" ] || die "failed to write version $VERSION to $MANIFEST (got \"$check\")"
	if [ -f "$LOCKFILE" ]; then
		check="$(read_lock_version)"
		[ "$check" = "$VERSION" ] || die "failed to write version $VERSION to $LOCKFILE (got \"$check\")"
	fi
	printf 'version.sh: applied %s to Cargo.toml and Cargo.lock\n' "$VERSION" >&2
fi

printf '%s\n' "$VERSION"
