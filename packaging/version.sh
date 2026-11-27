#!/usr/bin/env bash
# Decides llmman's release version:
#
#   packaging/version.sh            # print it, e.g. 0.1.324
#   packaging/version.sh --apply    # also write it into Cargo.toml + Cargo.lock
#                                   # (and desktop/'s)
#
# Every commit on main is a release, published to GitHub Releases,
# crates.io, Homebrew and scoop at once, so the version is
#
#     <MAJOR>.<MINOR>.<number of commits reachable from HEAD>
#
# MAJOR.MINOR come from Cargo.toml's [package] version (its PATCH is a
# placeholder; keep it 0). The commit count is the old b<N> build number,
# now in a shape that is valid semver for crates.io and sorts correctly
# for Homebrew and scoop. Bump MAJOR or MINOR in Cargo.toml to start a
# new series; the count only grows, so versions stay strictly increasing.
# desktop/ (llmman-desktop) gets the same version; its PATCH stays 0.
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
DESKTOP_MANIFEST="$REPO_ROOT/desktop/Cargo.toml"
DESKTOP_LOCKFILE="$REPO_ROOT/desktop/Cargo.lock"

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
	' "${1:-$MANIFEST}"
}

# The `version` line right after `name = "<crate>"` (exact: "llmman" must not
# match llmman-fuzz or llmman-desktop).
read_lock_version() {
	awk -v crate="$2" '
		hit && /^version = / { gsub(/^[^"]*"|".*$/, ""); print; exit }
		$0 == "name = \"" crate "\"" { hit = 1 }
	' "$1"
}

# Writes $VERSION to manifest $1 and to crate $3's entry in lock file $2.
apply_version() {
	awk -v v="$VERSION" '
		/^\[package\]/ { in_pkg = 1 }
		/^\[/ && !/^\[package\]/ { in_pkg = 0 }
		in_pkg && !done && /^[[:space:]]*version[[:space:]]*=/ {
			sub(/"[^"]*"/, "\"" v "\"")
			done = 1
		}
		{ print }
	' "$1" >"$1.tmp" && mv "$1.tmp" "$1"

	# `cargo publish --locked` refuses a Cargo.lock that disagrees. Edited
	# directly: `cargo update --workspace` would need the registry index.
	if [ -f "$2" ]; then
		awk -v v="$VERSION" -v crate="$3" '
			hit && /^version = / { sub(/"[^"]*"/, "\"" v "\""); hit = 0 }
			$0 == "name = \"" crate "\"" { hit = 1 }
			{ print }
		' "$2" >"$2.tmp" && mv "$2.tmp" "$2"
	fi

	# Read back, so a broken edit cannot publish the placeholder 0.x.0.
	check="$(read_manifest_version "$1")"
	[ "$check" = "$VERSION" ] || die "failed to write version $VERSION to $1 (got \"$check\")"
	if [ -f "$2" ]; then
		check="$(read_lock_version "$2" "$3")"
		[ "$check" = "$VERSION" ] || die "failed to write version $VERSION to $2 (got \"$check\")"
	fi
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
	apply_version "$MANIFEST" "$LOCKFILE" llmman
	apply_version "$DESKTOP_MANIFEST" "$DESKTOP_LOCKFILE" llmman-desktop
	printf 'version.sh: applied %s to Cargo.toml and Cargo.lock, and to desktop/\n' "$VERSION" >&2
fi

printf '%s\n' "$VERSION"
