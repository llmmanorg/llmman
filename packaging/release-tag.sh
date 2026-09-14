#!/usr/bin/env bash
# The release tag, for CI (needs gh, GH_TOKEN and GITHUB_REPOSITORY):
#
#   packaging/release-tag.sh ensure <tag> <sha>   # create <tag> at <sha>, or check it is there
#   packaging/release-tag.sh remove <tag> <sha>   # delete <tag> if at <sha> and unreleased
#
# GITHUB_TOKEN may only create a ref whose .github/workflows/ matches
# main's tip (it can never hold the `workflows` permission); the API
# reports anything else as "403 Resource not accessible by integration".
# So `ensure` runs first thing in a run, while its commit is the tip.
# Lookups treat only a 404 as "absent"; any other error is fatal.

set -euo pipefail

die() {
	printf 'release-tag.sh: %s\n' "$*" >&2
	exit 1
}

# get <path> <jq>: sets OUT. 0 = found, 1 = 404, anything else dies.
get() {
	if OUT="$(gh api "repos/$GITHUB_REPOSITORY/$1" --jq "$2" 2>&1)"; then
		return 0
	fi
	case "$OUT" in
	*"(HTTP 404)"*) return 1 ;;
	esac
	die "GET $1: $OUT"
}

ensure() {
	if get "git/ref/tags/$1" .object.sha; then
		[ "$OUT" = "$2" ] || die "$1 is at $OUT, not $2"
		echo "$1 already at $2"
		return
	fi
	gh api -X POST "repos/$GITHUB_REPOSITORY/git/refs" -f "ref=refs/tags/$1" -f "sha=$2" --jq .ref ||
		die "creating $1 at $2 failed; a 403 means .github/workflows/ changed on main since this commit, which GITHUB_TOKEN cannot tag across"
}

remove() {
	if ! get "git/ref/tags/$1" .object.sha; then
		echo "$1 does not exist"
		return
	fi
	[ "$OUT" = "$2" ] || die "$1 is at $OUT, not $2; leaving it"
	if get "releases/tags/$1" .id; then
		echo "$1 has a release; leaving it"
		return
	fi
	gh api -X DELETE "repos/$GITHUB_REPOSITORY/git/refs/tags/$1"
	echo "deleted $1"
}

[ $# -eq 3 ] || die "usage: $0 ensure|remove <tag> <sha>"
: "${GITHUB_REPOSITORY:?}"
case "$1" in
ensure | remove) "$1" "$2" "$3" ;;
*) die "unknown command: $1" ;;
esac
