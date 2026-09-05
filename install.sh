#!/bin/sh
# llmman installer (Linux, macOS) — downloads the llmman binary matching
# this host's OS/arch from GitHub Releases and installs it to
# ~/.local/bin. For Windows, use install.ps1 instead.
#
#   curl -fsSL https://raw.githubusercontent.com/llmmanorg/llmman/main/install.sh | sh
#
# GPU/backend detection happens at runtime inside the llmman binary
# itself, the first time `llmman serve` needs a `llama-server` (see
# src/hostgpu.rs and src/llama_release.rs) — this script just gets llmman
# onto PATH.
#
# Supported today (matches .github/workflows/ci.yml's build matrix):
#   Linux   x86_64, aarch64
#   macOS   aarch64 (Apple Silicon) — Intel (x86_64) Macs are not built
#
# Env overrides:
#   LLMMAN_VERSION   pin an exact release tag (e.g. "v0.2.0"); default: latest
#   LLMMAN_REPO      "owner/repo" to fetch from; default: llmmanorg/llmman
#   LLMMAN_BASE_URL  release base URL to fetch from, in place of GitHub's own
#                    "https://github.com/$LLMMAN_REPO/releases"; exists so
#                    CI (.github/workflows/ci.yml's e2e job) can point this
#                    script at a throwaway local HTTP server serving that
#                    job's own just-built binary instead — a real GitHub
#                    release for a commit/PR still under test wouldn't exist
#                    yet — while every other part of the install (OS/arch
#                    detection, the download itself, the --version smoke
#                    check, the final install-directory copy) still runs
#                    completely unmodified against a real HTTP round trip.
#   SKIP_INSTALL     download and verify only, don't install to ~/.local/bin

: "${LLMMAN_REPO:=llmmanorg/llmman}"

die() {
	printf "%s\n" "$@" >&2
	exit 111
}

check_bin() {
	command -v "$1" >/dev/null 2>/dev/null
}

check_path() {
	case ":$1:" in
	(*":$HOME/.local/bin:"*) return 0 ;;
	esac
	return 1
}

main() {
	check_bin curl || die "Please install curl"

	case "$(uname -m)" in
	(x86_64|amd64)  ARCH=x86_64  ;;
	(arm64|aarch64) ARCH=aarch64 ;;
	(*) die "Arch not supported: $(uname -m)" ;;
	esac

	case "$(uname -s)" in
	(Linux)
		TARGET="${ARCH}-unknown-linux-gnu"
		;;
	(Darwin)
		[ "$ARCH" = "aarch64" ] || die \
			"Intel (x86_64) macOS is not supported by llmman's published builds — only" \
			"Apple Silicon (arm64) is. Build from source instead: see docs/backends.md."
		TARGET="aarch64-apple-darwin"
		;;
	(*) die "OS not supported: $(uname -s). On Windows, use install.ps1 instead." ;;
	esac

	[ "$HOME" ] || die "No HOME, please check your OS"

	ASSET="llmman-${TARGET}"
	BASE_URL="${LLMMAN_BASE_URL:-https://github.com/$LLMMAN_REPO/releases}"
	if [ "$LLMMAN_VERSION" ]; then
		URL="$BASE_URL/download/$LLMMAN_VERSION/$ASSET"
		printf "Version: %s\n" "$LLMMAN_VERSION"
	else
		URL="$BASE_URL/latest/download/$ASSET"
		printf "Version: latest\n"
	fi

	DIR=$(mktemp -d) || die "Couldn't create a temporary directory"
	trap 'rm -rf "$DIR"' EXIT INT TERM

	printf "Downloading %s...\n" "$ASSET"
	curl -fsSL "$URL" -o "$DIR/llmman.tmp" || die \
		"Failed to download $URL" \
		"(has a release for $TARGET been published yet? see .github/workflows/ci.yml)"
	chmod +x "$DIR/llmman.tmp"

	"$DIR/llmman.tmp" --version >/dev/null 2>&1 || die \
		"Downloaded llmman binary failed to run"

	if [ "$SKIP_INSTALL" ]; then
		printf "Download verified, installation skipped (SKIP_INSTALL is set): %s\n" "$DIR/llmman.tmp"
		return
	fi

	mkdir -p "$HOME/.local/bin" &&
	cp "$DIR/llmman.tmp" "$HOME/.local/bin/llmman.new" &&
	mv "$HOME/.local/bin/llmman.new" "$HOME/.local/bin/llmman" || die \
		"Couldn't install llmman to $HOME/.local/bin"

	printf "Installation completed successfully\n\n"

	if ! check_path "$PATH"; then
		LOGIN_SHELL="${SHELL:-/bin/sh}"
		LOGIN_PATH=$("$LOGIN_SHELL" -l -c 'echo $PATH' 2>/dev/null)

		if ! check_path "$LOGIN_PATH"; then
			RC_FILE=
			case "${SHELL##*/}" in
			(bash) RC_FILE=".bash_profile" ;;
			(zsh)  RC_FILE=".zprofile" ;;
			esac
			if [ "$RC_FILE" ]; then
				cat <<-EOF
				To make llmman available in future sessions, add ~/.local/bin to your PATH by running:

				  echo 'export PATH="\$HOME/.local/bin:\$PATH"' >> ~/${RC_FILE}

				EOF
			else
				cat <<-EOF
				Add this line to your shell profile to include ~/.local/bin to your PATH:

				  export PATH="\$HOME/.local/bin:\$PATH"

				EOF
			fi
		fi
	fi
}

main "$@"
