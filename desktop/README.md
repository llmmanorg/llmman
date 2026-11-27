# llmman-desktop

A tray icon (Windows notification area, macOS menu bar, Linux
StatusNotifier tray) showing llmman's manatee. No window, no settings:

- **Left click** opens the web UI, <http://127.0.0.1:17434/>, in your browser.
- **Right click** shows **Quit**.
- It keeps `llmman serve` running: it uses a daemon already answering on
  `127.0.0.1:17434`, else starts the `llmman` next to the app (or on `PATH`).
  **Quit** stops only a daemon it started. A click while the daemon starts
  (the first run fetches llama.cpp) opens the browser once it answers.
- If the daemon cannot start, nothing pops up: see
  `llmman-desktop-<user>-serve.log` in the temp directory; the next click retries.
- Only one instance runs.
- `llmman-desktop --version` prints the version.

## Install

The version is llmman's. Every package includes `llmman`. Start llmman from
your launcher: the **Start menu** on Windows, **Launchpad** or the
Applications folder on macOS, the **application menu** on Linux.

**Windows** (x64 and ARM64): install
[`llmman-desktop.msixbundle`](https://github.com/llmmanorg/llmman/releases/latest/download/llmman-desktop.msixbundle).
Windows installs only signed packages; the bundle is signed when the
repository has the `AZURE_*` secrets (see Building). Otherwise it is
unsigned and installs only with `Add-AppxPackage -AllowUnsigned` (Windows 11,
as administrator).

**macOS** (Apple Silicon): `brew install --cask llmmanorg/tap/llmman-desktop`, or open
[`llmman-desktop-aarch64-apple-darwin.dmg`](https://github.com/llmmanorg/llmman/releases/latest/download/llmman-desktop-aarch64-apple-darwin.dmg)
and drag llmman.app to Applications. It is signed ad hoc and not notarized.
The cask removes the quarantine attribute for you; after the .dmg download,
right-click the app and choose Open, or run

```sh
xattr -dr com.apple.quarantine /Applications/llmman.app
```

**Linux** (x86_64 and aarch64; glibc 2.28+): `brew install llmmanorg/tap/llmman-desktop`
adds llmman to the application menu. It needs a StatusNotifierItem host (KDE
Plasma has one; GNOME needs its AppIndicator extension). Without one, the
icon appears once a host starts.

## Building

A crate of its own, so its dependencies stay out of llmman's lockfile:
`tray-icon` and `tao` on macOS and Windows, `ksni` (no GTK) on Linux.

```sh
cd desktop
cargo run                   # needs a desktop session
cargo test
cargo build --release
```

CI (the `desktop*` jobs in `.github/workflows/ci.yml`) sets the version with
`packaging/version.sh --apply`, then builds and packages the `.dmg`
(`packaging/desktop-dmg.sh`) and `.msixbundle` (`packaging/desktop-msix.ps1`).
The manifest `Publisher` is the repository variable `MSIX_PUBLISHER` (default
`CN=llmman`); signed, it must be the certificate's subject. Signing uses the
secrets `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`,
`AZURE_TS_ENDPOINT`, `AZURE_TS_ACCOUNT_NAME` and `AZURE_TS_PROFILE_NAME`;
without them the package is marked unsigned.

A debug build reads `LLMMAN_DESKTOP_TEST_HOST` (for example `127.0.0.1:18467`)
in place of `127.0.0.1:17434`; release builds ignore it.

The icons are not kept in git. `build.rs` downloads the tray icons, and
`packaging/desktop-dmg.sh` and `packaging/desktop-msix.ps1` the app icon and
logos, from the `docs-assets` release, each pinned by sha256; building
therefore needs network access and `curl`. `generate-assets.py` (needs Pillow)
makes the files and prints the `gh release upload` command and the new pins.
