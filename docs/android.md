# Android

The Android app is the [web UI](webui.md) in a WebView over an embedded
`llmman serve`: the same daemon, cross-compiled for `aarch64-linux-android`
and shipped with llama.cpp's own Android build, running on the phone. Every
model runs on-device; nothing in the app talks to a server that is not
`127.0.0.1:17434`. Source is `android/`.

Requirements: Android 9 (API 28) or newer, 64-bit ARM — the one Android
build llama.cpp publishes. Inference is CPU-only, so pick models by RAM:
a phone with 8 GB runs 1–4B parameter models at Q4 comfortably; 7–8B at
Q4 needs 12 GB or more.

## Install

Download `llmman-aarch64-linux-android.apk` from the
[latest release](https://github.com/llmmanorg/llmman/releases/latest) and
open it (sideloading has to be allowed for the browser or file manager
that opens it). The `checksums.txt` in the same release covers it.

The app is not on Google Play. Releases are signed with one key, so a
newer APK installs over an older one.

## What it does

On launch the app starts a foreground service that runs `llmman serve` and
shows the web UI once `/api/version` answers. The service keeps the daemon
up — a model stays loaded, a pull keeps going — with the app swiped away;
the notification's *Stop* ends it. If the daemon exits, the service restarts
it with backoff and the UI reloads. A failure shows the daemon's log in the
app.

Everything the web UI can do on a desktop works here: *Pull a model…* with
any reference `llmman pull` takes (`docker.io/ai/…`, `hf.co/…`, …), chat
with streaming replies, the model picker, Export chats (into Downloads),
and the *Shell* tab — a `/system/bin/sh` in the app's sandbox, with
`llmman` on `PATH`. Diffusion models are not supported: the mediagen
backend needs GPU libraries the phone lacks.

The daemon binds loopback only, so nothing else on the phone or the
network can reach it; the app exposes no `LLMMAN_HOST` setting. Other
apps on the same phone can, at `http://127.0.0.1:17434` — the Ollama,
OpenAI, Anthropic and Gemini APIs from [api.md](api.md), no key.

Model storage is the app's private data (`Android/data` is not used) and
is removed with the app or by *Clear storage*. Conversations live in the
WebView's IndexedDB, as in a browser.

## Building

Needs, in addition to the [usual build tools](../README.md#install) (Rust,
Go 1.22+): a JDK 17, the Android SDK with `platforms;android-35`,
`build-tools;35.0.0` and `ndk;27.2.12479018` (Android Studio or
`sdkmanager` installs them), `rustup target add aarch64-linux-android` and
`cargo install cargo-ndk`.

```sh
cd android
echo "sdk.dir=$ANDROID_HOME" > local.properties   # or set ANDROID_HOME
./gradlew assembleDebug           # app/build/outputs/apk/debug/app-debug.apk
./gradlew assembleRelease         # debug-signed unless LLMMAN_ANDROID_* are set
```

Gradle does all of it: `cargo ndk -t arm64-v8a --platform 28 build
--release` at the repository root, a download of the pinned
(`LLAMA_CPP_RELEASE`) `llama-<tag>-bin-android-arm64.tar.gz`, and staging
of both into the APK's native libraries. The version is `Cargo.toml`'s; in
CI `packaging/version.sh --apply` sets it first, as for every other
release asset.

To sign a release with your own key, set `LLMMAN_ANDROID_KEYSTORE` (the
keystore, base64), `LLMMAN_ANDROID_KEYSTORE_PASSWORD`,
`LLMMAN_ANDROID_KEY_ALIAS` and `LLMMAN_ANDROID_KEY_PASSWORD` in the
environment; without them `assembleRelease` signs with the debug key.

### How the pieces fit

Android forbids an app from executing anything under its writable storage
(since API 29), so every executable has to arrive through the APK's
native-library directory, which only takes `lib*.so`. The daemon is
therefore `libllmman.so`, llama.cpp's `llama-server` launcher is
`libllama-server.so`, and `extractNativeLibs` is on so they exist as files.
The service exposes both on `PATH` through symlinks named `llama-server`
and `llmman` in the app's files directory (the kernel resolves the link,
SELinux checks the target) and runs the daemon with `LLMMAN_RUNTIME=path`,
`LD_LIBRARY_PATH` set to that directory (llama.cpp's Android objects carry
no RUNPATH), `HOME` under the app's files directory and
`LLMMAN_SHELL=/system/bin/sh`.

Go has no `c-archive` build mode for Android, so there the Go shim is a
shared object (`libllmman_shim.so`, `-buildmode=c-shared`) next to the
binary, found through a `$ORIGIN` RUNPATH — `build.rs` switches on
`CARGO_CFG_TARGET_OS`. Every object is 16 KiB page-aligned, as Android 15
devices with 16 KiB pages require.

The same `aarch64-linux-android` binary runs outside the app too, e.g. in
Termux: `--runtime bin` downloads llama.cpp's Android release there, since
`src/llama_release.rs` knows its asset name.
