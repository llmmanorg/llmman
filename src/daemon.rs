//! Shared client-side helpers for talking to a local `llmman serve`
//! instance over its Ollama-protocol HTTP API (127.0.0.1:17434).
//!
//! Used by any CLI subcommand that acts as a client of that API rather than
//! calling the FFI/model-management logic directly — currently `pull`,
//! `push`, and `launch` — so bare model-name resolution (see
//! `shortnames::resolve_ollama_api`), the local model store, and any
//! already-loaded models are always the daemon's, never duplicated
//! per-invocation.

use std::io::BufRead;
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::Context;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;

/// The fixed loopback origin `llmman serve` always binds to (see
/// cmd::serve's own doc comment on why this isn't configurable).
pub const SERVER: &str = "http://127.0.0.1:17434";

/// Quick synchronous reachability check — none of this module's callers
/// run inside an async runtime, so a plain TCP connect attempt is enough
/// (no need to actually round-trip an HTTP request just to check liveness).
pub fn server_alive() -> bool {
    std::net::TcpStream::connect("127.0.0.1:17434").is_ok()
}

/// The daemon's self-identity from GET /api/version — the identity fields
/// are absent on daemons built before they were reported (which is itself
/// the strongest possible staleness signal: only a long-forgotten daemon
/// from an old install still runs such a build).
#[derive(Deserialize, Default)]
struct DaemonIdentity {
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    exe: Option<String>,
    #[serde(default)]
    pid: Option<u32>,
}

/// Returns the running daemon's identity if it should be stopped and
/// replaced rather than reused, matching how Ollama's app never lets a
/// server from a superseded install keep serving (its supervisor stops the
/// previous server on startup and after every update — see ollama's
/// app/server cleanup/killOtherInstances). A daemon is stale when:
///
/// - its own executable no longer exists on disk (the install that
///   provided it was upgraded or removed — e.g. a Homebrew cask upgrade
///   deleting the old version directory out from under a daemon it left
///   running),
/// - its version differs from this client's (the binary was replaced in
///   place by an update, so the same path now holds a newer build than
///   the one still serving — the installed binary wins), or
/// - it predates identity reporting entirely.
///
/// Returns None both for a healthy daemon and when /api/version can't be
/// fetched/parsed at all — whatever is holding the port in that case,
/// killing processes based on it would be a guess.
fn stale_daemon() -> Option<DaemonIdentity> {
    let resp = reqwest::blocking::Client::builder()
        .timeout(Some(Duration::from_secs(2)))
        .build()
        .ok()?
        .get(format!("{SERVER}/api/version"))
        .send()
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let identity: DaemonIdentity = resp.json().ok()?;
    match (&identity.version, &identity.exe) {
        (Some(version), Some(exe))
            if version == env!("LLMMAN_VERSION") && std::path::Path::new(exe).exists() =>
        {
            None
        }
        _ => Some(identity),
    }
}

/// Best-effort stop of a stale daemon (see `stale_daemon`), then waits for
/// the port to actually free up. Kills the daemon's whole process group —
/// it was started as a group leader (see `detach`/sbx's equivalent), so
/// this takes its spawned llama-server children down with it instead of
/// orphaning them with models still loaded in memory.
///
/// The stop sequence mirrors Ollama's app supervisor (app/server's
/// `stop()`): ask nicely first, poll for the port to free within a 5s
/// deadline, and only then escalate to a forceful kill — with one more 5s
/// wait before giving up entirely.
fn stop_stale_daemon(identity: &DaemonIdentity) -> anyhow::Result<()> {
    eprintln!("stopping stale llmman serve daemon (superseded by this install); restarting");
    kill_daemon(identity, false);
    if wait_for_port_free() {
        return Ok(());
    }
    // Graceful stop timed out — force-kill, like Ollama's supervisor
    // hard-killing a server that ignored its graceful signal for 5s.
    kill_daemon(identity, true);
    if wait_for_port_free() {
        return Ok(());
    }
    anyhow::bail!(
        "a stale llmman serve daemon is still holding 127.0.0.1:17434 after being asked to stop; \
         stop it manually (e.g. pkill -f 'llmman serve') and retry"
    )
}

/// Polls for the daemon port to free up, every 100ms for up to 5s.
fn wait_for_port_free() -> bool {
    for _ in 0..50 {
        if !server_alive() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    false
}

#[cfg(unix)]
fn kill_daemon(identity: &DaemonIdentity, force: bool) {
    let signal = if force { "-KILL" } else { "-TERM" };
    match identity.pid {
        // Negative pid = the whole process group (the daemon is its own
        // group leader), falling back to just the pid if that fails.
        Some(pid) => {
            let group = Command::new("kill")
                .args([signal, &format!("-{pid}")])
                .status();
            if !group.map(|s| s.success()).unwrap_or(false) {
                let _ = Command::new("kill")
                    .args([signal, &pid.to_string()])
                    .status();
            }
        }
        // A daemon too old to report its pid: match on the command line
        // ("<path>/llmman serve ..."), which no plain llmman CLI client
        // invocation shares.
        None => {
            let _ = Command::new("pkill")
                .args([signal, "-f", "llmman serve"])
                .status();
        }
    }
}

#[cfg(windows)]
fn kill_daemon(identity: &DaemonIdentity, _force: bool) {
    // taskkill /F and Stop-Process -Force are already forceful — Windows
    // has no in-between graceful signal for a detached, windowless
    // process, so the graceful/forceful distinction collapses here.
    match identity.pid {
        // /T takes the daemon's llama-server children down with it.
        Some(pid) => {
            let _ = Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/T", "/F"])
                .status();
        }
        None => {
            let _ = Command::new("powershell")
                .args([
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_Process -Filter \"Name='llmman.exe'\" | \
                     Where-Object { $_.CommandLine -match ' serve' } | \
                     ForEach-Object { Stop-Process -Id $_.ProcessId -Force }",
                ])
                .status();
        }
    }
}

/// Starts `llmman serve` as a background process if one isn't already
/// running, and waits (up to 60s) for it to start accepting connections.
/// The process is intentionally never stopped by this command — once
/// started it keeps running indefinitely, independent of this invocation,
/// so later commands (from this CLI or a concurrent one) reuse it instead
/// of starting a redundant copy.
///
/// stdin/stdout/stderr are all detached from this process (redirected to a
/// log file, or /dev/null if the log file can't be opened): the daemon
/// outlives this command, so anything that inherited its stdout/stderr
/// (a parent shell, a script capturing this command's output, `$(...)`,
/// etc.) would otherwise block forever waiting for those pipes to close,
/// since the daemon never exits to close them itself. The child is also
/// put in its own process group so terminal signals (e.g. Ctrl-C) sent to
/// this command's foreground process group don't reach it either.
///
/// `preload_model`, if non-empty, is passed through as `llmman serve`'s
/// optional positional argument so the daemon starts loading it
/// immediately (see cmd::serve::ServeArgs::model) instead of waiting for
/// the first request that references it. Pass "" when there's nothing to
/// preload — `run`/`pull`/`push` all pass "" so the daemon they spawn
/// stays a plain, model-agnostic `llmman serve` (only `launch` still
/// preloads, since its whole point is warming up one model for the
/// integration it's about to hand off to).
pub fn ensure_server(preload_model: &str) -> anyhow::Result<()> {
    if server_alive() {
        // Reuse the running daemon — unless it outlived its own install
        // (see stale_daemon): reusing that one means serving with a
        // long-obsolete build whose llama-server (and bug fixes) are gone.
        match stale_daemon() {
            None => return Ok(()),
            Some(identity) => stop_stale_daemon(&identity)?,
        }
    }
    let exe = std::env::current_exe().context("could not resolve own executable")?;

    let log_path = crate::default_store()
        .ok()
        .and_then(|store| store.parent().map(|p| p.join("serve.log")));

    // create_dir_all the log's parent (e.g. ~/.local/share/llmman) before
    // ever trying to open it below: `OpenOptions::create(true)` only
    // creates the *file*, never missing intermediate directories, so on a
    // genuinely fresh machine (nothing has ever pulled/served a model
    // yet — confirmed missing on a real macOS CI runner) the `.open()`
    // below would otherwise fail every single time, permanently losing
    // the daemon's entire stdout/stderr to the `None` branch's
    // `Stdio::null()` fallback instead of just this one first call.
    if let Some(p) = log_path.as_ref().and_then(|p| p.parent()) {
        let _ = std::fs::create_dir_all(p);
    }

    // Rotate the previous daemon's log out of the way (serve.log ->
    // serve-1.log -> ... -> serve-5.log) instead of appending forever,
    // the same scheme Ollama's app uses for server.log (app/logrotate):
    // each spawned daemon gets a fresh log, the last MAX_LOG_FILES
    // daemons' logs stay around for diagnostics, and nothing grows
    // without bound. Only rotated here, right before an actual spawn — a
    // call that reuses an already-running daemon never touches its log.
    if let Some(p) = &log_path {
        rotate_log(p);
    }

    let mut cmd = Command::new(&exe);
    cmd.arg("serve");
    if !preload_model.is_empty() {
        cmd.arg(preload_model);
    }
    cmd.stdin(Stdio::null());
    // Silently redirect the daemon's stdio to its log file (or /dev/null if
    // that file can't be opened) — no "starting serve" status line here:
    // every caller of ensure_server (run/pull/push/launch) wants to look
    // like a plain client of an already-running server, not announce that
    // it happened to be the one that started it this time. Anyone who
    // needs to know still can — see log_path above / `llmman serve.log`.
    match log_path
        .as_ref()
        .and_then(|p| std::fs::File::create(p).ok())
        .and_then(|f| f.try_clone().ok().map(|f2| (f, f2)))
    {
        Some((out, err)) => {
            cmd.stdout(out);
            cmd.stderr(err);
        }
        None => {
            cmd.stdout(Stdio::null());
            cmd.stderr(Stdio::null());
        }
    }
    detach(&mut cmd);
    cmd.spawn().context("spawn llmman serve")?;

    for _ in 0..120 {
        std::thread::sleep(Duration::from_millis(500));
        if server_alive() {
            return Ok(());
        }
    }
    anyhow::bail!("llmman serve did not start within 60s")
}

/// Puts the about-to-be-spawned child in its own process group (Unix) or
/// process group (Windows), so signals delivered to this process's
/// foreground process group (e.g. Ctrl-C in an interactive shell) don't
/// also terminate the daemon.
#[cfg(unix)]
fn detach(cmd: &mut Command) {
    use std::os::unix::process::CommandExt;
    cmd.process_group(0);
}

#[cfg(windows)]
fn detach(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    cmd.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
}

/// How many rotated generations of serve.log to keep — the same count as
/// Ollama's app/logrotate.MaxLogFiles.
const MAX_LOG_FILES: u32 = 5;

/// Rotates `path` (e.g. serve.log) through numbered generations
/// (serve-1.log ... serve-5.log), dropping the oldest — a direct port of
/// Ollama's app/logrotate.Rotate, which its supervisor runs on every
/// server spawn. Purely best-effort: a rotation failure must never stop
/// the daemon from starting, the worst case is just an appended log.
fn rotate_log(path: &std::path::Path) {
    if !path.exists() {
        return;
    }
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("serve");
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("log");
    let dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    for i in (1..=MAX_LOG_FILES).rev() {
        let older = dir.join(format!("{stem}-{i}.{ext}"));
        let newer = if i == 1 {
            path.to_path_buf()
        } else {
            dir.join(format!("{stem}-{}.{ext}", i - 1))
        };
        if newer.exists() {
            if older.exists() {
                let _ = std::fs::remove_file(&older);
            }
            let _ = std::fs::rename(&newer, &older);
        }
    }
}

// ---------------------------------------------------------------------------
// Windows-only: stop this process's own stdio handles from leaking into
// whatever it spawns
// ---------------------------------------------------------------------------

/// Root-caused a real Windows-only E2E hang: `cargo test`'s own harness
/// (tests/launch_e2e.rs's `spawn_with_timeout`) spawns `llmman run` with
/// piped stdout/stderr to capture its output live, and that child's own
/// `try_wait()`-detected exit was always reported correctly and promptly
/// — the hang was in the *next* step, `stdout_thread`/`stderr_thread`'s
/// `.join()`, which blocks on each reader thread's `read()` returning
/// `Ok(0)` (EOF). A pipe only reaches EOF once *every* handle to its
/// write end is closed — not just the one held by the process the reader
/// thinks it's reading from.
///
/// On Windows, any `CreateProcess` call that redirects a child's stdio via
/// a handle (a `File`, or another pipe — exactly what `ensure_server`
/// below does, redirecting the daemon's stdout/stderr to its log file)
/// requires `bInheritHandles = TRUE`, and that flag does not selectively
/// inherit only the handles you asked to redirect — it inherits *every*
/// handle in the parent's own handle table that's currently marked
/// inheritable. `llmman run`'s own stdout/stderr handles — received from
/// the test harness above specifically so they could be piped and
/// captured, which requires them to be inheritable in the first place —
/// are exactly such handles. So when `llmman run` calls `ensure_server`
/// and spawns the detached daemon below, that daemon process — which is
/// deliberately left running indefinitely — ends up with its own
/// duplicate, still-open handle to `llmman run`'s original stdout/stderr
/// pipes, even though its own stdio was separately, correctly redirected
/// to the log file. The pipe's write end therefore never fully closes,
/// the test harness's reader thread blocks on `read()` forever waiting
/// for an EOF that can now never come, and the *entire* E2E job times out
/// 45 minutes later with no error of its own — indistinguishable from
/// there being no bug at all up to that point, since `llmman run` itself
/// already exited successfully.
///
/// The fix: explicitly clear the inheritable flag on this process's own
/// three standard handles, once, as early as possible (see `main.rs`) —
/// before this process (or anything it calls into) ever spawns another
/// child with `bInheritHandles = TRUE` for any reason. This does not
/// affect this process's own use of those handles at all (only whether a
/// *future child* of this process can inherit them), and is safe to call
/// unconditionally even when nothing was ever piped in the first place
/// (`GetStdHandle` returning a real console handle, or even
/// `INVALID_HANDLE_VALUE`/null when there's no console at all, are all
/// handled as no-ops below).
#[cfg(windows)]
mod win_handles {
    use std::ffi::c_void;

    #[allow(non_snake_case)]
    extern "system" {
        fn GetStdHandle(nStdHandle: i32) -> *mut c_void;
        fn SetHandleInformation(hObject: *mut c_void, dwMask: u32, dwFlags: u32) -> i32;
    }

    const STD_INPUT_HANDLE: i32 = -10;
    const STD_OUTPUT_HANDLE: i32 = -11;
    const STD_ERROR_HANDLE: i32 = -12;
    const HANDLE_FLAG_INHERIT: u32 = 0x0000_0001;

    pub fn disable_std_handle_inheritance() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            for which in [STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE] {
                // SAFETY: GetStdHandle/SetHandleInformation are ordinary
                // kernel32 calls; `handle` is only ever passed to the one
                // API that accepts exactly this value (including the
                // documented NULL/INVALID_HANDLE_VALUE "no such stream"
                // cases, which SetHandleInformation simply fails on
                // harmlessly — the return value is deliberately ignored).
                unsafe {
                    let handle = GetStdHandle(which);
                    if !handle.is_null() && handle != usize::MAX as *mut c_void {
                        let _ = SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0);
                    }
                }
            }
        });
    }
}

/// See `win_handles`' own module doc comment. Must be called once, as
/// early as possible in `main()` — a no-op on every platform but Windows.
pub fn disable_std_handle_inheritance() {
    #[cfg(windows)]
    win_handles::disable_std_handle_inheritance();
}

/// A single line of Ollama's streamed NDJSON progress protocol (see
/// api.ProgressResponse) — status text plus an optional error, and
/// (unlike real Ollama's per-layer digest/total/completed) our own
/// aggregate total/completed byte counts across the whole pull/push, once
/// cmd::serve's stream_ffi_progress has one to report — see that
/// function's own doc comment for where these come from.
#[derive(Deserialize)]
struct ProgressLine {
    status: Option<String>,
    error: Option<String>,
    #[serde(default)]
    total: Option<u64>,
    #[serde(default)]
    completed: Option<u64>,
}

/// The indicatif template shared by `llmman pull`/`llmman push`'s
/// byte-level bar — deliberately similar in shape to `llmman transfer`'s
/// own mpb bars (go-shim/shared_oci.go's addLayerBar) so both commands'
/// output looks like the same family of progress bar.
///
/// The leading `{msg}` is the *reference* being pulled/pushed, not the
/// generic status word ("pulling"/"pushing") — that word was already
/// printed as its own plain line the moment byte-level reporting hadn't
/// started yet (see stream_progress), so repeating it here as the bar's
/// own label just showed "pulling" twice in a row, stacked directly on
/// top of each other, for no added information.
fn progress_bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{msg:<20} [{bar:32.cyan/blue}] {bytes:>10}/{total_bytes:<10} {bytes_per_sec:>12}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("=> ")
}

/// POSTs `{"model": reference}` to `path` (e.g. "/api/pull" or "/api/push")
/// on the local daemon and renders each streamed line to stderr as it
/// arrives: a real byte-level progress bar for any line carrying a nonzero
/// `total` (see ProgressLine), or a plain status line otherwise — matching
/// how `llmman transfer`'s own mpb bars render foreground FFI progress.
///
/// One case gets special-cased rather than handed to the animated bar:
/// llmman's local store is content-addressed, so a very common pull is
/// one where every blob is already on disk under a *different* reference
/// (see go-shim/backend_podman.go's ProgressEventSkipped handling) —
/// go.podman.io/image credits that instantly, so the very first
/// byte-count line this function ever sees already has completed==total.
/// Handing that straight to indicatif produced a bar that renders
/// "already 100% full" from its first frame (never visibly animating at
/// all — indistinguishable, at a glance, from a bar that's stuck) with a
/// `bytes_per_sec` figure computed as total/elapsed since the bar was
/// just created a moment ago — a wildly inflated number (multiple GiB/s
/// or even TiB/s) that then decays back down over the next several
/// polling ticks purely because the (fixed) numerator is being divided by
/// a (growing) denominator, not because anything is actually slowing
/// down. Nothing about that display was wrong, exactly, but it looked
/// exactly like a hang to a real reader, however briefly. Detecting that
/// exact shape up front and printing one plain "already have it" line
/// instead is both more honest (no bytes are moving) and reads instantly
/// as "done", not "stuck".
///
/// Returns an error if the stream reports one, or if it ends without ever
/// reporting "success".
pub fn stream_progress(path: &str, reference: &str) -> anyhow::Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(None) // model transfers can take much longer than any sane fixed timeout
        .build()
        .context("build http client")?;
    let resp = client
        .post(format!("{SERVER}{path}"))
        .json(&serde_json::json!({"model": reference}))
        .send()
        .with_context(|| format!("request {path} for {reference}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        anyhow::bail!("{path} {reference}: server returned {status}: {body}");
    }

    let mut saw_success = false;
    let mut last_status = String::new();
    let mut bar: Option<ProgressBar> = None;
    // Set once we've printed the "already have it" shortcut line (see this
    // function's own doc comment), so a pull whose bytes were all
    // instant-credited doesn't print that line again on every further
    // completed==total poll before "success" finally arrives.
    let mut printed_instant_complete = false;
    for line in std::io::BufReader::new(resp).lines() {
        let line = line.context("read response stream")?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(msg) = serde_json::from_str::<ProgressLine>(line) else {
            continue; // tolerate stray non-JSON keepalive output
        };
        if let Some(err) = msg.error.filter(|e| !e.is_empty()) {
            if let Some(b) = bar.take() {
                b.abandon(); // leave whatever was drawn in place instead of clearing it
            }
            // Only prefix with `reference` if the error doesn't already
            // mention it — many pull failures (e.g. containerd's "not
            // found") already embed the exact reference themselves, and
            // piling this prefix on unconditionally produced the same
            // reference two or three times over in one error line.
            if err.contains(reference) {
                anyhow::bail!("{err}");
            }
            anyhow::bail!("{reference}: {err}");
        }

        if let Some(total) = msg.total.filter(|&t| t > 0) {
            let completed = msg.completed.unwrap_or(0).min(total);

            // Instant-credit shortcut (see this function's own doc
            // comment): the *first* poll this function ever sees a
            // nonzero total on is already ≥99% complete, so whatever
            // sliver (if any) is still outstanding isn't a real,
            // animatable transfer — it's noise from total/completed being
            // credited by two separate calls a poll interval apart on the
            // Go side (go-shim/progress_state.go's progressAddTotal/
            // progressAddCompleted), not a meaningful amount of remaining
            // work. A genuine transfer's first-ever byte line is nowhere
            // near this close to done (that's what makes the animated
            // bar below worth showing at all). Checked only while
            // bar.is_none(): once a real bar has been created because an
            // earlier poll didn't clear this bar, later polls update it
            // normally even if they happen to also reach ≥99%.
            //
            // 99%, not 100% exactly: an earlier version of this required
            // completed==total on the nose and missed a real case where
            // two multi-hundred-MB blobs were skip-credited instantly but
            // a trailing few-hundred-byte config blob's own (separate,
            // genuinely tiny) transfer hadn't landed yet on that same
            // poll — 99.99996% done, not 100.000% done, but just as
            // clearly "nothing left worth animating" either way.
            if bar.is_none() && completed * 100 >= total * 99 {
                if !printed_instant_complete {
                    println!(
                        "Already have {reference} ({})",
                        crate::fmt::human_size(total)
                    );
                    printed_instant_complete = true;
                }
                last_status = "pulling".to_string(); // suppress a redundant plain "pulling" line below
                continue;
            }

            // A byte-level progress line: render/update the bar instead of
            // printing a new line for every update.
            let pb = bar.get_or_insert_with(|| {
                let pb = ProgressBar::new(total);
                pb.set_style(progress_bar_style());
                pb.set_message(reference.to_string());
                pb
            });
            pb.set_length(total);
            pb.set_position(completed);
            continue;
        }
        // No byte counts on this line: finish/clear any bar in progress
        // before falling back to plain status text, so the two don't
        // interleave on the same terminal lines.
        if let Some(b) = bar.take() {
            b.finish_and_clear();
        }
        if let Some(status) = msg.status {
            if !status.is_empty() && status != last_status {
                println!("{status}");
                last_status = status;
            }
            saw_success = last_status == "success";
        }
    }
    if let Some(b) = bar.take() {
        b.finish_and_clear();
    }
    if !saw_success {
        anyhow::bail!("{reference}: stream ended without a success status");
    }
    Ok(())
}

/// POSTs `{"model": reference}` to `/api/show` and reports whether the
/// daemon's local store already has it — a read-only existence check with
/// no download/pull side effects.
fn model_exists(reference: &str) -> anyhow::Result<bool> {
    let resp = reqwest::blocking::Client::new()
        .post(format!("{SERVER}/api/show"))
        .json(&serde_json::json!({"model": reference}))
        .send()
        .with_context(|| format!("request /api/show for {reference}"))?;
    Ok(resp.status().is_success())
}

/// Ensures `reference` is present in the daemon's local store, pulling it
/// (and streaming progress the same way `llmman pull` does) if it isn't.
///
/// Mirrors ollama's `RunHandler`, which calls `client.Show` before ever
/// entering the interactive/one-shot prompt loop and only falls back to
/// `PullHandler` on a miss — so a bad reference (typo'd tag, malformed
/// `hf.co/...` name, etc.) is reported and aborts the command immediately,
/// instead of only surfacing once the first message is submitted to
/// `/api/chat` (by which point the interactive `> ` prompt has already
/// been shown and read from).
pub fn ensure_model_pulled(reference: &str) -> anyhow::Result<()> {
    if model_exists(reference).unwrap_or(false) {
        return Ok(());
    }
    stream_progress("/api/pull", reference)
}

/// POSTs the Ollama unload sentinel (`{"model": reference, "keep_alive":
/// 0}`, no `prompt` field — i.e. an empty prompt) to `/api/generate` —
/// see `cmd::serve`'s `handle_ollama_generate` for the server side that
/// reads this exact shape as an immediate-unload request, mirroring real
/// Ollama's own `ollama stop` (`cmd/cmd.go`'s `loadOrUnloadModel`). Used
/// by `llmman stop`.
pub fn unload(reference: &str) -> anyhow::Result<()> {
    let resp = reqwest::blocking::Client::new()
        .post(format!("{SERVER}/api/generate"))
        .json(&serde_json::json!({"model": reference, "keep_alive": 0}))
        .send()
        .with_context(|| format!("request /api/generate (unload) for {reference}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        anyhow::bail!("stop {reference}: server returned {status}: {body}");
    }
    Ok(())
}

/// A plain `GET {SERVER}{path}` returning the parsed JSON body — for
/// callers (currently just `ps`) that don't need `stream_progress`'s
/// newline-delimited-JSON streaming, just a single request/response.
pub fn get_json<T: serde::de::DeserializeOwned>(path: &str) -> anyhow::Result<T> {
    let resp = reqwest::blocking::get(format!("{SERVER}{path}"))
        .with_context(|| format!("request {path}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        anyhow::bail!("{path}: server returned {status}: {body}");
    }
    resp.json()
        .with_context(|| format!("parse response from {path}"))
}
