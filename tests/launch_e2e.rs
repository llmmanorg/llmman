//! End-to-end tests for `llmman launch <integration> --model qwen3.5:0.8b`.
//!
//! These exercise the real launch path: a real, auto-started `llmman
//! serve`, a real pulled model (`docker.io/ai/qwen3.5:0.8b`, resolved
//! from the bare short name the same way `llmman launch`/`pull` always
//! resolve one — see `shortnames::resolve_ollama_api`), a real
//! `llama-server` backing it, and the real third-party CLI under test
//! (`claude`, `agy`, `opencode`, `pi`, `omp`, `codex`, `copilot`, `droid`, `cline`,
//! `grok`, `qwen`, `hermes`,
//! `openclaw`, `dsh`, `goose`) — not mocks. The one exception is
//! [`launch_goose_desktop_env`]: Goose Desktop is a GUI with no headless
//! mode, so it stubs its binary.
//! That's the only way this actually verifies anything: every one of the
//! three bugs this file's tests were written to catch (see below) only
//! ever showed up against the real binaries, never in isolation.
//!
//! Each test prints a message and skips (rather than failing) when a
//! prerequisite isn't installed in the current environment, since none of
//! that is under this crate's control:
//!
//!   - `llama-server` on PATH — required by every one of these (llmman's
//!     daemon can't serve any model without it).
//!   - the specific integration binary under test on PATH.
//!
//! Network access to pull `docker.io/ai/qwen3.5:0.8b` (~740MB, on the
//! first run only — later runs reuse whatever's already in the daemon's
//! store) is assumed available and NOT treated as skippable: a pull
//! failure is a real failure here, not an environment-setup gap.
//!
//! [`serve_mlx_safetensors_model`], [`serve_vllm_safetensors_model`] and
//! [`serve_sglang_safetensors_model`] are the tests in this file that
//! aren't about a third-party integration at all: they exercise `llmman
//! serve`'s own safetensors backends — `mlx_lm.server` (see
//! `cmd::serve::backend::use_mlx_for_safetensors`), `vllm` and `sglang`
//! (`LLMMAN_SAFETENSORS_ENGINE`) — directly via `llmman run`, against a
//! real safetensors model. Like every other test here they skip
//! themselves — rather than failing — when their own prerequisite (the
//! engine on `PATH`; for mlx, Apple Silicon macOS too) isn't met. The
//! vllm and sglang ones run a daemon of their own on a fresh port (the
//! engine choice is daemon-wide environment) — see
//! `serve_with_engine`. So does [`serve_ds4_gguf_model`], for
//! `ds4-server`.
//!
//! `llmman serve` is a process-wide singleton bound to a single loopback
//! port (127.0.0.1:17434 by default, or wherever `LLMMAN_HOST` points —
//! see `daemon::server`/`daemon::bind_addr`), so these tests can't
//! isolate their own daemon instance from each other or from one already
//! running on the machine: `llmman launch` (via `daemon::ensure_server`)
//! just reuses whatever's already listening there, preloaded with
//! whichever model first asked for it. What each test *does* isolate is
//! `HOME` (and so each integration's own config directory), by pointing
//! its child process at a fresh temp directory. `SERIAL` below keeps
//! these tests from running concurrently regardless, both to avoid
//! racing to spawn that one shared daemon and to keep real model
//! launches from competing for the same GPU/CPU at once.
//!
//! Not run as part of `cargo build`, and not run by the `test` job in
//! `.github/workflows/ci.yml` (that job only runs the in-crate
//! `#[cfg(test)]` unit tests) — this file is its own separate `e2e` job
//! there instead. To run it locally, invoke it explicitly:
//!
//!   cargo test --release --test launch_e2e -- --nocapture --test-threads=1
//!
//! # Regressions this file guards against
//!
//! All three were found by actually running these exact commands against
//! a real model, not by inspection:
//!
//!   - `claude`: real Claude Code sessions inject a second `role:"system"`
//!     message later in the conversation (e.g. an available-agents/skills
//!     reminder) in addition to its leading system prompt. Qwen3.5's chat
//!     template raises a hard Jinja error ("System message must be at the
//!     beginning") the moment that happens, so every real multi-turn
//!     request 500'd and Claude Code retried in a loop until giving up —
//!     fixed in `cmd::serve::anthropic::handle_anthropic_messages` by
//!     folding every system-role turn into one leading message.
//!   - `codex`: the config `codex::write_codex_config` wrote (a `[profiles.llmman]`
//!     table in `config.toml`) is a format current codex (0.134+) refuses
//!     to load at all — fixed by writing the sibling
//!     `~/.codex/llmman.config.toml` overlay codex now expects instead.
//!   - `codex`: real `codex exec` always includes Responses-API tool
//!     entries llama-server's `/v1/responses` rejects outright
//!     (`"'type' of tool must be 'function'"` for a `"namespace"`-typed
//!     sub-agent tool bundle, and for the bare `{"type":"web_search"}`
//!     entry) — fixed in
//!     `cmd::serve::responses::filter_non_function_tools`. Real `codex
//!     exec` also always carries a `developer`-role item alongside its
//!     top-level `instructions`, which llama-server's own Responses
//!     conversion turns into a second, misplaced `system`-role chat
//!     message (a confirmed, unresolved upstream llama.cpp gap — see
//!     ggml-org/llama.cpp#20733/#23423) — fixed in
//!     `cmd::serve::responses::consolidate_responses_instructions`.

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, Once, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// The exact short name every `llmman launch ... --model` invocation in
/// this suite uses — resolves (via `resolve_ollama_api`, the same path
/// real CLI usage takes) to `docker.io/ai/qwen3.5:0.8b`, a ~740MB Q4_K_M
/// quantization small enough to pull and run within a normal test
/// timeout.
const MODEL: &str = "qwen3.5:0.8b";

/// A short, literal prompt every integration is asked to answer. Kept
/// deliberately simple: the small quantized model's answer quality isn't
/// what's under test here, the launch/env/config plumbing is.
const PROMPT: &str = "Reply with exactly the single word: pong";

/// How long a single command (an `llmman launch` invocation, or
/// `warm_model`'s `llmman run`) may run before these tests give up and
/// fail it as hung, rather than waiting forever. This has to comfortably
/// cover a cold pull of the ~740MB model plus llama-server startup, which
/// only happens once per fresh daemon (in `warm_model`, not in every
/// individual test) — but that one pull is a real download over
/// whatever network the daemon's machine has, so this is generous rather
/// than tight: a CI run has already hit the low end of plausible pull
/// times at 300s.
const TIMEOUT: Duration = Duration::from_secs(600);

/// Serializes the tests in this file: they'd otherwise run in parallel
/// threads (the default `cargo test` behavior) and race to spawn
/// `llmman serve` for the one shared daemon slot every `llmman launch`
/// invocation targets (see the module doc comment).
static SERIAL: Mutex<()> = Mutex::new(());

/// Locks [`SERIAL`], recovering from poisoning instead of panicking with
/// an opaque `PoisonError` (via the default `.lock().unwrap()`) — this
/// mutex only ever guards *ordering*, never any data an earlier panicking
/// test could have left inconsistent (its payload is `()`), so there's
/// nothing to actually protect against here. Without this, one test
/// legitimately failing (including via `launch_and_assert`'s own retries
/// being exhausted) poisons the lock for the remaining two, which then
/// fail with a meaningless `PoisonError` instead of ever getting to run
/// and report their own real result — exactly what happened live in CI:
/// a `launch_claude_with_model` failure took down `launch_codex_with_model`
/// and `launch_opencode_with_model` too, hiding whether either of *those*
/// would have actually passed on their own.
fn lock_serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Guards `warm_model` so its real (slow, first-time) work happens only
/// once per test binary run, no matter how many of the three tests below
/// actually reach it.
static WARM: Once = Once::new();

/// The window `warm_model` loaded `MODEL` with; see [`restore_model_window`].
static DEFAULT_WINDOW: OnceLock<u64> = OnceLock::new();

/// The `llmman` binary every test in this file launches.
///
/// CI (see `.github/workflows/ci.yml`'s e2e job) installs the exact
/// binary `cargo build` just produced via the real `install.sh`/
/// `install.ps1` scripts (fetched from https://llmmanorg.github.io/)
/// before running this suite, then points
/// `LLMMAN_E2E_BIN` at the resulting installed copy (`~/.local/bin/llmman`
/// on Unix, `%LOCALAPPDATA%\Microsoft\WindowsApps\llmman.exe` on Windows)
/// — so what's actually under test is the same installed artifact a real
/// user's `curl ... | sh`/`irm ... | iex` produces, not just whatever
/// `cargo build` itself dropped in `target/`, closing the gap between
/// this suite and how llmman is actually obtained in practice.
///
/// Falls back to Cargo's own `CARGO_BIN_EXE_llmman` (the pre-existing
/// behavior) when that env var isn't set, so a plain local
/// `cargo test --release --test launch_e2e` (see this file's own module
/// doc comment) keeps working without requiring the installer to have
/// been run first.
fn llmman_bin() -> PathBuf {
    match std::env::var_os("LLMMAN_E2E_BIN") {
        Some(path) => PathBuf::from(path),
        None => PathBuf::from(env!("CARGO_BIN_EXE_llmman")),
    }
}

/// True if `bin` resolves on `PATH` — enough for test-skip purposes
/// without depending on llmman's own (private) `launch::find_on_path`.
/// Mirrors that function's own Windows handling (see its doc comment):
/// a bare name on Windows is checked against `.exe`/`.cmd`/`.bat`, not
/// just the bare file itself, since every integration under test here is
/// installed via `npm install -g` — which on Windows never produces a
/// bare, extension-less file — and this needs to agree with what
/// `find_on_path` can actually locate. A test skipping here despite the
/// CLI being on `PATH` (or, worse, not skipping and then hitting
/// `find_on_path`'s own "not installed" error inside `llmman launch`)
/// would both be this check silently drifting from that one.
fn on_path(bin: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    if cfg!(windows) {
        const EXTS: &[&str] = &["exe", "cmd", "bat"];
        std::env::split_paths(&path).any(|dir| {
            EXTS.iter()
                .any(|ext| dir.join(format!("{bin}.{ext}")).is_file())
        })
    } else {
        std::env::split_paths(&path).any(|dir| dir.join(bin).is_file())
    }
}

/// A fresh, unique temp `HOME` for one test's child process — isolates
/// each integration's own config directory (`~/.claude`, `~/.codex`,
/// `~/.config/opencode`) both from the real developer's and from the
/// other tests in this file.
fn fresh_home(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("llmman-e2e-{label}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("create temp HOME");
    dir
}

/// Returns the last `n` bytes of `buf` as a lossy string, prefixed with an
/// ellipsis marker when truncated — used by spawn_with_timeout's
/// heartbeat to show recent output without the printed heartbeat itself
/// growing unboundedly as a slow child produces more and more of it.
/// (The "bytes total" in that marker is really "bytes retained": the
/// buffer itself is capped — see [`READER_BUF_CAP`].)
fn tail_str(buf: &VecDeque<u8>, n: usize) -> String {
    let tail: Vec<u8> = buf
        .iter()
        .skip(buf.len().saturating_sub(n))
        .copied()
        .collect();
    if buf.len() > n {
        format!(
            "...<{} bytes total>...{}",
            buf.len(),
            String::from_utf8_lossy(&tail)
        )
    } else {
        String::from_utf8_lossy(&tail).into_owned()
    }
}

/// Kills `child` *and every descendant it spawned*, not just the direct
/// child process itself. `Child::kill` alone is not enough for any
/// command in this file: `llmman launch <integration>` execs a real
/// third-party CLI (which itself spawns more — codex runs a `node`
/// child), and `llmman run` auto-starts a daemon. Killing only the
/// direct child leaves those descendants running — and, worse, still
/// holding inherited handles to the stdout/stderr pipes this harness
/// reads, so the pipes never reach EOF (see `collect_reader`).
///
/// Directly observed in CI (x86_64 linux/docker, run 31772215140): a
/// `codex exec` whose small model degenerated into an endless agent loop
/// survived `child.kill()` of its `llmman launch` parent at the 600s
/// timeout, kept the pipe open, and the old unconditional
/// `stdout_thread.join()` then blocked forever — so the timeout panic
/// (and all its diagnostics) never fired and the whole job ran into the
/// workflow's 45-minute kill instead. Same failure class as the Windows
/// handle-inheritance hang documented at length in `src/daemon.rs`
/// (`disable_std_handle_inheritance`), new vector.
///
/// Unix: the child is spawned into its own process group (see
/// `spawn_with_timeout`), so one `kill(-pgid, SIGKILL)` takes out the
/// entire tree atomically — every descendant is in that group unless it
/// deliberately escaped (llmman's own daemon does, by design; it doesn't
/// inherit these pipes on any platform, so it can't wedge the readers).
/// Windows: `taskkill /T /F` walks and kills the tree by parent PID.
fn kill_process_tree(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        // The child was put in its own process group via
        // `process_group(0)`, so its pgid == its pid.
        unsafe {
            libc::kill(-(child.id() as i32), libc::SIGKILL);
        }
    }
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// Upper bound on how many bytes of one stream's output a reader thread
/// retains — generous (a passing run of anything in this file produces
/// kilobytes, even with opencode's `--log-level DEBUG` on), but *finite*,
/// because a reader thread's lifetime isn't reliably bounded: when
/// `collect_reader` gives up on a thread still blocked in `read()`, that
/// thread is detached, not terminated (safe Rust has no way to kill it),
/// and it keeps appending for as long as some descendant that escaped
/// `kill_process_tree` keeps the pipe's write end open and keeps
/// writing. That was always true, but it used to not matter: a timeout
/// panicked, the test process died, and any detached reader died with
/// it. Now that `launch_and_assert` *retries* a timed-out attempt, a
/// detached reader from a previous attempt can outlive it (each attempt
/// gets fresh pipes/buffers/threads, so its output can never leak into a
/// later attempt's assertions — only this cap's worth of memory), so an
/// endlessly-writing zombie must be able to cost at most this much, not
/// grow without bound for the rest of the test binary's life.
const READER_BUF_CAP: usize = 4 * 1024 * 1024;

/// Appends `chunk` to `buf`, discarding the *oldest* bytes to stay under
/// [`READER_BUF_CAP`] — the newest output is what every consumer here
/// wants: the heartbeat and the timeout diagnostics print tails, and the
/// "pong" assertion looks for the model's final answer.
///
/// `buf` is a `VecDeque` precisely so this stays cheap once the cap is
/// hit: dropping bytes off a `VecDeque`'s front just advances its head
/// index (it's a ring buffer), where the equivalent `Vec::drain(..n)`
/// would memmove the entire ~4 MiB retained tail down on *every* append
/// — per 4 KiB read, a ~1000x write amplification handed to exactly the
/// endlessly-writing zombie descendant this cap exists to contain,
/// converting the unbounded-memory problem into a busy-CPU one.
fn append_capped(buf: &Mutex<VecDeque<u8>>, chunk: &[u8]) {
    let mut buf = buf.lock().unwrap();
    // A single chunk larger than the whole cap can't happen with the 4 KiB
    // reads below, but don't rely on that from here.
    let chunk = &chunk[chunk.len().saturating_sub(READER_BUF_CAP)..];
    let excess = (buf.len() + chunk.len()).saturating_sub(READER_BUF_CAP);
    buf.drain(..excess);
    buf.extend(chunk);
}

/// Spawns the background thread that drains one of the child's output
/// pipes into a shared (capped — see [`append_capped`]) buffer until EOF
/// or a read error. See the comment above the call sites in
/// `try_spawn_with_timeout` for why draining must happen live on a
/// thread rather than after the child exits.
fn spawn_reader(
    mut pipe: impl Read + Send + 'static,
    buf: Arc<Mutex<VecDeque<u8>>>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut chunk = [0u8; 4096];
        loop {
            match pipe.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => append_capped(&buf, &chunk[..n]),
                // A signal landing mid-read isn't EOF — retry, per the
                // standard EINTR contract.
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                // Any other error still ends draining (there's nothing
                // useful to do with a broken pipe from here, and the
                // output captured so far is worth returning either way),
                // but say so: a truncated capture that *looked* like a
                // clean EOF would send whoever reads the resulting
                // diagnostics down the wrong path.
                Err(e) => {
                    eprintln!("[spawn_reader] pipe read failed (treating as EOF): {e}");
                    break;
                }
            }
        }
    })
}

/// Collects one reader thread's buffer, waiting at most `grace` for the
/// thread to see EOF — never unboundedly, because EOF is only guaranteed
/// if *every* process holding the pipe's write end has exited, which is
/// exactly the invariant that has now failed twice in this suite's
/// history (the Windows daemon handle-inheritance leak in
/// `src/daemon.rs`, and the surviving-`codex`-descendant hang this
/// bounded wait was added for — see `kill_process_tree`). If the thread
/// hasn't finished within `grace`, it's left detached (it parks in a
/// blocking `read()` costing nothing) and whatever output made it into
/// the shared buffer so far is returned — losing at most the final
/// unflushed chunk from a process that's being killed anyway, instead of
/// losing the timeout panic (and the whole job) to an unkillable wait.
fn collect_reader(
    thread: std::thread::JoinHandle<()>,
    buf: &Arc<Mutex<VecDeque<u8>>>,
    grace: Duration,
) -> Vec<u8> {
    let deadline = Instant::now() + grace;
    while !thread.is_finished() {
        if Instant::now() >= deadline {
            eprintln!(
                "[collect_reader] reader thread still blocked after {grace:?} — \
                 a killed process's descendant is likely still holding the pipe's \
                 write end; proceeding with the output captured so far"
            );
            return buf.lock().unwrap().iter().copied().collect();
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    thread.join().expect("join reader thread");
    Vec::from(std::mem::take(&mut *buf.lock().unwrap()))
}

/// A launch attempt that hit its deadline and was killed — carries the
/// fully formatted diagnostics (description, timeout, and both output
/// tails) so the caller can either panic with them verbatim
/// ([`spawn_with_timeout`]) or print them and retry
/// ([`launch_and_assert`]).
struct TimedOut {
    message: String,
}

/// Runs `cmd`, waiting up to `timeout` and killing (then panicking with
/// `description` in the message) it — *and its entire process tree*, see
/// `kill_process_tree` — if it hasn't exited by then. Thin panicking
/// wrapper over [`try_spawn_with_timeout`], for callers (`warm_model`)
/// for which a timeout is immediately fatal rather than retryable.
fn spawn_with_timeout(cmd: Command, timeout: Duration, description: &str) -> std::process::Output {
    try_spawn_with_timeout(cmd, timeout, description)
        .unwrap_or_else(|timed_out| panic!("{}", timed_out.message))
}

/// Runs `cmd`, waiting up to `timeout` and killing it — *and its entire
/// process tree*, see `kill_process_tree` — if it hasn't exited by then,
/// returning `Err(TimedOut)` instead of panicking so the caller decides
/// whether a timeout is fatal or retryable. stdout/stderr are drained on
/// background threads while polling for exit, rather than read only
/// after the child finishes, so a chatty child can't deadlock on a full
/// pipe buffer before the timeout ever gets a chance to fire.
fn try_spawn_with_timeout(
    mut cmd: Command,
    timeout: Duration,
    description: &str,
) -> Result<std::process::Output, TimedOut> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // Own process group (Unix): makes the whole descendant tree killable
    // as one unit at the timeout — see kill_process_tree.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    let mut child = cmd
        .spawn()
        .unwrap_or_else(|e| panic!("spawn {description}: {e}"));
    // Kept permanently (not removed once the investigation that added it
    // concluded — see this repo's own git history): a real Windows/macOS
    // hang in this file once showed only a bare "still running" heartbeat
    // with no insight into what the child itself was doing at that point
    // — its own stdout/stderr used to only become visible once it exited
    // (or was killed at the timeout), by which point a forceful GH
    // Actions job cancellation (the outer timeout-minutes, not this
    // function's own) can lose the last several minutes of log output
    // entirely. Reading into shared buffers the heartbeat can peek at
    // live (rather than only handing them back once each reader thread's
    // `read_to_end` finally returns) means a hung/slow child's progress
    // up to the very last heartbeat before that happens stays visible
    // even if everything after it is lost — useful for whatever the next
    // one of these turns out to be, not just the one this was built for.
    eprintln!(
        "[spawn_with_timeout] pid={} spawned: {description}",
        child.id()
    );
    let stdout_pipe = child.stdout.take().expect("child stdout");
    let stderr_pipe = child.stderr.take().expect("child stderr");
    let stdout_buf = Arc::new(Mutex::new(VecDeque::new()));
    let stderr_buf = Arc::new(Mutex::new(VecDeque::new()));
    let stdout_thread = spawn_reader(stdout_pipe, Arc::clone(&stdout_buf));
    let stderr_thread = spawn_reader(stderr_pipe, Arc::clone(&stderr_buf));

    let start = Instant::now();
    let mut last_heartbeat = start;
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll child status") {
            eprintln!(
                "[spawn_with_timeout] pid={} exited after {:?}: {description}",
                child.id(),
                start.elapsed()
            );
            break status;
        }
        if last_heartbeat.elapsed() > Duration::from_secs(30) {
            last_heartbeat = Instant::now();
            eprintln!(
                "[spawn_with_timeout] pid={} still running after {:?}: {description}\n  stdout tail: {:?}\n  stderr tail: {:?}",
                child.id(),
                start.elapsed(),
                tail_str(&stdout_buf.lock().unwrap(), 300),
                tail_str(&stderr_buf.lock().unwrap(), 300),
            );
        }
        if start.elapsed() > timeout {
            // Kill the *whole tree*, then collect (bounded — see
            // collect_reader) whatever both readers captured before
            // panicking: what the process printed before it got stuck is
            // exactly what's needed to tell "genuinely still downloading/
            // loading a large model" apart from "stuck in a real hang"
            // from the outside, instead of a bare timeout message that
            // can't distinguish either. Neither step here may block
            // unboundedly: this panic *is* the test's failure report, and
            // anything that can postpone it indefinitely turns a 10-minute
            // failure into a diagnostics-free 45-minute job cancellation
            // (which is precisely what the old unconditional join did —
            // see kill_process_tree).
            kill_process_tree(&mut child);
            let stdout = String::from_utf8_lossy(&collect_reader(
                stdout_thread,
                &stdout_buf,
                Duration::from_secs(5),
            ))
            .into_owned();
            let stderr = String::from_utf8_lossy(&collect_reader(
                stderr_thread,
                &stderr_buf,
                Duration::from_secs(5),
            ))
            .into_owned();
            return Err(TimedOut {
                message: format!(
                    "{description} did not finish within {timeout:?} — likely a hang\n\
                     --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
                ),
            });
        }
        std::thread::sleep(Duration::from_millis(200));
    };

    // Bounded here too (generously — the child has already *exited*, so
    // EOF is normally immediate): a well-behaved run must not lose output
    // to a tight grace period, but a leaked write end from some surviving
    // grandchild must not be able to wedge a *successful* run either.
    let stdout = collect_reader(stdout_thread, &stdout_buf, Duration::from_secs(30));
    let stderr = collect_reader(stderr_thread, &stderr_buf, Duration::from_secs(30));
    Ok(std::process::Output {
        status,
        stdout,
        stderr,
    })
}

/// Forces `MODEL` to be pulled and fully loaded — via `llmman run`, our
/// own trusted client, never a third-party CLI — before any of this
/// file's tests hand off to one. Real third-party AI-SDK-based clients
/// (opencode's included) have been observed to retry a failed connection
/// indefinitely rather than giving up on it, so if one of them happens to
/// be the first to reach the daemon while `MODEL` is still on its way in
/// (a cold pull of ~740MB, plus llama-server startup), any transient
/// connection hiccup during that window risks wedging that client forever
/// — a failure this suite can only detect (via each test's own `TIMEOUT`),
/// never recover from. Paying that cold-start cost here first, against
/// code this suite controls, removes the window entirely: by the time any
/// test execs a real integration, `MODEL` has already answered a real
/// prompt successfully at least once.
///
/// `--think false`: qwen3.5's thinking mode has been observed (directly,
/// against a real windows-2025/macos-15 run — see this repo's own git
/// history) to occasionally degenerate into repeating the same handful of
/// reasoning sentences indefinitely instead of ever reaching an answer,
/// hanging this warm-up for the full `TIMEOUT` and poisoning `WARM` for
/// every other test in the same run. That failure mode lives entirely
/// inside the "Thinking:" block this flag skips outright.
///
/// `--num-predict 64`: disabling thinking alone turned out not to be
/// enough — a *second*, real windows-2025 run (this one with `--think
/// false` already in effect) still hung the full `TIMEOUT`, this time
/// repeating a single token in the actual answer instead of inside a
/// thinking block. Nothing about *why* a small quantized model's sampling
/// might degenerate is reliably preventable from here; a hard ceiling on
/// how many tokens it's even allowed to generate is. 64 is generous for
/// this file's own PROMPT (a literal one-word answer) while still capping
/// a worst-case degenerate run at a few seconds, not `TIMEOUT`'s full 600.
///
/// Neither flag is used for the three real per-integration launches
/// below: they exercise real third-party clients exactly as a real user
/// would run them, unable to pass either flag those clients don't
/// themselves expose.
fn warm_model() {
    WARM.call_once(|| {
        eprintln!("[warm_model] starting");
        let output = reload_model("llmman run (model warm-up)")
            .unwrap_or_else(|message| panic!("{message}"));
        eprintln!("[warm_model] done, status={:?}", output.status);
        assert!(
            output.status.success(),
            "llmman run {MODEL} {PROMPT:?} (model warm-up) failed (status: {:?})\n\
             --- stdout ---\n{}\n--- stderr ---\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        // Only a measured window is a baseline, so a transient miss is retried.
        let window = (0..3).find_map(|attempt| {
            if attempt > 0 {
                std::thread::sleep(Duration::from_secs(1));
            }
            llmman::daemon::running_context_length(&model_reference())
        });
        match window {
            Some(window) => {
                let _ = DEFAULT_WINDOW.set(window);
            }
            None => eprintln!("[warm_model] no context window reported; not restoring it"),
        }
    });
    restore_model_window();
}

/// `MODEL` as the daemon names it, which is how `/api/ps` lists it.
fn model_reference() -> String {
    llmman::shortnames::resolve_ollama_api(MODEL).expect("MODEL resolves")
}

/// Unloads `MODEL` and loads it again with `llmman run`, at the daemon's
/// own window. A live daemon that will not unload it would hand back the
/// old window, so that is an error; no daemon yet is fine, the run starts one.
fn reload_model(description: &str) -> Result<std::process::Output, String> {
    if llmman::daemon::server_alive() {
        llmman::daemon::unload(&model_reference())
            .map_err(|error| format!("unloading {MODEL} failed: {error:#}"))?;
    }
    run_model_prompt(TIMEOUT, description).map_err(|timed_out| timed_out.message)
}

/// Reloads `MODEL` unless it has at least the window `warm_model` gave it.
///
/// The daemon is shared, and a request that names `num_ctx` reloads the
/// model at that window for every later request too (#684). Cline's Ollama
/// provider sends 32768, so later tests ran at an eighth of the warmed
/// 262144: Hermes refuses a window under 64K, and the qwen test's prompts
/// overflow it. A failed reload is logged and left for the test to hit.
fn restore_model_window() {
    let Some(&default) = DEFAULT_WINDOW.get() else {
        return;
    };
    let model = model_reference();
    // Read-only. An unknown window (idle, or `/api/ps` failed) reloads too,
    // which loads an idle model at the default.
    let window = llmman::daemon::running_context_length(&model);
    if window.is_some_and(|window| window >= default) {
        return;
    }
    eprintln!(
        "[restore_model_window] {model} has a window of {window:?}, want {default}+; reloading"
    );
    match reload_model("llmman run (model window restore)") {
        Ok(output) if output.status.success() => {}
        Ok(output) => eprintln!(
            "[restore_model_window] reload failed (status: {:?})\n--- stderr ---\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ),
        Err(message) => eprintln!("[restore_model_window] {message}"),
    }
    eprintln!(
        "[restore_model_window] window is now {:?}",
        llmman::daemon::running_context_length(&model)
    );
}

/// `llmman run MODEL PROMPT` with `warm_model`'s guard flags, bounded by
/// `timeout`. Shared by `warm_model` and `daemon_still_answers`.
fn run_model_prompt(
    timeout: Duration,
    description: &str,
) -> Result<std::process::Output, TimedOut> {
    let mut cmd = Command::new(llmman_bin());
    cmd.arg("run")
        .arg(MODEL)
        .arg("--think")
        .arg("false")
        .arg("--num-predict")
        .arg("64")
        .arg(PROMPT);
    try_spawn_with_timeout(cmd, timeout, description)
}

/// After a launch was killed at `TIMEOUT`: does the daemon still answer
/// our own client? Yes means the third-party CLI was the slow party (a
/// slow runner or a degenerate sampling loop); no means a daemon stall,
/// which a strict test must report.
fn daemon_still_answers(integration: &str) -> bool {
    match run_model_prompt(
        Duration::from_secs(180),
        &format!("llmman run (daemon probe after {integration} timed out)"),
    ) {
        Ok(output) if output.status.success() => true,
        Ok(output) => {
            eprintln!(
                "[test] {integration}: daemon probe failed (status: {:?})\n--- stderr ---\n{}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            );
            false
        }
        Err(timed_out) => {
            eprintln!(
                "[test] {integration}: daemon probe hung\n{}",
                timed_out.message
            );
            false
        }
    }
}

/// The shared daemon's output from one launch attempt on: its llama-server
/// timings, the only record of why a launch was slow. Each attempt runs
/// under a fresh `HOME`, but the daemon does not: on Linux CI it is the
/// install script's systemd user unit (`llmman.service`), whose output
/// goes to the journal; elsewhere it is whatever `ensure_server` spawned,
/// writing `serve.log` under this process's own data directory.
struct DaemonLog {
    /// `journalctl --since` for the systemd unit, when it is active.
    journal_since: Option<String>,
    /// `serve.log`, its length and its first bytes when the attempt began.
    serve_log: Option<(PathBuf, u64, Vec<u8>)>,
}

impl DaemonLog {
    fn mark() -> Self {
        let journal_since = (cfg!(target_os = "linux")
            && Command::new("systemctl")
                .args(["--user", "is-active", "--quiet", "llmman.service"])
                .status()
                .is_ok_and(|s| s.success()))
        .then(|| {
            // Microseconds, since a retry starts right after the attempt
            // before it.
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default();
            format!("@{}.{:06}", now.as_secs(), now.subsec_micros())
        });
        // Where `ensure_server` puts it: next to the store.
        let serve_log = llmman::promptlog::beside_store("serve.log")
            .ok()
            .map(|path| {
                let bytes = std::fs::read(&path).unwrap_or_default();
                let head = bytes[..bytes.len().min(256)].to_vec();
                (path, bytes.len() as u64, head)
            });
        Self {
            journal_since,
            serve_log,
        }
    }

    /// What the daemon wrote since [`DaemonLog::mark`].
    fn since(&self) -> String {
        let mut text = String::new();
        if let Some(since) = &self.journal_since {
            if let Ok(out) = Command::new("journalctl")
                .args(["--user", "-u", "llmman.service", "--no-pager", "-o", "cat"])
                .args(["--since", since])
                .output()
            {
                text.push_str(&String::from_utf8_lossy(&out.stdout));
            }
        }
        if let Some((path, start, head)) = &self.serve_log {
            if let Ok(bytes) = std::fs::read(path) {
                // A new daemon rotates serve.log before it starts, so a file
                // that no longer begins as it did at the mark (or is shorter)
                // is a replacement, all of it written during the attempt.
                let start = if (bytes.len() as u64) < *start || !bytes.starts_with(head) {
                    0
                } else {
                    *start as usize
                };
                text.push_str(&String::from_utf8_lossy(&bytes[start..]));
            }
        }
        text
    }
}

/// One line totalling llama-server's per-request `print_timing` lines:
/// how much prompt it prefilled, and how fast. `None` without any.
fn timing_summary(log: &str) -> Option<String> {
    // "... | prompt eval time =   20.78 ms /    47 tokens (...)"
    let parse = |line: &str, key: &str| -> Option<(f64, u64)> {
        let rest = line.split_once(key)?.1.trim_start().strip_prefix('=')?;
        let (ms, rest) = rest.split_once("ms /")?;
        let tokens = rest.split_whitespace().next()?;
        Some((ms.trim().parse().ok()?, tokens.parse().ok()?))
    };
    let (mut requests, mut prompt, mut generated) = (0, (0.0, 0), (0.0, 0));
    for line in log.lines() {
        if let Some((ms, n)) = parse(line, "prompt eval time") {
            requests += 1;
            prompt = (prompt.0 + ms, prompt.1 + n);
        } else if let Some((ms, n)) = parse(line, "eval time") {
            generated = (generated.0 + ms, generated.1 + n);
        }
    }
    if requests == 0 {
        return None;
    }
    let rate = |(ms, n): (f64, u64)| {
        if ms > 0.0 {
            n as f64 / (ms / 1000.0)
        } else {
            0.0
        }
    };
    Some(format!(
        "{requests} request(s); prompt eval {} tokens in {:.1}s ({:.0} tok/s); \
         generated {} tokens in {:.1}s ({:.0} tok/s)",
        prompt.1,
        prompt.0 / 1000.0,
        rate(prompt),
        generated.1,
        generated.0 / 1000.0,
        rate(generated),
    ))
}

/// A bounded excerpt of the daemon's output for a log that must stay
/// readable: the last 40 llama-server timing and slot lines, then the last
/// 50 lines of all, each cut at 300 characters.
fn daemon_log_excerpt(log: &str) -> String {
    const MARKERS: &[&str] = &[
        "print_timing",
        "launch_slot_",
        "release:",
        "cancel task",
        "prompt processing",
        "n_ctx_slot",
    ];
    let lines: Vec<&str> = log.lines().collect();
    let tail_start = lines.len().saturating_sub(50);
    let timing: Vec<&str> = lines[..tail_start]
        .iter()
        .filter(|l| MARKERS.iter().any(|m| l.contains(m)))
        .copied()
        .collect();
    let mut out = String::new();
    for line in timing[timing.len().saturating_sub(40)..]
        .iter()
        .chain(&lines[tail_start..])
    {
        let cut: String = line.chars().take(300).collect();
        out.push_str(&cut);
        out.push('\n');
    }
    out
}

/// The one-line llama-server timing summary for a launch that passed.
fn print_timing_summary(integration: &str, log: &DaemonLog) {
    if let Some(summary) = timing_summary(&log.since()) {
        eprintln!("[test] {integration}: llama-server: {summary}");
    }
}

/// Prints what the daemon wrote during an attempt that did not pass.
fn report_daemon_log(integration: &str, log: &DaemonLog) {
    let text = log.since();
    if text.trim().is_empty() {
        eprintln!("[test] {integration}: the daemon wrote nothing during this attempt");
        return;
    }
    if let Some(summary) = timing_summary(&text) {
        eprintln!("[test] {integration}: llama-server: {summary}");
    }
    eprintln!(
        "[test] {integration}: daemon output during this attempt (excerpt):\n{}",
        daemon_log_excerpt(&text)
    );
}

/// `llmman launch <integration> --model qwen3.5:0.8b <launch_flags> --
/// <extra_args>` with `home` as its `HOME`, and every directory an
/// integration keeps its state in under it.
fn launch_command(
    home: &Path,
    integration: &str,
    launch_flags: &[&str],
    extra_args: &[&str],
) -> Command {
    let mut cmd = Command::new(llmman_bin());
    cmd.arg("launch")
        .arg(integration)
        .arg("--model")
        .arg(MODEL)
        .args(launch_flags);
    if !extra_args.is_empty() {
        cmd.arg("--").args(extra_args);
    }
    cmd.env("HOME", home)
        .env("USERPROFILE", home)
        .env("XDG_CONFIG_HOME", home.join(".config"))
        .env("XDG_DATA_HOME", home.join(".local/share"))
        .env("XDG_STATE_HOME", home.join(".local/state"))
        // Set, not cleared: a `QWEN_HOME` (or `GROK_HOME`, `CLINE_DIR`) in
        // the developer's shell would send the settings a launch writes
        // past this `HOME`, and on Windows `dirs::home_dir` reads neither
        // `HOME` nor `USERPROFILE`.
        .env("QWEN_HOME", home.join(".qwen"))
        .env("GROK_HOME", home.join(".grok"))
        .env("CLINE_DIR", home.join(".cline"))
        .env("COPILOT_HOME", home.join(".copilot"))
        .env("PI_CODING_AGENT_DIR", home.join(".pi").join("agent"))
        // goose asks before each tool call otherwise, and a headless run
        // has nobody to answer. Granted here, not by `launch goose`:
        // auto-approving an agent's writes is the user's call.
        .env("GOOSE_MODE", "auto")
        // AGY's background updater replaces its binary in place (seen
        // within a minute of a first run), so without this a retry would
        // run a different AGY than the one CI installed and checksummed.
        .env("AGY_CLI_DISABLE_AUTO_UPDATE", "true")
        // Cline likewise reinstalls itself as `latest` (CI's pinned 2.18.0
        // became 3.0.x within ~20s), so a retry ran a different Cline.
        .env("CLINE_NO_AUTO_UPDATE", "1");
    cmd
}

/// Runs `llmman launch <integration> --model qwen3.5:0.8b -- <extra_args>`
/// with `home` as its `HOME`, returning `Err(TimedOut)` (rather than
/// panicking) if it had to be killed at `TIMEOUT` — see
/// `try_spawn_with_timeout` and `launch_and_assert` for who retries that
/// and why. The caller warms the model first (`warm_model`).
fn run_launch(
    home: &Path,
    integration: &str,
    extra_args: &[&str],
) -> Result<std::process::Output, TimedOut> {
    try_spawn_with_timeout(
        launch_command(home, integration, &[], extra_args),
        TIMEOUT,
        &format!("`llmman launch {integration} --model {MODEL} -- {extra_args:?}`"),
    )
}

/// Max retries for the "missing pong" shape (see [`launch_and_assert`]).
/// A *timeout* never retries, regardless of this constant — see below.
///
/// Previously bumped to 4 for `opencode` after CI run 32376290299, then
/// reverted once CI run 32633829292 showed the bump didn't help: a
/// degenerate sampling loop isn't fixed by more retries of the same 600s
/// budget, only delayed. That's also why timeouts no longer retry at all
/// (CI run 32726299596: two back-to-back 600s timeouts before a 11s
/// third attempt landed on the same non-failing outcome one immediate
/// timeout would have, 1200s slower).
const MAX_ATTEMPTS: u32 = 3;

/// Runs `llmman launch <integration> --model qwen3.5:0.8b -- <extra_args>`
/// (via [`run_launch`], fresh `HOME` per attempt) and asserts success and
/// that the reply contains "pong". Small-model sampling variance can
/// produce two failure shapes, handled differently:
///
///   - succeeded, but didn't say "pong" — retried up to [`MAX_ATTEMPTS`]:
///     cheap (a completed run, just wrong content) and retrying has a
///     real chance of a different, correct answer.
///   - killed at `TIMEOUT` — an endless agent loop (small-model sampling
///     degenerating under real concurrent batching, observed decoding
///     thousands of tokens at ~14 t/s on a CPU-only runner), or a slow
///     runner: aarch64 Linux prefills the same ~10k-token system prompt
///     at ~44 t/s where x86_64 does ~200 t/s (CI run 35602511987), so a
///     CLI needing several such requests per answer can't fit in
///     `TIMEOUT` there. NOT retried: a fresh `HOME` doesn't fix a slow
///     runner or the model's own sampling, and CI run 32633829292
///     already showed retrying this shape doesn't reliably help — only
///     costs more CI time.
///
/// A real regression (non-zero exit: a crash, a rejected request, a 500)
/// is never retried — it panics immediately via the `assert!` below, on
/// the first attempt, so a deterministic bug can't be masked or slowed
/// down by retries.
///
/// Exhausting attempts via only the two shapes above (never the
/// `assert!`) is logged loudly but does not panic: it's the model's own
/// sampling variance, not an llmman regression, so it must not turn CI
/// red on its own. The `strict` variants fail on the first shape (a CLI
/// that *completed* without saying "pong"); on a timeout they fail only
/// if `daemon_still_answers` finds the daemon stalled. CI run
/// 35602511987 went red on all six E2E targets through the two strict
/// tests timing out on these very shapes (cline on five, grok on one).
enum NonzeroDisposition {
    Reject,
    Retry,
    Accept,
}

fn launch_and_assert(integration: &str, extra_args: &[&str]) {
    launch_and_assert_with(
        integration,
        extra_args,
        |_stdout, _stderr| NonzeroDisposition::Reject,
        |_stdout| false,
        reply_contains_pong,
        false,
        |_home| {},
    );
}

/// [`launch_and_assert`], but a *completed* run that never said `pong`
/// is a test failure, and a timeout is one unless the daemon still
/// answers afterwards. Used where the integration is expected to be
/// deterministic enough that a zero exit without inference is not success.
fn launch_and_assert_strict(integration: &str, extra_args: &[&str]) {
    launch_and_assert_with(
        integration,
        extra_args,
        |_stdout, _stderr| NonzeroDisposition::Reject,
        |_stdout| false,
        reply_contains_pong,
        true,
        |_home| {},
    );
}

/// Strict inference plus assertions over the fresh HOME after the real CLI
/// exits. This catches launchers that infer correctly but leak credentials or
/// mutate user-owned state while doing it.
fn launch_and_assert_strict_inspecting(
    integration: &str,
    extra_args: &[&str],
    accept_nonzero: impl Fn(&str, &str) -> bool,
    accept_stdout: impl Fn(&str) -> bool,
    inspect_home: impl Fn(&Path),
) {
    launch_and_assert_with(
        integration,
        extra_args,
        |stdout, stderr| {
            if accept_nonzero(stdout, stderr) {
                NonzeroDisposition::Accept
            } else {
                NonzeroDisposition::Reject
            }
        },
        |_stdout| false,
        accept_stdout,
        true,
        inspect_home,
    );
}

/// [`launch_and_assert`], for a CLI that reports failure without a
/// non-zero exit: stdout matching `reject_stdout` panics instead of being
/// retried as sampling variance — see `goose_request_failed`.
fn launch_and_assert_rejecting(
    integration: &str,
    extra_args: &[&str],
    reject_stdout: impl Fn(&str) -> bool,
) {
    launch_and_assert_with(
        integration,
        extra_args,
        |_stdout, _stderr| NonzeroDisposition::Reject,
        reject_stdout,
        reply_contains_pong,
        false,
        |_home| {},
    );
}

/// [`launch_and_assert`], generalized with an extra tolerated failure
/// shape: a nonzero exit whose stderr matches `tolerate_stderr` is
/// treated the same as a timeout or a missing "pong" (logged, retried,
/// never panics) instead of the deterministic-bug path. Only meant for
/// an integration with its own independently-verified, non-llmman-caused
/// failure mode — see `openclaw_pull_registry_flake`'s own doc comment
/// for the one real case this exists for. `launch_and_assert` itself is
/// just this with `|_| false`: no additional shape tolerated, unchanged
/// behavior for every other integration.
fn launch_and_assert_tolerating(
    integration: &str,
    extra_args: &[&str],
    tolerate_stderr: impl Fn(&str) -> bool,
) {
    launch_and_assert_with(
        integration,
        extra_args,
        retry_on_stderr(tolerate_stderr),
        |_stdout| false,
        reply_contains_pong,
        false,
        |_home| {},
    );
}

/// A nonzero-exit disposition that retries when `tolerate_stderr` matches
/// and rejects anything else.
fn retry_on_stderr(
    tolerate_stderr: impl Fn(&str) -> bool,
) -> impl Fn(&str, &str) -> NonzeroDisposition {
    move |_stdout, stderr| {
        if tolerate_stderr(stderr) {
            NonzeroDisposition::Retry
        } else {
            NonzeroDisposition::Reject
        }
    }
}

fn reply_contains_pong(stdout: &str) -> bool {
    stdout.to_lowercase().contains("pong")
}

/// The shared body: `nonzero_disposition` rejects, retries, or conditionally
/// accepts a nonzero exit, `reject_stdout` narrows what a zero exit may be,
/// `accept_stdout` defines a successful model reply, and `strict` makes
/// exhausting the sampling attempts a test failure — for a timeout, only
/// when the daemon has stopped answering (see [`launch_and_assert`]).
fn launch_and_assert_with(
    integration: &str,
    extra_args: &[&str],
    nonzero_disposition: impl Fn(&str, &str) -> NonzeroDisposition,
    reject_stdout: impl Fn(&str) -> bool,
    accept_stdout: impl Fn(&str) -> bool,
    strict: bool,
    inspect_home: impl Fn(&Path),
) {
    let mut last_failure = None;
    // Set when the loop gives up on a timeout (not retried) rather than
    // exhausting MAX_ATTEMPTS via "missing pong"/a tolerated nonzero
    // exit — picks the right WARNING message below.
    let mut gave_up_after_timeout = None;
    for attempt in 1..=MAX_ATTEMPTS {
        let home = fresh_home(integration);
        // Warmed first, so this attempt's timings are the launch's own.
        warm_model();
        let daemon_log = DaemonLog::mark();
        let output = match run_launch(&home, integration, extra_args) {
            Ok(output) => output,
            Err(timed_out) => {
                eprintln!(
                    "[test] {integration}: attempt {attempt}/{MAX_ATTEMPTS} timed out \
                     (see launch_and_assert's doc comment); not retried, giving up\n{}",
                    timed_out.message
                );
                report_daemon_log(integration, &daemon_log);
                last_failure = Some(timed_out.message);
                gave_up_after_timeout = Some(attempt);
                break;
            }
        };
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !output.status.success() {
            match nonzero_disposition(&stdout, &stderr) {
                NonzeroDisposition::Accept if accept_stdout(&stdout) => {
                    print_timing_summary(integration, &daemon_log);
                    inspect_home(&home);
                    return;
                }
                NonzeroDisposition::Accept | NonzeroDisposition::Retry => {
                    report_daemon_log(integration, &daemon_log)
                }
                NonzeroDisposition::Reject => {
                    report_daemon_log(integration, &daemon_log);
                    panic!(
                        "`llmman launch {integration} --model {MODEL} -- {extra_args:?}` failed \
                         (status: {:?})\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
                        output.status
                    )
                }
            }
            eprintln!(
                "[test] {integration}: attempt {attempt}/{MAX_ATTEMPTS} failed via its own \
                 known non-llmman-caused failure shape; {}",
                if attempt < MAX_ATTEMPTS {
                    "retrying with a fresh HOME"
                } else {
                    "giving up"
                }
            );
            last_failure = Some(format!(
                "known non-llmman-caused failure (status: {:?})\n\
                 --- stdout (last attempt) ---\n{stdout}\n--- stderr (last attempt) ---\n{stderr}",
                output.status
            ));
            continue;
        }
        // A zero exit isn't proof the launch worked for every CLI: a
        // self-reported failure is a bug, not sampling variance.
        if reject_stdout(&stdout) {
            report_daemon_log(integration, &daemon_log);
            panic!(
                "`llmman launch {integration} --model {MODEL} -- {extra_args:?}` exited 0 but \
                 reported a failure of its own\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
            );
        }
        if accept_stdout(&stdout) {
            print_timing_summary(integration, &daemon_log);
            inspect_home(&home);
            return;
        }
        report_daemon_log(integration, &daemon_log);
        eprintln!(
            "[test] {integration}: attempt {attempt}/{MAX_ATTEMPTS} succeeded but the reply \
             didn't satisfy the output assertion; {}",
            if attempt < MAX_ATTEMPTS {
                "retrying with a fresh HOME"
            } else {
                "giving up"
            }
        );
        last_failure = Some(format!(
            "expected {integration}'s reply to satisfy its output assertion\n\
             --- stdout (last attempt) ---\n{stdout}\n--- stderr (last attempt) ---\n{stderr}"
        ));
    }
    let last_failure = last_failure.expect("loop runs at least once, so this is always set");
    let timed_out = gave_up_after_timeout.is_some();
    let why = if timed_out {
        "a timeout"
    } else {
        "an unexpected model reply (or a known non-llmman-caused failure)"
    };
    // A strict timeout is tolerated only once the daemon is shown to be
    // fine: then the CLI, not llmman, was the slow party.
    let tolerated = !strict || (timed_out && daemon_still_answers(integration));
    assert!(
        tolerated,
        "`llmman launch {integration} --model {MODEL} -- {extra_args:?}` gave up via {why}\n\
         {last_failure}"
    );
    eprintln!(
        "[test] {integration}: WARNING — gave up via {why} only (sampling variance, not an \
         llmman regression — see launch_and_assert's doc comment); does not fail this test. \
         Last failure, for investigation:\n{last_failure}"
    );
}

#[test]
fn launch_claude_with_model() {
    eprintln!("[test] launch_claude_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_claude_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("claude") {
        eprintln!("skipping: claude not on PATH — https://code.claude.com/docs/en/quickstart");
        return;
    }

    // `-p`/`--print`: Claude Code's non-interactive one-shot mode — the
    // scriptable equivalent of typing a message into the interactive TUI
    // `llmman launch claude --model qwen3.5:0.8b` would otherwise open.
    launch_and_assert("claude", &["-p", PROMPT]);
}

#[test]
fn launch_agy_with_model() {
    eprintln!("[test] launch_agy_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_agy_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("agy") {
        eprintln!("skipping: agy not on PATH — https://antigravity.google/docs/cli/install/");
        return;
    }

    // Real AGY print mode. Its built-in title and main-agent requests use
    // different Google model names; the llmman route must pin both to MODEL.
    //
    // AGY's own `--print-timeout` (default 5m) is kept past TIMEOUT so the
    // harness decides when a run is hung: AGY firing first is a nonzero
    // exit `launch_and_assert` treats as a bug, though the cause (a CPU
    // runner needing 83s to 300s+ just to prefill AGY's ~12k-token system
    // prompt, per CI runs 34698882471..34703049223) is the timeout shape
    // it already tolerates.
    let print_timeout = format!("{}s", TIMEOUT.as_secs() * 2);
    launch_and_assert(
        "agy",
        &[
            "-p",
            PROMPT,
            "--output-format",
            "text",
            "--print-timeout",
            &print_timeout,
            "--disable-slash-commands",
        ],
    );
}

#[test]
fn launch_opencode_with_model() {
    eprintln!("[test] launch_opencode_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_opencode_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("opencode") {
        eprintln!("skipping: opencode not on PATH — https://opencode.ai");
        return;
    }

    // `run <message>`: opencode's non-interactive one-shot mode.
    // --print-logs --log-level DEBUG: opencode's provider (configured via
    // OPENCODE_CONFIG_CONTENT's "npm" field — see launch::opencode::opencode_config)
    // is installed on demand into ~/.config/opencode/node_modules the
    // first time a fresh HOME uses it, which showed up as a slow/hanging
    // step in one environment during development; keep this on so a CI
    // failure's logs show exactly what opencode was doing right up to a
    // timeout, instead of just the banner and silence.
    launch_and_assert(
        "opencode",
        &["run", PROMPT, "--print-logs", "--log-level", "DEBUG"],
    );
}

#[test]
fn launch_codex_with_model() {
    eprintln!("[test] launch_codex_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_codex_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("codex") {
        eprintln!("skipping: codex not on PATH — npm install -g @openai/codex");
        return;
    }

    // `exec <prompt>`: codex's non-interactive one-shot mode.
    launch_and_assert("codex", &["exec", PROMPT]);
}

#[test]
fn launch_copilot_with_model() {
    eprintln!("[test] launch_copilot_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_copilot_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("copilot") {
        eprintln!("skipping: copilot not on PATH — npm install -g @github/copilot");
        return;
    }

    // `-p` is Copilot CLI's non-interactive prompt mode. run_launch's
    // fresh HOME proves BYOK mode does not depend on a prior GitHub
    // login or Copilot configuration. `--silent` leaves only the model's
    // reply on stdout, and the strict assertion prevents Copilot's unusual
    // zero-exit "Failed to get response" path from passing this test.
    launch_and_assert_strict("copilot", &["--silent", "-p", PROMPT]);
}

#[test]
fn launch_droid_with_model() {
    eprintln!("[test] launch_droid_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_droid_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("droid") {
        eprintln!(
            "skipping: droid not on PATH — https://docs.factory.ai/cli/getting-started/quickstart"
        );
        return;
    }

    // `exec` is Droid's non-interactive mode. The fresh HOME verifies that
    // launch creates a usable custom-model entry without prior setup/login.
    launch_and_assert_strict_inspecting(
        "droid",
        &["exec", "--output-format", "text", PROMPT],
        |_stdout, _stderr| false,
        reply_contains_pong,
        |home| {
            let path = home.join(".factory/settings.json");
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
            let settings: serde_json::Value = serde_json::from_str(&text)
                .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()));
            let model = model_reference();
            let entry = settings["customModels"]
                .as_array()
                .and_then(|models| models.iter().find(|entry| entry["model"] == model))
                .unwrap_or_else(|| panic!("Droid model missing from {}: {text}", path.display()));
            assert_eq!(entry["baseUrl"], format!("{}/v1", llmman::daemon::server()));
            assert_eq!(entry["apiKey"], "${LLMMAN_API_KEY}");
            assert_eq!(entry["provider"], "generic-chat-completion-api");
            assert_eq!(entry["id"], format!("custom:{model}-[llmman]-0"));
            assert_eq!(entry["index"], 0);
            assert_eq!(settings["sessionDefaultSettings"]["model"], entry["id"]);
            assert_eq!(
                settings["sessionDefaultSettings"]["reasoningEffort"],
                "none"
            );
        },
    );
}

/// `launch codex --sandbox docker|podman`: codex from Docker Sandboxes'
/// image, reading the profile the launcher wrote on the host. That needs
/// both halves of the container plumbing: the state directory mounted at
/// its host path, and the daemon reachable from inside. codex need not be
/// installed here. Opt-in, since it pulls a multi-GB image:
///
///   LLMMAN_E2E_SANDBOX=docker cargo test --release --test launch_e2e \
///     launch_codex_in_a_container_sandbox -- --nocapture
#[test]
fn launch_codex_in_a_container_sandbox() {
    let _guard = lock_serial();
    let Some(engine) = std::env::var("LLMMAN_E2E_SANDBOX")
        .ok()
        .filter(|e| e == "docker" || e == "podman")
    else {
        eprintln!("skipping: set LLMMAN_E2E_SANDBOX=docker or podman (pulls a multi-GB image)");
        return;
    };
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path(&engine) {
        eprintln!("skipping: {engine} not on PATH");
        return;
    }
    warm_model();

    let home = fresh_home("codex-sandbox");
    let mut cmd = Command::new(llmman_bin());
    cmd.args(["launch", "codex", "--model", MODEL, "--sandbox", &engine])
        .args(["--", "exec", "--skip-git-repo-check", PROMPT])
        .env("HOME", &home)
        .env("XDG_CONFIG_HOME", home.join(".config"))
        .env("XDG_DATA_HOME", home.join(".local/share"));
    let output = spawn_with_timeout(cmd, TIMEOUT, "`llmman launch codex --sandbox`");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "`llmman launch codex --sandbox {engine}` failed (status: {:?})\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
        output.status
    );
    // The sandbox's home is llmman's own, under this HOME, not HOME itself.
    assert!(
        home.join(".local/share/llmman/sandbox/codex").is_dir(),
        "no sandbox home under {}",
        home.display()
    );
    if !reply_contains_pong(&stdout) {
        eprintln!(
            "[test] codex --sandbox: WARNING — exited 0 without saying pong (sampling \
             variance, see launch_and_assert)\n--- stdout ---\n{stdout}"
        );
    }
}

// ---------------------------------------------------------------------------
// `--variant`
// ---------------------------------------------------------------------------

/// `MODEL` offers `none` and `thinking`; claude and codex take `thinking`
/// as `medium`.
const VARIANT: &str = "thinking";
const SPELLED: &str = "medium";
/// An effort the tool saved itself, for the launch to outrank. Its
/// background requests (claude's session title) send it regardless, so
/// only that the variant's effort is sent can be checked.
const SAVED: &str = "high";

/// A TCP relay in front of the daemon, keeping what clients send: the
/// tools don't print their effort, so their requests are where to look.
struct Relay {
    addr: String,
    sent: Arc<Mutex<Vec<u8>>>,
}

fn relay_to_daemon() -> Relay {
    let upstream = llmman::daemon::server()
        .trim_start_matches("http://")
        .to_string();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind the relay");
    let addr = listener.local_addr().expect("relay address").to_string();
    let sent = Arc::new(Mutex::new(Vec::new()));
    let record = Arc::clone(&sent);
    std::thread::spawn(move || {
        for client in listener.incoming().flatten() {
            let Ok(server) = std::net::TcpStream::connect(&upstream) else {
                continue;
            };
            let (Ok(from_client), Ok(to_server)) = (client.try_clone(), server.try_clone()) else {
                continue;
            };
            let record = Arc::clone(&record);
            std::thread::spawn(move || relay_bytes(from_client, to_server, Some(record)));
            std::thread::spawn(move || relay_bytes(server, client, None));
        }
    });
    Relay { addr, sent }
}

/// Copies `from` to `to` until either ends, keeping a copy in `record`.
fn relay_bytes(
    mut from: std::net::TcpStream,
    mut to: std::net::TcpStream,
    record: Option<Arc<Mutex<Vec<u8>>>>,
) {
    let mut buf = [0u8; 16 * 1024];
    while let Ok(n) = from.read(&mut buf) {
        if n == 0 {
            break;
        }
        if let Some(record) = &record {
            record.lock().unwrap().extend_from_slice(&buf[..n]);
        }
        if to.write_all(&buf[..n]).is_err() {
            break;
        }
    }
    let _ = to.shutdown(std::net::Shutdown::Write);
}

/// What a real integration sent the daemon when launched at [`VARIANT`].
struct VariantLaunch {
    home: PathBuf,
    sent: String,
    output: String,
}

impl VariantLaunch {
    /// The values of the string field `key` in what was sent.
    fn sent_values(&self, key: &str) -> Vec<&str> {
        let needle = format!("\"{key}\":\"");
        self.sent
            .match_indices(&needle)
            .map(|(at, _)| {
                let rest = &self.sent[at + needle.len()..];
                &rest[..rest.find('"').unwrap_or(rest.len())]
            })
            .collect()
    }
}

/// Launches `integration` at [`VARIANT`] behind a [`Relay`], after `saved`
/// has set up its fresh `HOME`. The request is what is checked, not the
/// reply, and a run killed at `TIMEOUT` has sent it already.
fn launch_at_variant(
    integration: &str,
    extra_args: &[&str],
    saved: impl FnOnce(&Path),
) -> VariantLaunch {
    warm_model();
    let home = fresh_home(integration);
    saved(&home);
    let relay = relay_to_daemon();
    let mut cmd = launch_command(&home, integration, &["--variant", VARIANT], extra_args);
    cmd.env("LLMMAN_HOST", &relay.addr);
    let description =
        format!("`llmman launch {integration} --variant {VARIANT} -- {extra_args:?}`");
    let output = match try_spawn_with_timeout(cmd, TIMEOUT, &description) {
        Ok(output) => format!(
            "--- stdout ---\n{}\n--- stderr ---\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
        Err(timed_out) => timed_out.message,
    };
    let sent = String::from_utf8_lossy(&relay.sent.lock().unwrap()).into_owned();
    VariantLaunch { home, sent, output }
}

fn write_saved(home: &Path, file: &str, contents: &str) {
    let path = home.join(file);
    std::fs::create_dir_all(path.parent().unwrap()).expect("create the saved file's directory");
    std::fs::write(path, contents).expect("write the saved file");
}

/// False, with a note, unless llama-server and `tool` are installed.
fn installed(tool: &str, hint: &str) -> bool {
    let found = on_path("llama-server") && on_path(tool);
    if !found {
        eprintln!("skipping: needs llama-server and {tool} on PATH ({hint})");
    }
    found
}

/// claude and codex send the launch's effort, not the one in their own
/// settings (`saved_file`).
fn variant_outranks_saved_effort(
    tool: &str,
    hint: &str,
    args: &[&str],
    saved_file: &str,
    saved: &str,
) {
    let _guard = lock_serial();
    if !installed(tool, hint) {
        return;
    }
    let launch = launch_at_variant(tool, args, |home| write_saved(home, saved_file, saved));
    let sent = launch.sent_values("effort");
    assert!(
        sent.contains(&SPELLED),
        "{tool} should send effort {SPELLED:?}, not just its saved {SAVED:?}; it sent {sent:?}\n{}",
        launch.output
    );
}

#[test]
fn launch_claude_variant_outranks_the_saved_effort() {
    variant_outranks_saved_effort(
        "claude",
        "https://code.claude.com/docs/en/quickstart",
        &["-p", PROMPT],
        ".claude/settings.json",
        &format!(r#"{{"effortLevel":"{SAVED}"}}"#),
    );
}

#[test]
fn launch_codex_variant_outranks_the_saved_effort() {
    variant_outranks_saved_effort(
        "codex",
        "npm install -g @openai/codex",
        &["exec", PROMPT],
        ".codex/config.toml",
        &format!("model_reasoning_effort = \"{SAVED}\"\n"),
    );
}

/// `opencode run` sends the model's options; its UI sends the selection
/// in `model.json`, which the launch has to replace.
#[test]
fn launch_opencode_variant_outranks_the_saved_selection() {
    let _guard = lock_serial();
    if !installed("opencode", "https://opencode.ai") {
        return;
    }
    let model = format!("ollama/{}", model_reference());
    let state = ".local/state/opencode/model.json";
    let launch = launch_at_variant("opencode", &["run", PROMPT], |home| {
        let saved = serde_json::json!({ "variant": { &model: "none" } });
        write_saved(home, state, &saved.to_string());
    });

    let sent = launch.sent_values("reasoning_effort");
    assert!(
        sent.contains(&SPELLED) && launch.sent.contains(r#""enable_thinking":true"#),
        "opencode should send effort {SPELLED:?} with thinking on; it sent {sent:?}\n{}",
        launch.output
    );
    let saved: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(launch.home.join(state)).unwrap()).unwrap();
    assert_eq!(
        saved["variant"][&model], VARIANT,
        "opencode's saved selection: {saved}\n{}",
        launch.output
    );
}

#[test]
fn launch_pi_with_model() {
    eprintln!("[test] launch_pi_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_pi_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("pi") {
        eprintln!("skipping: pi not on PATH — npm install -g @earendil-works/pi-coding-agent");
        return;
    }

    // `-p <prompt>`: pi's own print-and-exit mode. `run_launch` gives it a
    // fresh HOME and PI_CODING_AGENT_DIR, so this exercises writing
    // models.json and settings.json from nothing as well as the request.
    launch_and_assert("pi", &["-p", PROMPT]);
}

#[test]
fn launch_omp_with_model() {
    eprintln!("[test] launch_omp_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_omp_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("omp") {
        eprintln!("skipping: omp not on PATH — https://omp.sh/");
        return;
    }

    // `-p <prompt>` is OMP's print-and-exit mode. The launcher selects
    // `ollama/<model>` while run_launch's fresh HOME ensures the test does
    // not succeed because of a developer's pre-existing OMP configuration.
    // The 0.8B model can say any word ("Rust" x3 in run 36833820219): any reply counts.
    launch_and_assert_strict_inspecting(
        "omp",
        &["-p", PROMPT],
        |_stdout, _stderr| false,
        |stdout| !stdout.trim().is_empty(),
        |_home| {},
    );
}

#[test]
fn launch_cline_with_model() {
    eprintln!("[test] launch_cline_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_cline_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("cline") {
        eprintln!("skipping: cline not on PATH — npm install -g cline");
        return;
    }

    // `--json` selects NDJSON output and `--yolo` prevents interactive tool
    // approval. Parse only the final assistant text so an echoed prompt or a
    // question containing "pong" cannot satisfy the assertion. qwen3.5:0.8b
    // may answer correctly but never make a well-formed tool call, which
    // Cline requires to finish; accept that one known nonzero shape only
    // after this exact reply check.
    launch_and_assert_strict_inspecting(
        "cline",
        &["--json", "--yolo", PROMPT],
        cline_nonzero_is_a_malformed_tool_call,
        cline_json_reply_is_exact_pong,
        |home| {
            let path = home.join(".cline/data/settings/providers.json");
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
            let settings: serde_json::Value = serde_json::from_str(&text)
                .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()));
            let provider = &settings["providers"]["ollama"]["settings"];
            assert_eq!(settings["lastUsedProvider"], "ollama");
            assert_eq!(provider["provider"], "ollama");
            assert_eq!(provider["model"], "docker.io/ai/qwen3.5:0.8b");
            assert_eq!(
                provider["baseUrl"],
                format!("{}/v1", llmman::daemon::server())
            );
            // See CLINE_RESPONSE_START_TIMEOUT_MS in launch/cline.rs.
            assert!(
                provider["timeout"].as_u64().is_some_and(|ms| ms > 300_000),
                "expected `llmman launch cline` to raise Cline's response-start timeout: {text}"
            );
            assert!(
                provider.get("apiKey").is_none(),
                "Cline persisted the launch credential in {}: {text}",
                path.display()
            );

            let global_path = home.join(".cline/data/globalState.json");
            let global_text = std::fs::read_to_string(&global_path)
                .unwrap_or_else(|error| panic!("read {}: {error}", global_path.display()));
            let global: serde_json::Value = serde_json::from_str(&global_text)
                .unwrap_or_else(|error| panic!("parse {}: {error}", global_path.display()));
            assert_eq!(global["actModeApiProvider"], "ollama");
            assert_eq!(global["planModeApiProvider"], "ollama");
            assert_eq!(global["actModeOllamaModelId"], "docker.io/ai/qwen3.5:0.8b");
            assert_eq!(global["planModeOllamaModelId"], "docker.io/ai/qwen3.5:0.8b");
            assert_eq!(global["welcomeViewCompleted"], true);
            // Cline 2.18 reads its timeout from here, not providers.json.
            assert!(
                global["requestTimeoutMs"]
                    .as_u64()
                    .is_some_and(|ms| ms > 300_000),
                "expected `llmman launch cline` to raise Cline's requestTimeoutMs: {global_text}"
            );
        },
    );
}

fn cline_json_reply_is_exact_pong(stdout: &str) -> bool {
    stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|event| event["type"] == "say" && event["say"] == "text")
        .filter_map(|event| event["text"].as_str().map(str::to_owned))
        .next_back()
        .is_some_and(|text| text.trim() == "pong")
}

/// Cline ends a `--yolo` run with exit 1 when the model's tool call is
/// malformed: it gives up after "Too many consecutive mistakes", or reports
/// "Cline tried to use <tool> without value for required parameter '<name>'".
/// Cline takes its tool calls from the model's text, so this is the model's
/// sampling and not llmman. qwen3.5:0.8b has hit it on `attempt_completion`
/// and `ask_followup_question`, so match the shape and not the tool name.
fn cline_nonzero_is_a_malformed_tool_call(stdout: &str, stderr: &str) -> bool {
    stderr.trim().is_empty()
        && stdout.lines().any(|line| {
            serde_json::from_str::<serde_json::Value>(line)
                .ok()
                .is_some_and(|event| {
                    event["type"] == "error"
                        && event["message"].as_str().is_some_and(|message| {
                            message.contains("Too many consecutive mistakes")
                                || (message.starts_with("Cline tried to use ")
                                    && message.contains(" without value for required parameter "))
                        })
                })
        })
}

/// Real llama-server b11429 lines: two finished requests are totalled, the
/// generation progress line is not a request, and the excerpt stays bounded.
#[test]
fn daemon_timings_are_summarized_and_the_excerpt_is_bounded() {
    let log = "\
0.00.482.322 I slot launch_slot_: id  3 | task 0 | processing task, is_child = 0
0.00.616.768 I slot print_timing: id  3 | task 0 | prompt eval time =   20000.00 ms /  1000 tokens (   20.00 ms per token,    50.00 tokens per second)
0.00.616.771 I slot print_timing: id  3 | task 0 |        eval time =     113.61 ms /    33 tokens (    3.55 ms per token,   281.66 tokens per second)
0.00.616.772 I slot print_timing: id  3 | task 0 |       total time =   20113.61 ms /  1033 tokens
1.02.457.226 I slot print_timing: id  3 | task 39 | n_gen =    894, tg = 297.44 t/s, tg_3s = 297.78 t/s
1.03.671.356 W srv          stop: cancel task, id_task = 39
1.04.616.768 I slot print_timing: id  3 | task 40 | prompt eval time =   10000.00 ms /   500 tokens (   20.00 ms per token,    50.00 tokens per second)
1.04.616.771 I slot print_timing: id  3 | task 40 |        eval time =     100.00 ms /    10 tokens (   10.00 ms per token,   100.00 tokens per second)
";
    assert_eq!(
        timing_summary(log).as_deref(),
        Some(
            "2 request(s); prompt eval 1500 tokens in 30.0s (50 tok/s); \
             generated 43 tokens in 0.2s (201 tok/s)"
        )
    );
    assert_eq!(timing_summary("no timings here"), None);

    let noisy: String = (0..1000)
        .map(|i| format!("{i} {}\n", "x".repeat(400)))
        .chain(std::iter::once(log.to_string()))
        .collect();
    let excerpt = daemon_log_excerpt(&noisy);
    assert!(excerpt.lines().count() <= 90, "{}", excerpt.lines().count());
    assert!(excerpt.lines().all(|l| l.chars().count() <= 300));
    assert!(excerpt.contains("cancel task"), "{excerpt}");
}

#[test]
fn cline_json_reply_requires_the_final_result_to_be_exactly_pong() {
    assert!(cline_json_reply_is_exact_pong(
        "{\"type\":\"say\",\"say\":\"task\",\"text\":\"Reply with exactly the single word: pong\"}\n\
         {\"type\":\"say\",\"say\":\"text\",\"text\":\"  pong\\n\"}\n"
    ));
    assert!(!cline_json_reply_is_exact_pong(
        "{\"type\":\"say\",\"say\":\"task\",\"text\":\"Reply with exactly the single word: pong\"}\n\
         {\"type\":\"ask\",\"ask\":\"followup\",\"text\":\"Which teammate should I play in the pong game?\"}\n"
    ));
    assert!(!cline_json_reply_is_exact_pong(
        "{\"type\":\"say\",\"say\":\"text\",\"text\":\"pong\"}\n\
         {\"type\":\"say\",\"say\":\"text\",\"text\":\"not pong\"}\n"
    ));
    assert!(cline_nonzero_is_a_malformed_tool_call(
        "{\"type\":\"error\",\"message\":\"[YOLO MODE] Task failed: Too many consecutive mistakes (3).\"}\n",
        ""
    ));
    // The same failure on whichever tool the model botched (CI run
    // 36698710059 was `attempt_completion`, 36712104327 this one).
    assert!(cline_nonzero_is_a_malformed_tool_call(
        "{\"type\":\"error\",\"message\":\"Cline tried to use attempt_completion without value for required parameter 'result'. Retrying...\"}\n",
        ""
    ));
    assert!(cline_nonzero_is_a_malformed_tool_call(
        "{\"type\":\"error\",\"message\":\"Cline tried to use ask_followup_question without value for required parameter 'question'. Retrying...\"}\n",
        ""
    ));
    // CI run 36712104327, condensed: the model said `pong` as plain text
    // each turn, never made a valid tool call, and Cline exited 1.
    let botched = "{\"type\":\"say\",\"say\":\"task\",\"text\":\"Reply with exactly the single word: pong\"}\n\
         {\"type\":\"say\",\"say\":\"text\",\"text\":\"pong\",\"partial\":false}\n\
         {\"type\":\"say\",\"say\":\"text\",\"text\":\"pong\",\"partial\":false}\n\
         {\"type\":\"say\",\"say\":\"error\",\"text\":\"Cline tried to use ask_followup_question without value for required parameter 'question'. Retrying...\"}\n\
         {\"type\":\"error\",\"message\":\"Cline tried to use ask_followup_question without value for required parameter 'question'. Retrying...\"}\n";
    assert!(cline_nonzero_is_a_malformed_tool_call(botched, ""));
    assert!(cline_json_reply_is_exact_pong(botched));
    // Anything else, or anything on stderr, is not the model's tool call.
    assert!(!cline_nonzero_is_a_malformed_tool_call(
        "{\"type\":\"error\",\"message\":\"connection refused\"}\n",
        ""
    ));
    assert!(!cline_nonzero_is_a_malformed_tool_call(
        "{\"type\":\"error\",\"message\":\"Cline tried to use ask_followup_question without value for required parameter 'question'. Retrying...\"}\n",
        "Error: cline is not installed\n"
    ));
    assert!(!cline_nonzero_is_a_malformed_tool_call(
        "{\"type\":\"say\",\"say\":\"text\",\"text\":\"Cline tried to use x without value for required parameter 'y'\"}\n",
        ""
    ));
}

#[test]
fn launch_grok_with_model() {
    eprintln!("[test] launch_grok_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_grok_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("grok") {
        eprintln!("skipping: grok not on PATH — https://x.ai/cli");
        return;
    }

    // `-p <prompt>` is Grok Build's headless, single-turn mode. This
    // exercises its real model-catalog fetch and Chat Completions request
    // against llmman's `/v1` endpoint, using the model table llmman writes
    // into Grok's config.
    launch_and_assert_strict("grok", &["-p", PROMPT]);
}

#[test]
fn launch_docker_agent_with_model() {
    eprintln!("[test] launch_docker_agent_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_docker_agent_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    // `on_path` alone, though `find_docker_agent` also looks in
    // `~/.docker/cli-plugins`: `run_launch` replaces `HOME` with a fresh
    // temp directory, so that fallback can never fire under this
    // harness. CI installs the binary onto `PATH` for that reason.
    if !on_path("docker-agent") {
        eprintln!(
            "skipping: docker-agent not on PATH — https://github.com/docker/docker-agent/releases"
        );
        return;
    }

    // These arguments land after the `run <agent file>` that
    // `docker_agent_args` prepends, which is why neither appears here.
    // `--exec` is docker-agent's non-interactive mode. No `--yolo`,
    // though the generated agent carries the shell and filesystem
    // toolsets: a one-word reply calls neither, and granting an agent's
    // writes stays the caller's decision.
    //
    // Strict: `inspect_home` runs only after the reply assertion
    // passes, so a tolerated miss would skip the config checks below
    // and leave the test asserting nothing. A timeout is still
    // forgiven when the daemon is shown to be alive.
    //
    // Only docker-agent's own loop guard is retried; any other nonzero
    // exit still fails at once.
    launch_and_assert_with(
        "docker-agent",
        &["--exec", PROMPT],
        retry_on_stderr(docker_agent_stopped_a_degenerate_loop),
        |_stdout| false,
        docker_agent_reply_is_pong,
        true,
        docker_agent_left_the_users_own_config_alone,
    );
}

/// Whether docker-agent's own loop guard ended the run (`Error: Agent
/// terminated: detected 5 consecutive identical calls to write_file. This
/// indicates a degenerate loop where the model is not making progress.`):
/// model sampling variance, not an llmman failure.
fn docker_agent_stopped_a_degenerate_loop(stderr: &str) -> bool {
    stderr.contains("degenerate loop where the model is not making progress")
}

#[test]
fn docker_agent_retries_only_its_own_degenerate_loop_guard() {
    assert!(docker_agent_stopped_a_degenerate_loop(
        "Error: Agent terminated: detected 5 consecutive identical calls to write_file. \
         This indicates a degenerate loop where the model is not making progress.\n"
    ));
    assert!(!docker_agent_stopped_a_degenerate_loop(
        "Error: failed to create runtime: connection refused\n"
    ));
    assert!(!docker_agent_stopped_a_degenerate_loop(""));
}

/// Whether the model's *reply* was "pong", ignoring the reasoning
/// docker-agent prints above it.
///
/// A thinking model reasons about the prompt, so the word appears in its
/// reasoning whether or not it ever answers — `stdout.contains("pong")`
/// passes on a run that produced no reply. The reply is the last
/// non-empty line.
fn docker_agent_reply_is_pong(stdout: &str) -> bool {
    stdout
        .lines()
        .map(str::trim)
        .rev()
        .find(|line| !line.is_empty())
        .is_some_and(|reply| {
            reply
                .trim_matches(|c: char| !c.is_alphanumeric())
                .eq_ignore_ascii_case("pong")
        })
}

/// Checks what `launch_docker_agent` promises: it writes its own agent
/// file under `~/.config/llmman`, and puts nothing in docker-agent's own
/// `~/.config/cagent`. The second is asserted as "no llmman settings in
/// that file" rather than "that directory is absent", because
/// docker-agent creates it itself on first run.
///
/// Runs on every platform: `docker_agent_config_dir` hangs off
/// `llmman.conf`'s directory, which resolves `~` through
/// `HOME`/`USERPROFILE`, so Windows writes into this temp home too.
fn docker_agent_left_the_users_own_config_alone(home: &Path) {
    let generated = docker_agent_generated_file(home);
    let text = std::fs::read_to_string(&generated)
        .unwrap_or_else(|error| panic!("read {}: {error}", generated.display()));
    assert!(
        text.contains(&format!("base_url: \"{}/v1\"", llmman::daemon::server())),
        "generated agent does not route at the daemon: {text}"
    );
    assert!(
        // MODEL as `launch` resolves it before the launcher sees it,
        // spelled out the way `launch_cline_with_model` does.
        text.contains("model: \"docker.io/ai/qwen3.5:0.8b\""),
        "generated agent does not name the resolved model: {text}"
    );
    // The key is named, not written, so a `--provider` launch cannot
    // leave a real credential in this file.
    assert!(
        text.contains("token_key: LLMMAN_API_KEY"),
        "generated agent does not read its key from the environment: {text}"
    );
    assert!(
        !text.contains("api_key"),
        "generated agent persisted a credential: {text}"
    );

    let theirs = home.join(".config/cagent/config.yaml");
    if let Ok(text) = std::fs::read_to_string(&theirs) {
        assert!(
            !text.contains("llmman"),
            "llmman wrote into docker-agent's own config at {}: {text}",
            theirs.display()
        );
    }
}

/// The agent file the launch wrote, matched rather than spelled out
/// because `launch_docker_agent` names it after the model. One file is
/// also the assertion that a launch leaves nothing else behind.
fn docker_agent_generated_file(home: &Path) -> PathBuf {
    let dir = home.join(".config/llmman/launch/docker-agent");
    let mut written: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|error| panic!("read {}: {error}", dir.display()))
        .map(|entry| entry.expect("read directory entry").path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("agent-") && name.ends_with(".yaml"))
        })
        .collect();
    written.sort();
    assert_eq!(
        written.len(),
        1,
        "expected one generated agent file in {}, found {written:?}",
        dir.display()
    );
    written.remove(0)
}

#[test]
fn launch_hermes_with_model() {
    eprintln!("[test] launch_hermes_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_hermes_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("hermes") {
        eprintln!(
            "skipping: hermes not on PATH — https://hermes-agent.nousresearch.com/install.sh"
        );
        return;
    }

    // `-z <prompt>`: hermes's own "purest one-shot" mode — single prompt
    // in, final reply text out, nothing else on stdout/stderr (see
    // hermes-agent.nousresearch.com/docs/reference/cli-commands).
    launch_and_assert("hermes", &["-z", PROMPT]);
}

#[test]
fn launch_openclaw_with_model() {
    eprintln!("[test] launch_openclaw_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_openclaw_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("openclaw") {
        eprintln!("skipping: openclaw not on PATH — npm install -g openclaw");
        return;
    }

    // `agent --local --message <prompt> --agent main`: one embedded turn,
    // no Gateway/TUI. Verified directly against the real published
    // `openclaw` npm package (2026.7.1-2) — its `--help` doesn't actually
    // have an `exec` subcommand despite docs.openclaw.ai/cli/agent
    // describing one; `--local` still requires an explicit session
    // selector (`--agent main`, the default agent onboarding creates).
    launch_and_assert_tolerating(
        "openclaw",
        &["agent", "--local", "--message", PROMPT, "--agent", "main"],
        openclaw_pull_registry_flake,
    );
}

/// `reasoning_effort: "low"` on MODEL, whose template has only the
/// thinking switch, leaves room for the answer (#703). Without a thinking
/// budget, its thinking used all 120 tokens at temperature 0.
#[test]
fn low_reasoning_effort_leaves_room_for_the_answer() {
    let _guard = lock_serial();
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    warm_model();

    let reply: serde_json::Value = reqwest::blocking::Client::builder()
        .timeout(TIMEOUT)
        .build()
        .expect("build client")
        .post(format!("{}/v1/chat/completions", llmman::daemon::server()))
        .json(&serde_json::json!({
            "model": MODEL,
            "messages": [{ "role": "user", "content": "Summarize: the sky is blue." }],
            "max_tokens": 120,
            "temperature": 0,
            "reasoning_effort": "low",
        }))
        .send()
        .expect("send the request")
        .json()
        .expect("a JSON reply");
    let choice = &reply["choices"][0];
    assert_eq!(choice["finish_reason"], "stop", "{reply}");
    let content = choice["message"]["content"].as_str().unwrap_or_default();
    assert!(!content.trim().is_empty(), "{reply}");
}

#[test]
fn ollama_think_low_leaves_room_for_the_answer() {
    let _guard = lock_serial();
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    warm_model();

    let reply: serde_json::Value = reqwest::blocking::Client::builder()
        .timeout(TIMEOUT)
        .build()
        .expect("build client")
        .post(format!("{}/api/chat", llmman::daemon::server()))
        .json(&serde_json::json!({
            "model": MODEL,
            "messages": [{ "role": "user", "content": "Summarize: the sky is blue." }],
            "stream": false,
            "think": "low",
            "options": { "num_predict": 120, "temperature": 0 },
        }))
        .send()
        .expect("send the request")
        .json()
        .expect("a JSON reply");
    assert_eq!(reply["done_reason"], "stop", "{reply}");
    let content = reply["message"]["content"].as_str().unwrap_or_default();
    assert!(!content.trim().is_empty(), "{reply}");
}

#[test]
fn launch_qwen_with_model() {
    eprintln!("[test] launch_qwen_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_qwen_with_model: acquired SERIAL");

    // No on_path skips: ci.yml's install_cli already fails the job when
    // qwen did not become a working binary, and a skip here would hide
    // exactly that (the ask on #332).
    //
    // Positional prompt, and first: `-p` is deprecated in 0.22.3, and
    // `--exclude-tools` is an array option that would swallow a prompt
    // placed after it. `--safe-mode` trims Qwen Code's eager tool set from
    // 23 to 10, but its `tool_search` can add the rest back on a later
    // turn, so `report_findings` is excluded by name as well: that tool's
    // `summary` field carries `maxLength: 2000`, which llama.cpp's
    // json-schema-to-grammar renders as `char{0,2000}` and its grammar
    // parser refuses at exactly that repetition count, for any model whose
    // template constrains tool calls with a grammar — MODEL's does. The
    // request then fails with a 400 before generating (issue #352).
    launch_and_assert_tolerating(
        "qwen",
        &[PROMPT, "--safe-mode", "--exclude-tools", "report_findings"],
        qwen_loop_detection,
    );
}

/// Qwen Code's own loop detector stops a run and exits 1 when a small
/// model repeats a tool call — a property of MODEL's sampling, not of
/// llmman, so tolerated the same way a timeout or a missing "pong" already
/// are (see `openclaw_pull_registry_flake` for the same shape).
fn qwen_loop_detection(stderr: &str) -> bool {
    stderr.contains("Loop detection halted the run")
}

#[test]
fn launch_dsh_with_model() {
    eprintln!("[test] launch_dsh_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_dsh_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    if !on_path("dsh") {
        eprintln!("skipping: dsh not on PATH — npm install -g @deepseek-ai/dsh");
        return;
    }

    // `--profile headless <prompt>`: dsh's own one-shot mode — answers
    // one task, prints the final assistant message, and exits. The
    // default `web` profile `llmman launch dsh` execs otherwise boots a
    // persistent server with no text reply, so it alone wouldn't fit
    // this file's launch_and_assert/"pong" pattern at all; a caller-
    // supplied `--profile` overrides it (see `dsh_args` in
    // `cmd::launch`) for exactly this case.
    launch_and_assert("dsh", &["--profile", "headless", PROMPT]);
}

#[test]
fn launch_goose_with_model() {
    eprintln!("[test] launch_goose_with_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] launch_goose_with_model: acquired SERIAL");
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    // Skipped rather than failed, unlike the npm CLIs: ci.yml can't
    // install goose on aarch64-pc-windows (no asset at v1.50.0) or on a
    // Windows runner whose bash has no `unzip` — see its own comment.
    if !on_path("goose") {
        eprintln!("skipping: goose not on PATH — https://github.com/aaif-goose/goose");
        return;
    }

    // `run -t <prompt> --no-session`: goose's own headless mode — one
    // instruction in, reply out, no session file left behind.
    launch_and_assert_rejecting(
        "goose",
        &["run", "-t", PROMPT, "--no-session"],
        goose_request_failed,
    );
}

/// goose exits 0 even when no request reached a model — verified against
/// 1.50.0 at a dead port ("Network error: ...") and at a 500 ("Ran into
/// this error: ..."). Without this a broken `goose_env` would read as the
/// model not saying "pong": retried, warned, green. Its third zero-exit
/// shape, "The model returned an empty response", stays retried — a 0.8b
/// model can legitimately produce one.
fn goose_request_failed(stdout: &str) -> bool {
    stdout.contains("Network error:") || stdout.contains("Ran into this error:")
}

/// `launch goose-desktop` hands the desktop app the same endpoint, model
/// and key as the CLI, and leaves the user's goose config alone.
///
/// The one test here that stubs its binary: Goose Desktop is a GUI with
/// no headless mode, so a real launch could only ever skip on a CI
/// runner. The shim records the environment it was given, which is what
/// the desktop path actually risks getting wrong.
#[test]
fn launch_goose_desktop_env() {
    let _guard = lock_serial();
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH (required to serve any model)");
        return;
    }
    warm_model();

    let home = fresh_home("goose-desktop");
    let shim_dir = home.join("shim");
    std::fs::create_dir_all(&shim_dir).expect("create shim dir");
    let dump = home.join("env.txt");

    // Only the variables under test are recorded, never the whole
    // environment: this inherits the runner's, and a failing assertion
    // below prints what it read. The dump path is hardcoded into the
    // script, so the shim needs nothing from the environment it records.
    let shim = if cfg!(windows) {
        let path = shim_dir.join("goose-desktop.cmd");
        let d = dump.display();
        std::fs::write(
            &path,
            format!(
                "@echo off\r\n\
                 set GOOSE_ > \"{d}\" 2>nul\r\n\
                 set OPENAI_ >> \"{d}\" 2>nul\r\n\
                 set OLLAMA_HOST >> \"{d}\" 2>nul\r\n\
                 exit /b 0\r\n"
            ),
        )
        .expect("write shim");
        path
    } else {
        let path = shim_dir.join("goose-desktop");
        std::fs::write(
            &path,
            format!(
                "#!/bin/sh\n\
                 env | grep -E '^(GOOSE_|OPENAI_|OLLAMA_HOST=)' > '{}'\n\
                 exit 0\n",
                dump.display()
            ),
        )
        .expect("write shim");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
                .expect("chmod shim");
        }
        path
    };
    assert!(shim.is_file());

    // A configuration the user already has. Seeded rather than left
    // absent: an empty home only proves nothing was created, not that an
    // existing provider survives, which is what this target promises.
    let goose_config = home.join(".config").join("goose");
    std::fs::create_dir_all(&goose_config).expect("create goose config dir");
    std::fs::write(goose_config.join("config.yaml"), GOOSE_CONFIG_SENTINEL)
        .expect("seed goose config");

    // Prepended, so `find_on_path("goose-desktop")` reaches the shim
    // before any real install on this machine.
    let path_var = std::env::var_os("PATH").unwrap_or_default();
    let mut dirs = vec![shim_dir.clone()];
    dirs.extend(std::env::split_paths(&path_var));
    let new_path = std::env::join_paths(dirs).expect("join PATH");

    let mut cmd = Command::new(llmman_bin());
    cmd.arg("launch")
        .arg("goose-desktop")
        .arg("--model")
        .arg(MODEL)
        .env("PATH", new_path)
        .env("HOME", &home)
        .env("USERPROFILE", &home)
        .env("XDG_CONFIG_HOME", home.join(".config"))
        .env("XDG_DATA_HOME", home.join(".local/share"));
    let output = try_spawn_with_timeout(
        cmd,
        TIMEOUT,
        &format!("`llmman launch goose-desktop --model {MODEL}`"),
    )
    .unwrap_or_else(|timed_out| panic!("{}", timed_out.message));
    assert!(
        output.status.success(),
        "launch goose-desktop failed (status: {:?})\n--- stdout ---\n{}\n--- stderr ---\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let dumped = std::fs::read_to_string(&dump).expect("shim did not record its environment");
    // Read before the assertions, so a failing one cannot leave the
    // recorded values on disk.
    let config_after = std::fs::read_to_string(goose_config.join("config.yaml")).ok();
    let mut left_in_config: Vec<String> = std::fs::read_dir(&goose_config)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    left_in_config.sort();
    let _ = std::fs::remove_dir_all(&home);

    let get = |key: &str| {
        dumped
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{key}=")))
            .map(str::trim)
    };
    // What a failure is allowed to print: the key's value is llmman's to
    // hand out, not this log's to carry.
    let shown = dumped
        .lines()
        .map(|l| match l.split_once('=') {
            Some((k, _)) if k.ends_with("API_KEY") => format!("{k}=<redacted>"),
            _ => l.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(get("GOOSE_PROVIDER"), Some("openai"));
    assert_eq!(get("OPENAI_BASE_PATH"), Some("v1/chat/completions"));
    assert!(
        get("OPENAI_API_KEY").is_some_and(|k| !k.is_empty()),
        "no OPENAI_API_KEY in:\n{shown}"
    );
    // llmman resolves the short name before launching (see
    // `shortnames::resolve_ollama_api`), so this is the resolved
    // reference, not necessarily MODEL verbatim.
    assert!(
        get("GOOSE_MODEL").is_some_and(|m| m.contains(MODEL)),
        "GOOSE_MODEL does not name {MODEL} in:\n{shown}"
    );
    // Scheme and authority only: goose appends OPENAI_BASE_PATH, so a
    // `/v1` here would request `/v1/v1/chat/completions`, and any other
    // path, query or fragment lands somewhere the daemon does not serve.
    // Checked structurally — rejecting `/v1` alone lets the rest past.
    let host = get("OPENAI_HOST").expect("no OPENAI_HOST");
    let authority = host
        .strip_prefix("http://")
        .or_else(|| host.strip_prefix("https://"))
        .unwrap_or_else(|| panic!("OPENAI_HOST is not an http(s) URL: {host}"));
    assert!(
        // A lone trailing slash still names the origin; anything after
        // it does not.
        !authority.trim_end_matches('/').contains(['/', '?', '#']),
        "OPENAI_HOST is not a bare origin: {host}"
    );

    // Byte-identical, and alone: a launch that rewrote, replaced or
    // deleted the provider the user configured — or left a `.bak` beside
    // it, as the file-configured targets do — fails here.
    assert_eq!(
        config_after.as_deref(),
        Some(GOOSE_CONFIG_SENTINEL),
        "launch goose-desktop changed the user's config.yaml"
    );
    assert_eq!(
        left_in_config,
        ["config.yaml"],
        "launch goose-desktop left files in ~/.config/goose"
    );
}

/// Seeded into the test's `~/.config/goose/config.yaml`: a provider that
/// is not llmman, so a launch that honored the file instead of the
/// environment would also be visible in `GOOSE_PROVIDER`.
const GOOSE_CONFIG_SENTINEL: &str = "GOOSE_PROVIDER: anthropic\nGOOSE_MODEL: claude-sonnet-4\n";

/// Verifies `daemon::ensure_server`'s fast-fail path end to end: when the
/// auto-spawned `llmman serve` dies during startup, the client command
/// must report the daemon's exit within seconds (via the poll loop's
/// try_wait) instead of polling a dead port for the full 60s, and its
/// error must carry the exit status plus the daemon log's tail.
///
/// The daemon is made to die instantly and deterministically, with no
/// network and no interference with any real daemon on the machine:
///
///   - `LLMMAN_HOST` points at a loopback port nothing listens on, so
///     the client never reuses (or stops) the developer's real daemon;
///   - `LLMMAN_RUNTIME=path` with a fake `llama-server` first on `PATH`,
///     so serve's runtime resolution returns immediately without probing
///     a container engine or downloading anything (it is found but never
///     executed: serve dies before launching it);
///   - `LLMMAN_MODELS` points at `<tmp>/store` while `<tmp>/cache`
///     already exists as a regular FILE, so `serve_async`'s
///     `create_dir_all(cache)` fails right after resolving llama-server,
///     before the listener ever binds. `<tmp>/serve.log` itself stays
///     writable, so the failure reaches the log and the client's error
///     can quote its tail.
///
/// Unlike the launch tests above, this needs neither `SERIAL` (its
/// daemon slot is its own unused port), `warm_model`, nor any binary on
/// the real `PATH`.
#[test]
fn ensure_server_fails_fast_when_daemon_dies_at_startup() {
    let (dir, port, path) = dead_daemon_setup("dead-daemon");

    let mut cmd = Command::new(llmman_bin());
    cmd.arg("pull").arg(MODEL);
    cmd.env("LLMMAN_HOST", format!("127.0.0.1:{port}"))
        .env("LLMMAN_MODELS", dir.join("store"))
        .env("LLMMAN_RUNTIME", "path")
        .env("PATH", path);

    let start = Instant::now();
    // 45s: comfortably above the few seconds a fast-fail takes, but below
    // ensure_server's own 60s poll, so a regression to the old
    // wait-out-the-timeout behavior fails here as a timeout rather than
    // passing slowly.
    let output = spawn_with_timeout(
        cmd,
        Duration::from_secs(45),
        "`llmman pull` against a daemon that dies at startup",
    );
    let elapsed = start.elapsed();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "pull against a dead daemon unexpectedly succeeded\n--- stderr ---\n{stderr}"
    );
    assert!(
        stderr.contains("exited during startup"),
        "expected the daemon's startup exit to be reported\n--- stderr ---\n{stderr}"
    );
    assert!(
        stderr.contains("last lines of"),
        "expected the error to quote the daemon log's tail\n--- stderr ---\n{stderr}"
    );
    assert!(
        elapsed < Duration::from_secs(30),
        "fast-fail took {elapsed:?}; it should report within seconds, \
         not wait out ensure_server's 60s poll"
    );
}

/// `llmman launch qwen` with no `--model` is refused on the flag before
/// any daemon work: under the dead-daemon setup a build that checked
/// after `ensure_server` would report the daemon's startup exit instead.
#[test]
fn launch_qwen_without_a_model_is_refused_before_the_daemon() {
    let (dir, port, path) = dead_daemon_setup("qwen-no-model");

    let mut cmd = Command::new(llmman_bin());
    cmd.arg("launch").arg("qwen");
    cmd.env("HOME", &dir)
        .env("USERPROFILE", &dir)
        .env("QWEN_HOME", dir.join(".qwen"))
        .env("LLMMAN_HOST", format!("127.0.0.1:{port}"))
        .env("LLMMAN_MODELS", dir.join("store"))
        .env("LLMMAN_RUNTIME", "path")
        .env("PATH", path);

    let start = Instant::now();
    let output = spawn_with_timeout(
        cmd,
        Duration::from_secs(45),
        "`llmman launch qwen` with no --model",
    );
    let elapsed = start.elapsed();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "launch without a model succeeded\n{stderr}"
    );
    assert!(stderr.contains("qwen needs a model"), "{stderr}");
    assert!(
        !stderr.contains("exited during startup"),
        "the daemon was started first\n{stderr}"
    );
    // 30s, like the dead-daemon test: the refusal takes milliseconds, but
    // a runner busy with the model tests beside it does not.
    assert!(
        elapsed < Duration::from_secs(30),
        "refusal took {elapsed:?}"
    );
}

/// A fresh `HOME` in which any daemon `llmman` spawns dies at startup, a
/// port nothing answers on for `LLMMAN_HOST`, and the `PATH` that makes
/// it so; see `ensure_server_fails_fast_when_daemon_dies_at_startup`.
fn dead_daemon_setup(label: &str) -> (PathBuf, u16, std::ffi::OsString) {
    let dir = fresh_home(label);
    let bin_dir = dir.join("bin");
    std::fs::create_dir_all(&bin_dir).expect("create fake bin dir");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let fake = bin_dir.join("llama-server");
        std::fs::write(&fake, "#!/bin/sh\nexit 0\n").expect("write fake llama-server");
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755))
            .expect("chmod fake llama-server");
    }
    // find_on_path probes `.exe`, `.cmd` and `.bat` on Windows; `.exe` is
    // the natural fake, and any content works: it is found, not executed.
    #[cfg(windows)]
    std::fs::write(bin_dir.join("llama-server.exe"), "not a real executable")
        .expect("write fake llama-server");

    std::fs::write(dir.join("cache"), "a file where serve expects a directory")
        .expect("write cache blocker file");

    // Bound once to reserve a definitely-free port number, then dropped:
    // the port must NOT stay listening, or ensure_server would take a
    // successful connect as "a daemon is already running" and never spawn
    // the one under test.
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .expect("bind probe listener")
        .local_addr()
        .expect("probe listener addr")
        .port();

    let path = std::env::join_paths(std::iter::once(bin_dir).chain(std::env::split_paths(
        &std::env::var_os("PATH").unwrap_or_default(),
    )))
    .expect("join PATH");
    (dir, port, path)
}

/// True for the one openclaw-specific failure shape confirmed live in
/// CI that isn't an llmman bug: its own onboarding independently
/// re-verifies whatever model it's given against a real public Docker
/// Hub registry path (`docker.io/ai/<name>`) — unrelated to our own
/// server, since it happens even for a model our server already has —
/// and that registry lookup is itself flaky/rate-limited: "pull failed:
/// copy image: docker.io/ai/... requested access to the resource is
/// denied". A real external dependency outside llmman's control, so
/// tolerated the same way a timeout or a missing "pong" already are.
fn openclaw_pull_registry_flake(stderr: &str) -> bool {
    stderr.contains("pull failed: copy image") || stderr.contains("FailoverError")
}

/// A tiny (135M-parameter, 8-bit-quantized) real safetensors model
/// already published in `mlx-community`'s own MLX-converted form —
/// small enough to pull quickly in CI, unlike [`MODEL`] (a ~740MB GGUF,
/// llama-server's own territory, never this file's own concern for the
/// SafeTensors/mlx path this test exercises instead). Deliberately not
/// added to [`MODEL`]/[`PROMPT`]'s own constants: those are shared by
/// every `llmman launch <integration>` test in this file, none of which
/// have anything to do with safetensors or mlx.
const MLX_MODEL: &str = "mlx-community/SmolLM2-135M-Instruct-8bit";

/// Exercises `llmman serve`'s `mlx_lm.server` backend
/// (`cmd::serve::backend::use_mlx_for_safetensors`/`spawn_mlx_server`) end to end:
/// a real `llmman run` against [`MLX_MODEL`], pulled fresh from
/// HuggingFace, served locally by a real `mlx_lm.server` process — not a
/// third-party integration launch like every other test in this file
/// (there's no "MLX" CLI to launch; this is purely about `llmman serve`'s
/// own backend selection).
///
/// Skips itself (rather than failing) on anything other than Apple
/// Silicon macOS, or when the `mlx_lm.server` the daemon would run is
/// missing: under `LLMMAN_RUNTIME=path` (CI's setting) the one on `PATH`,
/// otherwise llmman's own install (`crate::mlx_release`), which must be
/// complete already — the download is not what this measures. CI
/// installs `mlx-lm` onto `PATH` before this suite, on the two macOS
/// aarch64 legs, since the shared daemon's PATH is fixed at its spawn.
///
/// Unlike `launch_and_assert`'s small-model sampling-variance tolerance
/// (this file's other tests, talking to real third-party agentic CLIs
/// that can spin into a degenerate loop), this only asserts that
/// `llmman run` itself succeeds — a real non-zero exit here is always a
/// deterministic llmman bug (a bad spawn, a backend that never became
/// ready, a wire-model mismatch — see `cmd::serve::backend_wire_model`'s
/// own doc comment for the specific bug class this exists to catch)
/// worth failing loudly on immediately, not sampling noise worth
/// retrying — [`MLX_MODEL`]'s own reply content is never asserted on for
/// the same reason `warm_model` doesn't either.
///
/// Then confirms via `llmman ps` that [`MLX_MODEL`] is actually reported
/// as running under `mlx (local)` — not silently falling back to `vllm`
/// (which would also happen to answer this request successfully, since
/// vllm's own CPU backend can serve any plain safetensors model too, and
/// so could mask a real `use_mlx_for_safetensors` regression instead of
/// catching it).
#[test]
fn serve_mlx_safetensors_model() {
    eprintln!("[test] serve_mlx_safetensors_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] serve_mlx_safetensors_model: acquired SERIAL");

    if !(cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")) {
        eprintln!("skipping: mlx_lm.server only runs on Apple Silicon macOS");
        return;
    }
    let runtime_path = std::env::var("LLMMAN_RUNTIME").is_ok_and(|r| r == "path");
    if runtime_path && !on_path("mlx_lm.server") {
        eprintln!(
            "skipping: LLMMAN_RUNTIME=path and mlx_lm.server not on PATH — pip install mlx-lm"
        );
        return;
    }
    if !runtime_path && !llmman::mlx_release::installed() {
        eprintln!("skipping: llmman's mlx-lm is not installed — run `llmman serve --pull-only`");
        return;
    }

    let mut cmd = Command::new(llmman_bin());
    cmd.arg("run")
        .arg(MLX_MODEL)
        .arg("--think")
        .arg("false")
        .arg("--num-predict")
        .arg("64")
        .arg(PROMPT);
    let output = spawn_with_timeout(cmd, TIMEOUT, "llmman run (mlx safetensors model)");
    assert!(
        output.status.success(),
        "llmman run {MLX_MODEL} {PROMPT:?} failed (status: {:?})\n\
         --- stdout ---\n{}\n--- stderr ---\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let ps = Command::new(llmman_bin())
        .arg("ps")
        .output()
        .expect("spawn `llmman ps`");
    let ps_stdout = String::from_utf8_lossy(&ps.stdout);
    assert!(
        ps.status.success(),
        "llmman ps failed (status: {:?})\n--- stdout ---\n{ps_stdout}\n--- stderr ---\n{}",
        ps.status,
        String::from_utf8_lossy(&ps.stderr),
    );
    let model_row = ps_stdout
        .lines()
        .find(|line| line.contains("SmolLM2-135M-Instruct-8bit"))
        .unwrap_or_else(|| panic!("{MLX_MODEL} missing from `llmman ps` output:\n{ps_stdout}"));
    assert!(
        model_row.contains("mlx (local)"),
        "{MLX_MODEL} was not served by mlx_lm.server (use_mlx_for_safetensors regression?) \
         — `llmman ps` row: {model_row:?}"
    );
}

/// [`MODEL`]'s safetensors twin (`ai/qwen3.5:0.8b-safetensors`, ~1.7GB),
/// which vllm-plugin/tests/test_e2e.py also pulls into the same store.
#[cfg(unix)]
const VLLM_MODEL: &str = "qwen3.5:0.8b-safetensors";

/// A small dense model for [`serve_sglang_safetensors_model`], not
/// [`VLLM_MODEL`]: Qwen3.5's SGLang model code binds the Intel AMX CPU
/// kernels at import, and the MLX runtime has no Gated DeltaNet.
#[cfg(unix)]
const SGLANG_MODEL: &str = "hf.co/HuggingFaceTB/SmolLM2-135M-Instruct";

/// A cold ~1.7GB pull plus the daemon's 15m load deadline (see
/// `serve_with_engine`).
#[cfg(unix)]
const ENGINE_TIMEOUT: Duration = Duration::from_secs(1200);

/// A loopback port nothing is listening on right now, for a test's own
/// daemon (see [`serve_with_engine`]).
#[cfg(unix)]
fn free_loopback_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("bind an ephemeral loopback port")
        .local_addr()
        .expect("local_addr of the ephemeral listener")
        .port()
}

/// SGLang compiles its CPU kernel library `-march=x86-64-v4` (plus AMX)
/// and it SIGILLs at import without AVX-512 — which GitHub's mixed
/// AMD/Intel x86_64 fleet does not guarantee. Always true elsewhere
/// (aarch64 builds `-march=native`; macOS uses MLX).
#[cfg(unix)]
fn host_can_run_sglang_cpu_kernels() -> bool {
    if !(cfg!(target_os = "linux") && cfg!(target_arch = "x86_64")) {
        return true;
    }
    std::fs::read_to_string("/proc/cpuinfo")
        .map(|s| s.contains(" avx512f") && s.contains(" avx512bw"))
        .unwrap_or(false)
}

/// SIGTERMs the `daemon`'s process group (the engine shares it), SIGKILL
/// after 30s. Reaps the daemon as it goes: a zombie still counts for
/// `kill(pgid, 0)`.
#[cfg(unix)]
fn terminate_daemon(daemon: &mut std::process::Child) {
    let pgid = -(daemon.id() as i32);
    unsafe {
        libc::kill(pgid, libc::SIGTERM);
    }
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        let _ = daemon.try_wait();
        // ESRCH once the group is empty.
        if unsafe { libc::kill(pgid, 0) } != 0 {
            return;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    eprintln!(
        "[test] process group of daemon {} still alive 30s after SIGTERM; sending SIGKILL",
        daemon.id()
    );
    unsafe {
        libc::kill(pgid, libc::SIGKILL);
    }
    let _ = daemon.wait();
}

/// A `llmman serve` of the test's own on `host` with `env` added, in its
/// own process group, returned once it accepts TCP connections — so
/// its log is the test's to report and a slow start can't race
/// `ensure_server` into a second, doomed spawn.
#[cfg(unix)]
fn spawn_test_daemon(host: &str, env: &[(&str, &str)]) -> std::process::Child {
    use std::os::unix::process::CommandExt;
    let mut cmd = Command::new(llmman_bin());
    cmd.arg("serve")
        .env("LLMMAN_HOST", host)
        // As CI's job env: `auto` would pull a llama.cpp image first.
        .env(
            "LLMMAN_RUNTIME",
            std::env::var_os("LLMMAN_RUNTIME").unwrap_or_else(|| "path".into()),
        )
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .process_group(0);
    for (k, v) in env {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().expect("spawn `llmman serve`");
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        if std::net::TcpStream::connect(host).is_ok() {
            return child;
        }
        if let Ok(Some(status)) = child.try_wait() {
            panic!("`llmman serve` on {host} exited before listening: {status}");
        }
        assert!(
            Instant::now() < deadline,
            "`llmman serve` on {host} did not start listening within 120s"
        );
        std::thread::sleep(Duration::from_millis(250));
    }
}

/// `llmman run <model>` against a daemon of its own (fresh port,
/// `<engine_var>=<engine>` plus `env`), asserting `llmman ps` reports
/// `<engine> (local)`: the daemon's default engine could also answer
/// the prompt and mask a selection regression. `--host 0.0.0.0` is
/// appended to `args_var`, and the engine must still listen only on
/// loopback ([`assert_engine_listens_only_on_loopback`]). The daemon is
/// torn down afterwards. The shared daemon's [`MODEL`] is
/// unloaded first: its 256k context is several GiB a CPU engine start on
/// a 16GB runner cannot spare. `--think false --num-predict 32`: see
/// `warm_model`.
#[cfg(unix)]
fn serve_with_engine(
    engine_var: &str,
    engine: &str,
    args_var: &str,
    model: &str,
    env: &[(&str, &str)],
) {
    let _ = Command::new(llmman_bin())
        .args(["stop", MODEL])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    let host = format!("127.0.0.1:{}", free_loopback_port());
    let args = format!(
        "{} --host 0.0.0.0",
        std::env::var(args_var).unwrap_or_default()
    );
    // 15m: what vllm-plugin's e2e allows a CPU `vllm serve` to become
    // healthy in; llmman's 10m default gave up first on slow runners.
    let mut daemon_env = vec![(engine_var, engine), ("LLMMAN_LOAD_TIMEOUT", "15m")];
    daemon_env.extend_from_slice(env);
    daemon_env.push((args_var, &args));
    let mut daemon = spawn_test_daemon(&host, &daemon_env);
    eprintln!("[test] {engine}: daemon pid {} on {host}", daemon.id());

    let result = std::panic::catch_unwind(|| {
        let mut cmd = Command::new(llmman_bin());
        cmd.env("LLMMAN_HOST", &host)
            .arg("run")
            .arg(model)
            .arg("--think")
            .arg("false")
            .arg("--num-predict")
            .arg("32")
            .arg(PROMPT);
        let output = spawn_with_timeout(cmd, ENGINE_TIMEOUT, &format!("llmman run ({engine})"));
        assert!(
            output.status.success(),
            "llmman run {model} {PROMPT:?} via {engine} failed (status: {:?})\n\
             --- stdout ---\n{}\n--- stderr ---\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        eprintln!(
            "[test] {engine}: reply: {}",
            String::from_utf8_lossy(&output.stdout).trim()
        );

        let ps = Command::new(llmman_bin())
            .env("LLMMAN_HOST", &host)
            .arg("ps")
            .output()
            .expect("spawn `llmman ps`");
        let ps_stdout = String::from_utf8_lossy(&ps.stdout);
        assert!(
            ps.status.success(),
            "llmman ps failed (status: {:?})\n--- stdout ---\n{ps_stdout}\n--- stderr ---\n{}",
            ps.status,
            String::from_utf8_lossy(&ps.stderr),
        );
        // `ps` prints the canonical reference; match on the repo name,
        // which every spelling of `model` contains.
        let needle = model
            .trim_start_matches("hf.co/")
            .split(':')
            .next()
            .unwrap_or(model);
        let model_row = ps_stdout
            .lines()
            .find(|line| line.contains(needle))
            .unwrap_or_else(|| panic!("{model} missing from `llmman ps` output:\n{ps_stdout}"));
        let expected = format!("{engine} (local)");
        assert!(
            model_row.contains(&expected),
            "{model} was not served by {engine} ({engine_var} regression?) \
             — `llmman ps` row: {model_row:?}"
        );
        assert_engine_listens_only_on_loopback(&host);
    });

    terminate_daemon(&mut daemon);
    if let Err(payload) = result {
        std::panic::resume_unwind(payload);
    }
}

/// `llmman serve`'s `vllm` backend end to end, wherever CI installs vLLM
/// (Linux: CPU wheel; macOS: vllm-metal). `LLMMAN_SAFETENSORS_ENGINE=vllm`
/// is what makes the macOS leg deterministic, where mlx_lm.server is also
/// installed. Platform flags (`--dtype bfloat16 --enforce-eager`,
/// `--max-num-seqs 1`, `VLLM_*`) come from the CI step's environment as
/// `LLMMAN_VLLM_ARGS`/`VLLM_*`; `LLMMAN_CONTEXT_LENGTH=1024` bounds the
/// KV cache.
#[cfg(unix)]
#[test]
fn serve_vllm_safetensors_model() {
    eprintln!("[test] serve_vllm_safetensors_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] serve_vllm_safetensors_model: acquired SERIAL");

    if !on_path("vllm") {
        eprintln!("skipping: vllm not on PATH — see ci.yml's \"Install vLLM (e2e)\" step");
        return;
    }
    serve_with_engine(
        "LLMMAN_SAFETENSORS_ENGINE",
        "vllm",
        "LLMMAN_VLLM_ARGS",
        VLLM_MODEL,
        &[("LLMMAN_CONTEXT_LENGTH", "1024")],
    );
}

#[cfg(unix)]
fn assert_engine_listens_only_on_loopback(host: &str) {
    let ps: serde_json::Value = llmman::daemon::client()
        .expect("HTTP client")
        .get(format!("http://{host}/api/ps"))
        .send()
        .and_then(|resp| resp.error_for_status())
        .and_then(|resp| resp.json())
        .expect("GET /api/ps");
    let models = ps["models"].as_array().expect("/api/ps models");
    assert_eq!(models.len(), 1, "expected one loaded model: {ps}");
    let port = u16::try_from(models[0]["port"].as_u64().expect("engine port")).unwrap();
    let connect = |ip: std::net::IpAddr| {
        std::net::TcpStream::connect_timeout(&(ip, port).into(), Duration::from_secs(5))
    };
    connect(std::net::Ipv4Addr::LOCALHOST.into())
        .unwrap_or_else(|e| panic!("engine is not listening on 127.0.0.1:{port}: {e}"));
    let ip = non_loopback_ipv4()
        .expect("no non-loopback IPv4 address to check the engine is not reachable on");
    // Only a refusal shows nothing listens there; a timeout may be a firewall.
    match connect(ip) {
        Ok(_) => panic!(
            "engine answers on {ip}:{port}: an LLMMAN_*_ARGS --host overrode llmman's loopback"
        ),
        Err(e) if e.kind() == std::io::ErrorKind::ConnectionRefused => {}
        Err(e) => panic!("inconclusive: connecting to {ip}:{port} failed with {e}, not a refusal"),
    }
}

/// This machine's address on its default route: the local end of a UDP
/// socket "connected" to a documentation address, which sends nothing.
#[cfg(unix)]
fn non_loopback_ipv4() -> Option<std::net::IpAddr> {
    let socket = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("192.0.2.1:9").ok()?;
    let ip = socket.local_addr().ok()?.ip();
    (!ip.is_loopback() && !ip.is_unspecified()).then_some(ip)
}

/// `llmman serve`'s `sglang` backend end to end, wherever CI installs
/// SGLang (Linux: its CPU engine; macOS: its MLX runtime). Platform
/// variables (`SGLANG_USE_CPU_ENGINE`/`SGLANG_USE_MLX`,
/// `LLMMAN_SGLANG_ARGS=--disable-cuda-graph`) come from the CI step.
/// Skips on x86_64 without AVX-512 ([`host_can_run_sglang_cpu_kernels`]).
#[cfg(unix)]
#[test]
fn serve_sglang_safetensors_model() {
    eprintln!("[test] serve_sglang_safetensors_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] serve_sglang_safetensors_model: acquired SERIAL");

    if !on_path("sglang") {
        eprintln!("skipping: sglang not on PATH — see ci.yml's \"Install SGLang (e2e)\" step");
        return;
    }
    if !host_can_run_sglang_cpu_kernels() {
        eprintln!(
            "skipping: this x86_64 CPU has no AVX-512, which SGLang's CPU kernel library \
             is compiled for (it SIGILLs at import)"
        );
        return;
    }
    serve_with_engine(
        "LLMMAN_SAFETENSORS_ENGINE",
        "sglang",
        "LLMMAN_SGLANG_ARGS",
        SGLANG_MODEL,
        &[("LLMMAN_CONTEXT_LENGTH", "1024")],
    );
}

/// `llmman serve`'s `ds4` backend end to end, on the ds4 model
/// `LLMMAN_E2E_DS4_MODEL` names; it skips, rather than pulls, a model
/// that is not already in the store. CI skips it: ds4-server needs Metal
/// or CUDA and a ds4 GGUF too large for a runner.
#[cfg(unix)]
#[test]
fn serve_ds4_gguf_model() {
    eprintln!("[test] serve_ds4_gguf_model: acquiring SERIAL");
    let _guard = lock_serial();
    eprintln!("[test] serve_ds4_gguf_model: acquired SERIAL");

    if !on_path("ds4-server") {
        eprintln!("skipping: ds4-server not on PATH");
        return;
    }
    let Some(model) = std::env::var("LLMMAN_E2E_DS4_MODEL")
        .ok()
        .filter(|m| !m.trim().is_empty())
    else {
        eprintln!("skipping: LLMMAN_E2E_DS4_MODEL is not set to a ds4 model ref");
        return;
    };
    // `llmman run` pulls a missing model; `show` only reads the local store,
    // so a model that is not there skips instead of starting a huge download.
    let in_store = Command::new(llmman_bin())
        .args(["show", "--parameters", &model])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    if !in_store {
        eprintln!("skipping: {model} is not in the local store; pull it first");
        return;
    }
    serve_with_engine("LLMMAN_GGUF_ENGINE", "ds4", "LLMMAN_DS4_ARGS", &model, &[]);
}

/// llmman never links against llama.cpp: `mediagen::ffi` dlopens the
/// ggml/llama libraries next to `llama-server` and mirrors two of its
/// structs. Checked here against the release CI pins (on `PATH`), not
/// only when a diffusion model loads: a dropped symbol fails `Api::load`,
/// a moved struct `check_layout`.
#[test]
fn mediagen_ffi_binds_the_llama_cpp_on_path() {
    let Some(dir) = llama_libs_on_path() else {
        return;
    };
    eprintln!("binding the ggml/llama libraries in {}", dir.display());
    let api = llmman::mediagen::ffi::Api::load(&dir).unwrap();
    api.check_layout().unwrap();
}

/// A GGUF metadata value on one line: strings bare, arrays joined,
/// numbers with their type.
fn gguf_value_text(v: &llmman::gguf::Value) -> String {
    use llmman::gguf::Value;
    match v {
        Value::String(s) => s.clone(),
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(gguf_value_text).collect();
            format!("[{}]", parts.join(", "))
        }
        other => format!("{other:?}"),
    }
}

/// Shared setup of the two ignored Canary tests: the serial lock and the
/// three GGUFs of `LLMMAN_CANARY_GGUF_DIR`, the acceptance list of issue
/// #714, as `(label, path)`.
struct CanaryFixture {
    // ggml registers backend modules without a lock; two `Api::load`
    // calls on parallel test threads race on that registry.
    _guard: std::sync::MutexGuard<'static, ()>,
    files: [(&'static str, PathBuf); 3],
}

/// Takes the serial lock and resolves the three required files. Skips
/// (None) when `LLMMAN_CANARY_GGUF_DIR` is unset or empty; a missing
/// file fails.
fn canary_fixture() -> Option<CanaryFixture> {
    let guard = lock_serial();
    let Some(dir) = std::env::var_os("LLMMAN_CANARY_GGUF_DIR")
        .map(PathBuf::from)
        .filter(|d| !d.as_os_str().is_empty())
    else {
        eprintln!("skipping: LLMMAN_CANARY_GGUF_DIR is not set");
        return None;
    };
    let files = [
        ("Q8_0", "canary-1b-v2-Q8_0.gguf"),
        ("F16", "canary-1b-v2-F16.gguf"),
        ("Q6_K", "canary-1b-v2-Q6_K.gguf"),
    ]
    .map(|(label, file)| {
        let path = dir.join(file);
        assert!(path.is_file(), "{file}: missing in {}", dir.display());
        (label, path)
    });
    Some(CanaryFixture {
        _guard: guard,
        files,
    })
}

/// The ggml/llama library directory of the `llama-server` on `PATH`, or
/// None (skip) when there is none.
fn llama_libs_on_path() -> Option<PathBuf> {
    if !on_path("llama-server") {
        eprintln!("skipping: llama-server not on PATH");
        return None;
    }
    let (_, dir) = llmman::cmd::serve::llama_libs(llmman::cmd::serve::Runtime::Path, None).unwrap();
    Some(dir)
}

/// Canary 1B v2 (handy-computer/canary-1b-v2-gguf): each GGUF in
/// `LLMMAN_CANARY_GGUF_DIR` meets the transcribe.cpp tensor contract
/// (`src/arch/canary/weights.cpp` at db678a0, kept in `speech::model`)
/// through the loader's own index, and a quantized tensor survives a
/// byte round-trip through the same `ggml_backend_tensor_set` the loader
/// uses. All three files are required: Q8_0, F16 and Q6_K, the
/// acceptance list of issue #714. Run by hand:
/// `LLMMAN_CANARY_GGUF_DIR=<dir> cargo test --release --test launch_e2e
/// canary -- --ignored --nocapture`.
#[test]
#[ignore]
fn canary_gguf_loads_through_mediagen_weights() {
    use llmman::mediagen::ffi;
    use llmman::speech::model::{check_index, contract_names, CanaryHParams};
    use std::collections::BTreeMap;
    use std::io::Seek;

    let Some(fixture) = canary_fixture() else {
        return;
    };

    // Every GGUF stays open for the byte round-trip below. Candidates per
    // ggml type: (index into `opened`, tensor).
    let mut opened: Vec<(PathBuf, std::fs::File)> = Vec::new();
    let mut q8_0: Option<(usize, llmman::mediagen::weights::TensorDesc)> = None;
    let mut q6_k: Option<(usize, llmman::mediagen::weights::TensorDesc)> = None;
    for (_, path) in fixture.files {
        let file = path.file_name().unwrap().to_string_lossy();
        eprintln!("== {}", path.display());

        let info = llmman::gguf::read_info(&path).unwrap();
        assert_eq!(info.architecture(), Some("canary"), "{file}");
        let mut keys: Vec<&String> = info
            .metadata
            .keys()
            .filter(|k| {
                k.starts_with("stt.canary.")
                    || k.starts_with("stt.frontend.")
                    || *k == "general.languages"
                    || *k == "stt.translation.pairs"
            })
            .collect();
        keys.sort();
        for k in keys {
            eprintln!("  {k} = {}", gguf_value_text(&info.metadata[k]));
        }
        let hp = CanaryHParams::from_info(&info)
            .unwrap_or_else(|e| panic!("{file}: hyperparameters: {e:#}"));

        let mut file_h = std::fs::File::open(&path).unwrap();
        let idx = llmman::mediagen::weights::read_index_from(&mut file_h, &path).unwrap();
        let mut histogram: BTreeMap<u32, usize> = BTreeMap::new();
        for t in &idx.tensors {
            *histogram.entry(t.ty).or_default() += 1;
        }
        eprintln!(
            "  tensors: {}, types {{ggml type id: count}}: {histogram:?}",
            idx.tensors.len()
        );

        // `check_index` lists every unmapped, missing and misshapen name.
        check_index(&idx, &hp).unwrap_or_else(|e| panic!("{file}: {e:#}"));
        let expected = contract_names(hp.enc_n_layers, hp.dec_n_layers, hp.encoder_decoder_proj);
        eprintln!(
            "  contract: {} names, all mapped (encoder_decoder_proj = {})",
            expected.len(),
            hp.encoder_decoder_proj
        );
        eprintln!("  shapes: every tensor matches the hyperparameters");

        for t in &idx.tensors {
            let slot = match t.ty {
                ffi::ty::Q8_0 => &mut q8_0,
                ffi::ty::Q6_K => &mut q6_k,
                _ => continue,
            };
            if slot.is_none() {
                *slot = Some((opened.len(), t.clone()));
            }
        }
        opened.push((path, file_h));
    }

    // All three files are required, so both quantized types must turn up.
    let picks: Vec<_> = [("Q8_0", q8_0), ("Q6_K", q6_k)]
        .into_iter()
        .map(|(label, pick)| {
            let pick = pick.unwrap_or_else(|| panic!("round-trip: no {label} tensor in any file"));
            (label, pick)
        })
        .collect();

    // The metadata and contract checks above need no libraries; only the
    // round-trip does, so an opted-in run never passes without them.
    let Some(lib_dir) = llama_libs_on_path() else {
        return;
    };
    eprintln!(
        "== round-trip through the ggml CPU backend in {}",
        lib_dir.display()
    );
    let api = ffi::Api::load(&lib_dir).unwrap();
    // SAFETY: plain ggml calls on handles this block creates and frees.
    // The backend, context, tensor and buffer pointers are null-checked
    // before use; `buft` is the backend's default buffer type and is not.
    unsafe {
        let backend =
            (api.ggml_backend_init_by_type)(ffi::GGML_BACKEND_DEVICE_TYPE_CPU, std::ptr::null());
        assert!(!backend.is_null(), "no ggml CPU backend");
        let buft = (api.ggml_backend_get_default_buffer_type)(backend);
        for (label, (i, desc)) in picks {
            let (path, file) = &mut opened[i];
            let ctx = (api.ggml_init)(ffi::GgmlInitParams {
                mem_size: (api.ggml_tensor_overhead)(),
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            });
            assert!(!ctx.is_null(), "ggml_init");
            let t = (api.ggml_new_tensor)(
                ctx,
                desc.ty,
                desc.ne.len() as std::ffi::c_int,
                desc.ne.as_ptr(),
            );
            assert!(!t.is_null(), "ggml_new_tensor {}", desc.name);
            let buf = (api.ggml_backend_alloc_ctx_tensors_from_buft)(ctx, buft);
            assert!(!buf.is_null(), "alloc {}", desc.name);
            // `read_gguf` leaves `nbytes` at 0; the loader sizes the read
            // with `ggml_row_size` (weights.rs `load_files`), as here.
            let elems: i64 = desc.ne.iter().product();
            let nbytes = (api.ggml_row_size)(desc.ty, elems);
            assert_eq!((api.ggml_nbytes)(t), nbytes, "{}: ggml_nbytes", desc.name);
            let file_len = file.metadata().unwrap().len();
            assert!(
                desc.offset + nbytes as u64 <= file_len,
                "{}: offset {} + {nbytes} past the end of {}",
                desc.name,
                desc.offset,
                path.display()
            );
            file.seek(std::io::SeekFrom::Start(desc.offset)).unwrap();
            let mut src = vec![0u8; nbytes];
            file.read_exact(&mut src).unwrap();
            (api.ggml_backend_tensor_set)(t, src.as_ptr() as *const _, 0, nbytes);
            let mut back = vec![0u8; nbytes];
            (api.ggml_backend_tensor_get)(t, back.as_mut_ptr() as *mut _, 0, nbytes);
            assert!(
                src == back,
                "{}: bytes differ after the round-trip",
                desc.name
            );
            eprintln!(
                "  {label} {} ne={:?} ty={} offset={} {nbytes} bytes: identical",
                desc.name, desc.ne, desc.ty, desc.offset
            );
            (api.ggml_backend_buffer_free)(buf);
            (api.ggml_free)(ctx);
        }
        (api.ggml_backend_free)(backend);
    }
}

/// Canary 1B v2 (handy-computer/canary-1b-v2-gguf) through
/// `speech::model::load` on the ggml CPU backend of the llama.cpp on
/// `PATH`: each GGUF in `LLMMAN_CANARY_GGUF_DIR` comes back as a
/// `CanaryModel` with its hyperparameters, vocabulary and every contract
/// tensor in one weight buffer. All three files are required: Q8_0, F16
/// and Q6_K, the acceptance list of issue #714. Run by hand:
/// `LLMMAN_CANARY_GGUF_DIR=<dir> cargo test --release --test launch_e2e
/// canary -- --ignored --nocapture`.
#[test]
#[ignore]
fn canary_model_loads_on_cpu_backend() {
    use llmman::mediagen::backend::Backend;
    use llmman::mediagen::ffi::{ty, Api, Tensor};
    use std::collections::BTreeMap;
    use std::io::Seek;

    let Some(fixture) = canary_fixture() else {
        return;
    };
    let Some(lib_dir) = llama_libs_on_path() else {
        return;
    };
    eprintln!("== CPU backend from {}", lib_dir.display());
    // the weight contexts hold `&'static Api`, as in `mediagen_backend`
    let api: &'static Api = Box::leak(Box::new(Api::load(&lib_dir).unwrap()));
    let n_threads = std::thread::available_parallelism().map_or(4, |n| n.get() as i32);
    let be = Backend::new(api, false, n_threads, 1024).unwrap();

    let mut weight_bytes: BTreeMap<&str, usize> = BTreeMap::new();
    for (label, path) in &fixture.files {
        let file = path.file_name().unwrap().to_string_lossy();
        eprintln!("== {}", path.display());
        let model = llmman::speech::model::load(path, &be).unwrap();
        eprintln!("  tensors: {}", model.tensor_count());
        eprintln!("  hparams: {:?}", model.hp);
        eprintln!(
            "  weights: {} bytes ({:.1} MiB)",
            model.weight_bytes(),
            model.weight_bytes() as f64 / 1024.0 / 1024.0
        );
        assert_eq!(model.tensor_count(), 1476, "{file}");
        assert_eq!(model.tensors.enc_blocks.len(), 32, "{file}");
        assert_eq!(model.tensors.dec_layers.len(), 8, "{file}");
        let hp = &model.hp;
        assert_eq!(
            (
                hp.enc_n_layers,
                hp.enc_d_model,
                hp.enc_n_heads,
                hp.enc_d_ff,
                hp.enc_conv_kernel,
                hp.enc_subsampling_factor,
                hp.enc_subsampling_channels,
                hp.enc_pos_emb_max_len,
            ),
            (32, 1024, 8, 4096, 9, 8, 256, 5000),
            "{file}: encoder"
        );
        assert_eq!(
            (hp.dec_n_layers, hp.dec_max_position, hp.dec_vocab_size),
            (8, 1024, 16384),
            "{file}: decoder"
        );
        assert_eq!(model.vocab.pieces.len(), 16384, "{file}: vocab");
        assert_eq!(
            (hp.eos, hp.bos, hp.pad, hp.startofcontext),
            (3, 4, 2, 7),
            "{file}: special tokens"
        );
        // The bytes `load` uploaded against the file: one conv kernel,
        // one quantized (or F16) linear, one F32 vector and the stored
        // position table. `read_index_from` offsets are absolute in the file.
        let mut file_h = std::fs::File::open(path).unwrap();
        let idx = llmman::mediagen::weights::read_index_from(&mut file_h, path).unwrap();
        let samples: [(&str, Tensor); 4] = [
            (
                "enc.pre_encode.conv.0.weight",
                model.tensors.pre_encode.conv0,
            ),
            (
                "enc.blocks.0.ff1.linear1.weight",
                model.tensors.enc_blocks[0].ff1_linear1,
            ),
            (
                "enc.blocks.0.norm_ff1.weight",
                model.tensors.enc_blocks[0].norm_ff1,
            ),
            ("dec.embed.pos_enc", model.tensors.dec_embed.pos_enc),
        ];
        for (name, t) in samples {
            let desc = idx
                .tensors
                .iter()
                .find(|d| d.name == name)
                .unwrap_or_else(|| panic!("{file}: {name}: not in the index"));
            // BF16 is the one type `Weights` converts; these files carry none.
            assert_ne!(desc.ty, ty::BF16, "{file}: {name}: BF16 in the file");
            assert!(!t.is_null(), "{file}: {name}: null tensor");
            // SAFETY: `model` keeps the tensor context and its backing
            // buffer alive; `t` is null-checked above and `back` has
            // `ggml_nbytes(t)` writable bytes.
            let (loaded_ty, back) = unsafe {
                assert_eq!((*t).type_, desc.ty, "{file}: {name}: type");
                let nbytes = (api.ggml_nbytes)(t);
                let mut back = vec![0u8; nbytes];
                (api.ggml_backend_tensor_get)(t, back.as_mut_ptr() as *mut _, 0, nbytes);
                ((*t).type_, back)
            };
            let elems: i64 = desc.ne.iter().product();
            // SAFETY: a pure size computation on a ggml type id.
            let nbytes = unsafe { (api.ggml_row_size)(desc.ty, elems) };
            file_h.seek(std::io::SeekFrom::Start(desc.offset)).unwrap();
            let mut src = vec![0u8; nbytes];
            file_h.read_exact(&mut src).unwrap();
            assert_eq!(src.len(), back.len(), "{file}: {name}: size");
            let first_mismatch = src.iter().zip(&back).position(|(a, b)| a != b);
            assert!(
                src == back,
                "{file}: {name}: first mismatch at byte {}",
                first_mismatch.unwrap()
            );
            eprintln!("  {name} ty={loaded_ty} {} bytes: identical", back.len());
        }
        weight_bytes.insert(*label, model.weight_bytes());
        // the buffer goes before the backend it lives on
        drop(model);
    }
    assert!(
        weight_bytes["Q8_0"] < weight_bytes["F16"],
        "weight buffers: {weight_bytes:?}"
    );
    drop(be);
}
