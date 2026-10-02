//! `llmman launch goose-desktop`.
//!
//! Everywhere but macOS the app takes a single-instance lock and hands a
//! second launch to the copy already running, which keeps its own
//! provider. Most of this file decides whether an instance holds that
//! lock, and offers to quit it.

use std::path::{Path, PathBuf};
#[cfg(any(windows, test))]
use std::process::Command;

use super::goose::goose_env;
use super::sandbox;
use super::{accepts_prompt, env_dir, exec_with_env, find_on_path, find_on_path_unless, server};

/// goose-desktop: [`super::goose::launch_goose`]'s environment, pointed at the desktop
/// app. Configured through the environment alone, so nothing is written
/// and a `goose configure` provider survives the launch.
///
/// Verified against Goose Desktop 1.52.0: the launched process carries
/// these variables, a prompt typed into it reaches the daemon, and
/// neither the launch nor the conversation touches `~/.config/goose`.
///
/// On macOS that is all of it: 1.52.0 takes no single-instance lock
/// there, so two launches make two processes, each with the environment
/// it was given. Elsewhere it does take one, and hands this launch to the
/// copy already running, which keeps its own provider — read out of
/// 1.52.0's `app.asar`, not run.
///
/// So the launch is refused when [`goose_desktop_instance`] finds a
/// holder, and warned about when it cannot tell. Nothing is inferred
/// afterwards from the exit code or from how long the app ran: a second
/// instance quitting and a user quitting look identical from here, and
/// guessing between them cried wolf on every quick exit.
pub(super) fn launch_goose_desktop(
    model: &str,
    api_key: &str,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_goose_desktop().ok_or_else(|| {
        anyhow::anyhow!(
            "goose-desktop is not installed\n\
             Install Goose Desktop from https://github.com/aaif-goose/goose, \
             or run 'llmman launch goose' for the CLI."
        )
    })?;
    // Before the launch, because afterwards there is nothing to tell a
    // handover from an ordinary quick exit: both are an exit 0.
    //
    // Not on macOS, which takes no lock, so there is no handover to find.
    // Not under a sandbox either: that launch has its own profile and
    // process table, so no window on this machine can be handed it.
    if !cfg!(target_os = "macos") && !sandbox::active() {
        match dirs::home_dir() {
            Some(home) => {
                let config = env_dir("XDG_CONFIG_HOME").unwrap_or_else(|| home.join(".config"));
                match goose_desktop_instance(&bin, &home, &config) {
                    GooseInstance::Confirmed(pid) => {
                        offer_to_quit_goose_desktop(pid, &bin, &home, &config)?
                    }
                    GooseInstance::Held => anyhow::bail!(goose_desktop_lock_held(
                        &goose_desktop_user_data(&home, &config)
                    )),
                    GooseInstance::Unknown => eprintln!("{}", goose_desktop_unknown_warning()),
                    GooseInstance::Free => {}
                }
            }
            // Without a home directory there is no profile to read the
            // lock from, which is the same not-knowing.
            None => eprintln!("{}", goose_desktop_unknown_warning()),
        }
    }
    let host = server();
    exec_with_env(&bin, extra_args, &goose_env(model, api_key, &host))
}

/// What a probe for a running Goose Desktop found.
///
/// Two questions, and they take different evidence. Whether this launch
/// will be handed over: anything alive holding the single-instance lock
/// does that, whatever binary it is, because that is what Chromium keys
/// on. Whether the holder may be signalled: only a pid confirmed to be
/// `bin`, or a `SIGTERM` lands on a stranger whose pid was reused. One
/// bar for both would either miss handovers or signal blind.
#[derive(Debug, PartialEq, Eq)]
enum GooseInstance {
    /// A live instance of the binary this launch resolved: this launch
    /// would be handed to it, and its pid is ours to signal.
    Confirmed(u32),
    /// Something live holds the lock, but not confirmably that binary —
    /// another build, a process this user cannot read, or a pid reused
    /// after a crash. The launch is refused, since a holder is what causes
    /// the handover, and nothing is signalled on this much.
    Held,
    /// The probe reached no answer — it could not run, or could not read
    /// what it needed. Nothing is known either way.
    Unknown,
    /// Nothing holds the lock.
    Free,
}

/// Electron's `userData`, named for the bundle rather than the CLI.
/// Shared with [`super::sandbox_state`] so the directory a sandbox mounts and
/// the one the lock is read from cannot drift apart.
pub(super) fn goose_desktop_user_data(home: &Path, config: &Path) -> PathBuf {
    if cfg!(target_os = "macos") {
        home.join("Library")
            .join("Application Support")
            .join("Goose")
    } else if cfg!(windows) {
        env_dir("APPDATA")
            .unwrap_or_else(|| home.join("AppData").join("Roaming"))
            .join("Goose")
    } else {
        config.join("Goose")
    }
}

/// Who holds Chromium's single-instance lock in `user_data`.
///
/// `SingletonLock` is Chromium's own lock, which Electron takes through
/// it: a symlink whose target is `<hostname>-<pid>`. Chromium breaks a
/// lock whose holder is gone and becomes the primary itself, so a stale
/// one is [`GooseInstance::Free`] here too — it must not stand in the way
/// of a launch that would have worked.
///
/// Not on Windows, which keeps this lock in a named mutex with nothing on
/// disk to read — see [`goose_desktop_running_windows`].
#[cfg(not(windows))]
fn single_instance_lock_holder(user_data: &Path, bin: &Path) -> GooseInstance {
    let target = match std::fs::read_link(user_data.join("SingletonLock")) {
        Ok(target) => target,
        // Conclusively no holder: no lock (`NotFound`), or something
        // there that Chromium did not write, since it always writes a
        // symlink and reading any other file type gives `EINVAL`.
        Err(e)
            if matches!(
                e.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::InvalidInput
            ) =>
        {
            return GooseInstance::Free;
        }
        // Anything else — an unreadable profile directory, most likely —
        // is not evidence that nothing is running.
        Err(_) => return GooseInstance::Unknown,
    };
    // A target that does not end in a number names no holder.
    let Some(pid) = target
        .to_str()
        .and_then(|t| t.rsplit_once('-'))
        .and_then(|(_, pid)| pid.parse::<u32>().ok())
    else {
        return GooseInstance::Free;
    };
    // `kill` and `/proc` take a signed pid. A number too large to be one
    // is not a pid Chromium wrote, and `pid_alive` rejects zero.
    let Ok(signed) = i32::try_from(pid) else {
        return GooseInstance::Free;
    };
    if !pid_alive(signed) {
        return GooseInstance::Free;
    }
    if process_is(signed, bin) {
        GooseInstance::Confirmed(pid)
    } else {
        GooseInstance::Held
    }
}

/// Whether `pid` exists, without signalling it: `kill(pid, 0)` performs
/// the permission and existence checks and delivers nothing. `EPERM` is a
/// live process this user may not signal, which counts as alive — the
/// caller only needs to know the lock has a holder.
///
/// Zero and negatives are rejected first, because to `kill` they address
/// process *groups*, this process's own among them.
#[cfg(not(windows))]
fn pid_alive(pid: i32) -> bool {
    if pid <= 0 {
        return false;
    }
    // SAFETY: kill(2) with signal 0 sends nothing; it only reports
    // whether the process exists and could be signalled.
    if unsafe { libc::kill(pid, 0) } == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// Whether `pid` is running `bin`, by the executable `/proc/<pid>/exe`
/// resolves to — what the Windows side matches as `ExecutablePath`.
///
/// This is the bar for *signalling*, not for refusing. `SingletonLock`
/// outlives an app that was hard-killed, and its pid may since have been
/// reused by something unrelated, which [`quit_goose_desktop`] would
/// otherwise SIGTERM after a prompt naming Goose Desktop. The same goes
/// for a lock written on another machine over a shared home directory,
/// whose pid means nothing here.
///
/// Only Linux has `/proc/<pid>/exe`; an unreadable one — another user's
/// process, or any other unix — counts as not it. The launch is still
/// refused in that case, as [`GooseInstance::Held`]; it is only the
/// SIGTERM that is withheld. That costs macOS nothing, since
/// [`goose_desktop_instance`] never reads a lock there.
#[cfg(not(windows))]
fn process_is(pid: i32, bin: &Path) -> bool {
    if pid <= 0 {
        return false;
    }
    let Ok(exe) = std::fs::read_link(format!("/proc/{pid}/exe")) else {
        return false;
    };
    let canonical = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    canonical(&exe) == canonical(bin)
}

/// Whether a Goose Desktop already holds the single-instance lock. `bin`
/// is the executable this launch resolved. Windows has no lock file, so
/// the process is what is looked for — see
/// [`goose_desktop_running_windows`].
#[cfg(windows)]
fn goose_desktop_instance(bin: &Path, _home: &Path, _config: &Path) -> GooseInstance {
    goose_desktop_running_windows(bin)
}

/// Whether a Goose Desktop already holds the single-instance lock.
///
/// macOS never takes it — `app.asar` guards the whole branch with
/// `process.platform !== 'darwin'` — so this is always
/// [`GooseInstance::Free`] there, and a `SingletonLock` left behind by
/// something else cannot refuse a launch that would have worked.
#[cfg(not(windows))]
fn goose_desktop_instance(bin: &Path, home: &Path, config: &Path) -> GooseInstance {
    if cfg!(target_os = "macos") {
        return GooseInstance::Free;
    }
    single_instance_lock_holder(&goose_desktop_user_data(home, config), bin)
}

/// A running Goose Desktop on Windows, where Chromium keeps the lock in a
/// named mutex with nothing on disk to read, so the process is what there
/// is to find. `Get-CimInstance` the way [`crate::daemon`]'s own stop path
/// finds a process by name.
///
/// A probe that cannot run reports [`GooseInstance::Unknown`] rather than
/// guessing either way: a machine without `powershell` is not a machine
/// without Goose Desktop, and the caller warns instead of refusing.
#[cfg(windows)]
fn goose_desktop_running_windows(bin: &Path) -> GooseInstance {
    let out = Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='Goose.exe'\" | \
             ForEach-Object { \"$($_.ProcessId)|$($_.ExecutablePath)|$($_.CommandLine)\" }",
        ])
        .stderr(std::process::Stdio::null())
        .output()
        .ok()
        .filter(|o| o.status.success());
    match out {
        Some(out) => goose_exe_pid(
            &String::from_utf8_lossy(&out.stdout),
            bin,
            is_electron_bundle,
        ),
        None => GooseInstance::Unknown,
    }
}

/// Reads `Get-CimInstance`'s `<pid>|<path>|<command>` lines. Split from the
/// shell-out, and given `is_bundle` rather than calling
/// [`is_electron_bundle`] itself, so it can be tested off Windows where
/// neither would work.
///
/// [`GooseInstance::Confirmed`] needs the main process — matched on the
/// path, so an unrelated `Goose.exe` is not taken for the app this launch
/// resolved, and on the absence of `--type=`, which every process Electron
/// spawns off the main one carries. A renderer holds no lock, and
/// `taskkill` without `/F` would not close one anyway.
///
/// Another Electron `Goose.exe` is [`GooseInstance::Held`] — a build
/// installed elsewhere, or a child, which implies a main process holding
/// the lock even when that row is not listed. Windows exposes nothing
/// profile-scoped to read, so a running process is all there is to go on:
/// one sharing this profile takes the launch, one with a profile of its own
/// would not, and refusing on it is the cost of not being able to tell.
///
/// `is_bundle` is what keeps the `goose` CLI out of that: Windows reads its
/// `goose.exe` and the app's `Goose.exe` as one name, and `Name=` in WQL is
/// case-insensitive, so a CLI session would otherwise refuse every desktop
/// launch.
///
/// A blank `ExecutablePath` — a process this user cannot open — is
/// [`GooseInstance::Unknown`]: it cannot be told from the CLI either, and a
/// refusal on it would block the launch on nothing. An app whose directory
/// cannot be read also fails the bundle check, so it is missed rather than
/// refused — though such a process rarely lists an `ExecutablePath` at all,
/// and lands here instead.
#[cfg(any(windows, test))]
fn goose_exe_pid(listing: &str, bin: &Path, is_bundle: impl Fn(&Path) -> bool) -> GooseInstance {
    let mut held = false;
    let mut unreadable = false;
    for line in listing.lines() {
        let mut fields = line.trim().splitn(3, '|');
        let Some(pid) = fields.next() else { continue };
        let Some(path) = fields.next() else { continue };
        // The command line keeps any `|` of its own: it is last, and
        // `splitn` leaves the remainder whole.
        let command = fields.next().unwrap_or_default();
        if path.trim().is_empty() {
            unreadable = true;
            continue;
        }
        if !is_bundle(Path::new(path.trim())) {
            // The CLI under its other name, which holds no lock.
            continue;
        }
        held = true;
        if command.contains("--type=") || !same_windows_path(path, bin) {
            continue;
        }
        if let Ok(pid) = pid.trim().parse() {
            return GooseInstance::Confirmed(pid);
        }
    }
    // A holder outranks not knowing: one is evidence, the other is its absence.
    if held {
        GooseInstance::Held
    } else if unreadable {
        GooseInstance::Unknown
    } else {
        GooseInstance::Free
    }
}

/// Whether two Windows paths name one file, comparing case-insensitively
/// and treating `/` as `\`. An empty path never matches:
/// `ExecutablePath` is blank for a process this user cannot open.
#[cfg(any(windows, test))]
fn same_windows_path(a: &str, b: &Path) -> bool {
    let normalize = |s: &str| s.trim().replace('/', "\\").to_ascii_lowercase();
    let a = normalize(a);
    !a.is_empty() && a == normalize(&b.to_string_lossy())
}

/// Asks the app to quit, the way [`crate::daemon`]'s own stop path does:
/// `SIGTERM` rather than `SIGKILL`, `taskkill` without `/F`, so it closes
/// its session rather than losing it. Best-effort — whether it went is
/// [`wait_for_goose_desktop_to_quit`]'s to say.
fn quit_goose_desktop(pid: u32) {
    #[cfg(unix)]
    if let Ok(pid) = i32::try_from(pid) {
        // SAFETY: kill(2) with SIGTERM only asks that process to exit.
        unsafe { libc::kill(pid, libc::SIGTERM) };
    }
    #[cfg(windows)]
    {
        // Nulled like `daemon`'s: taskkill reports a missing pid loudly,
        // and the poll below is what decides either way.
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
    }
}

/// Polls until nothing holds the lock, for as long as
/// [`QUIT_TIMEOUT`]. Bounded by the clock rather than by a count of
/// tries, unlike `daemon`'s `wait_for_port_free`: on Windows each probe
/// shells out to `powershell`, so a fixed number of tries would wait for
/// however long that costs on top of the sleeps.
fn wait_for_goose_desktop_to_quit(bin: &Path, home: &Path, config: &Path) -> bool {
    let deadline = std::time::Instant::now() + QUIT_TIMEOUT;
    loop {
        // `Free` only: `Held` is still a holder, and `Unknown` is a probe
        // that stopped working, which is no evidence that it quit.
        if goose_desktop_instance(bin, home, config) == GooseInstance::Free {
            return true;
        }
        if std::time::Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

/// How long a quitting Goose Desktop is given to release the lock.
const QUIT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// What a launch is told when the running instance would take it. A
/// function rather than a literal so a test can assert on it without
/// going through the prompt, which would block on stdin.
fn goose_desktop_refusal(pid: u32) -> String {
    format!(
        "Goose Desktop is already running (pid {pid}), and holds the lock that \
         makes a second launch hand over to it — with the provider and model it \
         already had, not llmman's.\n\
         Quit it and run this again."
    )
}

/// The refusal for a holder that could not be confirmed to be the binary
/// this launch resolved, so there is no pid llmman will offer to signal.
///
/// Off Windows this is also how a lock left by a crash whose pid has since
/// been reused presents itself, and deleting the file clears it — so the
/// path is named. Windows holds the lock in a kernel mutex with no file to
/// remove, hence nothing to point at there.
fn goose_desktop_lock_held(user_data: &Path) -> String {
    let mut message = "Something is already holding Goose Desktop's single-instance lock, so \
         this launch would be handed to it and keep that instance's provider \
         and model, not llmman's.\n\
         Quit Goose Desktop and run this again."
        .to_string();
    if !cfg!(windows) {
        message.push_str(&format!(
            " If it is not running, the lock is stale and can be removed:\n  {}",
            user_data.join("SingletonLock").display()
        ));
    }
    message
}

/// What a launch is told when llmman could not determine whether an
/// instance is running. Printed, not refused: the launch may well be the
/// only instance, and refusing on no evidence would block a working setup.
fn goose_desktop_unknown_warning() -> String {
    "[llmman] could not determine whether Goose Desktop is already running. \
     If it is, this launch is handed to it and keeps that instance's provider \
     and model instead of the ones llmman resolved."
        .to_string()
}

/// What to do about a quit the user approved, given what the lock says by
/// the time they answered.
#[derive(Debug, PartialEq, Eq)]
enum QuitDecision {
    /// Still the same confirmed instance: signal it.
    Signal,
    /// It quit while the prompt waited, so the launch can proceed.
    AlreadyGone,
    /// Something else holds the lock now, or it can no longer be read.
    /// Signal nothing.
    Changed,
}

/// Only an unchanged, still-confirmed pid may be signalled. Split from
/// [`offer_to_quit_goose_desktop`], which blocks on a prompt, so that rule
/// can be tested.
fn quit_decision(probe: &GooseInstance, approved: u32) -> QuitDecision {
    match probe {
        GooseInstance::Confirmed(pid) if *pid == approved => QuitDecision::Signal,
        GooseInstance::Free => QuitDecision::AlreadyGone,
        // A different pid, an unconfirmable holder, or no answer at all.
        _ => QuitDecision::Changed,
    }
}

/// Refuses a launch the running instance would take, or quits that
/// instance when there is a terminal to answer for it.
///
/// Follows [`super::ensure_cline_installed`]: a pipe or a CI job is told, never
/// asked, so nothing here can hang waiting on a stdin that will not
/// answer.
fn offer_to_quit_goose_desktop(
    pid: u32,
    bin: &Path,
    home: &Path,
    config: &Path,
) -> anyhow::Result<()> {
    use std::io::{BufRead, IsTerminal, Write};

    let refusal = goose_desktop_refusal(pid);
    anyhow::ensure!(
        std::io::stdin().is_terminal() && std::io::stderr().is_terminal(),
        "{refusal}"
    );

    eprint!("Goose Desktop is already running (pid {pid}) and would take this launch with its own provider and model. Quit it? [y/N] ");
    std::io::stderr().flush()?;
    let mut answer = String::new();
    std::io::stdin().lock().read_line(&mut answer)?;
    anyhow::ensure!(accepts_prompt(&answer), "{refusal}");

    // Re-probed after the prompt, which waits as long as the user does: by
    // then `pid` may name something else entirely — the app quit and the
    // number was reused — and this is the call that would SIGTERM it.
    match quit_decision(&goose_desktop_instance(bin, home, config), pid) {
        QuitDecision::Signal => {
            eprintln!("[llmman] asking Goose Desktop to quit...");
            quit_goose_desktop(pid);
        }
        // It quit while the prompt waited, so the launch can go ahead.
        QuitDecision::AlreadyGone => {
            eprintln!("[llmman] Goose Desktop already quit; nothing to signal.");
            return Ok(());
        }
        QuitDecision::Changed => anyhow::bail!(
            "Goose Desktop (pid {pid}) is no longer the instance llmman \
             confirmed, so nothing was signalled. Run this again."
        ),
    }
    anyhow::ensure!(
        wait_for_goose_desktop_to_quit(bin, home, config),
        "Goose Desktop (pid {pid}) did not quit, so this launch would still go \
         to it. Quit it and run this again."
    );
    Ok(())
}

/// `goose-desktop` is the name this integration is specified against;
/// `goose-gui` is the symlink `BUILDING_LINUX.md` installs. Never a bare
/// `Goose`: on a case-insensitive filesystem it resolves to the `goose`
/// CLI — as it does on macOS — and launching the CLI in place of the app
/// is the mistake upstream's own desktop entry made (block/goose#4079).
pub(super) fn find_goose_desktop() -> Option<PathBuf> {
    find_on_path("goose-desktop")
        .or_else(|| find_on_path("goose-gui"))
        .or_else(goose_exe_on_path)
        .or_else(|| goose_desktop_fallback(&dirs::home_dir()?))
}

/// The Windows zip unpacks to `Goose.exe` and upstream ships no
/// installer, so a directory on `PATH` is the only way to reach it.
/// Windows reads that name and the CLI's `goose.exe` as one, so a match
/// counts only where the Electron bundle sits beside it.
fn goose_exe_on_path() -> Option<PathBuf> {
    if !cfg!(windows) {
        return None;
    }
    find_on_path_unless("Goose", |exe| !is_electron_bundle(exe))
}

/// Whether `exe` is the unpacked Electron app rather than a same-named
/// binary, in either layout the packager produces: `resources/app.asar`
/// beside the executable on Windows and Linux, `Contents/Resources` a
/// level up from `Contents/MacOS` in a macOS `.app`.
///
/// Beside the target, not the link: `BUILDING_LINUX.md` installs the app
/// by symlinking it into a `bin` directory, where nothing sits beside
/// the link. A path that will not resolve falls back to its literal
/// parent, which is where a bundle reached directly keeps these anyway.
///
/// Keyed on `app.asar`, which every packaging so far produces. A build
/// with asar off keeps `resources/app/` instead, and would read as the
/// CLI here — check for both if upstream ever ships one.
pub(super) fn is_electron_bundle(exe: &Path) -> bool {
    let resolved = std::fs::canonicalize(exe);
    let Some(dir) = resolved.as_deref().unwrap_or(exe).parent() else {
        return false;
    };
    if dir.join("resources").join("app.asar").is_file() {
        return true;
    }
    // Only where the layout really is a `.app`: otherwise any executable
    // two levels below a `Resources` counts, and a case-insensitive
    // filesystem matches the flat `resources` above as one.
    dir.file_name() == Some(std::ffi::OsStr::new("MacOS"))
        && dir
            .parent()
            .is_some_and(|contents| contents.join("Resources").join("app.asar").is_file())
}

/// Where Goose Desktop lands when it is not on `PATH`: a macOS `.app`
/// bundle never is, and `~/.local/bin` is not always.
///
/// The bundle is verified against 1.52.0, whose `CFBundleExecutable` is
/// `Goose`; the Linux path is the `.deb`'s. `is_file` keeps a wrong one
/// harmless, falling through to "not installed" rather than naming
/// something that fails to spawn.
///
/// Windows has no path to probe: upstream ships a zip with no
/// installer, so an install is wherever it was unpacked, and
/// [`goose_exe_on_path`] is what finds it there.
fn goose_desktop_fallback(home: &Path) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if cfg!(target_os = "macos") {
        for root in [PathBuf::from("/Applications"), home.join("Applications")] {
            candidates.push(
                root.join("Goose.app")
                    .join("Contents")
                    .join("MacOS")
                    .join("Goose"),
            );
        }
    }
    if !cfg!(windows) {
        // `goose-gui` as well: that is the name `BUILDING_LINUX.md`
        // symlinks, and a `PATH` without `~/.local/bin` never sees it.
        for name in ["goose-desktop", "goose-gui"] {
            candidates.push(home.join(".local").join("bin").join(name));
        }
    }
    if cfg!(target_os = "linux") {
        // The packaged GUI binary, capital G beside the lowercase CLI. Both
        // spellings, because 1.52.0's Electron Forge makers give the deb and
        // the rpm their own desktop templates, and those differ in the case
        // of the directory alone: `Exec=/usr/lib/goose/Goose` in
        // `forge.deb.desktop`, `Exec=/usr/lib/Goose/Goose` in
        // `forge.rpm.desktop`. Linux tells the two apart; neither is on
        // `PATH`.
        //
        // Not the manual `/opt/goose` unpack: its executable is spelled
        // `goose`, so a CLI left there would be launched in place of the
        // app, and that install's own `goose-gui` symlink is on `PATH`.
        candidates.push(PathBuf::from("/usr/lib/goose/Goose"));
        candidates.push(PathBuf::from("/usr/lib/Goose/Goose"));
    }
    candidates.into_iter().find(|p| p.is_file())
}

#[cfg(test)]
mod tests {
    use super::super::goose::goose_fallback;
    use super::*;

    /// The environment the desktop target hands over. Both launchers call
    /// `goose_env`, so this pins that its result is still right for the
    /// desktop app: a `/v1` on the host or a missing key breaks it there
    /// exactly as it breaks the CLI.
    #[test]
    fn goose_desktop_shares_the_cli_environment() {
        let host = "http://127.0.0.1:17434";
        let env = goose_env("m", "k", host);
        let get = |k| env.iter().find(|(n, _)| *n == k).map(|(_, v)| *v);
        assert_eq!(get("GOOSE_PROVIDER"), Some("openai"));
        assert_eq!(get("GOOSE_MODEL"), Some("m"));
        assert_eq!(get("OPENAI_API_KEY"), Some("k"));
        // The bare origin: goose joins OPENAI_BASE_PATH onto it itself,
        // so a `/v1` here would request /v1/v1/chat/completions.
        assert_eq!(get("OPENAI_HOST"), Some(host));
        assert_eq!(get("OPENAI_BASE_PATH"), Some("v1/chat/completions"));
    }

    /// `Goose.exe` and the CLI's `goose.exe` are one name to Windows, so
    /// the bundle beside it is what tells them apart. Get this wrong and
    /// `launch goose-desktop` starts the CLI — block/goose#4079 again.
    #[test]
    fn only_an_electron_bundle_counts_as_the_desktop_app() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-goose-bundle-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let exe = dir.join("Goose.exe");
        std::fs::write(&exe, "").unwrap();
        // A bare executable is the CLI as far as this can tell.
        assert!(!is_electron_bundle(&exe));
        let resources = dir.join("resources");
        std::fs::create_dir(&resources).unwrap();
        // The directory alone is not the bundle either.
        assert!(!is_electron_bundle(&exe));
        std::fs::write(resources.join("app.asar"), "").unwrap();
        assert!(is_electron_bundle(&exe));
        // Through a symlink too: the documented Linux install is a link
        // into a bin directory, and nothing sits beside the link. Unix
        // only — `std::os::unix` is what makes one without a privilege
        // Windows asks for.
        #[cfg(unix)]
        {
            let link_dir = dir.join("bin");
            std::fs::create_dir(&link_dir).unwrap();
            let link = link_dir.join("goose");
            std::os::unix::fs::symlink(&exe, &link).unwrap();
            assert!(is_electron_bundle(&link));

            // A link to an ordinary file is still not the app.
            let target = link_dir.join("target");
            let plain = link_dir.join("plain");
            std::fs::write(&target, "").unwrap();
            std::os::unix::fs::symlink(&target, &plain).unwrap();
            assert!(!is_electron_bundle(&plain));
        }

        // The macOS `.app`, whose `app.asar` is a level up from the
        // executable rather than beside it.
        let macos = dir.join("Goose.app").join("Contents").join("MacOS");
        std::fs::create_dir_all(&macos).unwrap();
        let app = macos.join("Goose");
        std::fs::write(&app, "").unwrap();
        assert!(!is_electron_bundle(&app));
        let res = dir.join("Goose.app").join("Contents").join("Resources");
        std::fs::create_dir(&res).unwrap();
        std::fs::write(res.join("app.asar"), "").unwrap();
        assert!(is_electron_bundle(&app));

        // `goose_fallback` refuses it too, or the installer's own
        // directory becomes a way past the guard.
        let fallback_home = dir.join("home");
        let unpacked = if cfg!(windows) {
            fallback_home.join("goose")
        } else {
            fallback_home.join(".local").join("bin")
        };
        std::fs::create_dir_all(unpacked.join("resources")).unwrap();
        let bin = unpacked.join(if cfg!(windows) { "goose.exe" } else { "goose" });
        std::fs::write(&bin, "").unwrap();
        assert_eq!(goose_fallback(&fallback_home), Some(bin));
        std::fs::write(unpacked.join("resources").join("app.asar"), "").unwrap();
        assert_eq!(goose_fallback(&fallback_home), None);

        // Never consulted off Windows, where the CLI owns the name.
        if !cfg!(windows) {
            assert_eq!(goose_exe_on_path(), None);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The refusal has to name the pid and say what to do about it —
    /// without that, all the user sees is a launch that did not happen.
    #[test]
    fn the_refusal_names_the_instance_and_what_to_do() {
        let refusal = goose_desktop_refusal(4242);
        assert!(refusal.contains("4242"), "{refusal}");
        assert!(refusal.contains("Quit it and run this again"), "{refusal}");
        // Says whose provider wins, which is the part a user acts on.
        assert!(refusal.contains("not llmman's"), "{refusal}");
    }

    /// The pid reused while the prompt waited is the one this must never
    /// SIGTERM, so every outcome but an unchanged match refuses to signal.
    #[test]
    fn only_the_same_confirmed_instance_is_signalled_after_the_prompt() {
        use GooseInstance::{Confirmed, Free, Held, Unknown};
        assert_eq!(quit_decision(&Confirmed(4242), 4242), QuitDecision::Signal);
        // Reused while the prompt waited: a stranger wearing that number.
        assert_eq!(quit_decision(&Confirmed(99), 4242), QuitDecision::Changed);
        // Quit while the prompt waited, so there is nothing to signal.
        assert_eq!(quit_decision(&Free, 4242), QuitDecision::AlreadyGone);
        // A holder llmman cannot identify is not one it may signal, and a
        // probe that stopped answering is no licence either.
        assert_eq!(quit_decision(&Held, 4242), QuitDecision::Changed);
        assert_eq!(quit_decision(&Unknown, 4242), QuitDecision::Changed);
    }

    /// An unconfirmable holder is refused without a pid to offer, so off
    /// Windows the message names the lock file instead — the only way out
    /// when the lock is stale and its pid has been reused. Windows has no
    /// such file, and must not be told to delete one.
    #[test]
    fn the_held_refusal_names_the_lock_to_remove() {
        let user_data = Path::new("/home/me/.config/Goose");
        let held = goose_desktop_lock_held(user_data);
        assert!(held.contains("not llmman's"), "{held}");
        assert!(held.contains("Quit Goose Desktop"), "{held}");
        // No pid is claimed: none was confirmed.
        assert!(!held.contains("pid"), "{held}");
        if cfg!(windows) {
            assert!(!held.contains("SingletonLock"), "{held}");
            assert!(!held.contains("stale"), "{held}");
        } else {
            assert!(
                held.contains("/home/me/.config/Goose/SingletonLock"),
                "{held}"
            );
        }
    }

    /// Not knowing is said out loud rather than refused — a launch that
    /// would have worked must not be blocked on absent evidence.
    #[test]
    fn the_unknown_warning_says_what_it_could_not_determine() {
        let warning = goose_desktop_unknown_warning();
        assert!(warning.starts_with("[llmman]"), "{warning}");
        assert!(warning.contains("could not determine"), "{warning}");
        assert!(warning.contains("provider"), "{warning}");
    }

    /// macOS takes no single-instance lock, so a `SingletonLock` sitting
    /// in its profile must not refuse a launch that would have worked
    /// there. Everywhere else the same file is the lock, and does.
    #[test]
    #[cfg(unix)]
    fn goose_desktop_instance_reads_the_lock_only_where_it_is_taken() {
        let home = std::env::temp_dir().join(format!(
            "llmman-goose-instance-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let config = home.join(".config");
        let user_data = goose_desktop_user_data(&home, &config);
        std::fs::create_dir_all(&user_data).unwrap();
        // A path nothing resolves to: only the lock is under test, so a
        // holder can never be `Confirmed` here.
        let bin = Path::new("/nonexistent/Goose");
        assert_eq!(
            goose_desktop_instance(bin, &home, &config),
            GooseInstance::Free
        );

        let me = std::process::id();
        std::os::unix::fs::symlink(format!("host-{me}"), user_data.join("SingletonLock")).unwrap();
        let found = goose_desktop_instance(bin, &home, &config);
        if cfg!(target_os = "macos") {
            assert_eq!(found, GooseInstance::Free, "macOS never takes the lock");
        } else {
            // This test's own pid is alive, so the lock has a holder —
            // `Held`, not `Confirmed`, because it is not running `bin`.
            assert_eq!(found, GooseInstance::Held);
        }
        let _ = std::fs::remove_dir_all(&home);
    }

    /// The `<pid>|<path>|<command>` lines `Get-CimInstance` prints. Only
    /// the main process is `Confirmed` — matched on the path, and on the
    /// absence of `--type=`, so an Electron renderer is not signalled as
    /// the lock holder. Another Electron `Goose.exe` is still `Held`: it
    /// takes the launch whatever its path.
    ///
    /// Every path here is an Electron bundle unless a case says otherwise;
    /// the real probe asks the filesystem, which these paths are not on.
    #[test]
    fn goose_exe_pid_finds_the_main_process_only() {
        use GooseInstance::{Confirmed, Free, Held, Unknown};
        let bin = Path::new(r"C:\Users\me\Goose\Goose.exe");
        let exe = r"C:\Users\me\Goose\Goose.exe";
        let bundle = |_: &Path| true;
        // As Windows really lists it: the children come first, and they
        // share their parent's executable.
        let listing = format!(
            "1001|{exe}|\"{exe}\" --type=gpu-process --field-trial-handle=1\r\n\
             1002|{exe}|\"{exe}\" --type=renderer --lang=en-GB\r\n\
             1003|{exe}|\"{exe}\"\r\n"
        );
        assert_eq!(goose_exe_pid(&listing, bin, bundle), Confirmed(1003));

        // A child with no main process listed is not ours to signal, but
        // the app it belongs to still holds the lock.
        assert_eq!(
            goose_exe_pid(
                &format!("1002|{exe}|\"{exe}\" --type=renderer"),
                bin,
                bundle
            ),
            Held
        );
        // Neither case nor separator tells two Windows paths apart.
        assert_eq!(
            goose_exe_pid("9|c:/users/me/goose/GOOSE.EXE|x", bin, bundle),
            Confirmed(9)
        );
        // A command line of its own may hold `|`; it is last and stays whole.
        assert_eq!(
            goose_exe_pid(&format!("9|{exe}|\"{exe}\" --logfile a|b"), bin, bundle),
            Confirmed(9)
        );
        // A blank ExecutablePath cannot be told from the CLI, so it is not
        // grounds to refuse — only to say so.
        assert_eq!(goose_exe_pid("9||x", bin, bundle), Unknown);
        assert_eq!(goose_exe_pid("9|   |x", bin, bundle), Unknown);
        // A build installed elsewhere shares the profile and the mutex.
        assert_eq!(
            goose_exe_pid("9|C:\\elsewhere\\Goose.exe|x", bin, bundle),
            Held
        );
        // Only an empty listing means nothing is running.
        assert_eq!(goose_exe_pid("", bin, bundle), Free);
        // One field is not a row `Get-CimInstance` can produce.
        assert_eq!(goose_exe_pid("nonsense", bin, bundle), Free);

        // The `goose` CLI, which this listing picks up under the app's name
        // and which holds no lock.
        let cli = r"C:\Users\me\.local\bin\goose.exe";
        let not_a_bundle = |p: &Path| p != Path::new(cli);
        assert_eq!(
            goose_exe_pid(&format!("2001|{cli}|\"{cli}\" session"), bin, not_a_bundle),
            Free
        );
        // Nor does it mask a real instance listed beside it.
        assert_eq!(
            goose_exe_pid(
                &format!("2001|{cli}|\"{cli}\" session\r\n1003|{exe}|\"{exe}\"\r\n"),
                bin,
                not_a_bundle
            ),
            Confirmed(1003)
        );
    }

    /// A lock counts only while the pid in it is *this app*. Liveness
    /// alone would let a stale lock whose number has been reused name an
    /// unrelated process — which the prompt would call Goose Desktop and
    /// `quit_goose_desktop` would then SIGTERM.
    #[test]
    #[cfg(not(windows))]
    fn a_single_instance_lock_counts_only_for_the_app_that_holds_it() {
        let user_data = std::env::temp_dir().join(format!(
            "llmman-goose-lock-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&user_data).unwrap();
        let lock = user_data.join("SingletonLock");
        // This test is a live process running a real executable, so it
        // stands in for the app; anything else stands in for a pid that
        // was reused after the app died.
        let me = std::process::id() as i32;
        let this_exe = std::env::current_exe().unwrap();
        let elsewhere = Path::new("/nonexistent/Goose");

        // No lock at all: nothing to hand a launch to.
        assert_eq!(
            single_instance_lock_holder(&user_data, &this_exe),
            GooseInstance::Free
        );

        std::os::unix::fs::symlink(format!("somehost-{me}"), &lock).unwrap();
        // A live pid running something else still holds the lock, so the
        // launch is refused — but as `Held`, which is never signalled.
        // This is the reused-pid case that used to SIGTERM a stranger.
        assert_eq!(
            single_instance_lock_holder(&user_data, elsewhere),
            GooseInstance::Held
        );
        // `/proc/<pid>/exe` is what confirms the binary, and only Linux
        // has it. Elsewhere the holder is real but unconfirmable, which
        // costs macOS nothing since it never takes the lock.
        let confirmed = single_instance_lock_holder(&user_data, &this_exe);
        if cfg!(target_os = "linux") {
            assert_eq!(confirmed, GooseInstance::Confirmed(me as u32));
        } else {
            assert_eq!(confirmed, GooseInstance::Held);
        }

        // A pid nothing answers to is a lock Chromium would break itself,
        // so it must not stand in the way of a launch. Malformed targets
        // and a plain file name no holder at all.
        for target in ["somehost-2147483646", "somehost-notapid", "nodash"] {
            std::fs::remove_file(&lock).unwrap();
            std::os::unix::fs::symlink(target, &lock).unwrap();
            assert_eq!(
                single_instance_lock_holder(&user_data, &this_exe),
                GooseInstance::Free,
                "{target}"
            );
        }
        std::fs::remove_file(&lock).unwrap();
        std::fs::write(&lock, "").unwrap();
        assert_eq!(
            single_instance_lock_holder(&user_data, &this_exe),
            GooseInstance::Free
        );

        let _ = std::fs::remove_dir_all(&user_data);
    }

    /// A profile llmman cannot read is not a profile with nothing in it.
    /// Reporting `Free` there would launch into a silent handover, which
    /// is the whole failure this guard exists to prevent, so it reports
    /// `Unknown` and the caller warns instead.
    #[test]
    #[cfg(unix)]
    fn an_unreadable_profile_is_unknown_rather_than_free() {
        use std::os::unix::fs::PermissionsExt;
        let user_data = std::env::temp_dir().join(format!(
            "llmman-goose-perm-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&user_data).unwrap();
        let me = std::process::id();
        std::os::unix::fs::symlink(format!("somehost-{me}"), user_data.join("SingletonLock"))
            .unwrap();
        std::fs::set_permissions(&user_data, std::fs::Permissions::from_mode(0o000)).unwrap();

        let bin = std::env::current_exe().unwrap();
        let found = single_instance_lock_holder(&user_data, &bin);
        // Restored before asserting: a failing assertion would otherwise
        // leave a directory nothing can delete.
        std::fs::set_permissions(&user_data, std::fs::Permissions::from_mode(0o755)).unwrap();
        let _ = std::fs::remove_dir_all(&user_data);
        if unsafe { libc::geteuid() } == 0 {
            // root reads it regardless of the mode, so a holder is found —
            // `Confirmed` only where `/proc` can back that up.
            assert!(
                matches!(found, GooseInstance::Confirmed(p) if p == me)
                    || found == GooseInstance::Held,
                "{found:?}"
            );
        } else {
            assert_eq!(found, GooseInstance::Unknown);
        }
    }

    /// `kill(pid, 0)` asks without signalling. This process is alive; pid
    /// 0 and negatives address process *groups*, which must never be
    /// mistaken for a holder, and a freshly reaped child is gone.
    #[test]
    #[cfg(not(windows))]
    fn pid_alive_sees_this_process_and_not_a_group_or_a_corpse() {
        assert!(pid_alive(std::process::id() as i32));
        assert!(!pid_alive(0));
        assert!(!pid_alive(-1));
        assert!(!pid_alive(-(std::process::id() as i32)));

        let mut child = Command::new("true").spawn().expect("spawn true");
        let pid = child.id() as i32;
        child.wait().expect("wait");
        // Reaped, so the pid is free rather than a zombie still answering.
        assert!(!pid_alive(pid));
    }

    /// A GUI install is the one never on `PATH`, so the fallback is what
    /// finds it: the real app when this machine has one, a fake home
    /// otherwise, since a fake home cannot outrank an install the
    /// machine really has at an absolute path.
    #[test]
    fn goose_desktop_fallback_finds_the_installers_target() {
        // A real install is the better test: assert the fallback finds
        // it rather than skip. Every machine-absolute candidate counts,
        // not just the macOS one.
        let installed = [
            "/Applications/Goose.app/Contents/MacOS/Goose",
            "/usr/lib/goose/Goose",
            "/usr/lib/Goose/Goose",
        ]
        .into_iter()
        .map(Path::new)
        .find(|p| p.is_file());
        if let Some(installed) = installed {
            assert_eq!(
                goose_desktop_fallback(Path::new("/nonexistent")),
                Some(installed.to_path_buf())
            );
            return;
        }
        // Windows has no installer and so no candidate to exercise: an
        // unpacked zip is found on `PATH` or not at all.
        if cfg!(windows) {
            assert_eq!(goose_desktop_fallback(Path::new("/nonexistent")), None);
            return;
        }
        let (parts, name): (&[&str], &str) = if cfg!(target_os = "macos") {
            (&["Applications", "Goose.app", "Contents", "MacOS"], "Goose")
        } else {
            (&[".local", "bin"], "goose-desktop")
        };
        let home = std::env::temp_dir().join(format!(
            "llmman-goose-desktop-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let dir = parts.iter().fold(home.clone(), |p, part| p.join(part));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(goose_desktop_fallback(&home), None);
        let bin = dir.join(name);
        // A directory of that name is not the app: returning it would
        // report Goose Desktop as installed and then fail to spawn.
        std::fs::create_dir(&bin).unwrap();
        assert_eq!(goose_desktop_fallback(&home), None);
        std::fs::remove_dir(&bin).unwrap();
        std::fs::write(&bin, "").unwrap();
        assert_eq!(goose_desktop_fallback(&home), Some(bin));
        assert_eq!(goose_desktop_fallback(&home.join("nowhere")), None);

        let _ = std::fs::remove_dir_all(&home);
    }
}
