//! `llmman tk <command> [args...]`: runs a command and prints its output
//! compacted, so a coding agent spends fewer tokens on noise.
//!
//! The command keeps this process's stdin, cwd and environment, and `tk`
//! exits with its status (`128 + n` for signal `n`).
//!
//! - A command with a filter here (`git status`, `cargo test`, ...) gets it.
//! - Any other command gets only the generic clean-up in [`text`].
//! - Commands whose output is meant to be read exactly (JSON, porcelain,
//!   formats, NUL separators, `cat`, ...) or that are interactive run
//!   untouched, as does everything with `LLMMAN_TK=off`.
//!
//! Filters shorten and never hide: a failed command keeps its error text,
//! and an unrecognized shape comes back near-verbatim.

mod cargo;
mod docker;
mod git;
mod go;
pub mod hook;
mod js;
mod pytest;
mod search;
pub(crate) mod shell;
mod text;

use std::ffi::OsString;
use std::io::{IsTerminal, Read, Write};
use std::process::{Command, Stdio};
use std::sync::{mpsc, Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// What a filter is given.
pub(crate) struct Run<'a> {
    pub code: i32,
    /// Both streams as text, with escape sequences and redraws already removed.
    pub stdout: &'a str,
    pub stderr: &'a str,
}

/// What a filter returns: the text to print on each stream.
pub(crate) struct Compact {
    pub stdout: String,
    pub stderr: String,
}

/// The command families with a filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Family {
    GitStatus,
    GitDiff,
    GitLog,
    GitCommit,
    GitPush,
    GitPull,
    Cargo,
    CargoTest,
    Go,
    GoTest,
    Pytest,
    JsTest,
    Search,
    DockerPs,
}

impl Family {
    /// `None` when the filter does not recognize the output.
    fn apply(self, run: &Run<'_>) -> Option<Compact> {
        match self {
            Family::GitStatus => git::status(run),
            Family::GitDiff => git::diff(run),
            Family::GitLog => git::log(run),
            Family::GitCommit => git::commit(run),
            Family::GitPush => git::push(run),
            Family::GitPull => git::pull(run),
            Family::Cargo => cargo::build(run),
            Family::CargoTest => cargo::test(run),
            Family::Go => go::build(run),
            Family::GoTest => go::test(run),
            Family::Pytest => pytest::filter(run),
            Family::JsTest => js::test(run),
            Family::Search => search::filter(run),
            Family::DockerPs => docker::ps(run),
        }
    }
}

/// How a command line is run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Plan {
    /// Untouched, on this process's own stdio.
    Passthrough,
    /// Through a filter, then the generic clean-up.
    Filter(Family),
    /// Through the generic clean-up only.
    Generic,
}

/// Programs whose output is the point (file contents, data, an
/// interactive session), so a trimmed copy would be wrong.
const VERBATIM_PROGRAMS: &[&str] = &[
    "cat", "head", "tail", "less", "more", "bat", "sed", "awk", "jq", "yq", "echo", "printf",
    "xxd", "od", "hexdump", "strings", "base64", "tee", "dd", "sort", "cut", "tr", "curl", "wget",
    "ssh", "scp", "vi", "vim", "nvim", "nano", "emacs", "man", "top", "htop", "watch", "sudo",
    "doas", "env", "xargs", "exec", "time", "nohup", "script", "tmux", "screen",
];

/// Flags (long, as a word or `=`-joined) that ask for output a program
/// reads, a pager follows, or that never ends.
const VERBATIM_FLAGS: &[&str] = &[
    "--json",
    "--format",
    "--message-format",
    "--output-format",
    "--porcelain",
    "--template",
    "--null",
    "--null-data",
    "--help",
    "--version",
    "--watch",
    "--watchAll",
    "--follow",
];

/// The program's own name: no directory, no `.exe`.
fn program_name(arg: &str) -> &str {
    let base = arg.rsplit(['/', '\\']).next().unwrap_or(arg);
    base.strip_suffix(".exe").unwrap_or(base)
}

/// `args` up to a `--`: past it, they are operands.
pub(crate) fn before_dashes(args: &[String]) -> &[String] {
    let end = args.iter().position(|a| a == "--").unwrap_or(args.len());
    &args[..end]
}

/// Whether `args` spells one of `flags` as a word or `flag=value`.
pub(crate) fn has_flag(args: &[String], flags: &[&str]) -> bool {
    args.iter().any(|a| {
        flags
            .iter()
            .any(|f| a == f || a.strip_prefix(f).is_some_and(|r| r.starts_with('=')))
    })
}

/// Whether `args` has a short-flag bundle (`-sb`) with any of `letters`.
pub(crate) fn has_short(args: &[String], letters: &str) -> bool {
    args.iter().any(|a| {
        a.starts_with('-') && !a.starts_with("--") && a.chars().skip(1).any(|c| letters.contains(c))
    })
}

/// `args` from the first word that is not an option, an option's value or a
/// `+toolchain`. `value_options` take the next word, unless spelled `--opt=value`.
pub(crate) fn skip_options<'a>(args: &'a [String], value_options: &[&str]) -> &'a [String] {
    let mut i = 0;
    while let Some(arg) = args.get(i) {
        if value_options.contains(&arg.as_str()) {
            i += 2;
        } else if arg.starts_with(['-', '+']) {
            i += 1;
        } else {
            return &args[i..];
        }
    }
    &[]
}

/// A flag that makes the output something to parse or follow, for any
/// program: see [`VERBATIM_FLAGS`], and `-z` (NUL separators).
pub(crate) fn asks_for_exact_output(args: &[String]) -> bool {
    has_flag(args, VERBATIM_FLAGS) || has_short(args, "z") || args.iter().any(|a| a == "-V")
}

/// Decides how `args` (a program and its arguments) is run.
pub(crate) fn plan(args: &[String]) -> Plan {
    let Some(first) = args.first() else {
        return Plan::Passthrough;
    };
    let program = program_name(first);
    let rest = &args[1..];
    if VERBATIM_PROGRAMS.contains(&program) {
        return Plan::Passthrough;
    }
    let plan = match program {
        "git" => git::plan(rest),
        "cargo" => cargo::plan(rest),
        "go" => go::plan(rest),
        "pytest" | "py.test" => pytest::plan(rest),
        "python" | "python3" => match rest {
            [m, module, tail @ ..] if m == "-m" && module == "pytest" => pytest::plan(tail),
            _ => Plan::Generic,
        },
        "npm" | "pnpm" | "yarn" => js::plan(rest),
        "grep" | "egrep" | "fgrep" | "rg" => search::plan(rest),
        "docker" => docker::plan(rest),
        _ => Plan::Generic,
    };
    // The same flags mean "leave it alone" for every command.
    if asks_for_exact_output(rest) {
        return Plan::Passthrough;
    }
    plan
}

/// Whether `llmman tk` would filter this command: the commands the hook
/// is willing to route through it.
pub(crate) fn is_filtered(args: &[String]) -> bool {
    matches!(plan(args), Plan::Filter(_))
}

/// The text `plan` makes of a finished command.
pub(crate) fn render(plan: Plan, code: i32, stdout: &str, stderr: &str) -> Compact {
    let stdout = text::clean(stdout);
    let stderr = text::clean(stderr);
    let run = Run {
        code,
        stdout: &stdout,
        stderr: &stderr,
    };
    let filtered = match plan {
        Plan::Filter(family) => family.apply(&run),
        Plan::Generic | Plan::Passthrough => None,
    };
    let mut out = match filtered {
        Some(c) => Compact {
            stdout: text::cap(&c.stdout),
            stderr: text::cap(&c.stderr),
        },
        None => Compact {
            stdout: text::squeeze(&stdout),
            stderr: text::squeeze(&stderr),
        },
    };
    // A filter that came back with nothing for a stream that had text, or
    // that dropped the error text of a command that failed, was wrong:
    // the stream is shown near-verbatim instead.
    if out.stdout.trim().is_empty() && !stdout.trim().is_empty() {
        out.stdout = text::squeeze(&stdout);
    }
    if out.stderr.trim().is_empty() && !stderr.trim().is_empty() {
        out.stderr = text::squeeze(&stderr);
    }
    out
}

/// Whether `LLMMAN_TK` (its value, if set) turns the compaction off.
fn is_off(value: Option<&str>) -> bool {
    value.is_some_and(|v| v.trim().eq_ignore_ascii_case("off"))
}

/// What is done with `words`: its plan, unless the compaction is off or a
/// person is typing at the command (with a terminal on stdin, prompts and
/// editors would be captured).
fn effective_plan(words: &[String], off: bool, interactive: bool) -> Plan {
    if off || interactive {
        Plan::Passthrough
    } else {
        plan(words)
    }
}

/// Runs `argv` and returns the exit code `llmman tk` ends with.
pub fn run(argv: &[OsString]) -> i32 {
    let words: Vec<String> = argv
        .iter()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    let plan = effective_plan(
        &words,
        is_off(std::env::var("LLMMAN_TK").ok().as_deref()),
        std::io::stdin().is_terminal(),
    );
    run_with(
        argv,
        plan,
        &mut std::io::stdout().lock(),
        &mut std::io::stderr().lock(),
    )
}

/// Runs `argv` as `plan` says, writing what a filtered command prints to
/// `stdout` and `stderr`, and returns the exit code.
fn run_with(argv: &[OsString], plan: Plan, stdout: &mut dyn Write, stderr: &mut dyn Write) -> i32 {
    let Some(program) = argv.first() else {
        return 2;
    };
    let mut cmd = Command::new(program);
    cmd.args(&argv[1..]);
    if plan == Plan::Passthrough {
        return passthrough(cmd, program);
    }
    cmd.stdin(Stdio::inherit());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(e) => return spawn_failed(program, &e),
    };
    // Both pipes are read at once, so the child never blocks on a full one.
    let out = child.stdout.take().map(Reader::spawn);
    let err = child.stderr.take().map(Reader::spawn);
    let status = child.wait();
    let deadline = Instant::now() + PIPE_GRACE;
    let collect = |r: Option<Reader>| r.map_or((String::new(), true), |r| r.finish(deadline));
    let ((raw_out, out_ended), (raw_err, err_ended)) = (collect(out), collect(err));
    let code = match status {
        Ok(status) => exit_code(status),
        Err(e) => {
            eprintln!("llmman tk: waiting for {}: {e}", program.to_string_lossy());
            1
        }
    };
    let shown = render(plan, code, &raw_out, &raw_err);
    write_all(stdout, &shown.stdout);
    write_all(stderr, &shown.stderr);
    if !(out_ended && err_ended) {
        write_all(
            stderr,
            "llmman tk: a process the command left running still holds its output open; what it writes later is not shown\n",
        );
    }
    code
}

fn write_all(to: &mut dyn Write, text: &str) {
    // A reader that went away is not an error worth a message.
    let _ = to.write_all(text.as_bytes());
    let _ = to.flush();
}

/// Runs `cmd` on this process's stdio. Where the platform can, the
/// process becomes the command, so signals and the exit status are its own.
fn passthrough(mut cmd: Command, program: &OsString) -> i32 {
    // Not in a test build: exec would replace the test runner.
    #[cfg(all(unix, not(test)))]
    {
        use std::os::unix::process::CommandExt;
        let e = cmd.exec();
        spawn_failed(program, &e)
    }
    #[cfg(not(all(unix, not(test))))]
    {
        match cmd.status() {
            Ok(status) => exit_code(status),
            Err(e) => spawn_failed(program, &e),
        }
    }
}

/// The shell's codes: 127 for a missing program, 126 for one that cannot run.
fn spawn_failed(program: &OsString, e: &std::io::Error) -> i32 {
    let name = program.to_string_lossy();
    if e.kind() == std::io::ErrorKind::NotFound {
        eprintln!("llmman tk: {name}: command not found");
        127
    } else {
        eprintln!("llmman tk: {name}: {e}");
        126
    }
}

fn exit_code(status: std::process::ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        return code;
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return 128 + signal;
        }
    }
    1
}

/// The most of a stream kept from its start, and from its end.
const KEEP_HEAD: usize = 4 << 20;
const KEEP_TAIL: usize = 4 << 20;

/// A stream held in bounded memory: its first and last bytes, and how many
/// lay between.
#[derive(Default)]
struct Bounded {
    head: Vec<u8>,
    tail: std::collections::VecDeque<u8>,
    dropped: u64,
}

impl Bounded {
    fn push(&mut self, mut chunk: &[u8]) {
        if self.head.len() < KEEP_HEAD {
            let n = (KEEP_HEAD - self.head.len()).min(chunk.len());
            self.head.extend_from_slice(&chunk[..n]);
            chunk = &chunk[n..];
        }
        self.tail.extend(chunk);
        while self.tail.len() > KEEP_TAIL {
            self.tail.pop_front();
            self.dropped += 1;
        }
    }

    fn into_text(self) -> String {
        let mut bytes = self.head;
        if self.dropped > 0 {
            bytes.extend_from_slice(
                format!("\n... {} bytes omitted ...\n", self.dropped).as_bytes(),
            );
        }
        bytes.extend(self.tail);
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// How long, once the command has exited, to wait for its pipes to close.
/// A process it left running holds them open.
const PIPE_GRACE: Duration = Duration::from_secs(1);

/// A stream read on its own thread into bounded memory.
struct Reader {
    kept: Arc<Mutex<Bounded>>,
    ended: mpsc::Receiver<()>,
}

impl Reader {
    fn spawn(source: impl Read + Send + 'static) -> Self {
        let kept = Arc::new(Mutex::new(Bounded::default()));
        let (done, ended) = mpsc::channel();
        let into = Arc::clone(&kept);
        std::thread::spawn(move || {
            capture(source, &into);
            let _ = done.send(());
        });
        Reader { kept, ended }
    }

    /// What was read, once the stream ends or `deadline` passes, and whether
    /// it ended.
    fn finish(self, deadline: Instant) -> (String, bool) {
        let ended = self
            .ended
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .is_ok();
        let mut kept = self.kept.lock().unwrap_or_else(PoisonError::into_inner);
        (std::mem::take(&mut *kept).into_text(), ended)
    }
}

/// Reads `source` to its end, so the child never blocks on a full pipe.
fn capture(mut source: impl Read, into: &Mutex<Bounded>) {
    let mut buf = [0u8; 64 * 1024];
    loop {
        match source.read(&mut buf) {
            Ok(0) => return,
            Ok(n) => into
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(&buf[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => return,
        }
    }
}

#[cfg(test)]
mod tests;
