use std::io::{self, IsTerminal, Write};
use std::process::{Command, Stdio};

/// Through the pager when stdout is a terminal, as git: `$LLMMAN_PAGER`,
/// then the command-specific pager, `core.pager`, `$PAGER`, or `less`.
/// Empty or `cat` means none; false boolean values in `pager.<command>`
/// disable paging even when `$LLMMAN_PAGER` is set.
/// `LESS=FRX` is git's default too.
///
/// A pager that is not installed is skipped for the next in that order;
/// with none left the text goes straight to the terminal.
pub(crate) fn emit(subcommand: &str, text: &str, pager: bool) -> anyhow::Result<()> {
    if pager && io::stdout().is_terminal() {
        for cmd in pager_commands(subcommand) {
            let Ok(mut child) = spawn_pager(&cmd) else {
                continue;
            };
            if let Some(mut stdin) = child.stdin.take() {
                // Quitting the pager early closes the pipe; not an error.
                let _ = stdin.write_all(text.as_bytes());
            }
            // 127 (sh) / 9009 (cmd): no such pager, nothing was shown.
            if !matches!(child.wait()?.code(), Some(127 | 9009)) {
                return Ok(());
            }
        }
    }
    match io::stdout().write_all(text.as_bytes()) {
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        other => Ok(other?),
    }
}

fn pager_commands(subcommand: &str) -> Vec<String> {
    let (core, specific) = crate::config::pager(subcommand);
    resolve(
        std::env::var("LLMMAN_PAGER").ok().as_deref(),
        core.as_deref(),
        specific.as_deref(),
        std::env::var("PAGER").ok().as_deref(),
    )
}

/// The pagers to try, in order. An empty command or `cat` ends the list:
/// it is a choice of no pager, not a missing one.
fn resolve(
    env: Option<&str>,
    core: Option<&str>,
    specific: Option<&str>,
    pager: Option<&str>,
) -> Vec<String> {
    let enabled = specific.and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
        "true" | "yes" | "on" | "1" => Some(true),
        "false" | "no" | "off" | "0" => Some(false),
        _ => None,
    });
    if enabled == Some(false) {
        return Vec::new();
    }
    let specific = specific.filter(|_| enabled.is_none());
    let mut commands: Vec<String> = Vec::new();
    for cmd in [env, specific, core, pager, Some("less")]
        .into_iter()
        .flatten()
    {
        let cmd = cmd.trim();
        if cmd.is_empty() || cmd == "cat" {
            break;
        }
        if !commands.iter().any(|seen| seen == cmd) {
            commands.push(cmd.to_string());
        }
    }
    commands
}

/// `cmd /C <missing>` exits 1, not 9009, and shows nothing, so on Windows a
/// pager is looked up first. `where` searches the current directory and
/// `PATH` with `PATHEXT`, as `cmd` does. Commands using `cmd` syntax, and a
/// `where` that cannot run, are left to `cmd`.
#[cfg(windows)]
fn program_exists(cmd: &str) -> bool {
    let cmd = cmd.trim();
    if cmd.contains(['&', '|', '<', '>', '^', '%', '(', ')']) {
        return true;
    }
    let program = match cmd.strip_prefix('"') {
        Some(rest) => rest.split('"').next(),
        None => cmd.split_whitespace().next(),
    };
    Command::new("where")
        .arg(program.unwrap_or(""))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_or(true, |status| status.success())
}

fn spawn_pager(cmd: &str) -> io::Result<std::process::Child> {
    #[cfg(windows)]
    if !program_exists(cmd) {
        return Err(io::ErrorKind::NotFound.into());
    }
    let mut command = if cfg!(windows) {
        let mut c = Command::new("cmd");
        c.args(["/C", cmd]);
        c
    } else {
        let mut c = Command::new("sh");
        c.args(["-c", cmd]);
        c
    };
    for (var, default) in [("LESS", "FRX"), ("LV", "-c")] {
        if std::env::var_os(var).is_none() {
            command.env(var, default);
        }
    }
    command.stdin(Stdio::piped()).spawn()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn first(
        env: Option<&str>,
        core: Option<&str>,
        specific: Option<&str>,
        pager: Option<&str>,
    ) -> Option<String> {
        resolve(env, core, specific, pager).into_iter().next()
    }

    #[test]
    fn pager_precedence_and_opt_out_match_git() {
        assert_eq!(first(None, None, None, None).as_deref(), Some("less"));
        assert_eq!(
            first(None, None, None, Some("more")).as_deref(),
            Some("more")
        );
        assert_eq!(
            first(None, Some("less -S"), None, Some("more")).as_deref(),
            Some("less -S")
        );
        assert_eq!(
            first(None, Some("less"), Some("more"), None).as_deref(),
            Some("more")
        );
        assert_eq!(
            first(Some("env-pager"), Some("less"), Some("more"), None).as_deref(),
            Some("env-pager")
        );
        assert_eq!(
            first(None, Some("less"), Some("true"), None).as_deref(),
            Some("less")
        );
        for disabled in ["", "cat", "  cat  "] {
            assert!(resolve(Some(disabled), Some("less"), None, None).is_empty());
        }
        for value in ["false", "no", "off", "0", "FALSE", " No ", "OFF"] {
            assert!(resolve(Some("less"), None, Some(value), None).is_empty());
        }
        for value in ["true", "yes", "on", "1", "TRUE", " Yes ", "ON"] {
            assert_eq!(
                first(None, Some("less -S"), Some(value), None).as_deref(),
                Some("less -S")
            );
        }
        assert!(resolve(None, Some("less"), Some(""), None).is_empty());
    }

    /// A missing pager falls through to the next one, in precedence order
    /// and without repeats, ending at `less`; a pager turned off stops it.
    #[test]
    fn resolve_lists_every_fallback_in_order() {
        assert_eq!(
            resolve(Some("a"), Some("b"), Some("c"), Some("d")),
            ["a", "c", "b", "d", "less"]
        );
        assert_eq!(resolve(None, None, None, Some("less")), ["less"]);
        assert_eq!(resolve(Some("a"), None, None, Some("a")), ["a", "less"]);
        assert_eq!(resolve(Some("a"), Some("cat"), None, Some("d")), ["a"]);
        assert_eq!(resolve(None, None, None, Some("")), Vec::<String>::new());
    }

    #[cfg(windows)]
    #[test]
    fn missing_pager_is_detected_on_windows() {
        assert!(program_exists("cmd /K"));
        assert!(program_exists("\"cmd\""));
        assert!(program_exists("missing-pager-xyz | more"));
        assert!(program_exists("(missing-pager-xyz)"));
        assert!(!program_exists("missing-pager-xyz -S"));
        assert!(!program_exists("\"missing-pager-xyz\""));
        assert!(!program_exists(""));
    }

    /// A missing pager is `sh`'s 127; builtins such as `exec` need no file.
    #[cfg(unix)]
    #[test]
    fn missing_pager_is_sh_status_127() {
        let status = |cmd: &str| {
            let mut child = spawn_pager(cmd).unwrap();
            drop(child.stdin.take());
            child.wait().unwrap().code()
        };
        assert_eq!(status("missing-pager-xyz -S"), Some(127));
        assert_eq!(status("exec cat"), Some(0));
    }
}
