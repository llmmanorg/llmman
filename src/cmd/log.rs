//! `llmman log` — `git log` for the prompts `llmman serve` has seen. Reads
//! `crate::promptlog`'s file directly (no daemon, as `git log` needs no
//! server), newest first, through a pager on a terminal.

use std::io::{self, IsTerminal, Write};
use std::process::{Command, Stdio};

use anyhow::{bail, Context};
use chrono::{DateTime, Local, Utc};
use clap::Args;

use crate::promptlog::Entry;

#[derive(Args, Debug)]
pub struct LogArgs {
    /// Limit the number of prompts to output
    #[arg(short = 'n', long = "max-count", value_name = "NUMBER")]
    pub max_count: Option<usize>,
    /// Skip NUMBER prompts before starting to show the output
    #[arg(long, value_name = "NUMBER", default_value_t = 0)]
    pub skip: usize,
    /// Show prompts more recent than a specific date (RFC 3339, YYYY-MM-DD,
    /// "yesterday", or "2 hours ago")
    #[arg(long, visible_alias = "after", value_name = "DATE")]
    pub since: Option<String>,
    /// Show prompts older than a specific date
    #[arg(long, visible_alias = "before", value_name = "DATE")]
    pub until: Option<String>,
    /// Limit to prompts sent to a model matching the pattern (regular
    /// expression). Repeat to match any of several
    #[arg(long, value_name = "PATTERN")]
    pub model: Vec<String>,
    /// Limit to prompts whose text matches the pattern (regular
    /// expression). Repeat to match any of several
    #[arg(long, value_name = "PATTERN")]
    pub grep: Vec<String>,
    /// Match the limiting patterns without regard to letter case
    #[arg(short = 'i', long = "regexp-ignore-case")]
    pub ignore_case: bool,
    /// One line per prompt: its abbreviated id and first line
    #[arg(long)]
    pub oneline: bool,
    /// Oldest first
    #[arg(long)]
    pub reverse: bool,
    /// Do not pipe the output into a pager
    #[arg(long)]
    pub no_pager: bool,
}

/// git's `log -3` shorthand for `-n 3`, which clap has no spelling for:
/// a bare `-<digits>` after `log` is rewritten before parsing.
pub fn expand_count_shorthand<I: IntoIterator<Item = std::ffi::OsString>>(
    args: I,
) -> Vec<std::ffi::OsString> {
    let mut args: Vec<std::ffi::OsString> = args.into_iter().collect();
    if args.get(1).is_some_and(|a| a == "log") {
        for arg in &mut args[2..] {
            if let Some(n) = arg.to_str().and_then(|a| a.strip_prefix('-')) {
                if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) {
                    *arg = format!("-n{n}").into();
                }
            }
        }
    }
    args
}

pub fn run(args: &LogArgs) -> anyhow::Result<()> {
    let path = crate::promptlog::path()?;
    let mut entries = crate::promptlog::read(&path)
        .with_context(|| format!("read prompt log {}", path.display()))?;

    let now = Utc::now();
    let since = args
        .since
        .as_deref()
        .map(|s| parse_date(s, now))
        .transpose()?;
    let until = args
        .until
        .as_deref()
        .map(|s| parse_date(s, now))
        .transpose()?;
    let model = patterns(&args.model, args.ignore_case)?;
    let grep = patterns(&args.grep, args.ignore_case)?;
    entries.retain(|e| {
        let time = DateTime::parse_from_rfc3339(&e.time).ok();
        since.is_none_or(|s| time.is_some_and(|t| t >= s))
            && until.is_none_or(|u| time.is_some_and(|t| t <= u))
            && model.as_ref().is_none_or(|m| m.is_match(&e.model))
            && grep.as_ref().is_none_or(|g| g.is_match(&e.prompt))
    });

    let color = io::stdout().is_terminal();
    let mut out = String::new();
    for entry in select(&entries, args.skip, args.max_count, args.reverse) {
        if args.oneline {
            oneline(entry, color, &mut out);
        } else {
            if !out.is_empty() {
                out.push('\n');
            }
            full(entry, color, &mut out);
        }
    }
    emit(&out, !args.no_pager)
}

/// As git: `--skip`/`-n` select newest-first, `--reverse` then flips
/// only what's shown (`log --reverse -3` is the newest three, oldest
/// first).
fn select(entries: &[Entry], skip: usize, max: Option<usize>, reverse: bool) -> Vec<&Entry> {
    let mut shown: Vec<&Entry> = entries
        .iter()
        .rev()
        .skip(skip)
        .take(max.unwrap_or(usize::MAX))
        .collect();
    if reverse {
        shown.reverse();
    }
    shown
}

/// Several patterns match when any does, as in git; none is no filter.
fn patterns(patterns: &[String], ignore_case: bool) -> anyhow::Result<Option<regex::RegexSet>> {
    if patterns.is_empty() {
        return Ok(None);
    }
    regex::RegexSetBuilder::new(patterns)
        .case_insensitive(ignore_case)
        .build()
        .map(Some)
        .context("invalid pattern")
}

/// RFC 3339; local `YYYY-MM-DD` (midnight) or `YYYY-MM-DD HH:MM[:SS]`;
/// `yesterday`; git's `N <unit>[s] ago` (also `N.<unit>.ago`).
fn parse_date(s: &str, now: DateTime<Utc>) -> anyhow::Result<DateTime<Utc>> {
    let s = s.trim();
    if let Ok(t) = DateTime::parse_from_rfc3339(s) {
        return Ok(t.with_timezone(&Utc));
    }
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"] {
        if let Ok(t) = chrono::NaiveDateTime::parse_from_str(s, fmt) {
            return local_to_utc(t);
        }
    }
    if let Ok(d) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return local_to_utc(d.and_hms_opt(0, 0, 0).unwrap());
    }
    let words: Vec<&str> = s.split(['.', ' ']).filter(|w| !w.is_empty()).collect();
    let (n, unit): (i64, &str) = match words[..] {
        ["yesterday"] => (1, "day"),
        [n, unit, "ago"] => (n.parse().ok().unwrap_or(-1), unit.trim_end_matches('s')),
        _ => bail!("invalid date: {s:?}"),
    };
    let seconds: i64 = match unit {
        "second" | "sec" => 1,
        "minute" | "min" => 60,
        "hour" => 3600,
        "day" => 86400,
        "week" => 86400 * 7,
        "month" => 86400 * 30,
        "year" => 86400 * 365,
        _ => bail!("invalid date: {s:?}"),
    };
    n.checked_mul(seconds)
        .filter(|_| n >= 0)
        .and_then(chrono::Duration::try_seconds)
        .and_then(|d| now.checked_sub_signed(d))
        .with_context(|| format!("invalid date: {s:?}"))
}

fn local_to_utc(t: chrono::NaiveDateTime) -> anyhow::Result<DateTime<Utc>> {
    t.and_local_timezone(Local)
        .single()
        .map(|t| t.with_timezone(&Utc))
        .ok_or_else(|| anyhow::anyhow!("not a valid local time: {t}"))
}

const YELLOW: &str = "\x1b[33m";
const RESET: &str = "\x1b[0m";

fn paint(text: &str, color: bool) -> String {
    if color {
        format!("{YELLOW}{text}{RESET}")
    } else {
        text.to_string()
    }
}

/// Request-supplied text with control characters (tab aside) dropped:
/// no driving the terminal — `less -R` passes escapes through — and no
/// forged header lines from a newline in a model name.
fn clean(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_control() || *c == '\t')
        .collect()
}

/// `git log`'s layout: header, blank line, message indented four.
fn full(e: &Entry, color: bool, out: &mut String) {
    out.push_str(&paint(&format!("prompt {}", e.id), color));
    out.push('\n');
    out.push_str(&format!("Model:  {}\n", clean(&e.model)));
    if let Some(client) = &e.client {
        out.push_str(&format!("Client: {}\n", clean(client)));
    }
    out.push_str(&format!("Route:  {}\n", e.route));
    out.push_str(&format!("Date:   {}\n\n", git_date(&e.time)));
    for line in e.prompt.lines() {
        out.push_str("    ");
        out.push_str(&clean(line));
        out.push('\n');
    }
}

fn oneline(e: &Entry, color: bool, out: &mut String) {
    out.push_str(&paint(&crate::fmt::short_id(&e.id), color));
    out.push(' ');
    out.push_str(&clean(e.prompt.lines().next().unwrap_or_default()));
    out.push('\n');
}

/// git's date format, local time: `Sun Sep 6 12:34:56 2026 +0100`.
fn git_date(rfc3339: &str) -> String {
    match DateTime::parse_from_rfc3339(rfc3339) {
        Ok(t) => t
            .with_timezone(&Local)
            .format("%a %b %-d %H:%M:%S %Y %z")
            .to_string(),
        Err(_) => rfc3339.to_string(),
    }
}

/// Through the pager when stdout is a terminal, as git: `$LLMMAN_PAGER`,
/// else `$PAGER`, else `less`, via the shell; empty or `cat` means none.
/// `LESS=FRX` is git's default too.
fn emit(text: &str, pager: bool) -> anyhow::Result<()> {
    if pager && io::stdout().is_terminal() {
        if let Some(mut child) = pager_command().and_then(|cmd| spawn_pager(&cmd).ok()) {
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

fn pager_command() -> Option<String> {
    let cmd = ["LLMMAN_PAGER", "PAGER"]
        .iter()
        .find_map(|var| std::env::var(var).ok())
        .unwrap_or_else(|| "less".to_string());
    let cmd = cmd.trim().to_string();
    (!cmd.is_empty() && cmd != "cat").then_some(cmd)
}

fn spawn_pager(cmd: &str) -> io::Result<std::process::Child> {
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

    fn entry() -> Entry {
        Entry {
            id: "0123456789abcdef0123456789abcdef01234567".into(),
            time: "2026-09-06T11:34:56Z".into(),
            route: "/v1/chat/completions".into(),
            model: "qwen3:8b".into(),
            client: Some("claude-cli/1.2.3".into()),
            prompt: "fix the failing test\nthen commit".into(),
        }
    }

    #[test]
    fn full_format_mirrors_git_log() {
        let mut out = String::new();
        full(&entry(), false, &mut out);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "prompt 0123456789abcdef0123456789abcdef01234567");
        assert_eq!(lines[1], "Model:  qwen3:8b");
        assert_eq!(lines[2], "Client: claude-cli/1.2.3");
        assert_eq!(lines[3], "Route:  /v1/chat/completions");
        assert!(lines[4].starts_with("Date:   "), "{}", lines[4]);
        assert_eq!(lines[5], "");
        assert_eq!(lines[6], "    fix the failing test");
        assert_eq!(lines[7], "    then commit");
        assert_eq!(lines.len(), 8);
    }

    #[test]
    fn the_client_line_is_left_out_when_unknown() {
        let mut out = String::new();
        full(
            &Entry {
                client: None,
                ..entry()
            },
            false,
            &mut out,
        );
        assert!(!out.contains("Client:"));
    }

    #[test]
    fn oneline_is_the_short_id_and_first_line() {
        let mut out = String::new();
        oneline(&entry(), false, &mut out);
        assert_eq!(out, "0123456789ab fix the failing test\n");
    }

    #[test]
    fn color_wraps_only_the_header_in_yellow() {
        let mut out = String::new();
        full(&entry(), true, &mut out);
        assert!(out.starts_with("\x1b[33mprompt 0123456789abcdef0123456789abcdef01234567\x1b[0m\n"));
        assert_eq!(out.matches("\x1b[").count(), 2);
    }

    #[test]
    fn request_supplied_text_cannot_drive_the_terminal() {
        let hostile = Entry {
            model: "m\x1b[2J\nDate:   forged".into(),
            client: Some("c\u{9b}0m\r".into()),
            prompt: "\x1b]0;title\x07line one\n\tindented\x08".into(),
            ..entry()
        };
        let mut out = String::new();
        full(&hostile, false, &mut out);
        assert!(!out.contains('\x1b') && !out.contains('\u{9b}') && !out.contains('\x07'));
        assert!(out.contains("Model:  m[2JDate:   forged\n"));
        assert!(out.contains("Client: c0m\n"));
        assert!(out.contains("    ]0;titleline one\n    \tindented\n"));
        let mut one = String::new();
        oneline(&hostile, false, &mut one);
        assert_eq!(one, "0123456789ab ]0;titleline one\n");
    }

    #[test]
    fn git_date_uses_gits_layout() {
        // Local zone, so check the shape rather than the hour.
        let d = git_date("2026-09-06T11:34:56Z");
        let parts: Vec<&str> = d.split(' ').collect();
        assert_eq!(parts.len(), 6, "{d}");
        assert_eq!(parts[1], "Sep");
        assert_eq!(parts[3].len(), 8);
        assert_eq!(parts[4], "2026");
        assert!(parts[5].starts_with('+') || parts[5].starts_with('-'));
        assert_eq!(git_date("garbage"), "garbage");
    }

    #[test]
    fn relative_dates_count_back_from_now() {
        let now = DateTime::parse_from_rfc3339("2026-09-06T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let at = |s: &str| parse_date(s, now).unwrap().to_rfc3339();
        assert_eq!(at("2 hours ago"), "2026-09-06T10:00:00+00:00");
        assert_eq!(at("1.week.ago"), "2026-08-30T12:00:00+00:00");
        assert_eq!(at("yesterday"), "2026-09-05T12:00:00+00:00");
        assert_eq!(at("30 minutes ago"), "2026-09-06T11:30:00+00:00");
        assert_eq!(at("2026-09-01T00:00:00Z"), "2026-09-01T00:00:00+00:00");
        for bad in [
            "fortnight",
            "3 fortnights ago",
            "-2 days ago",
            "99999999999999 years ago",
        ] {
            assert!(parse_date(bad, now).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_plain_date_is_local_midnight() {
        let now = Utc::now();
        let t = parse_date("2026-09-06", now).unwrap().with_timezone(&Local);
        assert_eq!(
            t.format("%Y-%m-%d %H:%M:%S").to_string(),
            "2026-09-06 00:00:00"
        );
    }

    #[test]
    fn reverse_flips_the_selection_not_the_whole_log() {
        let entries: Vec<Entry> = (1..=5)
            .map(|i| Entry {
                prompt: format!("p{i}"),
                ..entry()
            })
            .collect();
        fn prompts(shown: Vec<&Entry>) -> Vec<&str> {
            shown.iter().map(|e| e.prompt.as_str()).collect()
        }
        assert_eq!(
            prompts(select(&entries, 0, None, false)),
            ["p5", "p4", "p3", "p2", "p1"]
        );
        assert_eq!(
            prompts(select(&entries, 0, Some(3), false)),
            ["p5", "p4", "p3"]
        );
        assert_eq!(
            prompts(select(&entries, 0, Some(3), true)),
            ["p3", "p4", "p5"]
        );
        assert_eq!(prompts(select(&entries, 1, Some(2), true)), ["p3", "p4"]);
    }

    #[test]
    fn a_bare_dash_number_after_log_becomes_max_count() {
        let expand = |args: &[&str]| -> Vec<String> {
            expand_count_shorthand(args.iter().map(std::ffi::OsString::from))
                .into_iter()
                .map(|a| a.into_string().unwrap())
                .collect()
        };
        assert_eq!(
            expand(&["llmman", "log", "-3", "--oneline"]),
            ["llmman", "log", "-n3", "--oneline"]
        );
        assert_eq!(
            expand(&["llmman", "log", "-n", "3"]),
            ["llmman", "log", "-n", "3"]
        );
        assert_eq!(expand(&["llmman", "log", "-"]), ["llmman", "log", "-"]);
        assert_eq!(expand(&["llmman", "ps", "-3"]), ["llmman", "ps", "-3"]);
    }

    #[test]
    fn several_patterns_match_any_and_ignore_case_applies_to_all() {
        let re = patterns(&["foo".into(), "bar".into()], true)
            .unwrap()
            .unwrap();
        assert!(re.is_match("FOO"));
        assert!(re.is_match("a Bar"));
        assert!(!re.is_match("baz"));
        // Independent patterns: a capture name may repeat across them.
        assert!(patterns(&["(?P<x>a)".into(), "(?P<x>b)".into()], false).is_ok());
        assert!(patterns(&[], false).unwrap().is_none());
        assert!(patterns(&["(".into()], false).is_err());
    }
}
