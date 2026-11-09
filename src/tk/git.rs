//! Filters for `git status`, `diff`, `log`, `commit`, `push` and `pull`.
//!
//! Any other git subcommand (`rev-parse`, `show`, `ls-files`, `branch`, ...)
//! runs untouched: much of what the agent runs there is plumbing whose
//! output it parses, or the contents of a file.

use std::sync::OnceLock;

use super::text::{cached, cut_line, join, omitted};
use super::{before_dashes, has_short, Compact, Family, Plan, Run};

/// Most entries listed under one `git status` heading.
const STATUS_ENTRIES: usize = 50;
/// Most body lines of a file's diff.
const DIFF_FILE_LINES: usize = 250;
/// Longest diff line.
const DIFF_LINE_CHARS: usize = 300;
/// Most commits listed.
const LOG_COMMITS: usize = 100;
/// Most message lines shown under a commit, after its subject.
const LOG_BODY_LINES: usize = 3;

/// The subcommand, and what follows it, past git's own leading options.
fn subcommand(rest: &[String]) -> Option<(&str, &[String])> {
    let mut i = 0;
    while let Some(arg) = rest.get(i) {
        match arg.as_str() {
            "-C" | "-c" | "--git-dir" | "--work-tree" | "--namespace" | "--super-prefix"
            | "--config-env" => i += 2,
            "--no-pager"
            | "--paginate"
            | "-p"
            | "-P"
            | "--bare"
            | "--no-replace-objects"
            | "--literal-pathspecs"
            | "--glob-pathspecs"
            | "--noglob-pathspecs"
            | "--icase-pathspecs"
            | "--no-optional-locks" => i += 1,
            a if a.starts_with("--") && a.contains('=') => i += 1,
            a if a.starts_with('-') => return None,
            a => return Some((a, &rest[i + 1..])),
        }
    }
    None
}

/// Whether `args` has a flag that is, or starts with, one of `names`.
fn starts_with_any(args: &[String], names: &[&str]) -> bool {
    args.iter().any(|a| names.iter().any(|n| a.starts_with(n)))
}

pub(super) fn plan(rest: &[String]) -> Plan {
    let Some((sub, args)) = subcommand(rest) else {
        return Plan::Passthrough;
    };
    let args = before_dashes(args);
    match sub {
        "status" if has_short(args, "s") || starts_with_any(args, &["--short", "--porcelain"]) => {
            Plan::Passthrough
        }
        "status" => Plan::Filter(Family::GitStatus),
        // A summary or a name list is already compact; the rest is not a diff.
        "diff"
            if starts_with_any(
                args,
                &[
                    "--stat",
                    "--name-only",
                    "--name-status",
                    "--numstat",
                    "--shortstat",
                    "--summary",
                    "--dirstat",
                    "--compact-summary",
                    "--check",
                    "--raw",
                    "--output",
                    "--quiet",
                    "--no-patch",
                    "--no-index",
                    "--patch-with-",
                    "--word-diff",
                    "--binary",
                ],
            ) || has_short(args, "s") =>
        {
            Plan::Passthrough
        }
        "diff" => Plan::Filter(Family::GitDiff),
        "log" if log_flags_known(args) => Plan::Filter(Family::GitLog),
        "commit" if has_message(args) && !is_interactive_commit(args) => {
            Plan::Filter(Family::GitCommit)
        }
        "push" => Plan::Filter(Family::GitPush),
        "pull" if !is_interactive_pull(args) => Plan::Filter(Family::GitPull),
        _ => Plan::Passthrough,
    }
}

/// A commit that prints no summary, or asks something: `--dry-run`,
/// `--short`, `-e`/`--edit`, `-p`/`--patch`, `--interactive`.
fn is_interactive_commit(args: &[String]) -> bool {
    starts_with_any(
        args,
        &["--dry-run", "--short", "--edit", "--patch", "--interactive"],
    ) || has_short(args, "ep")
}

/// A pull that opens an editor: `-e`/`--edit`, `--rebase=interactive`
/// (`--rebase=i`, `-ri`).
fn is_interactive_pull(args: &[String]) -> bool {
    starts_with_any(args, &["--edit", "--rebase=i", "-ri"]) || has_short(args, "e")
}

/// `-m`, `-am`, `--message`, `-F`, `--no-edit`: a commit that opens no editor.
fn has_message(args: &[String]) -> bool {
    args.iter().any(|a| {
        a == "--message"
            || a.starts_with("--message=")
            || a == "--no-edit"
            || a == "--file"
            || a.starts_with("--file=")
            || (a.starts_with('-')
                && !a.starts_with("--")
                && a[1..].chars().all(|c| c.is_ascii_alphabetic())
                && (a.contains('m') || a.contains('F')))
    })
}

/// The `git log` options the filter reads the output of. Any other (a
/// format, a graph, a patch, a diffstat, ...) changes the shape, so the
/// log is left as it comes.
fn log_flags_known(args: &[String]) -> bool {
    const KNOWN: &[&str] = &[
        "-n",
        "--max-count",
        "--skip",
        "--author",
        "--committer",
        "--since",
        "--until",
        "--after",
        "--before",
        "--grep",
        "--invert-grep",
        "--all-match",
        "--regexp-ignore-case",
        "-i",
        "--no-merges",
        "--merges",
        "--first-parent",
        "--all",
        "--branches",
        "--tags",
        "--remotes",
        "--date",
        "--decorate",
        "--no-decorate",
    ];
    args.iter().filter(|a| a.starts_with('-')).all(|a| {
        // `-5` for `-n 5`.
        let is_count = a.len() > 1 && a[1..].bytes().all(|b| b.is_ascii_digit());
        is_count || KNOWN.iter().any(|k| a == k || a.starts_with(k))
    })
}

pub(super) fn status(run: &Run<'_>) -> Option<Compact> {
    let first = run.stdout.lines().next()?;
    if run.code != 0
        || !(first.starts_with("On branch ")
            || first.starts_with("HEAD detached")
            || first.contains("rebase in progress"))
    {
        return None;
    }
    let mut out: Vec<String> = Vec::new();
    let (mut entries, mut skipped) = (0usize, 0usize);
    let flush = |out: &mut Vec<String>, skipped: &mut usize| {
        if *skipped > 0 {
            out.push(format!("\t{}", omitted(*skipped, "more entries")));
            *skipped = 0;
        }
    };
    for line in run.stdout.lines() {
        if line.trim().is_empty() || line.trim_start().starts_with("(use \"git ") {
            continue;
        }
        if line.starts_with('\t') {
            entries += 1;
            if entries > STATUS_ENTRIES {
                skipped += 1;
                continue;
            }
        } else {
            flush(&mut out, &mut skipped);
            entries = 0;
        }
        out.push(line.to_string());
    }
    flush(&mut out, &mut skipped);
    Some(Compact {
        stdout: join(&out),
        stderr: run.stderr.to_string(),
    })
}

pub(super) fn diff(run: &Run<'_>) -> Option<Compact> {
    // 1 is what `--exit-code` returns for a diff that has changes.
    if run.code > 1 || !run.stdout.starts_with("diff --git ") {
        return None;
    }
    let mut out: Vec<String> = Vec::new();
    let mut in_header = false;
    let (mut body, mut path) = (0usize, String::new());
    let end_file = |out: &mut Vec<String>, body: usize, path: &str| {
        if body > DIFF_FILE_LINES {
            out.push(omitted(
                body - DIFF_FILE_LINES,
                &format!("diff lines in {path}"),
            ));
        }
    };
    for line in run.stdout.lines() {
        if let Some(names) = line.strip_prefix("diff --git ") {
            end_file(&mut out, body, &path);
            body = 0;
            path = names
                .rsplit_once(" b/")
                .map_or(names, |(_, b)| b)
                .to_string();
            in_header = true;
            out.push(line.to_string());
            continue;
        }
        if in_header {
            if line.starts_with("@@") || line.starts_with("Binary files") {
                in_header = false;
            } else {
                // The blob names say nothing the paths do not.
                if !line.starts_with("index ") {
                    out.push(line.to_string());
                }
                continue;
            }
        }
        body += 1;
        if body <= DIFF_FILE_LINES {
            out.push(cut_line(line, DIFF_LINE_CHARS));
        }
    }
    end_file(&mut out, body, &path);
    Some(Compact {
        stdout: join(&out),
        stderr: run.stderr.to_string(),
    })
}

/// One commit of the default `git log`.
struct Commit<'a> {
    hash: &'a str,
    decorations: Option<&'a str>,
    author: &'a str,
    date: String,
    merge: bool,
    message: Vec<&'a str>,
}

/// `Tue Nov 3 10:22:11 2026 +0100` as `Nov 3 2026`; anything else as it is.
fn short_date(date: &str) -> String {
    let words: Vec<&str> = date.split_whitespace().collect();
    match words.as_slice() {
        [_, month, day, time, year, _] if time.contains(':') => format!("{month} {day} {year}"),
        _ => date.trim().to_string(),
    }
}

/// The commits in `stdout`, or `None` for anything but the default format.
fn parse_log(stdout: &str) -> Option<Vec<Commit<'_>>> {
    let mut commits: Vec<Commit<'_>> = Vec::new();
    let mut in_message = false;
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("commit ") {
            let (hash, decorations) = match rest.split_once(' ') {
                Some((h, d)) => (h, Some(d)),
                None => (rest, None),
            };
            let hex = matches!(hash.len(), 40 | 64) && hash.bytes().all(|b| b.is_ascii_hexdigit());
            if !hex {
                return None;
            }
            commits.push(Commit {
                hash,
                decorations,
                author: "",
                date: String::new(),
                merge: false,
                message: Vec::new(),
            });
            in_message = false;
            continue;
        }
        let commit = commits.last_mut()?;
        if in_message {
            if let Some(text) = line.strip_prefix("    ") {
                commit.message.push(text);
            } else if line.is_empty() {
                commit.message.push("");
            } else {
                return None;
            }
        } else if line.is_empty() {
            in_message = true;
        } else if let Some(a) = line.strip_prefix("Author:") {
            commit.author = a.split('<').next().unwrap_or(a).trim();
        } else if let Some(d) = line.strip_prefix("Date:") {
            commit.date = short_date(d);
        } else if line.starts_with("Merge:") {
            commit.merge = true;
        } else {
            return None;
        }
    }
    (!commits.is_empty()).then_some(commits)
}

pub(super) fn log(run: &Run<'_>) -> Option<Compact> {
    if run.code != 0 {
        return None;
    }
    let commits = parse_log(run.stdout)?;
    let mut out: Vec<String> = Vec::new();
    for c in commits.iter().take(LOG_COMMITS) {
        let mut head = c.hash.chars().take(7).collect::<String>();
        if let Some(d) = c.decorations {
            head.push(' ');
            head.push_str(d);
        }
        let mut message = c.message.iter().copied();
        let subject = message.next().unwrap_or("");
        let merge = if c.merge { " [merge]" } else { "" };
        out.push(format!(
            "{head} {subject}{merge} [{}, {}]",
            c.author, c.date
        ));
        let body: Vec<&str> = message.filter(|l| !l.trim().is_empty()).collect();
        out.extend(body.iter().take(LOG_BODY_LINES).map(|l| format!("    {l}")));
        if body.len() > LOG_BODY_LINES {
            out.push(format!(
                "    {}",
                omitted(body.len() - LOG_BODY_LINES, "message lines")
            ));
        }
    }
    if commits.len() > LOG_COMMITS {
        out.push(omitted(commits.len() - LOG_COMMITS, "more commits"));
    }
    Some(Compact {
        stdout: join(&out),
        stderr: run.stderr.to_string(),
    })
}

pub(super) fn commit(run: &Run<'_>) -> Option<Compact> {
    static HEAD: OnceLock<regex::Regex> = OnceLock::new();
    let head = cached(&HEAD, r"^\[.+ [0-9a-f]{7,}\]");
    let first = run.stdout.lines().next()?;
    if run.code != 0 || !head.is_match(first) {
        return None;
    }
    let out: Vec<String> = run
        .stdout
        .lines()
        .map(|line| {
            // `create mode 100644 path`: the mode is the part nobody reads.
            for verb in ["create", "delete"] {
                let prefix = format!(" {verb} mode ");
                if let Some(rest) = line.strip_prefix(&prefix) {
                    if let Some((_, path)) = rest.split_once(' ') {
                        return format!(" {verb} {path}");
                    }
                }
            }
            line.to_string()
        })
        .collect();
    Some(Compact {
        stdout: join(&out),
        stderr: run.stderr.to_string(),
    })
}

/// Lines `git push` and `git pull` print as they work.
fn is_transfer_progress(line: &str) -> bool {
    let line = line.strip_prefix("remote: ").unwrap_or(line);
    [
        "Enumerating objects:",
        "Counting objects:",
        "Delta compression using",
        "Compressing objects:",
        "Writing objects:",
        "Total ",
        "Resolving deltas:",
        "Receiving objects:",
        "Unpacking objects:",
        "Finding sources:",
    ]
    .iter()
    .any(|p| line.starts_with(p))
}

/// Drops the progress lines of a successful transfer, from both streams.
fn without_progress(run: &Run<'_>) -> Option<Compact> {
    if run.code != 0 {
        return None;
    }
    let keep = |text: &str| {
        let lines: Vec<&str> = text.lines().filter(|l| !is_transfer_progress(l)).collect();
        join(&lines)
    };
    Some(Compact {
        stdout: keep(run.stdout),
        stderr: keep(run.stderr),
    })
}

pub(super) fn push(run: &Run<'_>) -> Option<Compact> {
    without_progress(run)
}

pub(super) fn pull(run: &Run<'_>) -> Option<Compact> {
    static STAT: OnceLock<regex::Regex> = OnceLock::new();
    let stat = cached(&STAT, r"^ .+ \| +(\d+|Bin .*)( [+-]*)?$");
    let base = without_progress(run)?;
    let mut dropped = 0usize;
    let mut out: Vec<&str> = Vec::new();
    for line in base.stdout.lines() {
        let mode_line = [" create mode ", " delete mode ", " mode change "]
            .iter()
            .any(|p| line.starts_with(p));
        if stat.is_match(line) || mode_line {
            dropped += 1;
        } else {
            out.push(line);
        }
    }
    let note = (dropped > 0).then(|| omitted(dropped, "diffstat lines"));
    out.extend(note.as_deref());
    Some(Compact {
        stdout: join(&out),
        stderr: base.stderr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tk::tests::{run, words as args};

    const STATUS: &str = "On branch main
Your branch is up to date with 'origin/main'.

Changes to be committed:
  (use \"git restore --staged <file>...\" to unstage)
\tmodified:   src/a.rs
\tnew file:   src/b.rs

Changes not staged for commit:
  (use \"git add <file>...\" to update what will be committed)
  (use \"git restore <file>...\" to discard changes in working directory)
\tmodified:   src/c.rs

Untracked files:
  (use \"git add <file>...\" to include in what will be committed)
\tnotes.txt

";

    #[test]
    fn status_drops_the_hints_and_blank_lines_and_keeps_every_entry() {
        let out = status(&run(0, STATUS, "")).unwrap().stdout;
        assert_eq!(
            out,
            "On branch main
Your branch is up to date with 'origin/main'.
Changes to be committed:
\tmodified:   src/a.rs
\tnew file:   src/b.rs
Changes not staged for commit:
\tmodified:   src/c.rs
Untracked files:
\tnotes.txt
"
        );
        assert!(out.len() < STATUS.len());
    }

    #[test]
    fn status_keeps_the_conflict_and_rebase_wording() {
        let text = "On branch main
You have unmerged paths.
  (fix conflicts and run \"git commit\")
  (use \"git merge --abort\" to abort the merge)

Unmerged paths:
  (use \"git add <file>...\" to mark resolution)
\tboth modified:   src/a.rs

no changes added to commit (use \"git add\" and/or \"git commit -a\")
";
        let out = status(&run(0, text, "")).unwrap().stdout;
        assert!(out.contains("You have unmerged paths."));
        assert!(out.contains("\tboth modified:   src/a.rs"));
        assert!(out.contains("no changes added to commit"));
    }

    #[test]
    fn a_long_status_list_is_cut_with_a_count() {
        let mut text = String::from("On branch main\n\nUntracked files:\n");
        for i in 0..80 {
            text.push_str(&format!("\tf{i}.txt\n"));
        }
        let out = status(&run(0, &text, "")).unwrap().stdout;
        assert_eq!(out.lines().filter(|l| l.contains(".txt")).count(), 50);
        assert!(out.contains("... 30 more entries omitted ..."), "{out}");
    }

    #[test]
    fn status_of_a_shape_it_does_not_know_is_left_to_the_fallback() {
        assert!(status(&run(0, "fatal: weird\n", "")).is_none());
        assert!(status(&run(128, STATUS, "")).is_none());
        assert!(status(&run(0, "", "")).is_none());
    }

    const DIFF: &str = "diff --git a/src/a.rs b/src/a.rs
index 1234567..89abcde 100644
--- a/src/a.rs
+++ b/src/a.rs
@@ -1,3 +1,3 @@
 fn a() {
-    1
+    2
 }
diff --git a/src/b.rs b/src/b.rs
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/src/b.rs
@@ -0,0 +1 @@
+fn b() {}
";

    #[test]
    fn diff_drops_blob_lines_and_keeps_the_change() {
        let out = diff(&run(0, DIFF, "")).unwrap().stdout;
        assert!(!out.contains("index "));
        for kept in [
            "diff --git a/src/a.rs b/src/a.rs",
            "--- a/src/a.rs",
            "+++ b/src/a.rs",
            "@@ -1,3 +1,3 @@",
            "-    1",
            "+    2",
            "new file mode 100644",
            "+fn b() {}",
        ] {
            assert!(out.contains(kept), "{kept}\n{out}");
        }
    }

    #[test]
    fn a_huge_file_diff_is_cut_per_file_with_the_path_in_the_marker() {
        let mut text =
            String::from("diff --git a/big.txt b/big.txt\nindex 1..2 100644\n--- a/big.txt\n+++ b/big.txt\n@@ -0,0 +1,400 @@\n");
        for i in 0..400 {
            text.push_str(&format!("+line {i}\n"));
        }
        text.push_str("diff --git a/small.txt b/small.txt\n--- a/small.txt\n+++ b/small.txt\n@@ -1 +1 @@\n-a\n+b\n");
        let out = diff(&run(0, &text, "")).unwrap().stdout;
        assert!(out.contains("+line 248\n"));
        assert!(!out.contains("+line 249\n"));
        assert!(
            out.contains("... 151 diff lines in big.txt omitted ..."),
            "{out}"
        );
        // The next file is untouched by the first one's cut.
        assert!(out.contains("-a\n+b\n"));
    }

    #[test]
    fn diff_keeps_a_non_diff_output_and_a_failure_to_the_fallback() {
        assert!(diff(&run(0, "warning: x\n", "")).is_none());
        assert!(diff(&run(129, DIFF, "usage: git diff\n")).is_none());
        // `--exit-code` returns 1 with a diff, and that is still a diff.
        assert!(diff(&run(1, DIFF, "")).is_some());
    }

    const LOG: &str = "commit 0123456789abcdef0123456789abcdef01234567 (HEAD -> main, origin/main)
Author: Alice Example <alice@example.com>
Date:   Tue Nov 3 10:22:11 2026 +0100

    Fix the parser

    The old one split on commas inside quotes.
    Add a regression test.

commit 89abcdef0123456789abcdef0123456789abcdef
Merge: 1111111 2222222
Author: Bob <bob@example.com>
Date:   Mon Nov 2 09:00:00 2026 +0000

    Merge branch 'topic'
";

    #[test]
    fn log_becomes_one_line_per_commit_with_a_short_body() {
        let out = log(&run(0, LOG, "")).unwrap().stdout;
        assert_eq!(
            out,
            "0123456 (HEAD -> main, origin/main) Fix the parser [Alice Example, Nov 3 2026]
    The old one split on commas inside quotes.
    Add a regression test.
89abcde Merge branch 'topic' [merge] [Bob, Nov 2 2026]
"
        );
        assert!(out.len() < LOG.len());
    }

    #[test]
    fn log_caps_the_commit_count_and_a_long_body() {
        let mut text = String::new();
        for i in 0..130 {
            text.push_str(&format!(
                "commit {i:040x}\nAuthor: A <a@b>\nDate:   Tue Nov 3 10:22:11 2026 +0100\n\n    c{i}\n\n    b1\n    b2\n    b3\n    b4\n    b5\n\n"
            ));
        }
        let out = log(&run(0, &text, "")).unwrap().stdout;
        assert_eq!(
            out.lines()
                .filter(|l| l.contains("[A, Nov 3 2026]"))
                .count(),
            100
        );
        assert!(out.contains("... 30 more commits omitted ..."));
        assert!(out.contains("... 2 message lines omitted ..."));
    }

    #[test]
    fn log_of_another_shape_is_left_alone() {
        assert!(log(&run(0, "abc1234 subject\n", "")).is_none());
        assert!(log(&run(0, "commit nothex\n", "")).is_none());
        // A header it does not know is not guessed at.
        let odd =
            "commit 0123456789abcdef0123456789abcdef01234567\nAuthor: A <a>\nGPG: yes\n\n    x\n";
        assert!(log(&run(0, odd, "")).is_none());
        assert!(log(&run(128, LOG, "")).is_none());
    }

    #[test]
    fn commit_keeps_the_summary_and_shortens_the_mode_lines() {
        let text = "[main abc1234] Fix the parser
 3 files changed, 10 insertions(+), 2 deletions(-)
 create mode 100644 src/new.rs
 delete mode 100644 src/old.rs
 rename src/{a.rs => b.rs} (90%)
";
        let out = commit(&run(0, text, "")).unwrap().stdout;
        assert_eq!(
            out,
            "[main abc1234] Fix the parser
 3 files changed, 10 insertions(+), 2 deletions(-)
 create src/new.rs
 delete src/old.rs
 rename src/{a.rs => b.rs} (90%)
"
        );
    }

    #[test]
    fn a_failed_commit_or_one_with_hook_output_is_not_rewritten() {
        assert!(commit(&run(1, "", "error: gpg failed to sign the data\n")).is_none());
        assert!(commit(&run(0, "hook says hi\n[main abc1234] x\n", "")).is_none());
    }

    #[test]
    fn push_drops_the_progress_and_keeps_the_ref_lines_and_remote_messages() {
        let stderr = "Enumerating objects: 5, done.
Counting objects: 100% (5/5), done.
Delta compression using up to 8 threads
Compressing objects: 100% (3/3), done.
Writing objects: 100% (3/3), 312 bytes | 312.00 KiB/s, done.
Total 3 (delta 2), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (2/2), completed with 2 local objects.
remote: Create a pull request for 'topic' on GitHub by visiting:
remote:      https://example.com/x/y/pull/new/topic
To example.com:x/y.git
   abc1234..def5678  topic -> topic
";
        let out = push(&run(0, "", stderr)).unwrap();
        assert_eq!(
            out.stderr,
            "remote: Create a pull request for 'topic' on GitHub by visiting:
remote:      https://example.com/x/y/pull/new/topic
To example.com:x/y.git
   abc1234..def5678  topic -> topic
"
        );
    }

    #[test]
    fn a_rejected_push_keeps_all_of_its_error() {
        let stderr = "To example.com:x/y.git
 ! [rejected]        main -> main (fetch first)
error: failed to push some refs to 'example.com:x/y.git'
hint: Updates were rejected because the remote contains work that you do not
";
        assert!(push(&run(1, "", stderr)).is_none());
        let shown = crate::tk::render(Plan::Filter(Family::GitPush), 1, "", stderr);
        assert_eq!(shown.stderr, stderr);
    }

    #[test]
    fn pull_drops_the_diffstat_and_keeps_the_summary() {
        let stdout = "Updating abc1234..def5678
Fast-forward
 src/a.rs | 4 ++--
 src/b.rs | 10 +++++++---
 docs/img.png | Bin 0 -> 1024 bytes
 3 files changed, 10 insertions(+), 4 deletions(-)
 create mode 100644 docs/img.png
";
        let out = pull(&run(
            0,
            stdout,
            "From example.com:x/y\n   abc1234..def5678  main -> origin/main\n",
        ))
        .unwrap();
        assert_eq!(
            out.stdout,
            "Updating abc1234..def5678
Fast-forward
 3 files changed, 10 insertions(+), 4 deletions(-)
... 4 diffstat lines omitted ...
"
        );
        assert!(out.stderr.contains("main -> origin/main"));
    }

    #[test]
    fn pull_that_conflicts_is_left_whole() {
        let stdout = "Auto-merging src/a.rs\nCONFLICT (content): Merge conflict in src/a.rs\nAutomatic merge failed; fix conflicts and then commit the result.\n";
        assert!(pull(&run(1, stdout, "")).is_none());
    }

    #[test]
    fn plans_follow_the_subcommand_and_its_flags() {
        let p = |line: &str| crate::tk::plan(&args(line));
        assert_eq!(p("git status"), Plan::Filter(Family::GitStatus));
        assert_eq!(p("git -C repo status"), Plan::Filter(Family::GitStatus));
        assert_eq!(
            p("git --no-pager diff HEAD~1"),
            Plan::Filter(Family::GitDiff)
        );
        assert_eq!(
            p("git log -n 5 --author=alice"),
            Plan::Filter(Family::GitLog)
        );
        assert_eq!(p("git log -5 -- src"), Plan::Filter(Family::GitLog));
        assert_eq!(p("git commit -m msg"), Plan::Filter(Family::GitCommit));
        assert_eq!(p("git commit -am msg"), Plan::Filter(Family::GitCommit));
        assert_eq!(p("git push origin main"), Plan::Filter(Family::GitPush));
        assert_eq!(p("git pull --rebase"), Plan::Filter(Family::GitPull));
        assert_eq!(p("git pull --no-edit"), Plan::Filter(Family::GitPull));
        assert_eq!(p("/usr/bin/git status"), Plan::Filter(Family::GitStatus));
    }

    #[test]
    fn machine_readable_interactive_and_plumbing_git_runs_untouched() {
        let p = |line: &str| crate::tk::plan(&args(line));
        for line in [
            "git status --porcelain",
            "git status --porcelain=v2",
            "git status -s",
            "git status -sb",
            "git status -z",
            "git status -bz",
            "git diff --binary",
            "git diff --stat",
            "git diff --name-only",
            "git diff --numstat",
            "git diff --output=out.patch",
            "git log --oneline",
            "git log --format=%H",
            "git log --pretty=format:%s",
            "git log --graph",
            "git log -p",
            "git log --stat",
            "git log -z",
            "git commit",
            "git commit --amend",
            "git commit --dry-run -m x",
            "git commit -m x --edit",
            "git commit -m x --patch",
            "git commit -m x -p",
            "git commit -m x --interactive",
            "git pull --edit",
            "git pull -e",
            "git pull --rebase=interactive",
            "git pull --rebase=i",
            "git pull -ri",
            "git push --porcelain",
            "git rev-parse HEAD",
            "git show HEAD:src/main.rs",
            "git ls-files",
            "git cat-file -p HEAD",
            "git branch --show-current",
            "git config user.name",
            "git rebase -i HEAD~3",
            "git add -p",
            "git --version",
            "git status --help",
            "git",
        ] {
            assert_eq!(p(line), Plan::Passthrough, "{line}");
        }
    }
}
