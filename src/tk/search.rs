//! Filter for `grep`, `egrep`, `fgrep` and `rg`: `path:line:text` matches
//! grouped under their file, long lines cut.
//!
//! Output is only regrouped when every line has the `path:` shape of a file
//! that exists; context lines, `--heading`, a binary file notice or any
//! other shape comes back as it was. A search of one file is not touched at
//! all: it has no `path:` in front, whatever its lines look like.

use std::collections::HashSet;
use std::sync::OnceLock;

use super::text::{cached, cut_line, join, omitted};
use super::{before_dashes, has_flag, has_short, skip_options, Compact, Family, Plan, Run};

/// Longest match line shown.
const MATCH_CHARS: usize = 200;
/// Most matches listed under one file.
const MATCHES_PER_FILE: usize = 50;
/// Most distinct paths looked up on disk before a shape is trusted.
const PATHS_CHECKED: usize = 200;

pub(super) fn plan(rest: &[String]) -> Plan {
    // Letters that turn the output into names, counts, offsets, matched
    // parts only, context, NUL separators or nothing. A digit is `-NUM`,
    // grep's context; `0` is rg's `--null`.
    const SHORT: &str = "bclLqZhoABC0123456789";
    const LONG: &[&str] = &[
        "--count",
        "--count-matches",
        "--files",
        "--files-with-matches",
        "--files-without-match",
        "--quiet",
        "--no-filename",
        "--byte-offset",
        "--context",
        "--after-context",
        "--before-context",
        "--stats",
        "--only-matching",
    ];
    let flags = before_dashes(rest);
    if has_short(flags, SHORT) || has_flag(flags, LONG) || names_one_file(rest) {
        Plan::Passthrough
    } else {
        Plan::Filter(Family::Search)
    }
}

/// Whether the search is of one file that is not a directory, with no flag
/// that puts a name in front of matches: then a `path:` in the output is
/// the text's own. Any doubt is a no.
fn names_one_file(rest: &[String]) -> bool {
    // Options that take the next word, so it is not taken for an operand.
    const VALUE: &[&str] = &["-e", "-f", "-m", "--regexp", "--file", "--max-count"];
    const PATTERN: &[&str] = &["-e", "-f", "--regexp", "--file"];
    let flags = before_dashes(rest);
    if has_short(flags, "HrR")
        || has_flag(
            flags,
            &["--with-filename", "--recursive", "--dereference-recursive"],
        )
    {
        return false;
    }
    let mut operands: Vec<&String> = skip_options(flags, VALUE)
        .iter()
        .filter(|a| !a.starts_with('-'))
        .collect();
    operands.extend(rest.iter().skip(flags.len() + 1));
    // Without a pattern option, the first operand is the pattern.
    let paths = operands.get(usize::from(!has_flag(flags, PATTERN))..);
    matches!(paths, Some([path]) if !std::path::Path::new(path).is_dir())
}

/// A match: `path:text`, or `path:line:text` when the lines are numbered.
struct Hit<'a> {
    path: &'a str,
    line: Option<&'a str>,
    text: &'a str,
}

fn parse(stdout: &str) -> Option<Vec<Hit<'_>>> {
    static NUMBERED: OnceLock<regex::Regex> = OnceLock::new();
    let numbered = cached(&NUMBERED, r"^([^:]+):(\d+):(.*)$");
    let mut hits = Vec::new();
    let all_numbered = stdout.lines().all(|l| numbered.is_match(l));
    for line in stdout.lines() {
        if all_numbered {
            let c = numbered.captures(line)?;
            hits.push(Hit {
                path: c.get(1)?.as_str(),
                line: Some(c.get(2)?.as_str()),
                text: c.get(3)?.as_str(),
            });
        } else {
            let (path, text) = line.split_once(':')?;
            if path.is_empty() {
                return None;
            }
            hits.push(Hit {
                path,
                line: None,
                text,
            });
        }
    }
    // A path is a file this machine has; anything else is text with a colon.
    let mut seen = HashSet::new();
    for hit in &hits {
        if seen.insert(hit.path)
            && (seen.len() > PATHS_CHECKED || !std::path::Path::new(hit.path).is_file())
        {
            return None;
        }
    }
    (!hits.is_empty()).then_some(hits)
}

pub(super) fn filter(run: &Run<'_>) -> Option<Compact> {
    // 0 is "found"; 1 is "nothing found" and has no output; 2 is an error.
    if run.code != 0 {
        return None;
    }
    let hits = parse(run.stdout)?;
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < hits.len() {
        let path = hits[i].path;
        let group_len = hits[i..].iter().take_while(|h| h.path == path).count();
        let group = &hits[i..i + group_len];
        out.push(format!("{path} ({group_len})"));
        for hit in group.iter().take(MATCHES_PER_FILE) {
            let text = cut_line(hit.text.trim(), MATCH_CHARS);
            out.push(match hit.line {
                Some(n) => format!("  {n}: {text}"),
                None => format!("  {text}"),
            });
        }
        if group_len > MATCHES_PER_FILE {
            out.push(format!(
                "  {}",
                omitted(group_len - MATCHES_PER_FILE, "more matches")
            ));
        }
        i += group_len;
    }
    Some(Compact {
        stdout: join(&out),
        stderr: run.stderr.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use crate::tk::tests::run;
    use crate::tk::tests::words as args;

    /// A throwaway directory with the named files, and the process's
    /// working directory is not changed: paths are absolute.
    #[cfg(unix)]
    fn files(names: &[&str]) -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().unwrap();
        for name in names {
            let path = dir.path().join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, "x\n").unwrap();
        }
        let root = dir.path().to_string_lossy().into_owned();
        (dir, root)
    }

    #[cfg(unix)]
    #[test]
    fn numbered_matches_are_grouped_under_their_file() {
        let (_d, root) = files(&["src/a.rs", "src/b.rs"]);
        let stdout = format!(
            "{root}/src/a.rs:12:    let foo = 1;\n{root}/src/a.rs:40:fn foo() {{}}\n{root}/src/b.rs:3:// foo\n"
        );
        let r = run(0, &stdout, "");
        assert_eq!(
            filter(&r).unwrap().stdout,
            format!(
                "{root}/src/a.rs (2)\n  12: let foo = 1;\n  40: fn foo() {{}}\n{root}/src/b.rs (1)\n  3: // foo\n"
            )
        );
    }

    #[cfg(unix)]
    #[test]
    fn unnumbered_matches_are_grouped_too() {
        let (_d, root) = files(&["a.txt", "b.txt"]);
        let stdout = format!("{root}/a.txt:hello: world\n{root}/b.txt:hello\n");
        let r = run(0, &stdout, "");
        assert_eq!(
            filter(&r).unwrap().stdout,
            format!("{root}/a.txt (1)\n  hello: world\n{root}/b.txt (1)\n  hello\n")
        );
    }

    #[cfg(unix)]
    #[test]
    fn long_lines_are_cut_and_a_busy_file_is_capped() {
        let (_d, root) = files(&["big.log"]);
        let long = "y".repeat(500);
        let mut stdout = format!("{root}/big.log:1:{long}\n");
        for i in 2..=60 {
            stdout.push_str(&format!("{root}/big.log:{i}:hit {i}\n"));
        }
        let r = run(0, &stdout, "");
        let out = filter(&r).unwrap().stdout;
        assert!(out.contains("[300 more chars]"), "{out}");
        assert!(out.contains("big.log (60)"));
        assert!(out.contains("  50: hit 50"));
        assert!(!out.contains("hit 51"));
        assert!(out.contains("... 10 more matches omitted ..."));
    }

    #[test]
    fn text_that_is_not_a_list_of_files_is_left_as_it_is() {
        for stdout in [
            // One file searched: no path in front, and colons in the text.
            "Error: disk full\nError: retry: later\n",
            // Context separators.
            "a.txt-1-before\na.txt:2:match\n--\n",
            // Paths that are not files here.
            "no/such/file.rs:1:x\nno/such/other.rs:2:y\n",
            "Binary file x matches\n",
            "",
        ] {
            let r = Run {
                code: 0,
                stdout,
                stderr: "",
            };
            assert!(filter(&r).is_none(), "{stdout:?}");
        }
        // No match (1) and an error (2) are not reshaped.
        for code in [1, 2] {
            let r = Run {
                code,
                stdout: "",
                stderr: "grep: nope: No such file or directory\n",
            };
            assert!(filter(&r).is_none());
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_search_of_one_file_is_left_alone_but_one_that_names_files_is_not() {
        let (_d, root) = files(&["build.log", "a.rs"]);
        let p = |line: &str| crate::tk::plan(&args(line));
        // Its matches have no `path:` in front, even when the text has one.
        for line in [
            format!("grep -n error {root}/build.log"),
            format!("rg error {root}/build.log"),
            format!("grep -e error -e warn {root}/build.log"),
            format!("grep error -- {root}/build.log"),
            format!("grep error {root}/build.log -n"),
        ] {
            assert_eq!(p(&line), Plan::Passthrough, "{line}");
        }
        for line in [
            format!("grep -nH error {root}/build.log"),
            format!("grep --with-filename error {root}/build.log"),
            format!("grep -rn error {root}/build.log"),
            format!("grep -n error {root}/build.log {root}/a.rs"),
            format!("grep -n error {root}"),
            format!("rg error {root}"),
            "rg error".to_string(),
        ] {
            assert_eq!(p(&line), Plan::Filter(Family::Search), "{line}");
        }
    }

    #[test]
    fn plans_leave_counts_names_context_and_null_separated_output_alone() {
        let p = |line: &str| crate::tk::plan(&args(line));
        for line in [
            "grep -rn foo src",
            "grep -rniw foo .",
            "rg foo",
            "rg -n foo .",
            "egrep -e a -e b .",
        ] {
            assert_eq!(p(line), Plan::Filter(Family::Search), "{line}");
        }
        for line in [
            "grep -c foo f",
            "grep -rl foo .",
            "grep -L foo f",
            "grep -q foo f",
            "grep -h foo f",
            "grep -o foo f",
            "grep -on foo f",
            "grep --only-matching foo f",
            "rg -o foo",
            "grep -bH foo f",
            "grep --byte-offset foo f",
            "grep -nz foo f",
            "grep --null-data foo f",
            "rg -n0 foo",
            "grep -A3 foo f",
            "grep -rnC 2 foo .",
            "grep -2 foo f",
            "grep --count foo f",
            "grep --context=3 foo f",
            "grep -Z foo f",
            "grep --null foo f",
            "rg --json foo",
            "rg --files",
            "rg --files-with-matches foo",
            "rg -l foo",
            "rg --stats foo",
            "grep -V",
        ] {
            assert_eq!(p(line), Plan::Passthrough, "{line}");
        }
    }
}
