//! Filter for `npm test`, `pnpm test` and `yarn test`: the package manager's
//! banner and the lines of passing tests and files go. Whatever else the
//! runner printed (failures, stack traces, summaries) is kept whole, from
//! both streams: jest reports on stderr.

use std::sync::OnceLock;

use super::text::{cached, join, Dropped};
use super::{Compact, Family, Plan, Run};

pub(super) fn plan(rest: &[String]) -> Plan {
    // `test`, `t`, `run test`, `run-script test`, before any `--`.
    let words: Vec<&str> = super::before_dashes(rest)
        .iter()
        .map(String::as_str)
        .filter(|a| !a.starts_with('-'))
        .collect();
    let is_test = matches!(
        words.as_slice(),
        ["test" | "t" | "tst"] | ["run" | "run-script", "test"]
    );
    // A reporter other than the default is a format to read.
    let custom_reporter = super::has_flag(rest, &["--reporter", "--reporters"]);
    if is_test && !custom_reporter {
        Plan::Filter(Family::JsTest)
    } else {
        Plan::Passthrough
    }
}

/// Drops the banner and the passing lines from one stream.
fn compact(text: &str) -> String {
    static PASSING: OnceLock<regex::Regex> = OnceLock::new();
    let passing = cached(&PASSING, r"^(\s*(✓|✔|√)\s|PASS\s)");
    let mut out: Vec<String> = Vec::new();
    let mut dropped = Dropped::default();
    // `> pkg@1.0.0 test` and `> jest`: only at the top.
    let mut banner = true;
    for line in text.lines() {
        if banner && (line.starts_with("> ") || line.trim().is_empty()) {
            continue;
        }
        banner = false;
        if passing.is_match(line) {
            dropped.one(&mut out);
        } else {
            out.push(line.to_string());
        }
    }
    dropped.finish(&mut out, "passing-test lines");
    join(&out)
}

pub(super) fn test(run: &Run<'_>) -> Option<Compact> {
    Some(Compact {
        stdout: compact(run.stdout),
        stderr: compact(run.stderr),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tk::tests::words as args;

    #[test]
    fn passing_files_and_the_banner_go_and_the_failure_stays() {
        let stdout = "
> app@1.0.0 test
> jest

";
        let stderr = "PASS src/a.test.js
PASS src/b.test.js
FAIL src/c.test.js
  ● total › adds

    expect(received).toBe(expected) // Object.is equality

    Expected: 4
    Received: 3

      12 |   expect(total([1, 2])).toBe(4);

      at Object.<anonymous> (src/c.test.js:12:27)

Test Suites: 1 failed, 2 passed, 3 total
Tests:       1 failed, 5 passed, 6 total
";
        let r = Run {
            code: 1,
            stdout,
            stderr,
        };
        let out = test(&r).unwrap();
        assert_eq!(out.stdout, "");
        for kept in [
            "FAIL src/c.test.js",
            "● total › adds",
            "Expected: 4",
            "Received: 3",
            "at Object.<anonymous> (src/c.test.js:12:27)",
            "Tests:       1 failed, 5 passed, 6 total",
            "... 2 passing-test lines omitted ...",
        ] {
            assert!(out.stderr.contains(kept), "{kept}\n{}", out.stderr);
        }
        assert!(!out.stderr.contains("PASS src"));
    }

    #[test]
    fn mocha_style_passing_lines_go_and_the_summary_stays() {
        let shown = crate::tk::render(
            Plan::Filter(Family::JsTest),
            0,
            "> x@1 test\n> mocha\n\n  ✔ works\n  ✔ also works\n\n  2 passing (3ms)\n",
            "",
        );
        assert!(shown.stdout.contains("2 passing (3ms)"));
        assert!(!shown.stdout.contains("✔ works"));
    }

    #[test]
    fn plans() {
        let p = |line: &str| crate::tk::plan(&args(line));
        for line in [
            "npm test",
            "npm t",
            "npm run test",
            "npm test -- --coverage",
            "pnpm test",
            "yarn test",
            "yarn run test",
        ] {
            assert_eq!(p(line), Plan::Filter(Family::JsTest), "{line}");
        }
        for line in [
            "npm install",
            "npm run build",
            "npm run test:e2e",
            "npm test --watch",
            "npm test -- --watchAll",
            "npm test -- --json",
            "npm test -- --reporter=json",
            "npm --version",
            "yarn",
        ] {
            assert_ne!(p(line), Plan::Filter(Family::JsTest), "{line}");
        }
        assert_eq!(p("npm install"), Plan::Passthrough);
    }
}
