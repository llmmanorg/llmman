//! Filters for `go test`, `go build`, `go vet` and `go install`.
//!
//! `go test` loses the lines of tests that passed and of packages without
//! tests; a failing test's output, build errors and vet findings are kept
//! whole. Module download notices are counted.

use std::sync::OnceLock;

use super::text::{cached, drop_lines, join, Dropped};
use super::{Compact, Family, Plan, Run};

pub(super) fn plan(rest: &[String]) -> Plan {
    let sub = rest.iter().find(|a| !a.starts_with('-'));
    // `-json` is a stream for a program to read.
    if super::has_flag(rest, &["-json", "-test.json"]) {
        return Plan::Passthrough;
    }
    match sub.map(String::as_str) {
        Some("test") => Plan::Filter(Family::GoTest),
        Some("build" | "vet" | "install") => Plan::Filter(Family::Go),
        _ => Plan::Passthrough,
    }
}

/// `go: downloading x v1`, from the module machinery.
fn is_module_notice(line: &str) -> bool {
    ["go: downloading ", "go: finding ", "go: extracting "]
        .iter()
        .any(|p| line.starts_with(p))
}

fn without_module_notices(text: &str) -> String {
    drop_lines(text, "module download lines", is_module_notice)
}

pub(super) fn build(run: &Run<'_>) -> Option<Compact> {
    Some(Compact {
        stdout: without_module_notices(run.stdout),
        stderr: without_module_notices(run.stderr),
    })
}

pub(super) fn test(run: &Run<'_>) -> Option<Compact> {
    static RUNNING: OnceLock<regex::Regex> = OnceLock::new();
    static PASSED: OnceLock<regex::Regex> = OnceLock::new();
    static NO_TESTS: OnceLock<regex::Regex> = OnceLock::new();
    static SWITCH: OnceLock<regex::Regex> = OnceLock::new();
    // `-v` narration of each test, and the pass line of each.
    let running = cached(&RUNNING, r"^=== (RUN|PAUSE|CONT|NAME)\s");
    let passed = cached(&PASSED, r"^\s*--- PASS: ");
    let no_tests = cached(&NO_TESTS, r"^\?\s+\S+\s+\[no test files\]$");
    // These say whose output follows: a test's own (RUN), or, when parallel
    // tests interleave, another's (CONT, NAME).
    let switch = cached(&SWITCH, r"^=== (RUN|CONT|NAME)\s");
    let text = without_module_notices(run.stdout);
    let mut out: Vec<String> = Vec::new();
    let (mut narration, mut untested) = (Dropped::default(), Dropped::default());
    // The last switch line: kept only when output follows it.
    let mut owner: Option<&str> = None;
    for line in text.lines() {
        let narrates = running.is_match(line) || passed.is_match(line) || line == "PASS";
        if narrates || no_tests.is_match(line) {
            if owner.take().is_some() {
                narration.one(&mut out);
            }
            if switch.is_match(line) {
                owner = Some(line);
            } else if narrates {
                narration.one(&mut out);
            } else {
                untested.one(&mut out);
            }
        } else {
            out.extend(owner.take().map(String::from));
            out.push(line.to_string());
        }
    }
    if owner.is_some() {
        narration.one(&mut out);
    }
    narration.finish(&mut out, "passing-test lines");
    untested.finish(&mut out, "packages without test files");
    Some(Compact {
        stdout: join(&out),
        stderr: without_module_notices(run.stderr),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tk::tests::{run, words as args};

    #[test]
    fn a_passing_run_keeps_the_package_lines_and_counts_the_untested() {
        let stdout = "?   \texample.com/app/cmd\t[no test files]
?   \texample.com/app/internal/gen\t[no test files]
ok  \texample.com/app/parser\t0.012s
ok  \texample.com/app/store\t(cached)
";
        let out = test(&run(0, stdout, "")).unwrap().stdout;
        assert_eq!(
            out,
            "... 2 packages without test files omitted ...
ok  \texample.com/app/parser\t0.012s
ok  \texample.com/app/store\t(cached)
"
        );
    }

    #[test]
    fn a_failing_run_keeps_the_failure_output_and_drops_the_narration() {
        let stdout = "=== RUN   TestParse
=== RUN   TestParse/empty
--- PASS: TestParse/empty (0.00s)
=== RUN   TestParse/nested
    parser_test.go:42: got 1, want 2
--- FAIL: TestParse/nested (0.00s)
--- FAIL: TestParse (0.00s)
=== RUN   TestStore
--- PASS: TestStore (0.01s)
FAIL
FAIL\texample.com/app/parser\t0.013s
FAIL
";
        let out = test(&run(1, stdout, "")).unwrap().stdout;
        for kept in [
            "    parser_test.go:42: got 1, want 2",
            "--- FAIL: TestParse/nested (0.00s)",
            "--- FAIL: TestParse (0.00s)",
            "FAIL\texample.com/app/parser\t0.013s",
        ] {
            assert!(out.contains(kept), "{kept}\n{out}");
        }
        // Only the RUN line that the failure's output follows is kept.
        assert_eq!(out.matches("=== RUN").count(), 1, "{out}");
        assert!(out.contains("=== RUN   TestParse/nested\n    parser_test.go"));
        assert!(!out.contains("--- PASS"));
        // The bare FAIL lines are not mistaken for the PASS line.
        assert_eq!(out.lines().filter(|l| *l == "FAIL").count(), 2);
    }

    #[test]
    fn interleaved_output_keeps_the_line_that_names_its_test() {
        let stdout = "=== RUN   TestA
=== PAUSE TestA
=== RUN   TestB
=== PAUSE TestB
=== CONT  TestA
    a_test.go:10: from A
=== CONT  TestB
=== NAME  TestA
    a_test.go:11: more from A
=== CONT  TestB
--- PASS: TestA (0.00s)
--- PASS: TestB (0.00s)
PASS
";
        let out = test(&run(0, stdout, "")).unwrap().stdout;
        assert_eq!(
            out,
            "... 9 passing-test lines omitted ...
=== CONT  TestA
    a_test.go:10: from A
=== NAME  TestA
    a_test.go:11: more from A
"
        );
    }

    #[test]
    fn a_passing_test_that_logged_keeps_the_run_line_its_output_follows() {
        let stdout = "=== RUN   TestA
--- PASS: TestA (0.00s)
=== RUN   TestB
    b_test.go:7: diagnostic
--- PASS: TestB (0.00s)
PASS
";
        let out = test(&run(0, stdout, "")).unwrap().stdout;
        assert_eq!(
            out,
            "... 4 passing-test lines omitted ...
=== RUN   TestB
    b_test.go:7: diagnostic
"
        );
    }

    #[test]
    fn a_compile_error_is_kept_whole() {
        let stdout = "# example.com/app/parser [example.com/app/parser.test]
parser/parser_test.go:9:2: undefined: Parse
FAIL\texample.com/app/parser [build failed]
FAIL
";
        let shown = crate::tk::render(Plan::Filter(Family::GoTest), 1, stdout, "");
        assert_eq!(shown.stdout, stdout);
    }

    #[test]
    fn build_counts_module_downloads_and_keeps_errors() {
        let stderr = "go: downloading github.com/x/y v1.2.3
go: downloading github.com/z/w v0.4.0
# example.com/app
./main.go:7:2: undefined: foo
";
        let out = build(&run(1, "", stderr)).unwrap().stderr;
        assert_eq!(
            out,
            "... 2 module download lines omitted ...
# example.com/app
./main.go:7:2: undefined: foo
"
        );
    }

    #[test]
    fn plans() {
        let p = |line: &str| crate::tk::plan(&args(line));
        assert_eq!(p("go test ./..."), Plan::Filter(Family::GoTest));
        assert_eq!(p("go build ./..."), Plan::Filter(Family::Go));
        assert_eq!(p("go vet ./..."), Plan::Filter(Family::Go));
        for line in [
            "go test -json ./...",
            "go test -json=true ./...",
            "go run .",
            "go env GOPATH",
            "go mod tidy",
            "go version",
            "go",
        ] {
            assert_eq!(p(line), Plan::Passthrough, "{line}");
        }
    }
}
