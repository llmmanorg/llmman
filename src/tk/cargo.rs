//! Filters for `cargo build`, `check`, `clippy` and `test`.
//!
//! Only the lines cargo prints to say it is working (`Compiling ...`) and
//! the lines of tests that passed are dropped. Warnings, errors, panics and
//! failure reports are kept as they are, whether or not the command failed.

use std::cell::Cell;
use std::sync::OnceLock;

use super::text::{cached, drop_lines, join, Dropped};
use super::{Compact, Family, Plan, Run};

pub(super) fn plan(rest: &[String]) -> Plan {
    // The subcommand is the first word after `+nightly` and cargo's own options.
    let sub = super::skip_options(rest, &["--color", "--config", "-C", "-Z"]).first();
    match sub.map(String::as_str) {
        Some("build" | "b" | "check" | "c" | "clippy") => Plan::Filter(Family::Cargo),
        Some("test" | "t") => Plan::Filter(Family::CargoTest),
        _ => Plan::Passthrough,
    }
}

/// Whether `line` is cargo narrating its work. Cargo right-aligns the verb
/// to 12 columns, so the indent says the line is not a build script's.
fn is_progress(line: &str) -> bool {
    static PROGRESS: OnceLock<regex::Regex> = OnceLock::new();
    cached(
        &PROGRESS,
        r"^( {3}(Compiling|Unpacking)| {4}(Checking|Updating)| {1}Downloading| {2}Downloaded| {5}Locking| {6}Adding| {7}Fresh) ",
    )
    .is_match(line)
}

fn without_progress(text: &str) -> String {
    // A failed build script's output follows `  --- stdout` or `  --- stderr`,
    // indented like cargo's own lines; it is kept whole, up to the next
    // line that is not indented.
    let in_script = Cell::new(false);
    drop_lines(text, "progress lines", |line| {
        if line == "  --- stdout" || line == "  --- stderr" {
            in_script.set(true);
        } else if !line.is_empty() && !line.starts_with("  ") {
            in_script.set(false);
        }
        !in_script.get() && is_progress(line)
    })
}

pub(super) fn build(run: &Run<'_>) -> Option<Compact> {
    Some(Compact {
        stdout: run.stdout.to_string(),
        stderr: without_progress(run.stderr),
    })
}

/// `test path::name ... ok`: the whole report of a test that passed.
fn is_passing_test(line: &str) -> bool {
    line.starts_with("test ") && line.ends_with(" ... ok")
}

/// `---- name stdout ----`: what a failed test printed follows.
fn is_report_header(line: &str) -> bool {
    line.starts_with("---- ") && (line.ends_with(" stdout ----") || line.ends_with(" stderr ----"))
}

pub(super) fn test(run: &Run<'_>) -> Option<Compact> {
    let lines: Vec<&str> = run.stdout.lines().collect();
    let mut out: Vec<String> = Vec::new();
    let mut passing = Dropped::default();
    // A failed test's output, up to the next `failures:`, is kept as it is.
    let mut in_report = false;
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let in_output = in_report && line != "failures:";
        if !in_output && is_passing_test(line) {
            passing.one(&mut out);
        } else if !in_output && line == "running 0 tests" {
            // A target with no tests: its header and its all-zero result
            // say nothing, and a workspace prints one per target.
            let next = lines[i + 1..]
                .iter()
                .position(|l| !l.trim().is_empty())
                .map(|k| i + 1 + k);
            match next {
                Some(j) if lines[j].starts_with("test result: ok. 0 passed; 0 failed;") => {
                    i = j;
                }
                _ => out.push(line.to_string()),
            }
        } else {
            // A different line ends a run of passing tests: the next run
            // gets its own marker.
            if !out.is_empty() {
                passing.finish_run(&mut out, "passing test lines");
            }
            in_report = in_output || is_report_header(line);
            out.push(line.to_string());
        }
        i += 1;
    }
    passing.finish_run(&mut out, "passing test lines");
    Some(Compact {
        stdout: join(&out),
        stderr: without_progress(run.stderr),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tk::tests::{run, words as args};

    const BUILD_OK: &str = "    Updating crates.io index
 Downloading crates ...
  Downloaded serde v1.0.200
   Compiling serde v1.0.200
   Compiling mylib v0.1.0 (/work/mylib)
warning: unused variable: `x`
 --> src/lib.rs:3:9
  |
3 |     let x = 1;
  |         ^ help: if this is intentional, prefix it with an underscore: `_x`
  |
  = note: `#[warn(unused_variables)]` on by default

warning: `mylib` (lib) generated 1 warning
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 4.31s
";

    #[test]
    fn build_drops_progress_and_keeps_warnings_whole() {
        let out = build(&run(0, "", BUILD_OK)).unwrap().stderr;
        assert!(out.starts_with("... 5 progress lines omitted ...\nwarning: unused variable"));
        for kept in [
            " --> src/lib.rs:3:9",
            "3 |     let x = 1;",
            "= note: `#[warn(unused_variables)]` on by default",
            "warning: `mylib` (lib) generated 1 warning",
            "Finished `dev` profile",
        ] {
            assert!(out.contains(kept), "{kept}\n{out}");
        }
        assert!(!out.contains("Compiling"));
    }

    #[test]
    fn a_failed_build_keeps_every_error_line() {
        let stderr = "   Compiling mylib v0.1.0 (/work/mylib)
error[E0308]: mismatched types
 --> src/lib.rs:2:18
  |
2 |     let n: u32 = \"a\";
  |            ---   ^^^ expected `u32`, found `&str`
  |            |
  |            expected due to this

error: could not compile `mylib` (lib) due to 1 previous error
";
        let shown = crate::tk::render(Plan::Filter(Family::Cargo), 101, "", stderr);
        assert!(!shown.stderr.contains("Compiling"));
        for line in stderr.lines().filter(|l| !l.contains("Compiling")) {
            assert!(shown.stderr.lines().any(|l| l == line), "{line}");
        }
    }

    const TEST_OUT: &str = "
running 4 tests
test parser::tests::empty ... ok
test parser::tests::quoted ... ok
test parser::tests::nested ... FAILED
test parser::tests::long ... ok

failures:

---- parser::tests::nested stdout ----
thread 'parser::tests::nested' panicked at src/parser.rs:42:9:
assertion `left == right` failed
  left: 1
 right: 2
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace


failures:
    parser::tests::nested

test result: FAILED. 3 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s

";

    #[test]
    fn test_drops_passing_lines_and_keeps_the_failure_report() {
        let stderr = "   Compiling mylib v0.1.0 (/work/mylib)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 1.20s
     Running unittests src/lib.rs (target/debug/deps/mylib-0123456789abcdef)
error: test failed, to rerun pass `--lib`
";
        let out = test(&run(101, TEST_OUT, stderr)).unwrap();
        assert!(out.stdout.contains("test parser::tests::nested ... FAILED"));
        assert!(!out.stdout.contains("::empty ... ok"));
        for kept in [
            "---- parser::tests::nested stdout ----",
            "thread 'parser::tests::nested' panicked at src/parser.rs:42:9:",
            "  left: 1",
            " right: 2",
            "test result: FAILED. 3 passed; 1 failed;",
            "... 2 passing test lines omitted ...",
            "... 1 passing test lines omitted ...",
        ] {
            assert!(out.stdout.contains(kept), "{kept}\n{}", out.stdout);
        }
        assert!(out
            .stderr
            .contains("error: test failed, to rerun pass `--lib`"));
        assert!(out.stderr.contains("Running unittests"));
        assert!(!out.stderr.contains("Compiling"));
    }

    #[test]
    fn test_with_nothing_but_passes_and_empty_targets_is_short() {
        let stdout = "
running 3 tests
test a ... ok
test b ... ok
test c ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s


running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

";
        let shown = crate::tk::render(Plan::Filter(Family::CargoTest), 0, stdout, "");
        assert_eq!(
            shown.stdout,
            "\nrunning 3 tests\n... 3 passing test lines omitted ...\n\ntest result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n"
        );
        assert!(shown.stdout.len() < stdout.len());
    }

    #[test]
    fn output_printed_by_a_test_is_not_taken_for_a_passing_line() {
        let stdout = "running 1 test\nprogress... test x ... ok\ntest x ... ok\n";
        let out = test(&run(0, stdout, "")).unwrap().stdout;
        assert!(out.contains("progress... test x ... ok"));
    }

    #[test]
    fn a_failed_tests_own_output_is_kept_whole() {
        let stdout = "running 1 test
test t ... FAILED

failures:

---- t stdout ----
test child_case ... ok
thread 't' panicked at src/lib.rs:1:1:
boom

failures:
    t

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
";
        let out = test(&run(101, stdout, "")).unwrap().stdout;
        assert_eq!(out, stdout);
    }

    #[test]
    fn only_cargos_own_progress_lines_are_dropped() {
        // A build script's text is indented by two, and is not cargo's.
        let stderr = "   Compiling x v0.1.0
error: failed to run custom build command for `x v0.1.0`

Caused by:
  process didn't exit successfully: `build-script-build` (exit status: 1)
  --- stderr
  Checking required library failed: missing libfoo
";
        let out = build(&run(101, "", stderr)).unwrap().stderr;
        assert!(out.contains("  Checking required library failed: missing libfoo"));
        assert!(!out.contains("   Compiling x"));
    }

    #[test]
    fn a_build_scripts_downloaded_line_is_part_of_the_failure() {
        // Cargo indents the script's text by two, as it does `Downloaded`.
        let stderr = "  Downloaded serde v1.0.200
   Compiling x v0.1.0
error: failed to run custom build command for `x v0.1.0`

Caused by:
  process didn't exit successfully: `build-script-build` (exit status: 1)
  --- stdout
  Downloaded libfoo 1.2 from the mirror
  --- stderr
  Downloaded libfoo 1.2 does not match its checksum

  Fetching libfoo again failed
warning: build failed, waiting for other jobs to finish...
   Compiling y v0.1.0
";
        let out = build(&run(101, "", stderr)).unwrap().stderr;
        for kept in [
            "  Downloaded libfoo 1.2 from the mirror",
            "  Downloaded libfoo 1.2 does not match its checksum",
            "  Fetching libfoo again failed",
            "warning: build failed, waiting for other jobs to finish...",
        ] {
            assert!(out.contains(kept), "{kept}\n{out}");
        }
        // Cargo's own lines before and after the script's text still go.
        assert!(!out.contains("serde"), "{out}");
        assert!(!out.contains("Compiling"), "{out}");
    }

    #[test]
    fn plans_cover_the_build_family_and_leave_the_rest() {
        let p = |line: &str| crate::tk::plan(&args(line));
        assert_eq!(p("cargo build --release"), Plan::Filter(Family::Cargo));
        assert_eq!(p("cargo +nightly check"), Plan::Filter(Family::Cargo));
        assert_eq!(
            p("cargo --color always test"),
            Plan::Filter(Family::CargoTest)
        );
        assert_eq!(p("cargo --color=never build"), Plan::Filter(Family::Cargo));
        assert_eq!(p("cargo clippy --all-targets"), Plan::Filter(Family::Cargo));
        assert_eq!(p("cargo test --lib"), Plan::Filter(Family::CargoTest));
        for line in [
            "cargo build --message-format=json",
            "cargo test -- --format json",
            "cargo test -- --list --format=terse",
            "cargo run",
            "cargo fmt --check",
            "cargo metadata",
            "cargo --version",
            "cargo",
        ] {
            assert_eq!(p(line), Plan::Passthrough, "{line}");
        }
    }
}
