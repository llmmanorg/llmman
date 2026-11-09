//! Filter for `pytest`: the session banner and the per-file progress lines
//! go, and so do the lines of tests that passed under `-v`. Everything from
//! the first `=== ... ===` rule on (failure reports, captured output,
//! warnings and the summary) is kept whole.

use std::sync::OnceLock;

use super::text::{cached, join, Dropped};
use super::{Compact, Family, Plan, Run};

pub(super) fn plan(_rest: &[String]) -> Plan {
    Plan::Filter(Family::Pytest)
}

pub(super) fn filter(run: &Run<'_>) -> Option<Compact> {
    static BANNER: OnceLock<regex::Regex> = OnceLock::new();
    static PROGRESS: OnceLock<regex::Regex> = OnceLock::new();
    static PASSED: OnceLock<regex::Regex> = OnceLock::new();
    let banner = cached(
        &BANNER,
        r"^(platform |rootdir: |configfile: |inifile: |plugins: |cachedir: |testpaths: )",
    );
    // `tests/test_a.py ..F.. [ 40%]`, or `..F.. [ 40%]` under `-q`.
    let progress = cached(&PROGRESS, r"^(\S+\.py )?[.FEsxX]+\s*(\[\s*\d+%\])?$");
    let passed = cached(&PASSED, r"^\S+::\S.* PASSED( +\[\s*\d+%\])?$");
    // 1 is "some tests failed"; the other codes are not a test report.
    if !matches!(run.code, 0 | 1) || !run.stdout.contains("test session starts") {
        return None;
    }
    let mut out: Vec<String> = Vec::new();
    let (mut progress_lines, mut passes) = (Dropped::default(), Dropped::default());
    // The banner is the session header, up to the `collected N items` line,
    // or the blank line after the worker lines of xdist: later, such a line
    // is output a test printed. From the first `=== ... ===` rule on, the
    // text is a report, which may hold a nested pytest's own lines.
    let (mut in_header, mut in_report) = (false, false);
    for line in run.stdout.lines() {
        let rule = line.starts_with('=');
        if in_report {
            out.push(line.to_string());
            continue;
        }
        if rule && line.contains(" test session starts ") {
            in_header = true;
            continue;
        }
        if in_header {
            if banner.is_match(line) {
                continue;
            }
            in_header = !(rule || line.trim().is_empty() || line.starts_with("collect"));
        }
        in_report = rule;
        if progress.is_match(line) && (line.contains(".py ") || line.trim_end().ends_with("%]")) {
            progress_lines.one(&mut out);
        } else if passed.is_match(line) {
            passes.one(&mut out);
        } else {
            out.push(line.to_string());
        }
    }
    progress_lines.finish(&mut out, "progress lines");
    passes.finish(&mut out, "passed-test lines");
    Some(Compact {
        stdout: join(&out),
        stderr: run.stderr.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tk::tests::{run, words as args};

    const FAILING: &str =
        "============================= test session starts ==============================
platform linux -- Python 3.12.1, pytest-8.0.0, pluggy-1.4.0
rootdir: /work/app
configfile: pyproject.toml
plugins: cov-4.1.0, xdist-3.5.0
collected 5 items

tests/test_a.py ..F..                                                    [100%]

=================================== FAILURES ===================================
__________________________________ test_total __________________________________

    def test_total():
>       assert total([1, 2]) == 4
E       assert 3 == 4
E        +  where 3 = total([1, 2])

tests/test_a.py:14: AssertionError
=========================== short test summary info ============================
FAILED tests/test_a.py::test_total - assert 3 == 4
========================= 1 failed, 4 passed in 0.12s ==========================
";

    #[test]
    fn a_failing_session_keeps_the_report_and_drops_the_banner_and_dots() {
        let r = run(1, FAILING, "");
        let out = filter(&r).unwrap().stdout;
        for kept in [
            "collected 5 items",
            "= FAILURES =",
            ">       assert total([1, 2]) == 4",
            "E       assert 3 == 4",
            "tests/test_a.py:14: AssertionError",
            "FAILED tests/test_a.py::test_total - assert 3 == 4",
            "1 failed, 4 passed in 0.12s",
            "... 1 progress lines omitted ...",
        ] {
            assert!(out.contains(kept), "{kept}\n{out}");
        }
        assert!(!out.contains("platform linux"));
        assert!(!out.contains("rootdir"));
        assert!(!out.contains("tests/test_a.py ..F.."));
        assert!(out.len() < FAILING.len());
    }

    #[test]
    fn a_banner_like_line_a_test_printed_is_kept() {
        let stdout =
            "============================= test session starts ==============================
platform linux -- Python 3.12.1, pytest-8.0.0, pluggy-1.4.0
rootdir: /work/app
collected 1 item

tests/test_a.py F                                                        [100%]

=================================== FAILURES ===================================
__________________________________ test_dir ____________________________________
----------------------------- Captured stdout call -----------------------------
rootdir: missing fixture directory
plugins: none loaded
E   assert False
========================= 1 failed in 0.01s ==========================
";
        let out = filter(&run(1, stdout, "")).unwrap().stdout;
        assert!(
            out.contains("\nrootdir: missing fixture directory\n"),
            "{out}"
        );
        assert!(out.contains("\nplugins: none loaded\n"), "{out}");
        assert!(!out.contains("rootdir: /work/app"));
        assert!(!out.contains("platform linux"));
        assert!(out.contains("collected 1 item"));
    }

    #[test]
    fn xdist_has_no_collected_line_and_its_banner_still_ends() {
        // `pytest -n 2`: worker lines and a blank line end the header.
        let stdout =
            "============================= test session starts ==============================
platform linux -- Python 3.12.1, pytest-8.0.0, pluggy-1.4.0
rootdir: /work/app
plugins: xdist-3.5.0
created: 2/2 workers
2 workers [2 items]

.F                                                                       [100%]
=================================== FAILURES ===================================
__________________________________ test_dir ____________________________________
----------------------------- Captured stdout call -----------------------------
rootdir: missing fixture directory
plugins: none loaded
E   assert False
========================= 1 failed, 1 passed in 0.01s ==========================
";
        let out = filter(&run(1, stdout, "")).unwrap().stdout;
        assert!(
            out.contains("\nrootdir: missing fixture directory\n"),
            "{out}"
        );
        assert!(out.contains("\nplugins: none loaded\n"), "{out}");
        assert!(out.contains("2 workers [2 items]"), "{out}");
        assert!(!out.contains("rootdir: /work/app"), "{out}");
        assert!(!out.contains("platform linux"), "{out}");
        assert!(out.contains("... 1 progress lines omitted ..."), "{out}");
    }

    #[test]
    fn a_report_is_kept_verbatim_even_when_it_looks_like_progress() {
        let head =
            "============================= test session starts ==============================
platform linux -- Python 3.12.1, pytest-8.0.0, pluggy-1.4.0
rootdir: /work/app
collected 1 item

tests/test_outer.py F                                                    [100%]

";
        // A test that ran pytest itself: its output is in the report.
        let report =
            "=================================== FAILURES ===================================
________________________________ test_runs_child _______________________________
----------------------------- Captured stdout call -----------------------------
============================= test session starts ==============================
platform linux -- Python 3.12.1, pytest-8.0.0, pluggy-1.4.0
rootdir: /tmp/child
collected 2 items

tests/test_child.py F.                                                   [100%]
tests/test_child.py::test_ok PASSED                                      [100%]

========================= 1 failed, 1 passed in 0.01s ==========================
E   assert 1 == 0
=========================== short test summary info ============================
FAILED tests/test_outer.py::test_runs_child - assert 1 == 0
========================= 1 failed in 0.05s ==========================
";
        let out = filter(&run(1, &format!("{head}{report}"), ""))
            .unwrap()
            .stdout;
        assert!(out.ends_with(report), "{out}");
        assert!(out.contains("... 1 progress lines omitted ..."), "{out}");
        assert!(!out.contains("rootdir: /work/app"), "{out}");
    }

    #[test]
    fn verbose_passes_are_counted_and_failures_stay() {
        let stdout =
            "============================= test session starts ==============================
collected 3 items

tests/test_a.py::test_one PASSED                                         [ 33%]
tests/test_a.py::test_two PASSED                                         [ 66%]
tests/test_a.py::test_three FAILED                                       [100%]

=================================== FAILURES ===================================
E   boom
========================= 1 failed, 2 passed in 0.05s ==========================
";
        let r = Run {
            code: 1,
            stdout,
            stderr: "",
        };
        let out = filter(&r).unwrap().stdout;
        assert!(out.contains("... 2 passed-test lines omitted ..."));
        assert!(out.contains("tests/test_a.py::test_three FAILED"));
        assert!(out.contains("E   boom"));
    }

    #[test]
    fn a_line_of_dots_that_is_not_progress_is_kept() {
        let stdout =
            "============================= test session starts ==============================
collected 1 item

....
========================= 1 passed in 0.01s ==========================
";
        let r = Run {
            code: 0,
            stdout,
            stderr: "",
        };
        assert!(filter(&r).unwrap().stdout.contains("\n....\n"));
    }

    #[test]
    fn what_is_not_a_test_report_is_not_filtered() {
        let usage = run(4, "", "ERROR: file or directory not found: nope\n");
        assert!(filter(&usage).is_none());
        let other = run(0, "something else\n", "");
        assert!(filter(&other).is_none());
        // No tests collected is exit code 5.
        let none = run(5, FAILING, "");
        assert!(filter(&none).is_none());
    }

    #[test]
    fn plans() {
        let p = |line: &str| crate::tk::plan(&args(line));
        assert_eq!(p("pytest -x tests/"), Plan::Filter(Family::Pytest));
        assert_eq!(p("python -m pytest tests"), Plan::Filter(Family::Pytest));
        assert_eq!(p("python3 -m pytest"), Plan::Filter(Family::Pytest));
        assert_eq!(p("pytest --json-report"), Plan::Filter(Family::Pytest));
        assert_eq!(p("pytest --version"), Plan::Passthrough);
        assert_eq!(p("python script.py"), Plan::Generic);
    }
}
