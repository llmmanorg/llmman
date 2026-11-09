//! Tests of the engine across every filter family: the guarantees that hold
//! whichever filter handles a command.

use super::*;

/// A command line as the words `plan` is given.
pub(crate) fn words(line: &str) -> Vec<String> {
    line.split_whitespace().map(String::from).collect()
}

/// What a filter is given for a finished command.
pub(crate) fn run<'a>(code: i32, stdout: &'a str, stderr: &'a str) -> Run<'a> {
    Run {
        code,
        stdout,
        stderr,
    }
}

#[cfg(unix)]
fn os(argv: &[&str]) -> Vec<OsString> {
    argv.iter().map(OsString::from).collect()
}

/// Runs `argv` as `plan` and returns the exit code and both streams.
#[cfg(unix)]
fn execute(argv: &[&str], plan: Plan) -> (i32, String, String) {
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let code = run_with(&os(argv), plan, &mut out, &mut err);
    (
        code,
        String::from_utf8(out).unwrap(),
        String::from_utf8(err).unwrap(),
    )
}

const FAMILIES: &[Family] = &[
    Family::GitStatus,
    Family::GitDiff,
    Family::GitLog,
    Family::GitCommit,
    Family::GitPush,
    Family::GitPull,
    Family::Cargo,
    Family::CargoTest,
    Family::Go,
    Family::GoTest,
    Family::Pytest,
    Family::JsTest,
    Family::Search,
    Family::DockerPs,
];

#[cfg(unix)]
mod processes {
    use super::*;

    #[test]
    fn the_childs_exit_status_comes_back_unchanged() {
        for plan in [
            Plan::Generic,
            Plan::Filter(Family::Cargo),
            Plan::Passthrough,
        ] {
            for code in [0, 1, 7, 42, 255] {
                let script = format!("exit {code}");
                let (got, _, _) = execute(&["sh", "-c", &script], plan);
                assert_eq!(got, code, "{plan:?}");
            }
        }
    }

    #[test]
    fn a_signal_is_128_plus_its_number() {
        for plan in [Plan::Generic, Plan::Passthrough] {
            let (code, _, _) = execute(&["sh", "-c", "kill -TERM $$"], plan);
            assert_eq!(code, 128 + 15, "{plan:?}");
            let (code, _, _) = execute(&["sh", "-c", "kill -KILL $$"], plan);
            assert_eq!(code, 128 + 9, "{plan:?}");
        }
    }

    #[test]
    fn a_program_that_is_missing_or_cannot_run_gets_the_shells_codes() {
        let (code, _, _) = execute(&["/no/such/program-llmman-tk"], Plan::Generic);
        assert_eq!(code, 127);
        let (code, _, _) = execute(&["/no/such/program-llmman-tk"], Plan::Passthrough);
        assert_eq!(code, 127);
        // A directory is not something that runs.
        let (code, _, _) = execute(&["/"], Plan::Generic);
        assert_eq!(code, 126);
    }

    #[test]
    fn both_streams_are_cleaned_separately_and_the_status_survives() {
        let script = "printf 'a\\na\\na\\na\\n'; printf 'oops\\n' >&2; exit 3";
        let (code, out, err) = execute(&["sh", "-c", script], Plan::Generic);
        assert_eq!(
            (code, out.as_str(), err.as_str()),
            (3, "a (x4)\n", "oops\n")
        );
    }

    #[test]
    fn escape_sequences_and_progress_redraws_do_not_reach_the_reader() {
        let script = "printf '\\033[32mok\\033[0m\\n'; printf '10%%\\r50%%\\r100%%\\n'";
        let (_, out, _) = execute(&["sh", "-c", script], Plan::Generic);
        assert_eq!(out, "ok\n100%\n");
    }

    #[test]
    fn a_failing_command_keeps_its_error_text_through_a_filter() {
        let script =
            "echo 'error: could not compile `x`' >&2; echo 'caused by: nothing' >&2; exit 101";
        let (code, _, err) = execute(&["sh", "-c", script], Plan::Filter(Family::Cargo));
        assert_eq!(code, 101);
        assert!(err.contains("error: could not compile `x`"), "{err}");
        assert!(err.contains("caused by: nothing"), "{err}");
    }

    /// A process the command leaves running keeps the pipes open: the
    /// output is shown once the grace period is over, not when it exits,
    /// with a note that the capture may be incomplete.
    #[test]
    fn a_process_left_running_does_not_hold_the_output_back() {
        let start = std::time::Instant::now();
        let (code, out, err) = execute(&["sh", "-c", "sleep 5 & echo done"], Plan::Generic);
        assert_eq!((code, out.as_str()), (0, "done\n"));
        assert!(err.contains("still holds its output open"), "{err}");
        assert!(start.elapsed() < std::time::Duration::from_secs(4));
    }

    /// A noisy command with far more output than either side of a pipe holds
    /// finishes, and what is shown is bounded with a marker.
    #[test]
    fn a_huge_output_is_read_without_stalling_and_cut() {
        let script = "yes 'a line that differs only by its number' | head -c 20000000 | nl -ba; yes e | head -c 3000000 >&2";
        let (code, out, err) = execute(&["sh", "-c", script], Plan::Generic);
        assert_eq!(code, 0);
        assert!(out.lines().count() < 700, "{} lines", out.lines().count());
        assert!(out.contains(" lines omitted ..."));
        assert!(err.contains("(x"), "{err}");
        // The tail is the real end of the output.
        assert!(out.lines().last().unwrap().contains("a line that differs"));
    }
}

#[test]
fn memory_for_a_stream_is_bounded_and_the_cut_is_marked() {
    let mut kept = Bounded::default();
    let chunk = vec![b'x'; 1 << 20];
    for _ in 0..10 {
        kept.push(&chunk);
    }
    assert_eq!(kept.head.len(), KEEP_HEAD);
    assert_eq!(kept.tail.len(), KEEP_TAIL);
    assert_eq!(kept.dropped, (2 << 20) as u64);
    assert!(kept.into_text().contains("... 2097152 bytes omitted ..."));

    let mut small = Bounded::default();
    small.push(b"hello ");
    small.push(b"world");
    assert_eq!(small.into_text(), "hello world");
}

#[test]
fn off_and_a_person_at_the_terminal_leave_everything_untouched() {
    let git = words("git status");
    assert_eq!(
        effective_plan(&git, false, false),
        Plan::Filter(Family::GitStatus)
    );
    assert_eq!(effective_plan(&git, true, false), Plan::Passthrough);
    assert_eq!(effective_plan(&git, false, true), Plan::Passthrough);
    assert_eq!(
        effective_plan(&words("make test"), true, false),
        Plan::Passthrough
    );
    assert!(is_off(Some("off")));
    assert!(is_off(Some(" OFF ")));
    assert!(!is_off(Some("on")));
    assert!(!is_off(Some("")));
    assert!(!is_off(None));
}

#[test]
fn commands_that_must_be_read_exactly_or_are_interactive_run_untouched() {
    for line in [
        "cat Cargo.toml",
        "/bin/cat Cargo.toml",
        "head -n 5 src/main.rs",
        "tail -f log",
        "sed -n 1,5p f",
        "jq . data.json",
        "echo hi",
        "curl https://example.com",
        "vim notes.txt",
        "ssh host",
        "git rev-parse --show-toplevel",
        "git show HEAD:README.md",
        "git cat-file -p HEAD",
        "git ls-files",
        "git status --porcelain",
        "git status -z",
        "git status -bz",
        "git diff --name-only",
        "git log --format=%H",
        "git commit",
        "gh repo view --json description",
        "python script.py --json",
        "grep -nz foo f",
        "git rebase -i HEAD~3",
        "cargo build --message-format=json",
        "cargo test -- --format=json",
        "cargo run",
        "go test -json ./...",
        "go run .",
        "pytest --version",
        "npm run dev",
        "npm test -- --watch",
        "rg --json foo",
        "grep -rl foo .",
        "grep -Z foo f",
        "docker ps --format {{.ID}}",
        "docker ps -q",
        "docker run -it ubuntu bash",
        "docker logs -f web",
        "git --help",
        "cargo --version",
    ] {
        assert_eq!(plan(&words(line)), Plan::Passthrough, "{line}");
    }
    assert_eq!(plan(&[]), Plan::Passthrough);
}

#[test]
fn any_other_command_gets_the_generic_cleanup_and_is_never_a_filtered_family() {
    for line in [
        "make test",
        "ls -la",
        "tsc --noEmit",
        "python script.py",
        "terraform plan",
    ] {
        assert_eq!(plan(&words(line)), Plan::Generic, "{line}");
        assert!(!is_filtered(&words(line)), "{line}");
    }
}

/// Safety rule: a command that failed keeps its error text, whichever
/// filter looked at it.
#[test]
fn no_family_drops_the_error_text_of_a_failed_command() {
    let stderr = "error: the thing broke\n  --> where.rs:1:1\nhelp: try again\n";
    let stdout = "fatal: could not do it\nFAIL something\n";
    for &family in FAMILIES {
        for code in [1, 2, 101, 128] {
            let shown = render(Plan::Filter(family), code, stdout, stderr);
            for expected in stderr.lines() {
                assert!(
                    shown.stderr.lines().any(|l| l == expected),
                    "{family:?} code {code} lost {expected:?}: {:?}",
                    shown.stderr
                );
            }
            for expected in stdout.lines() {
                assert!(
                    shown.stdout.lines().any(|l| l == expected),
                    "{family:?} code {code} lost {expected:?}: {:?}",
                    shown.stdout
                );
            }
        }
    }
}

/// Safety rule: output of a shape a filter does not know comes back, never
/// as nothing.
#[test]
fn no_family_turns_output_it_does_not_recognize_into_nothing() {
    let odd = "something entirely unexpected\nsecond line with a: colon\n";
    for &family in FAMILIES {
        for code in [0, 1] {
            let shown = render(Plan::Filter(family), code, odd, odd);
            assert_eq!(shown.stdout, odd, "{family:?} code {code}");
            assert_eq!(shown.stderr, odd, "{family:?} code {code}");
        }
    }
}

/// A filter that wrongly returns nothing for a stream that had text is
/// overridden.
#[test]
fn an_empty_result_for_a_stream_with_text_is_replaced_by_the_text() {
    // The JS filter drops banner and passing lines: all of this is that.
    let shown = render(Plan::Filter(Family::JsTest), 0, "> x@1 test\n> jest\n", "");
    assert_eq!(shown.stdout, "> x@1 test\n> jest\n");
}

#[test]
fn escape_sequences_are_removed_before_a_filter_looks_at_the_text() {
    let status = "\x1b[1mOn branch main\x1b[0m\nnothing to commit, working tree clean\n";
    let shown = render(Plan::Filter(Family::GitStatus), 0, status, "");
    assert_eq!(
        shown.stdout,
        "On branch main\nnothing to commit, working tree clean\n"
    );
}

/// Backs the claim in docs/optimizers.md that noisy output gets smaller,
/// without putting a number in the docs.
#[test]
fn noisy_fixtures_shrink() {
    let passing: String = (0..300)
        .map(|i| format!("test module::case_{i} ... ok\n"))
        .collect();
    let cargo_stdout = format!(
        "\nrunning 300 tests\n{passing}\ntest result: ok. 300 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.20s\n\n"
    );
    let cargo_stderr: String = (0..80)
        .map(|i| format!("   Compiling crate{i} v0.{i}.0\n"))
        .collect::<String>()
        + "    Finished `test` profile [unoptimized + debuginfo] target(s) in 41.00s\n";

    let pytest_stdout = {
        let mut s = String::from(
            "============================= test session starts ==============================\nplatform linux -- Python 3.12.1, pytest-8.0.0, pluggy-1.4.0\nrootdir: /work/app\nplugins: cov-4.1.0\ncollected 400 items\n\n",
        );
        for i in 0..40 {
            s.push_str(&format!(
                "tests/test_{i}.py {}  [{:>3}%]\n",
                ".".repeat(10),
                (i + 1) * 100 / 40
            ));
        }
        s.push_str("\n========================= 400 passed in 3.21s ==========================\n");
        s
    };

    let go_stdout: String = (0..40)
        .map(|i| format!("?   \texample.com/app/pkg{i}\t[no test files]\n"))
        .collect::<String>()
        + "ok  \texample.com/app/core\t0.4s\n";

    let status = "On branch main\nYour branch is up to date with 'origin/main'.\n\nChanges not staged for commit:\n  (use \"git add <file>...\" to update what will be committed)\n  (use \"git restore <file>...\" to discard changes in working directory)\n\tmodified:   src/a.rs\n\nno changes added to commit (use \"git add\" and/or \"git commit -a\")\n";

    let repeated = "retrying connection...\n".repeat(500);

    let fixtures: Vec<(Plan, &str, String, String)> = vec![
        (
            Plan::Filter(Family::CargoTest),
            "cargo test",
            cargo_stdout,
            cargo_stderr,
        ),
        (
            Plan::Filter(Family::Pytest),
            "pytest",
            pytest_stdout,
            String::new(),
        ),
        (
            Plan::Filter(Family::GoTest),
            "go test ./...",
            go_stdout,
            String::new(),
        ),
        (
            Plan::Filter(Family::GitStatus),
            "git status",
            status.to_string(),
            String::new(),
        ),
        (Plan::Generic, "some-tool", repeated, String::new()),
    ];
    for (plan, line, stdout, stderr) in fixtures {
        let shown = render(plan, 0, &stdout, &stderr);
        let (before, after) = (
            stdout.len() + stderr.len(),
            shown.stdout.len() + shown.stderr.len(),
        );
        assert!(after < before, "{line}: {after} bytes from {before}");
        assert!(after > 0, "{line}");
    }
    // The largest ones lose most of their bytes.
    let big = render(
        Plan::Filter(Family::CargoTest),
        0,
        &format!("running 300 tests\n{passing}\ntest result: ok. 300 passed; 0 failed;\n"),
        "",
    );
    assert!(big.stdout.len() * 4 < passing.len(), "{}", big.stdout);
}
