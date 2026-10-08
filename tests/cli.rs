//! Cheap CLI checks that spawn the built binary; no model or network.

use std::process::Command;

fn llmman() -> Command {
    Command::new(env!("CARGO_BIN_EXE_llmman"))
}

/// A bare `llmman` prints help and exits 0 (clap's default for a missing
/// subcommand is 2, which package validators flag as a broken install).
#[test]
fn bare_invocation_prints_help_and_exits_0() {
    let out = llmman().output().expect("spawn llmman");
    assert!(out.status.success(), "exit status: {}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Usage: llmman"), "stdout: {stdout}");
    assert!(stdout.contains("launch"), "stdout: {stdout}");
}

/// An unknown subcommand is still a usage error.
#[test]
fn unknown_subcommand_exits_2() {
    let out = llmman().arg("bogus").output().expect("spawn llmman");
    assert_eq!(out.status.code(), Some(2));
}

#[cfg(unix)]
#[test]
fn log_follow_filters_new_prompts_and_does_not_limit_them_to_max_count() {
    use std::io::{BufRead, BufReader};
    use std::time::Duration;

    struct Fixture {
        child: std::process::Child,
        _dir: tempfile::TempDir,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prompts.jsonl");
    let entry = |id: &str, model: &str, prompt: &str| llmman::promptlog::Entry {
        id: id.into(),
        model: model.into(),
        prompt: prompt.into(),
        time: "2026-10-06T12:00:00Z".into(),
        route: "/api/chat".into(),
        client: None,
    };
    for value in [
        entry("first", "wanted", "keep first"),
        entry("second", "wanted", "keep second"),
    ] {
        llmman::promptlog::append(&path, &value).unwrap();
    }
    let child = llmman()
        .args([
            "log",
            "-f",
            "--oneline",
            "-n",
            "1",
            "--model",
            "wanted",
            "--grep",
            "keep",
        ])
        .env("LLMMAN_MODELS", dir.path().join("store"))
        .env("LLMMAN_PAGER", "exit 1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut fixture = Fixture { child, _dir: dir };
    let stdout = fixture.child.stdout.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            if tx.send(line.unwrap()).is_err() {
                break;
            }
        }
    });
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(10)).unwrap(),
        "second keep second"
    );
    for value in [
        entry("wrong-model", "other", "keep hidden"),
        entry("wrong-text", "wanted", "hidden"),
        entry("third", "wanted", "keep third"),
        entry("fourth", "wanted", "keep fourth"),
    ] {
        llmman::promptlog::append(&path, &value).unwrap();
    }
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(10)).unwrap(),
        "third keep third"
    );
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(10)).unwrap(),
        "fourth keep fourth"
    );
    unsafe {
        libc::kill(fixture.child.id() as i32, libc::SIGINT);
    }
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(status) = fixture.child.try_wait().unwrap() {
            assert!(status.success(), "{status}");
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "follow did not exit on Ctrl-C"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    reader.join().unwrap();
}

/// A launch whose agent is not installed fails on that, before it starts
/// a daemon or pulls the model: nothing listens at LLMMAN_HOST, so any
/// attempt to reach one would be a different error, and PATH is empty.
#[test]
fn launch_reports_a_missing_agent_before_starting_or_pulling_anything() {
    let home = std::env::temp_dir().join(format!("llmman-cli-launch-{}", std::process::id()));
    let empty_path = home.join("bin");
    std::fs::create_dir_all(&empty_path).unwrap();
    let out = llmman()
        .args(["launch", "claude", "--model", "ai/smollm2"])
        .env("PATH", &empty_path)
        .env("HOME", &home)
        .env("LLMMAN_HOST", "127.0.0.1:9")
        .env_remove("LLMMAN_API_KEY")
        .output()
        .expect("spawn llmman");
    let _ = std::fs::remove_dir_all(&home);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "stderr: {stderr}");
    assert!(
        stderr.contains("claude is not installed"),
        "stderr: {stderr}"
    );
    assert!(!stderr.contains("pull"), "stderr: {stderr}");
}
