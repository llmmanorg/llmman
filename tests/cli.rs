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

/// A pinned vLLM release and an explicit vLLM image conflict.
#[test]
fn serve_vllm_image_conflicts_with_vllm_version() {
    let out = llmman()
        .args(["serve", "--vllm-image", "dhi.io/vllm-openai:0.32"])
        .args(["--vllm-version", "v0.28.0"])
        .output()
        .expect("spawn llmman");
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("cannot be used with"), "stderr: {stderr}");
    assert!(stderr.contains("--vllm-version"), "stderr: {stderr}");
}

#[test]
fn serve_detach_conflicts_with_non_daemon_modes() {
    for args in [
        vec!["serve", "--detach", "--pull-only"],
        vec!["serve", "model", "--detach", "--port", "12345"],
    ] {
        let out = llmman().args(args).output().expect("spawn llmman");
        assert_eq!(out.status.code(), Some(2));
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("cannot be used with"),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
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

/// Runs `llmman launch` with an empty PATH and nothing at LLMMAN_HOST,
/// so any attempt to start a daemon or reach one would be a different
/// error from the one under test. `tag` keeps parallel tests' homes apart.
/// Returns whether it succeeded, and its stderr.
fn launch_with_empty_path(tag: &str, args: &[&str]) -> (bool, String) {
    let home = std::env::temp_dir().join(format!("llmman-cli-{tag}-{}", std::process::id()));
    let empty_path = home.join("bin");
    std::fs::create_dir_all(&empty_path).unwrap();
    let out = llmman()
        .arg("launch")
        .args(args)
        .env("PATH", &empty_path)
        .env("HOME", &home)
        .env("LLMMAN_HOST", "127.0.0.1:9")
        .env_remove("LLMMAN_API_KEY")
        .output()
        .expect("spawn llmman");
    let _ = std::fs::remove_dir_all(&home);
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// A launch whose agent is not installed fails on that, before it starts
/// a daemon or pulls the model.
#[test]
fn launch_reports_a_missing_agent_before_starting_or_pulling_anything() {
    let (ok, stderr) = launch_with_empty_path("launch", &["claude", "--model", "ai/smollm2"]);
    assert!(!ok, "stderr: {stderr}");
    assert!(
        stderr.contains("claude is not installed"),
        "stderr: {stderr}"
    );
    assert!(!stderr.contains("pull"), "stderr: {stderr}");
}

/// claude yields to a `--model` after `--`, so a `--variant` checked
/// against llmman's `--model` is refused before anything starts.
#[test]
fn launch_claude_refuses_a_variant_with_a_forwarded_model() {
    let (ok, stderr) = launch_with_empty_path(
        "variant",
        &[
            "claude",
            "--model",
            "ai/smollm2",
            "--variant",
            "high",
            "--",
            "--model",
            "other/model",
        ],
    );
    assert!(!ok, "stderr: {stderr}");
    assert!(
        stderr.contains("--variant needs the model as llmman's --model, not after --"),
        "stderr: {stderr}"
    );
}

#[cfg(unix)]
struct Daemon(std::process::Child);

#[cfg(unix)]
impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Starts `llmman serve` in a cleared environment. Returns the daemon, its
/// stderr up to the shell line, and the rest of its stderr.
#[cfg(unix)]
fn serve_until_shell_line(
    home: &std::path::Path,
    host: &str,
    shell: Option<&str>,
) -> (Daemon, String, std::sync::mpsc::Receiver<String>) {
    use std::io::{BufRead, BufReader};
    use std::process::Stdio;
    use std::time::Duration;

    let mut cmd = llmman();
    cmd.args(["serve", "--runtime", "path"])
        .env_clear()
        .env("HOME", home)
        .env("PATH", "/usr/bin:/bin")
        .env("LLMMAN_HOST", host)
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    if let Some(value) = shell {
        cmd.env("LLMMAN_SHELL", value);
    }
    let mut daemon = Daemon(cmd.spawn().unwrap());
    let stderr = daemon.0.stderr.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    // Drains to EOF, so the daemon never writes to a closed pipe.
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            let _ = tx.send(line);
        }
    });
    let mut log = Vec::new();
    while !log.iter().any(|l: &String| l.contains("[llmman] shell:")) {
        match rx.recv_timeout(Duration::from_secs(30)) {
            Ok(line) => log.push(line),
            Err(_) => panic!("no shell line in {log:#?}"),
        }
    }
    (daemon, log.join("\n"), rx)
}

#[cfg(unix)]
fn shell_upgrade(shell_url: &str, protocol: Option<&str>) -> u16 {
    let mut req = reqwest::blocking::Client::new()
        .get(shell_url)
        .header("connection", "upgrade")
        .header("upgrade", "websocket")
        .header("sec-websocket-version", "13")
        .header("sec-websocket-key", "dGhlIHNhbXBsZSBub25jZQ==");
    if let Some(protocol) = protocol {
        req = req.header("sec-websocket-protocol", protocol);
    }
    req.send().unwrap().status().as_u16()
}

/// The client sends no `Origin` (not a browser). stderr is a pipe, so the
/// token must not appear in it.
#[cfg(unix)]
#[test]
fn a_default_keyless_serve_opens_its_shell_only_with_the_token() {
    use std::os::unix::fs::PermissionsExt as _;

    let home = tempfile::tempdir().unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let token_file = home
        .path()
        .join(format!(".local/share/llmman/shell-127.0.0.1-{port}.token"));
    let host = format!("127.0.0.1:{port}");
    let start = |shell| serve_until_shell_line(home.path(), &host, shell);
    let shell_url = format!("http://127.0.0.1:{port}/llmman/shell");
    let client = reqwest::blocking::Client::new();
    let upgrade = |protocol| shell_upgrade(&shell_url, protocol);

    // A token file the daemon cannot open for writing, from an earlier start.
    let data_dir = token_file.parent().unwrap().to_path_buf();
    let chmod = |path: &std::path::Path, mode| {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap()
    };
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::write(&token_file, "stale").unwrap();
    chmod(&token_file, 0o444);

    let (daemon, log, rest) = start(None);
    assert!(
        log.contains(&format!(
            "[llmman] shell: on (bound to loopback; LLMMAN_SHELL=off removes it), its token is in {}",
            token_file.display()
        )),
        "{log}"
    );
    let mode = std::fs::metadata(&token_file).unwrap().permissions().mode();
    assert_eq!(mode & 0o777, 0o600);
    let token = std::fs::read_to_string(&token_file).unwrap();
    assert_ne!(token, "stale");

    assert_eq!(upgrade(None), 401);
    assert_eq!(upgrade(Some("llmman.shell.wrong")), 401);
    assert_eq!(upgrade(Some(&format!("llmman.shell.{token}"))), 101);
    let status: serde_json::Value = client.get(&shell_url).send().unwrap().json().unwrap();
    assert_eq!(
        status,
        serde_json::json!({ "enabled": true, "token_required": true })
    );

    drop(daemon);
    let stderr = format!("{log}\n{}", rest.iter().collect::<Vec<_>>().join("\n"));
    assert!(!stderr.contains(&token), "{stderr}");

    let (daemon, log, _) = start(Some("off"));
    assert!(
        log.contains("[llmman] shell: off (LLMMAN_SHELL is off)"),
        "{log}"
    );
    assert!(!token_file.exists());
    drop(daemon);

    // Root ignores permissions, so only another user can see a refusal.
    if unsafe { libc::geteuid() } == 0 {
        return;
    }
    chmod(&data_dir, 0o555);
    let (daemon, log, _) = start(None);
    let status: serde_json::Value = client.get(&shell_url).send().unwrap().json().unwrap();
    drop(daemon);
    chmod(&data_dir, 0o755);
    assert!(
        log.contains("[llmman] shell: off (llmman serve could not save its shell token: "),
        "{log}"
    );
    assert_eq!(status["enabled"], false);
    assert!(
        status["reason"]
            .as_str()
            .unwrap()
            .starts_with("llmman serve could not save its shell token: "),
        "{status}"
    );
}

/// The OS picks the port for `LLMMAN_HOST=…:0`; the token file is named by
/// it too.
#[cfg(unix)]
#[test]
fn a_serve_on_port_0_reports_the_port_it_got() {
    let home = tempfile::tempdir().unwrap();
    let (_daemon, log, _) = serve_until_shell_line(home.path(), "127.0.0.1:0", None);
    let port: u16 = log
        .lines()
        .find_map(|l| l.strip_prefix("llmman serve listening on http://127.0.0.1:"))
        .unwrap_or_else(|| panic!("no listening line in {log}"))
        .parse()
        .unwrap();
    assert_ne!(port, 0);
    let token_file = home
        .path()
        .join(format!(".local/share/llmman/shell-127.0.0.1-{port}.token"));
    assert!(
        log.contains(&format!("its token is in {}", token_file.display())),
        "{log}"
    );
    let token = std::fs::read_to_string(&token_file).unwrap();
    let shell_url = format!("http://127.0.0.1:{port}/llmman/shell");
    assert_eq!(
        shell_upgrade(&shell_url, Some(&format!("llmman.shell.{token}"))),
        101
    );
}
