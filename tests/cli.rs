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

/// The `METHOD /path` of every request a fake server saw.
type Seen = std::sync::Arc<std::sync::Mutex<Vec<String>>>;

/// A stand-in for a real Ollama: `/api/version` answers a bare version,
/// anything llmman-only is a 404, and `/api/pull` streams one line per
/// layer. Returns its address and the `METHOD /path` of every request.
fn fake_ollama() -> (String, Seen) {
    fake_server(r#"{"version":"0.40.17"}"#.to_string())
}

/// [`fake_ollama`] with `version` as the body of `/api/version`.
fn fake_server(version: String) -> (String, Seen) {
    use std::io::{BufRead, Read, Write};

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let log = seen.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let log = log.clone();
            let version = version.clone();
            std::thread::spawn(move || {
                let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
                    return;
                }
                let mut length = 0;
                loop {
                    let mut header = String::new();
                    reader.read_line(&mut header).unwrap();
                    if header.trim().is_empty() {
                        break;
                    }
                    if let Some(value) = header.to_ascii_lowercase().strip_prefix("content-length:")
                    {
                        length = value.trim().parse().unwrap();
                    }
                }
                let mut body = String::new();
                reader
                    .by_ref()
                    .take(length)
                    .read_to_string(&mut body)
                    .unwrap();
                let mut parts = request_line.split_whitespace();
                let (method, path) = (parts.next().unwrap(), parts.next().unwrap());
                log.lock().unwrap().push(format!("{method} {path} {body}"));
                let (status, body) = match (method, path) {
                    ("POST", "/api/show") => ("200 OK", "{\"capabilities\":[\"completion\"]}\n".to_string()),
                    ("POST", "/api/chat") => (
                        "200 OK",
                        [
                            r#"{"message":{"role":"assistant","content":"hi"},"done":false}"#,
                            r#"{"message":{"role":"assistant","content":""},"done":true}"#,
                        ]
                        .join("\n")
                            + "\n",
                    ),
                    ("GET", "/api/version") => ("200 OK", format!("{version}\n")),
                    ("POST", "/api/pull") => (
                        "200 OK",
                        [
                            r#"{"status":"pulling manifest"}"#,
                            r#"{"status":"pulling aaaa","digest":"sha256:aaaa","total":100,"completed":0}"#,
                            r#"{"status":"pulling bbbb","digest":"sha256:bbbb","total":50,"completed":50}"#,
                            r#"{"status":"pulling aaaa","digest":"sha256:aaaa","total":100,"completed":100}"#,
                            r#"{"status":"verifying sha256 digest"}"#,
                            r#"{"status":"writing manifest"}"#,
                            r#"{"status":"success"}"#,
                        ]
                        .join("\n")
                            + "\n",
                    ),
                    _ => ("404 Not Found", "404 page not found\n".to_string()),
                };
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
            });
        }
    });
    (addr, seen)
}

/// `llmman` against the server at `addr`, keeping everything it would write
/// under `home`: a daemon it started would put `serve.log` there.
fn llmman_at(addr: &str, home: &std::path::Path) -> Command {
    let mut cmd = llmman();
    cmd.env("LLMMAN_HOST", addr)
        .env("LLMMAN_MODELS", home.join("store"))
        .env("HOME", home)
        .env("USERPROFILE", home);
    cmd
}

/// Whether the CLI started a daemon: `serve.log` beside the store, which
/// `llmman_at` points at `home`, is the first thing a spawn creates.
fn spawned_a_daemon(home: &std::path::Path) -> bool {
    home.join("serve.log").exists()
}

/// `pull` against the server at `addr` reuses it as is: it relays the layers,
/// sends nothing but the pull, and starts no daemon.
fn assert_pull_reuses(addr: &str, seen: &Seen, what: &str) {
    let home = tempfile::tempdir().unwrap();
    let out = llmman_at(addr, home.path())
        .args(["pull", "all-minilm:22m"])
        .output()
        .expect("spawn llmman");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{what}: {stdout}\n{stderr}");
    assert_eq!(
        stdout,
        "pulling manifest\nverifying sha256 digest\nwriting manifest\nsuccess\nPulled all-minilm:22m\n",
        "{what}"
    );
    assert!(!spawned_a_daemon(home.path()), "{what}");
    assert_eq!(
        *seen.lock().unwrap(),
        ["POST /api/pull {\"model\":\"all-minilm:22m\"}"],
        "{what}"
    );
}

#[test]
fn pull_reuses_an_ollama_at_llmman_host() {
    let (addr, seen) = fake_ollama();
    assert_pull_reuses(&addr, &seen, "an Ollama");
}

/// Stands for the daemon a fake server claims to be: dies if the CLI
/// signals the pid it reports.
#[cfg(unix)]
struct Bystander(std::process::Child);

#[cfg(unix)]
impl Bystander {
    fn start() -> Bystander {
        use std::os::unix::process::CommandExt;
        let mut sleeper = Command::new("sleep");
        sleeper.arg("60").process_group(0);
        Bystander(sleeper.spawn().expect("spawn sleep"))
    }

    fn pid(&self) -> u32 {
        self.0.id()
    }

    fn alive(&mut self) -> bool {
        self.0.try_wait().unwrap().is_none()
    }
}

#[cfg(unix)]
impl Drop for Bystander {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// A llmman daemon of another version, or whose exe is gone, is reused like
/// an Ollama: not signalled, and no second daemon is started.
#[cfg(unix)]
#[test]
fn pull_reuses_whatever_daemon_answers_at_llmman_host() {
    let own_exe = std::env::current_exe().unwrap();
    for (what, exe, version) in [
        ("an old version", own_exe.to_str().unwrap(), "0.0.1-old"),
        (
            "a missing exe",
            "/nonexistent/llmman",
            env!("CARGO_PKG_VERSION"),
        ),
    ] {
        let mut daemon = Bystander::start();
        let identity = serde_json::json!({"version": version, "exe": exe, "pid": daemon.pid()});
        let (addr, seen) = fake_server(identity.to_string());
        assert_pull_reuses(&addr, &seen, what);
        assert!(daemon.alive(), "{what}: the daemon was signalled");
    }
}

/// `run` hands an Ollama the model name as typed: llmman's own bare-name
/// default (`docker.io/ai/...`) means nothing to it.
#[test]
fn run_uses_the_model_name_as_typed_with_an_ollama_at_llmman_host() {
    let home = tempfile::tempdir().unwrap();
    let (addr, seen) = fake_ollama();
    let out = llmman_at(&addr, home.path())
        .args(["run", "smollm2:135m", "say hi"])
        .stdin(std::process::Stdio::null())
        .output()
        .expect("spawn llmman");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stdout}\n{stderr}");
    assert!(stdout.contains("hi"), "{stdout}");
    assert!(!spawned_a_daemon(home.path()));
    let seen = seen.lock().unwrap();
    for request in seen.iter().filter(|r| r.starts_with("POST /api/")) {
        assert!(!request.contains("docker.io/ai"), "{request}");
    }
    assert!(
        seen.iter()
            .any(|r| r.starts_with("POST /api/chat ") && r.contains("\"smollm2:135m\"")),
        "{seen:?}"
    );
}

/// A loopback address nothing listens on.
fn closed_addr() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().to_string()
}

/// With `LLMMAN_NO_AUTOSTART` set, client commands fail when no server
/// answers instead of starting a daemon of their own.
#[test]
fn no_autostart_makes_client_commands_fail_without_spawning_a_daemon() {
    let addr = closed_addr();
    for args in [
        &["pull", "all-minilm:22m"][..],
        &["push", "all-minilm:22m"],
        &["list", "--provider", "openai"],
        &["providers"],
    ] {
        let home = tempfile::tempdir().unwrap();
        let out = llmman_at(&addr, home.path())
            .args(args)
            .env("LLMMAN_NO_AUTOSTART", "1")
            .output()
            .expect("spawn llmman");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "{args:?}: {stderr}");
        assert!(
            stderr.contains("LLMMAN_NO_AUTOSTART") && stderr.contains(&addr),
            "{args:?}: {stderr}"
        );
        assert!(!spawned_a_daemon(home.path()), "{args:?}");
        assert!(
            std::net::TcpStream::connect(&addr).is_err(),
            "{args:?}: something started listening"
        );
    }
}

/// `LLMMAN_NO_AUTOSTART` only forbids starting: a running server is used.
#[test]
fn no_autostart_still_uses_a_running_server() {
    let home = tempfile::tempdir().unwrap();
    let (addr, seen) = fake_ollama();
    let out = llmman_at(&addr, home.path())
        .args(["pull", "all-minilm:22m"])
        .env("LLMMAN_NO_AUTOSTART", "1")
        .output()
        .expect("spawn llmman");
    assert!(out.status.success(), "{out:?}");
    assert_eq!(
        *seen.lock().unwrap(),
        ["POST /api/pull {\"model\":\"all-minilm:22m\"}"]
    );
}

#[test]
fn llama_cpp_assets_is_listed_in_the_help() {
    let out = llmman().arg("--help").output().expect("spawn llmman");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout
            .lines()
            .any(|l| l.trim_start().starts_with("llama-cpp-assets ")),
        "{stdout}"
    );
    let out = llmman()
        .args(["llama-cpp-assets", "--help"])
        .output()
        .expect("spawn llmman");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--llama-cpp-version") && stdout.contains("--json"),
        "{stdout}"
    );
}

/// Repo and pin errors come before any network access.
#[test]
fn llama_cpp_assets_rejects_a_bad_repo_and_latest_from_a_mirror() {
    for (repo, args, needle) in [
        ("not-a-repo", &[][..], "<owner>/<repo>"),
        ("a/b/c", &["--json"][..], "<owner>/<repo>"),
        (
            "docker/sbx-releases",
            &["--llama-cpp-version", "latest"][..],
            "mirrors are pinned",
        ),
    ] {
        let out = llmman()
            .arg("llama-cpp-assets")
            .args(args)
            .env("LLMMAN_LLAMA_CPP_REPO", repo)
            .output()
            .expect("spawn llmman");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "{repo}: {stderr}");
        assert!(stderr.contains(needle), "{repo}: {stderr}");
        assert!(out.stdout.is_empty(), "{repo}");
    }
}

#[cfg(unix)]
#[test]
fn clients_reach_a_serve_on_port_0_while_it_holds_its_lock() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let home = tempfile::tempdir().unwrap();
    let data_dir = home.path().join(".local/share/llmman");
    let port_file = data_dir.join("serve-127.0.0.1-0.port");
    let llmman_here = || {
        let mut cmd = llmman();
        cmd.env_clear()
            .env("HOME", home.path())
            .env("PATH", "/usr/bin:/bin")
            .env("LLMMAN_HOST", "127.0.0.1:0");
        cmd
    };
    let output = |cmd: &mut Command| {
        let out = cmd.output().unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        (out.status.success(), stdout, stderr)
    };
    let ps = || output(llmman_here().arg("ps"));
    let reachable = || {
        let (ok, stdout, stderr) = ps();
        assert!(ok && stdout.contains("NAME"), "{stdout}{stderr}");
    };
    let unreachable = || {
        let (ok, _, stderr) = ps();
        assert!(!ok && stderr.contains("not running"), "{stderr}");
    };
    let start = || serve_until_shell_line(home.path(), "127.0.0.1:0", None);

    let (mut daemon, _, _) = start();
    reachable();

    // A second daemon for the same LLMMAN_HOST refuses to start. Waited on
    // with a deadline, since one that started would serve until killed.
    let mut second = Daemon(
        llmman_here()
            .args(["serve", "--runtime", "path"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let status = loop {
        if let Some(status) = second.0.try_wait().unwrap() {
            break status;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "a second daemon started"
        );
        std::thread::sleep(std::time::Duration::from_millis(100));
    };
    let mut stderr = String::new();
    std::io::Read::read_to_string(&mut second.0.stderr.take().unwrap(), &mut stderr).unwrap();
    assert!(
        !status.success() && stderr.contains("another llmman serve"),
        "{stderr}"
    );
    reachable();

    // SIGTERM, not Daemon's SIGKILL: a graceful stop removes the port file.
    let status = Command::new("kill")
        .args(["-TERM", &daemon.0.id().to_string()])
        .status()
        .unwrap();
    assert!(status.success());
    daemon.0.wait().unwrap();
    assert!(!port_file.exists());
    unreachable();

    // A crash leaves the port file, and something else takes the port. The
    // squatter closes each connection, so a client that trusted the file
    // fails instead of waiting on it.
    let (daemon, log, _) = start();
    let port: u16 = log
        .lines()
        .find_map(|l| l.strip_prefix("llmman serve listening on http://127.0.0.1:"))
        .unwrap_or_else(|| panic!("no listening line in {log}"))
        .parse()
        .unwrap();
    drop(daemon);
    assert!(port_file.exists());
    let squatter = std::net::TcpListener::bind(("127.0.0.1", port)).unwrap();
    let connections = std::sync::Arc::new(AtomicUsize::new(0));
    let counted = connections.clone();
    std::thread::spawn(move || {
        for stream in squatter.incoming() {
            counted.fetch_add(1, Ordering::SeqCst);
            drop(stream);
        }
    });
    unreachable();
    assert_eq!(connections.load(Ordering::SeqCst), 0);

    // A directory in the port file's place fails startup at once, instead of
    // leaving the client to wait out its timeout.
    std::fs::remove_file(&port_file).unwrap();
    std::fs::create_dir(&port_file).unwrap();
    let started = std::time::Instant::now();
    let (ok, _, stderr) = output(llmman_here().args(["serve", "--detach", "--runtime", "path"]));
    assert!(
        !ok && stderr.contains(&port_file.display().to_string()),
        "{stderr}"
    );
    assert!(started.elapsed() < std::time::Duration::from_secs(30));
}
