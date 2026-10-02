//! `llmman launch --sandbox`.
//!
//! `sbx` hands the whole launch to `sbx run`, which serves the model
//! itself. The other backends keep launch's flow (local daemon, launcher
//! config) and only change how the integration process starts:
//!
//! - `seatbelt`: the installed binary under `sandbox-exec`, writes
//!   confined to the workspace, temp/cache dirs and the integration's state.
//! - `docker`, `podman`, `apple-container`, `microsandbox`: the
//!   integration from an image, with the workspace and state dirs mounted
//!   at their host paths over a per-integration `HOME`.
//! - `openshell`: the workspace uploaded to a sandbox that is kept for the
//!   changes to be downloaded.
//!
//! Launchers write [`agent_server`], the guest's address for the daemon.

use std::collections::BTreeMap;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use anyhow::Context;
use clap::ValueEnum;

/// Where `--sandbox` runs the integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Sandbox {
    /// Docker Sandboxes (`sbx run`), which also serves the model
    Sbx,
    /// macOS `sandbox-exec`, around the installed integration
    Seatbelt,
    /// A Docker container
    Docker,
    /// A Podman container
    Podman,
    /// Apple's `container` (macOS on Apple silicon)
    AppleContainer,
    /// A microsandbox (`msb`) microVM
    Microsandbox,
    /// An NVIDIA OpenShell sandbox
    Openshell,
}

impl Sandbox {
    fn name(self) -> &'static str {
        match self {
            Sandbox::Sbx => "sbx",
            Sandbox::Seatbelt => "seatbelt",
            Sandbox::Docker => "docker",
            Sandbox::Podman => "podman",
            Sandbox::AppleContainer => "apple-container",
            Sandbox::Microsandbox => "microsandbox",
            Sandbox::Openshell => "openshell",
        }
    }

    fn cli(self) -> &'static str {
        match self {
            Sandbox::Sbx => "sbx",
            Sandbox::Seatbelt => SANDBOX_EXEC,
            Sandbox::Docker => "docker",
            Sandbox::Podman => "podman",
            Sandbox::AppleContainer => "container",
            Sandbox::Microsandbox => "msb",
            Sandbox::Openshell => "openshell",
        }
    }

    fn uses_image(self) -> bool {
        !matches!(self, Sandbox::Sbx | Sandbox::Seatbelt)
    }

    /// The guest's name for the host's loopback, if not its own. On Linux
    /// docker/podman use `--network host`: a bridge can't reach a daemon
    /// bound to 127.0.0.1.
    fn host_alias(self, linux: bool) -> Option<&'static str> {
        match self {
            Sandbox::Sbx | Sandbox::Seatbelt => None,
            Sandbox::Docker => (!linux).then_some("host.docker.internal"),
            Sandbox::Podman => (!linux).then_some("host.containers.internal"),
            Sandbox::AppleContainer => Some(APPLE_HOST_DOMAIN),
            Sandbox::Microsandbox => Some("host.microsandbox.internal"),
            Sandbox::Openshell => Some(OPENSHELL_HOST),
        }
    }
}

const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";

/// Created with `container system dns create --localhost` (needs sudo).
const APPLE_HOST_DOMAIN: &str = "host.container.internal";

/// The gateway's host, reachable from a sandbox once policy allows it.
const OPENSHELL_HOST: &str = "host.openshell.internal";

/// Overrides [`DEFAULT_IMAGES`].
const IMAGE_ENV: &str = "LLMMAN_SANDBOX_IMAGE";

/// Docker Sandboxes' agent images: the agent on `PATH`, `tini` entrypoint.
const TEMPLATES: &str = "docker.io/docker/sandbox-templates";

const DEFAULT_IMAGES: &[(&str, &str)] = &[
    ("claude", "claude-code"),
    ("codex", "codex"),
    ("opencode", "opencode"),
    ("gemini", "gemini"),
    ("docker-agent", "docker-agent"),
];

const SBX_AGENTS: &[(&str, &str)] = &[
    ("claude", "claude"),
    ("codex", "codex"),
    ("copilot", "copilot"),
    ("copilot-cli", "copilot"),
    ("docker-agent", "docker-agent"),
    ("gemini", "gemini"),
    ("opencode", "opencode"),
];

/// Host variables an image-based guest also gets: terminal, locale, and
/// what the state directories are found by.
const PASSTHROUGH_ENV: &[&str] = &[
    "TERM",
    "COLORTERM",
    "LANG",
    "LC_ALL",
    "CLAUDE_CONFIG_DIR",
    "CLINE_DIR",
    "COPILOT_HOME",
    "GH_CONFIG_DIR",
    "GROK_HOME",
    "HERMES_HOME",
    "PI_CODING_AGENT_DIR",
    "PI_CONFIG_DIR",
    "QWEN_HOME",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
];

/// What an integration keeps on this machine and may write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum State {
    Dir(PathBuf),
    /// A file and siblings extending its name (`~/.claude.json.backup`).
    /// Seatbelt only: a bind-mounted file can't be replaced by rename.
    Files(PathBuf),
}

struct Active {
    sandbox: Sandbox,
    integration: String,
    /// The daemon's URL as the guest reaches it.
    server: String,
    workspace: PathBuf,
    home: PathBuf,
    image: Option<String>,
    state: Vec<State>,
}

static ACTIVE: OnceLock<Active> = OnceLock::new();

/// The daemon's URL for the integration.
pub fn agent_server() -> String {
    ACTIVE
        .get()
        .map_or_else(crate::daemon::server, |a| a.server.clone())
}

/// Whether the image's copy of the integration runs, not the host's.
pub fn runs_from_image() -> bool {
    ACTIVE.get().is_some_and(|a| a.sandbox.uses_image())
}

pub fn active() -> bool {
    ACTIVE.get().is_some()
}

/// Checks `sandbox` can run `integration`, before the daemon starts.
/// `configured_by_file`: the launcher writes config under `~`, which
/// OpenShell never sees. `carries_key`: the integration is handed a real
/// key, which OpenShell could only take on its command line.
pub fn prepare(
    sandbox: Sandbox,
    integration: &str,
    state: Vec<State>,
    configured_by_file: bool,
    carries_key: bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(sandbox != Sandbox::Sbx, "--sandbox sbx is not prepared");
    anyhow::ensure!(
        !cfg!(windows),
        "--sandbox {} is not supported on Windows",
        sandbox.name()
    );
    anyhow::ensure!(
        sandbox != Sandbox::Openshell || !configured_by_file,
        "--sandbox openshell cannot run {integration}: llmman configures it through files in \
         your home directory, which an OpenShell sandbox does not receive"
    );
    anyhow::ensure!(
        sandbox != Sandbox::Openshell || !carries_key,
        "--sandbox openshell takes the integration's environment only on its command line, \
         so it cannot be given an API key (--provider, --overflow-provider, LLMMAN_API_KEY)"
    );
    let daemon = crate::daemon::server();
    check_host(sandbox, &daemon)?;
    let image = sandbox
        .uses_image()
        .then(|| image(integration, std::env::var(IMAGE_ENV).ok().as_deref()))
        .transpose()?;
    let server = agent_server_for(&daemon, sandbox.host_alias(cfg!(target_os = "linux")))?;
    let home = dirs::home_dir().context("no home directory")?;
    let cwd = std::env::current_dir().context("current directory")?;
    // A relative state path (`GROK_HOME=state`) is relative to the cwd,
    // which the guest shares; mount destinations must be absolute.
    let state: Vec<State> = state
        .into_iter()
        .map(|s| match s {
            State::Dir(p) => State::Dir(cwd.join(p)),
            State::Files(p) => State::Files(cwd.join(p)),
        })
        .collect();
    // They come from the environment (`CLAUDE_CONFIG_DIR=/`).
    for state in &state {
        if let State::Dir(dir) = state {
            anyhow::ensure!(
                !contains_home(dir, &home),
                "--sandbox would let {integration} write all of {}, which contains your home \
                 directory",
                dir.display()
            );
        }
    }
    ACTIVE
        .set(Active {
            sandbox,
            integration: integration.to_string(),
            server,
            workspace: workspace(&cwd, &home)?,
            home,
            image,
            state,
        })
        .map_err(|_| anyhow::anyhow!("a sandbox is already prepared"))
}

/// Runs the integration in the prepared sandbox; returns its exit code.
/// `env` overlays the inherited environment, later entries winning.
pub fn run(bin: &Path, args: &[String], env: &[(String, String)]) -> anyhow::Result<i32> {
    let active = ACTIVE.get().context("no sandbox prepared")?;
    let filtered;
    let env = if active.sandbox.uses_image() {
        filtered = without_path(env);
        &filtered[..]
    } else {
        env
    };
    match active.sandbox {
        Sandbox::Sbx => anyhow::bail!("--sandbox sbx is not prepared"),
        Sandbox::Seatbelt => run_seatbelt(active, bin, args, env),
        Sandbox::Openshell => run_openshell(active, bin, args, env),
        sandbox => {
            let plan = plan(active, bin, args, env)?;
            let linux = cfg!(target_os = "linux");
            let argv = image_args(
                sandbox,
                &plan,
                interactive(),
                linux,
                &identity(sandbox, linux),
            );
            // `-e NAME` takes the value from here, keeping keys off argv.
            let mut cmd = Command::new(sandbox.cli());
            cmd.args(argv).envs(&plan.env);
            status_code(cmd, sandbox.cli())
        }
    }
}

/// `sbx run <agent>` with the same model flags, in the current directory.
pub fn run_sbx(
    integration: &str,
    model: Option<&str>,
    provider: Option<&str>,
    overflow: Option<(&str, &str)>,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let agent = sbx_agent(integration)?;
    let sbx = crate::find_on_path("sbx").context("sbx is not installed")?;
    let mut cmd = Command::new(sbx);
    cmd.args(sbx_args(agent, model, provider, overflow, extra_args));
    std::process::exit(status_code(cmd, "sbx")?);
}

fn sbx_agent(integration: &str) -> anyhow::Result<&'static str> {
    let name = integration.to_lowercase();
    SBX_AGENTS
        .iter()
        .find(|(id, _)| *id == name)
        .map(|(_, agent)| *agent)
        .ok_or_else(|| {
            let supported: Vec<&str> = SBX_AGENTS.iter().map(|(id, _)| *id).collect();
            anyhow::anyhow!(
                "sbx has no {name} agent; --sandbox sbx runs {}",
                supported.join(", ")
            )
        })
}

fn sbx_args(
    agent: &str,
    model: Option<&str>,
    provider: Option<&str>,
    overflow: Option<(&str, &str)>,
    extra_args: &[String],
) -> Vec<String> {
    let mut args = vec!["run".to_string(), agent.to_string()];
    let (overflow_provider, overflow_model) = overflow.unzip();
    for (name, value) in [
        ("--model", model),
        ("--provider", provider),
        ("--overflow-provider", overflow_provider),
        ("--overflow-model", overflow_model),
    ] {
        if let Some(value) = value.map(str::trim).filter(|v| !v.is_empty()) {
            args.extend([name.to_string(), value.to_string()]);
        }
    }
    if !extra_args.is_empty() {
        args.push("--".to_string());
        args.extend_from_slice(extra_args);
    }
    args
}

// ---------------------------------------------------------------------------
// Pre-flight
// ---------------------------------------------------------------------------

fn check_host(sandbox: Sandbox, daemon: &str) -> anyhow::Result<()> {
    match sandbox {
        Sandbox::Seatbelt => {
            anyhow::ensure!(cfg!(target_os = "macos"), "--sandbox seatbelt needs macOS");
            anyhow::ensure!(
                Path::new(SANDBOX_EXEC).is_file(),
                "--sandbox seatbelt needs {SANDBOX_EXEC}, which is missing"
            );
            return Ok(());
        }
        Sandbox::AppleContainer => anyhow::ensure!(
            cfg!(all(target_os = "macos", target_arch = "aarch64")),
            "--sandbox apple-container needs macOS on Apple silicon"
        ),
        _ => {}
    }
    let cli = sandbox.cli();
    anyhow::ensure!(
        crate::find_on_path(cli).is_some(),
        "{cli} is not installed; --sandbox {} runs the integration with it",
        sandbox.name()
    );
    let daemon_local = reqwest::Url::parse(daemon).is_ok_and(|u| is_loopback(&u));
    match sandbox {
        // Its host network is RootlessKit's, without the host's loopback.
        Sandbox::Docker if cfg!(target_os = "linux") && daemon_local && docker_is_rootless() => {
            anyhow::bail!(
                "--sandbox docker cannot reach llmman serve on loopback through rootless Docker; \
                 use --sandbox podman, or set LLMMAN_HOST to an address the container can reach"
            )
        }
        Sandbox::AppleContainer => {
            let out = output("container", &["system", "dns", "list"])?;
            anyhow::ensure!(
                lists_domain(&out, APPLE_HOST_DOMAIN),
                "--sandbox apple-container reaches llmman serve at {APPLE_HOST_DOMAIN}, which \
                 needs a one-time setup:\n  sudo container system dns create \
                 {APPLE_HOST_DOMAIN} --localhost 203.0.113.113"
            );
        }
        Sandbox::Openshell => {
            // `openshell status` exits 0 whatever the state.
            let out = output("openshell", &["status", "-o", "json"])?;
            check_openshell_status(&out, daemon_local)?;
        }
        _ => {}
    }
    Ok(())
}

/// Stdout of a successful `program args`.
fn output(program: &str, args: &[&str]) -> anyhow::Result<String> {
    let out = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("run {program}"))?;
    anyhow::ensure!(
        out.status.success(),
        "{program} {} failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr).trim()
    );
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Whether `container system dns list` output names `domain`.
fn lists_domain(output: &str, domain: &str) -> bool {
    output
        .split_whitespace()
        .any(|word| word.trim_end_matches('.').eq_ignore_ascii_case(domain))
}

/// The gateway must speak gRPC (`connected`, not `connected_http`), and a
/// loopback daemon is only reachable through a gateway on this machine,
/// since `host.openshell.internal` is the gateway's host.
fn check_openshell_status(json: &str, daemon_local: bool) -> anyhow::Result<()> {
    let status: serde_json::Value = serde_json::from_str(json).unwrap_or_default();
    let state = status["status"].as_str().unwrap_or("unknown");
    anyhow::ensure!(
        state == "connected",
        "--sandbox openshell needs a connected OpenShell gateway; openshell status reports \
         {state:?}"
    );
    let gateway = status["server"].as_str().unwrap_or_default();
    anyhow::ensure!(
        !daemon_local || reqwest::Url::parse(gateway).is_ok_and(|u| is_loopback(&u)),
        "the OpenShell gateway at {gateway} is not on this machine, so its sandboxes cannot \
         reach llmman serve on loopback; set LLMMAN_HOST to a daemon the gateway can reach"
    );
    Ok(())
}

/// [`IMAGE_ENV`], else the template for `integration`.
fn image(integration: &str, configured: Option<&str>) -> anyhow::Result<String> {
    if let Some(image) = configured.map(str::trim).filter(|i| !i.is_empty()) {
        return Ok(image.to_string());
    }
    DEFAULT_IMAGES
        .iter()
        .find(|(id, _)| *id == integration)
        .map(|(_, tag)| format!("{TEMPLATES}:{tag}"))
        .with_context(|| {
            format!(
                "there is no default sandbox image for {integration}; set {IMAGE_ENV} to an \
                 image with {integration} installed"
            )
        })
}

fn is_loopback(url: &reqwest::Url) -> bool {
    url.host_str().is_some_and(|host| {
        let host = host.trim_start_matches('[').trim_end_matches(']');
        host.eq_ignore_ascii_case("localhost")
            || host
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback())
    })
}

/// `server` with a loopback host renamed `alias`; other hosts unchanged.
fn agent_server_for(server: &str, alias: Option<&str>) -> anyhow::Result<String> {
    let Some(alias) = alias else {
        return Ok(server.to_string());
    };
    let mut url = reqwest::Url::parse(server).with_context(|| format!("parse {server}"))?;
    if !is_loopback(&url) {
        return Ok(server.to_string());
    }
    anyhow::ensure!(
        url.scheme() != "https",
        "the sandbox reaches llmman serve at {alias}, a name its TLS certificate does not \
         carry; use plain http on loopback, or set LLMMAN_HOST to a name the certificate has"
    );
    url.set_host(Some(alias))
        .with_context(|| format!("set host {alias}"))?;
    Ok(url.origin().ascii_serialization())
}

/// The enclosing Git work tree, else the cwd; never `~` or above it (a
/// dotfiles repo in `~` falls back to the cwd).
fn workspace(cwd: &Path, home: &Path) -> anyhow::Result<PathBuf> {
    let git_root = cwd.ancestors().find(|dir| dir.join(".git").exists());
    git_root
        .into_iter()
        .chain([cwd])
        .find(|dir| !contains_home(dir, home))
        .map(Path::to_path_buf)
        .with_context(|| {
            format!(
                "--sandbox would let the integration write all of {}, which contains your home \
                 directory; run it from a project directory",
                cwd.display()
            )
        })
}

/// `env` less `PATH`: a launcher's (qwen sets one) names host directories.
fn without_path(env: &[(String, String)]) -> Vec<(String, String)> {
    env.iter().filter(|(k, _)| k != "PATH").cloned().collect()
}

/// Whether writing all of `dir` would include the home directory.
fn contains_home(dir: &Path, home: &Path) -> bool {
    real_path(home).starts_with(real_path(dir))
}

fn interactive() -> bool {
    std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}

fn utf8(path: &Path) -> anyhow::Result<&str> {
    path.to_str()
        .with_context(|| format!("{} is not UTF-8", path.display()))
}

/// The integration's command line inside an image: found on its `PATH`.
fn guest_command(bin: &Path, args: &[String]) -> anyhow::Result<Vec<String>> {
    let program = bin.file_name().map(Path::new).unwrap_or(bin);
    let mut command = vec![utf8(program)?.to_string()];
    command.extend_from_slice(args);
    Ok(command)
}

// ---------------------------------------------------------------------------
// seatbelt
// ---------------------------------------------------------------------------

fn run_seatbelt(
    active: &Active,
    bin: &Path,
    args: &[String],
    env: &[(String, String)],
) -> anyhow::Result<i32> {
    let (dirs, files) = seatbelt_writable(active);
    let workspace = real_path(&active.workspace);
    let mut cmd = Command::new(SANDBOX_EXEC);
    cmd.arg("-p")
        .arg(seatbelt_profile(&dirs, &files, &workspace)?)
        .arg(bin)
        .args(args);
    cmd.envs(env.iter().map(|(k, v)| (k, v)));
    status_code(cmd, SANDBOX_EXEC)
}

/// Real paths, which seatbelt matches (`/tmp` and `/var` are symlinks).
fn seatbelt_writable(active: &Active) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let home = &active.home;
    let mut dirs = vec![
        active.workspace.clone(),
        PathBuf::from("/private/tmp"),
        home.join("Library").join("Caches"),
        home.join(".cache"),
        home.join(".npm"),
    ];
    if let Some(tmp) = std::env::var_os("TMPDIR").map(|t| real_path(Path::new(&t))) {
        // `/private/var/folders/<xx>/<id>/T`: its parent holds the user's
        // cache dir (`C`) too.
        match tmp.parent() {
            Some(parent) if tmp.ends_with("T") => dirs.push(parent.to_path_buf()),
            _ => dirs.push(tmp),
        }
    }
    let mut files = Vec::new();
    for state in &active.state {
        match state {
            State::Dir(dir) => dirs.push(dir.clone()),
            State::Files(file) => files.push(file.clone()),
        }
    }
    let real = |paths: Vec<PathBuf>| paths.iter().map(|p| real_path(p)).collect();
    (real(dirs), real(files))
}

/// `path` with its deepest existing ancestor resolved.
fn real_path(path: &Path) -> PathBuf {
    let mut rest = Vec::new();
    let mut existing = path;
    loop {
        if let Ok(real) = existing.canonicalize() {
            return rest.iter().rev().fold(real, |acc, part| acc.join(part));
        }
        match (existing.parent(), existing.file_name()) {
            (Some(parent), Some(name)) => {
                rest.push(name.to_os_string());
                existing = parent;
            }
            _ => return path.to_path_buf(),
        }
    }
}

/// Everything readable; writes only to `dirs`, `files` (and names
/// extending them) and terminal devices, never to a `.git` in
/// `workspace`: Git runs hooks and config settings (`core.fsmonitor`)
/// from there on the host. The later rule wins.
fn seatbelt_profile(
    dirs: &[PathBuf],
    files: &[PathBuf],
    workspace: &Path,
) -> anyhow::Result<String> {
    let mut profile = String::from(
        "(version 1)\n\
         (allow default)\n\
         (deny file-write*)\n\
         (allow file-write*\n\
         \x20   (literal \"/dev/null\")\n\
         \x20   (literal \"/dev/zero\")\n\
         \x20   (literal \"/dev/tty\")\n\
         \x20   (literal \"/dev/ptmx\")\n\
         \x20   (literal \"/dev/dtracehelper\")\n\
         \x20   (regex #\"^/dev/ttys[0-9]+$\")\n\
         \x20   (regex #\"^/dev/fd/\")\n",
    );
    for dir in dirs {
        profile.push_str(&format!("    (subpath \"{}\")\n", sbpl_path(dir)?));
    }
    for file in files {
        let path = regex_escape(sbpl_path(file)?);
        profile.push_str(&format!("    (regex #\"^{path}\")\n"));
    }
    profile.push_str(")\n");
    let workspace = regex_escape(sbpl_path(workspace)?);
    profile.push_str(&format!(
        "(deny file-write* (regex #\"^{workspace}/(.*/)?\\.git(/|$)\"))\n"
    ));
    Ok(profile)
}

/// Refuses quotes, backslashes and control characters rather than
/// escaping them: SBPL string and regex literals escape differently.
fn sbpl_path(path: &Path) -> anyhow::Result<&str> {
    let text = utf8(path)?;
    anyhow::ensure!(
        !text
            .chars()
            .any(|c| c == '"' || c == '\\' || c.is_control()),
        "--sandbox seatbelt cannot express the path {text:?}"
    );
    Ok(text)
}

fn regex_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if ".^$|?*+()[]{}".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

// ---------------------------------------------------------------------------
// docker, podman, apple-container, microsandbox
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct Mount {
    host: String,
    guest: String,
    read_only: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Plan {
    image: String,
    /// Parents first.
    mounts: Vec<Mount>,
    workdir: String,
    env: BTreeMap<String, String>,
    command: Vec<String>,
}

/// Mounts the workspace and state dirs at their host paths over a `HOME`
/// llmman keeps per integration, pre-creating what an engine would
/// otherwise create as root.
fn plan(
    active: &Active,
    bin: &Path,
    args: &[String],
    env: &[(String, String)],
) -> anyhow::Result<Plan> {
    let home = &active.home;
    let sandbox_home = crate::data_root()?
        .join("sandbox")
        .join(&active.integration);
    let create = |dir: &Path| {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))
    };
    // Resolved: microsandbox refuses a source reached through a symlink.
    let source = |dir: &Path| -> anyhow::Result<String> {
        mount_path(
            &dir.canonicalize()
                .with_context(|| format!("resolve {}", dir.display()))?,
        )
    };
    create(&sandbox_home)?;
    let mount = |host: &Path, guest: &Path, read_only| -> anyhow::Result<Mount> {
        Ok(Mount {
            host: source(host)?,
            guest: mount_path(guest)?,
            read_only,
        })
    };
    let mut mounts = vec![mount(&sandbox_home, home, false)?];
    let state_dirs = active.state.iter().filter_map(|s| match s {
        State::Dir(dir) => Some(dir),
        State::Files(_) => None,
    });
    for dir in std::iter::once(&active.workspace).chain(state_dirs) {
        create(dir)?;
        if let Ok(rel) = dir.strip_prefix(home) {
            create(&sandbox_home.join(rel))?;
        }
        let next = mount(dir, dir, false)?;
        if !mounts.iter().any(|m| m.guest == next.guest) {
            mounts.push(next);
        }
    }
    // Git runs hooks and config settings (`core.fsmonitor`) from here on
    // the host. A mount point also can't be renamed away. A linked worktree
    // has a `.git` file instead of a directory; its gitdir and commondir are
    // outside the workspace and must be mounted too for read-only Git access.
    for (host, guest) in git_metadata_paths(&active.workspace)? {
        let next = mount(&host, &guest, true)?;
        if !mounts.iter().any(|m| m.guest == next.guest) {
            mounts.push(next);
        }
    }
    let mut env = guest_env(home, env, |name| std::env::var(name).ok())?;
    if active.sandbox == Sandbox::Microsandbox {
        // Its agent is PID 1, so the image's `tini` must be a subreaper.
        env.insert("TINI_SUBREAPER".to_string(), "1".to_string());
    }
    Ok(Plan {
        image: active.image.clone().context("no sandbox image")?,
        mounts,
        workdir: mount_path(&std::env::current_dir()?)?,
        env,
        command: guest_command(bin, args)?,
    })
}

/// A path every engine's `-v host:guest` can take.
fn mount_path(path: &Path) -> anyhow::Result<String> {
    let text = utf8(path)?;
    anyhow::ensure!(
        !text.contains([':', ',']),
        "--sandbox cannot mount {text}: it contains ':' or ','"
    );
    Ok(text.to_string())
}

/// Paths Git needs to read repository metadata in `workspace`. Normal
/// repositories have a `.git` directory. Linked worktrees and submodules
/// instead have a `.git` file pointing at a gitdir; linked worktrees also
/// have a `commondir` file pointing at the shared object database and refs.
fn git_metadata_paths(workspace: &Path) -> anyhow::Result<Vec<(PathBuf, PathBuf)>> {
    let dot_git = workspace.join(".git");
    let dot_git_metadata = match std::fs::symlink_metadata(&dot_git) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("inspect Git metadata path {}", dot_git.display()));
        }
    };
    // The workspace itself is mounted writable into the sandbox. Never follow
    // a workspace-controlled `.git` symlink into an arbitrary host directory.
    anyhow::ensure!(
        !dot_git_metadata.file_type().is_symlink(),
        "Git metadata path {} must not be a symlink",
        dot_git.display()
    );
    let mut paths = vec![(dot_git.clone(), dot_git.clone())];
    if !dot_git_metadata.is_file() {
        return Ok(paths);
    }

    let contents = std::fs::read_to_string(&dot_git)
        .with_context(|| format!("read Git worktree pointer {}", dot_git.display()))?;
    let Some(target) = contents.trim().strip_prefix("gitdir: ") else {
        return Ok(paths);
    };
    let target = Path::new(target);
    let git_guest = if target.is_absolute() {
        target.to_path_buf()
    } else {
        workspace.join(target)
    };
    let git_dir = git_guest
        .canonicalize()
        .with_context(|| format!("resolve Git metadata directory {}", git_guest.display()))?;
    anyhow::ensure!(
        git_dir.is_dir(),
        "Git metadata directory {} referenced by {} does not exist",
        git_guest.display(),
        dot_git.display()
    );
    let workspace_real = workspace.canonicalize()?;
    let dot_git_real = dot_git.canonicalize()?;
    let git_inside_workspace = git_dir.starts_with(&workspace_real);
    if !git_inside_workspace {
        // A writable workspace can replace `.git` with an arbitrary pointer.
        // Only accept an external gitdir that proves it is a real linked
        // worktree by pointing back to this workspace's `.git` file.
        let backlink_file = git_dir.join("gitdir");
        anyhow::ensure!(backlink_file.is_file(), "Git metadata directory {} is outside the workspace and has no linked-worktree backlink", git_dir.display());
        let backlink = std::fs::read_to_string(&backlink_file)?;
        let backlink = Path::new(backlink.trim());
        let backlink = if backlink.is_absolute() {
            backlink.to_path_buf()
        } else {
            git_dir.join(backlink)
        };
        anyhow::ensure!(
            backlink.canonicalize()? == dot_git_real,
            "Git metadata directory {} does not point back to this workspace",
            git_dir.display()
        );
        let worktrees_dir = git_dir
            .parent()
            .context("Git metadata directory has no parent")?;
        anyhow::ensure!(
            worktrees_dir
                .file_name()
                .is_some_and(|name| name == "worktrees"),
            "Git metadata directory {} is not below a Git worktrees directory",
            git_dir.display()
        );
        let common_dir = worktrees_dir
            .parent()
            .context("Git worktrees directory has no common directory")?;
        anyhow::ensure!(
            common_dir.join("objects").is_dir(),
            "Git metadata directory {} is not below a Git common directory",
            git_dir.display()
        );
    }
    paths.push((git_dir.clone(), git_guest.clone()));

    let common_file = git_dir.join("commondir");
    if common_file.is_file() {
        let common = std::fs::read_to_string(&common_file)
            .with_context(|| format!("read Git common directory {}", common_file.display()))?;
        let common = Path::new(common.trim());
        let common_guest = if common.is_absolute() {
            common.to_path_buf()
        } else {
            // Reject symlinks only on the relative traversal from git_guest
            // to common_guest. A symlink shared by both paths is harmless:
            // the container mounts both destinations in the same alias tree.
            let mut path = git_guest.clone();
            for component in common.components() {
                match component {
                    std::path::Component::CurDir => {}
                    std::path::Component::ParentDir => {
                        anyhow::ensure!(
                            !std::fs::symlink_metadata(&path)?.file_type().is_symlink(),
                            "cannot safely mount relative Git common directory through symlinked gitdir path {}",
                            git_guest.display()
                        );
                        path.pop();
                    }
                    std::path::Component::Normal(name) => {
                        anyhow::ensure!(
                            !std::fs::symlink_metadata(&path)?.file_type().is_symlink(),
                            "cannot safely mount relative Git common directory through symlinked gitdir path {}",
                            git_guest.display()
                        );
                        path.push(name);
                    }
                    std::path::Component::RootDir | std::path::Component::Prefix(_) => {
                        anyhow::bail!("relative Git common directory has an absolute component")
                    }
                }
            }
            anyhow::ensure!(
                !std::fs::symlink_metadata(&path)?.file_type().is_symlink(),
                "cannot safely mount relative Git common directory through symlinked gitdir path {}",
                git_guest.display()
            );
            git_guest.join(common)
        };
        anyhow::ensure!(
            common_guest.is_dir(),
            "Git common directory {} referenced by {} does not exist",
            common_guest.display(),
            common_file.display()
        );
        let common_real = common_guest
            .canonicalize()
            .with_context(|| format!("resolve Git common directory {}", common_guest.display()))?;
        let trusted_common = if git_inside_workspace {
            common_real.starts_with(&workspace_real)
        } else {
            let worktrees_dir = git_dir.parent();
            worktrees_dir.is_some_and(|worktrees| {
                worktrees
                    .file_name()
                    .is_some_and(|name| name == "worktrees")
                    && worktrees.parent() == Some(common_real.as_path())
                    && common_real.join("objects").is_dir()
            })
        };
        anyhow::ensure!(
            trusted_common,
            "Git common directory {} is not trusted metadata for {}",
            common_real.display(),
            git_dir.display()
        );
        paths.push((common_real, common_guest));
    }
    Ok(paths)
}

fn guest_env(
    home: &Path,
    env: &[(String, String)],
    host: impl Fn(&str) -> Option<String>,
) -> anyhow::Result<BTreeMap<String, String>> {
    let mut out: BTreeMap<String, String> = PASSTHROUGH_ENV
        .iter()
        .filter_map(|name| Some((name.to_string(), host(name)?)))
        .collect();
    out.insert("HOME".to_string(), mount_path(home)?);
    out.extend(env.iter().cloned());
    Ok(out)
}

/// Who the guest writes files as. A Linux bind mount keeps the guest's
/// uid; Docker Desktop and the VM backends map it to the host user.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Identity {
    Image,
    /// `docker run --user`: the user, or root for a rootless daemon.
    User(String),
    /// Rootless podman.
    KeepId,
}

fn identity(sandbox: Sandbox, linux: bool) -> Identity {
    let Some((uid, gid, euid)) = user_ids() else {
        return Identity::Image;
    };
    match sandbox {
        Sandbox::Docker if linux && docker_is_rootless() => Identity::User("0:0".to_string()),
        Sandbox::Docker if linux => Identity::User(format!("{uid}:{gid}")),
        Sandbox::Podman if euid != 0 => Identity::KeepId,
        _ => Identity::Image,
    }
}

#[cfg(unix)]
fn user_ids() -> Option<(u32, u32, u32)> {
    // SAFETY: these cannot fail and touch no memory.
    Some(unsafe { (libc::getuid(), libc::getgid(), libc::geteuid()) })
}

#[cfg(not(unix))]
fn user_ids() -> Option<(u32, u32, u32)> {
    None
}

fn docker_is_rootless() -> bool {
    output("docker", &["info", "--format", "{{json .SecurityOptions}}"])
        .is_ok_and(|out| out.contains("rootless"))
}

/// On Linux, docker/podman also get the host network and no SELinux
/// labelling, which would deny writes to unrelabelled mounts (and
/// relabelling would change the user's own files).
fn image_args(
    sandbox: Sandbox,
    plan: &Plan,
    tty: bool,
    linux: bool,
    identity: &Identity,
) -> Vec<String> {
    let mut args: Vec<String> = vec!["run".into()];
    if sandbox == Sandbox::Microsandbox {
        // Unnamed, so removed on exit; attaches when stdin is a terminal.
        args.extend(["--net".into(), "public,host".into()]);
    } else {
        args.extend(["--rm".into(), "-i".into()]);
        if tty {
            args.push("-t".into());
        }
    }
    if matches!(sandbox, Sandbox::Docker | Sandbox::Podman) && linux {
        args.extend(["--network".into(), "host".into()]);
        args.extend(["--security-opt".into(), "label=disable".into()]);
    }
    match identity {
        Identity::Image => {}
        Identity::User(user) => args.extend(["--user".into(), user.clone()]),
        Identity::KeepId => args.push("--userns=keep-id".into()),
    }
    for Mount {
        host,
        guest,
        read_only,
    } in &plan.mounts
    {
        let volume = match (sandbox, read_only) {
            // `-v` has no documented read-only option for `container`.
            (Sandbox::AppleContainer, true) => {
                args.extend([
                    "--mount".into(),
                    format!("type=bind,source={host},target={guest},readonly"),
                ]);
                continue;
            }
            (_, true) => format!("{host}:{guest}:ro"),
            // microsandbox otherwise creates host files owner-only.
            (Sandbox::Microsandbox, false) => format!("{host}:{guest}:host-perms=mirror"),
            (_, false) => format!("{host}:{guest}"),
        };
        args.extend(["-v".into(), volume]);
    }
    args.extend(["-w".into(), plan.workdir.clone()]);
    for name in plan.env.keys() {
        args.extend(["-e".into(), name.clone()]);
    }
    args.push(plan.image.clone());
    if sandbox == Sandbox::Microsandbox {
        args.push("--".into());
    }
    args.extend(plan.command.iter().cloned());
    args
}

// ---------------------------------------------------------------------------
// openshell
// ---------------------------------------------------------------------------

/// OpenShell can't upload and start a command in one step: create the
/// sandbox detached with the workspace uploaded, allow the daemon in its
/// policy, then exec where the workspace landed. It is kept afterwards,
/// since the changes are only in there.
fn run_openshell(
    active: &Active,
    bin: &Path,
    args: &[String],
    env: &[(String, String)],
) -> anyhow::Result<i32> {
    // Kept sandboxes outlive the PID, which gets reused.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let name = format!("llmman-{}-{now}-{}", active.integration, std::process::id());
    let workspace = utf8(&active.workspace)?;
    // `--upload` splits at the first ':'.
    anyhow::ensure!(
        !workspace.contains(':'),
        "--sandbox openshell cannot upload {workspace}: it contains ':'"
    );
    let image = active.image.as_deref().context("no sandbox image")?;
    let endpoint = openshell_endpoint(&active.server)?;
    let openshell = |args: Vec<String>, what: &str| -> anyhow::Result<()> {
        let mut cmd = Command::new("openshell");
        cmd.args(args);
        anyhow::ensure!(
            status_code(cmd, "openshell")? == 0,
            "openshell {what} failed"
        );
        Ok(())
    };

    openshell(
        openshell_create_args(&name, image, workspace),
        "sandbox create",
    )?;
    let workdir = (|| {
        openshell(openshell_policy_args(&name, &endpoint), "policy update")?;
        let root = output(
            "openshell",
            &[
                "sandbox", "exec", "-n", &name, "--no-tty", "--", "pwd", "-P",
            ],
        )?;
        let root = root
            .lines()
            .map(str::trim)
            .rfind(|line| line.starts_with('/'))
            .context("openshell sandbox exec printed no directory")?;
        let base = active
            .workspace
            .file_name()
            .and_then(|n| n.to_str())
            .context("workspace has no name")?;
        anyhow::Ok(format!("{}/{base}", root.trim_end_matches('/')))
    })()
    .inspect_err(|_| {
        let _ = Command::new("openshell")
            .args(["sandbox", "delete", &name])
            .status();
    })?;

    let command = guest_command(bin, args)?;
    let mut exec = Command::new("openshell");
    exec.args(openshell_exec_args(
        &name,
        &workdir,
        interactive(),
        env,
        &command,
    ));
    let code = status_code(exec, "openshell")?;
    eprintln!(
        "[llmman] The changes are in OpenShell sandbox {name}, at {workdir}.\n\
         [llmman] Copy them back:  openshell sandbox download {name} {workdir} {workspace}\n\
         [llmman] Remove it:       openshell sandbox delete {name}"
    );
    Ok(code)
}

/// `host:port` of the daemon as the sandbox reaches it.
fn openshell_endpoint(server: &str) -> anyhow::Result<String> {
    let url = reqwest::Url::parse(server).with_context(|| format!("parse {server}"))?;
    let host = url.host_str().context("llmman serve has no host")?;
    let port = url
        .port_or_known_default()
        .context("llmman serve has no port")?;
    Ok(format!("{host}:{port}"))
}

fn openshell_create_args(name: &str, image: &str, workspace: &str) -> Vec<String> {
    [
        "sandbox", "create", "--name", name, "--from", image, "--upload", workspace, "--detach",
    ]
    .map(String::from)
    .to_vec()
}

/// Every binary (`/**`): rules match real paths, and an npm-installed
/// agent's is `node`.
fn openshell_policy_args(name: &str, endpoint: &str) -> Vec<String> {
    [
        "policy",
        "update",
        name,
        "--rule-name",
        "llmman",
        "--binary",
        "/**",
        "--add-endpoint",
        endpoint,
        "--wait",
    ]
    .map(String::from)
    .to_vec()
}

fn openshell_exec_args(
    name: &str,
    workdir: &str,
    tty: bool,
    env: &[(String, String)],
    command: &[String],
) -> Vec<String> {
    let tty = if tty { "--tty" } else { "--no-tty" };
    let mut args: Vec<String> = ["sandbox", "exec", "-n", name, tty, "--workdir", workdir]
        .map(String::from)
        .to_vec();
    let env: BTreeMap<&str, &str> = env.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    for (name, value) in env {
        args.extend(["--env".into(), format!("{name}={value}")]);
    }
    args.push("--".into());
    args.extend(command.iter().cloned());
    args
}

fn status_code(mut cmd: Command, what: &str) -> anyhow::Result<i32> {
    let status = cmd
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .with_context(|| format!("failed to run {what}"))?;
    Ok(status.code().unwrap_or(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn every_sandbox_parses_under_its_documented_name() {
        for (value, sandbox) in [
            ("sbx", Sandbox::Sbx),
            ("seatbelt", Sandbox::Seatbelt),
            ("docker", Sandbox::Docker),
            ("podman", Sandbox::Podman),
            ("apple-container", Sandbox::AppleContainer),
            ("microsandbox", Sandbox::Microsandbox),
            ("openshell", Sandbox::Openshell),
        ] {
            assert_eq!(Sandbox::from_str(value, false).unwrap(), sandbox);
            assert_eq!(sandbox.name(), value);
        }
    }

    #[test]
    fn sbx_gets_the_model_flags_and_the_forwarded_arguments() {
        assert_eq!(
            sbx_args(
                "claude",
                Some("qwen3.8"),
                Some("openrouter"),
                None,
                &strings(&["--continue"])
            ),
            strings(&[
                "run",
                "claude",
                "--model",
                "qwen3.8",
                "--provider",
                "openrouter",
                "--",
                "--continue"
            ])
        );
        assert_eq!(
            sbx_args(
                "opencode",
                Some("gemma4"),
                None,
                Some(("anthropic", "claude-sonnet-5")),
                &[]
            ),
            strings(&[
                "run",
                "opencode",
                "--model",
                "gemma4",
                "--overflow-provider",
                "anthropic",
                "--overflow-model",
                "claude-sonnet-5"
            ])
        );
        // No model: sbx's own default, not an empty flag.
        assert_eq!(
            sbx_args("codex", Some("  "), None, None, &[]),
            strings(&["run", "codex"])
        );
    }

    /// A table entry for a name `launch` does not dispatch is dead, and
    /// hides that the real name is missing.
    #[test]
    fn every_integration_named_here_is_a_real_one() {
        let known = |id: &str| {
            id == "copilot-cli" || super::super::INTEGRATIONS.iter().any(|i| i.name == id)
        };
        for (id, _) in SBX_AGENTS.iter().chain(DEFAULT_IMAGES) {
            assert!(known(id), "{id} is not an integration");
        }
    }

    #[test]
    fn sbx_maps_integrations_to_its_agents_and_refuses_the_rest() {
        assert_eq!(sbx_agent("Claude").unwrap(), "claude");
        assert_eq!(sbx_agent("copilot-cli").unwrap(), "copilot");
        let err = sbx_agent("aider").unwrap_err().to_string();
        assert!(err.contains("sbx has no aider agent"), "{err}");
    }

    #[test]
    fn images_default_to_the_sandbox_templates_and_can_be_overridden() {
        assert_eq!(
            image("claude", None).unwrap(),
            "docker.io/docker/sandbox-templates:claude-code"
        );
        assert_eq!(image("aider", Some(" my/aider:1 ")).unwrap(), "my/aider:1");
        let err = image("aider", Some("")).unwrap_err().to_string();
        assert!(err.contains(IMAGE_ENV), "{err}");
    }

    #[test]
    fn a_loopback_daemon_is_renamed_for_the_guest() {
        let alias = Some("host.docker.internal");
        for server in [
            "http://127.0.0.1:17434",
            "http://localhost:17434",
            "http://[::1]:17434",
        ] {
            assert_eq!(
                agent_server_for(server, alias).unwrap(),
                "http://host.docker.internal:17434"
            );
        }
        // No alias: the guest shares this machine's loopback.
        assert_eq!(
            agent_server_for("http://127.0.0.1:17434", None).unwrap(),
            "http://127.0.0.1:17434"
        );
        // A daemon elsewhere is reached as it is.
        assert_eq!(
            agent_server_for("https://inferencebox:443", alias).unwrap(),
            "https://inferencebox:443"
        );
        // Its certificate would not name the alias.
        assert!(agent_server_for("https://127.0.0.1:17434", alias).is_err());
    }

    #[test]
    fn linux_containers_share_the_host_network_and_other_hosts_use_an_alias() {
        assert_eq!(Sandbox::Docker.host_alias(true), None);
        assert_eq!(Sandbox::Podman.host_alias(true), None);
        assert_eq!(
            Sandbox::Docker.host_alias(false),
            Some("host.docker.internal")
        );
        assert_eq!(
            Sandbox::Podman.host_alias(false),
            Some("host.containers.internal")
        );
        for linux in [true, false] {
            assert_eq!(Sandbox::Seatbelt.host_alias(linux), None);
            assert_eq!(
                Sandbox::Microsandbox.host_alias(linux),
                Some("host.microsandbox.internal")
            );
            assert_eq!(
                Sandbox::AppleContainer.host_alias(linux),
                Some(APPLE_HOST_DOMAIN)
            );
            assert_eq!(Sandbox::Openshell.host_alias(linux), Some(OPENSHELL_HOST));
        }
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "llmman-sandbox-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn the_workspace_is_the_git_work_tree_but_never_the_home_directory() {
        let root = temp_dir("workspace");
        let home = root.join("home");
        let repo = home.join("src").join("repo");
        let sub = repo.join("crate");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        std::fs::create_dir_all(&sub).unwrap();
        assert_eq!(workspace(&sub, &home).unwrap(), repo);

        let plain = home.join("scratch");
        std::fs::create_dir_all(&plain).unwrap();
        assert_eq!(workspace(&plain, &home).unwrap(), plain);

        // Not the home directory, nor anything above it.
        assert!(workspace(&home, &home).is_err());
        assert!(workspace(&root, &home).is_err());

        // A home directory under Git falls back to the current one.
        std::fs::create_dir_all(home.join(".git")).unwrap();
        assert_eq!(workspace(&plain, &home).unwrap(), plain);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn container_plan_mounts_linked_worktree_git_metadata_read_only() {
        let root = temp_dir("linked-worktree");
        let repo = root.join("repo");
        let worktree = root.join("worktree");
        std::fs::create_dir_all(&repo).unwrap();
        let git = |cwd: &Path, args: &[&str]| {
            let output = std::process::Command::new("git")
                .current_dir(cwd)
                .args(args)
                .output()
                .expect("git should be installed for this test");
            assert!(
                output.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            String::from_utf8(output.stdout).unwrap().trim().to_string()
        };
        git(&repo, &["init", "-q"]);
        std::fs::write(repo.join("README.md"), "worktree").unwrap();
        git(&repo, &["add", "README.md"]);
        git(
            &repo,
            &[
                "-c",
                "user.name=llmman test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-m",
                "initial",
                "--quiet",
            ],
        );
        git(
            &repo,
            &["worktree", "add", "-b", "linked", "../worktree", "--quiet"],
        );

        let git_dir = PathBuf::from(git(
            &worktree,
            &["rev-parse", "--path-format=absolute", "--git-dir"],
        ));
        let common_dir = PathBuf::from(git(
            &worktree,
            &["rev-parse", "--path-format=absolute", "--git-common-dir"],
        ));

        // Windows drive letters are rejected by the existing `mount_path` contract
        // before a container plan can be built. Still check real Git worktree
        // pointers there; assert the resulting read-only mounts on Unix.
        let metadata_paths = git_metadata_paths(&worktree).unwrap();
        assert_eq!(metadata_paths[0].0, worktree.join(".git"));
        assert_eq!(metadata_paths[0].1, worktree.join(".git"));
        for metadata in [git_dir, common_dir] {
            let metadata = metadata.canonicalize().unwrap();
            assert!(
                metadata_paths.iter().any(|(host, _)| host == &metadata),
                "missing Git metadata path {}: {metadata_paths:?}",
                metadata.display()
            );
        }

        #[cfg(not(windows))]
        {
            let home = dirs::home_dir().unwrap();
            let workspace = workspace(&worktree, &home).unwrap();
            let active = Active {
                sandbox: Sandbox::Docker,
                integration: format!("worktree-test-{}", std::process::id()),
                server: "http://127.0.0.1:17434".into(),
                workspace,
                home,
                image: Some("test/image".into()),
                state: vec![],
            };
            let plan = plan(&active, Path::new("codex"), &[], &[]).unwrap();
            for (metadata, guest) in metadata_paths {
                let metadata_host = metadata.canonicalize().unwrap();
                let metadata_guest = if metadata == worktree.join(".git") {
                    metadata.clone()
                } else {
                    guest
                };
                assert!(
                    plan.mounts.iter().any(|mount| {
                        mount.host == metadata_host.to_string_lossy()
                            && mount.guest == metadata_guest.to_string_lossy()
                            && mount.read_only
                    }),
                    "missing read-only git metadata mount for {}: {:?}",
                    metadata.display(),
                    plan.mounts
                        .iter()
                        .map(|m| (&m.host, &m.guest, m.read_only))
                        .collect::<Vec<_>>()
                );
            }
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn container_plan_preserves_symlinked_git_metadata_guest_paths() {
        use std::os::unix::fs::symlink;

        let root = temp_dir("linked-worktree-symlink");
        let repo = root.join("repo");
        let worktree = root.join("worktree");
        let git_dir_alias = root.join("gitdir-alias");
        let common_dir_alias = root.join("common-alias");
        std::fs::create_dir_all(&repo).unwrap();
        let git = |cwd: &Path, args: &[&str]| {
            let output = std::process::Command::new("git")
                .current_dir(cwd)
                .args(args)
                .output()
                .expect("git should be installed for this test");
            assert!(
                output.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            String::from_utf8(output.stdout).unwrap().trim().to_string()
        };
        git(&repo, &["init", "-q"]);
        std::fs::write(repo.join("README.md"), "worktree").unwrap();
        git(&repo, &["add", "README.md"]);
        git(
            &repo,
            &[
                "-c",
                "user.name=llmman test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-m",
                "initial",
                "--quiet",
            ],
        );
        git(
            &repo,
            &["worktree", "add", "-b", "linked", "../worktree", "--quiet"],
        );

        let git_dir = PathBuf::from(git(
            &worktree,
            &["rev-parse", "--path-format=absolute", "--git-dir"],
        ))
        .canonicalize()
        .unwrap();
        let common_dir = PathBuf::from(git(
            &worktree,
            &["rev-parse", "--path-format=absolute", "--git-common-dir"],
        ))
        .canonicalize()
        .unwrap();
        symlink(&git_dir, &git_dir_alias).unwrap();
        symlink(&common_dir, &common_dir_alias).unwrap();
        std::fs::write(
            worktree.join(".git"),
            format!("gitdir: {}\n", git_dir_alias.display()),
        )
        .unwrap();
        // Git follows both aliases on the host, so the container must mount
        // the canonical source at each pointer-resolved guest path. A
        // relative commondir cannot safely survive an aliased gitdir mount:
        // `..` inside the guest is relative to the alias location.
        assert_eq!(
            git(&worktree, &["rev-parse", "--show-toplevel"]),
            worktree.canonicalize().unwrap().display().to_string()
        );
        let home = dirs::home_dir().unwrap();
        let active = Active {
            sandbox: Sandbox::Docker,
            integration: format!("worktree-symlink-test-{}", std::process::id()),
            server: "http://127.0.0.1:17434".into(),
            workspace: worktree.clone(),
            home,
            image: Some("test/image".into()),
            state: vec![],
        };
        assert!(
            plan(&active, Path::new("codex"), &[], &[]).is_err(),
            "a relative commondir through a symlinked gitdir must fail closed"
        );
        std::fs::write(
            git_dir.join("commondir"),
            format!("{}\n", common_dir_alias.display()),
        )
        .unwrap();
        let plan = plan(&active, Path::new("codex"), &[], &[]).unwrap();
        for (host, guest) in [(&git_dir, &git_dir_alias), (&common_dir, &common_dir_alias)] {
            assert!(
                plan.mounts.iter().any(|mount| {
                    mount.host == host.to_string_lossy()
                        && mount.guest == guest.to_string_lossy()
                        && mount.read_only
                }),
                "missing host {} mounted at guest {}: {:?}",
                host.display(),
                guest.display(),
                plan.mounts
                    .iter()
                    .map(|m| (&m.host, &m.guest, m.read_only))
                    .collect::<Vec<_>>()
            );
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn container_plan_allows_shared_symlinked_ancestor_for_relative_commondir() {
        use std::os::unix::fs::symlink;

        let root = temp_dir("linked-worktree-shared-ancestor-symlink");
        let repo = root.join("repo");
        let worktree = root.join("worktree");
        let root_alias = root.with_file_name(format!(
            "{}-alias",
            root.file_name().unwrap().to_string_lossy()
        ));
        std::fs::create_dir_all(&repo).unwrap();
        let git = |cwd: &Path, args: &[&str]| {
            let output = std::process::Command::new("git")
                .current_dir(cwd)
                .args(args)
                .output()
                .expect("git should be installed for this test");
            assert!(
                output.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            String::from_utf8(output.stdout).unwrap().trim().to_string()
        };
        git(&repo, &["init", "-q"]);
        std::fs::write(repo.join("README.md"), "worktree").unwrap();
        git(&repo, &["add", "README.md"]);
        git(
            &repo,
            &[
                "-c",
                "user.name=llmman test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-m",
                "initial",
                "--quiet",
            ],
        );
        git(
            &repo,
            &[
                "worktree",
                "add",
                "-b",
                "linked",
                worktree.to_str().unwrap(),
                "--quiet",
            ],
        );
        symlink(&root, &root_alias).unwrap();
        let git_dir = PathBuf::from(git(
            &worktree,
            &["rev-parse", "--path-format=absolute", "--git-dir"],
        ))
        .canonicalize()
        .unwrap();
        let git_guest = root_alias
            .join("repo")
            .join(".git/worktrees")
            .join(git_dir.file_name().unwrap());
        std::fs::write(
            worktree.join(".git"),
            format!("gitdir: {}\n", git_guest.display()),
        )
        .unwrap();
        assert_eq!(
            git(&worktree, &["rev-parse", "--show-toplevel"]),
            worktree.canonicalize().unwrap().display().to_string()
        );

        let active = Active {
            sandbox: Sandbox::Docker,
            integration: format!("worktree-shared-ancestor-test-{}", std::process::id()),
            server: "http://127.0.0.1:17434".into(),
            workspace: worktree,
            home: dirs::home_dir().unwrap(),
            image: Some("test/image".into()),
            state: vec![],
        };
        let plan = plan(&active, Path::new("codex"), &[], &[])
            .expect("a shared symlinked ancestor must not invalidate relative commondir");
        let common = git_dir
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .canonicalize()
            .unwrap();
        let common_guest = git_guest.join("../..");
        assert!(
            plan.mounts.iter().any(|mount| {
                mount.host == git_dir.to_string_lossy()
                    && mount.guest == git_guest.to_string_lossy()
                    && mount.read_only
            }),
            "missing read-only gitdir mount at alias: {:?}",
            plan.mounts
                .iter()
                .map(|m| (&m.host, &m.guest, m.read_only))
                .collect::<Vec<_>>()
        );
        assert!(
            plan.mounts.iter().any(|mount| {
                mount.host == common.to_string_lossy()
                    && mount.guest == common_guest.to_string_lossy()
                    && mount.read_only
            }),
            "missing read-only common metadata mount at alias-relative path: {:?}",
            plan.mounts
                .iter()
                .map(|m| (&m.host, &m.guest, m.read_only))
                .collect::<Vec<_>>()
        );
        let _ = std::fs::remove_dir_all(&root_alias);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn container_plan_rejects_relative_commondir_through_symlinked_gitdir_ancestor() {
        use std::os::unix::fs::symlink;

        let root = temp_dir("linked-worktree-ancestor-symlink");
        let repo = root.join("repo");
        let worktree = root.join("worktree");
        let gitdir_parent_alias = root.join("gitdir-parent-alias");
        std::fs::create_dir_all(&repo).unwrap();
        let git = |cwd: &Path, args: &[&str]| {
            let output = std::process::Command::new("git")
                .current_dir(cwd)
                .args(args)
                .output()
                .expect("git should be installed for this test");
            assert!(
                output.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            String::from_utf8(output.stdout).unwrap().trim().to_string()
        };
        git(&repo, &["init", "-q"]);
        std::fs::write(repo.join("README.md"), "worktree").unwrap();
        git(&repo, &["add", "README.md"]);
        git(
            &repo,
            &[
                "-c",
                "user.name=llmman test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-m",
                "initial",
                "--quiet",
            ],
        );
        git(
            &repo,
            &["worktree", "add", "-b", "linked", "../worktree", "--quiet"],
        );

        let git_dir = PathBuf::from(git(
            &worktree,
            &["rev-parse", "--path-format=absolute", "--git-dir"],
        ))
        .canonicalize()
        .unwrap();
        symlink(git_dir.parent().unwrap(), &gitdir_parent_alias).unwrap();
        let git_guest = gitdir_parent_alias.join(git_dir.file_name().unwrap());
        std::fs::write(
            worktree.join(".git"),
            format!("gitdir: {}\n", git_guest.display()),
        )
        .unwrap();
        assert_eq!(
            git(&worktree, &["rev-parse", "--show-toplevel"]),
            worktree.canonicalize().unwrap().display().to_string()
        );

        let active = Active {
            sandbox: Sandbox::Docker,
            integration: format!("worktree-ancestor-symlink-test-{}", std::process::id()),
            server: "http://127.0.0.1:17434".into(),
            workspace: worktree,
            home: dirs::home_dir().unwrap(),
            image: Some("test/image".into()),
            state: vec![],
        };
        assert!(
            plan(&active, Path::new("codex"), &[], &[]).is_err(),
            "relative commondir through a symlinked gitdir ancestor must fail closed"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn container_plan_rejects_a_git_pointer_to_host_ssh() {
        let root = temp_dir("untrusted-git-pointer");
        let workspace = root.join("workspace");
        let home = root.join("home");
        let ssh = home.join(".ssh");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&ssh).unwrap();
        std::fs::write(ssh.join("id_ed25519"), "private key").unwrap();

        // A sandboxed integration can write this file into its RW workspace.
        // On the next launch the host must not turn its pointer into a mount.
        std::fs::write(
            workspace.join(".git"),
            format!("gitdir: {}\n", ssh.display()),
        )
        .unwrap();
        let active = Active {
            sandbox: Sandbox::Docker,
            integration: format!("untrusted-git-pointer-test-{}", std::process::id()),
            server: "http://127.0.0.1:17434".into(),
            workspace,
            home,
            image: Some("test/image".into()),
            state: vec![],
        };

        assert!(plan(&active, Path::new("codex"), &[], &[]).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn container_plan_rejects_a_symlinked_git_directory_to_host_ssh() {
        use std::os::unix::fs::symlink;

        let root = temp_dir("symlinked-git-directory");
        let workspace = root.join("workspace");
        let home = root.join("home");
        let ssh = home.join(".ssh");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&ssh).unwrap();
        std::fs::write(ssh.join("id_ed25519"), "private key").unwrap();
        symlink(&ssh, workspace.join(".git")).unwrap();

        let active = Active {
            sandbox: Sandbox::Docker,
            integration: format!("symlinked-git-directory-test-{}", std::process::id()),
            server: "http://127.0.0.1:17434".into(),
            workspace,
            home,
            image: Some("test/image".into()),
            state: vec![],
        };

        assert!(
            plan(&active, Path::new("codex"), &[], &[]).is_err(),
            "a symlinked .git directory must not expose host metadata to the sandbox"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn container_plan_rejects_an_internal_gitdir_with_external_commondir() {
        let root = temp_dir("untrusted-commondir");
        let workspace = root.join("workspace");
        let home = root.join("home");
        let ssh = home.join(".ssh");
        let git_dir = workspace.join(".git-metadata").join("repo");
        std::fs::create_dir_all(&git_dir).unwrap();
        std::fs::create_dir_all(&ssh).unwrap();
        std::fs::write(ssh.join("id_ed25519"), "private key").unwrap();
        std::fs::write(workspace.join(".git"), "gitdir: .git-metadata/repo\n").unwrap();
        std::fs::write(git_dir.join("commondir"), format!("{}\n", ssh.display())).unwrap();

        let active = Active {
            sandbox: Sandbox::Docker,
            integration: format!("untrusted-commondir-test-{}", std::process::id()),
            server: "http://127.0.0.1:17434".into(),
            workspace,
            home,
            image: Some("test/image".into()),
            state: vec![],
        };

        assert!(plan(&active, Path::new("codex"), &[], &[]).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn container_plan_rejects_external_gitdir_outside_worktrees_without_commondir() {
        let root = temp_dir("external-gitdir-outside-worktrees");
        let workspace = root.join("workspace");
        let git_dir = root.join("host/metadata/arbitrary/gitdir");
        let dot_git = workspace.join(".git");
        std::fs::create_dir_all(&git_dir).unwrap();
        std::fs::create_dir_all(root.join("host/metadata/objects")).unwrap();
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::write(&dot_git, format!("gitdir: {}\n", git_dir.display())).unwrap();
        std::fs::write(git_dir.join("gitdir"), format!("{}\n", dot_git.display())).unwrap();

        assert!(
            git_metadata_paths(&workspace).is_err(),
            "external Git metadata without the common/worktrees/name layout must be rejected"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn container_plan_rejects_symlinked_final_component_of_relative_commondir() {
        use std::os::unix::fs::symlink;

        let root = temp_dir("linked-worktree-final-commondir-symlink");
        let repo = root.join("repo");
        let worktree = root.join("worktree");
        std::fs::create_dir_all(&repo).unwrap();
        let git = |cwd: &Path, args: &[&str]| {
            let output = std::process::Command::new("git")
                .current_dir(cwd)
                .args(args)
                .output()
                .expect("git should be installed for this test");
            assert!(
                output.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            String::from_utf8(output.stdout).unwrap().trim().to_string()
        };
        git(&repo, &["init", "-q"]);
        std::fs::write(repo.join("README.md"), "worktree").unwrap();
        git(&repo, &["add", "README.md"]);
        git(
            &repo,
            &[
                "-c",
                "user.name=llmman test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-m",
                "initial",
                "--quiet",
            ],
        );
        git(
            &repo,
            &["worktree", "add", "-b", "linked", "../worktree", "--quiet"],
        );

        let git_dir = PathBuf::from(git(
            &worktree,
            &["rev-parse", "--path-format=absolute", "--git-dir"],
        ));
        let common_dir = PathBuf::from(git(
            &worktree,
            &["rev-parse", "--path-format=absolute", "--git-common-dir"],
        ));
        let worktrees = git_dir.parent().unwrap();
        let common_alias = worktrees.join("common-alias");
        symlink(&common_dir, &common_alias).unwrap();
        std::fs::write(git_dir.join("commondir"), "../common-alias\n").unwrap();

        assert!(
            git_metadata_paths(&worktree).is_err(),
            "a relative commondir whose final component is a symlink must fail closed"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn seatbelt_confines_writes_to_the_listed_paths() {
        let profile = seatbelt_profile(
            &[PathBuf::from("/Users/me/src/app")],
            &[PathBuf::from("/Users/me/.claude.json")],
            Path::new("/Users/me/src/app"),
        )
        .unwrap();
        // Order matters: the later rule wins.
        let deny = profile.find("(deny file-write*)").unwrap();
        let allow = profile.find("(allow file-write*").unwrap();
        let protect = profile.rfind("(deny file-write* (regex").unwrap();
        assert!(profile.contains("(allow default)"));
        assert!(deny < allow && allow < protect, "{profile}");
        assert!(profile.contains("(subpath \"/Users/me/src/app\")"));
        assert!(profile.contains(r#"(regex #"^/Users/me/\.claude\.json")"#));
        assert!(profile[protect..].contains(r#"(regex #"^/Users/me/src/app/(.*/)?\.git(/|$)"))"#));
        assert!(profile.trim_end().ends_with(')'));
        // A quote would end the string early.
        assert!(seatbelt_profile(&[PathBuf::from("/tmp/a\"b")], &[], Path::new("/tmp")).is_err());
    }

    #[test]
    fn real_path_resolves_the_existing_part_of_a_missing_path() {
        let root = temp_dir("realpath");
        let missing = root.join("not").join("yet");
        assert_eq!(
            real_path(&missing),
            root.canonicalize().unwrap().join("not").join("yet")
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    fn sample_plan() -> Plan {
        Plan {
            image: "docker.io/docker/sandbox-templates:codex".into(),
            mounts: [
                ("/data/sandbox/codex", "/home/me", false),
                ("/home/me/src/app", "/home/me/src/app", false),
                ("/home/me/.codex", "/home/me/.codex", false),
                ("/home/me/src/app/.git", "/home/me/src/app/.git", true),
            ]
            .map(|(host, guest, read_only)| Mount {
                host: host.into(),
                guest: guest.into(),
                read_only,
            })
            .to_vec(),
            workdir: "/home/me/src/app".into(),
            env: BTreeMap::from([
                ("HOME".to_string(), "/home/me".to_string()),
                ("OPENAI_API_KEY".to_string(), "secret".to_string()),
            ]),
            command: strings(&["codex", "--profile", "llmman"]),
        }
    }

    #[test]
    fn docker_on_linux_runs_as_the_user_on_the_host_network() {
        let args = image_args(
            Sandbox::Docker,
            &sample_plan(),
            true,
            true,
            &Identity::User("1000:1000".into()),
        );
        assert_eq!(
            args,
            strings(&[
                "run",
                "--rm",
                "-i",
                "-t",
                "--network",
                "host",
                "--security-opt",
                "label=disable",
                "--user",
                "1000:1000",
                "-v",
                "/data/sandbox/codex:/home/me",
                "-v",
                "/home/me/src/app:/home/me/src/app",
                "-v",
                "/home/me/.codex:/home/me/.codex",
                "-v",
                "/home/me/src/app/.git:/home/me/src/app/.git:ro",
                "-w",
                "/home/me/src/app",
                "-e",
                "HOME",
                "-e",
                "OPENAI_API_KEY",
                "docker.io/docker/sandbox-templates:codex",
                "codex",
                "--profile",
                "llmman"
            ])
        );
        // The key's value is never on the command line.
        assert!(!args.iter().any(|a| a.contains("secret")));
    }

    #[test]
    fn podman_keeps_the_users_id_and_macos_engines_keep_their_network() {
        let linux = image_args(
            Sandbox::Podman,
            &sample_plan(),
            false,
            true,
            &Identity::KeepId,
        );
        assert!(linux.contains(&"--userns=keep-id".to_string()));
        assert!(!linux.contains(&"-t".to_string()));
        assert!(linux.windows(2).any(|w| w == ["--network", "host"]));

        let mac = image_args(
            Sandbox::Podman,
            &sample_plan(),
            true,
            false,
            &Identity::Image,
        );
        assert!(!mac.contains(&"--network".to_string()));
        assert!(!mac.contains(&"--security-opt".to_string()));
    }

    #[test]
    fn apple_container_and_microsandbox_take_the_same_mounts() {
        let apple = image_args(
            Sandbox::AppleContainer,
            &sample_plan(),
            true,
            false,
            &Identity::Image,
        );
        assert_eq!(&apple[..4], &strings(&["run", "--rm", "-i", "-t"])[..]);
        assert!(apple.ends_with(&strings(&[
            "docker.io/docker/sandbox-templates:codex",
            "codex",
            "--profile",
            "llmman"
        ])));

        let msb = image_args(
            Sandbox::Microsandbox,
            &sample_plan(),
            true,
            false,
            &Identity::Image,
        );
        assert_eq!(&msb[..3], &strings(&["run", "--net", "public,host"])[..]);
        assert!(!msb.contains(&"--rm".to_string()));
        assert!(msb.ends_with(&strings(&[
            "docker.io/docker/sandbox-templates:codex",
            "--",
            "codex",
            "--profile",
            "llmman"
        ])));
        assert!(msb
            .windows(2)
            .any(|w| w == ["-v", "/home/me/.codex:/home/me/.codex:host-perms=mirror"]));
        assert!(apple
            .windows(2)
            .any(|w| w == ["-v", "/home/me/.codex:/home/me/.codex"]));
    }

    #[test]
    fn the_guest_env_is_home_the_passthrough_and_the_launchers_last_word() {
        let host = |name: &str| match name {
            "TERM" => Some("xterm-256color".to_string()),
            "COPILOT_HOME" => Some("/home/me/copilot-state".to_string()),
            "PATH" => Some("/usr/bin".to_string()),
            _ => None,
        };
        let env = guest_env(
            Path::new("/home/me"),
            &[
                ("OLLAMA_HOST".into(), "http://127.0.0.1:17434".into()),
                (
                    "OLLAMA_HOST".into(),
                    "http://host.docker.internal:17434".into(),
                ),
            ],
            host,
        )
        .unwrap();
        assert_eq!(env["HOME"], "/home/me");
        assert_eq!(env["TERM"], "xterm-256color");
        assert_eq!(env["COPILOT_HOME"], "/home/me/copilot-state");
        assert_eq!(env["OLLAMA_HOST"], "http://host.docker.internal:17434");
        // The host's PATH would name directories the image does not have.
        assert!(!env.contains_key("PATH"));
    }

    #[test]
    fn mount_paths_refuse_the_engines_separators() {
        assert!(mount_path(Path::new("/home/me/src/a:b")).is_err());
        assert!(mount_path(Path::new("/home/me/src/a,b")).is_err());
        assert_eq!(mount_path(Path::new("/home/me")).unwrap(), "/home/me");
    }

    #[test]
    fn apple_dns_listing_is_matched_by_whole_domain() {
        let table = "DOMAIN\nhost.container.internal.\n";
        assert!(lists_domain(table, APPLE_HOST_DOMAIN));
        assert!(lists_domain("host.container.internal\n", APPLE_HOST_DOMAIN));
        assert!(!lists_domain(
            "myhost.container.internal.example\n",
            APPLE_HOST_DOMAIN
        ));
    }

    #[test]
    fn openshell_opens_the_daemons_port_to_every_program_and_execs_in_the_workspace() {
        assert_eq!(
            openshell_create_args("llmman-claude-1", "img", "/src/app"),
            strings(&[
                "sandbox",
                "create",
                "--name",
                "llmman-claude-1",
                "--from",
                "img",
                "--upload",
                "/src/app",
                "--detach"
            ])
        );
        assert_eq!(
            openshell_policy_args(
                "llmman-claude-1",
                &openshell_endpoint("http://host.openshell.internal:17434").unwrap()
            ),
            strings(&[
                "policy",
                "update",
                "llmman-claude-1",
                "--rule-name",
                "llmman",
                "--binary",
                "/**",
                "--add-endpoint",
                "host.openshell.internal:17434",
                "--wait"
            ])
        );
        assert_eq!(
            openshell_exec_args(
                "llmman-claude-1",
                "/sandbox/app",
                true,
                &[
                    ("B".into(), "2".into()),
                    ("A".into(), "1".into()),
                    ("B".into(), "3".into())
                ],
                &strings(&["claude", "--model", "m"])
            ),
            strings(&[
                "sandbox",
                "exec",
                "-n",
                "llmman-claude-1",
                "--tty",
                "--workdir",
                "/sandbox/app",
                "--env",
                "A=1",
                "--env",
                "B=3",
                "--",
                "claude",
                "--model",
                "m"
            ])
        );
    }

    #[test]
    fn openshell_allows_the_daemon_where_the_sandbox_reaches_it() {
        assert_eq!(
            openshell_endpoint("http://host.openshell.internal:17434").unwrap(),
            "host.openshell.internal:17434"
        );
        // A daemon elsewhere keeps its own name, so policy must name it.
        assert_eq!(
            openshell_endpoint("https://inferencebox").unwrap(),
            "inferencebox:443"
        );
        // `host_str` keeps an IPv6 address bracketed.
        assert_eq!(
            openshell_endpoint("http://[2001:db8::1]:17434").unwrap(),
            "[2001:db8::1]:17434"
        );
    }

    #[test]
    fn openshell_needs_a_grpc_gateway_on_this_machine_for_a_loopback_daemon() {
        let local = r#"{"status":"connected","server":"https://127.0.0.1:17670"}"#;
        let remote = r#"{"status":"connected","server":"https://gw.example:17670"}"#;
        assert!(check_openshell_status(local, true).is_ok());
        assert!(check_openshell_status(remote, true).is_err());
        // A daemon the remote gateway can reach itself is fine.
        assert!(check_openshell_status(remote, false).is_ok());
        let http_only = r#"{"status":"connected_http","server":"https://127.0.0.1:17670"}"#;
        assert!(check_openshell_status(http_only, true).is_err());
        assert!(check_openshell_status("not json", true).is_err());
    }

    #[test]
    fn state_or_workspace_containing_home_is_refused() {
        let home = Path::new("/home/me");
        assert!(contains_home(Path::new("/"), home));
        assert!(contains_home(home, home));
        assert!(!contains_home(Path::new("/home/me/.claude"), home));
    }

    #[test]
    fn read_only_mounts_use_each_engines_syntax() {
        let git = "/home/me/src/app/.git";
        let apple = image_args(
            Sandbox::AppleContainer,
            &sample_plan(),
            false,
            false,
            &Identity::Image,
        );
        let readonly = format!("type=bind,source={git},target={git},readonly");
        assert!(apple
            .windows(2)
            .any(|w| w == ["--mount", readonly.as_str()]));
        let msb = image_args(
            Sandbox::Microsandbox,
            &sample_plan(),
            false,
            false,
            &Identity::Image,
        );
        let ro = format!("{git}:{git}:ro");
        assert!(msb.windows(2).any(|w| w == ["-v", ro.as_str()]));
    }

    #[test]
    fn an_image_keeps_its_own_path() {
        let env = [
            ("PATH".to_string(), ":/host/bin".to_string()),
            ("OPENAI_API_KEY".to_string(), "k".to_string()),
        ];
        assert_eq!(without_path(&env), env[1..].to_vec());
    }

    #[test]
    fn guest_commands_run_the_programs_name_from_the_images_path() {
        assert_eq!(
            guest_command(Path::new("/usr/local/bin/claude"), &strings(&["-p", "hi"])).unwrap(),
            strings(&["claude", "-p", "hi"])
        );
        assert_eq!(
            guest_command(Path::new("aider"), &[]).unwrap(),
            strings(&["aider"])
        );
    }
}
