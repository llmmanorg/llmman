//! `llmman launch dsh` (DeepSeek Harness).

use std::path::{Path, PathBuf};

use anyhow::Context;

use super::common;
use super::{exec_with_env, find_on_path, has_flag, server, Effort};

/// The env var dsh's generated provider entry reads its key from, so no
/// key value is ever written to disk (same role as `qwen::QWEN_ENV_KEY`).
const DSH_API_KEY_ENV: &str = "LLMMAN_API_KEY";

pub(super) const DSH_MISSING: &str =
    "dsh is not installed, and there is no npx on PATH to run it with";

/// dsh: unlike qwen, hermes and codex, nothing here merges into a file
/// dsh reads by default. dsh's own `--patch` overlay mechanism lets both
/// files live under llmman's own config dir and be rewritten in full on
/// every launch, without ever touching the user's real `$DSH_HOME`.
///
/// Defaults to the `web` profile, but a caller-supplied `--profile`
/// after `--` wins instead — e.g. `--profile headless "<task>"` for a
/// one-shot, scriptable run, the same way every other flag here already
/// yields to what the caller explicitly asked for.
///
/// Both files sit at one fixed path, rewritten in place per launch:
/// dsh hot-reloads the settings document, so two *concurrent* launches
/// naming different models would retarget each other — accepted
/// deliberately, since a per-launch directory costs a cleanup hook on
/// every exit path (signals included) for a case that needs two
/// simultaneous sessions on different models to bite at all.
pub(super) fn launch_dsh(
    model: &str,
    api_key: &str,
    vision: bool,
    effort: Option<&Effort>,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let launcher = dsh_launcher(extra_args);
    if has_flag(launcher.args, "--patch", None) {
        anyhow::bail!("llmman launch dsh manages --patch itself; pass other dsh flags after --");
    }
    if let Some(command) = launcher.command {
        anyhow::bail!(
            "dsh's `{command}` command cannot be combined with the --patch llmman passes it.\n\
             Select a profile with `--profile {}` instead, or omit it for the default.",
            if command == "web" { "web" } else { "<name>" }
        );
    }
    let (bin, prefix) = find_dsh().ok_or_else(|| anyhow::anyhow!("{DSH_MISSING}"))?;

    let dir = dsh_config_dir()?;
    let settings_path = dir.join("settings.yaml");
    write_dsh_settings(&settings_path, model, vision, effort)?;
    let patch_path = dir.join("llmman.cordis.yml");
    write_dsh_patch(&patch_path, &settings_path)?;

    let mut args = prefix;
    if !args.is_empty() {
        // Said before it happens: this launch downloads a package.
        eprintln!("[llmman] dsh is not installed; running {DSH_NPM_PACKAGE} with npx");
    }
    args.extend(dsh_args(&patch_path, extra_args));
    exec_with_env(&bin, &args, &[(DSH_API_KEY_ENV, api_key)])
}

/// The npm package `npx` fetches when dsh isn't installed. Unpinned, so
/// a one-off run gets what a global install would have.
const DSH_NPM_PACKAGE: &str = "@deepseek-ai/dsh@latest";

/// dsh, and the arguments that must lead whatever it is handed: none for
/// an installed `dsh`, `--yes <package>` for the `npx` that stands in when
/// there is none. `find_integration_binary` resolves it the same way, so
/// the listing agrees with what a launch would run.
pub(super) fn find_dsh() -> Option<(PathBuf, Vec<String>)> {
    dsh_command(find_on_path("dsh"), || find_on_path("npx"))
}

/// Split from [`find_dsh`] so which binary wins can be asserted without
/// depending on what the test machine has installed.
fn dsh_command(
    dsh: Option<PathBuf>,
    npx: impl FnOnce() -> Option<PathBuf>,
) -> Option<(PathBuf, Vec<String>)> {
    match dsh {
        Some(bin) => Some((bin, Vec::new())),
        None => Some((npx()?, vec!["--yes".into(), DSH_NPM_PACKAGE.into()])),
    }
}

/// The tokens dsh reads as its own launcher flags, rather than forwards
/// to the selected profile's app. Verified against dsh 0.1.2-rc.1: it
/// stops at `--`, and also at the first token that isn't one of its own
/// options — `--dump-config sometask --patch <file>` reports `--patch`
/// and the file as app arguments and never reads it.
///
/// Scanning past either boundary reads app arguments as launcher ones:
/// an app-level `--profile` would count as a profile selection and drop
/// the default `web`, and an app-level `--patch` would be refused here
/// as though it were ours to manage. (The app may well reject that
/// token itself — headless answers `unknown option '--patch'` — but
/// that is dsh's own argument to make, in its own words.)
///
/// `--profile`/`--patch` are the two that take a value, which has to be
/// stepped over so it isn't mistaken for the first app argument; every
/// other dsh option (`--dump-config`, `--version`, ...) is a bare flag.
fn dsh_launcher(extra_args: &[String]) -> DshLauncher<'_> {
    let mut end = 0;
    let mut command = None;
    while let Some(arg) = extra_args.get(end) {
        if arg == "--" {
            break;
        }
        end += match arg.as_str() {
            // The two that take a value: step over it as well, so a
            // profile or path isn't read as a command or as the first
            // app argument (`--profile web`'s value is not the `web`
            // command).
            "--profile" | "--patch" => 2,
            // dsh's command spellings, which it refuses to combine with
            // any parent option — "web takes none of parent --profile,
            // --patch, ..." — so no argument order pairs one with the
            // `--patch` this injects.
            found @ ("web" | "plugin") => {
                command = command.or(Some(found));
                1
            }
            _ if arg.starts_with('-') => 1,
            // Anything else is dsh's first app argument.
            _ => break,
        };
    }
    DshLauncher {
        args: &extra_args[..end.min(extra_args.len())],
        command,
    }
}

/// dsh's own launcher section: the tokens it reads rather than forwards,
/// and the command spelling inside them, if any.
struct DshLauncher<'a> {
    args: &'a [String],
    command: Option<&'a str>,
}

/// The argv dsh is invoked with. `--patch` is always injected; the
/// default `web` profile is omitted when the caller already named one
/// (however spelled) after `--`, so `--profile headless "task"` selects
/// dsh's real one-shot mode instead of being appended onto `web`, which
/// does not accept it.
fn dsh_args(patch_path: &Path, extra_args: &[String]) -> Vec<String> {
    let mut args = Vec::new();
    if !has_flag(dsh_launcher(extra_args).args, "--profile", None) {
        args.push("web".to_string());
    }
    args.push("--patch".to_string());
    args.push(patch_path.to_string_lossy().into_owned());
    args.extend_from_slice(extra_args);
    args
}

/// `~/.config/llmman/launch/dsh`. Derived from `llmman.conf`'s own
/// directory rather than rebuilt by hand, so the two cannot drift.
/// dsh never looks here on its own; only `--patch` points it there.
pub(super) fn dsh_config_dir() -> anyhow::Result<PathBuf> {
    let conf = crate::config::user_path().context("no home directory")?;
    let dir = conf.parent().context("llmman.conf has no directory")?;
    Ok(dir.join("launch").join("dsh"))
}

/// The settings document `llmman.cordis.yml` points dsh at: registers
/// `llmman` as an `llm-pi-ai` provider route at this daemon's `/v1`, and
/// selects it as the `agent-default-model`. A `--variant` becomes the
/// route's default `reasoning`, among the model's `reasoningEfforts`.
fn write_dsh_settings(
    path: &Path,
    model: &str,
    vision: bool,
    effort: Option<&Effort>,
) -> anyhow::Result<()> {
    let quoted_model = common::yaml_quote(model);
    let base_url = common::yaml_quote(&format!("{}/v1", server()));
    // Claiming image input a text-only model can't serve would have dsh
    // attach what the daemon then rejects.
    let input = if vision { "[text, image]" } else { "[text]" };
    let (reasoning, efforts) = effort.map_or_else(Default::default, |e| {
        let map: Vec<String> = (e.levels.iter())
            .map(|&l| format!("{l}: {}", if l == "off" { "none" } else { l }))
            .collect();
        (
            format!("      reasoning: {}\n", e.default),
            format!("          reasoningEfforts: {{ {} }}\n", map.join(", ")),
        )
    });
    let contents = format!(
        "# Written by `llmman launch dsh`; edits are overwritten.\n\
         agent-default-model:\n  provider: llmman\n  model: {quoted_model}\n\
         llm-pi-ai:\n  providers:\n    llmman:\n      displayName: llmman\n      \
         apiKeyEnv: {DSH_API_KEY_ENV}\n      api: openai-completions\n      baseURL: {base_url}\n\
         {reasoning}      \
         models:\n        - id: {quoted_model}\n          name: {quoted_model}\n          input: {input}\n\
         {efforts}"
    );
    write_dsh_file(path, &contents)
}

/// dsh's patch shape: points its `settings` provider at the document above.
fn write_dsh_patch(path: &Path, settings_path: &Path) -> anyhow::Result<()> {
    write_dsh_file(path, &dsh_patch_document(settings_path))
}

/// Split from `write_dsh_patch` so a test can render a Windows-shaped
/// path on any platform: `yaml_quote` escapes the `\` separators, and
/// forgetting that is what once turned the Windows leg red.
fn dsh_patch_document(settings_path: &Path) -> String {
    let quoted_settings_path = common::yaml_quote(&settings_path.to_string_lossy());
    format!(
        "# Written by `llmman launch dsh`; edits are overwritten.\n\
         - id: settings\n  config:\n    path: {quoted_settings_path}\n"
    )
}

fn write_dsh_file(path: &Path, contents: &str) -> anyhow::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    }
    crate::fsutil::write_atomic(path, contents.as_bytes())
        .with_context(|| format!("write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daemon;

    /// A shortname like `qwen3.5:0.8b` must round-trip quoted, or the
    /// `:` breaks YAML parsing; the key must never appear literally.
    #[test]
    fn write_dsh_settings_points_at_llmman_with_the_key_in_the_environment() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-dsh-settings-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = dir.join("settings.yaml");
        write_dsh_settings(&path, "qwen3.5:0.8b", false, None).unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("provider: llmman"));
        assert!(contents.contains("model: \"qwen3.5:0.8b\""));
        assert!(contents.contains(&format!("apiKeyEnv: {DSH_API_KEY_ENV}")));
        assert!(contents.contains("api: openai-completions"));
        assert!(contents.contains(&format!("baseURL: \"{}/v1\"", daemon::server())));
        assert!(contents.contains("id: \"qwen3.5:0.8b\""));
        assert!(!contents.contains("apiKey:"), "no literal key in the file");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// dsh sends an image only to a model whose `input` lists one — and
    /// must not attach one to a text-only model the daemon would reject.
    #[test]
    fn write_dsh_settings_declares_image_input_only_for_a_vision_model() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-dsh-vision-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = dir.join("settings.yaml");
        write_dsh_settings(&path, "m", true, None).unwrap();
        assert!(std::fs::read_to_string(&path)
            .unwrap()
            .contains("input: [text, image]"));
        write_dsh_settings(&path, "m", false, None).unwrap();
        assert!(std::fs::read_to_string(&path)
            .unwrap()
            .contains("input: [text]"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The fallback that makes `llmman launch dsh` work without a global
    /// install — and stays out of the way of one that exists.
    #[test]
    fn dsh_falls_back_to_the_published_package_under_npx() {
        let dsh = PathBuf::from("/usr/local/bin/dsh");
        let npx = PathBuf::from("/usr/local/bin/npx");

        // An install wins, and npx is never even looked for.
        assert_eq!(
            dsh_command(Some(dsh.clone()), || panic!("npx looked up anyway")),
            Some((dsh, Vec::new()))
        );
        // Without one, npx runs the package: `--yes` so a first run
        // isn't blocked on a prompt, ahead of dsh's own arguments.
        assert_eq!(
            dsh_command(None, || Some(npx.clone())),
            Some((npx, vec!["--yes".to_string(), DSH_NPM_PACKAGE.to_string()]))
        );
        assert!(DSH_NPM_PACKAGE.starts_with("@deepseek-ai/dsh@"));
        // Neither: "dsh is not installed", not an npm error.
        assert_eq!(dsh_command(None, || None), None);
    }

    #[test]
    fn write_dsh_patch_names_the_settings_document() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-dsh-patch-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let settings_path = dir.join("settings.yaml");
        let patch_path = dir.join("llmman.cordis.yml");
        write_dsh_patch(&patch_path, &settings_path).unwrap();
        let contents = std::fs::read_to_string(&patch_path).unwrap();
        assert!(contents.contains("id: settings"));
        // Through `yaml_quote`, not the raw path: on Windows a path's
        // `\` separators are escaped in the document, so the raw string
        // never matches (a real red Windows CI leg).
        assert!(contents.contains(&format!(
            "path: {}",
            common::yaml_quote(&settings_path.to_string_lossy())
        )));
        let _ = std::fs::remove_dir_all(&dir);

        // A Windows-shaped path on every platform, so the escaping this
        // depends on is covered without needing the Windows CI leg to
        // be the thing that catches it (which is how it was caught).
        let win = Path::new(r"C:\Users\hb\.config\llmman\launch\dsh\settings.yaml");
        let rendered = dsh_patch_document(win);
        assert!(rendered.contains(r#"path: "C:\\Users\\hb\\"#), "{rendered}");
        assert!(!rendered.contains(r#"path: "C:\Users"#), "{rendered}");
    }

    /// Past dsh's own `--` boundary, a token is an app argument rather
    /// than a launcher flag (verified against dsh 0.1.2-rc.1), so
    /// neither check may scan there: a task whose text is `--patch`
    /// must not be refused, and one reading `--profile` must not
    /// suppress the default `web`.
    #[test]
    fn dsh_checks_stop_at_dshs_own_argument_boundary() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();

        // Asserted on the boundary helper, never by calling `launch_dsh`
        // itself: past the refusal it goes on to exec dsh and
        // `std::process::exit`, which on a machine that has dsh
        // installed would take the test runner with it.
        let forwarded_patch = args(&["--profile", "headless", "--", "--patch"]);
        assert_eq!(
            dsh_launcher(&forwarded_patch).args,
            args(&["--profile", "headless"])
        );
        assert!(!has_flag(
            dsh_launcher(&forwarded_patch).args,
            "--patch",
            None
        ));

        let forwarded_profile = args(&["--", "--profile", "headless"]);
        assert!(dsh_launcher(&forwarded_profile).args.is_empty());
        assert_eq!(
            dsh_args(Path::new("/p.yml"), &forwarded_profile),
            ["web", "--patch", "/p.yml", "--", "--profile", "headless"]
        );

        // The same boundary without a `--`: dsh stops at the first
        // token of its own it doesn't recognize, so a task's wording
        // is app text, not flags. `--profile headless` before it is
        // still dsh's, value stepped over rather than read as the
        // first app argument.
        let task_mentions_patch = args(&["--profile", "headless", "explain", "the", "--patch"]);
        assert_eq!(
            dsh_launcher(&task_mentions_patch).args,
            args(&["--profile", "headless"])
        );
        assert!(!has_flag(
            dsh_launcher(&task_mentions_patch).args,
            "--patch",
            None
        ));

        // An app-level `--profile` past that boundary must not suppress
        // the default `web`.
        let app_level_profile = args(&["sometask", "--profile", "headless"]);
        assert!(dsh_launcher(&app_level_profile).args.is_empty());
        assert_eq!(
            dsh_args(Path::new("/p.yml"), &app_level_profile),
            [
                "web",
                "--patch",
                "/p.yml",
                "sometask",
                "--profile",
                "headless"
            ]
        );

        // Bare flags take no value, and the `=`-joined spelling is dsh's
        // own either way.
        assert_eq!(
            dsh_launcher(&args(&["--dump-config", "task"])).args,
            args(&["--dump-config"])
        );
        assert_eq!(
            dsh_launcher(&args(&["--profile=headless", "task"])).args,
            args(&["--profile=headless"])
        );

        // dsh's command spellings are found where dsh itself reads them
        // (before any app argument), so `launch_dsh` can refuse them up
        // front: dsh rejects a command combined with a parent --patch,
        // which this always injects (verified against 0.1.2-rc.1).
        for command in ["web", "plugin"] {
            let via_command = args(&[command, "--port", "8080"]);
            assert_eq!(dsh_launcher(&via_command).command, Some(command));
            let err = launch_dsh("m", "k", false, None, &via_command).unwrap_err();
            assert!(err.to_string().contains("--profile"), "{err}");
        }
        // `--profile web`'s *value* is not the `web` command — refusing
        // it would break the most ordinary explicit invocation there is
        // (a real bug this caught, found only by running it).
        let profile_web = args(&["--profile", "web", "--no-open"]);
        assert_eq!(dsh_launcher(&profile_web).command, None);
        assert_eq!(
            dsh_args(Path::new("/p.yml"), &profile_web),
            ["--patch", "/p.yml", "--profile", "web", "--no-open"]
        );
        // Same for a patch path that happens to be named `web`.
        assert_eq!(dsh_launcher(&args(&["--patch", "web"])).command, None);
        // Past the boundary it is app text, not a command.
        assert_eq!(dsh_launcher(&args(&["sometask", "web"])).command, None);

        // All launcher flags, no app arguments: the whole slice is dsh's.
        let plain = args(&["--profile", "headless"]);
        assert_eq!(dsh_launcher(&plain).args, plain);
        // A value-taking flag with its value missing must not run past
        // the end of the slice.
        assert_eq!(
            dsh_launcher(&args(&["--profile"])).args,
            args(&["--profile"])
        );
    }

    /// A caller-supplied `--patch` after `--` must be refused, however spelled.
    #[test]
    fn launch_dsh_refuses_a_conflicting_patch_flag() {
        let word = vec!["--patch".to_string(), "/tmp/x.yml".to_string()];
        let err = launch_dsh("m", "k", false, None, &word).unwrap_err();
        assert!(err.to_string().contains("--patch"), "{err}");
        let joined = vec!["--patch=/tmp/x.yml".to_string()];
        let err = launch_dsh("m", "k", false, None, &joined).unwrap_err();
        assert!(err.to_string().contains("--patch"), "{err}");
    }

    /// A caller-supplied `--profile` (however spelled) must win over the
    /// default `web`, since `web` doesn't accept `--profile` at all;
    /// `--patch` is injected either way and nothing else is reordered.
    #[test]
    fn dsh_args_defaults_to_web_but_yields_to_a_caller_supplied_profile() {
        let path = Path::new("/tmp/x/llmman.cordis.yml");
        let none: Vec<String> = vec![];
        assert_eq!(
            dsh_args(path, &none),
            ["web", "--patch", "/tmp/x/llmman.cordis.yml"]
        );

        let headless = vec![
            "--profile".to_string(),
            "headless".to_string(),
            "hi".to_string(),
        ];
        assert_eq!(
            dsh_args(path, &headless),
            [
                "--patch",
                "/tmp/x/llmman.cordis.yml",
                "--profile",
                "headless",
                "hi"
            ]
        );

        let joined = vec!["--profile=headless".to_string(), "hi".to_string()];
        assert_eq!(
            dsh_args(path, &joined),
            [
                "--patch",
                "/tmp/x/llmman.cordis.yml",
                "--profile=headless",
                "hi"
            ]
        );
    }

    #[test]
    fn dsh_settings_start_the_route_at_the_variant() {
        let path = std::env::temp_dir()
            .join(format!("llmman-dsh-variant-{}", std::process::id()))
            .join("settings.yaml");
        let effort = Effort {
            default: "low",
            levels: vec!["off", "low", "high"],
        };
        write_dsh_settings(&path, "m", false, Some(&effort)).unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        let doc: serde_json::Value = yaml_serde::from_str(&contents).expect("valid YAML");
        let route = &doc["llm-pi-ai"]["providers"]["llmman"];
        assert_eq!(route["reasoning"], "low");
        assert_eq!(route["baseURL"], format!("{}/v1", daemon::server()));
        assert_eq!(
            route["models"][0]["reasoningEfforts"],
            serde_json::json!({ "off": "none", "low": "low", "high": "high" })
        );
    }
}
