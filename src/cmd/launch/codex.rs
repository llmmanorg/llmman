//! `llmman launch codex`.

use std::path::{Path, PathBuf};

use anyhow::Context;
use toml_edit::{value, DocumentMut, Item, Table};

use super::common::write_if_changed;
use super::models::{write_transient, ModelEntry, ModelList};
use super::optimizer::Optimizer;
use super::{exec_with_env, find_on_path, server};

/// codex: write the `llmman` profile overlay (`~/.codex/llmman.config.toml`,
/// a provider named `llmman` at our /v1 endpoint), then run codex with
/// `--profile llmman`, `api_key` as OPENAI_API_KEY and, for a launch with a
/// model, a catalog of the list (`-c model_catalog_json=`), which codex
/// shows in `/model` with each model's reasoning levels.
pub(super) fn launch_codex(
    model: &str,
    api_key: &str,
    models: &ModelList,
    extra_args: &[String],
    optimizer: Optimizer,
) -> anyhow::Result<()> {
    let added = optimizer.addition("codex")?;
    // Write codex config
    write_codex_config(extra_args)?;
    let catalog = if models.is_empty() {
        None
    } else {
        Some(write_transient(
            &codex_dir()?,
            "models",
            "json",
            &codex_model_catalog(models),
        )?)
    };

    // Regression: this used to pass a bare PathBuf::from("codex") straight
    // to exec_with_env instead of resolving it via find_on_path like every
    // other integration here does. That happened to work on Unix (bare
    // relative names go through $PATH search via execvp with no extension
    // needed), but on Windows, Command::status() calls CreateProcess
    // directly (not cmd.exe), which — unlike a shell — does not consult
    // PATHEXT to try .cmd/.bat alternatives for an extensionless name: it
    // only ever auto-appends a single ".exe". Since `npm install -g
    // @openai/codex` installs a "codex.cmd" shim on Windows, not a
    // "codex.exe", every real Windows codex launch failed with "program
    // not found" — a real E2E-verified failure, not a theoretical one.
    let bin = find_on_path("codex").ok_or_else(|| anyhow::anyhow!("codex is not installed"))?;

    let mut args: Vec<String> = Vec::new();
    if !model.is_empty() {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    // codex profile flag
    args.extend(["--profile".to_string(), "llmman".to_string()]);
    if let Some(catalog) = &catalog {
        args.extend(["-c".to_string(), catalog_override(catalog)]);
    }
    args.extend_from_slice(extra_args);
    args.splice(0..0, added.args.iter().cloned());

    if let Some(notice) = optimizer.notice("codex") {
        eprintln!("{notice}");
    }
    exec_with_env(&bin, &args, &[("OPENAI_API_KEY", api_key)])
}

/// The model codex last ran: `model` in its config.
pub(super) fn remembered_model() -> Option<String> {
    let text = std::fs::read_to_string(codex_dir().ok()?.join("config.toml")).ok()?;
    let doc: DocumentMut = text.parse().ok()?;
    doc.get("model")?.as_str().map(str::to_string)
}

const KEY: &str = "developer_instructions";

/// The `-c` value that sets codex's config `key` to the string `text`.
fn config_override(key: &str, text: &str) -> String {
    format!("{key}={}", toml_edit::Value::from(text))
}

/// The `-c` value that sets codex's `developer_instructions` to `text`.
pub(super) fn developer_instructions(text: &str) -> String {
    config_override(KEY, text)
}

/// The `-c` value that points codex's model catalog at `path`.
fn catalog_override(path: &Path) -> String {
    config_override("model_catalog_json", &path.display().to_string())
}

/// The `-c` value that points the `llmman` provider at `server`.
pub(super) fn server_override(server: &str) -> String {
    config_override("model_providers.llmman.base_url", &format!("{server}/v1"))
}

/// Whether a `-c`/`--config` before `--` sets `developer_instructions`.
fn sets_developer_instructions(extra_args: &[String]) -> bool {
    let args = super::before_terminator(extra_args);
    args.iter().enumerate().any(|(i, a)| {
        let setting = match a.as_str() {
            "-c" | "--config" => args.get(i + 1).map(String::as_str),
            _ => a.strip_prefix("--config=").or(a.strip_prefix("-c=")),
        };
        setting.is_some_and(|s| s.split('=').next().is_some_and(|k| k.trim() == KEY))
    })
}

/// What keeps `--optimizer cvmn` from working with codex: instructions of
/// the caller's, which the override would replace.
pub(super) fn cvmn_conflict(extra_args: &[String]) -> Option<&'static str> {
    let configured = codex_dir().ok().is_some_and(|dir| {
        ["config.toml", "llmman.config.toml"].iter().any(|name| {
            std::fs::read_to_string(dir.join(name))
                .ok()
                .and_then(|text| text.parse::<DocumentMut>().ok())
                .is_some_and(|doc| doc.contains_key(KEY))
        })
    });
    (configured || sets_developer_instructions(extra_args)).then_some(
        "developer_instructions of your own: llmman's would replace them.\nRemove them, or launch without --optimizer.",
    )
}

/// Writes codex's `llmman` profile.
///
/// Codex 0.134+ dropped support for `--profile <name>` reading a
/// `[profiles.<name>]` table out of `config.toml`: it now only overlays a
/// sibling `~/.codex/<name>.config.toml`, using top-level keys instead of a
/// `[profiles.<name>]` wrapper (see
/// <https://developers.openai.com/codex/config-advanced#profiles>). An
/// older llmman wrote the now-unsupported `[profiles.llmman]` form directly
/// into `config.toml`, which current codex refuses to start with at all
/// ("cannot be used while config.toml contains legacy ... table") — so any
/// leftover copy of that table is stripped from `config.toml` first, then
/// the real settings are (re)written to the profile overlay file codex
/// actually reads.
fn write_codex_config(extra_args: &[String]) -> anyhow::Result<()> {
    let config_dir = codex_dir()?;
    std::fs::create_dir_all(&config_dir)?;

    let config_path = config_dir.join("config.toml");
    let user_config = std::fs::read_to_string(&config_path);
    // An unreadable file may hold a decision about the project.
    let user_config_unreadable = user_config
        .as_ref()
        .is_err_and(|e| e.kind() != std::io::ErrorKind::NotFound);
    let user_config = user_config.ok();
    if let Some(existing) = user_config.as_deref() {
        if existing.contains("[profiles.llmman]") {
            std::fs::write(&config_path, strip_legacy_llmman_profile(existing))?;
        }
    }

    let profile_path = config_dir.join("llmman.config.toml");
    let profile = codex_profile(&server(), None);
    let existing = std::fs::read_to_string(&profile_path).ok();
    let overlay = codex_overlay(
        existing.as_deref(),
        user_config.as_deref(),
        &profile,
        codex_trust_root(extra_args)
            .filter(|_| !user_config_unreadable)
            .as_deref(),
    );
    write_if_changed(&profile_path, &overlay)
}

/// The overlay's keys that are llmman's; the rest of the file is codex's.
const CODEX_OVERLAY_OWNED: &[&str] = &["model_provider", "model_catalog_json", "model_providers"];

/// The overlay to write: `profile`, what codex saved in the overlay it ran
/// with (a trust decision, `[tui]` notices), and `trusted` as a trusted
/// project. A plain rewrite erased those, so codex asked to trust the folder
/// on every launch. A project with a `trust_level` already, in the overlay or
/// in `user_config` (which the overlay would override), keeps it; so does
/// one in a `user_config` that does not parse.
fn codex_overlay(
    existing: Option<&str>,
    user_config: Option<&str>,
    profile: &str,
    trusted: Option<&Path>,
) -> String {
    let mut doc: DocumentMut = profile.parse().expect("the profile is valid TOML");
    // An overlay that does not parse is replaced, as it always was.
    if let Some(old) = existing.and_then(|text| text.parse::<DocumentMut>().ok()) {
        for (key, item) in old.iter() {
            if !CODEX_OVERLAY_OWNED.contains(&key) {
                doc.insert(key, item.clone());
            }
        }
    }
    let user_config = user_config.map(|text| text.parse::<DocumentMut>());
    if let Some(root) = trusted {
        let key = root.display().to_string();
        let decided = |doc: &DocumentMut| {
            doc.get("projects")
                .and_then(Item::as_table_like)
                .and_then(|projects| projects.get(&key))
                .and_then(Item::as_table_like)
                .is_some_and(|project| project.contains_key("trust_level"))
        };
        let user_decided = match &user_config {
            Some(Ok(user)) => decided(user),
            Some(Err(_)) => true,
            None => false,
        };
        if !decided(&doc) && !user_decided {
            // Not tables: codex's to fix, not llmman's to overwrite.
            let projects = doc.entry("projects").or_insert_with(|| {
                let mut table = Table::new();
                table.set_implicit(true);
                Item::Table(table)
            });
            let project = projects
                .as_table_like_mut()
                .map(|projects| projects.entry(&key).or_insert(Item::Table(Table::new())))
                .and_then(Item::as_table_like_mut);
            if let Some(project) = project {
                project.insert("trust_level", value("trusted"));
            }
        }
    }
    doc.to_string()
}

/// The project codex would ask to trust: the repository `dir` is in, else
/// `dir` itself.
fn codex_project_root(dir: &Path) -> PathBuf {
    dir.ancestors()
        .find(|d| d.join(".git").exists())
        .unwrap_or(dir)
        .to_path_buf()
}

/// [`codex_project_root`] of the directory codex will run in: a forwarded
/// `--cd`/`-C` (relative to the working directory), else the working
/// directory. `None` when that directory is not there, as codex will refuse
/// to start in it.
fn codex_trust_root(extra_args: &[String]) -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    if let Some(cd) = super::common::forwarded_value(extra_args, "--cd", "-C") {
        dir = dir.join(cd);
    }
    if !dir.is_dir() {
        return None;
    }
    Some(codex_project_root(&codex_key_path(dir)))
}

/// `dir` as codex keys it: symlinks resolved, except on Windows, where
/// `canonicalize` gives a `\\?\` path codex does not use.
fn codex_key_path(dir: PathBuf) -> PathBuf {
    if cfg!(unix) {
        dir.canonicalize().unwrap_or(dir)
    } else {
        dir
    }
}

pub(super) fn codex_dir() -> anyhow::Result<PathBuf> {
    Ok(dirs::home_dir()
        .context("no home directory")?
        .join(".codex"))
}

/// The catalog's `context_window` for a model whose window is unknown;
/// ollama's fallback too. codex's catalog cannot omit the field, so it
/// guesses where every other integration stays quiet.
pub(super) const CODEX_FALLBACK_CONTEXT_WINDOW: u64 = 128_000;

/// The `model_catalog_json` for the list, one entry per model. The other
/// fields are ones codex requires, valued as ollama's `buildCodexModelEntry`
/// does.
fn codex_model_catalog(models: &ModelList) -> String {
    let entries: Vec<serde_json::Value> = models.entries().iter().map(codex_model_entry).collect();
    serde_json::to_string_pretty(&serde_json::json!({ "models": entries }))
        .expect("codex catalog serializes")
        + "\n"
}

/// What codex calls a variant, if it has one: `none`, `minimal`, `low`,
/// `medium`, `high`, `xhigh`; `max` is none of them, and a switch-only
/// template's `thinking` is `medium` (see `spell_variant`).
fn codex_reasoning_levels(entry: &ModelEntry) -> Vec<&'static str> {
    let mut levels: Vec<&'static str> = Vec::new();
    for variant in entry.variants() {
        let spelled = super::spell_variant("codex", &variant).ok().and_then(|l| {
            ["none", "minimal", "low", "medium", "high", "xhigh"]
                .into_iter()
                .find(|c| *c == l)
        });
        if let Some(level) = spelled.filter(|l| !levels.contains(l)) {
            levels.push(level);
        }
    }
    levels
}

/// One model's catalog entry: its plain id as `display_name`, its window,
/// its image input and its reasoning levels as presets.
fn codex_model_entry(entry: &ModelEntry) -> serde_json::Value {
    let input: &[&str] = if entry.vision {
        &["text", "image"]
    } else {
        &["text"]
    };
    let levels = codex_reasoning_levels(entry);
    let presets: Vec<serde_json::Value> = levels
        .iter()
        .map(|level| serde_json::json!({ "effort": level, "description": level }))
        .collect();
    let mut model = serde_json::json!({
        "slug": entry.reference,
        "display_name": entry.id,
        "context_window": entry.context_window.unwrap_or(CODEX_FALLBACK_CONTEXT_WINDOW),
        "shell_type": "default",
        "visibility": "list",
        "supported_in_api": true,
        "priority": 0,
        "truncation_policy": { "mode": "bytes", "limit": 10000 },
        "input_modalities": input,
        "base_instructions": "",
        "support_verbosity": true,
        "default_verbosity": "low",
        "supports_parallel_tool_calls": false,
        "supports_reasoning_summaries": false,
        "supported_reasoning_levels": presets,
        "experimental_supported_tools": [],
    });
    if let Some(default) = ["medium", "high", "low"]
        .into_iter()
        .find(|l| levels.contains(l))
    {
        model["default_reasoning_level"] = default.into();
    }
    model
}

/// The contents of `~/.codex/llmman.config.toml`: a provider of llmman's
/// own rather than `openai_base_url` on codex's built-in one, which codex
/// treats as WebSocket-capable and so opened every session with five
/// failed `ws://` attempts (~6s of "Reconnecting...") before HTTP.
fn codex_profile(server: &str, catalog: Option<&Path>) -> String {
    // A JSON string is also a valid TOML string.
    let catalog = catalog
        .map(|p| {
            let quoted = serde_json::Value::from(p.display().to_string());
            format!("model_catalog_json = {quoted}\n")
        })
        .unwrap_or_default();
    format!(
        "# Written by `llmman launch codex`; edits are overwritten.\n\
         model_provider = \"llmman\"\n\
         {catalog}\
         \n\
         [model_providers.llmman]\n\
         name = \"llmman\"\n\
         base_url = \"{server}/v1\"\n\
         env_key = \"OPENAI_API_KEY\"\n\
         wire_api = \"responses\"\n\
         supports_websockets = false\n"
    )
}

/// Removes a `[profiles.llmman]` table (and everything up to the next
/// top-level `[...]` header or end of file) from `config.toml`'s text —
/// the shape an older llmman wrote there, now rejected by current codex.
/// Line-based rather than a real TOML parser: this only ever needs to
/// undo llmman's own prior output, not handle arbitrary user TOML.
fn strip_legacy_llmman_profile(existing: &str) -> String {
    let mut out = String::new();
    let mut skipping = false;
    for line in existing.lines() {
        let trimmed = line.trim();
        if trimmed == "[profiles.llmman]" {
            skipping = true;
            continue;
        }
        if skipping && trimmed.starts_with('[') {
            skipping = false;
        }
        if skipping {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the codex config bug described on
    /// `write_codex_config`'s own doc comment: an older llmman's
    /// `[profiles.llmman]` table (a format current codex refuses to load
    /// at all) must be fully removed, leaving everything else in
    /// `config.toml` untouched.
    #[test]
    fn strip_legacy_llmman_profile_removes_only_that_table() {
        let existing = "\
[some_other_setting]
foo = \"bar\"

[profiles.llmman]
openai_base_url = \"http://127.0.0.1:17434/v1\"

[profiles.other]
model = \"gpt-5\"
";
        let cleaned = strip_legacy_llmman_profile(existing);
        assert!(!cleaned.contains("[profiles.llmman]"));
        assert!(!cleaned.contains("openai_base_url"));
        assert!(cleaned.contains("[some_other_setting]"));
        assert!(cleaned.contains("foo = \"bar\""));
        assert!(cleaned.contains("[profiles.other]"));
        assert!(cleaned.contains("model = \"gpt-5\""));
    }

    #[test]
    fn strip_legacy_llmman_profile_is_a_no_op_without_the_legacy_table() {
        let existing = "[profiles.other]\nmodel = \"gpt-5\"\n";
        assert_eq!(strip_legacy_llmman_profile(existing), existing);
    }

    #[test]
    fn strip_legacy_llmman_profile_handles_the_table_at_end_of_file() {
        let existing = "[profiles.llmman]\nopenai_base_url = \"http://127.0.0.1:17434/v1\"\n";
        assert_eq!(strip_legacy_llmman_profile(existing), "");
    }

    #[test]
    fn codex_profile_is_a_websocket_free_provider_at_the_daemon() {
        let profile: toml::Value = codex_profile("http://127.0.0.1:17434", None)
            .parse()
            .expect("valid TOML");
        assert_eq!(profile["model_provider"].as_str(), Some("llmman"));
        let provider = &profile["model_providers"]["llmman"];
        assert_eq!(
            provider["base_url"].as_str(),
            Some("http://127.0.0.1:17434/v1")
        );
        assert_eq!(provider["env_key"].as_str(), Some("OPENAI_API_KEY"));
        assert_eq!(provider["wire_api"].as_str(), Some("responses"));
        assert_eq!(provider["supports_websockets"].as_bool(), Some(false));
        assert!(
            profile.get("openai_base_url").is_none(),
            "the built-in openai provider is not the one in use"
        );
        assert!(
            profile.get("model_catalog_json").is_none(),
            "no model, no catalog to point at"
        );
    }

    const PROFILE: &str = "# Written by `llmman launch codex`; edits are overwritten.\n\
        model_provider = \"llmman\"\n\
        \n\
        [model_providers.llmman]\n\
        base_url = \"http://h/v1\"\n";

    fn overlay(existing: Option<&str>, trusted: Option<&str>) -> toml::Value {
        overlay_over(existing, None, trusted)
    }

    fn overlay_over(
        existing: Option<&str>,
        user_config: Option<&str>,
        trusted: Option<&str>,
    ) -> toml::Value {
        codex_overlay(existing, user_config, PROFILE, trusted.map(Path::new))
            .parse()
            .expect("the overlay is valid TOML")
    }

    /// codex asked to trust the folder on every launch because llmman
    /// rewrote the overlay codex had saved its answer in.
    #[test]
    fn overlay_trusts_the_project_and_keeps_what_codex_saved() {
        let first = overlay(None, Some("/work/repo"));
        assert_eq!(
            first["projects"]["/work/repo"]["trust_level"].as_str(),
            Some("trusted")
        );
        assert_eq!(first["model_provider"].as_str(), Some("llmman"));

        // codex then saves a notice it has shown, and another project it
        // was asked about; the next launch must not lose either.
        let saved = format!(
            "{}\n[tui]\nscreen_reader_detection_done = true\n\n[projects.\"/work/other\"]\ntrust_level = \"trusted\"\n",
            codex_overlay(None, None, PROFILE, Some(Path::new("/work/repo")))
        );
        let next = overlay(Some(&saved), Some("/work/repo"));
        assert_eq!(
            next["tui"]["screen_reader_detection_done"].as_bool(),
            Some(true)
        );
        assert_eq!(
            next["projects"]["/work/other"]["trust_level"].as_str(),
            Some("trusted")
        );
        assert_eq!(
            next["projects"]["/work/repo"]["trust_level"].as_str(),
            Some("trusted")
        );
    }

    #[test]
    fn overlay_replaces_llmmans_own_keys_and_nothing_else() {
        let stale = "model_provider = \"old\"\n\
            model_catalog_json = \"/gone.json\"\n\
            [model_providers.llmman]\n\
            base_url = \"http://stale/v1\"\n\
            [model_providers.mine]\n\
            base_url = \"http://mine\"\n";
        let next = overlay(Some(stale), None);
        assert_eq!(next["model_provider"].as_str(), Some("llmman"));
        assert!(
            next.get("model_catalog_json").is_none(),
            "no model, no catalog"
        );
        assert_eq!(
            next["model_providers"]["llmman"]["base_url"].as_str(),
            Some("http://h/v1")
        );
        assert!(next["model_providers"].get("mine").is_none());
        assert!(next.get("projects").is_none(), "no project to trust");
    }

    /// A folder codex was told not to trust stays untrusted: that was the
    /// user's answer.
    #[test]
    fn overlay_keeps_an_existing_decision_about_the_project() {
        let saved = "[projects.\"/work/repo\"]\ntrust_level = \"untrusted\"\n";
        let next = overlay(Some(saved), Some("/work/repo"));
        assert_eq!(
            next["projects"]["/work/repo"]["trust_level"].as_str(),
            Some("untrusted")
        );
    }

    /// The overlay is layered over `~/.codex/config.toml`, so a trust level
    /// added to it would override the one the user set there.
    #[test]
    fn overlay_defers_to_a_decision_in_the_user_config() {
        for level in ["untrusted", "trusted"] {
            let user = format!("[projects.\"/work/repo\"]\ntrust_level = \"{level}\"\n");
            let next = overlay_over(None, Some(&user), Some("/work/repo"));
            assert!(next.get("projects").is_none(), "{level}: nothing added");
        }
        // another project's entry decides nothing about this one
        let user = "[projects.\"/work/other\"]\ntrust_level = \"untrusted\"\n";
        let next = overlay_over(None, Some(user), Some("/work/repo"));
        assert_eq!(
            next["projects"]["/work/repo"]["trust_level"].as_str(),
            Some("trusted")
        );
    }

    /// A user config that does not parse may hold an `untrusted` that codex
    /// reads, so nothing is trusted over it.
    #[test]
    fn overlay_trusts_nothing_over_a_user_config_that_does_not_parse() {
        let next = overlay_over(
            None,
            Some("[projects.\"/work/repo\"\ntrust_level = "),
            Some("/work/repo"),
        );
        assert!(next.get("projects").is_none());
        assert_eq!(next["model_provider"].as_str(), Some("llmman"));
    }

    /// A project table with no `trust_level` is no decision: codex asks.
    #[test]
    fn overlay_adds_the_trust_level_a_project_table_lacks() {
        let saved = "[projects.\"/work/repo\"]\nnote = \"mine\"\n";
        let next = overlay(Some(saved), Some("/work/repo"));
        let project = &next["projects"]["/work/repo"];
        assert_eq!(project["trust_level"].as_str(), Some("trusted"));
        assert_eq!(project["note"].as_str(), Some("mine"));
        // and a user-config table that lacks one is no decision either
        let next = overlay_over(None, Some(saved), Some("/work/repo"));
        assert_eq!(
            next["projects"]["/work/repo"]["trust_level"].as_str(),
            Some("trusted")
        );
    }

    #[test]
    fn overlay_starts_over_from_one_that_does_not_parse() {
        let next = overlay(Some("this is [not toml"), Some("/work/repo"));
        assert_eq!(next["model_provider"].as_str(), Some("llmman"));
        assert_eq!(
            next["projects"]["/work/repo"]["trust_level"].as_str(),
            Some("trusted")
        );
    }

    /// An unchanged overlay is not rewritten (`write_if_changed`), so
    /// writing it twice must give the same text.
    #[test]
    fn overlay_is_stable() {
        let once = codex_overlay(None, None, PROFILE, Some(Path::new("/work/repo")));
        let twice = codex_overlay(Some(&once), None, PROFILE, Some(Path::new("/work/repo")));
        assert_eq!(once, twice);
    }

    #[test]
    fn project_root_is_the_enclosing_repository_else_the_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let sub = dir.join("repo/src/deep");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::create_dir_all(dir.join("repo/.git")).unwrap();
        assert_eq!(codex_project_root(&sub), dir.join("repo"));
        assert_eq!(codex_project_root(&dir.join("repo")), dir.join("repo"));
        // a worktree's `.git` is a file
        std::fs::create_dir_all(dir.join("wt/sub")).unwrap();
        std::fs::write(dir.join("wt/.git"), "gitdir: ../elsewhere").unwrap();
        assert_eq!(codex_project_root(&dir.join("wt/sub")), dir.join("wt"));
    }

    /// `--cd`/`-C` makes codex run in another directory, which is the one
    /// it asks about, so that is the one to trust.
    #[test]
    fn trust_root_follows_a_forwarded_working_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let sub = dir.join("repo/src");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::create_dir_all(dir.join("repo/.git")).unwrap();
        let repo = codex_key_path(dir.join("repo"));
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let sub = sub.display().to_string();
        for forwarded in [
            args(&["--cd", &sub]),
            args(&["-C", &sub]),
            args(&[&format!("--cd={sub}"), "--", "-C", "/elsewhere"]),
        ] {
            assert_eq!(codex_trust_root(&forwarded).as_deref(), Some(&*repo));
        }
        // a directory that is not there is not trusted
        let gone = dir.join("gone").display().to_string();
        assert_eq!(codex_trust_root(&args(&["-C", &gone])), None);
        // none forwarded: the working directory
        assert_eq!(
            codex_trust_root(&[]),
            codex_trust_root(&args(&["--", "-C", &gone]))
        );
        assert!(codex_trust_root(&[]).is_some());
    }

    #[test]
    fn codex_profile_names_the_catalog_it_was_given() {
        let path = PathBuf::from("/home/we\"ird/.codex/llmman-model.json");
        let profile: toml::Value = codex_profile("http://h", Some(&path))
            .parse()
            .expect("valid TOML");
        assert_eq!(
            profile["model_catalog_json"].as_str(),
            Some(path.to_str().unwrap())
        );
    }

    fn row(reference: &str, window: Option<u64>, vision: bool, thinking: &[&str]) -> ModelEntry {
        ModelEntry {
            context_window: window,
            vision,
            thinking: Some(thinking.iter().map(|t| t.to_string()).collect()),
            ..ModelEntry::unlisted(reference)
        }
    }

    #[test]
    fn the_catalog_has_every_row_with_its_plain_id_window_image_input_and_levels() {
        let models = ModelList::of(vec![
            row("docker.io/ai/m", Some(32768), true, &["none", "thinking"]),
            row(
                "llmman.provider/p/gpt",
                None,
                false,
                &["low", "high", "max"],
            ),
            row("llmman.provider/p/plain", Some(8192), false, &[]),
        ]);
        let catalog: serde_json::Value =
            serde_json::from_str(&codex_model_catalog(&models)).expect("valid JSON");
        let rows = catalog["models"].as_array().unwrap();
        assert_eq!(rows.len(), 3);
        let entry = &rows[0];
        assert_eq!(
            entry["input_modalities"],
            serde_json::json!(["text", "image"])
        );
        assert_eq!(entry["slug"], "docker.io/ai/m");
        assert_eq!(entry["display_name"], "m");
        assert_eq!(entry["context_window"], 32768);
        // A switch-only template's `thinking` is codex's `medium`.
        let efforts = |row: &serde_json::Value| -> Vec<String> {
            row["supported_reasoning_levels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|l| l["effort"].as_str().unwrap().to_string())
                .collect()
        };
        assert_eq!(efforts(entry), ["none", "medium"]);
        assert_eq!(entry["default_reasoning_level"], "medium");
        // `max` is not a codex level; no window is codex's fallback.
        assert_eq!(efforts(&rows[1]), ["low", "high"]);
        assert_eq!(rows[1]["display_name"], "gpt");
        assert_eq!(rows[1]["context_window"], CODEX_FALLBACK_CONTEXT_WINDOW);
        assert_eq!(rows[1]["input_modalities"], serde_json::json!(["text"]));
        assert!(efforts(&rows[2]).is_empty());
        assert!(rows[2].get("default_reasoning_level").is_none());
        // Fields codex requires of an entry.
        for key in [
            "context_window",
            "shell_type",
            "visibility",
            "supported_in_api",
            "priority",
            "truncation_policy",
            "support_verbosity",
            "supported_reasoning_levels",
            "experimental_supported_tools",
        ] {
            for row in rows {
                assert!(row.get(key).is_some(), "missing {key}");
            }
        }
    }

    #[test]
    fn the_catalog_is_passed_by_path_and_holds_no_credential() {
        let path = PathBuf::from("/home/we\"ird/.codex/llmman-models-1.json");
        let over = catalog_override(&path);
        assert!(over.starts_with("model_catalog_json="), "{over}");
        let value: toml::Value = over
            .replacen("model_catalog_json=", "x=", 1)
            .parse()
            .unwrap();
        assert_eq!(value["x"].as_str(), Some(path.to_str().unwrap()));
        let models = ModelList::of(vec![row("m", None, false, &[])]);
        assert!(!codex_model_catalog(&models).contains("key"));
    }

    #[test]
    fn developer_instructions_of_the_callers_are_seen_before_a_second_terminator() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(sets_developer_instructions(&args(&[
            "-c",
            "developer_instructions=\"x\""
        ])));
        assert!(sets_developer_instructions(&args(&[
            "--config=developer_instructions='x'"
        ])));
        // Text that only mentions the key is not a setting.
        assert!(!sets_developer_instructions(&args(&[
            "--cd",
            "/w/developer_instructions-demo",
            "exec",
            "explain developer_instructions"
        ])));
        assert!(!sets_developer_instructions(&args(&["-c", "model=\"m\""])));
        assert!(!sets_developer_instructions(&args(&[
            "exec",
            "--",
            "-c",
            "developer_instructions=x"
        ])));
    }

    #[test]
    fn the_override_for_developer_instructions_is_valid_toml_that_keeps_the_text() {
        let text = "Be brief.\nQuote: \"a\" and \\ and 'b'";
        let arg = developer_instructions(text);
        let doc: DocumentMut = arg.parse().unwrap();
        assert_eq!(doc["developer_instructions"].as_str(), Some(text));
    }
}
