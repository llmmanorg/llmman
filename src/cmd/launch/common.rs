//! Helpers shared by the per-integration launchers.

use std::path::{Path, PathBuf};

use anyhow::Context;

const MISE_TOML: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/.mise.toml"));

/// Launchers that hand the daemon's real client key to another process must
/// not send it to a remote cleartext endpoint. Call this before anything
/// probes or starts the daemon; loopback HTTP and remote HTTPS remain valid.
pub(super) fn check_daemon_key_transport(integration: &str) -> anyhow::Result<()> {
    check_daemon_key_transport_to(
        integration,
        crate::auth::client_key().is_some(),
        crate::daemon::connects_securely(),
        &crate::daemon::server(),
    )
}

fn check_daemon_key_transport_to(
    integration: &str,
    has_client_key: bool,
    connects_securely: bool,
    server: &str,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        !has_client_key || connects_securely,
        "{integration} needs a local llmman serve, or one over TLS: LLMMAN_HOST points at \
         {server}, and the daemon API key would cross the network in cleartext."
    );
    Ok(())
}

pub(super) fn write_yaml_merged(
    path: &Path,
    label: &str,
    merge: impl FnOnce(&serde_json::Value) -> serde_json::Value,
) -> anyhow::Result<()> {
    write_structured_merged(
        path,
        label,
        ("YAML", "yml.bak"),
        |text| Ok(yaml_serde::from_str(text)?),
        |value| Ok(yaml_serde::to_string(value)?),
        |raw, backup| {
            if !backup.exists() {
                return true;
            }
            yaml_serde::from_str::<serde_json::Value>(raw)
                .and_then(|value| yaml_serde::to_string(&value))
                .is_ok_and(|canonical| canonical != raw)
        },
        merge,
    )
}

/// Renders `s` as a double-quoted YAML scalar, escaping backslashes and
/// double quotes — enough to keep any value we generate (a model name, a
/// URL) a literal string regardless of YAML keywords or metacharacters
/// it might contain.
pub(super) fn yaml_quote(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// The value of the last `--model`/`-m` after `--`, as a word or
/// `=`-joined, the forms `has_flag` takes; yargs keeps the last too.
/// Nothing after a second `--` counts, as in `has_forwarded_model_flag`.
pub(super) fn forwarded_model(extra_args: &[String]) -> Option<&str> {
    let mut found = None;
    let mut args = super::before_terminator(extra_args)
        .iter()
        .map(String::as_str);
    while let Some(a) = args.next() {
        match a {
            "--model" | "-m" => found = args.next().or(found),
            _ => {
                if let Some(v) = a.strip_prefix("--model=").or_else(|| a.strip_prefix("-m=")) {
                    found = Some(v);
                }
            }
        }
    }
    found.filter(|v| !v.is_empty())
}

/// A leading `~` is `home`, as Qwen Code's `Storage.resolvePath` reads
/// it; a quoted export leaves it for the program to expand.
pub(super) fn expand_tilde(dir: &str, home: &Path) -> PathBuf {
    match dir.strip_prefix('~') {
        Some(rest) if rest.is_empty() || rest.starts_with(['/', '\\']) => {
            home.join(rest.trim_start_matches(['/', '\\']))
        }
        _ => PathBuf::from(dir),
    }
}

/// Rewrites the JSON object at `path` through `merge`, leaving a file it
/// does not understand alone and saying so under `label`.
///
/// The settings files these integrations read are the user's as much as
/// llmman's, so: comments survive a parse (`strip_json_comments`) and the
/// text carrying them is kept as `.bak` before the first rewrite that
/// would drop them; a rendering that already matches is not written at
/// all; and the write is atomic, since a half-written settings file is
/// worse than a stale one. A leading BOM is stripped because a
/// node-based reader tolerates one and `serde_json` does not.
pub(super) fn write_json_merged(
    path: &Path,
    label: &str,
    merge: impl FnOnce(&serde_json::Value) -> serde_json::Value,
) -> anyhow::Result<()> {
    write_structured_merged(
        path,
        label,
        ("JSON", "json.bak"),
        |text| {
            let text = text.trim_start_matches('\u{feff}').trim();
            Ok(serde_json::from_str(&strip_json_comments(text))?)
        },
        |value| {
            let mut out = serde_json::to_string_pretty(value)?;
            out.push('\n');
            Ok(out)
        },
        |raw, backup| !backup.exists() || strip_json_comments(raw) != raw,
        merge,
    )
}

/// Parse, merge, back up, and atomically rewrite a user-owned structured
/// settings file. Format-specific parsing and rendering stay at the call site.
fn write_structured_merged(
    path: &Path,
    label: &str,
    format: (&str, &str),
    parse: impl FnOnce(&str) -> anyhow::Result<serde_json::Value>,
    serialize: impl FnOnce(&serde_json::Value) -> anyhow::Result<String>,
    should_backup: impl FnOnce(&str, &Path) -> bool,
    merge: impl FnOnce(&serde_json::Value) -> serde_json::Value,
) -> anyhow::Result<()> {
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => Some(raw),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e).with_context(|| format!("read {}", path.display())),
    };
    let existing = match raw.as_deref().map(str::trim) {
        None | Some("") => serde_json::json!({}),
        Some(text) => match parse(text) {
            Ok(value) if value.is_object() => value,
            _ => {
                eprintln!(
                    "[llmman] {label}: {} is not a {} object; leaving it alone",
                    path.display(),
                    format.0
                );
                return Ok(());
            }
        },
    };
    let merged = merge(&existing);
    if merged == existing {
        return Ok(());
    }
    let dir = path.parent().context("settings path has no directory")?;
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    if let Some(raw) = &raw {
        let backup = path.with_extension(format.1);
        if should_backup(raw, &backup) {
            std::fs::copy(path, &backup)
                .with_context(|| format!("back up {} to {}", path.display(), backup.display()))?;
        }
    }
    let out = serialize(&merged)?;
    crate::fsutil::write_atomic(path, out.as_bytes())
        .with_context(|| format!("write {}", path.display()))
}

/// `//` and `/* */` comments outside strings replaced by spaces, so a
/// parse error still points at the right place; what `strip-json-comments`
/// does for Qwen Code before `JSON.parse`.
fn strip_json_comments(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    let mut in_string = false;
    let mut escaped = false;
    while let Some(c) = chars.next() {
        if in_string {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                out.push(c);
            }
            '/' if chars.peek() == Some(&'/') => {
                out.push(' ');
                while chars.peek().is_some_and(|&n| n != '\n') {
                    chars.next();
                    out.push(' ');
                }
            }
            '/' if chars.peek() == Some(&'*') => {
                chars.next();
                out.push_str("  ");
                let mut prev = ' ';
                for n in chars.by_ref() {
                    out.push(if n == '\n' { '\n' } else { ' ' });
                    if prev == '*' && n == '/' {
                        break;
                    }
                    prev = n;
                }
            }
            _ => out.push(c),
        }
    }
    out
}

/// The object at `key` in `parent`, put there if absent or not an object.
pub(super) fn object_under<'a>(
    parent: &'a mut serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> &'a mut serde_json::Map<String, serde_json::Value> {
    let slot = parent.entry(key).or_insert_with(|| serde_json::json!({}));
    if !slot.is_object() {
        *slot = serde_json::json!({});
    }
    slot.as_object_mut().expect("set to an object just above")
}

/// The explicit agent directory understood by both Pi and OMP.
pub(super) fn configured_pi_agent_dir() -> anyhow::Result<Option<PathBuf>> {
    let Some(dir) = std::env::var("PI_CODING_AGENT_DIR")
        .ok()
        .filter(|dir| !dir.trim().is_empty())
    else {
        return Ok(None);
    };
    let dir = dir.trim();
    if !dir.starts_with('~') {
        return Ok(Some(PathBuf::from(dir)));
    }
    let home = crate::config::home_dir().context("no home directory")?;
    Ok(Some(expand_tilde(dir, &home)))
}

/// The model entry Pi and OMP are told about: its display name, what it can
/// take in, whether it thinks, and how much it can hold. Every example in
/// Pi's own `models.md`
/// declares these, and what it assumes for an entry that omits them is
/// not written down, so they are stated rather than left to it.
pub(super) fn pi_model_entry(
    model: &str,
    reasons: bool,
    vision: bool,
    context_window: Option<u64>,
) -> serde_json::Value {
    let input: &[&str] = if vision {
        &["text", "image"]
    } else {
        &["text"]
    };
    let mut entry = serde_json::json!({
        "id": model,
        "name": model,
        "input": input,
    });
    if reasons {
        entry["reasoning"] = serde_json::json!(true);
    }
    if let Some(context) = context_window {
        entry["contextWindow"] = serde_json::json!(context);
    }
    entry
}

/// Merge one model into a named provider while retaining unrelated providers,
/// provider options, and models. OMP also retains unowned fields on the
/// matching model; Pi rebuilds its matching entry from daemon metadata.
pub(super) fn provider_models_merged(
    existing: &serde_json::Value,
    provider_name: &str,
    entry: &serde_json::Value,
    preserve_existing_model_fields: bool,
    configure: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>),
) -> serde_json::Value {
    let mut root = existing.clone();
    let Some(root_map) = root.as_object_mut() else {
        return serde_json::json!({});
    };
    let provider = object_under(object_under(root_map, "providers"), provider_name);
    let old_models = provider
        .get("models")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut replaced = false;
    let mut models = Vec::with_capacity(old_models.len() + 1);
    for old in old_models {
        if old.get("id") != entry.get("id") {
            models.push(old);
            continue;
        }
        if replaced {
            continue;
        }
        let mut merged = if preserve_existing_model_fields {
            old
        } else {
            serde_json::json!({})
        };
        if let (Some(merged), Some(update)) = (merged.as_object_mut(), entry.as_object()) {
            merged.extend(update.clone());
        } else {
            merged = entry.clone();
        }
        models.push(merged);
        replaced = true;
    }
    if !replaced {
        models.push(entry.clone());
    }
    configure(provider);
    provider.insert("models".into(), serde_json::json!(models));
    root
}

/// The `npm:` tools in `.mise.toml`, package name to version.
pub(super) fn npm_pins() -> anyhow::Result<std::collections::BTreeMap<String, String>> {
    use std::str::FromStr as _;
    let value = toml::Value::from_str(MISE_TOML).context("failed to parse .mise.toml")?;
    let tools = value
        .get("tools")
        .and_then(|v| v.as_table())
        .context(".mise.toml has no [tools] table")?;
    let mut pins = std::collections::BTreeMap::new();
    for (key, val) in tools {
        let Some(pkg) = key.strip_prefix("npm:") else {
            continue;
        };
        let version = match val {
            toml::Value::String(s) => s.clone(),
            toml::Value::Table(t) => t
                .get("version")
                .and_then(|v| v.as_str())
                .with_context(|| format!("npm:{pkg} table has no string `version`"))?
                .to_owned(),
            other => anyhow::bail!("npm:{pkg} has unexpected TOML type: {other:?}"),
        };
        pins.insert(pkg.to_owned(), version);
    }
    Ok(pins)
}

pub(super) fn npm_install_spec(pkg: &str) -> anyhow::Result<String> {
    let pins = npm_pins()?;
    let version = pins
        .get(pkg)
        .with_context(|| format!("`{pkg}` is not pinned in llmman's .mise.toml"))?;
    Ok(format!("{pkg}@{version}"))
}

pub(super) fn npm_missing(name: &str, pkg: &str) -> anyhow::Error {
    let spec = match npm_install_spec(pkg) {
        Ok(s) => s,
        Err(e) => return e,
    };
    anyhow::anyhow!(
        "{name} is not installed\n\n\
         From an llmman source checkout, install the pinned release with mise:\n  \
         mise install npm:{pkg}\n\
         mise puts it on PATH only through `mise exec`, its shims, or shell activation, so retry with:\n  \
         mise exec -- llmman launch {name}\n\
         or activate mise in your shell first: eval \"$(mise activate bash)\" (or zsh/fish)\n\n\
         From a release binary, install globally with npm:\n  \
         npm install -g {spec}\n\n\
         Ensure npm's global bin directory is on PATH, then retry."
    )
}

pub(super) fn npm_install(npm: &std::path::Path, spec: &str) -> anyhow::Result<()> {
    let status = std::process::Command::new(npm)
        .args(["install", "-g", spec])
        .status()
        .with_context(|| format!("failed to run {}", npm.display()))?;
    anyhow::ensure!(status.success(), "npm install -g {spec} failed: {status}");
    Ok(())
}

pub(super) fn install_missing_with_npm(name: &str, pkg: &str) -> anyhow::Result<()> {
    let spec = npm_install_spec(pkg)?;
    let Some(npm) = super::find_on_path("npm") else {
        return Err(npm_missing(name, pkg));
    };
    if !std::io::IsTerminal::is_terminal(&std::io::stdin())
        || !std::io::IsTerminal::is_terminal(&std::io::stderr())
    {
        return Err(npm_missing(name, pkg));
    }
    eprint!("[llmman] {name} is not installed. Install {spec} globally with npm? [y/N] ");
    use std::io::Write as _;
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    if !super::accepts_prompt(&line) {
        anyhow::bail!("{name} installation cancelled");
    }
    eprintln!("[llmman] {name} not found; installing {spec} via npm...");
    npm_install(&npm, &spec)?;
    if super::find_on_path(name).is_none() {
        anyhow::bail!(
            "npm installed {spec}, but {name} is still not on PATH\n\n\
             Add npm's global bin directory (`npm prefix -g`) to PATH, then retry."
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{pair_context_window, served_context_window};
    use super::*;

    #[test]
    fn a_real_daemon_key_never_crosses_remote_cleartext() {
        let remote_http = "http://192.0.2.1:17434";
        let error = check_daemon_key_transport_to("agent", true, false, remote_http)
            .unwrap_err()
            .to_string();
        assert!(error.contains("agent"));
        assert!(error.contains(remote_http));
        assert!(error.contains("cleartext"));

        assert!(check_daemon_key_transport_to("agent", false, false, remote_http).is_ok());
        assert!(check_daemon_key_transport_to("agent", true, true, remote_http).is_ok());
    }

    /// Regression test for a real CodeRabbit finding: an unquoted model
    /// value in generated YAML could be misparsed as a non-string
    /// (`null`, `true`, ...) or broken outright by metacharacters.
    #[test]
    fn yaml_quote_escapes_keywords_and_metacharacters() {
        assert_eq!(yaml_quote("qwen3.5:0.8b"), "\"qwen3.5:0.8b\"");
        assert_eq!(yaml_quote("null"), "\"null\"");
        assert_eq!(yaml_quote("true"), "\"true\"");
        assert_eq!(
            yaml_quote(r#"a "quoted" \ value"#),
            r#""a \"quoted\" \\ value""#
        );
    }

    /// The last forwarded model wins, in either spelling; none is none.
    #[test]
    fn forwarded_model_takes_the_last_spelling() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(forwarded_model(&args(&["--model", "b"])), Some("b"));
        assert_eq!(forwarded_model(&args(&["-m=b", "--model", "c"])), Some("c"));
        assert_eq!(forwarded_model(&args(&["--model=b", "-m", "c"])), Some("c"));
        assert_eq!(forwarded_model(&args(&["-p", "x"])), None);
        assert_eq!(forwarded_model(&args(&["--model"])), None);
        assert_eq!(forwarded_model(&args(&["--model="])), None);
        // After a second `--` it is not a flag.
        assert_eq!(forwarded_model(&args(&["--", "--model", "c"])), None);
        assert_eq!(
            forwarded_model(&args(&["--model", "b", "--", "--model", "c"])),
            Some("b")
        );
    }

    /// Comments go, as `strip-json-comments` takes them out for Qwen Code,
    /// and nothing else moves: not a `//` inside a string, not a column.
    #[test]
    fn strip_json_comments_keeps_strings_and_columns() {
        let raw =
            "{\n  // note\n  \"url\": \"http://h//v1\", /* block\n  */ \"q\": \"a\\\"//b\"\n}\n";
        let stripped = strip_json_comments(raw);
        assert_eq!(stripped.chars().count(), raw.chars().count());
        assert_eq!(stripped.lines().count(), raw.lines().count());
        let v: serde_json::Value = serde_json::from_str(&stripped).unwrap();
        assert_eq!(v["url"], "http://h//v1");
        assert_eq!(v["q"], "a\"//b");
        assert_eq!(strip_json_comments("{\"a\": 1}"), "{\"a\": 1}");
    }

    /// A leading `~` is the home directory; anything else is as given.
    #[test]
    fn expand_tilde_reads_the_forms_qwen_code_reads() {
        let home = Path::new("/h");
        assert_eq!(expand_tilde("~/alt", home), PathBuf::from("/h/alt"));
        assert_eq!(expand_tilde("~", home), PathBuf::from("/h"));
        assert_eq!(expand_tilde("/abs", home), PathBuf::from("/abs"));
        assert_eq!(expand_tilde("~user/x", home), PathBuf::from("~user/x"));
    }

    #[test]
    fn pi_model_entry_declares_what_the_daemon_serves() {
        let entry = pi_model_entry("qwen3.5:0.8b", true, true, Some(32768));
        assert_eq!(entry["id"], "qwen3.5:0.8b");
        assert_eq!(entry["name"], "qwen3.5:0.8b");
        assert_eq!(entry["input"], serde_json::json!(["text", "image"]));
        assert_eq!(entry["reasoning"], true);
        assert_eq!(entry["contextWindow"], 32768);

        // A text-only model that does not think, and no window to
        // declare: pi keeps its own default rather than being told a guess.
        let plain = pi_model_entry("smol", false, false, None);
        assert_eq!(plain["input"], serde_json::json!(["text"]));
        assert_eq!(plain.get("reasoning"), None);
        assert_eq!(plain.get("contextWindow"), None);
    }

    /// The entry — pi's and omp's alike — carries the window `launch`
    /// resolved, not the raw trained context. Which file it lands in is
    /// covered by `write_omp_config_in_dir`'s own tests; this pins where
    /// the number comes from.
    #[test]
    fn pi_model_entry_declares_the_resolved_window_not_the_trained_context() {
        // A pair's larger half, as `pair_context_window` resolves it —
        // a figure no trained context could have produced.
        let pair = pair_context_window(Some(32_768), Some(200_000));
        assert_eq!(pair, Some(200_000));
        let entry = pi_model_entry("gemma4", false, false, pair);
        assert_eq!(entry["contextWindow"], 200_000);

        // And the daemon's clamp travels: a model trained past
        // DEFAULT_CTX_SIZE is served only the default, so it declares
        // only the default.
        let default = u64::from(crate::cmd::serve::DEFAULT_CTX_SIZE);
        let served = served_context_window(None, Some(1 << 20));
        assert_eq!(served, Some(default));
        let entry = pi_model_entry("big", false, false, served);
        assert_eq!(entry["contextWindow"], default);
    }
}
