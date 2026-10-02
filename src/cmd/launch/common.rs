//! Helpers shared by the per-integration launchers.

use std::path::{Path, PathBuf};

use anyhow::Context;

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
pub(super) fn forwarded_model(extra_args: &[String]) -> Option<&str> {
    let mut found = None;
    let mut args = extra_args.iter().map(String::as_str);
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
