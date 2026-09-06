//! The prompt log: `llmman serve` appends one JSON line per generation
//! request to `prompts.jsonl` beside the store; `llmman log` reads it.
//!
//! An entry is the prompt — the last user message's text — not the whole
//! transcript an agent re-sends every turn, so the file grows with what
//! was typed.

use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const FILE: &str = "prompts.jsonl";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    /// 40 hex chars, git-commit width.
    pub id: String,
    /// RFC 3339, UTC.
    pub time: String,
    pub route: String,
    pub model: String,
    /// The request's `User-Agent`, when it sent one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client: Option<String>,
    pub prompt: String,
}

/// `<store>/../prompts.jsonl`, honoring `LLMMAN_MODELS` like `serve.log`.
pub fn path() -> anyhow::Result<PathBuf> {
    let store = crate::default_store()?;
    Ok(store.parent().unwrap_or(&store).join(FILE))
}

/// Off under `LLMMAN_NOHISTORY` (as ollama's `OLLAMA_NOHISTORY`).
pub fn enabled_from_env() -> bool {
    !crate::env_flag_set("LLMMAN_NOHISTORY")
}

pub fn is_generation_route(route: &str) -> bool {
    matches!(
        route,
        "/api/chat"
            | "/api/generate"
            | "/v1/chat/completions"
            | "/v1/completions"
            | "/v1/responses"
            | "/v1/messages"
    )
}

/// `None` for a body with no prompt text: ollama's empty-`messages`
/// load/unload requests, or non-JSON the handler will reject anyway.
pub fn entry(route: &str, body: &[u8], client: Option<&str>, time: String) -> Option<Entry> {
    let req: serde_json::Value = serde_json::from_slice(body).ok()?;
    let prompt = prompt_of(&req);
    if prompt.is_empty() {
        return None;
    }
    // Nanoseconds too, so identical requests within the same displayed
    // second still get distinct ids.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(time.as_bytes());
    hasher.update(nanos.to_le_bytes());
    hasher.update(route.as_bytes());
    hasher.update(body);
    let id = hex::encode(hasher.finalize())[..40].to_string();
    Some(Entry {
        id,
        time,
        route: route.to_string(),
        model: req["model"].as_str().unwrap_or("").to_string(),
        client: client.map(str::to_string),
        prompt,
    })
}

/// The last user turn's text from `messages` (Ollama, OpenAI, Anthropic)
/// or the Responses API's `input` list — never another role's, nor an
/// older turn's when the last has no text; otherwise the bare
/// `prompt`/`input`.
fn prompt_of(req: &serde_json::Value) -> String {
    if let Some(messages) = req["messages"]
        .as_array()
        .or_else(|| req["input"].as_array())
    {
        return messages
            .iter()
            .rev()
            .find(|m| m["role"] == "user")
            .map(|m| text_of(&m["content"]))
            .unwrap_or_default();
    }
    let prompt = if req["prompt"].is_null() {
        &req["input"]
    } else {
        &req["prompt"]
    };
    text_of(prompt)
}

/// A string as is; a content-part list as its parts' `text` (OpenAI,
/// Anthropic, Responses all use that key) joined by newlines.
fn text_of(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(s) => s.trim().to_string(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.as_str().or_else(|| p["text"].as_str()))
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Appends one entry as one line, to a file only its owner can read. A
/// tail left unterminated by a crash mid-write gets its newline first,
/// so it costs only itself.
pub fn append(path: &Path, entry: &Entry) -> std::io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut options = std::fs::OpenOptions::new();
    options.create(true).append(true).read(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let mut file = options.open(path)?;
    let mut line = Vec::new();
    if file.seek(SeekFrom::End(-1)).is_ok() {
        let mut last = [0u8];
        file.read_exact(&mut last)?;
        if last != *b"\n" {
            line.push(b'\n');
        }
    }
    serde_json::to_writer(&mut line, entry)?;
    line.push(b'\n');
    file.write_all(&line)
}

/// Every entry, oldest first. A missing file is an empty log; a line
/// that doesn't parse (a torn write, a hand edit) is skipped.
pub fn read(path: &Path) -> std::io::Result<Vec<Entry>> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    Ok(bytes
        .split(|b| *b == b'\n')
        .filter_map(|line| serde_json::from_slice(line).ok())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prompt(body: &str) -> String {
        prompt_of(&serde_json::from_str(body).unwrap())
    }

    #[test]
    fn the_last_user_message_is_the_prompt() {
        let body = r#"{"messages":[
            {"role":"system","content":"be terse"},
            {"role":"user","content":"first"},
            {"role":"assistant","content":"ok"},
            {"role":"user","content":"  second  "}]}"#;
        assert_eq!(prompt(body), "second");
    }

    #[test]
    fn a_trailing_tool_turn_does_not_hide_the_user_prompt() {
        let body = r#"{"messages":[
            {"role":"user","content":"list files"},
            {"role":"assistant","content":"","tool_calls":[{}]},
            {"role":"tool","content":"a.txt"}]}"#;
        assert_eq!(prompt(body), "list files");
    }

    #[test]
    fn without_text_in_the_last_user_turn_nothing_is_recorded() {
        let body = r#"{"messages":[{"role":"system","content":"sys"},{"role":"assistant","content":"hi"}]}"#;
        assert_eq!(prompt(body), "");
        // An image-only turn must not re-record the previous prompt.
        let body = r#"{"messages":[
            {"role":"user","content":"older"},
            {"role":"assistant","content":"hi"},
            {"role":"user","content":[{"type":"image_url","image_url":{"url":"data:..."}}]}]}"#;
        assert_eq!(prompt(body), "");
    }

    #[test]
    fn content_parts_are_joined_and_images_skipped() {
        let body = r#"{"messages":[{"role":"user","content":[
            {"type":"text","text":"what is this"},
            {"type":"image_url","image_url":{"url":"data:..."}},
            {"type":"input_text","text":"in detail"}]}]}"#;
        assert_eq!(prompt(body), "what is this\nin detail");
    }

    #[test]
    fn generate_completions_and_responses_shapes_are_read() {
        assert_eq!(prompt(r#"{"prompt":"why"}"#), "why");
        assert_eq!(prompt(r#"{"prompt":["a","b"]}"#), "a\nb");
        assert_eq!(prompt(r#"{"input":"resp"}"#), "resp");
        assert_eq!(
            prompt(
                r#"{"input":[{"role":"user","content":[{"type":"input_text","text":"codex"}]}]}"#
            ),
            "codex"
        );
    }

    #[test]
    fn empty_and_non_json_bodies_make_no_entry() {
        let t = || "2026-01-01T00:00:00Z".to_string();
        assert!(entry("/api/chat", br#"{"model":"m","messages":[]}"#, None, t()).is_none());
        assert!(entry("/api/generate", br#"{"model":"m","prompt":""}"#, None, t()).is_none());
        assert!(entry("/api/chat", b"not json", None, t()).is_none());
    }

    #[test]
    fn an_entry_carries_model_client_and_a_git_width_id() {
        let body = br#"{"model":"qwen3:8b","messages":[{"role":"user","content":"hi"}]}"#;
        let make = || {
            entry(
                "/v1/chat/completions",
                body,
                Some("claude-cli/1.0"),
                "2026-01-01T00:00:00Z".to_string(),
            )
            .unwrap()
        };
        let e = make();
        assert_eq!(e.model, "qwen3:8b");
        assert_eq!(e.client.as_deref(), Some("claude-cli/1.0"));
        assert_eq!(e.prompt, "hi");
        assert_eq!(e.id.len(), 40);
        assert!(e.id.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(e.id, make().id, "the same request twice is two entries");
    }

    #[test]
    fn append_then_read_round_trips_and_skips_a_bad_line() {
        let dir = std::env::temp_dir().join(format!("llmman-promptlog-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("sub").join(FILE);
        assert_eq!(read(&path).unwrap(), Vec::<Entry>::new());

        let e = |i: u8| Entry {
            id: format!("{i:040}"),
            time: "2026-01-01T00:00:00Z".into(),
            route: "/api/chat".into(),
            model: "m".into(),
            client: None,
            prompt: format!("p{i}"),
        };
        append(&path, &e(1)).unwrap();
        append(&path, &e(2)).unwrap();
        // A write torn inside a multibyte character, with no newline.
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"{\"prompt\":\"\xC3")
            .unwrap();
        append(&path, &e(3)).unwrap();

        assert_eq!(read(&path).unwrap(), vec![e(1), e(2), e(3)]);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "prompts are private to their owner");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn only_the_generation_routes_are_recorded() {
        for r in [
            "/api/chat",
            "/api/generate",
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/responses",
            "/v1/messages",
        ] {
            assert!(is_generation_route(r), "{r}");
        }
        for r in [
            "/api/embed",
            "/v1/embeddings",
            "/api/pull",
            "/v1/responses/input_tokens",
        ] {
            assert!(!is_generation_route(r), "{r}");
        }
    }
}
