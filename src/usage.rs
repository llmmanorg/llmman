//! The usage ledger: `llmman serve` appends a line per reply that reports
//! token usage to `usage.jsonl` beside the store; `llmman usage` reads it.
//!
//! Separate from the prompt log so `LLMMAN_NOHISTORY` doesn't also drop
//! the bill; `LLMMAN_NOUSAGE` turns this one off.
//!
//! The Messages API counts cached tokens beside `input_tokens`, the other
//! wires inside their prompt count. [`Decoder`] folds both into
//! [`Tokens`], so the ledger sums without knowing the wire.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::providers::Cost;

pub const FILE: &str = "usage.jsonl";

pub fn path() -> anyhow::Result<PathBuf> {
    crate::promptlog::beside_store(FILE)
}

pub fn enabled_from_env() -> bool {
    !crate::env_flag_set("LLMMAN_NOUSAGE")
}

/// One request's token counts; the cache counts are parts of `input`,
/// `reasoning` of `output`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tokens {
    pub input: u64,
    /// Read from a prompt cache (a provider's, or llama-server's KV cache).
    #[serde(default)]
    pub cache_read: u64,
    /// Written to a provider's prompt cache.
    #[serde(default)]
    pub cache_write: u64,
    pub output: u64,
    #[serde(default)]
    pub reasoning: u64,
}

impl Tokens {
    /// Clamps the cache counts into `input`, so an over-report never
    /// bills a token twice.
    pub fn new(input: u64, cache_read: u64, cache_write: u64, output: u64) -> Self {
        let cache_read = cache_read.min(input);
        let cache_write = cache_write.min(input - cache_read);
        Self {
            input,
            cache_read,
            cache_write,
            output,
            reasoning: 0,
        }
    }

    /// With `reasoning` of the output tokens, clamped into it.
    pub fn with_reasoning(self, reasoning: u64) -> Self {
        Self {
            reasoning: reasoning.min(self.output),
            ..self
        }
    }

    /// Prompt tokens billed at the plain input rate.
    pub fn uncached(&self) -> u64 {
        self.input
            .saturating_sub(self.cache_read)
            .saturating_sub(self.cache_write)
    }

    pub fn add(&mut self, other: &Self) {
        self.input = self.input.saturating_add(other.input);
        self.cache_read = self.cache_read.saturating_add(other.cache_read);
        self.cache_write = self.cache_write.saturating_add(other.cache_write);
        self.output = self.output.saturating_add(other.output);
        self.reasoning = self.reasoning.saturating_add(other.reasoning);
    }
}

/// The rates a request was charged, USD per million tokens, all filled
/// in so `rate` alone explains `cost`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rate {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub reasoning: f64,
}

impl Rate {
    /// The highest tier `input` is past (exclusive, as opencode reads
    /// models.dev), else the base price. A rate the tier lacks is the base
    /// one; a cache rate the catalog lacks is the input rate, a reasoning
    /// rate the output rate.
    pub fn of(cost: &Cost, input: u64) -> Self {
        let price = cost
            .tiers
            .iter()
            .rev()
            .find(|t| input > t.above)
            .map_or(cost, |t| &t.price);
        Self {
            input: price.input,
            output: price.output,
            cache_read: price.cache_read.or(cost.cache_read).unwrap_or(price.input),
            cache_write: price
                .cache_write
                .or(cost.cache_write)
                .unwrap_or(price.input),
            reasoning: price.reasoning.or(cost.reasoning).unwrap_or(price.output),
        }
    }

    /// US dollars.
    pub fn cost(&self, tokens: &Tokens) -> f64 {
        (tokens.uncached() as f64 * self.input
            + tokens.cache_read as f64 * self.cache_read
            + tokens.cache_write as f64 * self.cache_write
            + (tokens.output - tokens.reasoning) as f64 * self.output
            + tokens.reasoning as f64 * self.reasoning)
            / 1_000_000.0
    }
}

/// One line of the ledger.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    /// The prompt log's id for the request when it has one.
    pub id: String,
    /// RFC 3339, UTC, when the request arrived.
    pub time: String,
    pub route: String,
    /// What served it; for a hybrid pair, the half that answered.
    pub model: String,
    /// `null` for a local model or a peer.
    #[serde(default)]
    pub provider: Option<String>,
    /// `User-Agent`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client: Option<String>,
    pub tokens: Tokens,
    /// The price at the time; absent (not zero) where there is none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate: Option<Rate>,
    /// US dollars.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
}

impl Entry {
    pub fn price(&mut self, cost: &Cost) {
        let rate = Rate::of(cost, self.tokens.input);
        self.cost = Some(rate.cost(&self.tokens));
        self.rate = Some(rate);
    }
}

pub fn append(path: &Path, entry: &Entry) -> std::io::Result<()> {
    crate::promptlog::append_line(path, entry)
}

pub fn read(path: &Path) -> std::io::Result<Vec<Entry>> {
    crate::promptlog::read_lines(path)
}

// ---------------------------------------------------------------------------
// Reading usage off a reply
// ---------------------------------------------------------------------------

/// Which reply shape a route answers in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
    Ollama,
    OpenAi,
    Anthropic,
    Responses,
    Gemini,
}

impl Dialect {
    /// The dialect of a route template, or `None` if it reports no usage.
    pub fn of_route(route: &str) -> Option<Self> {
        match route {
            "/api/chat" | "/api/generate" | "/api/embed" => Some(Self::Ollama),
            "/v1/chat/completions" | "/v1/completions" | "/v1/embeddings" => Some(Self::OpenAi),
            "/v1/messages" => Some(Self::Anthropic),
            "/v1/responses" => Some(Self::Responses),
            "/gemini/:model/*gemini_path" => Some(Self::Gemini),
            _ => None,
        }
    }

    /// One is in every line that carries usage, so token deltas are never
    /// parsed.
    fn markers(self) -> &'static [&'static [u8]] {
        match self {
            Self::Ollama => &[b"eval_count"],
            // llama-server's `timings`, sent unasked.
            Self::OpenAi => &[b"usage", b"timings"],
            _ => &[b"usage"],
        }
    }
}

/// Longest stream line parsed (`response.completed` repeats the output).
const LINE_LIMIT: usize = 16 * 1024 * 1024;
/// Largest whole body parsed.
pub const BODY_LIMIT: usize = 64 * 1024 * 1024;
/// Longer lines can't be the usage chunk, so `Strip` doesn't hold them.
const STRIP_LINE_LIMIT: usize = 64 * 1024;

/// Counts as the dialect reported them.
#[derive(Debug, Default)]
struct Counts {
    input: Option<u64>,
    cache_read: Option<u64>,
    cache_write: Option<u64>,
    output: Option<u64>,
    reasoning: Option<u64>,
}

/// Reads token usage off a reply as it streams past, holding only the
/// current line (or a whole-document body). With `strip`, it also drops
/// the choiceless usage chunk the daemon asked for on a client's behalf.
pub struct Decoder {
    dialect: Dialect,
    /// SSE or NDJSON rather than one JSON document.
    lines: bool,
    buf: Vec<u8>,
    overflowed: bool,
    finished: bool,
    counts: Counts,
    strip: Option<Strip>,
}

impl Decoder {
    pub fn new(dialect: Dialect, content_type: Option<&str>, strip: bool) -> Self {
        let essence = content_type
            .and_then(|ct| ct.split(';').next())
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        Self {
            dialect,
            lines: essence == "text/event-stream" || essence.ends_with("ndjson"),
            buf: Vec::new(),
            overflowed: false,
            finished: false,
            counts: Counts::default(),
            // Whatever the content type says: a mislabelled stream still
            // carries the chunk, and a document has no `data:` lines.
            strip: strip.then(Strip::default),
        }
    }

    /// Reads a chunk; returns what to send on in its place.
    pub fn feed(&mut self, chunk: bytes::Bytes) -> bytes::Bytes {
        if !self.finished {
            self.inspect(&chunk);
        }
        match &mut self.strip {
            Some(strip) => strip.filter(&chunk),
            None => chunk,
        }
    }

    /// Reads the rest; returns what a strip still held. Idempotent.
    pub fn finish(&mut self) -> bytes::Bytes {
        if std::mem::replace(&mut self.finished, true) {
            return bytes::Bytes::new();
        }
        let rest = std::mem::take(&mut self.buf);
        if !self.overflowed {
            if self.lines {
                self.line(&rest);
            } else {
                self.document(&rest);
            }
        }
        match &mut self.strip {
            Some(strip) => strip.flush(),
            None => bytes::Bytes::new(),
        }
    }

    /// What the reply reported, if anything.
    pub fn tokens(&self) -> Option<Tokens> {
        let c = &self.counts;
        if c.input.is_none() && c.output.is_none() {
            return None;
        }
        let read = c.cache_read.unwrap_or(0);
        let written = c.cache_write.unwrap_or(0);
        let mut input = c.input.unwrap_or(0);
        if self.dialect == Dialect::Anthropic {
            input = input.saturating_add(read).saturating_add(written);
        }
        let tokens = Tokens::new(input, read, written, c.output.unwrap_or(0));
        Some(tokens.with_reasoning(c.reasoning.unwrap_or(0)))
    }

    fn inspect(&mut self, chunk: &[u8]) {
        if !self.lines {
            if self.overflowed {
                return;
            }
            if self.buf.len() + chunk.len() > BODY_LIMIT {
                self.overflowed = true;
                self.buf = Vec::new();
            } else {
                self.buf.extend_from_slice(chunk);
            }
            return;
        }
        let mut rest = chunk;
        while let Some(pos) = rest.iter().position(|&b| b == b'\n') {
            let (head, tail) = (&rest[..pos], &rest[pos + 1..]);
            if self.buf.is_empty() && !self.overflowed {
                self.line(head);
            } else if !self.overflowed && self.buf.len() + head.len() <= LINE_LIMIT {
                let mut line = std::mem::take(&mut self.buf);
                line.extend_from_slice(head);
                self.line(&line);
                line.clear();
                self.buf = line;
            }
            self.buf.clear();
            self.overflowed = false;
            rest = tail;
        }
        if self.overflowed {
            return;
        }
        if self.buf.len() + rest.len() > LINE_LIMIT {
            self.overflowed = true;
            self.buf.clear();
        } else {
            self.buf.extend_from_slice(rest);
        }
    }

    /// An SSE `data:` line or an NDJSON object.
    fn line(&mut self, line: &[u8]) {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        let payload = sse_data(line).unwrap_or(line);
        if !self.dialect.markers().iter().any(|m| contains(payload, m)) {
            return;
        }
        if let Ok(value) = serde_json::from_slice::<Value>(payload) {
            self.read(&value);
        }
    }

    /// One JSON document (an array for Gemini without `alt=sse`), or
    /// failing that, lines after all.
    fn document(&mut self, body: &[u8]) {
        match serde_json::from_slice::<Value>(body) {
            Ok(Value::Array(items)) => items.iter().for_each(|item| self.read(item)),
            Ok(value) => self.read(&value),
            Err(_) => body.split(|&b| b == b'\n').for_each(|line| self.line(line)),
        }
    }

    fn read(&mut self, v: &Value) {
        let n = |v: &Value, ptr: &str| v.pointer(ptr).and_then(Value::as_u64);
        let c = &mut self.counts;
        match self.dialect {
            Dialect::OpenAi => {
                if let Some(u) = v.get("usage").filter(|u| n(u, "/prompt_tokens").is_some()) {
                    *c = Counts {
                        input: n(u, "/prompt_tokens"),
                        // `prompt_cache_hit_tokens` is DeepSeek's.
                        cache_read: n(u, "/prompt_tokens_details/cached_tokens")
                            .or_else(|| n(u, "/prompt_cache_hit_tokens")),
                        // OpenRouter's, and llmman's from an Anthropic provider.
                        cache_write: n(u, "/prompt_tokens_details/cache_write_tokens"),
                        output: Some(n(u, "/completion_tokens").unwrap_or(0)),
                        reasoning: n(u, "/completion_tokens_details/reasoning_tokens"),
                    };
                } else if let (Some(t), None) = (v.get("timings"), c.input) {
                    // llama-server's, sent unasked; a later `usage` wins.
                    let cached = n(t, "/cache_n").unwrap_or(0);
                    if let (Some(prompt), Some(output)) = (n(t, "/prompt_n"), n(t, "/predicted_n"))
                    {
                        *c = Counts {
                            input: Some(prompt.saturating_add(cached)),
                            cache_read: Some(cached),
                            output: Some(output),
                            ..Counts::default()
                        };
                    }
                }
            }
            Dialect::Responses => {
                for u in [v.get("usage"), v.pointer("/response/usage")]
                    .into_iter()
                    .flatten()
                {
                    if let Some(input) = n(u, "/input_tokens") {
                        *c = Counts {
                            input: Some(input),
                            cache_read: n(u, "/input_tokens_details/cached_tokens"),
                            cache_write: n(u, "/input_tokens_details/cache_write_tokens"),
                            output: Some(n(u, "/output_tokens").unwrap_or(0)),
                            reasoning: n(u, "/output_tokens_details/reasoning_tokens"),
                        };
                    }
                }
            }
            Dialect::Anthropic => {
                // Spread over `message_start` and `message_delta`, either
                // of which may carry any field, so each is kept apart.
                for u in [v.get("usage"), v.pointer("/message/usage")]
                    .into_iter()
                    .flatten()
                {
                    let set = |field: &mut Option<u64>, key: &str| {
                        if let Some(value) = u.get(key).and_then(Value::as_u64) {
                            *field = Some(value);
                        }
                    };
                    set(&mut c.input, "input_tokens");
                    set(&mut c.cache_read, "cache_read_input_tokens");
                    set(&mut c.cache_write, "cache_creation_input_tokens");
                    set(&mut c.output, "output_tokens");
                }
            }
            Dialect::Ollama => {
                let prompt = n(v, "/prompt_eval_count");
                let output = n(v, "/eval_count");
                if prompt.is_none() && output.is_none() {
                    return;
                }
                // `prompt_eval_count` is what was evaluated (cache writes
                // included); llmman reports the cached rest beside it.
                let cached = n(v, "/prompt_eval_cached_count").unwrap_or(0);
                *c = Counts {
                    input: Some(prompt.unwrap_or(0).saturating_add(cached)),
                    cache_read: Some(cached),
                    cache_write: n(v, "/prompt_eval_cache_write_count"),
                    output: Some(output.unwrap_or(0)),
                    reasoning: None,
                };
            }
            Dialect::Gemini => {
                let Some(m) = v.get("usageMetadata") else {
                    return;
                };
                let Some(prompt) = n(m, "/promptTokenCount") else {
                    return;
                };
                // The prompt count includes cached content; candidates
                // exclude thoughts.
                let tool_prompt = n(m, "/toolUsePromptTokenCount").unwrap_or(0);
                let candidates = n(m, "/candidatesTokenCount").unwrap_or(0);
                let thoughts = n(m, "/thoughtsTokenCount").unwrap_or(0);
                *c = Counts {
                    input: Some(prompt.saturating_add(tool_prompt)),
                    cache_read: n(m, "/cachedContentTokenCount"),
                    cache_write: None,
                    output: Some(candidates.saturating_add(thoughts)),
                    reasoning: Some(thoughts),
                };
            }
        }
    }
}

/// Holds each line until whole and drops the choiceless usage chunk with
/// the blank line ending its event; everything else passes byte for byte.
#[derive(Debug, Default)]
struct Strip {
    held: Vec<u8>,
    /// The current line is too long to be the chunk: relayed as it comes.
    passing: bool,
    drop_blank: bool,
}

impl Strip {
    fn filter(&mut self, chunk: &[u8]) -> bytes::Bytes {
        let mut out = Vec::with_capacity(chunk.len() + self.held.len());
        let mut rest = chunk;
        while let Some(pos) = rest.iter().position(|&b| b == b'\n') {
            let (line, tail) = rest.split_at(pos + 1);
            if self.passing {
                out.extend_from_slice(line);
                self.passing = false;
            } else {
                self.held.extend_from_slice(line);
                let line = std::mem::take(&mut self.held);
                if !self.drops(&line) {
                    out.extend_from_slice(&line);
                }
            }
            rest = tail;
        }
        if self.passing {
            out.extend_from_slice(rest);
        } else if self.held.len() + rest.len() > STRIP_LINE_LIMIT {
            out.append(&mut self.held);
            out.extend_from_slice(rest);
            self.passing = true;
        } else {
            self.held.extend_from_slice(rest);
        }
        bytes::Bytes::from(out)
    }

    fn flush(&mut self) -> bytes::Bytes {
        let held = std::mem::take(&mut self.held);
        if held.is_empty() || self.drops(&held) {
            return bytes::Bytes::new();
        }
        bytes::Bytes::from(held)
    }

    fn drops(&mut self, line: &[u8]) -> bool {
        let content = line.strip_suffix(b"\n").unwrap_or(line);
        let content = content.strip_suffix(b"\r").unwrap_or(content);
        if content.is_empty() {
            return std::mem::take(&mut self.drop_blank);
        }
        self.drop_blank = is_usage_chunk(content);
        self.drop_blank
    }
}

/// `data: {"choices": [], "usage": {...}}`, what `include_usage` adds.
fn is_usage_chunk(line: &[u8]) -> bool {
    let Some(payload) = sse_data(line).filter(|p| contains(p, b"usage")) else {
        return false;
    };
    serde_json::from_slice::<Value>(payload)
        .is_ok_and(|v| v["choices"].as_array().is_some_and(Vec::is_empty) && v["usage"].is_object())
}

fn sse_data(line: &[u8]) -> Option<&[u8]> {
    let payload = line.strip_prefix(b"data:")?;
    Some(payload.strip_prefix(b" ").unwrap_or(payload))
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::Tier;
    use bytes::Bytes;

    /// Feeds `body` in `size`-byte chunks; returns tokens and output.
    fn decode_in(
        dialect: Dialect,
        content_type: &str,
        strip: bool,
        body: &str,
        size: usize,
    ) -> (Option<Tokens>, String) {
        let mut decoder = Decoder::new(dialect, Some(content_type), strip);
        let mut out = Vec::new();
        for chunk in body.as_bytes().chunks(size) {
            out.extend_from_slice(&decoder.feed(Bytes::copy_from_slice(chunk)));
        }
        out.extend_from_slice(&decoder.finish());
        (decoder.tokens(), String::from_utf8(out).unwrap())
    }

    fn decode(dialect: Dialect, content_type: &str, body: &str) -> Option<Tokens> {
        let whole = decode_in(dialect, content_type, false, body, body.len().max(1));
        for size in [1, 3, 7] {
            assert_eq!(
                decode_in(dialect, content_type, false, body, size),
                whole,
                "chunks of {size}"
            );
        }
        assert_eq!(whole.1, body, "relayed unchanged");
        whole.0
    }

    #[test]
    fn a_chat_completion_counts_its_cached_prefix_inside_the_prompt() {
        let json = r#"{"id":"x","choices":[{"message":{"content":"hi"}}],
            "usage":{"prompt_tokens":100,"completion_tokens":7,"prompt_tokens_details":{"cached_tokens":80},
            "completion_tokens_details":{"reasoning_tokens":5}}}"#;
        assert_eq!(
            decode(Dialect::OpenAi, "application/json", json),
            Some(Tokens::new(100, 80, 0, 7).with_reasoning(5))
        );
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"usage\"}}]}\n\n\
                   data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                   data: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3,\
                   \"prompt_tokens_details\":{\"cached_tokens\":4,\"cache_write_tokens\":6}}}\n\n\
                   data: [DONE]\n\n";
        assert_eq!(
            decode(Dialect::OpenAi, "text/event-stream", sse),
            Some(Tokens::new(12, 4, 6, 3))
        );
        // DeepSeek's spelling; an embeddings reply has no output.
        assert_eq!(
            decode(
                Dialect::OpenAi,
                "application/json",
                r#"{"usage":{"prompt_tokens":9,"prompt_cache_hit_tokens":5}}"#
            ),
            Some(Tokens::new(9, 5, 0, 0))
        );
    }

    /// llama-server's `timings` stand in for an unasked `usage`, and give
    /// way to one.
    #[test]
    fn llama_server_timings_count_when_there_is_no_usage() {
        let finish = "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\
                      \"timings\":{\"prompt_n\":5,\"cache_n\":95,\"predicted_n\":7}}\n\n";
        assert_eq!(
            decode(Dialect::OpenAi, "text/event-stream", finish),
            Some(Tokens::new(100, 95, 0, 7))
        );
        let both = format!(
            "{finish}data: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":9,\"completion_tokens\":1}}}}\n\n"
        );
        assert_eq!(
            decode(Dialect::OpenAi, "text/event-stream", &both),
            Some(Tokens::new(9, 0, 0, 1))
        );
    }

    #[test]
    fn a_reply_without_usage_records_nothing() {
        for (dialect, body) in [
            (Dialect::OpenAi, r#"{"choices":[]}"#),
            (Dialect::OpenAi, r#"{"usage":null}"#),
            (Dialect::Anthropic, r#"{"type":"error","error":{}}"#),
            (
                Dialect::Ollama,
                r#"{"model":"m","done":true,"done_reason":"load"}"#,
            ),
            (Dialect::Gemini, r#"{"totalTokens":31}"#),
            (Dialect::Responses, "not json"),
        ] {
            assert_eq!(decode(dialect, "application/json", body), None, "{body}");
        }
    }

    #[test]
    fn a_messages_stream_adds_the_cache_counts_to_the_prompt() {
        let sse = "event: message_start\n\
            data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":25,\
            \"cache_read_input_tokens\":1000,\"cache_creation_input_tokens\":200,\"output_tokens\":1}}}\n\n\
            event: content_block_delta\n\
            data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"hi\"}}\n\n\
            event: message_delta\n\
            data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":42}}\n\n";
        assert_eq!(
            decode(Dialect::Anthropic, "text/event-stream", sse),
            Some(Tokens::new(1225, 1000, 200, 42))
        );
        let json = r#"{"type":"message","usage":{"input_tokens":5,"output_tokens":7,"cache_read_input_tokens":3}}"#;
        assert_eq!(
            decode(Dialect::Anthropic, "application/json", json),
            Some(Tokens::new(8, 3, 0, 7))
        );
    }

    #[test]
    fn a_responses_stream_reads_the_completed_event() {
        let sse = "event: response.created\n\
            data: {\"type\":\"response.created\",\"response\":{\"usage\":null}}\n\n\
            event: response.completed\n\
            data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":50,\
            \"input_tokens_details\":{\"cached_tokens\":40,\"cache_write_tokens\":5},\"output_tokens\":9,\
            \"output_tokens_details\":{\"reasoning_tokens\":4}}}}\n\n";
        assert_eq!(
            decode(Dialect::Responses, "text/event-stream; charset=utf-8", sse),
            Some(Tokens::new(50, 40, 5, 9).with_reasoning(4))
        );
        let json = r#"{"object":"response","usage":{"input_tokens":3,"output_tokens":2}}"#;
        assert_eq!(
            decode(Dialect::Responses, "application/json", json),
            Some(Tokens::new(3, 0, 0, 2))
        );
    }

    #[test]
    fn ollama_counts_come_off_the_done_object() {
        let ndjson = "{\"message\":{\"content\":\"a\"},\"done\":false}\n\
            {\"message\":{\"content\":\"\"},\"done\":true,\"prompt_eval_count\":10,\
            \"prompt_eval_cached_count\":90,\"prompt_eval_cache_write_count\":6,\"eval_count\":4}\n";
        assert_eq!(
            decode(Dialect::Ollama, "application/x-ndjson", ndjson),
            Some(Tokens::new(100, 90, 6, 4))
        );
        assert_eq!(
            decode(
                Dialect::Ollama,
                "application/json",
                r#"{"embeddings":[[0.1]],"prompt_eval_count":6}"#
            ),
            Some(Tokens::new(6, 0, 0, 0))
        );
    }

    #[test]
    fn gemini_takes_the_last_metadata_and_bills_thinking_as_output() {
        let sse = "data: {\"candidates\":[],\"usageMetadata\":{\"promptTokenCount\":20,\"candidatesTokenCount\":1}}\n\n\
            data: {\"candidates\":[],\"usageMetadata\":{\"promptTokenCount\":20,\
            \"cachedContentTokenCount\":15,\"candidatesTokenCount\":8,\"thoughtsTokenCount\":30}}\n\n";
        assert_eq!(
            decode(Dialect::Gemini, "text/event-stream", sse),
            Some(Tokens::new(20, 15, 0, 38).with_reasoning(30))
        );
        // `streamGenerateContent` without `alt=sse` is one JSON array.
        let array = r#"[{"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":1}},
            {"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":5}}]"#;
        assert_eq!(
            decode(Dialect::Gemini, "application/json", array),
            Some(Tokens::new(2, 0, 0, 5))
        );
    }

    #[test]
    fn a_stream_without_its_content_type_is_still_read() {
        let sse = "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n";
        assert_eq!(
            decode(Dialect::OpenAi, "application/octet-stream", sse),
            Some(Tokens::new(2, 0, 0, 1))
        );
    }

    #[test]
    fn strip_removes_only_the_usage_event() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\r\n\r\n\
                   data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":null}\r\n\r\n\
                   data: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3}}\r\n\r\n\
                   data: [DONE]\r\n\r\n";
        let want = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\r\n\r\n\
                    data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":null}\r\n\r\n\
                    data: [DONE]\r\n\r\n";
        for size in [1, 2, 5, 64, sse.len()] {
            let (tokens, out) = decode_in(Dialect::OpenAi, "text/event-stream", true, sse, size);
            assert_eq!(out, want, "chunks of {size}");
            assert_eq!(tokens, Some(Tokens::new(12, 0, 0, 3)));
        }
        // A trailing usage chunk with no newline goes too.
        let tail = "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}";
        let (tokens, out) = decode_in(Dialect::OpenAi, "text/event-stream", true, tail, 4);
        assert_eq!(out, "");
        assert_eq!(tokens, Some(Tokens::new(1, 0, 0, 1)));
    }

    /// Usage on a chunk with a choice stays; a long line is not held.
    #[test]
    fn strip_keeps_everything_else_byte_for_byte() {
        let long = format!(
            "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{}\"}}}}]}}\n\n",
            "x".repeat(STRIP_LINE_LIMIT * 2)
        );
        let sse = format!(
            "{long}data: {{\"choices\":[{{\"delta\":{{}}}}],\"usage\":{{\"prompt_tokens\":4,\"completion_tokens\":2}}}}\n\nid: 7\n\n"
        );
        let (tokens, out) = decode_in(Dialect::OpenAi, "text/event-stream", true, &sse, 4096);
        assert_eq!(out, sse);
        assert_eq!(tokens, Some(Tokens::new(4, 0, 0, 2)));
    }

    #[test]
    fn strip_works_on_a_mislabelled_stream() {
        let sse = "data: {\"choices\":[{\"delta\":{}}]}\n\n\
                   data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1}}\n\n\
                   data: [DONE]\n\n";
        let (tokens, out) = decode_in(Dialect::OpenAi, "application/octet-stream", true, sse, 9);
        assert_eq!(
            out,
            "data: {\"choices\":[{\"delta\":{}}]}\n\ndata: [DONE]\n\n"
        );
        assert_eq!(tokens, Some(Tokens::new(2, 0, 0, 1)));
    }

    #[test]
    fn strip_does_nothing_to_a_whole_document() {
        let json = r#"{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#;
        let (_, out) = decode_in(Dialect::OpenAi, "application/json", true, json, 5);
        assert_eq!(out, json);
    }

    #[test]
    fn tokens_are_clamped_into_the_prompt() {
        assert_eq!(Tokens::new(10, 50, 5, 1), Tokens::new(10, 10, 0, 1));
        assert_eq!(Tokens::new(10, 4, 50, 1), Tokens::new(10, 4, 6, 1));
        assert_eq!(Tokens::new(10, 4, 6, 1).uncached(), 0);
    }

    fn sonnet() -> Cost {
        Cost {
            cache_read: Some(0.3),
            cache_write: Some(3.75),
            tiers: vec![Tier {
                above: 200_000,
                price: Cost {
                    cache_read: Some(0.6),
                    ..Cost::flat(6.0, 22.5)
                },
            }],
            ..Cost::flat(3.0, 15.0)
        }
    }

    #[test]
    fn cost_prices_each_class_of_token_at_its_rate() {
        let tokens = Tokens::new(18_240, 17_010, 1_000, 412);
        let rate = Rate::of(&sonnet(), tokens.input);
        assert_eq!(
            rate,
            Rate {
                input: 3.0,
                output: 15.0,
                cache_read: 0.3,
                cache_write: 3.75,
                reasoning: 15.0,
            }
        );
        let want = (230.0 * 3.0 + 17_010.0 * 0.3 + 1_000.0 * 3.75 + 412.0 * 15.0) / 1e6;
        assert!((rate.cost(&tokens) - want).abs() < 1e-12);
    }

    /// Reasoning tokens are billed at a published reasoning rate.
    #[test]
    fn reasoning_is_priced_at_its_own_rate() {
        let cost = Cost {
            reasoning: Some(2.1),
            ..Cost::flat(0.2, 0.7)
        };
        let tokens = Tokens::new(0, 0, 0, 1_000).with_reasoning(600);
        let want = (400.0 * 0.7 + 600.0 * 2.1) / 1e6;
        assert!((Rate::of(&cost, 0).cost(&tokens) - want).abs() < 1e-12);
        assert_eq!(Tokens::new(0, 0, 0, 5).with_reasoning(9).reasoning, 5);
    }

    /// A cache rate the tier lacks is the base one, not the input rate.
    #[test]
    fn a_long_prompt_is_priced_at_its_tier() {
        assert_eq!(Rate::of(&sonnet(), 200_000).input, 3.0);
        let rate = Rate::of(&sonnet(), 200_001);
        assert_eq!(
            rate,
            Rate {
                input: 6.0,
                output: 22.5,
                cache_read: 0.6,
                cache_write: 3.75,
                reasoning: 22.5,
            }
        );
        let tiered = Cost {
            tiers: vec![
                Tier {
                    above: 32_000,
                    price: Cost::flat(2.0, 2.0),
                },
                Tier {
                    above: 128_000,
                    price: Cost::flat(4.0, 4.0),
                },
            ],
            ..Cost::flat(1.0, 1.0)
        };
        assert_eq!(Rate::of(&tiered, 64_000).input, 2.0);
        assert_eq!(Rate::of(&tiered, 500_000).input, 4.0);
    }

    #[test]
    fn a_missing_cache_rate_is_the_input_rate() {
        let rate = Rate::of(&Cost::flat(2.0, 8.0), 10);
        assert_eq!((rate.cache_read, rate.cache_write), (2.0, 2.0));
        let free = Rate::of(&Cost::flat(0.0, 0.0), 10);
        assert_eq!(free.cost(&Tokens::new(10, 5, 0, 10)), 0.0);
    }

    #[test]
    fn append_then_read_round_trips() {
        let dir = std::env::temp_dir().join(format!("llmman-usage-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join(FILE);
        let mut priced = Entry {
            id: "a".repeat(40),
            time: "2026-09-10T14:02:11Z".into(),
            route: "/v1/messages".into(),
            model: "llmman.provider/openrouter/anthropic/claude-sonnet-4".into(),
            provider: Some("openrouter".into()),
            client: Some("claude-cli/1.0.83".into()),
            tokens: Tokens::new(18_240, 17_010, 0, 412),
            rate: None,
            cost: None,
        };
        priced.price(&sonnet());
        let local = Entry {
            model: "qwen3:8b".into(),
            provider: None,
            client: None,
            rate: None,
            cost: None,
            ..priced.clone()
        };
        append(&path, &priced).unwrap();
        append(&path, &local).unwrap();
        assert_eq!(read(&path).unwrap(), vec![priced, local]);
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert!(lines[1]["provider"].is_null());
        assert!(lines[1].get("rate").is_none() && lines[1].get("cost").is_none());
        assert_eq!(lines[0]["rate"]["cache_read"], 0.3);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
