//! The Responses API for providers that don't speak it.
//!
//! Codex speaks only `/v1/responses` (`wire_api = "responses"` is the only
//! value it accepts). llama-server, `openai`, `groq` and `openrouter`
//! implement it; most OpenAI-compatible providers stop at
//! `/v1/chat/completions` — `anthropic` 404s the route, `opencode` 500s it
//! for every non-OpenAI model — so the daemon bridges the two.
//!
//! A port of Ollama's `openai/responses.go` (`FromResponsesRequest`,
//! `ResponsesStreamConverter`), speaking chat completions to a provider
//! instead of `/api/chat` to a local runner. Differences forced by that:
//!
//! - Tool calls arrive as argument fragments, so they are assembled and
//!   emitted at completion rather than as each arrives.
//! - Reasoning input items are dropped: an OpenAI assistant message has
//!   no `thinking` field to carry them back in.
//! - Ollama's server-side `web_search`/`tool_search` tools are dropped.
//! - `tool_choice` and `text.format` are forwarded (`/api/chat` has no
//!   `tool_choice`).
//! - `namespace` members flatten to `namespace_member`, not
//!   `namespace.member`: providers enforce `^[a-zA-Z0-9_-]{1,128}$`.
//!
//! Everything here is pure — JSON in, SSE text out — so it is testable
//! without a network. `remote_responses` in the parent decides when to
//! use it: only after the provider answered the native route with a
//! [`falls_back`] status, so a provider that has the API keeps it.

use std::collections::BTreeMap;

use axum::http::StatusCode;
use serde_json::{json, Value};

/// Whether a provider's answer on `/v1/responses` means "retry as a chat
/// completion": 404/405/501 (no such route) or any 5xx (`opencode` 500s
/// the route for every non-OpenAI model). Any other 4xx is the provider's
/// answer about the caller's key or request, and a retry would bury it.
pub(super) fn falls_back(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::NOT_FOUND | StatusCode::METHOD_NOT_ALLOWED | StatusCode::NOT_IMPLEMENTED
    ) || status.is_server_error()
}

// ---------------------------------------------------------------------------
// Request: Responses -> chat completions (Ollama's FromResponsesRequest)
// ---------------------------------------------------------------------------

/// Ollama's `FromResponsesRequest`. `developer` becomes `system` (Mistral
/// rejects `developer`; Anthropic folds both into one prompt anyway).
/// Always streams with `usage` in the final chunk so [`StreamConverter`]
/// can report token counts; a `stream: false` caller folds the stream.
/// Errors on `input_file`, as Ollama does.
pub(super) fn from_responses_request(req: &Value) -> anyhow::Result<Value> {
    let available_tools = req
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut messages: Vec<Value> = Vec::new();

    if let Some(instructions) = req.get("instructions").and_then(Value::as_str) {
        if !instructions.is_empty() {
            messages.push(json!({ "role": "system", "content": instructions }));
        }
    }

    match req.get("input") {
        Some(Value::String(prompt)) => {
            if !prompt.is_empty() {
                messages.push(json!({ "role": "user", "content": prompt }));
            }
        }
        Some(Value::Array(items)) => {
            for (i, item) in items.iter().enumerate() {
                let next = items.get(i + 1);
                push_input_item(&mut messages, item, next, &available_tools)?;
            }
        }
        _ => {}
    }

    let mut chat = json!({
        "model": req.get("model").cloned().unwrap_or(Value::Null),
        "messages": messages,
        "stream": true,
        "stream_options": { "include_usage": true },
    });

    let mut tools: Vec<Value> = Vec::new();
    for tool in &available_tools {
        convert_tools(tool, &mut tools);
    }
    if !tools.is_empty() {
        chat["tools"] = Value::Array(tools);
        if let Some(choice) = req.get("tool_choice") {
            chat["tool_choice"] = tool_choice(choice, &available_tools);
        }
        if let Some(parallel) = req.get("parallel_tool_calls").filter(|v| v.is_boolean()) {
            chat["parallel_tool_calls"] = parallel.clone();
        }
    }

    for (from, to) in [
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("max_output_tokens", "max_tokens"),
    ] {
        if let Some(v) = req.get(from).filter(|v| !v.is_null()) {
            chat[to] = v.clone();
        }
    }

    // Ollama's `think` level; chat completions calls it `reasoning_effort`.
    if let Some(effort) = req.pointer("/reasoning/effort").and_then(Value::as_str) {
        if !effort.is_empty() {
            chat["reasoning_effort"] = Value::String(effort.to_string());
        }
    }

    // Ollama's `format`; chat completions calls it `response_format`.
    if let Some(format) = req.pointer("/text/format").filter(|f| f.is_object()) {
        if format.get("type").and_then(Value::as_str) == Some("json_object") {
            chat["response_format"] = json!({ "type": "json_object" });
        } else if format.get("type").and_then(Value::as_str) == Some("json_schema") {
            if let Some(schema) = format.get("schema") {
                let mut json_schema = json!({ "schema": schema });
                for key in ["name", "strict"] {
                    if let Some(v) = format.get(key).filter(|v| !v.is_null()) {
                        json_schema[key] = v.clone();
                    }
                }
                chat["response_format"] =
                    json!({ "type": "json_schema", "json_schema": json_schema });
            }
        }
    }
    Ok(chat)
}

/// `tool_choice` differs only in its object form: the Responses API
/// names the function at the top level, chat completions nests it, under
/// the name [`convert_tools`] gave it. A choice of a hosted tool
/// [`convert_tools`] dropped becomes `auto`.
fn tool_choice(choice: &Value, tools: &[Value]) -> Value {
    let Value::Object(o) = choice else {
        return choice.clone();
    };
    if o.get("type").and_then(Value::as_str) != Some("function") {
        return json!("auto");
    }
    let name = o.get("name").and_then(Value::as_str).unwrap_or("");
    let (namespace, name) = match o.get("namespace").and_then(Value::as_str) {
        Some(ns) if !ns.is_empty() => (ns.to_string(), name.to_string()),
        _ => responses_tool_call_name(tools, name),
    };
    json!({ "type": "function", "function": { "name": qualify_namespace_tool_name(&namespace, &name) } })
}

/// Appends the chat message(s) one `input` item stands for; `next` is the
/// following item, for the assistant-message merge.
fn push_input_item(
    messages: &mut Vec<Value>,
    item: &Value,
    next: Option<&Value>,
    tools: &[Value],
) -> anyhow::Result<()> {
    let role = item.get("role").and_then(Value::as_str);
    // The `{"role", "content"}` shorthand has no `type`.
    let kind = item
        .get("type")
        .and_then(Value::as_str)
        .filter(|t| !t.is_empty())
        .unwrap_or(if role.is_some() { "message" } else { "" });
    match kind {
        "message" => {
            let role = match role.unwrap_or("user") {
                "developer" => "system",
                role => role,
            };
            let content = convert_responses_content(item.get("content").unwrap_or(&Value::Null))?;
            if role == "assistant" && merge_replayed_assistant(messages, &content, next) {
                return Ok(());
            }
            messages.push(json!({ "role": role, "content": content }));
        }
        "function_call" => {
            let arguments = match item.get("arguments") {
                Some(Value::String(s)) => s.clone(),
                Some(other) if !other.is_null() => other.to_string(),
                _ => String::new(),
            };
            let name = item.get("name").and_then(Value::as_str).unwrap_or("");
            let namespace = item.get("namespace").and_then(Value::as_str).unwrap_or("");
            let (namespace, name) = if namespace.is_empty() {
                responses_tool_call_name(tools, name)
            } else {
                (namespace.to_string(), name.to_string())
            };
            let call = json!({
                "id": item.get("call_id").cloned().unwrap_or(Value::Null),
                "type": "function",
                "function": {
                    "name": qualify_namespace_tool_name(&namespace, &name),
                    "arguments": arguments,
                },
            });
            append_response_tool_call(messages, call);
        }
        "function_call_output" => {
            let output = match item.get("output") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Array(_)) => content_text(&convert_responses_content(&item["output"])?),
                Some(other) if !other.is_null() => other.to_string(),
                _ => String::new(),
            };
            messages.push(json!({
                "role": "tool",
                "tool_call_id": item.get("call_id").cloned().unwrap_or(Value::Null),
                "content": output,
            }));
        }
        // Reasoning cannot be handed back (see the module doc); a built-in
        // tool call is history metadata, as Ollama treats it.
        "reasoning" | "web_search_call" => {}
        // `compaction` and anything else carries context this cannot
        // replay; refused, as Ollama's decoder refuses unknown items.
        other => anyhow::bail!("unsupported input item type: {other:?}"),
    }
    Ok(())
}

/// Ollama's merge: an assistant message followed by the output of a call
/// the previous assistant message made is folded into that message, so
/// the tool result still directly follows its call.
fn merge_replayed_assistant(messages: &mut [Value], content: &Value, next: Option<&Value>) -> bool {
    let Some(output_call_id) = next
        .filter(|n| n.get("type").and_then(Value::as_str) == Some("function_call_output"))
        .and_then(|n| n.get("call_id").and_then(Value::as_str))
    else {
        return false;
    };
    let Some(last) = messages.last_mut() else {
        return false;
    };
    if last.get("role").and_then(Value::as_str) != Some("assistant") {
        return false;
    }
    let carries_call = last
        .get("tool_calls")
        .and_then(Value::as_array)
        .is_some_and(|calls| {
            calls
                .iter()
                .any(|c| c.get("id").and_then(Value::as_str) == Some(output_call_id))
        });
    if !carries_call {
        return false;
    }
    let content = content_text(content);
    if !content.is_empty() {
        let existing = last.get("content").and_then(Value::as_str).unwrap_or("");
        let joined = if existing.is_empty() {
            content
        } else {
            format!("{existing}\n{content}")
        };
        last["content"] = Value::String(joined);
    }
    true
}

/// Ollama's `appendResponseToolCall`.
fn append_response_tool_call(messages: &mut Vec<Value>, call: Value) {
    if let Some(last) = messages.last_mut() {
        if last.get("role").and_then(Value::as_str) == Some("assistant") {
            match last.get_mut("tool_calls").and_then(Value::as_array_mut) {
                Some(calls) => calls.push(call),
                None => last["tool_calls"] = json!([call]),
            }
            return;
        }
    }
    messages.push(json!({ "role": "assistant", "tool_calls": [call] }));
}

/// Ollama's `convertTools`: a `namespace` tool becomes each of its
/// members under a qualified name; non-function tools are dropped.
fn convert_tools(tool: &Value, out: &mut Vec<Value>) {
    match tool.get("type").and_then(Value::as_str) {
        Some("function") => out.push(convert_tool(tool, "")),
        Some("namespace") => {
            let namespace = tool.get("name").and_then(Value::as_str).unwrap_or("");
            for member in tool
                .get("tools")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                match member.get("type").and_then(Value::as_str) {
                    Some("function") => out.push(convert_tool(member, namespace)),
                    Some("namespace") => {
                        let mut inner = Vec::new();
                        convert_tools(member, &mut inner);
                        for mut t in inner {
                            let name = t["function"]["name"].as_str().unwrap_or("").to_string();
                            t["function"]["name"] =
                                Value::String(qualify_namespace_tool_name(namespace, &name));
                            out.push(t);
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

/// Ollama's `convertTool`, plus `strict`, which chat completions has too.
fn convert_tool(tool: &Value, namespace: &str) -> Value {
    let name = tool.get("name").and_then(Value::as_str).unwrap_or("");
    let mut function = json!({ "name": qualify_namespace_tool_name(namespace, name) });
    for key in ["description", "parameters", "strict"] {
        if let Some(v) = tool.get(key).filter(|v| !v.is_null()) {
            function[key] = v.clone();
        }
    }
    json!({ "type": "function", "function": function })
}

/// Ollama's `qualifyNamespaceToolName`, with `_` wherever Ollama has `.`:
/// providers enforce `^[a-zA-Z0-9_-]{1,128}$` on function names (Anthropic
/// rejects Codex's `multi_agent_v1.spawn_agent`).
fn qualify_namespace_tool_name(namespace: &str, member: &str) -> String {
    if namespace.is_empty() || member.is_empty() {
        return member.to_string();
    }
    if let Some(rest) = member.strip_prefix(&format!("{namespace}.")) {
        return format!("{namespace}_{rest}");
    }
    if member.starts_with(&format!("{namespace}_")) {
        return member.to_string();
    }
    if member.starts_with('_') {
        return format!("{namespace}{member}");
    }
    format!("{namespace}_{member}")
}

/// Ollama's `responsesToolCallName`: `(namespace, name)` for a qualified
/// member of one of the request's `namespace` tools, else `("", qualified)`.
fn responses_tool_call_name(tools: &[Value], qualified: &str) -> (String, String) {
    for tool in tools {
        if tool.get("type").and_then(Value::as_str) != Some("namespace") {
            continue;
        }
        let Some(namespace) = tool
            .get("name")
            .and_then(Value::as_str)
            .filter(|n| !n.is_empty())
        else {
            continue;
        };
        for member in tool
            .get("tools")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if member.get("type").and_then(Value::as_str) == Some("namespace") {
                continue;
            }
            let name = member.get("name").and_then(Value::as_str).unwrap_or("");
            let native = qualify_namespace_tool_name(namespace, name);
            if qualified == native
                || qualified == format!("{namespace}.{name}")
                || qualified == format!("{namespace}:{name}")
            {
                return (namespace.to_string(), name.to_string());
            }
        }
    }
    (String::new(), qualified.to_string())
}

/// Ollama's `convertResponsesContent`: a bare string when all text (every
/// provider accepts that), ordered `text`/`image_url` parts when an image
/// forces them. Unknown block types are an error, as in Ollama's decoder.
fn convert_responses_content(content: &Value) -> anyhow::Result<Value> {
    let Value::Array(blocks) = content else {
        return Ok(Value::String(content.as_str().unwrap_or("").to_string()));
    };
    let mut parts = Vec::new();
    let mut has_image = false;
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("input_text" | "output_text" | "text") => {
                let text = block.get("text").and_then(Value::as_str).unwrap_or("");
                parts.push(json!({ "type": "text", "text": text }));
            }
            Some("input_image") => {
                // A `file_id` image has no URL to send; skipped, as Ollama skips it.
                if let Some(url) = block
                    .get("image_url")
                    .and_then(Value::as_str)
                    .filter(|u| !u.is_empty())
                {
                    has_image = true;
                    let mut image_url = json!({ "url": url });
                    if let Some(detail) = block.get("detail").filter(|d| d.is_string()) {
                        image_url["detail"] = detail.clone();
                    }
                    parts.push(json!({ "type": "image_url", "image_url": image_url }));
                }
            }
            Some("input_file") => anyhow::bail!("file inputs are not currently supported"),
            other => anyhow::bail!("unknown content type: {}", other.unwrap_or("")),
        }
    }
    if has_image {
        Ok(Value::Array(parts))
    } else {
        Ok(Value::String(content_text(&Value::Array(parts))))
    }
}

/// The text of a chat `content` value: the string itself, or its `text`
/// parts joined.
fn content_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect(),
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Response: chat completion SSE -> Responses SSE (Ollama's
// ResponsesStreamConverter)
// ---------------------------------------------------------------------------

/// One tool call, assembled from streamed argument fragments.
#[derive(Default)]
struct ToolCall {
    id: Option<String>,
    name: String,
    arguments: String,
}

/// Ollama's `ResponsesStreamConverter`, fed chat-completion SSE lines.
///
/// Feed each upstream line to [`StreamConverter::line`], then
/// [`StreamConverter::finish`] at end of stream; each returns SSE text to
/// relay. Completion is emitted on `[DONE]`, not on the `finish_reason`
/// chunk, because `usage` follows in one more chunk.
pub(super) struct StreamConverter {
    // Configuration.
    response_id: String,
    item_id: String,
    model: String,
    request: Value,
    created_at: u64,

    // State.
    first_write: bool,
    output_index: usize,
    content_started: bool,
    accumulated_text: String,
    sequence_number: u64,
    accumulated_thinking: String,
    reasoning_item_id: String,
    reasoning_started: bool,
    tool_calls: BTreeMap<usize, ToolCall>,
    completed_items: Vec<Value>,
    usage: Option<Value>,
    saw_finish: bool,
    done: bool,
    error: Option<Value>,
}

impl StreamConverter {
    /// `model` is the name the client asked for; `request` is echoed back
    /// into every `response` object as Ollama's `buildResponseObject` does.
    pub(super) fn new(model: &str, request: &Value) -> Self {
        let suffix = super::gen_id();
        Self::with_ids(
            model,
            request,
            &format!("resp_{suffix}"),
            &format!("msg_{suffix}"),
            now_unix(),
        )
    }

    fn with_ids(
        model: &str,
        request: &Value,
        response_id: &str,
        item_id: &str,
        created_at: u64,
    ) -> Self {
        Self {
            response_id: response_id.to_string(),
            item_id: item_id.to_string(),
            model: model.to_string(),
            request: request.clone(),
            created_at,
            first_write: true,
            output_index: 0,
            content_started: false,
            accumulated_text: String::new(),
            sequence_number: 0,
            accumulated_thinking: String::new(),
            reasoning_item_id: String::new(),
            reasoning_started: false,
            tool_calls: BTreeMap::new(),
            completed_items: Vec::new(),
            usage: None,
            saw_finish: false,
            done: false,
            error: None,
        }
    }

    /// Translates one upstream SSE line, line ending already stripped.
    pub(super) fn line(&mut self, line: &str) -> String {
        if self.done {
            return String::new();
        }
        let Some(payload) = line.strip_prefix("data:") else {
            return String::new();
        };
        let payload = payload.trim();
        let mut out = self.first_write();
        if payload == "[DONE]" {
            out.push_str(&self.process_completion());
            return out;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(payload) else {
            return out;
        };

        if let Some(usage) = chunk.get("usage").filter(|u| u.is_object()) {
            self.usage = Some(usage_of(usage));
        }

        let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
        else {
            // The trailing usage chunk, or an in-band error.
            if let Some(error) = chunk.get("error") {
                let message = error
                    .get("message")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .unwrap_or_else(|| error.to_string());
                out.push_str(&self.response_failed(&message));
            }
            return out;
        };
        let delta = choice.get("delta").unwrap_or(&Value::Null);

        // Reasoning first, as Ollama orders it. Field names: llama-server
        // and most providers, OpenRouter, llama.cpp git.
        let thinking = ["reasoning_content", "reasoning", "thinking"]
            .into_iter()
            .find_map(|k| delta.get(k).and_then(Value::as_str))
            .filter(|s| !s.is_empty());
        if let Some(thinking) = thinking {
            out.push_str(&self.process_thinking(thinking));
        }

        // Fragments, unlike Ollama's whole `api.ToolCall`s: accumulated
        // until completion.
        if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for (position, call) in calls.iter().enumerate() {
                let index = call
                    .get("index")
                    .and_then(Value::as_u64)
                    .map(|i| i as usize)
                    .unwrap_or(position);
                let entry = self.tool_calls.entry(index).or_default();
                if let Some(id) = call
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                {
                    entry.id.get_or_insert_with(|| id.to_string());
                }
                if let Some(function) = call.get("function") {
                    if let Some(name) = function.get("name").and_then(Value::as_str) {
                        entry.name.push_str(name);
                    }
                    if let Some(args) = function.get("arguments").and_then(Value::as_str) {
                        entry.arguments.push_str(args);
                    }
                }
            }
        }

        if let Some(content) = delta
            .get("content")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            out.push_str(&self.process_text_content(content));
        }

        if choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .is_some_and(|r| !r.is_empty() && r != "null")
        {
            self.saw_finish = true;
        }
        out
    }

    /// End of upstream stream without `[DONE]`: completes if a
    /// `finish_reason` was seen, else fails — `bytes_to_lines` turns a
    /// dropped connection into EOF, and partial output is not completion.
    pub(super) fn finish(&mut self) -> String {
        if self.done {
            return String::new();
        }
        let mut out = self.first_write();
        if self.saw_finish {
            out.push_str(&self.process_completion());
        } else {
            out.push_str(&self.response_failed("upstream stream ended before completion"));
        }
        out
    }

    /// Whether the response ended in `response.failed`.
    pub(super) fn failed(&self) -> bool {
        self.error.is_some()
    }

    /// Ollama's `ToResponse`: the one `response` object a `stream: false`
    /// caller expects.
    pub(super) fn fold<I, S>(&mut self, lines: I) -> Value
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for line in lines {
            self.line(line.as_ref());
        }
        self.finish();
        self.final_response()
    }

    /// The `response` object as it stands once the stream is over.
    fn final_response(&self) -> Value {
        let mut response = if self.error.is_some() {
            let mut r = self.build_response_object(
                "failed",
                self.completed_items.clone(),
                self.usage.clone(),
            );
            r["error"] = self.error.clone().unwrap_or(Value::Null);
            r
        } else {
            self.build_response_object(
                "completed",
                self.completed_items.clone(),
                self.usage.clone(),
            )
        };
        response["completed_at"] = Value::from(now_unix());
        response
    }

    /// Ollama emits `created`/`in_progress` with the first chunk.
    fn first_write(&mut self) -> String {
        if !self.first_write {
            return String::new();
        }
        self.first_write = false;
        let mut out = self.new_event(
            "response.created",
            json!({ "response": self.build_response_object("in_progress", Vec::new(), None) }),
        );
        out.push_str(&self.new_event(
            "response.in_progress",
            json!({ "response": self.build_response_object("in_progress", Vec::new(), None) }),
        ));
        out
    }

    /// Ollama's `buildResponseObject`, echoing the request.
    fn build_response_object(
        &self,
        status: &str,
        output: Vec<Value>,
        usage: Option<Value>,
    ) -> Value {
        let req = &self.request;
        let instructions = req
            .get("instructions")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(|s| Value::String(s.to_string()))
            .unwrap_or(Value::Null);
        let truncation = req
            .get("truncation")
            .cloned()
            .filter(|v| v.is_string())
            .unwrap_or_else(|| json!("disabled"));
        let tools = req
            .get("tools")
            .cloned()
            .filter(|v| v.is_array())
            .unwrap_or_else(|| json!([]));
        let text_format = req
            .pointer("/text/format")
            .cloned()
            .filter(|f| f.is_object())
            .unwrap_or_else(|| json!({ "type": "text" }));
        let reasoning = match req.get("reasoning") {
            Some(r) if r.get("effort").is_some() || r.get("summary").is_some() => json!({
                "effort": r.get("effort").cloned().unwrap_or(Value::Null),
                "summary": r.get("summary").cloned().unwrap_or(Value::Null),
            }),
            _ => Value::Null,
        };
        json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "completed_at": Value::Null,
            "status": status,
            "incomplete_details": Value::Null,
            "model": self.model,
            "previous_response_id": Value::Null,
            "instructions": instructions,
            "output": output,
            "error": Value::Null,
            "tools": tools,
            "tool_choice": req.get("tool_choice").cloned().filter(|v| !v.is_null()).unwrap_or_else(|| json!("auto")),
            "truncation": truncation,
            "parallel_tool_calls": req.get("parallel_tool_calls").and_then(Value::as_bool).unwrap_or(true),
            "text": { "format": text_format },
            "top_p": req.get("top_p").cloned().filter(|v| v.is_number()).unwrap_or(json!(1.0)),
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "top_logprobs": 0,
            "temperature": req.get("temperature").cloned().filter(|v| v.is_number()).unwrap_or(json!(1.0)),
            "reasoning": reasoning,
            "usage": usage,
            "max_output_tokens": req.get("max_output_tokens").cloned().unwrap_or(Value::Null),
            "max_tool_calls": Value::Null,
            "store": false,
            "background": req.get("background").and_then(Value::as_bool).unwrap_or(false),
            "service_tier": "default",
            "metadata": {},
            "safety_identifier": Value::Null,
            "prompt_cache_key": Value::Null,
        })
    }

    /// Ollama's `processThinking`.
    fn process_thinking(&mut self, thinking: &str) -> String {
        let mut out = String::new();
        if !self.reasoning_started {
            self.reasoning_started = true;
            self.reasoning_item_id = format!("rs_{}", self.response_id.trim_start_matches("resp_"));
            out.push_str(&self.new_event(
                "response.output_item.added",
                json!({
                    "output_index": self.output_index,
                    "item": { "id": self.reasoning_item_id, "type": "reasoning", "summary": [] },
                }),
            ));
        }
        self.accumulated_thinking.push_str(thinking);
        out.push_str(&self.new_event(
            "response.reasoning_summary_text.delta",
            json!({
                "item_id": self.reasoning_item_id,
                "output_index": self.output_index,
                "summary_index": 0,
                "delta": thinking,
            }),
        ));
        out
    }

    /// Ollama's `finishReasoning`.
    fn finish_reasoning(&mut self) -> String {
        if !self.reasoning_started {
            return String::new();
        }
        let item_id = std::mem::take(&mut self.reasoning_item_id);
        let thinking = std::mem::take(&mut self.accumulated_thinking);
        let output_index = self.output_index;
        let item = json!({
            "id": item_id,
            "type": "reasoning",
            "summary": [{ "type": "summary_text", "text": thinking }],
            "encrypted_content": thinking,
        });
        self.completed_items.push(item.clone());
        self.reasoning_started = false;
        self.output_index += 1;

        let mut out = self.new_event(
            "response.reasoning_summary_text.done",
            json!({ "item_id": item_id, "output_index": output_index, "summary_index": 0, "text": thinking }),
        );
        out.push_str(&self.new_event(
            "response.output_item.done",
            json!({ "output_index": output_index, "item": item }),
        ));
        out
    }

    /// Ollama's `processTextContent`.
    fn process_text_content(&mut self, content: &str) -> String {
        let mut out = self.finish_reasoning();
        if !self.content_started {
            self.content_started = true;
            out.push_str(&self.new_event(
                "response.output_item.added",
                json!({
                    "output_index": self.output_index,
                    "item": {
                        "id": self.item_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                }),
            ));
            out.push_str(&self.new_event(
                "response.content_part.added",
                json!({
                    "item_id": self.item_id,
                    "output_index": self.output_index,
                    "content_index": 0,
                    "part": { "type": "output_text", "text": "", "annotations": [], "logprobs": [] },
                }),
            ));
        }
        self.accumulated_text.push_str(content);
        out.push_str(&self.new_event(
            "response.output_text.delta",
            json!({
                "item_id": self.item_id,
                "output_index": self.output_index,
                "content_index": 0,
                "delta": content,
                "logprobs": [],
            }),
        ));
        out
    }

    /// Ollama's `FinishMessageItem`.
    fn finish_message_item(&mut self) -> String {
        if !self.content_started {
            return String::new();
        }
        self.content_started = false;
        let text = std::mem::take(&mut self.accumulated_text);
        let item_id = self.item_id.clone();
        let output_index = self.output_index;
        let part =
            json!({ "type": "output_text", "text": text, "annotations": [], "logprobs": [] });
        let item = json!({
            "id": item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [part.clone()],
        });
        self.completed_items.push(item.clone());
        self.output_index += 1;

        let mut out = self.new_event(
            "response.output_text.done",
            json!({ "item_id": item_id, "output_index": output_index, "content_index": 0, "text": text, "logprobs": [] }),
        );
        out.push_str(&self.new_event(
            "response.content_part.done",
            json!({ "item_id": item_id, "output_index": output_index, "content_index": 0, "part": part }),
        ));
        out.push_str(&self.new_event(
            "response.output_item.done",
            json!({ "output_index": output_index, "item": item }),
        ));
        out
    }

    /// Ollama's `emitFunctionCallEvents`, over the assembled calls.
    fn emit_function_call_events(&mut self) -> String {
        let mut out = String::new();
        let calls = std::mem::take(&mut self.tool_calls);
        let tools = self
            .request
            .get("tools")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let suffix = self.response_id.trim_start_matches("resp_").to_string();
        let mut emitted = 0;
        for (i, call) in calls.into_values().enumerate() {
            let output_index = self.output_index + i;
            let fc_item_id = format!("fc_{suffix}_{i}");
            let call_id = call.id.unwrap_or_else(|| format!("call_{suffix}_{i}"));
            let (namespace, name) = responses_tool_call_name(&tools, &call.name);

            let mut in_progress = json!({
                "id": fc_item_id,
                "type": "function_call",
                "status": "in_progress",
                "call_id": call_id,
                "name": name,
                "arguments": "",
            });
            let mut item = in_progress.clone();
            item["status"] = json!("completed");
            item["arguments"] = Value::String(call.arguments.clone());
            if !namespace.is_empty() {
                in_progress["namespace"] = Value::String(namespace.clone());
                item["namespace"] = Value::String(namespace);
            }
            self.completed_items.push(item.clone());

            out.push_str(&self.new_event(
                "response.output_item.added",
                json!({ "output_index": output_index, "item": in_progress }),
            ));
            out.push_str(&self.new_event(
                "response.function_call_arguments.delta",
                json!({ "item_id": fc_item_id, "output_index": output_index, "delta": call.arguments }),
            ));
            out.push_str(&self.new_event(
                "response.function_call_arguments.done",
                json!({ "item_id": fc_item_id, "output_index": output_index, "arguments": call.arguments }),
            ));
            out.push_str(&self.new_event(
                "response.output_item.done",
                json!({ "output_index": output_index, "item": item }),
            ));
            emitted += 1;
        }
        self.output_index += emitted;
        out
    }

    /// Ollama's `processCompletion`: text closes before the function calls
    /// it announced, as in OpenAI's own output.
    fn process_completion(&mut self) -> String {
        if self.done {
            return String::new();
        }
        self.done = true;
        let mut out = self.finish_reasoning();
        out.push_str(&self.finish_message_item());
        out.push_str(&self.emit_function_call_events());
        let response = self.final_response();
        out.push_str(&self.new_event("response.completed", json!({ "response": response })));
        out
    }

    /// Ollama's `ResponseFailed`.
    fn response_failed(&mut self, message: &str) -> String {
        if self.done {
            return String::new();
        }
        self.done = true;
        self.error = Some(json!({ "code": "server_error", "message": message }));
        let response = self.final_response();
        self.new_event("response.failed", json!({ "response": response }))
    }

    /// Ollama's `newEvent`, already framed as SSE.
    fn new_event(&mut self, kind: &str, mut data: Value) -> String {
        data["type"] = Value::String(kind.to_string());
        data["sequence_number"] = Value::from(self.sequence_number);
        self.sequence_number += 1;
        format!("event: {kind}\ndata: {data}\n\n")
    }
}

/// A chat completion's `usage` in the Responses API's spelling.
fn usage_of(usage: &Value) -> Value {
    let n = |v: Option<&Value>| v.and_then(Value::as_u64).unwrap_or(0);
    let input = n(usage.get("prompt_tokens"));
    let output = n(usage.get("completion_tokens"));
    json!({
        "input_tokens": input,
        "output_tokens": output,
        "total_tokens": input + output,
        "input_tokens_details": {
            "cached_tokens": n(usage.pointer("/prompt_tokens_details/cached_tokens")),
        },
        "output_tokens_details": {
            "reasoning_tokens": n(usage.pointer("/completion_tokens_details/reasoning_tokens")),
        },
    })
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parses every `data:` line of an SSE text back into JSON events.
    fn events(sse: &str) -> Vec<Value> {
        sse.lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    fn kinds(events: &[Value]) -> Vec<&str> {
        events.iter().map(|e| e["type"].as_str().unwrap()).collect()
    }

    fn roles(messages: &[Value]) -> Vec<&str> {
        messages
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect()
    }

    fn converter(request: Value) -> StreamConverter {
        StreamConverter::with_ids(
            "llmman.provider/mockprov/mock-model",
            &request,
            "resp_test",
            "msg_test",
            1_700_000_000,
        )
    }

    fn run(request: Value, lines: &[&str]) -> (StreamConverter, Vec<Value>) {
        let mut c = converter(request);
        let mut sse = String::new();
        for line in lines {
            sse.push_str(&c.line(line));
        }
        sse.push_str(&c.finish());
        (c, events(&sse))
    }

    // -- request ------------------------------------------------------------

    /// The shape Codex actually sends on its first turn: instructions, a
    /// developer message, a user message, function tools and a couple of
    /// tool types chat completions has no name for.
    #[test]
    fn a_codex_first_turn_becomes_chat_messages_and_function_tools() {
        let req = json!({
            "model": "claude-sonnet-5",
            "instructions": "You are Codex.",
            "input": [
                { "type": "message", "role": "developer", "content": [{ "type": "input_text", "text": "Sandbox: read-only." }] },
                { "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "Reply OK" }] }
            ],
            "tools": [
                { "type": "function", "name": "shell", "description": "Run a command", "strict": false,
                  "parameters": { "type": "object", "properties": { "cmd": { "type": "string" } } } },
                { "type": "web_search" }
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": true,
            "store": false,
            "stream": true,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "abc",
            "max_output_tokens": 512
        });
        let chat = from_responses_request(&req).unwrap();

        assert_eq!(chat["model"], "claude-sonnet-5");
        assert_eq!(chat["stream"], true);
        assert_eq!(chat["stream_options"]["include_usage"], true);
        assert_eq!(chat["max_tokens"], 512);
        for gone in [
            "store",
            "include",
            "prompt_cache_key",
            "max_output_tokens",
            "instructions",
            "input",
        ] {
            assert!(chat.get(gone).is_none(), "{gone} leaked through");
        }

        let messages = chat["messages"].as_array().unwrap();
        assert_eq!(
            roles(messages),
            ["system", "system", "user"],
            "{messages:#?}"
        );
        assert_eq!(messages[0]["content"], "You are Codex.");
        assert_eq!(messages[1]["content"], "Sandbox: read-only.");
        assert_eq!(messages[2]["content"], "Reply OK");

        let tools = chat["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1, "web_search must be dropped: {tools:#?}");
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "shell");
        assert_eq!(tools[0]["function"]["description"], "Run a command");
        assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
        assert_eq!(tools[0]["function"]["strict"], false);
        assert_eq!(chat["tool_choice"], "auto");
        assert_eq!(chat["parallel_tool_calls"], true);
    }

    /// The second turn after a tool call: the model's own text and
    /// function_call items come back as one assistant message with
    /// `tool_calls`, its output as a `tool` message, and its reasoning
    /// not at all.
    #[test]
    fn a_tool_call_round_trip_becomes_assistant_tool_calls_and_a_tool_message() {
        let req = json!({
            "model": "m",
            "input": [
                { "role": "user", "content": "list files" },
                { "type": "reasoning", "id": "rs_1", "summary": [{ "type": "summary_text", "text": "I should ls." }], "encrypted_content": "I should ls." },
                { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "Listing." }] },
                { "type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{\"cmd\":\"ls\"}" },
                { "type": "function_call", "call_id": "call_2", "name": "shell", "arguments": "{\"cmd\":\"pwd\"}" },
                { "type": "function_call_output", "call_id": "call_1", "output": "a\nb\n" },
                { "type": "function_call_output", "call_id": "call_2", "output": [{ "type": "input_text", "text": "/tmp" }] }
            ]
        });
        let chat = from_responses_request(&req).unwrap();
        let messages = chat["messages"].as_array().unwrap();
        assert_eq!(
            roles(messages),
            ["user", "assistant", "tool", "tool"],
            "{messages:#?}"
        );
        assert_eq!(messages[1]["content"], "Listing.");
        let calls = messages[1]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["id"], "call_1");
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "shell");
        assert_eq!(calls[0]["function"]["arguments"], "{\"cmd\":\"ls\"}");
        assert_eq!(calls[1]["id"], "call_2");
        assert_eq!(messages[2]["tool_call_id"], "call_1");
        assert_eq!(messages[2]["content"], "a\nb\n");
        assert_eq!(messages[3]["tool_call_id"], "call_2");
        assert_eq!(messages[3]["content"], "/tmp");
        assert!(chat.get("tools").is_none(), "no tools were offered");
        let all: String = chat.to_string();
        assert!(
            !all.contains("I should ls."),
            "reasoning must not reach the provider"
        );
    }

    /// Ollama's merge rule: an assistant message replayed between a
    /// function_call and its output is folded into the message that
    /// made the call, so the tool result still directly follows it.
    #[test]
    fn an_assistant_message_between_a_call_and_its_output_is_merged() {
        let req = json!({ "model": "m", "input": [
            { "role": "user", "content": "go" },
            { "type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}" },
            { "type": "message", "role": "assistant", "content": "Working on it." },
            { "type": "function_call_output", "call_id": "c1", "output": "done" },
            { "type": "message", "role": "assistant", "content": "All done." }
        ]});
        let messages = from_responses_request(&req).unwrap()["messages"].clone();
        let messages = messages.as_array().unwrap();
        assert_eq!(
            roles(messages),
            ["user", "assistant", "tool", "assistant"],
            "{messages:#?}"
        );
        assert_eq!(messages[1]["content"], "Working on it.");
        assert_eq!(messages[1]["tool_calls"][0]["id"], "c1");
        assert_eq!(messages[3]["content"], "All done.");
    }

    /// A function call with no assistant text before it still needs an
    /// assistant message to hang off — with no `content` at all, which
    /// every provider accepts where an empty string is rejected by some.
    #[test]
    fn a_bare_function_call_gets_its_own_assistant_message() {
        let req = json!({ "model": "m", "input": [
            { "role": "user", "content": "go" },
            { "type": "function_call", "call_id": "c", "name": "f", "arguments": {"x": 1} },
            { "type": "function_call_output", "call_id": "c", "output": "done" }
        ]});
        let messages = from_responses_request(&req).unwrap()["messages"].clone();
        assert_eq!(messages[1]["role"], "assistant");
        assert!(messages[1].get("content").is_none(), "{:#?}", messages[1]);
        // Non-string arguments are serialized, never dropped.
        assert_eq!(
            messages[1]["tool_calls"][0]["function"]["arguments"],
            "{\"x\":1}"
        );
    }

    /// Ollama's namespace handling: a `namespace` tool is flattened into
    /// `namespace.member` functions on the way out, a `function_call`
    /// naming a member is qualified the same way on the way back in
    /// (whether it carries `namespace` or a dotted/colon name), and the
    /// model's call to one is reported under its namespace.
    #[test]
    fn namespace_tools_are_flattened_and_calls_to_them_qualified() {
        let tools = json!([
            { "type": "namespace", "name": "agents", "tools": [
                { "type": "function", "name": "spawn", "parameters": { "type": "object" } },
                { "type": "function", "name": "_wait" }
            ]},
            { "type": "function", "name": "shell" }
        ]);
        let req = json!({ "model": "m", "tools": tools, "input": [
            { "role": "user", "content": "go" },
            { "type": "function_call", "call_id": "c1", "namespace": "agents", "name": "spawn", "arguments": "{}" },
            { "type": "function_call", "call_id": "c2", "name": "agents:_wait", "arguments": "{}" },
            { "type": "function_call", "call_id": "c3", "name": "shell", "arguments": "{}" }
        ]});
        let chat = from_responses_request(&req).unwrap();
        let names: Vec<&str> = chat["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["agents_spawn", "agents_wait", "shell"]);
        for name in names {
            assert!(
                name.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
                "{name} is not a chat-completions function name"
            );
        }
        let calls = chat["messages"][1]["tool_calls"].as_array().unwrap();
        assert_eq!(calls[0]["function"]["name"], "agents_spawn");
        assert_eq!(calls[1]["function"]["name"], "agents_wait");
        assert_eq!(calls[2]["function"]["name"], "shell");

        let lines = [
            r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_x","function":{"name":"agents_spawn","arguments":"{}"}}]},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
            "data: [DONE]",
        ];
        let (_, events) = run(json!({ "tools": tools }), &lines);
        let item = &events
            .iter()
            .find(|e| e["type"] == "response.output_item.done")
            .unwrap()["item"];
        assert_eq!(item["namespace"], "agents");
        assert_eq!(item["name"], "spawn");
    }

    #[test]
    fn a_string_input_is_one_user_message() {
        let chat = from_responses_request(&json!({ "model": "m", "input": "hi" })).unwrap();
        assert_eq!(
            chat["messages"],
            json!([{ "role": "user", "content": "hi" }])
        );
    }

    #[test]
    fn an_image_forces_the_array_content_form_and_text_alone_does_not() {
        let with_image = from_responses_request(&json!({ "model": "m", "input": [{
            "role": "user",
            "content": [
                { "type": "input_text", "text": "what is this" },
                { "type": "input_image", "image_url": "data:image/png;base64,AAAA" }
            ]
        }]}))
        .unwrap();
        let content = with_image["messages"][0]["content"].as_array().unwrap();
        assert_eq!(
            content[0],
            json!({ "type": "text", "text": "what is this" })
        );
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(content[1]["image_url"]["url"], "data:image/png;base64,AAAA");

        // Parts keep their order: an image before its caption stays before it.
        let image_first = from_responses_request(&json!({ "model": "m", "input": [{
            "role": "user",
            "content": [
                { "type": "input_image", "image_url": "data:image/png;base64,AAAA" },
                { "type": "input_text", "text": "caption" }
            ]
        }]}))
        .unwrap();
        let content = image_first["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "image_url");
        assert_eq!(content[1]["text"], "caption");

        let detailed = from_responses_request(&json!({ "model": "m", "input": [{
            "role": "user",
            "content": [{ "type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "high" }]
        }]})).unwrap();
        assert_eq!(
            detailed["messages"][0]["content"][0]["image_url"]["detail"],
            "high"
        );

        let audio = from_responses_request(&json!({ "model": "m", "input": [{
            "role": "user",
            "content": [{ "type": "input_audio", "audio_url": "data:audio/wav;base64,AAAA" }]
        }]}));
        assert!(audio.unwrap_err().to_string().contains("input_audio"));

        // An item carrying context this cannot replay is refused too.
        let compaction = from_responses_request(&json!({ "model": "m", "input": [
            { "type": "compaction", "encrypted_content": "..." },
            { "role": "user", "content": "hi" }
        ]}));
        assert!(compaction.unwrap_err().to_string().contains("compaction"));

        let file = from_responses_request(&json!({ "model": "m", "input": [{
            "role": "user",
            "content": [{ "type": "input_file", "filename": "a.pdf", "file_data": "..." }]
        }]}));
        assert!(file.unwrap_err().to_string().contains("file inputs"));

        let text_only = from_responses_request(&json!({ "model": "m", "input": [{
            "role": "user",
            "content": [{ "type": "input_text", "text": "a" }, { "type": "input_text", "text": "b" }]
        }]})).unwrap();
        assert_eq!(text_only["messages"][0]["content"], "ab");
    }

    #[test]
    fn tool_choice_object_form_is_nested_the_chat_way() {
        let req = json!({ "model": "m", "input": "x",
            "tools": [{ "type": "function", "name": "f" }],
            "tool_choice": { "type": "function", "name": "f" } });
        assert_eq!(
            from_responses_request(&req).unwrap()["tool_choice"],
            json!({ "type": "function", "function": { "name": "f" } })
        );
        let req = json!({ "model": "m", "input": "x",
            "tools": [{ "type": "function", "name": "f" }], "tool_choice": "required" });
        assert_eq!(
            from_responses_request(&req).unwrap()["tool_choice"],
            "required"
        );

        // A forced namespace member is named as convert_tools named it,
        // whether the namespace is explicit or in the name.
        let ns_tools = json!([{ "type": "namespace", "name": "ns", "tools": [{ "type": "function", "name": "f" }] }]);
        for choice in [
            json!({ "type": "function", "namespace": "ns", "name": "f" }),
            json!({ "type": "function", "name": "ns.f" }),
        ] {
            let req =
                json!({ "model": "m", "input": "x", "tools": ns_tools, "tool_choice": choice });
            assert_eq!(
                from_responses_request(&req).unwrap()["tool_choice"]["function"]["name"],
                "ns_f"
            );
        }
        // A choice of a hosted tool convert_tools dropped is `auto`.
        let req = json!({ "model": "m", "input": "x",
            "tools": [{ "type": "function", "name": "f" }, { "type": "web_search" }],
            "tool_choice": { "type": "web_search" } });
        assert_eq!(from_responses_request(&req).unwrap()["tool_choice"], "auto");
    }

    /// `tool_choice` and `parallel_tool_calls` without any tools is a 400
    /// from OpenAI itself; when every tool was a non-function one they
    /// must go with them.
    #[test]
    fn tool_options_are_dropped_with_the_last_tool() {
        let req = json!({ "model": "m", "input": "x",
            "tools": [{ "type": "web_search" }], "tool_choice": "auto", "parallel_tool_calls": false });
        let chat = from_responses_request(&req).unwrap();
        assert!(chat.get("tools").is_none());
        assert!(chat.get("tool_choice").is_none());
        assert!(chat.get("parallel_tool_calls").is_none());
    }

    /// Ollama maps `reasoning.effort` to its think level and a
    /// `json_schema` text format to its `format`; the chat-completions
    /// spellings are `reasoning_effort` and `response_format`.
    #[test]
    fn reasoning_effort_and_json_schema_take_their_chat_spellings() {
        let req = json!({ "model": "m", "input": "x",
            "reasoning": { "effort": "high", "summary": "auto" },
            "text": { "format": { "type": "json_schema", "name": "answer", "strict": true,
                                  "schema": { "type": "object", "properties": { "ok": { "type": "boolean" } } } } } });
        let chat = from_responses_request(&req).unwrap();
        assert_eq!(chat["reasoning_effort"], "high");
        assert_eq!(chat["response_format"]["type"], "json_schema");
        assert_eq!(chat["response_format"]["json_schema"]["name"], "answer");
        assert_eq!(chat["response_format"]["json_schema"]["strict"], true);
        assert_eq!(
            chat["response_format"]["json_schema"]["schema"]["type"],
            "object"
        );

        let plain = from_responses_request(
            &json!({ "model": "m", "input": "x", "text": { "format": { "type": "text" } } }),
        )
        .unwrap();
        assert!(plain.get("response_format").is_none());

        let object = from_responses_request(
            &json!({ "model": "m", "input": "x", "text": { "format": { "type": "json_object" } } }),
        )
        .unwrap();
        assert_eq!(object["response_format"], json!({ "type": "json_object" }));
        assert!(plain.get("reasoning_effort").is_none());
    }

    // -- response -----------------------------------------------------------

    const TEXT_STREAM: &[&str] = &[
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}",
        "",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"O\"},\"finish_reason\":null}]}",
        "",
        ": keep-alive",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"K\"},\"finish_reason\":null}]}",
        "",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}",
        "",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":4,\"total_tokens\":14,\"prompt_tokens_details\":{\"cached_tokens\":3}}}",
        "",
        "data: [DONE]",
    ];

    fn text_request() -> Value {
        json!({ "model": "mock-model", "instructions": "Be terse.", "input": "hi",
                "temperature": 0.2, "max_output_tokens": 64, "tools": [{ "type": "function", "name": "f" }] })
    }

    /// The event sequence Codex requires — Ollama's, exactly — with the
    /// usage that only arrives after `finish_reason` still making it
    /// into `completed`, and the request echoed back into every
    /// `response` object.
    #[test]
    fn a_text_reply_is_ollamas_event_sequence() {
        let (c, events) = run(text_request(), TEXT_STREAM);
        assert_eq!(
            kinds(&events),
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        assert!(!c.failed());

        let created = &events[0]["response"];
        assert_eq!(created["id"], "resp_test");
        assert_eq!(created["object"], "response");
        assert_eq!(created["status"], "in_progress");
        assert_eq!(created["model"], "llmman.provider/mockprov/mock-model");
        assert_eq!(created["output"], json!([]));
        assert!(created["usage"].is_null());
        assert!(created["completed_at"].is_null());
        // Ollama's buildResponseObject echoes the request.
        assert_eq!(created["instructions"], "Be terse.");
        assert_eq!(created["temperature"], 0.2);
        assert_eq!(created["top_p"], 1.0);
        assert_eq!(created["max_output_tokens"], 64);
        assert_eq!(created["tools"][0]["name"], "f");
        assert_eq!(created["tool_choice"], "auto");
        assert_eq!(created["truncation"], "disabled");
        assert_eq!(created["parallel_tool_calls"], true);

        // Supplied values are echoed rather than the defaults.
        let (_, echoed) = run(
            json!({ "tool_choice": "required", "parallel_tool_calls": false }),
            TEXT_STREAM,
        );
        assert_eq!(echoed[0]["response"]["tool_choice"], "required");
        assert_eq!(echoed[0]["response"]["parallel_tool_calls"], false);
        assert_eq!(created["text"]["format"]["type"], "text");
        assert_eq!(created["store"], false);
        assert_eq!(created["service_tier"], "default");
        assert!(created["reasoning"].is_null());

        assert_eq!(events[4]["delta"], "O");
        assert_eq!(events[4]["item_id"], "msg_test");
        assert_eq!(events[4]["output_index"], 0);
        assert_eq!(events[4]["content_index"], 0);
        assert_eq!(events[4]["logprobs"], json!([]));
        assert_eq!(events[5]["delta"], "K");

        let item = &events[8]["item"];
        assert_eq!(item["type"], "message");
        assert_eq!(item["role"], "assistant");
        assert_eq!(item["status"], "completed");
        assert_eq!(
            item["content"],
            json!([{ "type": "output_text", "text": "OK", "annotations": [], "logprobs": [] }])
        );

        let completed = &events[9]["response"];
        assert_eq!(completed["status"], "completed");
        assert!(completed["completed_at"].is_number());
        assert_eq!(completed["output"].as_array().unwrap().len(), 1);
        assert_eq!(completed["output"][0], *item);
        assert_eq!(completed["usage"]["input_tokens"], 10);
        assert_eq!(completed["usage"]["output_tokens"], 4);
        assert_eq!(completed["usage"]["total_tokens"], 14);
        assert_eq!(
            completed["usage"]["input_tokens_details"]["cached_tokens"],
            3
        );
        assert_eq!(
            completed["usage"]["output_tokens_details"]["reasoning_tokens"],
            0
        );

        // Sequence numbers are contiguous from zero.
        let seq: Vec<u64> = events
            .iter()
            .map(|e| e["sequence_number"].as_u64().unwrap())
            .collect();
        assert_eq!(seq, (0..events.len() as u64).collect::<Vec<_>>());
    }

    /// Every event is framed `event: <type>` / `data: <json>` with the
    /// type repeated inside, since Codex reads the JSON, not the name.
    #[test]
    fn events_are_framed_as_sse() {
        let mut c = converter(json!({}));
        let sse = c.line(TEXT_STREAM[0]);
        let mut lines = sse.lines();
        assert_eq!(lines.next(), Some("event: response.created"));
        let data = lines.next().unwrap().strip_prefix("data: ").unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(data).unwrap()["type"],
            "response.created"
        );
        assert_eq!(lines.next(), Some(""));
        assert_eq!(lines.next(), Some("event: response.in_progress"));
    }

    /// Fragmented, interleaved tool-call deltas (as OpenAI-compatible
    /// servers really stream parallel calls) come out as whole
    /// function_call items, in index order, after any text, each with
    /// Ollama's four events.
    #[test]
    fn tool_calls_are_assembled_into_function_call_items() {
        let lines = [
            r#"data: {"choices":[{"index":0,"delta":{"content":"Running."},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"shell","arguments":""}}]},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"shell","arguments":"{\"cmd\":"}}]},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"cmd\":\"ls\"}"}}]},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"pwd\"}"}}]},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
            "data: [DONE]",
        ];
        let (_, events) = run(json!({}), &lines);
        let after_text: Vec<&str> = kinds(&events)[8..].to_vec();
        assert_eq!(
            after_text,
            [
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        let done: Vec<&Value> = events
            .iter()
            .filter(|e| e["type"] == "response.output_item.done")
            .map(|e| &e["item"])
            .collect();
        assert_eq!(done.len(), 3, "{events:#?}");
        assert_eq!(done[0]["type"], "message");
        assert_eq!(done[0]["content"][0]["text"], "Running.");
        assert_eq!(done[1]["type"], "function_call");
        assert_eq!(done[1]["call_id"], "call_a");
        assert_eq!(done[1]["name"], "shell");
        assert_eq!(done[1]["arguments"], "{\"cmd\":\"ls\"}");
        assert_eq!(done[1]["status"], "completed");
        assert!(done[1].get("namespace").is_none());
        assert_eq!(done[2]["call_id"], "call_b");
        assert_eq!(done[2]["arguments"], "{\"cmd\":\"pwd\"}");
        let added: Vec<&Value> = events
            .iter()
            .filter(|e| {
                e["type"] == "response.output_item.added" && e["item"]["type"] == "function_call"
            })
            .collect();
        assert_eq!(added[0]["item"]["status"], "in_progress");
        assert_eq!(added[0]["item"]["arguments"], "");
        assert_eq!(added[0]["output_index"], 1);
        assert_eq!(added[1]["output_index"], 2);

        let completed = events.last().unwrap();
        assert_eq!(completed["response"]["status"], "completed");
        let output = completed["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 3);
        assert_eq!(output[1]["id"], "fc_test_0");
        assert_eq!(output[2]["id"], "fc_test_1");
        let arg_deltas: Vec<&str> = events
            .iter()
            .filter(|e| e["type"] == "response.function_call_arguments.delta")
            .map(|e| e["delta"].as_str().unwrap())
            .collect();
        assert_eq!(arg_deltas, ["{\"cmd\":\"ls\"}", "{\"cmd\":\"pwd\"}"]);
    }

    /// A provider that streams tool calls without ids still yields a
    /// usable `call_id`, since Codex sends it back verbatim.
    #[test]
    fn a_tool_call_without_an_id_is_given_one() {
        let lines = [
            r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"f","arguments":"{}"}}]},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
            "data: [DONE]",
        ];
        let (_, events) = run(json!({}), &lines);
        let item = &events
            .iter()
            .find(|e| e["type"] == "response.output_item.done")
            .unwrap()["item"];
        assert_eq!(item["call_id"], "call_test_0");
        assert_eq!(item["name"], "f");
    }

    /// Reasoning is its own item, streamed as Ollama streams it —
    /// `reasoning_summary_text` deltas, a `summary_text` summary and the
    /// text again as `encrypted_content` — closed the moment the answer
    /// starts, under whichever of the three field names the provider
    /// uses.
    #[test]
    fn reasoning_precedes_the_message_as_its_own_item() {
        for field in ["reasoning_content", "reasoning", "thinking"] {
            let lines = [
                format!(
                    r#"data: {{"choices":[{{"index":0,"delta":{{"{field}":"Let me "}},"finish_reason":null}}]}}"#
                ),
                format!(
                    r#"data: {{"choices":[{{"index":0,"delta":{{"{field}":"think."}},"finish_reason":null}}]}}"#
                ),
                r#"data: {"choices":[{"index":0,"delta":{"content":"OK"},"finish_reason":null}]}"#
                    .to_string(),
                r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#.to_string(),
                "data: [DONE]".to_string(),
            ];
            let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
            let (_, events) = run(json!({}), &refs);
            assert_eq!(
                kinds(&events),
                [
                    "response.created",
                    "response.in_progress",
                    "response.output_item.added",
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_summary_text.done",
                    "response.output_item.done",
                    "response.output_item.added",
                    "response.content_part.added",
                    "response.output_text.delta",
                    "response.output_text.done",
                    "response.content_part.done",
                    "response.output_item.done",
                    "response.completed",
                ],
                "{field}"
            );
            assert_eq!(
                events[2]["item"],
                json!({ "id": "rs_test", "type": "reasoning", "summary": [] })
            );
            assert_eq!(events[3]["delta"], "Let me ");
            assert_eq!(events[3]["summary_index"], 0);
            assert_eq!(events[5]["text"], "Let me think.");
            let reasoning = &events[6]["item"];
            assert_eq!(reasoning["type"], "reasoning");
            assert_eq!(reasoning["id"], "rs_test");
            assert_eq!(
                reasoning["summary"],
                json!([{ "type": "summary_text", "text": "Let me think." }])
            );
            assert_eq!(reasoning["encrypted_content"], "Let me think.");
            assert_eq!(events[6]["output_index"], 0);
            assert_eq!(events[12]["output_index"], 1);
            let output = events[13]["response"]["output"].as_array().unwrap();
            assert_eq!(output.len(), 2);
            assert_eq!(output[0]["type"], "reasoning");
            assert_eq!(output[1]["type"], "message");
        }
    }

    /// Ollama reports a truncated reply as `completed` with the text it
    /// got, not as `incomplete`; Codex shows the former and errors on
    /// the latter.
    #[test]
    fn a_length_finish_is_still_a_completed_response() {
        let lines = [
            r#"data: {"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}"#,
            "data: [DONE]",
        ];
        let (c, events) = run(json!({}), &lines);
        let last = events.last().unwrap();
        assert_eq!(last["type"], "response.completed");
        assert_eq!(last["response"]["status"], "completed");
        assert!(last["response"]["incomplete_details"].is_null());
        assert_eq!(
            last["response"]["output"][0]["content"][0]["text"],
            "partial"
        );
        assert!(!c.failed());
    }

    /// Not every provider sends `[DONE]`; the end of the stream closes
    /// the response the same way, and only once.
    #[test]
    fn a_stream_without_done_is_completed_at_its_end_exactly_once() {
        let lines = [
            r#"data: {"choices":[{"index":0,"delta":{"content":"OK"},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        ];
        let mut c = converter(json!({}));
        let mut sse = String::new();
        for line in lines {
            sse.push_str(&c.line(line));
        }
        sse.push_str(&c.finish());
        assert_eq!(c.finish(), "", "a second finish emits nothing");
        assert_eq!(c.line("data: [DONE]"), "", "nor does a late [DONE]");
        let events = events(&sse);
        assert_eq!(kinds(&events).last(), Some(&"response.completed"));
        assert_eq!(
            events
                .iter()
                .filter(|e| e["type"] == "response.completed")
                .count(),
            1
        );
    }

    /// A stream that ends without `[DONE]` or a `finish_reason` is a
    /// dropped connection, whether it produced nothing or half a reply;
    /// Codex must see a failure, not a truncated success.
    #[test]
    fn a_stream_cut_off_before_finishing_fails_the_response() {
        for lines in [
            &[][..],
            &[
                r#"data: {"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}"#,
            ][..],
        ] {
            let (c, events) = run(json!({}), lines);
            assert!(c.failed(), "{lines:?}");
            let failed = events.last().unwrap();
            assert_eq!(failed["type"], "response.failed");
            assert_eq!(failed["response"]["status"], "failed");
            assert_eq!(failed["response"]["error"]["code"], "server_error");
            assert!(failed["response"]["error"]["message"]
                .as_str()
                .unwrap()
                .contains("ended"));
            assert_eq!(
                events
                    .iter()
                    .filter(|e| e["type"] == "response.completed")
                    .count(),
                0
            );
        }
    }

    /// Some providers report an error in-band, as a `data:` line with an
    /// `error` object and no `choices`, after a 200. That is
    /// `response.failed`, with the provider's own message.
    #[test]
    fn an_in_band_error_fails_the_response_with_the_providers_message() {
        let lines = [
            r#"data: {"error":{"message":"Rate limit exceeded. Please try again later.","type":"rate_limit_error"}}"#,
            "data: [DONE]",
        ];
        let (c, events) = run(json!({}), &lines);
        assert!(c.failed());
        let failed = events
            .iter()
            .find(|e| e["type"] == "response.failed")
            .unwrap();
        assert_eq!(
            failed["response"]["error"]["message"],
            "Rate limit exceeded. Please try again later."
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| e["type"] == "response.completed")
                .count(),
            0
        );
    }

    /// Non-`data:` lines — blanks, comments, `event:` names some servers
    /// add — and unparseable payloads produce nothing rather than noise
    /// (beyond the opening pair the first `data:` line triggers).
    #[test]
    fn noise_lines_are_ignored() {
        let mut c = converter(json!({}));
        for line in ["", ": ping", "event: message", "id: 7"] {
            assert_eq!(c.line(line), "", "{line:?}");
        }
        let opening = c.line("data: not json");
        assert_eq!(
            kinds(&events(&opening)),
            ["response.created", "response.in_progress"]
        );
        assert_eq!(c.line("data: still not json"), "");
    }

    /// `stream: false` callers get the one object `response.completed`
    /// would have carried.
    #[test]
    fn fold_yields_the_completed_response_object() {
        let response = converter(text_request()).fold(TEXT_STREAM);
        assert_eq!(response["object"], "response");
        assert_eq!(response["status"], "completed");
        assert_eq!(response["output"][0]["content"][0]["text"], "OK");
        assert_eq!(response["usage"]["total_tokens"], 14);
        assert_eq!(response["instructions"], "Be terse.");
        assert!(response["completed_at"].is_number());
        assert!(response["error"].is_null());
    }

    #[test]
    fn usage_totals_are_the_sum_as_ollamas_are() {
        let usage =
            usage_of(&json!({ "prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 99 }));
        assert_eq!(usage["total_tokens"], 12);
        assert_eq!(usage["input_tokens_details"]["cached_tokens"], 0);
    }

    #[test]
    fn qualify_namespace_tool_name_ported_cases() {
        assert_eq!(qualify_namespace_tool_name("", "f"), "f");
        assert_eq!(qualify_namespace_tool_name("ns", ""), "");
        assert_eq!(qualify_namespace_tool_name("ns", "f"), "ns_f");
        assert_eq!(qualify_namespace_tool_name("ns", "ns.f"), "ns_f");
        assert_eq!(qualify_namespace_tool_name("ns", "ns_f"), "ns_f");
        assert_eq!(qualify_namespace_tool_name("ns", "_f"), "ns_f");
    }

    // -- fallback policy ----------------------------------------------------

    #[test]
    fn a_missing_or_broken_route_falls_back_and_a_refusal_does_not() {
        for status in [
            StatusCode::NOT_FOUND,
            StatusCode::METHOD_NOT_ALLOWED,
            StatusCode::NOT_IMPLEMENTED,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::BAD_GATEWAY,
            StatusCode::SERVICE_UNAVAILABLE,
        ] {
            assert!(falls_back(status), "{status}");
        }
        for status in [
            StatusCode::OK,
            StatusCode::BAD_REQUEST,
            StatusCode::UNAUTHORIZED,
            StatusCode::FORBIDDEN,
            StatusCode::PAYMENT_REQUIRED,
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::UNPROCESSABLE_ENTITY,
        ] {
            assert!(!falls_back(status), "{status}");
        }
    }
}
