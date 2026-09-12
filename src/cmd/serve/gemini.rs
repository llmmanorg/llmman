//! Gemini protocol translation for the AGY integration.

use axum::body::{Body, Bytes};
use axum::extract::{Path as UrlPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use axum::Json;
use base64::Engine as _;
use futures::StreamExt;
use serde::Deserialize;

use super::sched::{begin_activity, ActivityGuard};
use super::stream::{bytes_to_lines, oai_chunk_to_content};
use super::types::{OAIChatRequest, OAIChunk, OAIMessage, OAIToolCall, OAIToolCallFunction};
use super::{
    accumulate_tool_call_deltas, backend_wire_model, post_chat, send_with_hybrid_fallback,
    AppError, AppState, Target, ToolCallAccumulator,
};

// ---------------------------------------------------------------------------
// Gemini API types (AGY integration)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub(super) struct GeminiRequest {
    #[serde(default)]
    contents: Vec<GeminiContent>,
    #[serde(default, rename = "systemInstruction")]
    system_instruction: Option<GeminiContent>,
    #[serde(default, rename = "generationConfig")]
    generation_config: Option<serde_json::Value>,
    #[serde(default)]
    tools: Option<serde_json::Value>,
    #[serde(default, rename = "toolConfig")]
    tool_config: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct GeminiContent {
    #[serde(default)]
    role: String,
    #[serde(default)]
    parts: Vec<serde_json::Value>,
}

/// Converts Gemini content parts into the OpenAI-compatible content shape.
fn gemini_content(parts: &[serde_json::Value]) -> Result<serde_json::Value, String> {
    let mut content = Vec::new();
    for part in parts {
        // Replayed reasoning must not become visible assistant content.
        if part.get("thought").and_then(serde_json::Value::as_bool) == Some(true) {
            continue;
        }
        if let Some(text) = part.get("text").and_then(serde_json::Value::as_str) {
            content.push(serde_json::json!({"type": "text", "text": text}));
        } else if let Some(data) = part.get("inlineData") {
            let mime_type = data
                .get("mimeType")
                .and_then(serde_json::Value::as_str)
                .filter(|mime| {
                    matches!(
                        *mime,
                        "image/png" | "image/jpeg" | "image/webp" | "image/gif"
                    )
                })
                .ok_or_else(|| "unsupported Gemini inlineData MIME type".to_string())?;
            let bytes = data
                .get("data")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "Gemini inlineData requires base64 data".to_string())?;
            base64::engine::general_purpose::STANDARD
                .decode(bytes)
                .map_err(|_| "Gemini inlineData must be valid base64".to_string())?;
            content.push(serde_json::json!({
                "type": "image_url",
                "image_url": {"url": format!("data:{mime_type};base64,{bytes}")}
            }));
        } else if part.get("functionCall").is_none() && part.get("functionResponse").is_none() {
            return Err("unsupported Gemini content part".to_string());
        }
    }
    match content.len() {
        0 => Ok(serde_json::Value::String(String::new())),
        1 if content[0]["type"] == "text" => Ok(serde_json::Value::String(
            content[0]["text"].as_str().unwrap_or_default().to_string(),
        )),
        _ => Ok(serde_json::Value::Array(content)),
    }
}

/// Converts a complete Gemini conversation, including function calls, to chat messages.
fn gemini_messages(req: &GeminiRequest) -> Result<Vec<OAIMessage>, String> {
    let mut messages = Vec::new();
    let mut pending_calls =
        std::collections::BTreeMap::<String, std::collections::VecDeque<String>>::new();
    let mut used_call_ids = req
        .contents
        .iter()
        .flat_map(|content| &content.parts)
        .flat_map(|part| [part.get("functionCall"), part.get("functionResponse")])
        .flatten()
        .filter_map(|call| call.get("id").and_then(serde_json::Value::as_str))
        .map(str::to_string)
        .collect::<std::collections::BTreeSet<_>>();
    let mut next_call_id = 0;
    let mut new_call_id = |name: &str| loop {
        let id = format!("call_{name}_{next_call_id}");
        next_call_id += 1;
        if used_call_ids.insert(id.clone()) {
            break id;
        }
    };
    if let Some(system) = &req.system_instruction {
        let content = gemini_content(&system.parts)?;
        if content != serde_json::Value::String(String::new()) {
            messages.push(OAIMessage {
                role: "system".into(),
                content,
                tool_calls: None,
                name: None,
                tool_call_id: None,
            });
        }
    }
    for content in &req.contents {
        let role = if content.role == "model" {
            "assistant"
        } else {
            "user"
        };
        let message_content = gemini_content(&content.parts)?;
        let mut calls = Vec::new();
        for call in content
            .parts
            .iter()
            .filter_map(|part| part.get("functionCall"))
        {
            let Some(name) = call.get("name").and_then(serde_json::Value::as_str) else {
                continue;
            };
            let id = call
                .get("id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| new_call_id(name));
            pending_calls
                .entry(name.to_string())
                .or_default()
                .push_back(id.clone());
            calls.push(OAIToolCall {
                id,
                type_: "function",
                function: OAIToolCallFunction {
                    name: name.to_string(),
                    arguments: call
                        .get("args")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!({}))
                        .to_string(),
                },
            });
        }
        let message = (message_content != serde_json::Value::String(String::new())
            || !calls.is_empty())
        .then_some(OAIMessage {
            role: role.to_string(),
            content: message_content,
            tool_calls: (!calls.is_empty()).then_some(calls),
            name: None,
            tool_call_id: None,
        });
        let mut responses = Vec::new();
        for response in content
            .parts
            .iter()
            .filter_map(|part| part.get("functionResponse"))
        {
            let Some(name) = response.get("name").and_then(serde_json::Value::as_str) else {
                continue;
            };
            let id = if let Some(id) = response.get("id").and_then(serde_json::Value::as_str) {
                if let Some(calls) = pending_calls.get_mut(name) {
                    if let Some(index) = calls.iter().position(|call| call == id) {
                        calls.remove(index);
                    }
                }
                id.to_string()
            } else {
                pending_calls
                    .get_mut(name)
                    .and_then(std::collections::VecDeque::pop_front)
                    .unwrap_or_else(|| new_call_id(name))
            };
            responses.push(OAIMessage {
                role: "tool".into(),
                content: serde_json::Value::String(
                    response
                        .get("response")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null)
                        .to_string(),
                ),
                tool_calls: None,
                name: Some(name.to_string()),
                tool_call_id: Some(id),
            });
        }
        if role == "assistant" {
            messages.extend(message);
            messages.extend(responses);
        } else {
            messages.extend(responses);
            messages.extend(message);
        }
    }
    Ok(messages)
}

/// Normalize protobuf Schema nodes without changing example/default data.
fn normalize_gemini_schema(schema: &mut serde_json::Value) {
    if let Some(serde_json::Value::String(kind)) = schema.get_mut("type") {
        match kind.as_str() {
            "STRING" | "NUMBER" | "INTEGER" | "BOOLEAN" | "ARRAY" | "OBJECT" | "NULL" => {
                *kind = kind.to_ascii_lowercase();
            }
            _ => {}
        }
    }
    if let Some(properties) = schema
        .get_mut("properties")
        .and_then(serde_json::Value::as_object_mut)
    {
        for property in properties.values_mut() {
            normalize_gemini_schema(property);
        }
    }
    if let Some(items) = schema.get_mut("items") {
        normalize_gemini_schema(items);
    }
    if let Some(variants) = schema
        .get_mut("anyOf")
        .and_then(serde_json::Value::as_array_mut)
    {
        for variant in variants {
            normalize_gemini_schema(variant);
        }
    }
}

fn gemini_tools(
    tools: &Option<serde_json::Value>,
    config: &Option<serde_json::Value>,
) -> Option<serde_json::Value> {
    let allowed_names = config
        .as_ref()
        .and_then(|config| config.get("functionCallingConfig"))
        .filter(|config| config.get("mode").and_then(serde_json::Value::as_str) == Some("ANY"))
        .and_then(|config| config.get("allowedFunctionNames"))
        .and_then(serde_json::Value::as_array)
        .filter(|names| !names.is_empty())
        .map(|names| {
            names
                .iter()
                .filter_map(serde_json::Value::as_str)
                .collect::<std::collections::BTreeSet<_>>()
        });
    let tools = tools
        .as_ref()?
        .as_array()?
        .iter()
        .flat_map(|tool| {
            tool.get("functionDeclarations")
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter_map(|declaration| {
            let name = declaration.get("name")?.as_str()?;
            if allowed_names
                .as_ref()
                .is_some_and(|names| !names.contains(name))
            {
                return None;
            }
            let parameters = declaration
                .get("parametersJsonSchema")
                .cloned()
                .or_else(|| {
                    declaration.get("parameters").cloned().map(|mut schema| {
                        normalize_gemini_schema(&mut schema);
                        schema
                    })
                })
                .unwrap_or_else(|| serde_json::json!({"type": "object"}));
            let mut function = serde_json::json!({"name": name, "parameters": parameters});
            if let Some(description) = declaration.get("description") {
                function["description"] = description.clone();
            }
            Some(serde_json::json!({"type": "function", "function": function}))
        })
        .collect::<Vec<_>>();
    (!tools.is_empty()).then_some(serde_json::Value::Array(tools))
}

fn gemini_tool_choice(
    config: &Option<serde_json::Value>,
) -> Result<Option<serde_json::Value>, String> {
    let Some(config) = config
        .as_ref()
        .and_then(|config| config.get("functionCallingConfig"))
    else {
        return Ok(None);
    };
    let mode = config
        .get("mode")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("AUTO");
    let names = config
        .get("allowedFunctionNames")
        .and_then(serde_json::Value::as_array)
        .map(|names| {
            names
                .iter()
                .filter_map(serde_json::Value::as_str)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    match (mode, names.as_slice()) {
        ("AUTO", []) => Ok(None),
        ("NONE", []) => Ok(Some(serde_json::json!("none"))),
        ("ANY", _) => Ok(Some(serde_json::json!("required"))),
        _ => Err("unsupported Gemini functionCallingConfig".to_string()),
    }
}

/// Maps backend completion reasons to Gemini's finish-reason vocabulary.
fn gemini_finish_reason(reason: Option<&str>) -> &'static str {
    match reason {
        Some("length") | Some("max_tokens") => "MAX_TOKENS",
        Some("content_filter") => "SAFETY",
        _ => "STOP",
    }
}

/// Rewrites one OpenAI SSE payload into a Gemini SSE payload.
fn gemini_sse_line(
    payload: &str,
    calls: &std::cell::RefCell<std::collections::BTreeMap<usize, ToolCallAccumulator>>,
    finished: &std::cell::Cell<bool>,
) -> String {
    if let Ok(event) = serde_json::from_str::<serde_json::Value>(payload) {
        if event.get("error").is_some_and(|error| !error.is_null()) {
            return format!("data: {event}\n\n");
        }
    }
    accumulate_tool_call_deltas(payload, calls);
    let usage = gemini_usage_metadata(payload);
    let Some((content, thinking, done)) = oai_chunk_to_content(payload) else {
        return usage
            .map(|usage| format!("data: {}\n\n", serde_json::json!({"usageMetadata": usage})))
            .unwrap_or_default();
    };
    let finish_reason = serde_json::from_str::<OAIChunk>(payload)
        .ok()
        .and_then(|chunk| chunk.choices.into_iter().next())
        .and_then(|choice| choice.finish_reason)
        .map(|reason| gemini_finish_reason(Some(&reason)))
        .unwrap_or("STOP");

    if done.is_some() && !finished.replace(true) {
        let tool_parts = calls
            .borrow()
            .iter()
            .map(|(index, call)| {
                let id = if !call.id.is_empty() {
                    call.id.clone()
                } else {
                    format!("call_{}_{}", call.name, index)
                };
                serde_json::json!({
                    "functionCall": {
                        "id": id,
                        "name": call.name,
                        "args": serde_json::from_str::<serde_json::Value>(&call.arguments)
                            .unwrap_or_else(|_| serde_json::json!({}))
                    }
                })
            })
            .collect::<Vec<_>>();
        let mut parts = Vec::new();
        if let Some(thinking) = thinking.filter(|thinking| !thinking.is_empty()) {
            parts.push(serde_json::json!({"text": thinking, "thought": true}));
        }
        if !content.is_empty() {
            parts.push(serde_json::json!({"text": content}));
        }
        parts.extend(tool_parts);
        let mut chunk = serde_json::json!({
            "candidates": [{
                "content": {"role": "model", "parts": parts},
                "finishReason": finish_reason,
            }]
        });
        if let Some(usage) = usage {
            chunk["usageMetadata"] = usage;
        }
        format!("data: {chunk}\n\n")
    } else if !content.is_empty()
        || thinking
            .as_deref()
            .is_some_and(|thinking| !thinking.is_empty())
    {
        let mut parts = Vec::new();
        if let Some(thinking) = thinking.filter(|thinking| !thinking.is_empty()) {
            parts.push(serde_json::json!({"text": thinking, "thought": true}));
        }
        if !content.is_empty() {
            parts.push(serde_json::json!({"text": content}));
        }
        let chunk = serde_json::json!({
            "candidates": [{
                "content": {"role": "model", "parts": parts}
            }]
        });
        format!("data: {chunk}\n\n")
    } else {
        String::new()
    }
}

/// Extracts OpenAI token usage into Gemini's usage metadata field names.
fn gemini_usage_metadata(payload: &str) -> Option<serde_json::Value> {
    let usage = serde_json::from_str::<OAIChunk>(payload).ok()?.usage?;
    Some(serde_json::json!({
        "promptTokenCount": usage.prompt_tokens,
        "candidatesTokenCount": usage.completion_tokens,
        "totalTokenCount": if usage.total_tokens == 0 {
            usage.prompt_tokens + usage.completion_tokens
        } else {
            usage.total_tokens
        },
    }))
}

/// Extracts a Gemini model name from a streaming method path.
pub(super) fn gemini_stream_model(path: &str) -> Option<&str> {
    path.strip_suffix(":streamGenerateContent")
        .filter(|model| !model.is_empty())
}

// -- Gemini /gemini/<model>/v1beta/models/... ------------------------------

/// AGY appends its own model name to `GOOGLE_GEMINI_BASE_URL`, including
/// separate hard-coded models for title/planning work. The first path segment
/// instead carries the model selected by `llmman launch agy`, encoded so OCI
/// references containing `/` remain one safe URL segment. Every AGY request in
/// the session is intentionally pinned to that model.
pub(super) async fn handle_pinned_gemini(
    State(state): State<AppState>,
    UrlPath((encoded_model, gemini_path)): UrlPath<(String, String)>,
    headers: HeaderMap,
    Json(req): Json<GeminiRequest>,
) -> Result<Response, AppError> {
    let model_bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(encoded_model)
        .map_err(|_| AppError::status(StatusCode::BAD_REQUEST, "invalid AGY model route"))?;
    let selected_model = String::from_utf8(model_bytes)
        .map_err(|_| AppError::status(StatusCode::BAD_REQUEST, "invalid AGY model route"))?;

    if gemini_stream_model(&gemini_path).is_some() {
        return handle_gemini_request(state, &selected_model, headers, req).await;
    }
    Err(AppError::status(
        StatusCode::NOT_FOUND,
        "AGY endpoint supports streamGenerateContent only",
    ))
}

async fn handle_gemini_request(
    state: AppState,
    selected_model: &str,
    headers: HeaderMap,
    req: GeminiRequest,
) -> Result<Response, AppError> {
    send_with_hybrid_fallback(
        &state,
        selected_model,
        Some(&headers),
        None,
        |model, target, guard| gemini_request_to(&state, &req, model, target, guard),
    )
    .await
}

async fn gemini_request_to(
    state: &AppState,
    req: &GeminiRequest,
    canonical_model: String,
    target: Target,
    guard: ActivityGuard,
) -> Result<Response, AppError> {
    let activity = begin_activity(guard, None).await;
    let wire_model = backend_wire_model(state, &target, &canonical_model).await;
    let mut oai = gemini_oai_request(wire_model, req)?;
    let resp = post_chat(&state.0.client, &target, &mut oai).await?;

    let calls = std::cell::RefCell::new(std::collections::BTreeMap::new());
    let finished = std::cell::Cell::new(false);
    let stream = bytes_to_lines(resp).map(move |line| {
        let _activity = &activity;
        let Some(payload) = line.strip_prefix("data: ") else {
            return Ok::<_, std::convert::Infallible>(Bytes::new());
        };
        Ok(Bytes::from(gemini_sse_line(payload, &calls, &finished)))
    });

    Ok(Response::builder()
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from_stream(stream))
        .unwrap())
}

/// Builds the backend chat request for one Gemini request.
fn gemini_oai_request(model: String, req: &GeminiRequest) -> Result<OAIChatRequest, AppError> {
    let generation = req.generation_config.as_ref();
    let seed = generation
        .and_then(|config| config.get("seed"))
        .map(|value| {
            let signed = value
                .as_i64()
                .and_then(|value| i32::try_from(value).ok())
                .ok_or_else(|| {
                    AppError::status(
                        StatusCode::BAD_REQUEST,
                        "Gemini generationConfig.seed must be a signed 32-bit integer",
                    )
                })?;
            u64::try_from(signed).map_err(|_| {
                AppError::status(
                    StatusCode::BAD_REQUEST,
                    "negative Gemini generationConfig.seed is unsupported by this backend adapter",
                )
            })
        })
        .transpose()?;
    Ok(OAIChatRequest {
        model,
        messages: gemini_messages(req)
            .map_err(|error| AppError::status(StatusCode::BAD_REQUEST, error))?,
        stream: true,
        temperature: generation
            .and_then(|config| config.get("temperature"))
            .and_then(serde_json::Value::as_f64)
            .map(|value| value as f32),
        top_p: generation
            .and_then(|config| config.get("topP"))
            .and_then(serde_json::Value::as_f64)
            .map(|value| value as f32),
        max_tokens: generation
            .and_then(|config| config.get("maxOutputTokens"))
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u32::try_from(value).ok()),
        repeat_penalty: None,
        chat_template_kwargs: None,
        tools: gemini_tools(&req.tools, &req.tool_config),
        tool_choice: gemini_tool_choice(&req.tool_config)
            .map_err(|error| AppError::status(StatusCode::BAD_REQUEST, error))?,
        response_format: None,
        seed,
        stop: generation
            .and_then(|config| config.get("stopSequences"))
            .and_then(serde_json::Value::as_array)
            .map(|sequences| {
                sequences
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .map(str::to_string)
                    .collect()
            }),
        presence_penalty: generation
            .and_then(|config| config.get("presencePenalty"))
            .and_then(serde_json::Value::as_f64)
            .map(|value| value as f32),
        frequency_penalty: generation
            .and_then(|config| config.get("frequencyPenalty"))
            .and_then(serde_json::Value::as_f64)
            .map(|value| value as f32),
        top_k: generation
            .and_then(|config| config.get("topK"))
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u32::try_from(value).ok()),
        min_p: None,
        stream_options: Some(serde_json::json!({"include_usage": true})),
    })
}

#[cfg(test)]
mod tests {
    use super::super::tests::{headers_with, remote_target, test_state, HOSTED, PAIR};
    use super::super::{with_hybrid_fallback, ContextOverflow};
    use super::*;
    use axum::{response::IntoResponse, routing::post, Router};
    use std::sync::Arc;
    #[test]
    fn gemini_terminal_stream_event_preserves_content_and_finish_reason() {
        let calls = std::cell::RefCell::new(std::collections::BTreeMap::new());
        let finished = std::cell::Cell::new(false);
        let event = gemini_sse_line(
            r#"{"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}"#,
            &calls,
            &finished,
        );

        let payload = event
            .strip_prefix("data: ")
            .and_then(|event| event.strip_suffix("\n\n"))
            .expect("Gemini SSE event");
        let response: serde_json::Value = serde_json::from_str(payload).expect("valid JSON event");
        assert_eq!(
            response["candidates"][0]["content"]["parts"][0]["text"],
            "done"
        );
        assert_eq!(response["candidates"][0]["finishReason"], "STOP");
        assert_eq!(
            response["usageMetadata"],
            serde_json::json!({
                "promptTokenCount": 3,
                "candidatesTokenCount": 2,
                "totalTokenCount": 5,
            })
        );
    }

    #[test]
    fn gemini_inline_image_becomes_an_openai_image_part() {
        let content = gemini_content(&[serde_json::json!({
            "inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}
        })])
        .expect("supported inline image");

        assert_eq!(
            content,
            serde_json::json!([{
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,aGVsbG8="}
            }])
        );
    }

    #[test]
    fn gemini_generation_controls_are_forwarded() {
        for (seed, presence, frequency, top_k) in [
            (42, -0.5, 0.75, 40),
            (0, 0.0, 0.0, 0),
            (i32::MAX as u64, 1.0, -1.0, u32::MAX),
        ] {
            let req: GeminiRequest = serde_json::from_value(serde_json::json!({
                "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                "generationConfig": {
                    "seed": seed,
                    "presencePenalty": presence,
                    "frequencyPenalty": frequency,
                    "topK": top_k
                }
            }))
            .unwrap();
            let request = gemini_oai_request("selected-model".into(), &req).unwrap();
            let body = serde_json::to_value(request).unwrap();
            assert_eq!(body["seed"], seed);
            assert_eq!(body["presence_penalty"], presence);
            assert_eq!(body["frequency_penalty"], frequency);
            assert_eq!(body["top_k"], top_k);
        }
    }

    #[test]
    fn gemini_generation_controls_without_representable_values_remain_unset() {
        for generation in [
            serde_json::Value::Null,
            serde_json::json!({}),
            serde_json::json!({"presencePenalty": "0.5", "frequencyPenalty": false, "topK": u64::from(u32::MAX) + 1}),
            serde_json::json!({"topK": 1.5}),
        ] {
            let req: GeminiRequest = serde_json::from_value(serde_json::json!({
                "generationConfig": generation
            }))
            .unwrap();
            let request = gemini_oai_request("selected-model".into(), &req).unwrap();
            assert_eq!(request.seed, None);
            assert_eq!(request.presence_penalty, None);
            assert_eq!(request.frequency_penalty, None);
            assert_eq!(request.top_k, None);
        }
    }

    #[test]
    fn gemini_tool_config_constrains_backend_tools_to_all_allowed_names() {
        let config = Some(serde_json::json!({
            "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["inspect", "search"]}
        }));
        let tools = Some(serde_json::json!([{
            "functionDeclarations": [
                {"name": "inspect", "parameters": {"type": "object"}},
                {"name": "search", "parameters": {"type": "object"}},
                {"name": "delete", "parameters": {"type": "object"}}
            ]
        }]));

        assert_eq!(
            gemini_tool_choice(&config).expect("supported tool config"),
            Some(serde_json::json!("required"))
        );
        assert_eq!(
            gemini_tools(&tools, &config),
            Some(serde_json::json!([
                {"type": "function", "function": {"name": "inspect", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}
            ]))
        );
    }

    #[test]
    fn gemini_empty_or_omitted_allow_list_keeps_all_tools() {
        let tools = Some(serde_json::json!([{
            "functionDeclarations": [{"name": "inspect"}, {"name": "search"}]
        }]));
        let expected = Some(serde_json::json!([
            {"type": "function", "function": {"name": "inspect", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}
        ]));
        for config in [
            serde_json::json!({"functionCallingConfig": {"mode": "ANY"}}),
            serde_json::json!({"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": []}}),
        ] {
            let config = Some(config);
            assert_eq!(gemini_tools(&tools, &config), expected);
            assert_eq!(
                gemini_tool_choice(&config).unwrap(),
                Some(serde_json::json!("required"))
            );
        }
    }

    #[test]
    fn gemini_function_responses_without_ids_match_prior_calls_across_content_blocks() {
        let req = GeminiRequest {
            contents: vec![
                GeminiContent {
                    role: "model".into(),
                    parts: vec![
                        serde_json::json!({"functionCall": {"name": "lookup", "args": {"id": 1}}}),
                        serde_json::json!({"functionCall": {"name": "lookup", "args": {"id": 2}}}),
                    ],
                },
                GeminiContent {
                    role: "user".into(),
                    parts: vec![
                        serde_json::json!({"functionResponse": {"name": "lookup", "response": {"id": 1}}}),
                    ],
                },
                GeminiContent {
                    role: "user".into(),
                    parts: vec![
                        serde_json::json!({"functionResponse": {"name": "lookup", "response": {"id": 2}}}),
                    ],
                },
            ],
            system_instruction: None,
            generation_config: None,
            tools: None,
            tool_config: None,
        };

        let messages = gemini_messages(&req).expect("valid Gemini tool conversation");
        let calls = messages[0]
            .tool_calls
            .as_ref()
            .expect("assistant tool calls");
        assert_eq!(
            messages[1].tool_call_id.as_deref(),
            Some(calls[0].id.as_str())
        );
        assert_eq!(
            messages[2].tool_call_id.as_deref(),
            Some(calls[1].id.as_str())
        );
    }

    #[test]
    fn gemini_explicit_function_response_id_is_not_reused_as_a_fallback() {
        let req = GeminiRequest {
            contents: vec![
                GeminiContent {
                    role: "model".into(),
                    parts: vec![
                        serde_json::json!({"functionCall": {"id": "first", "name": "lookup"}}),
                        serde_json::json!({"functionCall": {"id": "second", "name": "lookup"}}),
                    ],
                },
                GeminiContent {
                    role: "user".into(),
                    parts: vec![
                        serde_json::json!({"functionResponse": {"id": "first", "name": "lookup"}}),
                    ],
                },
                GeminiContent {
                    role: "user".into(),
                    parts: vec![serde_json::json!({"functionResponse": {"name": "lookup"}})],
                },
            ],
            system_instruction: None,
            generation_config: None,
            tools: None,
            tool_config: None,
        };

        let messages = gemini_messages(&req).expect("valid Gemini tool conversation");
        assert_eq!(messages[1].tool_call_id.as_deref(), Some("first"));
        assert_eq!(messages[2].tool_call_id.as_deref(), Some("second"));
    }

    #[test]
    fn gemini_generated_ids_do_not_collide_with_supplied_ids() {
        for explicit_first in [true, false] {
            let mut parts = vec![
                serde_json::json!({"functionCall": {"id": "call_lookup_0", "name": "lookup"}}),
                serde_json::json!({"functionCall": {"name": "lookup"}}),
            ];
            if !explicit_first {
                parts.reverse();
            }
            let req: GeminiRequest = serde_json::from_value(serde_json::json!({
                "contents": [
                    {"role": "model", "parts": parts},
                    {"role": "user", "parts": [
                        {"functionResponse": {"name": "lookup", "response": {}}},
                        {"functionResponse": {"name": "lookup", "response": {}}}
                    ]},
                    {"role": "model", "parts": [
                        {"functionCall": {"id": "call_lookup_1", "name": "lookup"}},
                        {"functionCall": {"name": "lookup"}}
                    ]},
                    {"role": "user", "parts": [
                        {"functionResponse": {"name": "lookup", "response": {}}},
                        {"functionResponse": {"name": "lookup", "response": {}}}
                    ]}
                ]
            }))
            .unwrap();
            let messages = gemini_messages(&req).unwrap();
            let ids: Vec<_> = messages
                .iter()
                .filter_map(|message| message.tool_calls.as_ref())
                .flatten()
                .map(|call| call.id.as_str())
                .collect();
            assert_eq!(
                ids.iter().collect::<std::collections::BTreeSet<_>>().len(),
                4
            );
            assert_eq!(ids[usize::from(!explicit_first)], "call_lookup_0");
            assert_eq!(ids[2], "call_lookup_1");
            let response_ids: Vec<_> = messages
                .iter()
                .filter_map(|message| message.tool_call_id.as_deref())
                .collect();
            assert_eq!(response_ids, ids);
        }
    }

    #[tokio::test]
    async fn gemini_context_overflow_retries_hosted_unless_pinned_local() {
        let seen = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let captured = seen.clone();
        let app = Router::new().route(
        "/v1/chat/completions",
        post(move |Json(body): Json<serde_json::Value>| {
            let captured = captured.clone();
            async move {
                captured.lock().await.push(body.clone());
                if body["model"] == "gemma4" {
                    return (
                        StatusCode::BAD_REQUEST,
                        r#"{"error":{"message":"request exceeds the available context size","type":"exceed_context_size_error"}}"#,
                    ).into_response();
                }
                (
                    [("content-type", "text/event-stream")],
                    "data: {\"choices\":[{\"delta\":{\"content\":\"hosted answer\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
                ).into_response()
            }
        }),
    );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let state = test_state();
        let req: GeminiRequest = serde_json::from_value(serde_json::json!({
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "generationConfig": {"temperature": 0.25}
        }))
        .unwrap();

        for pinned in [false, true] {
            seen.lock().await.clear();
            let headers = if pinned {
                headers_with(&[("x-llmman-route", "local")])
            } else {
                HeaderMap::new()
            };
            let resolve = |model: String| {
                let state = &state;
                async move {
                    let model = crate::hybrid::local_half(&model).to_string();
                    let target = if model == HOSTED {
                        remote_target(&format!("http://127.0.0.1:{port}/v1"))
                    } else {
                        Target::Local(port)
                    };
                    let guard = ActivityGuard::new(state, &model);
                    Ok((model, target, guard))
                }
            };
            let result =
                with_hybrid_fallback(PAIR, Some(&headers), resolve, |model, target, guard| {
                    gemini_request_to(&state, &req, model, target, guard)
                })
                .await;
            if pinned {
                assert!(result
                    .unwrap_err()
                    .0
                    .downcast_ref::<ContextOverflow>()
                    .is_some());
                assert_eq!(seen.lock().await.len(), 1);
            } else {
                let response = result.unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap();
                assert!(std::str::from_utf8(&body)
                    .unwrap()
                    .contains("hosted answer"));
                let requests = seen.lock().await;
                assert_eq!(requests.len(), 2);
                assert_eq!(requests[0]["model"], "gemma4");
                assert_eq!(requests[1]["model"], "mock-model");
                assert_eq!(requests[0]["messages"], requests[1]["messages"]);
                assert_eq!(requests[1]["temperature"], 0.25);
            }
        }
    }

    #[test]
    fn gemini_invalid_or_unsupported_seeds_return_bad_request() {
        for seed in [
            serde_json::json!(-1),
            serde_json::json!(i32::MIN),
            serde_json::json!(i64::from(i32::MIN) - 1),
            serde_json::json!(i64::from(i32::MAX) + 1),
            serde_json::json!(u64::MAX),
            serde_json::json!(1.5),
            serde_json::json!("42"),
            serde_json::json!(true),
            serde_json::Value::Null,
        ] {
            let req: GeminiRequest = serde_json::from_value(serde_json::json!({
                "generationConfig": {"seed": seed}
            }))
            .unwrap();
            let error = gemini_oai_request("model".into(), &req).unwrap_err();
            assert_eq!(error.1, StatusCode::BAD_REQUEST);
            assert!(error.0.to_string().contains("seed"));
        }
    }

    #[test]
    fn gemini_stream_relays_provider_errors_after_partial_content() {
        let calls = std::cell::RefCell::new(std::collections::BTreeMap::new());
        let finished = std::cell::Cell::new(false);
        assert!(!gemini_sse_line(
            r#"{"choices":[{"delta":{"content":"partial"}}]}"#,
            &calls,
            &finished
        )
        .is_empty());
        for event in [
            serde_json::json!({"error": {"message": "upstream failed", "type": "api_error"}}),
            serde_json::json!({"error": "upstream failed"}),
        ] {
            let output = gemini_sse_line(&event.to_string(), &calls, &finished);
            let actual: serde_json::Value =
                serde_json::from_str(output.strip_prefix("data: ").unwrap().trim()).unwrap();
            assert_eq!(actual, event);
        }
    }

    #[test]
    fn gemini_tool_schema_normalizes_nested_types_without_changing_data() {
        let schema = serde_json::json!({
            "type": "OBJECT", "required": ["type"],
            "properties": {
                "type": {"type": "STRING", "enum": ["OBJECT", "STRING"]},
                "values": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
                    "count": {"type": "INTEGER"}, "score": {"type": "NUMBER"},
                    "enabled": {"type": "BOOLEAN"}}}},
                "optional": {"anyOf": [{"type": "STRING"}, {"type": "NULL"}]}
            },
            "default": {"type": "OBJECT"}, "example": {"type": "STRING"}
        });
        let tools = Some(serde_json::json!([{"functionDeclarations": [
            {"name": "proto", "parameters": schema},
            {"name": "json", "parametersJsonSchema": schema}
        ]}]));
        let result = gemini_tools(&tools, &None).unwrap();
        let normalized = &result[0]["function"]["parameters"];
        assert_eq!(normalized["type"], "object");
        assert_eq!(normalized["properties"]["type"]["type"], "string");
        assert_eq!(
            normalized["properties"]["type"]["enum"],
            schema["properties"]["type"]["enum"]
        );
        assert_eq!(normalized["properties"]["values"]["type"], "array");
        let items = &normalized["properties"]["values"]["items"];
        assert_eq!(items["type"], "object");
        assert_eq!(items["properties"]["count"]["type"], "integer");
        assert_eq!(items["properties"]["score"]["type"], "number");
        assert_eq!(items["properties"]["enabled"]["type"], "boolean");
        assert_eq!(
            normalized["properties"]["optional"]["anyOf"][1]["type"],
            "null"
        );
        assert_eq!(normalized["default"], schema["default"]);
        assert_eq!(normalized["example"], schema["example"]);
        assert_eq!(normalized["required"], schema["required"]);
        assert_eq!(result[1]["function"]["parameters"], schema);
    }

    #[test]
    fn replayed_thought_parts_do_not_become_assistant_content() {
        let req: GeminiRequest = serde_json::from_value(serde_json::json!({
            "contents": [
                {"role": "model", "parts": [{"text": "private reasoning", "thought": true}]},
                {"role": "model", "parts": [
                    {"text": "more reasoning", "thought": true},
                    {"text": "visible answer", "thought": false},
                    {"functionCall": {"name": "lookup", "args": {}}}
                ]}
            ]
        }))
        .unwrap();
        let messages = gemini_messages(&req).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "visible answer");
        assert_eq!(
            messages[0].tool_calls.as_ref().unwrap()[0].function.name,
            "lookup"
        );
    }
}
