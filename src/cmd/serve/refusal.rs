//! Routes a target has no equivalent of, answered with a 501 and a
//! sentence saying why.
//!
//! Two shapes, one for each side of the request. [`unsupported_on_wire`]
//! and [`wire_refusal`] refuse up front, from what the provider's
//! [`Wire`] is known to carry; [`explain_missing_route`] rewrites a 404
//! that already came back, because OpenAI-compatible does not mean every
//! OpenAI route exists.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use super::{responses, Target, CHAT_COMPLETIONS_ROUTE};
use crate::providers::Wire;

/// Routes the Messages API has no equivalent of (legacy completions,
/// embeddings, Responses token counting), refused with a 501 instead of
/// a round trip that could only 404.
pub(super) fn unsupported_on_wire(target: &Target, route: &str) -> Option<Response> {
    let message = wire_refusal(target, route)?;
    let body = serde_json::json!({
        "error": { "message": message, "type": "invalid_request_error" }
    });
    Some((StatusCode::NOT_IMPLEMENTED, Json(body)).into_response())
}

/// The message behind [`unsupported_on_wire`], for the Ollama embedding
/// routes.
pub(super) fn wire_refusal(target: &Target, route: &str) -> Option<String> {
    let Target::Remote(remote) = target else {
        return None;
    };
    if remote.wire != Wire::Anthropic
        || matches!(route, CHAT_COMPLETIONS_ROUTE | responses::RESPONSES_ROUTE)
    {
        return None;
    }
    Some(format!(
        "provider {} speaks the Anthropic Messages API, which has no equivalent of {route}; \
         only chat completions, /v1/responses and /v1/messages reach it",
        remote.provider
    ))
}

/// Explains a provider's bare 404 on the Responses API.
///
/// Being OpenAI-wire-format does not mean implementing every OpenAI
/// route: `openai`, `groq` and `openrouter` answer `/v1/responses*`,
/// `mistral` 404s. Generation is bridged by
/// [`responses::remote_responses`], so this only fires for
/// `/v1/responses/input_tokens`, which has no chat-completions
/// equivalent. It reports the 404 actually received rather than
/// predicting one from a list that would go stale.
pub(super) fn explain_missing_route(target: &Target, route: &str, resp: Response) -> Response {
    if resp.status() != StatusCode::NOT_FOUND || !responses::is_responses_route(route) {
        return resp;
    }
    // A 404 on any other route means something else entirely — an
    // unknown model on `/v1/chat/completions`, most often — and claiming
    // a missing Responses API for it would be a worse answer than the
    // provider's own.
    let Target::Remote(remote) = target else {
        return resp;
    };
    let body = serde_json::json!({
        "error": {
            "message": format!(
                "provider {} has no {route} — it is OpenAI-compatible but does not \
                 implement the Responses API's token counting. Generation on \
                 /v1/responses is bridged; this route cannot be.",
                remote.provider
            ),
            "type": "invalid_request_error",
        }
    });
    (StatusCode::NOT_IMPLEMENTED, Json(body)).into_response()
}
