//! Hybrid pairs: which half of a `local+cloud` reference serves a
//! request, and the one retry when the local half turns it down.
//!
//! The half is chosen from an estimate of the request's size (see
//! [`crate::hybrid`] for the reference syntax and the routing rule
//! itself), so the choice can be wrong. A local backend's own context
//! refusal is exact and arrives before any output, which is what makes
//! [`with_hybrid_fallback`]'s second attempt on the hosted half
//! possible at all. What the rest of the daemon sees of a resolved pair
//! is [`resolve_hybrid_side`]'s own doc comment.

use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use futures::StreamExt;

use super::sched::ActivityGuard;
use super::{context_overflow_message, ensure_model, AppError, AppState, ContextOverflow, Target};

/// Picks which half of a hybrid pair serves this request and returns
/// that half's own ordinary reference (see [`crate::hybrid`]).
/// Substitution rather than a third [`Target`] variant, so the rest of
/// [`ensure_model`] and every proxy past it serve a pair unchanged. Does
/// no I/O.
pub(super) fn resolve_hybrid_side(
    state: &AppState,
    pair: &crate::hybrid::Pair<'_>,
    headers: Option<&HeaderMap>,
) -> Result<String, AppError> {
    let pin = request_pin(headers)?;
    // The declared length is all that is knowable before the body is
    // parsed. A chunked request declares none and stays local.
    let request_bytes = headers
        .and_then(|h| h.get(reqwest::header::CONTENT_LENGTH))
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok());
    let decision = crate::hybrid::route(pin, request_bytes, state.0.hybrid_local_bytes);

    let why = match decision.reason {
        crate::hybrid::Reason::Pinned => format!("pinned by {}", crate::hybrid::ROUTE_HEADER),
        crate::hybrid::Reason::Overflow { bytes, budget } => format!(
            "{} request exceeds the {} this host serves locally",
            crate::fmt::human_size(bytes),
            crate::fmt::human_size(budget)
        ),
        crate::hybrid::Reason::LocalFirst => "no reason to leave this machine".to_string(),
    };
    // The sides differ in cost and in where the data goes, so every
    // request says which way it went. `{:?}`, as the request logs do:
    // both names come straight from the request.
    eprintln!(
        "[llmman] hybrid {:?} + {:?} -> {} ({why})",
        pair.local,
        pair.remote_ref(),
        decision.side.as_str()
    );
    Ok(pair.side_ref(decision.side))
}

/// The side a request pinned itself to, if any; a 400 when unreadable.
/// Raw bytes reach [`crate::hybrid::parse_pin`] so a non-UTF-8 value is
/// rejected rather than read as absent.
pub(super) fn request_pin(
    headers: Option<&HeaderMap>,
) -> Result<Option<crate::hybrid::Side>, AppError> {
    let mut values = headers
        .map(|h| h.get_all(crate::hybrid::ROUTE_HEADER).iter())
        .into_iter()
        .flatten();
    let value = values.next();
    // Two values is not a pin, whichever came first.
    if values.next().is_some() {
        return Err(AppError::status(
            StatusCode::BAD_REQUEST,
            format!("{} given more than once", crate::hybrid::ROUTE_HEADER),
        ));
    }
    crate::hybrid::parse_pin(value.map(|v| v.as_bytes()))
        .map_err(|e| AppError(e, StatusCode::BAD_REQUEST))
}

/// The hosted half a hybrid pair falls back to when its local half
/// refuses a request as too large: `None` for anything but a pair, and
/// for a pair pinned local, whose pin is never overridden.
fn hybrid_fallback(
    model_ref: &str,
    headers: Option<&HeaderMap>,
) -> Result<Option<String>, AppError> {
    let Some(pair) = crate::hybrid::split_ref(model_ref) else {
        return Ok(None);
    };
    Ok((request_pin(headers)? != Some(crate::hybrid::Side::Local)).then(|| pair.remote_ref()))
}

/// Serves a generating request through `send` against the target
/// [`ensure_model`] picks, retrying once on a hybrid pair's hosted half
/// when the local half refuses the request as over its context. The
/// byte budget is an estimate; the refusal is exact and arrives before
/// any output. Without the retry an agent sees the context error,
/// compacts its history and stays local.
///
/// The refusal is `post_chat`'s [`ContextOverflow`] or, for a raw
/// relay, the backend's own 400, read only when a fallback exists so a
/// plain local model's error passes through untouched.
pub(super) async fn send_with_hybrid_fallback<F, Fut>(
    state: &AppState,
    model_ref: &str,
    headers: Option<&HeaderMap>,
    request_threads: Option<u32>,
    send: F,
) -> Result<Response, AppError>
where
    F: Fn(String, Target, ActivityGuard) -> Fut,
    Fut: std::future::Future<Output = Result<Response, AppError>>,
{
    let resolve =
        |m: String| async move { ensure_model(state, &m, headers, request_threads).await };
    with_hybrid_fallback(model_ref, headers, resolve, send).await
}

/// [`send_with_hybrid_fallback`] with `ensure_model` abstracted, so the
/// retry itself is testable.
pub(super) async fn with_hybrid_fallback<R, RFut, F, Fut>(
    model_ref: &str,
    headers: Option<&HeaderMap>,
    resolve: R,
    send: F,
) -> Result<Response, AppError>
where
    R: Fn(String) -> RFut,
    RFut: std::future::Future<Output = Result<(String, Target, ActivityGuard), AppError>>,
    F: Fn(String, Target, ActivityGuard) -> Fut,
    Fut: std::future::Future<Output = Result<Response, AppError>>,
{
    let (model, target, guard) = resolve(model_ref.to_string()).await?;
    let fallback = match target {
        Target::Local(_) => hybrid_fallback(model_ref, headers)?,
        _ => None,
    };
    let Some(cloud) = fallback else {
        return send(model, target, guard).await;
    };
    let refusal = match send(model, target, guard).await {
        Ok(resp) => match local_context_overflow(resp).await {
            Ok(resp) => return Ok(resp),
            Err(refusal) => refusal,
        },
        Err(err) => match err.0.downcast_ref::<ContextOverflow>() {
            Some(overflow) => overflow.refusal.clone(),
            None => return Err(err),
        },
    };
    eprintln!("[llmman] hybrid {model_ref:?} -> cloud ({refusal})");
    let (model, target, guard) = resolve(cloud).await?;
    send(model, target, guard).await
}

/// Largest 400 body [`local_context_overflow`] reads to classify it.
/// llama-server's is one short JSON object.
const OVERFLOW_BODY_LIMIT: usize = 64 * 1024;

/// Splits a relayed response into the backend's context refusal (`Err`,
/// with its message) or anything else (`Ok`, the response intact). Only
/// a 400 is read, up to [`OVERFLOW_BODY_LIMIT`]; whatever was read is
/// put back in front of the rest when it is some other error.
async fn local_context_overflow(resp: Response) -> Result<Response, String> {
    if resp.status() != StatusCode::BAD_REQUEST {
        return Ok(resp);
    }
    let (parts, body) = resp.into_parts();
    let mut rest = body.into_data_stream();
    let mut head = Vec::new();
    while head.len() <= OVERFLOW_BODY_LIMIT {
        match rest.next().await {
            Some(Ok(chunk)) => head.extend_from_slice(&chunk),
            // A read error is the client's to see, as it would have been.
            Some(Err(_)) | None => break,
        }
    }
    if head.len() <= OVERFLOW_BODY_LIMIT {
        if let Some(refusal) =
            context_overflow_message(parts.status, &String::from_utf8_lossy(&head))
        {
            return Err(refusal);
        }
    }
    let head = futures::stream::once(futures::future::ready(Ok(Bytes::from(head))));
    Ok(Response::from_parts(
        parts,
        Body::from_stream(head.chain(rest)),
    ))
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{headers_with, HOSTED, PAIR};
    use super::*;

    /// Only a pair falls back, and never one the caller pinned local:
    /// that pin is the promise the data stays on this machine.
    #[test]
    fn only_an_unpinned_pair_has_a_hosted_half_to_fall_back_to() {
        assert_eq!(
            hybrid_fallback(PAIR, Some(&headers_with(&[]))).unwrap(),
            Some(HOSTED.to_string())
        );
        assert_eq!(
            hybrid_fallback(PAIR, None).unwrap(),
            Some(HOSTED.to_string())
        );
        assert_eq!(
            hybrid_fallback(PAIR, Some(&headers_with(&[("x-llmman-route", "local")]))).unwrap(),
            None
        );
        assert_eq!(hybrid_fallback("gemma4", None).unwrap(), None);
        assert_eq!(hybrid_fallback(HOSTED, None).unwrap(), None);
    }

    /// A relayed 400 is inspected and either taken as the refusal or
    /// handed back intact; nothing else is touched.
    #[tokio::test]
    async fn a_relayed_response_is_only_intercepted_when_it_is_the_refusal() {
        let llama = r#"{"error":{"code":400,"message":"request (9 tokens) exceeds the available context size (8 tokens), try increasing it","type":"exceed_context_size_error"}}"#;
        // No Content-Length: proxy_rewriting_model strips it.
        let resp = |status: StatusCode, body: &'static str| {
            Response::builder()
                .status(status)
                .body(Body::from(body))
                .unwrap()
        };
        let refusal = local_context_overflow(resp(StatusCode::BAD_REQUEST, llama))
            .await
            .expect_err("the refusal must be intercepted");
        assert!(
            refusal.contains("exceeds the available context size"),
            "{refusal}"
        );

        let other = r#"{"error":{"code":400,"message":"invalid grammar"}}"#;
        let passed = local_context_overflow(resp(StatusCode::BAD_REQUEST, other))
            .await
            .expect("another 400 passes through");
        assert_eq!(passed.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(passed.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body, other.as_bytes(), "body reattached intact");

        let ok = local_context_overflow(resp(StatusCode::OK, "data: {}"))
            .await
            .expect("a success is never read");
        assert_eq!(ok.status(), StatusCode::OK);

        // Past the read limit: not classified, and nothing lost.
        let big: &'static str = String::from_utf8(vec![b'x'; OVERFLOW_BODY_LIMIT + 10])
            .unwrap()
            .leak();
        let passed = local_context_overflow(resp(StatusCode::BAD_REQUEST, big))
            .await
            .expect("an oversized 400 passes through");
        let body = axum::body::to_bytes(passed.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body.len(), big.len());
    }
}
