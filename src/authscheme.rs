//! How a credential travels to a hosted provider, apart from which one it
//! is: the header or query parameter it rides in, the extra static headers
//! and query parameters a provider wants, and the `${VAR}` placeholders of
//! a per-account base URL.
//!
//! The default — [`Scheme::Default`] — is what `llmman serve` has always
//! done: `Authorization: Bearer <key>` on the OpenAI wire, `x-api-key` on
//! the Anthropic one. Everything here is opt-in through `llmman.conf`
//! (`auth`, `auth_header`, `auth_query`, `headers`, `query`, `vars`; see
//! `docs/providers.md`) or a built-in provider's own needs (Azure's
//! `api-key`, Copilot's editor headers).

use std::collections::BTreeMap;

use reqwest::header::{HeaderName, HeaderValue, AUTHORIZATION};

use crate::credentials::{Resolved, Style};
use crate::providers::Wire;

/// Where the credential goes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Scheme {
    /// The wire's own: `Authorization: Bearer` on [`Wire::OpenAi`],
    /// `x-api-key` on [`Wire::Anthropic`].
    #[default]
    Default,
    /// `Authorization: Bearer <key>`.
    Bearer,
    /// `x-api-key: <key>`.
    XApiKey,
    /// `<name>: <key>` — Azure's `api-key`, Google's `x-goog-api-key`.
    Header(String),
    /// `?<name>=<key>`.
    Query(String),
}

impl Scheme {
    /// Reads the `auth` / `auth_header` / `auth_query` fields of a
    /// `[providers.<id>]` table. Naming only `auth_header` or `auth_query`
    /// is enough: there is nothing else it could mean.
    pub fn from_fields(
        auth: Option<&str>,
        header: Option<&str>,
        query: Option<&str>,
    ) -> Result<Self, String> {
        let auth = auth.map(|a| a.trim().to_ascii_lowercase());
        let header = header.map(str::trim).filter(|h| !h.is_empty());
        let query = query.map(str::trim).filter(|q| !q.is_empty());
        if header.is_some() && query.is_some() {
            return Err("auth_header and auth_query are alternatives; set one".into());
        }
        let named = |header: Option<&str>, query: Option<&str>| match (header, query) {
            (Some(h), _) => header_name(h).map(|_| Scheme::Header(h.to_string())),
            (_, Some(q)) => query_name(q).map(|_| Scheme::Query(q.to_string())),
            _ => Ok(Scheme::Default),
        };
        match auth.as_deref() {
            None | Some("") => named(header, query),
            Some(plain @ ("bearer" | "x-api-key")) => {
                if header.is_some() || query.is_some() {
                    return Err(format!(
                        "auth_header and auth_query only go with auth = \"header\" or \"query\", not {plain:?}"
                    ));
                }
                Ok(if plain == "bearer" {
                    Self::Bearer
                } else {
                    Self::XApiKey
                })
            }
            Some("header") if header.is_some() => named(header, None),
            Some("header") => Err("auth = \"header\" needs auth_header = \"<name>\"".into()),
            Some("query") if query.is_some() => named(None, query),
            Some("query") => Err("auth = \"query\" needs auth_query = \"<name>\"".into()),
            Some(other) => Err(format!(
                "auth: {other:?} is not one of bearer, x-api-key, header, query"
            )),
        }
    }

    /// The name `llmman providers` and the daemon's API report.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Bearer => "bearer",
            Self::XApiKey => "x-api-key",
            Self::Header(_) => "header",
            Self::Query(_) => "query",
        }
    }

    /// [`Scheme::Default`] resolved against `wire`.
    fn effective(&self, wire: Wire) -> Self {
        match (self, wire) {
            (Self::Default, Wire::OpenAi) => Self::Bearer,
            (Self::Default, Wire::Anthropic) => Self::XApiKey,
            (other, _) => other.clone(),
        }
    }
}

/// Everything about authenticating to one provider except the credential
/// itself.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct Auth {
    pub scheme: Scheme,
    /// Sent on every request. Values may be secret (a gateway token), so
    /// they are marked sensitive and never printed.
    pub headers: Vec<(String, String)>,
    /// Query parameters on every request (`api-version` for Azure); never
    /// printed.
    pub query: Vec<(String, String)>,
    /// Values for the `${VAR}` of a catalog provider's base URL, from
    /// `[providers.<id>] vars`. The environment is asked first.
    pub vars: BTreeMap<String, String>,
}

impl std::fmt::Debug for Auth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Auth")
            .field("scheme", &self.scheme)
            .field(
                "headers",
                &self.headers.iter().map(|(k, _)| k).collect::<Vec<_>>(),
            )
            .field(
                "query",
                &self.query.iter().map(|(k, _)| k).collect::<Vec<_>>(),
            )
            .field("vars", &self.vars.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl Auth {
    /// The scheme alone: the static headers and query values belong to
    /// the daemon's operator and go only where its own key may.
    pub fn scheme_only(&self) -> Auth {
        Auth {
            scheme: self.scheme.clone(),
            ..Auth::default()
        }
    }

    /// `self` with what `over` sets laid on top: its scheme when it names
    /// one, its headers and query parameters by name, its variables by
    /// name. `llmman.conf` over a built-in provider's own.
    pub fn overlaid(&self, over: &Auth) -> Auth {
        let mut out = self.clone();
        if over.scheme != Scheme::Default {
            out.scheme = over.scheme.clone();
        }
        for (k, v) in &over.headers {
            out.headers
                .retain(|(name, _)| !name.eq_ignore_ascii_case(k));
            out.headers.push((k.clone(), v.clone()));
        }
        for (k, v) in &over.query {
            out.query.retain(|(name, _)| name != k);
            out.query.push((k.clone(), v.clone()));
        }
        out.vars.extend(over.vars.clone());
        out
    }

    /// Puts this provider's static headers and query parameters, and the
    /// credential `cred` the way its scheme says, on `req`. With no
    /// credential, only the static parts go: a keyless provider.
    ///
    /// Every value is marked sensitive, so no `Debug` of the request
    /// prints one.
    pub fn apply(
        &self,
        wire: Wire,
        mut req: reqwest::RequestBuilder,
        cred: Option<&Resolved>,
    ) -> reqwest::RequestBuilder {
        for (name, value) in &self.headers {
            req = with_header(req, name, value);
        }
        if !self.query.is_empty() {
            req = req.query(&self.query);
        }
        let Some(cred) = cred else { return req };
        for (name, value) in &cred.headers {
            req = with_header(req, name, value);
        }
        // A signing credential has no header of its own: the signature
        // goes on once the body is final.
        if cred.signer.is_some() {
            return req;
        }
        let secret = cred.secret.expose();
        let scheme = match cred.style {
            Style::Bearer => Scheme::Bearer,
            Style::Key => self.scheme.effective(wire),
        };
        match scheme {
            Scheme::Bearer | Scheme::Default => {
                with_header(req, AUTHORIZATION.as_str(), &format!("Bearer {secret}"))
            }
            Scheme::XApiKey => with_header(req, "x-api-key", secret),
            Scheme::Header(name) => with_header(req, &name, secret),
            Scheme::Query(name) => req.query(&[(name, secret)]),
        }
    }
}

/// Sends `req`, signing it first when `signer` is given. The signature
/// covers the request as built — its URL, headers and exact body bytes —
/// so it is made after everything else is set, on the request that goes out.
/// Without a signer this is `req.send()`.
pub async fn send(
    req: reqwest::RequestBuilder,
    signer: Option<&dyn crate::credentials::RequestSigner>,
) -> anyhow::Result<reqwest::Response> {
    // reqwest quotes the URL in its errors, and a credential may be in it.
    let Some(signer) = signer else {
        return Ok(req.send().await.map_err(reqwest::Error::without_url)?);
    };
    let (client, request) = req.build_split();
    let mut request = request.map_err(reqwest::Error::without_url)?;
    signer.sign(&mut request)?;
    Ok(client
        .execute(request)
        .await
        .map_err(reqwest::Error::without_url)?)
}

/// `req` with `name: value`, the value marked sensitive. A value that is
/// not a legal header value is left for reqwest to report when the
/// request is built, as it always has for a key with a stray newline.
fn with_header(req: reqwest::RequestBuilder, name: &str, value: &str) -> reqwest::RequestBuilder {
    match HeaderValue::from_str(value) {
        Ok(mut value) => {
            value.set_sensitive(true);
            req.header(name, value)
        }
        Err(_) => req.header(name, value),
    }
}

/// Header names a provider table cannot set: the ones that frame the
/// request rather than describe it.
const FRAMING_HEADERS: &[&str] = &[
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "upgrade",
    "te",
    "trailer",
];

/// `name` as a header name, or what is wrong with it.
pub fn header_name(name: &str) -> Result<HeaderName, String> {
    let parsed = HeaderName::from_bytes(name.trim().as_bytes())
        .map_err(|_| format!("{name:?} is not a usable header name"))?;
    if FRAMING_HEADERS.contains(&parsed.as_str()) {
        return Err(format!("{name:?} frames the request and cannot be set"));
    }
    Ok(parsed)
}

/// `value` as a header value for `name`, or what is wrong with it. The
/// value is not echoed: it may be a secret.
pub fn header_value(name: &str, value: &str) -> Result<HeaderValue, String> {
    HeaderValue::from_str(value.trim())
        .map_err(|_| format!("the value of header {name:?} has a character a header cannot hold"))
}

/// `name` as a query parameter name.
fn query_name(name: &str) -> Result<(), String> {
    if !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
    {
        Ok(())
    } else {
        Err(format!("{name:?} is not a usable query parameter name"))
    }
}

/// A `query = { ... }` entry's name, validated as `auth_query` is.
pub fn query_param(name: &str) -> Result<(), String> {
    query_name(name).map_err(|e| format!("query: {e}"))
}

// ---------------------------------------------------------------------------
// ${VAR} in a base URL
// ---------------------------------------------------------------------------

/// The `${NAME}` placeholders of `template`, in order.
pub fn template_vars(template: &str) -> Vec<String> {
    placeholders(template)
        .into_iter()
        .map(|p| p.name.to_string())
        .collect()
}

struct Placeholder<'a> {
    name: &'a str,
    start: usize,
    end: usize,
}

fn placeholders(template: &str) -> Vec<Placeholder<'_>> {
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(open) = template[at..].find("${") {
        let start = at + open;
        let Some(close) = template[start..].find('}') else {
            break;
        };
        let end = start + close + 1;
        let name = &template[start + 2..end - 1];
        if !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
            out.push(Placeholder { name, start, end });
        }
        at = end;
    }
    out
}

/// Where in a URL a placeholder sits, which decides what it may hold.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Position {
    /// The whole URL, as models.dev's Neon entry has it.
    Whole,
    /// In the authority: a host, or a label of one. `entire` when it is
    /// all of the authority, which is the only place a port may be.
    Host { entire: bool },
    /// In the path: one segment.
    Path,
}

/// `template` with each `${NAME}` filled from `lookup`, or the first
/// reason it cannot be. A variable's value is checked against where it
/// lands — a value that would change which host the request goes to, or
/// add a query, is refused rather than interpolated — and never echoed,
/// so the error is safe to log.
///
/// A variable standing for a host (Databricks) may be given with its
/// `https://` and a trailing `/`: both are dropped, as opencode's
/// `plugin/provider/databricks.ts` does.
pub fn expand(template: &str, lookup: &dyn Fn(&str) -> Option<String>) -> Result<String, String> {
    let scheme_end = template.find("://").map(|i| i + 3);
    let authority_end =
        scheme_end.map(|s| template[s..].find('/').map_or(template.len(), |j| s + j));
    let mut out = String::new();
    let mut at = 0;
    for p in placeholders(template) {
        out.push_str(&template[at..p.start]);
        at = p.end;
        let value = lookup(p.name)
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .ok_or_else(|| format!("{} is not set", p.name))?;
        let position = match (scheme_end, authority_end) {
            _ if p.start == 0 => Position::Whole,
            (Some(s), Some(e)) if p.start >= s && p.start < e => Position::Host {
                entire: p.start == s && p.end == e,
            },
            _ => Position::Path,
        };
        out.push_str(&fit(p.name, &value, position)?);
    }
    out.push_str(&template[at..]);
    Ok(out)
}

fn fit(name: &str, value: &str, position: Position) -> Result<String, String> {
    let refused = || {
        Err(format!(
            "{name} holds something this part of the URL cannot take"
        ))
    };
    match position {
        Position::Whole => {
            let Ok(url) = reqwest::Url::parse(value) else {
                return refused();
            };
            if url.scheme() != "https"
                || url.host_str().is_none()
                || !url.username().is_empty()
                || url.password().is_some()
                || url.query().is_some()
                || url.fragment().is_some()
            {
                return refused();
            }
            Ok(value.trim_end_matches('/').to_string())
        }
        Position::Host { entire } => {
            // Only a variable that is the whole host may arrive dressed as a
            // URL; a label in one may not bring a slash.
            let host = if entire {
                value
                    .strip_prefix("https://")
                    .unwrap_or(value)
                    .trim_end_matches('/')
            } else {
                value
            };
            let (host_part, port) = match host.rsplit_once(':') {
                Some((h, p)) if entire => (h, Some(p)),
                _ => (host, None),
            };
            let host_ok = !host_part.is_empty()
                && !host_part.starts_with(['.', '-'])
                && host_part
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-');
            let port_ok = port.is_none_or(|p| {
                !p.is_empty() && p.len() <= 5 && p.bytes().all(|b| b.is_ascii_digit())
            });
            if host_ok && port_ok {
                Ok(host.to_string())
            } else {
                refused()
            }
        }
        Position::Path => {
            let ok = value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._~-".contains(&b))
                && value.bytes().any(|b| b.is_ascii_alphanumeric());
            if ok {
                Ok(value.to_string())
            } else {
                refused()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Redirects
// ---------------------------------------------------------------------------

/// Follows a redirect only to the same scheme, host and port. reqwest
/// strips `Authorization` when a redirect changes host, but not `x-api-key`,
/// `api-key` or a key in the query — and a provider's credential must not
/// follow a `Location` to somewhere else whichever way it travels.
pub fn same_origin_redirects() -> reqwest::redirect::Policy {
    reqwest::redirect::Policy::custom(|attempt| {
        let same = attempt.previous().last().is_some_and(|from| {
            let to = attempt.url();
            from.scheme() == to.scheme()
                && from.host_str() == to.host_str()
                && from.port_or_known_default() == to.port_or_known_default()
        });
        if same && attempt.previous().len() < 10 {
            attempt.follow()
        } else {
            attempt.stop()
        }
    })
}

/// Percent-encodes `text` (RFC 3986's unreserved set stays).
pub(crate) fn url_encode(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for b in text.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credentials::Secret;

    fn lookup<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            pairs
                .iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| v.to_string())
        }
    }

    fn resolved(secret: &str, style: Style) -> Resolved {
        Resolved {
            secret: Secret::new(secret.to_string()),
            style,
            kind: "api_key",
            flow: None,
            metadata: BTreeMap::new(),
            headers: Vec::new(),
            signer: None,
        }
    }

    /// Builds the request a scheme produces, for inspection.
    fn built(auth: &Auth, wire: Wire, cred: Option<&Resolved>) -> reqwest::Request {
        let client = reqwest::Client::new();
        auth.apply(wire, client.post("https://example.test/v1/chat"), cred)
            .build()
            .expect("a request")
    }

    #[test]
    fn the_default_scheme_is_each_wires_own() {
        let auth = Auth::default();
        let cred = resolved("sk-1", Style::Key);

        let openai = built(&auth, Wire::OpenAi, Some(&cred));
        assert_eq!(openai.headers()["authorization"], "Bearer sk-1");
        assert!(openai.headers()["authorization"].is_sensitive());
        assert!(openai.headers().get("x-api-key").is_none());
        assert_eq!(openai.url().query(), None);

        let anthropic = built(&auth, Wire::Anthropic, Some(&cred));
        assert_eq!(anthropic.headers()["x-api-key"], "sk-1");
        assert!(anthropic.headers().get("authorization").is_none());

        // A keyless provider sends no credential at all.
        let bare = built(&auth, Wire::OpenAi, None);
        assert!(bare.headers().is_empty());
    }

    #[test]
    fn scheme_only_drops_static_headers_and_query_but_keeps_the_placement() {
        let auth = Auth {
            scheme: Scheme::XApiKey,
            headers: vec![("X-Gateway".into(), "t".into())],
            query: vec![("api-version".into(), "v1".into())],
            vars: BTreeMap::new(),
        };
        let req = built(
            &auth.scheme_only(),
            Wire::OpenAi,
            Some(&resolved("sk-3", Style::Key)),
        );
        assert!(req.headers().get("x-gateway").is_none());
        assert_eq!(req.url().query(), None);
        assert_eq!(req.headers()["x-api-key"], "sk-3");
    }

    #[test]
    fn each_scheme_puts_the_credential_where_it_says() {
        let cred = resolved("sk-2", Style::Key);
        let with = |scheme| Auth {
            scheme,
            ..Auth::default()
        };

        let r = built(&with(Scheme::XApiKey), Wire::OpenAi, Some(&cred));
        assert_eq!(r.headers()["x-api-key"], "sk-2");
        assert!(r.headers().get("authorization").is_none());

        let r = built(&with(Scheme::Bearer), Wire::Anthropic, Some(&cred));
        assert_eq!(r.headers()["authorization"], "Bearer sk-2");

        let r = built(
            &with(Scheme::Header("api-key".into())),
            Wire::OpenAi,
            Some(&cred),
        );
        assert_eq!(r.headers()["api-key"], "sk-2");
        assert!(r.headers()["api-key"].is_sensitive());
        assert!(r.headers().get("authorization").is_none());

        let r = built(
            &with(Scheme::Query("key".into())),
            Wire::OpenAi,
            Some(&cred),
        );
        assert_eq!(r.url().query(), Some("key=sk-2"));
        assert!(r.headers().get("authorization").is_none());
    }

    #[test]
    fn static_headers_and_query_ride_along_with_or_without_a_credential() {
        let auth = Auth {
            scheme: Scheme::Header("api-key".into()),
            headers: vec![("HTTP-Referer".into(), "https://llmman.test".into())],
            query: vec![("api-version".into(), "v1".into())],
            ..Auth::default()
        };
        let r = built(&auth, Wire::OpenAi, Some(&resolved("k", Style::Key)));
        assert_eq!(r.headers()["http-referer"], "https://llmman.test");
        assert!(r.headers()["http-referer"].is_sensitive());
        assert_eq!(r.url().query(), Some("api-version=v1"));
        assert_eq!(r.headers()["api-key"], "k");

        let r = built(&auth, Wire::OpenAi, None);
        assert_eq!(r.headers()["http-referer"], "https://llmman.test");
        assert!(r.headers().get("api-key").is_none());
    }

    /// A login's token is always a bearer, whatever the provider's own
    /// scheme says: it is not the provider's API key.
    #[test]
    fn a_bearer_style_credential_overrides_the_scheme() {
        let auth = Auth {
            scheme: Scheme::Header("api-key".into()),
            ..Auth::default()
        };
        let mut cred = resolved("tok", Style::Bearer);
        cred.headers = vec![("ChatGPT-Account-Id".into(), "acct-1".into())];
        let r = built(&auth, Wire::OpenAi, Some(&cred));
        assert_eq!(r.headers()["authorization"], "Bearer tok");
        assert!(r.headers().get("api-key").is_none());
        assert_eq!(r.headers()["chatgpt-account-id"], "acct-1");
    }

    #[test]
    fn scheme_fields_parse_and_reject_contradictions() {
        let s = Scheme::from_fields;
        assert_eq!(s(None, None, None), Ok(Scheme::Default));
        assert_eq!(s(Some("Bearer"), None, None), Ok(Scheme::Bearer));
        assert_eq!(s(Some("x-api-key"), None, None), Ok(Scheme::XApiKey));
        assert_eq!(
            s(Some("header"), Some("api-key"), None),
            Ok(Scheme::Header("api-key".into()))
        );
        assert_eq!(
            s(None, Some("x-goog-api-key"), None),
            Ok(Scheme::Header("x-goog-api-key".into()))
        );
        assert_eq!(
            s(Some("query"), None, Some("key")),
            Ok(Scheme::Query("key".into()))
        );
        assert!(s(Some("header"), None, None).is_err());
        assert!(s(Some("query"), None, None).is_err());
        assert!(s(Some("bearer"), Some("x"), None).is_err());
        assert!(s(Some("basic"), None, None).is_err());
        assert!(s(None, Some("a"), Some("b")).is_err());
        assert!(s(Some("header"), Some("bad name"), None).is_err());
        assert!(s(Some("header"), Some("Host"), None).is_err());
        assert!(s(Some("query"), None, Some("a&b")).is_err());
    }

    #[test]
    fn overlay_replaces_by_name_and_keeps_the_rest() {
        let base = Auth {
            scheme: Scheme::Header("api-key".into()),
            headers: vec![("X-A".into(), "1".into()), ("X-B".into(), "2".into())],
            query: vec![("api-version".into(), "v1".into())],
            vars: BTreeMap::from([("A".into(), "a".into())]),
        };
        let over = Auth {
            scheme: Scheme::Default,
            headers: vec![("x-a".into(), "9".into())],
            query: vec![("api-version".into(), "2024".into())],
            vars: BTreeMap::from([("B".into(), "b".into())]),
        };
        let merged = base.overlaid(&over);
        assert_eq!(merged.scheme, Scheme::Header("api-key".into()));
        assert_eq!(
            merged.headers,
            vec![("X-B".into(), "2".into()), ("x-a".into(), "9".into())]
        );
        assert_eq!(merged.query, vec![("api-version".into(), "2024".into())]);
        assert_eq!(merged.vars.len(), 2);
        let named = Auth {
            scheme: Scheme::Bearer,
            ..Auth::default()
        };
        assert_eq!(base.overlaid(&named).scheme, Scheme::Bearer);
    }

    #[test]
    fn debug_never_prints_a_header_value() {
        let auth = Auth {
            headers: vec![("X-Token".into(), "super-secret".into())],
            ..Auth::default()
        };
        let printed = format!("{auth:?}");
        assert!(printed.contains("X-Token"));
        assert!(!printed.contains("super-secret"), "{printed}");
    }

    #[test]
    fn templates_fill_from_the_lookup_by_position() {
        let vars = [
            ("ACCOUNT", "acme-1"),
            ("ID", "0af3"),
            ("HOST", "https://adb-1.azuredatabricks.net/"),
            ("BASE", "https://gw.example.com/x/"),
        ];
        let get = lookup(&vars);
        assert_eq!(
            expand(
                "https://${ACCOUNT}.snowflakecomputing.com/api/v2/cortex/v1",
                &get
            ),
            Ok("https://acme-1.snowflakecomputing.com/api/v2/cortex/v1".into())
        );
        assert_eq!(
            expand(
                "https://api.cloudflare.com/client/v4/accounts/${ID}/ai/v1",
                &get
            ),
            Ok("https://api.cloudflare.com/client/v4/accounts/0af3/ai/v1".into())
        );
        // A host variable may carry its scheme and trailing slash.
        assert_eq!(
            expand("https://${HOST}/ai-gateway/mlflow/v1", &get),
            Ok("https://adb-1.azuredatabricks.net/ai-gateway/mlflow/v1".into())
        );
        assert_eq!(
            expand("${BASE}/v1", &get),
            Ok("https://gw.example.com/x/v1".into())
        );
        assert_eq!(
            expand("https://plain.example/v1", &get),
            Ok("https://plain.example/v1".into())
        );
        assert_eq!(template_vars("https://${A}.x/${B}/y"), vec!["A", "B"]);
    }

    #[test]
    fn templates_refuse_what_would_move_the_request() {
        let bad = |value: &str, template: &str| {
            let vars = [("V", value)];
            let found = expand(template, &lookup(&vars));
            found
        };
        // Unset or blank.
        assert!(expand("https://${V}.x.com/v1", &|_| None).is_err());
        assert!(bad("  ", "https://${V}.x.com/v1").is_err());
        // Breaking out of a host label.
        assert!(bad("evil.com/", "https://${V}.x.com/v1").is_err());
        assert!(bad("a@evil.com", "https://${V}.x.com/v1").is_err());
        assert!(bad("a#", "https://${V}.x.com/v1").is_err());
        assert!(bad("a:80", "https://${V}.x.com/v1").is_err());
        assert!(bad("-a", "https://${V}.x.com/v1").is_err());
        // A path segment is one segment.
        assert!(bad("a/b", "https://x.com/${V}/v1").is_err());
        assert!(bad("..", "https://x.com/${V}/v1").is_err());
        assert!(bad("a?b=1", "https://x.com/${V}/v1").is_err());
        assert!(bad("a b", "https://x.com/${V}/v1").is_err());
        // A whole-URL variable must be a plain https URL.
        assert!(bad("http://gw.example.com", "${V}/v1").is_err());
        assert!(bad("https://u:p@gw.example.com", "${V}/v1").is_err());
        assert!(bad("https://gw.example.com/?x=1", "${V}/v1").is_err());
        assert!(bad("gw.example.com", "${V}/v1").is_err());
        // A host variable that is the whole authority may carry a port.
        assert_eq!(
            bad("gw.example.com:8443", "https://${V}/v1"),
            Ok("https://gw.example.com:8443/v1".into())
        );
        // The message names the variable, not its value.
        let err = bad("evil.com/", "https://${V}.x.com/v1").unwrap_err();
        assert!(err.contains('V') && !err.contains("evil"), "{err}");
    }

    #[tokio::test]
    async fn a_transport_error_omits_the_query_credential() {
        let req = reqwest::Client::new()
            .get("http://127.0.0.1:1/v1/models")
            .query(&[("key", "sk-q&secret")]);
        let err = send(req, None).await.unwrap_err();
        let message = format!("{err:#}");
        assert!(
            !message.contains("sk-q") && !message.contains("127.0.0.1:1/"),
            "{message}"
        );
    }
}
