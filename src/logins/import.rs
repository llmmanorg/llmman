//! Reusing a login another CLI holds: Claude Code's
//! `~/.claude/.credentials.json` and Codex's `~/.codex/auth.json`.
//!
//! Read-only by default: their refresh tokens rotate, so a copy would race
//! the original and log one of them out. An
//! [`crate::credentials::ExternalSource::CliLogin`] is read on every request
//! and never written back; `--copy` moves the login into llmman's store.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{anyhow, Context};

use crate::credentials::CliLogin;

/// What one of those files holds.
#[derive(Clone)]
pub struct CliTokens {
    /// The CLI, as the user runs it.
    pub cli_name: &'static str,
    /// The llmman provider the login belongs to.
    pub provider: &'static str,
    /// The [`crate::logins`] flow it is spent through.
    pub flow: &'static str,
    pub access: String,
    pub refresh: Option<String>,
    pub expires_ms: Option<u64>,
    pub metadata: BTreeMap<String, String>,
}

impl std::fmt::Debug for CliTokens {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CliTokens")
            .field("cli_name", &self.cli_name)
            .field("expires_ms", &self.expires_ms)
            .finish_non_exhaustive()
    }
}

/// Where `cli` keeps its login: `$CLAUDE_CONFIG_DIR/.credentials.json`
/// (OpenClaw `plugin-sdk/provider-auth-claude-compat.ts`), `$CODEX_HOME/
/// auth.json` (OpenClaw `agents/cli-credentials.ts`), each defaulting to
/// the dot-directory under the home directory.
pub fn default_path(cli: CliLogin) -> Option<PathBuf> {
    default_path_with(cli, |var| std::env::var_os(var), crate::config::home_dir())
}

fn default_path_with(
    cli: CliLogin,
    env: impl Fn(&str) -> Option<std::ffi::OsString>,
    home: Option<PathBuf>,
) -> Option<PathBuf> {
    let dir = |var: &str, dot: &str| {
        env(var)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .or_else(|| home.map(|h| h.join(dot)))
    };
    match cli {
        CliLogin::Claude => {
            dir("CLAUDE_CONFIG_DIR", ".claude").map(|d| d.join(".credentials.json"))
        }
        CliLogin::Codex => dir("CODEX_HOME", ".codex").map(|d| d.join("auth.json")),
    }
}

/// Reads `cli`'s login from `path`, or from where it keeps it.
pub fn read(cli: CliLogin, path: Option<&str>) -> anyhow::Result<CliTokens> {
    let path = match path {
        Some(p) => crate::credentials::expand_home(p),
        None => {
            default_path(cli).ok_or_else(|| anyhow!("no home directory to find the login in"))?
        }
    };
    let text = std::fs::read_to_string(&path).with_context(|| {
        format!(
            "read {} — is that CLI logged in on this computer? (macOS keeps Claude Code's \
             login in the Keychain, which llmman does not read)",
            path.display()
        )
    })?;
    match cli {
        CliLogin::Claude => parse_claude(&text),
        CliLogin::Codex => parse_codex(&text),
    }
}

/// Claude Code's file: `{"claudeAiOauth": {"accessToken", "refreshToken"?,
/// "expiresAt"}}`, `expiresAt` in milliseconds. A login without a refresh
/// token is the `claude setup-token` variety. OpenClaw
/// `plugin-sdk/provider-auth-claude-compat.ts` `parseClaudeCliOauthCredential`.
pub fn parse_claude(text: &str) -> anyhow::Result<CliTokens> {
    let json: serde_json::Value = serde_json::from_str(text)
        .map_err(|_| anyhow!("Claude Code's credentials file is not JSON"))?;
    let oauth = json
        .get("claudeAiOauth")
        .ok_or_else(|| anyhow!("Claude Code's credentials file has no claudeAiOauth login"))?;
    let access = non_empty(oauth.get("accessToken"))
        .ok_or_else(|| anyhow!("Claude Code's login has no access token"))?;
    let expires_ms = oauth
        .get("expiresAt")
        .and_then(serde_json::Value::as_f64)
        .filter(|e| e.is_finite() && *e > 0.0)
        .map(|e| e as u64);
    Ok(CliTokens {
        cli_name: "claude",
        provider: "anthropic",
        flow: super::FLOW_CLAUDE,
        access,
        refresh: non_empty(oauth.get("refreshToken")),
        expires_ms,
        metadata: BTreeMap::new(),
    })
}

/// Codex's file: `{"auth_mode", "OPENAI_API_KEY"?, "tokens": {"access_token",
/// "refresh_token", "id_token", "account_id"}}`. Only a ChatGPT login is
/// reused here; an API-key one is an `openai` API key, which is
/// `llmman login --provider openai --method key`. OpenClaw
/// `agents/cli-credentials.ts` `parseCodexOauthCredential`.
pub fn parse_codex(text: &str) -> anyhow::Result<CliTokens> {
    let json: serde_json::Value =
        serde_json::from_str(text).map_err(|_| anyhow!("Codex's auth file is not JSON"))?;
    let mode = json
        .get("auth_mode")
        .and_then(serde_json::Value::as_str)
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| {
            if json.get("OPENAI_API_KEY").is_some_and(|k| k.is_string()) {
                "api_key".into()
            } else {
                "chatgpt".into()
            }
        });
    if !matches!(mode.as_str(), "chatgpt" | "chatgptauthtokens") {
        anyhow::bail!(
            "Codex is logged in with an API key, not a ChatGPT account; \
             use `llmman login --provider openai --method key` for that"
        );
    }
    let tokens = json
        .get("tokens")
        .ok_or_else(|| anyhow!("Codex's auth file has no tokens"))?;
    let access = non_empty(tokens.get("access_token"))
        .ok_or_else(|| anyhow!("Codex's login has no access token"))?;
    let mut metadata = BTreeMap::new();
    let account = non_empty(tokens.get("account_id"))
        .or_else(|| non_empty(tokens.get("id_token")).and_then(|t| super::chatgpt_account_id(&t)))
        .or_else(|| super::chatgpt_account_id(&access));
    if let Some(account) = account {
        metadata.insert("account_id".to_string(), account);
    }
    Ok(CliTokens {
        cli_name: "codex",
        provider: "openai-chatgpt",
        flow: super::FLOW_CHATGPT,
        expires_ms: super::jwt_expiry_ms(&access),
        refresh: non_empty(tokens.get("refresh_token")),
        access,
        metadata,
    })
}

fn non_empty(value: Option<&serde_json::Value>) -> Option<String> {
    value
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A JWT whose payload is `claims`; the signature is not checked here
    /// or anywhere in llmman.
    pub(crate) fn jwt(claims: &serde_json::Value) -> String {
        use base64::Engine as _;
        let b64 = |v: &[u8]| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(v);
        format!(
            "{}.{}.sig",
            b64(b"{\"alg\":\"none\"}"),
            b64(claims.to_string().as_bytes())
        )
    }

    #[test]
    fn a_claude_login_with_a_refresh_token_is_oauth_and_without_is_a_token() {
        let both = r#"{"claudeAiOauth":{"accessToken":"sk-ant-oat01-a","refreshToken":"r","expiresAt":1893456000000,"subscriptionType":"max"}}"#;
        let t = parse_claude(both).unwrap();
        assert_eq!(t.access, "sk-ant-oat01-a");
        assert_eq!(t.refresh.as_deref(), Some("r"));
        assert_eq!(t.expires_ms, Some(1_893_456_000_000));
        assert_eq!(
            (t.provider, t.flow),
            ("anthropic", super::super::FLOW_CLAUDE)
        );

        let token_only = r#"{"claudeAiOauth":{"accessToken":"sk-ant-oat01-b","expiresAt":5}}"#;
        assert_eq!(parse_claude(token_only).unwrap().refresh, None);

        for bad in [
            "not json",
            "{}",
            r#"{"claudeAiOauth":{"refreshToken":"r"}}"#,
            r#"{"claudeAiOauth":{"accessToken":"  "}}"#,
        ] {
            let err = parse_claude(bad).unwrap_err().to_string();
            assert!(
                !err.contains("not json") || err.contains("not JSON"),
                "{err}"
            );
        }
        assert!(format!("{:?}", parse_claude(both).unwrap())
            .find("sk-ant")
            .is_none());
    }

    #[test]
    fn a_codex_chatgpt_login_yields_tokens_and_the_account_id() {
        let access = jwt(&serde_json::json!({
            "exp": 1_893_456_000u64,
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct-from-jwt"}
        }));
        let file = serde_json::json!({
            "auth_mode": "chatgpt",
            "tokens": {"access_token": access, "refresh_token": "rt", "account_id": "acct-1"}
        })
        .to_string();
        let t = parse_codex(&file).unwrap();
        assert_eq!(t.refresh.as_deref(), Some("rt"));
        assert_eq!(t.expires_ms, Some(1_893_456_000_000));
        assert_eq!(t.metadata["account_id"], "acct-1");
        assert_eq!(t.provider, "openai-chatgpt");

        // No explicit account id: the claim in the token.
        let file = serde_json::json!({"tokens": {"access_token": access, "refresh_token": "rt"}})
            .to_string();
        assert_eq!(
            parse_codex(&file).unwrap().metadata["account_id"],
            "acct-from-jwt"
        );
    }

    #[test]
    fn a_codex_api_key_login_is_not_a_chatgpt_login() {
        let file = r#"{"OPENAI_API_KEY":"sk-x","tokens":null}"#;
        let err = parse_codex(file).unwrap_err().to_string();
        assert!(err.contains("--provider openai --method key"), "{err}");
        assert!(!err.contains("sk-x"));
        assert!(parse_codex(r#"{"auth_mode":"chatgpt"}"#).is_err());
        assert!(parse_codex("nope").is_err());
    }

    #[test]
    fn default_paths_follow_the_cli_own_overrides() {
        let home = Some(PathBuf::from("/home/u"));
        let set = |var: &'static str, value: &'static str| {
            move |name: &str| (name == var).then(|| std::ffi::OsString::from(value))
        };
        assert_eq!(
            default_path_with(
                CliLogin::Claude,
                set("CLAUDE_CONFIG_DIR", "/c"),
                home.clone()
            ),
            Some(PathBuf::from("/c/.credentials.json"))
        );
        assert_eq!(
            default_path_with(CliLogin::Codex, set("CODEX_HOME", "/x"), home.clone()),
            Some(PathBuf::from("/x/auth.json"))
        );
        assert_eq!(
            default_path_with(CliLogin::Claude, |_| None, home.clone()),
            Some(PathBuf::from("/home/u/.claude/.credentials.json"))
        );
        assert_eq!(
            default_path_with(CliLogin::Codex, set("CODEX_HOME", ""), home),
            Some(PathBuf::from("/home/u/.codex/auth.json"))
        );
        assert_eq!(default_path_with(CliLogin::Codex, |_| None, None), None);
    }
}
