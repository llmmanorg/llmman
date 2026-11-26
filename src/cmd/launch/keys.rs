//! What an agent authenticates to `llmman serve` with.
//!
//! An agent offered every provider's models can switch provider, so its one credential cannot be a
//! provider's key: the daemon would send it to whichever provider the next request names. The
//! launch registers the keys this shell holds with the daemon, each under its provider, and hands
//! the agent a session token instead (see `cmd::serve::sessions`). Keys stay in memory: nothing is
//! written to disk, put on a command line, or sent anywhere but the daemon, over a connection
//! [`check_provider_supported`] has made sure is loopback or TLS.
//!
//! [`check_provider_supported`]: super::check_provider_supported

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::daemon::ProviderDetail;

use super::{integration_key, sandbox, PROVIDER_NEEDS_DAEMON_KEY};

/// The token of the session this launch opened, closed by [`close`].
static OPEN: Mutex<Option<String>> = Mutex::new(None);

/// Closes the session this launch opened, if it did.
pub(super) fn close() {
    let token = OPEN.lock().unwrap_or_else(|e| e.into_inner()).take();
    if let Some(token) = token {
        crate::daemon::close_session(&token);
    }
}

/// The key this shell holds for `provider`, read from the variable this launcher itself ties to
/// that provider (see [`crate::providers::own_key_env`]) and never one the daemon names: the key is
/// sent back to the daemon, and a name that is not ours could point at any secret in this
/// environment. A loopback daemon is trusted no more than a remote one, since another user's
/// process may hold the port. The cheap checks come first: resolving a key asks about stored logins
/// and key commands.
pub(super) fn key_here(provider: &ProviderDetail) -> Option<String> {
    let own = crate::providers::own_key_env(&provider.id).flatten();
    let set = own
        .as_deref()
        .and_then(crate::providers::key_from_env)
        .is_some()
        || crate::config::provider_api_key(&provider.id).is_some();
    // Neither a placeholder nor a parent launch's session token is a key.
    set.then(|| provider.api_key()).flatten().filter(|k| {
        !crate::providers::is_placeholder_key(k) && !crate::providers::is_session_token(k)
    })
}

/// Whether `integration` can send a credential per request. The ones in
/// [`PROVIDER_NEEDS_DAEMON_KEY`] are configured through a file, which must
/// not hold one.
pub(super) fn carries_key(integration: &str) -> bool {
    !PROVIDER_NEEDS_DAEMON_KEY.contains(&integration.to_lowercase().as_str())
}

/// Whether a request for `provider` would go upstream with a credential
/// for an agent launched as `integration`: a key this shell holds and the
/// agent can carry, one the daemon will spend, or none needed.
pub(super) fn usable(integration: &str, provider: &ProviderDetail) -> bool {
    provider.key_optional
        || provider.daemon_key_usable()
        || (can_open_session(integration) && key_here(provider).is_some())
}

/// Whether this launch may register keys with the daemon: not over a cleartext network hop, not
/// when the daemon takes a key of its own from the agent, not for a file-configured integration,
/// and not where the agent's environment travels on a command line (OpenShell).
fn can_open_session(integration: &str) -> bool {
    carries_key(integration)
        && crate::daemon::connects_securely()
        && crate::auth::client_key().is_none()
        && !sandbox::env_on_command_line()
}

/// The credential `integration` is handed, after refusing what could only fail inside the agent:
/// `start`, the provider the launch starts on, must have a credential. Keys are registered with the
/// daemon for a session (every provider's when it is on this machine, `start`'s alone across the
/// network) and the agent gets the token; with none to register it gets [`integration_key`], so the
/// daemon's own are used.
pub(super) fn authenticate(
    integration: &str,
    providers: &[ProviderDetail],
    start: Option<&str>,
) -> anyhow::Result<String> {
    let start = start.and_then(|id| providers.iter().find(|p| p.id == id));
    if let Some(start) = start {
        // A provider this launcher cannot place by itself is looked up in
        // its own catalog, which is fetched if it has none: the daemon's
        // word does not say where a key lives.
        if crate::providers::own_key_env(&start.id).is_none() {
            let _ = crate::providers::catalog();
        }
    }
    let keys = if can_open_session(integration) {
        held_keys(
            providers,
            start,
            crate::daemon::connects_over_loopback(),
            key_here,
        )
    } else {
        BTreeMap::new()
    };
    if let Some(start) = start {
        check_start(integration, start, keys.contains_key(&start.id))?;
    }
    if keys.is_empty() {
        return Ok(integration_key());
    }
    let token = match crate::daemon::open_session(&keys) {
        Ok(token) => token,
        // A local start needs none of them: they only let the agent switch to a hosted provider.
        Err(e) if start.is_none() => {
            eprintln!(
                "[llmman] could not register provider keys ({e:#}); the agent will not be able \
                 to switch to a hosted provider"
            );
            return Ok(integration_key());
        }
        Err(e) => return Err(e),
    };
    *OPEN.lock().unwrap_or_else(|e| e.into_inner()) = Some(token.clone());
    Ok(token)
}

/// The keys this shell holds, each under the provider it belongs to: every
/// provider's when the daemon is on this machine (`everyone`), else only
/// `start`'s, which is all `--provider` ever disclosed to another machine.
fn held_keys(
    providers: &[ProviderDetail],
    start: Option<&ProviderDetail>,
    everyone: bool,
    key: impl Fn(&ProviderDetail) -> Option<String>,
) -> BTreeMap<String, String> {
    providers
        .iter()
        .filter(|p| everyone || start.is_some_and(|s| s.id == p.id))
        .filter_map(|p| Some((p.id.clone(), key(p)?)))
        .collect()
}

/// What each way the start provider can be reached asks of the launch: checks the daemon would
/// otherwise make at the first request, when the agent already has the terminal.
pub(super) fn check_start(
    integration: &str,
    start: &ProviderDetail,
    has_key: bool,
) -> anyhow::Result<()> {
    let daemon_authenticates = crate::auth::client_key().is_some();
    if !carries_key(integration) {
        // Fatal, not a warning: this integration cannot carry a key, so the daemon's is the only
        // one its first request can use, and `key_usable` says whether it would spend it.
        anyhow::ensure!(
            start.key_usable || start.key_optional,
            "{integration} is configured through a file, so it cannot send an API key: \
             llmman serve needs a key of its own, and must be bound to loopback (or \
             require an API key) to spend it.\n\
             Where the daemon runs, {}, then restart it.",
            start.key_hint()
        );
        return Ok(());
    }
    if has_key {
        // Warns once if it would cross the network in cleartext.
        start.client_key();
        return Ok(());
    }
    if start.daemon_key_usable() {
        if !daemon_authenticates {
            eprintln!(
                "[llmman] warning: no API key for {} here; using the key llmman serve has",
                start.name
            );
        }
        return Ok(());
    }
    if start.key_optional {
        return Ok(());
    }
    if daemon_authenticates {
        anyhow::bail!(
            "llmman serve requires an API key, so {integration} sends that one and cannot \
             also carry a key for {}: llmman serve needs a key of its own.\n\
             Where the daemon runs, {}, then restart it.",
            start.name,
            start.key_hint()
        );
    }
    anyhow::bail!("no API key for {} — {}", start.name, start.key_hint())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(id: &str, key_env: &str, usable: bool, optional: bool) -> ProviderDetail {
        serde_json::from_value(serde_json::json!({
            "id": id, "name": id, "base_url": format!("https://{id}.example"),
            "key_env": key_env, "key_set": usable, "key_usable": usable,
            "key_optional": optional, "models": [],
        }))
        .expect("a provider")
    }

    #[test]
    fn each_key_is_registered_under_its_own_provider_and_nowhere_else() {
        let a = provider("test-a", "A_KEY", false, false);
        let b = provider("test-b", "B_KEY", false, false);
        let none = provider("test-none", "NONE_KEY", false, false);
        let all = [a, b, none];
        let held = |p: &ProviderDetail| match p.id.as_str() {
            "test-a" => Some("sk-a".to_string()),
            "test-b" => Some("sk-b".to_string()),
            _ => None,
        };
        let keys = held_keys(&all, Some(&all[0]), true, held);
        assert_eq!(keys.get("test-a").map(String::as_str), Some("sk-a"));
        assert_eq!(keys.get("test-b").map(String::as_str), Some("sk-b"));
        assert_eq!(keys.len(), 2, "a provider without a key gets none");
        // Across the network only the starting provider's goes.
        let remote = held_keys(&all, Some(&all[0]), false, held);
        assert_eq!(remote.keys().collect::<Vec<_>>(), ["test-a"]);
        assert!(held_keys(&all, None, false, held).is_empty());
    }

    /// The daemon names the variable a key is read from, and a daemon that
    /// is not ours could name any secret: only a variable this launcher ties
    /// to the provider itself is read, and nothing else is sent.
    #[test]
    fn a_daemon_named_variable_is_not_read_and_not_sent() {
        let secret = "sk-not-for-the-daemon-3f9a";
        std::env::set_var("LLMMAN_TEST_UNRELATED_SECRET", secret);
        // A provider this launcher knows (a built-in) and one it does not.
        let known = provider("anthropic", "LLMMAN_TEST_UNRELATED_SECRET", false, false);
        let unknown = provider("test-unknown", "LLMMAN_TEST_UNRELATED_SECRET", false, false);
        assert_ne!(key_here(&known).as_deref(), Some(secret));
        assert_eq!(key_here(&unknown), None);
        let all = [known, unknown];
        for everyone in [true, false] {
            let sent = held_keys(&all, Some(&all[1]), everyone, key_here);
            assert!(sent.values().all(|key| key != secret), "{sent:?}");
            assert!(!sent.contains_key("test-unknown"));
        }
        std::env::remove_var("LLMMAN_TEST_UNRELATED_SECRET");
    }

    #[test]
    fn a_start_provider_with_no_credential_is_refused_before_the_handoff() {
        let bare = provider("test-bare", "LLMMAN_TEST_KEYS_UNSET", false, false);
        let err = check_start("claude", &bare, false).unwrap_err().to_string();
        assert!(
            err.contains("no API key for test-bare") && err.contains("LLMMAN_TEST_KEYS_UNSET"),
            "{err}"
        );
        // A key held here, a key the daemon will spend, or none needed.
        assert!(check_start("claude", &bare, true).is_ok());
        assert!(check_start("claude", &provider("p", "V", true, false), false).is_ok());
        assert!(check_start("claude", &provider("p", "V", false, true), false).is_ok());
    }

    #[test]
    fn a_file_configured_integration_needs_the_daemons_key() {
        assert!(!carries_key("hermes"));
        assert!(!carries_key("OpenClaw"));
        assert!(carries_key("claude"));
        let bare = provider("test-bare", "LLMMAN_TEST_KEYS_UNSET", false, false);
        let err = check_start("hermes", &bare, false).unwrap_err().to_string();
        assert!(err.contains("configured through a file"), "{err}");
        // A key of this shell's does not help it: it cannot send one.
        assert!(check_start("hermes", &bare, true).is_err());
        assert!(check_start("hermes", &provider("p", "V", true, false), false).is_ok());
    }

    #[test]
    fn without_a_key_to_register_the_agent_is_handed_the_placeholder() {
        let bare = [provider(
            "test-bare",
            "LLMMAN_TEST_KEYS_UNSET",
            false,
            false,
        )];
        assert_eq!(
            authenticate("claude", &bare, None).unwrap(),
            integration_key()
        );
    }
}
