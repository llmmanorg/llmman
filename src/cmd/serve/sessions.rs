//! Provider keys held for one agent session.
//!
//! An agent started by `llmman launch` can switch to any provider's model, so the `Authorization`
//! header it sends cannot be one provider's key: the daemon would forward it to whichever provider
//! the next request names. The launcher registers its keys here by provider id and gives the agent
//! a session token instead; a request with the token is served with the key registered for the
//! provider it routes to and no other (else the daemon's own credential, or an error). Keys live in
//! this process's memory only.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use anyhow::{ensure, Context};

use crate::providers::SESSION_TOKEN_PREFIX as TOKEN_PREFIX;

/// What every token ends with. Claude Code asks whether to use "a custom API
/// key" and remembers the answer by the key's last 20 characters, so a token
/// that varied to its last character would ask at every launch.
const TOKEN_SUFFIX: &str = "-llmman-session-token";

/// A session unused for this long is forgotten.
const IDLE_TTL: Duration = Duration::from_secs(12 * 60 * 60);
/// At most this many sessions are open; a new one is refused past it (see
/// [`RECENTLY_USED`]).
const MAX_SESSIONS: usize = 64;
/// A session used within this long belongs to an agent that is running, and
/// is never evicted to make room: anything that reaches the daemon could
/// otherwise push a running launch's keys out by opening sessions.
const RECENTLY_USED: Duration = Duration::from_secs(60 * 60);
/// What one launch can need: a key per provider in the catalog (about 300)
/// and the ones in llmman.conf. With the lengths below, one session holds at
/// most [`MAX_BYTES`] and the store `MAX_SESSIONS` times that.
const MAX_KEYS: usize = 512;
const MAX_ID_LEN: usize = 128;
const MAX_KEY_LEN: usize = 4096;
const MAX_BYTES: usize = 256 * 1024;
/// The request body limit of `POST /llmman/sessions`: [`MAX_BYTES`] plus
/// the JSON around it.
pub(super) const BODY_LIMIT_BYTES: usize = 2 * MAX_BYTES;
const _: () = assert!(MAX_BYTES < BODY_LIMIT_BYTES);

struct Session {
    keys: HashMap<String, String>,
    last_used: Instant,
}

/// The open sessions by token. Process-wide ([`SESSIONS`]); a value of its
/// own so the limits are testable without every other test's sessions.
#[derive(Default)]
struct Store {
    open: HashMap<String, Session>,
}

/// The store holds [`MAX_SESSIONS`] sessions, every one used lately.
#[derive(Debug)]
pub(super) struct Full;

impl std::fmt::Display for Full {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "this llmman serve already holds {MAX_SESSIONS} launch sessions, all used within the \
             last hour: close a running `llmman launch` (or wait for one to idle) and launch again"
        )
    }
}

impl std::error::Error for Full {}

impl Store {
    /// Opens a session. When the store is full, the least recently used
    /// session not used within [`RECENTLY_USED`] makes room; if there is
    /// none the new session is refused.
    fn open(
        &mut self,
        token: String,
        keys: HashMap<String, String>,
        now: Instant,
    ) -> Result<(), Full> {
        self.open
            .retain(|_, s| now.duration_since(s.last_used) < IDLE_TTL);
        while self.open.len() >= MAX_SESSIONS {
            let oldest = self
                .open
                .iter()
                .filter(|(_, s)| now.duration_since(s.last_used) >= RECENTLY_USED)
                .min_by_key(|(_, s)| s.last_used)
                .map(|(t, _)| t.clone())
                .ok_or(Full)?;
            self.open.remove(&oldest);
        }
        self.open.insert(
            token,
            Session {
                keys,
                last_used: now,
            },
        );
        Ok(())
    }

    fn key_for(&mut self, token: &str, provider: &str, now: Instant) -> Option<String> {
        let session = self.open.get_mut(token)?;
        if now.duration_since(session.last_used) >= IDLE_TTL {
            self.open.remove(token);
            return None;
        }
        session.last_used = now;
        session.keys.get(provider).cloned()
    }
}

static SESSIONS: LazyLock<Mutex<Store>> = LazyLock::new(|| Mutex::new(Store::default()));

fn sessions() -> std::sync::MutexGuard<'static, Store> {
    SESSIONS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether `value` is a session token rather than a provider's key.
pub(super) fn is_token(value: &str) -> bool {
    crate::providers::is_session_token(value)
}

/// Opens a session holding `keys` (provider id to key), and returns its token.
pub(super) fn open(keys: HashMap<String, String>) -> anyhow::Result<String> {
    ensure!(keys.len() <= MAX_KEYS, "too many keys for one session");
    let bytes: usize = keys.iter().map(|(p, k)| p.len() + k.len()).sum();
    ensure!(bytes <= MAX_BYTES, "the keys of one session are too large");
    for (provider, key) in &keys {
        ensure!(
            !provider.is_empty()
                && provider.len() <= MAX_ID_LEN
                && !key.is_empty()
                && key.len() <= MAX_KEY_LEN,
            "a session key must be a non-empty provider id and key of a sane length"
        );
        ensure!(
            !key.contains(['\n', '\r'])
                && !is_token(key)
                && !crate::providers::is_placeholder_key(key),
            "a session key must be one line that is neither a session token nor a placeholder"
        );
    }
    let mut random = [0u8; 24];
    getrandom::getrandom(&mut random).context("no randomness for a session token")?;
    let token = format!("{TOKEN_PREFIX}{}{TOKEN_SUFFIX}", hex::encode(random));
    sessions().open(token.clone(), keys, Instant::now())?;
    Ok(token)
}

/// The key `token`'s session holds for `provider`, if the session is still
/// open and has one.
pub(super) fn key_for(token: &str, provider: &str) -> Option<String> {
    sessions().key_for(token, provider, Instant::now())
}

/// Closes `token`'s session; whether there was one.
pub(super) fn close(token: &str) -> bool {
    sessions().open.remove(token).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(p, k)| (p.to_string(), k.to_string()))
            .collect()
    }

    #[test]
    fn a_session_hands_out_each_providers_own_key_and_no_other() {
        let token = open(keys(&[("openai", "sk-a"), ("anthropic", "sk-b")])).unwrap();
        assert!(is_token(&token));
        assert_eq!(key_for(&token, "openai").as_deref(), Some("sk-a"));
        assert_eq!(key_for(&token, "anthropic").as_deref(), Some("sk-b"));
        assert_eq!(key_for(&token, "openrouter"), None);
        assert_eq!(key_for("llmman-session-unknown", "openai"), None);
        assert!(close(&token));
        assert_eq!(key_for(&token, "openai"), None);
        assert!(!close(&token));
    }

    #[test]
    fn tokens_are_distinct_and_keys_that_look_like_tokens_are_refused() {
        let a = open(keys(&[])).unwrap();
        let b = open(keys(&[])).unwrap();
        assert_ne!(a, b);
        assert!(open(keys(&[("openai", &a)])).is_err());
        assert!(open(keys(&[("openai", "")])).is_err());
        assert!(open(keys(&[("openai", "llmman")])).is_err());
        assert!(open(keys(&[("openai", "two\nlines")])).is_err());
        close(&a);
        close(&b);
    }

    #[test]
    fn one_session_holds_a_bounded_amount() {
        let many: HashMap<String, String> = (0..=MAX_KEYS)
            .map(|i| (format!("p{i}"), "k".into()))
            .collect();
        assert!(open(many).is_err(), "too many keys");
        let long = keys(&[("openai", &"k".repeat(MAX_KEY_LEN + 1))]);
        assert!(open(long).is_err(), "a key too long");
        let id = "p".repeat(MAX_ID_LEN + 1);
        assert!(open(keys(&[(&id, "k")])).is_err(), "an id too long");
        let bulky: HashMap<String, String> = (0..MAX_KEYS)
            .map(|i| (format!("p{i}"), "k".repeat(MAX_KEY_LEN)))
            .collect();
        assert!(open(bulky).is_err(), "too many bytes in all");
        let token = open(keys(&[("openai", &"k".repeat(MAX_KEY_LEN))])).unwrap();
        close(&token);
    }

    #[test]
    fn an_unused_session_makes_room_and_a_recent_one_is_never_evicted() {
        let mut store = Store::default();
        let start = Instant::now();
        for i in 0..MAX_SESSIONS {
            let at = start + Duration::from_secs(i as u64);
            store
                .open(format!("t{i}"), keys(&[("p", "k")]), at)
                .unwrap();
        }
        // All used within the hour: the new one is refused, none goes.
        let soon = start + Duration::from_secs(MAX_SESSIONS as u64);
        let err = store
            .open("new".into(), keys(&[("p", "k")]), soon)
            .unwrap_err();
        assert!(err.to_string().contains("64 launch sessions"), "{err}");
        assert_eq!(store.open.len(), MAX_SESSIONS);
        assert!(store.key_for("t0", "p", soon).is_some());
        // Past the hour only the least recently used of the rest goes, and
        // one used since stays.
        let later = start + RECENTLY_USED + Duration::from_secs(10);
        assert!(store.key_for("t0", "p", later).is_some());
        store
            .open("new".into(), keys(&[("p", "k")]), later)
            .unwrap();
        assert_eq!(store.open.len(), MAX_SESSIONS);
        assert!(store.key_for("t0", "p", later).is_some(), "used since");
        assert_eq!(store.key_for("t1", "p", later), None, "the oldest unused");
        assert!(store.key_for("t2", "p", later).is_some());
        assert!(store.key_for("new", "p", later).is_some());
    }

    #[test]
    fn an_idle_session_is_forgotten_and_use_keeps_one_alive() {
        let mut store = Store::default();
        let start = Instant::now();
        store
            .open("idle".into(), keys(&[("p", "k")]), start)
            .unwrap();
        store
            .open("used".into(), keys(&[("p", "k")]), start)
            .unwrap();
        let later = start + IDLE_TTL - Duration::from_secs(1);
        assert!(store.key_for("used", "p", later).is_some());
        let much_later = later + Duration::from_secs(2);
        assert_eq!(store.key_for("idle", "p", much_later), None);
        assert!(store.key_for("used", "p", much_later).is_some());
    }
}
