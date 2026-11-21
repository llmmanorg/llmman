//! The polling half of an OAuth device-authorization login (RFC 8628),
//! with the clock injected so every outcome can be tested without waiting.
//!
//! `crate::oauth` has the same dance for Docker Hub; that one is ported
//! line for line from `docker/cli` and stays as it is. The providers here
//! differ in how they say "not yet" — ChatGPT answers 403/404, GitHub a
//! 200 with an `error` field, xAI a 400 — so the loop takes a `poll`
//! closure that has already read the response and returns a [`Poll`].

use std::time::{Duration, Instant};

use anyhow::anyhow;

/// What one poll of the token endpoint said.
#[derive(Debug, PartialEq)]
pub enum Poll<T> {
    Done(T),
    /// `authorization_pending`: the user has not finished.
    Pending,
    /// `slow_down`: poll less often, by the server's new interval if it
    /// gave one, else five seconds more (RFC 8628 §3.5).
    SlowDown(Option<Duration>),
    /// `access_denied`: the user said no.
    Denied,
    /// `expired_token`: the code ran out.
    Expired,
    /// Anything else; the message is safe to show.
    Failed(String),
}

/// Time, as the loop needs it.
pub trait Clock {
    fn sleep(&self, d: Duration);
    /// Time passed since the loop began.
    fn elapsed(&self) -> Duration;
}

/// The wall clock.
pub struct RealClock(Instant);

impl RealClock {
    pub fn start() -> Self {
        Self(Instant::now())
    }
}

impl Clock for RealClock {
    fn sleep(&self, d: Duration) {
        std::thread::sleep(d);
    }

    fn elapsed(&self) -> Duration {
        self.0.elapsed()
    }
}

/// Floor on the polling interval, whatever the server asks for: a zero
/// would be a busy loop.
const MIN_INTERVAL: Duration = Duration::from_secs(1);

/// Polls until the user finishes, refuses, or `expires_in` passes.
///
/// Sleeps *before* each poll, as `docker login` does: the user has just
/// been shown a code and cannot have entered it yet.
pub fn run<T>(
    clock: &dyn Clock,
    interval: Duration,
    expires_in: Duration,
    mut poll: impl FnMut() -> Poll<T>,
) -> anyhow::Result<T> {
    let mut interval = interval.max(MIN_INTERVAL);
    let start = clock.elapsed();
    loop {
        clock.sleep(interval);
        if clock.elapsed() - start >= expires_in {
            return Err(anyhow!("timed out waiting for the login to be approved"));
        }
        match poll() {
            Poll::Done(token) => return Ok(token),
            Poll::Pending => {}
            Poll::SlowDown(Some(next)) => interval = next.max(interval).max(MIN_INTERVAL),
            Poll::SlowDown(None) => interval += Duration::from_secs(5),
            Poll::Denied => return Err(anyhow!("the login was denied")),
            Poll::Expired => return Err(anyhow!("the login code expired; run the command again")),
            Poll::Failed(why) => return Err(anyhow!("the login failed: {why}")),
        }
    }
}

/// Reads an RFC 8628 §3.5 token response — GitHub's 200-with-`error`,
/// xAI's 400-with-`error` — into a [`Poll`]. `done` turns a success body
/// into the result; the error codes are the RFC's.
pub fn rfc8628<T>(
    ok: bool,
    body: &serde_json::Value,
    done: impl FnOnce(&serde_json::Value) -> Result<T, String>,
) -> Poll<T> {
    let error = body.get("error").and_then(serde_json::Value::as_str);
    match error {
        None if ok => match done(body) {
            Ok(token) => Poll::Done(token),
            Err(why) => Poll::Failed(why),
        },
        Some("authorization_pending") => Poll::Pending,
        Some("slow_down") => Poll::SlowDown(
            body.get("interval")
                .and_then(serde_json::Value::as_u64)
                .filter(|i| *i > 0)
                .map(Duration::from_secs),
        ),
        // `authorization_denied` is xAI's spelling (OpenClaw
        // `extensions/xai/xai-oauth.ts`).
        Some("access_denied" | "authorization_denied") => Poll::Denied,
        Some("expired_token") => Poll::Expired,
        Some(other) => Poll::Failed(super::clean(other)),
        None => Poll::Failed("the server's reply had no token".to_string()),
    }
}

#[cfg(test)]
pub(crate) mod fake {
    use super::Clock;
    use std::cell::{Cell, RefCell};
    use std::time::Duration;

    /// Time that only passes when slept through.
    pub(crate) struct FakeClock {
        pub slept: RefCell<Vec<Duration>>,
        now: Cell<Duration>,
    }
    impl FakeClock {
        pub fn new() -> Self {
            Self {
                slept: RefCell::default(),
                now: Cell::new(Duration::ZERO),
            }
        }
    }
    impl Clock for FakeClock {
        fn sleep(&self, d: Duration) {
            self.slept.borrow_mut().push(d);
            self.now.set(self.now.get() + d);
        }
        fn elapsed(&self) -> Duration {
            self.now.get()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::FakeClock;
    use super::*;
    use std::cell::Cell;

    fn secs(n: u64) -> Duration {
        Duration::from_secs(n)
    }

    #[test]
    fn pending_then_slow_down_then_done() {
        let clock = FakeClock::new();
        let mut script = vec![
            Poll::Pending,
            Poll::SlowDown(None),
            Poll::SlowDown(Some(secs(20))),
            Poll::Done("token"),
        ]
        .into_iter();
        let got = run(&clock, secs(5), secs(900), || script.next().unwrap()).unwrap();
        assert_eq!(got, "token");
        // 5 before the first poll, 5 after pending, 10 after +5, 20 as told.
        assert_eq!(
            *clock.slept.borrow(),
            [secs(5), secs(5), secs(10), secs(20)]
        );
    }

    #[test]
    fn the_window_counts_from_the_start_of_polling() {
        let clock = FakeClock::new();
        clock.sleep(secs(20));
        let got = run(&clock, secs(5), secs(30), || Poll::Done("token")).unwrap();
        assert_eq!(got, "token");
    }

    #[test]
    fn a_server_cannot_speed_the_loop_up() {
        let clock = FakeClock::new();
        let mut script = vec![Poll::SlowDown(Some(secs(1))), Poll::Done(())].into_iter();
        run(&clock, secs(8), secs(900), || script.next().unwrap()).unwrap();
        assert_eq!(*clock.slept.borrow(), [secs(8), secs(8)]);
        // And a zero interval is a second.
        let clock = FakeClock::new();
        run(&clock, Duration::ZERO, secs(900), || Poll::Done(())).unwrap();
        assert_eq!(*clock.slept.borrow(), [secs(1)]);
    }

    #[test]
    fn terminal_outcomes_end_the_loop() {
        let outcome = |poll: fn() -> Poll<()>| {
            let clock = FakeClock::new();
            let calls = Cell::new(0);
            let err = run(&clock, secs(5), secs(900), || {
                calls.set(calls.get() + 1);
                poll()
            })
            .unwrap_err()
            .to_string();
            assert_eq!(calls.get(), 1);
            err
        };
        assert!(outcome(|| Poll::Denied).contains("denied"));
        assert!(outcome(|| Poll::Expired).contains("expired"));
        assert!(outcome(|| Poll::Failed("boom".into())).contains("boom"));
    }

    #[test]
    fn it_gives_up_when_the_code_expires() {
        let clock = FakeClock::new();
        let polls = Cell::new(0);
        let err = run::<()>(&clock, secs(5), secs(20), || {
            polls.set(polls.get() + 1);
            Poll::Pending
        })
        .unwrap_err();
        assert!(err.to_string().contains("timed out"), "{err}");
        // 5, 10, 15 poll; the 20th second is past the deadline.
        assert_eq!(polls.get(), 3);
    }

    #[test]
    fn rfc8628_errors_map_to_outcomes() {
        let read = |ok, body: serde_json::Value| {
            rfc8628(ok, &body, |b| {
                b.get("access_token")
                    .and_then(|t| t.as_str())
                    .map(str::to_string)
                    .ok_or_else(|| "no token".to_string())
            })
        };
        use serde_json::json;
        assert_eq!(
            read(true, json!({"access_token": "t"})),
            Poll::Done("t".into())
        );
        assert_eq!(
            read(false, json!({"error": "authorization_pending"})),
            Poll::Pending
        );
        assert_eq!(
            read(true, json!({"error": "authorization_pending"})),
            Poll::Pending
        );
        assert_eq!(
            read(true, json!({"error": "slow_down", "interval": 10})),
            Poll::SlowDown(Some(secs(10)))
        );
        assert_eq!(
            read(false, json!({"error": "slow_down"})),
            Poll::SlowDown(None)
        );
        assert_eq!(read(false, json!({"error": "access_denied"})), Poll::Denied);
        assert_eq!(
            read(false, json!({"error": "authorization_denied"})),
            Poll::Denied
        );
        assert_eq!(
            read(false, json!({"error": "expired_token"})),
            Poll::Expired
        );
        assert_eq!(
            read(false, json!({"error": "incorrect_client_credentials"})),
            Poll::Failed("incorrect_client_credentials".into())
        );
        assert_eq!(read(true, json!({})), Poll::Failed("no token".into()));
        assert!(matches!(read(false, json!({})), Poll::Failed(_)));
    }
}
