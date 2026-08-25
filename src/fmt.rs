//! Small display-formatting helpers shared by CLI subcommands that print a
//! table of images/models (`list`, `ps`) — kept in one place rather than
//! duplicated per-file so the two commands' NAME/ID/SIZE columns always
//! render identically for the same underlying digest/byte count.

use std::time::SystemTime;

/// First 12 hex chars of a `sha256:...` digest, matching `docker images`'s
/// convention (and Ollama's `ollama ps`/`ollama list`, which truncate to 12
/// as well).
pub fn short_id(digest: &str) -> String {
    let hex = digest.strip_prefix("sha256:").unwrap_or(digest);
    hex.chars().take(12).collect()
}

/// Decimal (not binary) byte units, matching `llmman list`'s existing
/// convention — GB/MB/kB, not GiB/MiB/KiB.
pub fn human_size(bytes: u64) -> String {
    const GB: u64 = 1_000_000_000;
    const MB: u64 = 1_000_000;
    const KB: u64 = 1_000;
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} kB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Human-scaled parameter/element count (e.g. "7.2B", "600.0M",
/// "125.0K") — used by `llmman show` to render a GGUF/safetensors model's
/// total parameter count, mirroring `ollama show`'s own
/// `format.HumanNumber`.
pub fn human_count(n: u64) -> String {
    const B: u64 = 1_000_000_000;
    const M: u64 = 1_000_000;
    const K: u64 = 1_000;
    if n >= B {
        format!("{:.1}B", n as f64 / B as f64)
    } else if n >= M {
        format!("{:.1}M", n as f64 / M as f64)
    } else if n >= K {
        format!("{:.1}K", n as f64 / K as f64)
    } else {
        n.to_string()
    }
}

/// Relative-time string ("5 minutes ago", "yesterday", ...) for a duration
/// expressed in seconds-in-the-past. Shared core of [`relative_time`] and
/// [`relative_time_rfc3339`], which just compute `secs` differently.
fn relative_time_secs(secs: u64) -> String {
    match secs {
        s if s < 60 => "just now".into(),
        s if s < 3600 => format!("{} minutes ago", s / 60),
        s if s < 86400 => format!("{} hours ago", s / 3600),
        s if s < 86400 * 2 => "yesterday".into(),
        s if s < 86400 * 7 => format!("{} days ago", s / 86400),
        s if s < 86400 * 14 => "1 week ago".into(),
        s if s < 86400 * 30 => format!("{} weeks ago", s / (86400 * 7)),
        s if s < 86400 * 60 => "1 month ago".into(),
        s if s < 86400 * 365 => format!("{} months ago", s / (86400 * 30)),
        s if s < 86400 * 730 => "1 year ago".into(),
        s => format!("{} years ago", s / (86400 * 365)),
    }
}

/// Relative-time string for a `SystemTime` (e.g. filesystem mtime) — used
/// by `llmman list`'s MODIFIED column.
pub fn relative_time(t: Option<SystemTime>) -> String {
    let secs = match t {
        Some(t) => SystemTime::now()
            .duration_since(t)
            .unwrap_or_default()
            .as_secs(),
        None => return "unknown".into(),
    };
    relative_time_secs(secs)
}

/// Relative-time string for an RFC 3339 timestamp (e.g. `llmman serve`'s
/// own `now_rfc3339()` output) — used by `llmman ps`'s STARTED column.
pub fn relative_time_rfc3339(s: &str) -> String {
    let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(s) else {
        return "unknown".into();
    };
    let secs = (chrono::Utc::now() - parsed.with_timezone(&chrono::Utc))
        .num_seconds()
        .max(0) as u64;
    relative_time_secs(secs)
}

/// "Time until" string for a duration expressed in seconds-in-the-future.
/// Shared core of [`until_rfc3339`] — split out (mirroring
/// `relative_time_secs`) so it can be tested with fixed inputs instead of
/// real wall-clock timestamps, which are inherently a little racy against
/// integer-seconds truncation right at a unit boundary.
fn until_secs(secs: u64) -> String {
    match secs {
        s if s < 60 => "just now".into(),
        s if s < 3600 => format!("{} minutes from now", s / 60),
        s if s < 86400 => format!("{} hours from now", s / 3600),
        s => format!("{} days from now", s / 86400),
    }
}

/// "Time until" string for an RFC 3339 timestamp in the future (a model's
/// computed `keep_alive` expiry) — used by `llmman ps`'s UNTIL column,
/// mirroring `ollama ps`'s own phrasing ("4 minutes from now"). A
/// timestamp already in the past (the idle-unload reaper just hasn't
/// caught up with it yet — see `reap_idle_models`) renders as "just now"
/// rather than a nonsensical negative duration.
pub fn until_rfc3339(s: &str) -> String {
    let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(s) else {
        return "unknown".into();
    };
    let secs = (parsed.with_timezone(&chrono::Utc) - chrono::Utc::now())
        .num_seconds()
        .max(0) as u64;
    until_secs(secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_id_truncates_to_12_hex_chars() {
        assert_eq!(
            short_id("sha256:0123456789abcdef0123456789abcdef"),
            "0123456789ab"
        );
        // No prefix: still truncates.
        assert_eq!(short_id("0123456789abcdef"), "0123456789ab");
    }

    #[test]
    fn human_size_picks_the_right_unit() {
        assert_eq!(human_size(512), "512 B");
        assert_eq!(human_size(1_500), "1.5 kB");
        assert_eq!(human_size(1_500_000), "1.5 MB");
        assert_eq!(human_size(1_500_000_000), "1.5 GB");
    }

    #[test]
    fn human_count_picks_the_right_unit() {
        assert_eq!(human_count(512), "512");
        assert_eq!(human_count(999), "999");
        assert_eq!(human_count(1_000), "1.0K");
        assert_eq!(human_count(1_500), "1.5K");
        assert_eq!(human_count(7_200_000_000), "7.2B");
        assert_eq!(human_count(600_000_000), "600.0M");
        assert_eq!(human_count(999_999_999), "1000.0M");
    }

    #[test]
    fn relative_time_rfc3339_handles_recent_and_invalid_timestamps() {
        let now = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        assert_eq!(relative_time_rfc3339(&now), "just now");
        assert_eq!(relative_time_rfc3339("not a timestamp"), "unknown");
    }

    #[test]
    fn until_secs_formats_the_unit_ladder() {
        assert_eq!(until_secs(0), "just now");
        assert_eq!(until_secs(59), "just now");
        assert_eq!(until_secs(60), "1 minutes from now");
        assert_eq!(until_secs(300), "5 minutes from now");
        assert_eq!(until_secs(3_600), "1 hours from now");
        assert_eq!(until_secs(7_200), "2 hours from now");
        assert_eq!(until_secs(86_400), "1 days from now");
    }

    #[test]
    fn until_rfc3339_clamps_a_past_timestamp_and_rejects_garbage() {
        // Already past its deadline (the reaper hasn't caught up yet) —
        // clamp rather than print a negative duration.
        let a_minute_ago = (chrono::Utc::now() - chrono::Duration::minutes(1))
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        assert_eq!(until_rfc3339(&a_minute_ago), "just now");

        assert_eq!(until_rfc3339("not a timestamp"), "unknown");
    }

    /// Ported from ollama's format/bytes_test.go (TestHumanBytes), adapted
    /// to this module's own decimal conventions: units are kB/MB/GB with no
    /// TB tier, and every unit above bytes always renders one decimal place
    /// ("1.0 kB" where ollama prints "1 KB"), so values that sit just under
    /// a unit boundary render as e.g. "1000.0 kB" rather than ollama's
    /// truncated "999 KB".
    #[test]
    fn human_size_ported_ollama_humanbytes_boundaries() {
        // Bytes.
        assert_eq!(human_size(0), "0 B");
        assert_eq!(human_size(1), "1 B");
        assert_eq!(human_size(999), "999 B");

        // Kilobytes.
        assert_eq!(human_size(1_000), "1.0 kB");
        assert_eq!(human_size(1_234), "1.2 kB");
        assert_eq!(human_size(999_999), "1000.0 kB");

        // Megabytes.
        assert_eq!(human_size(1_000_000), "1.0 MB");
        assert_eq!(human_size(1_234_567), "1.2 MB");

        // Gigabytes.
        assert_eq!(human_size(1_000_000_000), "1.0 GB");
        assert_eq!(human_size(1_234_567_890), "1.2 GB");

        // No TB tier: terabyte-scale sizes keep counting in GB.
        assert_eq!(human_size(1_000_000_000_000), "1000.0 GB");
        assert_eq!(human_size(1_500_000_000_000), "1500.0 GB");
    }

    /// Ported from ollama's format/time_test.go (TestHumanTime) plus its
    /// humanDuration unit ladder, adapted to this module's own phrasing
    /// ("just now"/"yesterday"/"N weeks ago" instead of ollama's
    /// "Less than a second"/"2 days"). Walks every unit boundary in
    /// relative_time_secs.
    #[test]
    fn relative_time_secs_ported_ollama_humantime_unit_ladder() {
        const DAY: u64 = 86_400;
        assert_eq!(relative_time_secs(0), "just now");
        assert_eq!(relative_time_secs(59), "just now");
        assert_eq!(relative_time_secs(60), "1 minutes ago");
        assert_eq!(relative_time_secs(120), "2 minutes ago");
        assert_eq!(relative_time_secs(3_600), "1 hours ago");
        assert_eq!(relative_time_secs(7_200), "2 hours ago");
        assert_eq!(relative_time_secs(DAY), "yesterday");
        assert_eq!(relative_time_secs(2 * DAY - 1), "yesterday");
        // ollama's "time in the past" case: now - 48h -> "2 days ago".
        assert_eq!(relative_time_secs(2 * DAY), "2 days ago");
        assert_eq!(relative_time_secs(6 * DAY), "6 days ago");
        assert_eq!(relative_time_secs(7 * DAY), "1 week ago");
        assert_eq!(relative_time_secs(14 * DAY), "2 weeks ago");
        assert_eq!(relative_time_secs(30 * DAY), "1 month ago");
        assert_eq!(relative_time_secs(60 * DAY), "2 months ago");
        assert_eq!(relative_time_secs(365 * DAY), "1 year ago");
        assert_eq!(relative_time_secs(730 * DAY), "2 years ago");
        // ollama's "time way in the future" renders "Forever"; this module
        // has no such cap and just keeps counting years.
        assert_eq!(relative_time_secs(200 * 365 * DAY), "200 years ago");
    }

    /// Ported from ollama's format/time_test.go: the zero value renders the
    /// caller's fallback ("never" there, "unknown" here), and a timestamp
    /// in the future clamps rather than panicking (ollama phrases it as
    /// "N days from now"; this module clamps to "just now" since nothing
    /// it formats — mtimes, daemon start times — can legitimately be in
    /// the future).
    #[test]
    fn relative_time_ported_ollama_humantime_edge_cases() {
        assert_eq!(relative_time(None), "unknown");
        let future = SystemTime::now() + std::time::Duration::from_secs(2 * 86_400);
        assert_eq!(relative_time(Some(future)), "just now");
        let past = SystemTime::now() - std::time::Duration::from_secs(2 * 86_400);
        assert_eq!(relative_time(Some(past)), "2 days ago");
    }
}
