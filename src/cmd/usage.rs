//! `llmman usage` — requests, tokens and cost from `crate::usage`'s
//! ledger, per model, provider, client, day or route. Reads the file
//! directly, as `llmman log` does.

use std::collections::HashMap;

use anyhow::Context;
use chrono::{DateTime, Local, Utc};
use clap::{Args, ValueEnum};
use serde_json::{json, Value};

use super::log::{clean, emit, patterns, window};
use crate::usage::{Entry, Tokens};

const AFTER_HELP: &str = "\
COST is US dollars at the catalog price when each request was made. `-`
means no request in the row had a price (a local model, a peer, a catalog
gap), which is not free; a trailing `+` means only some did, so the
figure is a floor. Written by `llmman serve` unless LLMMAN_NOUSAGE is set.";

#[derive(Args, Debug)]
#[command(after_help = AFTER_HELP)]
pub struct UsageArgs {
    /// Count requests more recent than a specific date (RFC 3339,
    /// YYYY-MM-DD, "yesterday", or "2 hours ago")
    #[arg(long, visible_alias = "after", value_name = "DATE")]
    pub since: Option<String>,
    /// Count requests older than a specific date
    #[arg(long, visible_alias = "before", value_name = "DATE")]
    pub until: Option<String>,
    /// Limit to models matching the pattern (regular expression).
    /// Repeat to match any of several
    #[arg(long, value_name = "PATTERN")]
    pub model: Vec<String>,
    /// Limit to a provider id; `-` for local models and peers. Repeat to
    /// match any of several
    #[arg(long, value_name = "ID")]
    pub provider: Vec<String>,
    /// Limit to clients (User-Agent) matching the pattern. Repeatable
    #[arg(long, value_name = "PATTERN")]
    pub client: Vec<String>,
    /// Limit to routes matching the pattern. Repeatable
    #[arg(long, value_name = "PATTERN")]
    pub route: Vec<String>,
    /// Match the limiting patterns without regard to letter case
    #[arg(short = 'i', long = "regexp-ignore-case")]
    pub ignore_case: bool,
    /// One row per model, provider, client, day (local time) or route
    #[arg(long, value_enum, default_value_t = By::Model)]
    pub by: By,
    /// One row summing every matching request
    #[arg(long, conflicts_with = "by")]
    pub total: bool,
    /// Print the rows as JSON
    #[arg(long, conflicts_with = "raw")]
    pub json: bool,
    /// Print the matching ledger entries, one JSON object per line
    #[arg(long, conflicts_with_all = ["by", "total"])]
    pub raw: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum By {
    Model,
    Provider,
    Client,
    Day,
    Route,
}

impl By {
    fn label(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Provider => "provider",
            Self::Client => "client",
            Self::Day => "day",
            Self::Route => "route",
        }
    }
}

pub fn run(args: &UsageArgs) -> anyhow::Result<()> {
    let path = crate::usage::path()?;
    let mut entries = crate::usage::read(&path)
        .with_context(|| format!("read usage ledger {}", path.display()))?;
    filter(&mut entries, args, Utc::now())?;

    let out = if args.raw {
        let mut out = String::new();
        for entry in &entries {
            out.push_str(&serde_json::to_string(entry)?);
            out.push('\n');
        }
        out
    } else if args.total {
        let total = total(&entries);
        if args.json {
            format!(
                "{}\n",
                serde_json::to_string_pretty(&row_json(&total, None))?
            )
        } else {
            table(&[total], None)
        }
    } else {
        let rows = group(&entries, args.by);
        if args.json {
            let rows: Vec<Value> = rows.iter().map(|r| row_json(r, Some(args.by))).collect();
            format!("{}\n", serde_json::to_string_pretty(&rows)?)
        } else {
            table(&rows, Some(args.by))
        }
    };
    emit(&out, false)
}

fn filter(entries: &mut Vec<Entry>, args: &UsageArgs, now: DateTime<Utc>) -> anyhow::Result<()> {
    let in_window = window(args.since.as_deref(), args.until.as_deref(), now)?;
    let model = patterns(&args.model, args.ignore_case)?;
    let client = patterns(&args.client, args.ignore_case)?;
    let route = patterns(&args.route, args.ignore_case)?;
    entries.retain(|e| {
        in_window(&e.time)
            && model.as_ref().is_none_or(|m| m.is_match(&e.model))
            && route.as_ref().is_none_or(|r| r.is_match(&e.route))
            && client
                .as_ref()
                .is_none_or(|c| e.client.as_deref().is_some_and(|ua| c.is_match(ua)))
            && (args.provider.is_empty()
                || args
                    .provider
                    .iter()
                    .any(|p| p == e.provider.as_deref().unwrap_or("-")))
    });
    Ok(())
}

/// A group's sums.
#[derive(Debug, Default, PartialEq)]
struct Row {
    key: String,
    /// A per-model row's provider column.
    provider: Option<String>,
    requests: u64,
    tokens: Tokens,
    /// US dollars, over the `priced` requests.
    cost: f64,
    priced: u64,
}

impl Row {
    fn add(&mut self, e: &Entry) {
        self.requests += 1;
        self.tokens.add(&e.tokens);
        if let Some(cost) = e.cost {
            self.cost += cost;
            self.priced += 1;
        }
    }
}

fn total(entries: &[Entry]) -> Row {
    let mut row = Row::default();
    entries.iter().for_each(|e| row.add(e));
    row
}

/// Days in order, anything else most expensive first, unpriced last.
fn group(entries: &[Entry], by: By) -> Vec<Row> {
    let mut rows: HashMap<(String, Option<String>), Row> = HashMap::new();
    for e in entries {
        let key = match by {
            By::Model => (e.model.clone(), e.provider.clone()),
            By::Provider => (e.provider.clone().unwrap_or_else(|| "-".into()), None),
            By::Client => (e.client.clone().unwrap_or_else(|| "-".into()), None),
            By::Day => (day(&e.time), None),
            By::Route => (e.route.clone(), None),
        };
        rows.entry(key.clone())
            .or_insert_with(|| Row {
                key: key.0,
                provider: key.1,
                ..Row::default()
            })
            .add(e);
    }
    let mut rows: Vec<Row> = rows.into_values().collect();
    if by == By::Day {
        rows.sort_by(|a, b| a.key.cmp(&b.key));
    } else {
        rows.sort_by(|a, b| {
            (b.priced > 0)
                .cmp(&(a.priced > 0))
                .then(b.cost.total_cmp(&a.cost))
                .then(b.requests.cmp(&a.requests))
                .then_with(|| a.key.cmp(&b.key))
                .then_with(|| a.provider.cmp(&b.provider))
        });
    }
    rows
}

/// The local calendar day of an RFC 3339 time.
fn day(time: &str) -> String {
    DateTime::parse_from_rfc3339(time)
        .map(|t| t.with_timezone(&Local).format("%Y-%m-%d").to_string())
        .unwrap_or_else(|_| "-".into())
}

/// Aligned columns as `llmman providers` prints; `by` is `None` for the
/// `--total` row, which has no group column.
fn table(rows: &[Row], by: Option<By>) -> String {
    let mut lines: Vec<Vec<String>> = Vec::new();
    let mut header: Vec<String> = Vec::new();
    if let Some(by) = by {
        header.push(by.label().to_ascii_uppercase());
        if by == By::Model {
            header.push("PROVIDER".into());
        }
    }
    header.extend(["REQUESTS", "INPUT", "CACHED", "OUTPUT", "COST"].map(String::from));
    lines.push(header);
    for row in rows {
        let mut line = Vec::new();
        if let Some(by) = by {
            line.push(clean(&row.key));
            if by == By::Model {
                line.push(clean(row.provider.as_deref().unwrap_or("-")));
            }
        }
        line.push(row.requests.to_string());
        line.push(crate::fmt::human_count(row.tokens.input));
        line.push(crate::fmt::human_count(row.tokens.cache_read));
        line.push(crate::fmt::human_count(row.tokens.output));
        line.push(dollars(row));
        lines.push(line);
    }
    let widths: Vec<usize> = (0..lines[0].len())
        .map(|i| {
            lines
                .iter()
                .map(|l| l[i].chars().count())
                .max()
                .unwrap_or(0)
        })
        .collect();
    let mut out = String::new();
    for line in lines {
        let last = line.len() - 1;
        for (i, cell) in line.iter().enumerate() {
            if i == last {
                out.push_str(cell);
            } else {
                out.push_str(cell);
                let pad = widths[i] - cell.chars().count() + 4;
                out.extend(std::iter::repeat_n(' ', pad));
            }
        }
        out.push('\n');
    }
    out
}

/// `-` for no price, not free (`$0.00`); `+` when only some had one.
fn dollars(row: &Row) -> String {
    if row.priced == 0 {
        return "-".into();
    }
    let partial = if row.priced < row.requests { "+" } else { "" };
    format!("{}{partial}", money(row.cost))
}

/// Two places above a cent, four below.
fn money(value: f64) -> String {
    if value == 0.0 {
        "$0.00".into()
    } else if value < 0.0001 {
        "<$0.0001".into()
    } else if value < 0.01 {
        format!("${value:.4}")
    } else {
        format!("${value:.2}")
    }
}

/// camelCase, as `verify --json`; `cost` is `null` without a price.
fn row_json(row: &Row, by: Option<By>) -> Value {
    let mut value = json!({
        "requests": row.requests,
        "inputTokens": row.tokens.input,
        "cacheReadTokens": row.tokens.cache_read,
        "cacheWriteTokens": row.tokens.cache_write,
        "outputTokens": row.tokens.output,
        "reasoningTokens": row.tokens.reasoning,
        "cost": (row.priced > 0).then_some(row.cost),
        "unpricedRequests": row.requests - row.priced,
    });
    if let Some(by) = by {
        value[by.label()] = json!(row.key);
        if by == By::Model {
            value["provider"] = json!(row.provider);
        }
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn entry(time: &str, model: &str, provider: Option<&str>, cost: Option<f64>) -> Entry {
        Entry {
            id: "0".repeat(40),
            time: time.into(),
            route: "/v1/messages".into(),
            model: model.into(),
            provider: provider.map(str::to_string),
            client: Some("claude-cli/1.0".into()),
            tokens: Tokens::new(1_000, 800, 0, 100),
            rate: None,
            cost,
        }
    }

    fn ledger() -> Vec<Entry> {
        vec![
            entry("2026-09-01T10:00:00Z", "qwen3:8b", None, None),
            entry(
                "2026-09-01T11:00:00Z",
                "llmman.provider/openrouter/deepseek/deepseek-v3",
                Some("openrouter"),
                Some(0.001),
            ),
            entry(
                "2026-09-02T10:00:00Z",
                "llmman.provider/openrouter/anthropic/claude-sonnet-4",
                Some("openrouter"),
                Some(1.5),
            ),
            entry(
                "2026-09-03T10:00:00Z",
                "llmman.provider/openrouter/anthropic/claude-sonnet-4",
                Some("openrouter"),
                Some(2.5),
            ),
        ]
    }

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        args: UsageArgs,
    }

    fn args(argv: &[&str]) -> UsageArgs {
        Cli::try_parse_from(std::iter::once("usage").chain(argv.iter().copied()))
            .unwrap()
            .args
    }

    #[test]
    fn rows_are_most_expensive_first_with_the_unpriced_last() {
        let rows = group(&ledger(), By::Model);
        let keys: Vec<&str> = rows.iter().map(|r| r.key.as_str()).collect();
        assert_eq!(
            keys,
            [
                "llmman.provider/openrouter/anthropic/claude-sonnet-4",
                "llmman.provider/openrouter/deepseek/deepseek-v3",
                "qwen3:8b"
            ]
        );
        assert_eq!(rows[0].requests, 2);
        assert_eq!(rows[0].tokens, Tokens::new(2_000, 1_600, 0, 200));
        assert!((rows[0].cost - 4.0).abs() < 1e-9);
        assert_eq!(rows[2].provider, None);

        let text = table(&rows, Some(By::Model));
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines[0].split_whitespace().collect::<Vec<_>>(),
            ["MODEL", "PROVIDER", "REQUESTS", "INPUT", "CACHED", "OUTPUT", "COST"]
        );
        assert_eq!(
            lines[1].split_whitespace().collect::<Vec<_>>(),
            [
                "llmman.provider/openrouter/anthropic/claude-sonnet-4",
                "openrouter",
                "2",
                "2.0K",
                "1.6K",
                "200",
                "$4.00"
            ]
        );
        assert!(lines[2].ends_with("$0.0010"), "{}", lines[2]);
        assert!(lines[3].ends_with(" -"), "{}", lines[3]);
        assert_eq!(lines.len(), 4, "nothing after the last row");
        // Columns line up under their headers.
        let col = lines[0].find("REQUESTS").unwrap();
        assert!(lines[1..]
            .iter()
            .all(|l| l[col..].starts_with(char::is_numeric)));
    }

    #[test]
    fn an_unpriced_row_sorts_after_a_free_one() {
        let mut entries = vec![entry("2026-09-01T10:00:00Z", "free", Some("p"), Some(0.0))];
        entries.extend((0..3).map(|_| entry("2026-09-01T10:00:00Z", "local", None, None)));
        let keys: Vec<String> = group(&entries, By::Model)
            .into_iter()
            .map(|r| r.key)
            .collect();
        assert_eq!(keys, ["free", "local"]);
    }

    #[test]
    fn days_are_in_order_and_a_partly_priced_row_is_a_floor() {
        let mut entries = ledger();
        entries.push(entry(
            "2026-09-02T12:00:00Z",
            "llmman.provider/cohere/command",
            Some("cohere"),
            None,
        ));
        let rows = group(&entries, By::Day);
        assert!(rows.windows(2).all(|w| w[0].key < w[1].key));
        let by_provider = group(&entries, By::Provider);
        let keys: Vec<&str> = by_provider.iter().map(|r| r.key.as_str()).collect();
        assert_eq!(keys, ["openrouter", "-", "cohere"]);
        let everything = total(&entries);
        assert_eq!(everything.requests, 5);
        assert_eq!(dollars(&everything), "$4.00+");
    }

    #[test]
    fn money_keeps_a_cheap_turn_visible_and_free_distinct_from_unknown() {
        assert_eq!(money(0.0), "$0.00");
        assert_eq!(money(0.00001), "<$0.0001");
        assert_eq!(money(0.0042), "$0.0042");
        assert_eq!(money(4.123), "$4.12");
        let unpriced = Row {
            requests: 3,
            ..Row::default()
        };
        assert_eq!(dollars(&unpriced), "-");
        let free = Row {
            requests: 3,
            priced: 3,
            ..Row::default()
        };
        assert_eq!(dollars(&free), "$0.00");
    }

    #[test]
    fn filters_combine() {
        let now = DateTime::parse_from_rfc3339("2026-09-04T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let kept = |argv: &[&str]| {
            let mut entries = ledger();
            filter(&mut entries, &args(argv), now).unwrap();
            entries.len()
        };
        assert_eq!(kept(&[]), 4);
        assert_eq!(kept(&["--provider", "-"]), 1);
        assert_eq!(kept(&["--provider", "openrouter"]), 3);
        assert_eq!(kept(&["--model", "SONNET", "-i"]), 2);
        assert_eq!(kept(&["--model", "sonnet", "--model", "qwen"]), 3);
        assert_eq!(kept(&["--since", "2026-09-02T00:00:00Z"]), 2);
        assert_eq!(kept(&["--before", "2026-09-02T00:00:00Z"]), 2);
        assert_eq!(kept(&["--since", "1 day ago"]), 1);
        assert_eq!(kept(&["--client", "^claude-cli/"]), 4);
        assert_eq!(kept(&["--client", "codex"]), 0);
        assert_eq!(kept(&["--route", "^/v1/messages$"]), 4);
        assert_eq!(kept(&["--route", "chat"]), 0);
    }

    #[test]
    fn the_json_rows_name_their_group() {
        let rows = group(&ledger(), By::Model);
        let local = row_json(&rows[2], Some(By::Model));
        assert_eq!(local["model"], "qwen3:8b");
        assert!(local["provider"].is_null());
        assert!(local["cost"].is_null(), "unknown, not free");
        assert_eq!(local["unpricedRequests"], 1);
        assert_eq!(local["cacheReadTokens"], 800);
        let total = row_json(&total(&ledger()), None);
        assert!(total.get("model").is_none());
        assert_eq!(total["requests"], 4);
        let day = row_json(&group(&ledger(), By::Day)[0], Some(By::Day));
        assert!(day["day"].is_string() && day.get("provider").is_none());
    }

    #[test]
    fn total_and_raw_do_not_take_a_grouping() {
        let parse = |argv: &[&str]| {
            Cli::try_parse_from(std::iter::once("usage").chain(argv.iter().copied())).is_ok()
        };
        assert!(parse(&["--total", "--json"]));
        assert!(parse(&["--raw", "--model", "x"]));
        assert!(!parse(&["--total", "--by", "day"]));
        assert!(!parse(&["--raw", "--json"]));
        assert!(!parse(&["--by", "week"]));
    }
}
