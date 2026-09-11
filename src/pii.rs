//! Local detection of personal data, for the one question `llmman serve`
//! has to answer before a [hybrid pair](crate::hybrid) sends a request
//! away: does this body carry something that should not leave the
//! machine?
//!
//! The gate is deliberately narrow. It reports only identifiers that a
//! deterministic check can confirm — an address that parses, a card
//! number that passes Luhn, an IBAN that passes mod-97 — and says
//! nothing about names, places or anything else that needs a model to
//! judge. That leaves real personal data undetected, so this is a floor
//! rather than a promise: it catches the identifiers that are cheap to
//! be sure about, and never pretends to more.
//!
//! Both error directions are one-sided on purpose. A false positive
//! keeps a request on the local model, which costs a worse answer. A
//! false negative sends it to a provider, which cannot be undone. So
//! every ambiguous case here resolves towards detecting, and the whole
//! module only ever makes [`crate::hybrid::route`] pick `Local`.
//!
//! Scanning finds *every* mention, not the first: an identifier repeated
//! across turns of a long conversation is exactly the case a
//! first-match scan gets wrong, and the count is what the routing log
//! reports.

use regex::Regex;
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Kinds
// ---------------------------------------------------------------------------

/// A class of identifier this module can confirm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Email,
    Phone,
    CreditCard,
    Iban,
    /// A US Social Security number: the one national ID with a format
    /// checkable without knowing the country.
    NationalId,
    /// A public IPv4 address. Private, loopback and link-local ones are
    /// not personal data — they are every agent's own `127.0.0.1:8080`.
    IpAddress,
    /// An API key, token or private key. Not personal data as such, but
    /// it is the other thing a request must not carry off the machine,
    /// and it is detectable by exactly the same means.
    Secret,
}

impl Kind {
    /// How a kind is named in the routing log, singular and plural.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Email => "email address",
            Self::Phone => "phone number",
            Self::CreditCard => "payment card number",
            Self::Iban => "IBAN",
            Self::NationalId => "national ID number",
            Self::IpAddress => "public IP address",
            Self::Secret => "credential",
        }
    }

    fn plural(self) -> &'static str {
        match self {
            Self::Email => "email addresses",
            Self::Phone => "phone numbers",
            Self::CreditCard => "payment card numbers",
            Self::Iban => "IBANs",
            Self::NationalId => "national ID numbers",
            Self::IpAddress => "public IP addresses",
            Self::Secret => "credentials",
        }
    }

    /// The capture group naming this kind in [`CANDIDATES`].
    fn group(self) -> &'static str {
        match self {
            Self::Email => "email",
            Self::Phone => "phone",
            Self::CreditCard => "card",
            Self::Iban => "iban",
            Self::NationalId => "ssn",
            Self::IpAddress => "ip",
            Self::Secret => "secret",
        }
    }
}

/// Every kind, in the order [`CANDIDATES`] prefers them. Longer and
/// more distinctive shapes come first: the alternation is leftmost-first,
/// so at a position where two could match, the one listed earlier wins
/// and the other never sees that text.
const KINDS: [Kind; 7] = [
    Kind::Email,
    Kind::Secret,
    Kind::Iban,
    Kind::NationalId,
    Kind::CreditCard,
    Kind::Phone,
    Kind::IpAddress,
];

// ---------------------------------------------------------------------------
// Candidate patterns
// ---------------------------------------------------------------------------

/// One pass over the text for all seven kinds, as a single alternation
/// of named groups rather than seven passes. Every branch is a
/// *candidate* only: [`confirmed`] decides whether a match is real, so
/// these can be loose where a checksum tightens them afterwards
/// (`card`, `iban`, `ssn`, `ip`) and must be tight where nothing does
/// (`email`, `secret`, `phone`).
static CANDIDATES: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(&PATTERNS.join("|")).expect("PII candidate patterns must compile"));

/// In [`KINDS`] order; joined with `|` into [`CANDIDATES`].
static PATTERNS: [&str; 7] = [
    // An address whose domain has a dot and a plausible TLD.
    r"(?P<email>[A-Za-z0-9._%+\-]+@[A-Za-z0-9][A-Za-z0-9.\-]*\.[A-Za-z]{2,24})",
    // Only issuer-prefixed shapes, never "a long random-looking string":
    // a request full of base64 or hashes must not read as full of keys.
    // OpenAI, GitHub (token and fine-grained PAT), Slack, AWS, Google,
    // a PEM private key header, and a JWT (both segments start `eyJ`,
    // which is `{"` in base64 — not something random data produces).
    concat!(
        r"(?P<secret>sk-[A-Za-z0-9_\-]{20,}",
        r"|(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}",
        r"|github_pat_[A-Za-z0-9_]{22,}",
        r"|xox[abprs]-[A-Za-z0-9\-]{10,}",
        r"|AKIA[0-9A-Z]{16}",
        r"|AIza[0-9A-Za-z_\-]{35}",
        r"|-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"|eyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,})",
    ),
    // Country, check digits, then 11-30 more, optionally spaced as
    // people write them. mod-97 rules out the uppercase-hex runs that
    // otherwise fit this shape.
    r"(?P<iban>\b[A-Z]{2}[0-9]{2}(?:[ ]?[A-Z0-9]){11,30}\b)",
    // Dashed only. A bare nine-digit run is any number at all.
    r"(?P<ssn>\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b)",
    // 13-19 digits, optionally grouped; Luhn decides.
    r"(?P<card>\b[0-9](?:[ \-]?[0-9]){12,18}\b)",
    // Either an international number, or a separated 3-3-4. Separators
    // are required: ten bare digits are not evidence of anything.
    concat!(
        r"(?P<phone>\+[0-9][0-9 \-.()]{7,18}[0-9]",
        r"|\([0-9]{3}\)[ \-.]?[0-9]{3}[ \-.]?[0-9]{4}",
        r"|\b[0-9]{3}[ \-.][0-9]{3}[ \-.][0-9]{4}\b)",
    ),
    // Dotted quad; the octets and the range are checked after.
    r"(?P<ip>\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b)",
];

// ---------------------------------------------------------------------------
// Scanning
// ---------------------------------------------------------------------------

/// One confirmed identifier, as byte offsets into the text it was found
/// in. Offsets rather than a copy of the value: nothing here should hold
/// on to the personal data it detects, and a redacting caller needs the
/// span anyway.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub kind: Kind,
}

/// Every confirmed identifier in `text`, in order, including repeats.
pub fn scan(text: &str) -> Vec<Span> {
    CANDIDATES
        .captures_iter(text)
        .filter_map(|caps| {
            let kind = KINDS
                .iter()
                .copied()
                .find(|k| caps.name(k.group()).is_some())?;
            let m = caps.name(kind.group())?;
            confirmed(kind, m.as_str()).then_some(Span {
                start: m.start(),
                end: m.end(),
                kind,
            })
        })
        .collect()
}

/// What one request body was found to contain: how many of each kind,
/// in the order the kinds were first seen. Counts, not values — this
/// ends up in a log line.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Findings {
    counts: Vec<(Kind, usize)>,
}

impl Findings {
    /// Nothing was confirmed. Note that this is also what an
    /// unscannable body returns, so it means "no reason found to keep
    /// this local", never "no personal data present".
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// The kinds found, first-seen order.
    pub fn kinds(&self) -> impl Iterator<Item = Kind> + '_ {
        self.counts.iter().map(|(kind, _)| *kind)
    }

    /// Readable for the routing log: `2 email addresses, 1 IBAN`. Empty
    /// for empty findings, which no caller logs.
    pub fn summary(&self) -> String {
        self.counts
            .iter()
            .map(|(kind, n)| {
                let name = if *n == 1 {
                    kind.as_str()
                } else {
                    kind.plural()
                };
                format!("{n} {name}")
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn add(&mut self, spans: &[Span]) {
        for span in spans {
            match self.counts.iter_mut().find(|(kind, _)| *kind == span.kind) {
                Some((_, n)) => *n += 1,
                None => self.counts.push((span.kind, 1)),
            }
        }
    }
}

/// Most text ever scanned for one request. Bounds the worst case rather
/// than any real request: inline media is skipped by [`is_blob`] below
/// and prose this long does not fit a context window anyway. A body
/// past it is scanned up to here and routed on what was found, since
/// refusing to route is not an option this far into a request.
const SCAN_LIMIT: usize = 8 << 20;

/// Longest string still treated as prose. Past this, a value with no
/// whitespace anywhere is an encoded blob — an inline image, a data
/// URL, an attached file — where a scan finds nothing and costs the
/// most.
const BLOB_LEN: usize = 4096;

fn is_blob(text: &str) -> bool {
    text.len() > BLOB_LEN && !text.bytes().any(|b| b.is_ascii_whitespace())
}

/// Every string value anywhere in a request body: message turns, the
/// system prompt, tool results and arguments alike. Keys are not
/// scanned — they are the wire schema, not the user's content.
///
/// Shape-agnostic on purpose. `/api/chat`, `/v1/chat/completions`,
/// `/v1/responses` and `/v1/messages` all nest their text differently
/// and grow new fields every release; walking the whole document means
/// a field this daemon has never heard of is still scanned.
pub fn scan_request(req: &serde_json::Value) -> Findings {
    let mut findings = Findings::default();
    let mut budget = SCAN_LIMIT;
    walk(req, &mut findings, &mut budget);
    findings
}

fn walk(value: &serde_json::Value, findings: &mut Findings, budget: &mut usize) {
    match value {
        serde_json::Value::String(s) => scan_text(s, findings, budget),
        serde_json::Value::Array(items) => {
            for item in items {
                walk(item, findings, budget);
            }
        }
        serde_json::Value::Object(fields) => {
            for (_, field) in fields {
                walk(field, findings, budget);
            }
        }
        _ => {}
    }
}

/// [`scan_request`] for text a caller already has out of its own typed
/// request (`/api/embed`'s inputs), rather than as a JSON document.
pub fn scan_texts<'a>(texts: impl IntoIterator<Item = &'a str>) -> Findings {
    let mut findings = Findings::default();
    let mut budget = SCAN_LIMIT;
    for text in texts {
        scan_text(text, &mut findings, &mut budget);
    }
    findings
}

fn scan_text(text: &str, findings: &mut Findings, budget: &mut usize) {
    if *budget == 0 || is_blob(text) {
        return;
    }
    *budget = budget.saturating_sub(text.len());
    findings.add(&scan(text));
}

// ---------------------------------------------------------------------------
// Confirmation
// ---------------------------------------------------------------------------

/// Whether a candidate match is really an identifier of its kind. The
/// four checkable kinds are checked; the three whose pattern is already
/// the whole evidence are taken as they are.
fn confirmed(kind: Kind, text: &str) -> bool {
    match kind {
        Kind::Email | Kind::Secret | Kind::Phone => true,
        Kind::CreditCard => luhn(text),
        Kind::Iban => iban(text),
        Kind::NationalId => ssn(text),
        Kind::IpAddress => public_ipv4(text),
    }
}

/// The Luhn check digit, over 13-19 digits. Roughly one random number
/// of that length in ten passes, which is the false-positive rate this
/// kind carries and the direction it should err in.
fn luhn(text: &str) -> bool {
    let digits: Vec<u32> = text.chars().filter_map(|c| c.to_digit(10)).collect();
    if !(13..=19).contains(&digits.len()) {
        return false;
    }
    let sum: u32 = digits
        .iter()
        .rev()
        .enumerate()
        .map(|(i, d)| match i % 2 {
            1 if *d > 4 => d * 2 - 9,
            1 => d * 2,
            _ => *d,
        })
        .sum();
    sum.is_multiple_of(10)
}

/// ISO 13616's mod-97: move the first four characters to the end, read
/// letters as 10-35, and the whole number must be 1 modulo 97. Folded
/// digit by digit so no big integer is needed.
fn iban(text: &str) -> bool {
    let compact: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    if !(15..=34).contains(&compact.len()) {
        return false;
    }
    let mut remainder: u32 = 0;
    for c in compact[4..].iter().chain(compact[..4].iter()) {
        let value = match c {
            '0'..='9' => c.to_digit(10).unwrap_or_default(),
            'A'..='Z' => *c as u32 - 'A' as u32 + 10,
            _ => return false,
        };
        // One or two decimal digits per character, hence the two moduli.
        remainder = if value > 9 {
            (remainder * 100 + value) % 97
        } else {
            (remainder * 10 + value) % 97
        };
    }
    remainder == 1
}

/// The SSA's own never-issued ranges, so the `000-00-0000` and
/// `123-45-6789` shaped placeholders that fill documentation and test
/// fixtures do not route real requests.
fn ssn(text: &str) -> bool {
    let mut parts = text.split('-');
    let (Some(area), Some(group), Some(serial)) = (parts.next(), parts.next(), parts.next()) else {
        return false;
    };
    area != "000" && area != "666" && !area.starts_with('9') && group != "00" && serial != "0000"
}

/// A routable IPv4 address. Everything a machine says about itself —
/// loopback, private ranges, link-local, the documentation blocks — is
/// not a person and appears in half of all agent traffic.
fn public_ipv4(text: &str) -> bool {
    let Ok(ip) = text.parse::<std::net::Ipv4Addr>() else {
        return false;
    };
    let first = ip.octets()[0];
    !ip.is_private()
        && !ip.is_loopback()
        && !ip.is_link_local()
        && !ip.is_multicast()
        && !ip.is_broadcast()
        && !ip.is_documentation()
        && !ip.is_unspecified()
        // 0.0.0.0/8 ("this network") and 240.0.0.0/4 (reserved), which
        // std has no stable predicate for.
        && first != 0
        && first < 240
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

/// Whether the privacy gate is on, from `LLMMAN_HYBRID_PII`: `off`
/// (`0`, `false`, `no`) turns it off, anything else leaves it on.
/// On by default, and an unreadable value leaves it on, because the
/// gate only ever keeps a request here — the failure it prevents is
/// unrecoverable and the failure it causes is a worse answer.
pub fn gate_enabled(value: Option<&str>) -> bool {
    !matches!(
        value
            .map(str::trim)
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "off" | "0" | "false" | "no" | "none" | "disabled"
    )
}

/// [`gate_enabled`] against this process's own environment.
pub fn gate_enabled_from_env() -> bool {
    gate_enabled(std::env::var("LLMMAN_HYBRID_PII").ok().as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(text: &str) -> Vec<Kind> {
        scan(text).into_iter().map(|s| s.kind).collect()
    }

    /// The whole point of scanning rather than sampling: an identifier
    /// repeated across turns is detected every time it appears, not
    /// once.
    #[test]
    fn every_mention_of_a_repeated_identifier_is_found() {
        let conversation = "Book it for ada@example.com.\n\
             Confirmed, ada@example.com is on the list.\n\
             Use ada@example.com for the receipt too.";
        let spans = scan(conversation);
        assert_eq!(spans.len(), 3);
        assert!(spans.iter().all(|s| s.kind == Kind::Email));
        // Distinct positions, in order.
        assert!(spans.windows(2).all(|w| w[0].end <= w[1].start));
        for span in &spans {
            assert_eq!(&conversation[span.start..span.end], "ada@example.com");
        }
    }

    #[test]
    fn an_email_address_is_detected() {
        assert_eq!(
            kinds("write to ada.lovelace+tag@example.co.uk"),
            [Kind::Email]
        );
        assert_eq!(kinds("no address here, just an @ sign"), []);
        assert_eq!(kinds("@example.com"), []);
    }

    /// Only Luhn-valid runs of 13-19 digits, so an order number or a
    /// timestamp is not a card.
    #[test]
    fn a_card_number_is_confirmed_by_luhn() {
        assert_eq!(
            kinds("card 4111 1111 1111 1111 exp 12/28"),
            [Kind::CreditCard]
        );
        assert_eq!(kinds("4111-1111-1111-1111"), [Kind::CreditCard]);
        assert_eq!(kinds("378282246310005"), [Kind::CreditCard]);
        // Same shape, wrong check digit.
        assert_eq!(kinds("4111 1111 1111 1112"), []);
        // Too short to be a card at all.
        assert_eq!(kinds("order 123456789012"), []);
    }

    #[test]
    fn an_iban_is_confirmed_by_mod_97() {
        assert_eq!(kinds("IBAN GB82 WEST 1234 5698 7654 32"), [Kind::Iban]);
        assert_eq!(kinds("DE89370400440532013000"), [Kind::Iban]);
        // One digit changed: the checksum fails and nothing is reported.
        assert_eq!(kinds("DE89370400440532013001"), []);
        // An uppercase hex run of the same shape.
        assert_eq!(kinds("AB12CDEF0123456789ABCD"), []);
    }

    #[test]
    fn a_national_id_needs_an_issuable_range() {
        assert_eq!(kinds("SSN 123-45-6789"), [Kind::NationalId]);
        for never_issued in [
            "000-45-6789",
            "666-45-6789",
            "900-45-6789",
            "123-00-6789",
            "123-45-0000",
        ] {
            assert_eq!(kinds(never_issued), [], "{never_issued} is never issued");
        }
        // Undashed, it is just a number.
        assert_eq!(kinds("123456789"), []);
    }

    #[test]
    fn a_phone_number_needs_its_separators() {
        assert_eq!(kinds("call +1 415 555 2671"), [Kind::Phone]);
        assert_eq!(kinds("(415) 555-2671"), [Kind::Phone]);
        assert_eq!(kinds("415-555-2671"), [Kind::Phone]);
        // Ten bare digits are not evidence of anything.
        assert_eq!(kinds("4155552671"), []);
    }

    /// A machine's own addresses are in half of all agent traffic and
    /// identify nobody; flagging them would keep every request local.
    #[test]
    fn only_a_routable_ip_address_counts() {
        assert_eq!(kinds("connect to 203.0.113.1"), []); // documentation range
        assert_eq!(kinds("seen from 8.8.4.4"), [Kind::IpAddress]);
        for local in [
            "127.0.0.1:17434",
            "192.168.1.20",
            "10.0.0.7",
            "172.16.3.4",
            "169.254.1.1",
            "0.0.0.0",
            "255.255.255.255",
        ] {
            assert_eq!(kinds(local), [], "{local} is not personal data");
        }
        assert_eq!(kinds("999.1.1.1"), []);
    }

    /// Prefixed shapes only: a body full of hashes or base64 must not
    /// read as a body full of keys.
    #[test]
    fn a_credential_is_detected_by_its_issuer_prefix() {
        for secret in [
            "sk-abcdefghijklmnopqrstuvwxyz012345",
            "ghp_012345678901234567890123456789012345",
            "AKIAIOSFODNN7EXAMPLE",
            "-----BEGIN RSA PRIVATE KEY-----",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk",
        ] {
            assert_eq!(kinds(secret), [Kind::Secret], "{secret} is a credential");
        }
        // A sha256 and a plain base64 blob are neither.
        assert_eq!(
            kinds("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
            []
        );
        assert_eq!(kinds("dGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVy"), []);
    }

    #[test]
    fn ordinary_prose_and_code_are_left_alone() {
        for clean in [
            "Summarise the last three commits and open a PR.",
            "fn main() { println!(\"{}\", 2 + 2); }",
            "Released v1.2.3 on 2026-01-30, 14:05:00, exit code 0.",
            "SELECT id, total FROM orders WHERE total > 1000;",
        ] {
            assert_eq!(kinds(clean), [], "{clean} has nothing in it");
        }
    }

    // -- Request bodies -----------------------------------------------------

    /// Every shape at once: the walk does not know which API it is
    /// looking at, so a nested tool result counts the same as a turn.
    #[test]
    fn a_request_is_scanned_wherever_its_text_sits() {
        let req = serde_json::json!({
            "model": "llmman.hybrid/gemma4,anthropic/claude-sonnet-4-5",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [{"type": "text", "text": "mail ada@example.com"}]},
                {"role": "tool", "content": "{\"card\": \"4111 1111 1111 1111\"}"},
            ],
        });
        let findings = scan_request(&req);
        assert!(!findings.is_empty());
        assert_eq!(
            findings.kinds().collect::<Vec<_>>(),
            [Kind::Email, Kind::CreditCard]
        );
        assert_eq!(findings.summary(), "1 email address, 1 payment card number");
    }

    #[test]
    fn a_clean_request_finds_nothing() {
        let req = serde_json::json!({
            "model": "gemma4",
            "messages": [{"role": "user", "content": "what is 2 + 2?"}],
        });
        assert!(scan_request(&req).is_empty());
        assert_eq!(scan_request(&serde_json::json!(null)), Findings::default());
    }

    /// Field names are the wire schema, not the user's content.
    #[test]
    fn object_keys_are_not_scanned() {
        let req = serde_json::json!({ "ada@example.com": "hello" });
        assert!(scan_request(&req).is_empty());
    }

    /// An inline image is where the bytes are and where the identifiers
    /// are not.
    #[test]
    fn an_encoded_blob_is_skipped() {
        let blob = "A".repeat(BLOB_LEN) + "ada@example.com";
        assert!(is_blob(&blob));
        assert!(scan_request(&serde_json::json!({ "images": [blob] })).is_empty());
        // The same length with whitespace in it is prose, and is read.
        let prose = "word ".repeat(BLOB_LEN) + "ada@example.com";
        assert!(!is_blob(&prose));
        assert!(!scan_request(&serde_json::json!({ "prompt": prose })).is_empty());
    }

    #[test]
    fn counts_accumulate_across_turns_and_read_as_english() {
        let req = serde_json::json!({
            "messages": [
                {"content": "ada@example.com"},
                {"content": "ada@example.com and grace@example.com"},
                {"content": "GB82 WEST 1234 5698 7654 32"},
            ],
        });
        let findings = scan_request(&req);
        assert_eq!(findings.summary(), "3 email addresses, 1 IBAN");
    }

    #[test]
    fn scan_texts_reads_a_typed_requests_own_strings() {
        let findings = scan_texts(["nothing here", "reach me at ada@example.com"]);
        assert_eq!(findings.summary(), "1 email address");
        assert!(scan_texts(std::iter::empty()).is_empty());
    }

    // -- Span invariants ----------------------------------------------------

    /// The oracle `fuzz/fuzz_targets/pii_scan.rs` runs on arbitrary
    /// input, pinned here against the seed corpus so a change that
    /// breaks it fails in `cargo test` too. A span that is not a real,
    /// in-bounds, non-overlapping character range would make a
    /// redacting caller cut the wrong bytes, or panic.
    #[test]
    fn spans_are_ordered_in_bounds_ranges_of_the_text_scanned() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fuzz/corpus/pii_scan");
        let mut seeds = 0usize;
        for entry in std::fs::read_dir(&dir).expect("read the seed corpus directory") {
            let path = entry.expect("read a corpus directory entry").path();
            let data = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
            seeds += 1;
            let Ok(text) = std::str::from_utf8(&data) else {
                continue;
            };
            let mut previous_end = 0;
            for span in scan(text) {
                assert!(span.start < span.end, "{path:?}: empty span {span:?}");
                assert!(span.end <= text.len(), "{path:?}: span past the end");
                assert!(span.start >= previous_end, "{path:?}: overlapping spans");
                // Panics unless both ends are character boundaries.
                let _ = &text[span.start..span.end];
                previous_end = span.end;
            }
        }
        assert!(seeds > 0, "seed corpus at {dir:?} is empty");
        // Multi-byte text: offsets are bytes, and must still land on
        // boundaries either side of a match.
        let text = "写信给 ada@example.com 谢谢";
        let spans = scan(text);
        assert_eq!(spans.len(), 1);
        assert_eq!(&text[spans[0].start..spans[0].end], "ada@example.com");
    }

    // -- The gate -----------------------------------------------------------

    #[test]
    fn the_gate_is_on_unless_it_is_turned_off() {
        assert!(gate_enabled(None));
        assert!(gate_enabled(Some("")));
        assert!(gate_enabled(Some("local")));
        assert!(gate_enabled(Some("on")));
        // Unreadable is not off: the gate only ever keeps data here.
        assert!(gate_enabled(Some("maybe")));
        for off in [
            "off", "OFF", " off ", "0", "false", "no", "none", "disabled",
        ] {
            assert!(!gate_enabled(Some(off)), "{off} must turn the gate off");
        }
    }
}
