//! The System One request: parsing, validation and the prompt wording,
//! following sglang's `/v1/systemone`. Objects keep their key order: it is
//! part of the question (the first option is labelled `A`) and of the prompt.

use std::collections::HashMap;
use std::fmt;

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};
use serde_json::{Map, Number};

pub const MAX_QUESTIONS: usize = 64;
/// Past 26 options, labels are two letters.
pub const MAX_OPTIONS: usize = 255;
/// Levels are labelled `0` to `9`.
pub const MAX_LEVELS: usize = 10;

/// JSON whose objects keep their key order.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(Number),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(o) => o.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn f64(v: f64) -> Json {
        Number::from_f64(v).map_or(Json::Null, Json::Num)
    }

    /// Python's `json.dumps(v, ensure_ascii=False, separators=(",", ":"))`.
    pub fn compact(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

impl From<&str> for Json {
    fn from(s: &str) -> Json {
        Json::Str(s.into())
    }
}

impl Serialize for Json {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Json::Null => s.serialize_unit(),
            Json::Bool(b) => s.serialize_bool(*b),
            Json::Num(n) => n.serialize(s),
            Json::Str(v) => s.serialize_str(v),
            Json::Arr(a) => {
                let mut seq = s.serialize_seq(Some(a.len()))?;
                a.iter().try_for_each(|v| seq.serialize_element(v))?;
                seq.end()
            }
            Json::Obj(o) => {
                let mut map = s.serialize_map(Some(o.len()))?;
                o.iter().try_for_each(|(k, v)| map.serialize_entry(k, v))?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Json, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Json;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_unit<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_bool<E>(self, v: bool) -> Result<Json, E> {
                Ok(Json::Bool(v))
            }
            fn visit_i64<E>(self, v: i64) -> Result<Json, E> {
                Ok(Json::Num(v.into()))
            }
            fn visit_u64<E>(self, v: u64) -> Result<Json, E> {
                Ok(Json::Num(v.into()))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> Result<Json, E> {
                Number::from_f64(v)
                    .map(Json::Num)
                    .ok_or_else(|| E::custom("a number JSON cannot hold"))
            }
            fn visit_str<E>(self, v: &str) -> Result<Json, E> {
                Ok(Json::Str(v.into()))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
                let mut v = Vec::new();
                while let Some(e) = seq.next_element()? {
                    v.push(e);
                }
                Ok(Json::Arr(v))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
                let mut o: Vec<(String, Json)> = Vec::new();
                let mut at: HashMap<String, usize> = HashMap::new();
                while let Some((k, v)) = map.next_entry::<String, Json>()? {
                    // a repeated key keeps its first place and last value, as in Python
                    match at.get(&k) {
                        Some(&i) => o[i].1 = v,
                        None => {
                            at.insert(k.clone(), o.len());
                            o.push((k, v));
                        }
                    }
                }
                Ok(Json::Obj(o))
            }
        }
        d.deserialize_any(V)
    }
}

/// One refusal, in FastAPI's `detail` shape: `loc` has integers for list indices.
#[derive(Debug, PartialEq)]
pub struct Problem {
    pub kind: &'static str,
    pub loc: Vec<serde_json::Value>,
    pub msg: String,
}

impl Problem {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({"type": self.kind, "loc": self.loc, "msg": self.msg})
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Noul,
    Choice,
    Score,
}

#[derive(Debug, Clone)]
pub struct Question {
    pub kind: Kind,
    /// `instructions`, if any.
    pub text: Option<Json>,
    /// Option names, level indices, or `yes` and `no`.
    pub names: Vec<String>,
    /// Option descriptions, levels, or the `true` and `false` descriptions.
    pub details: Vec<Option<Json>>,
}

#[derive(Debug)]
pub struct Request {
    pub state: Json,
    pub model: String,
    pub questions: Vec<(String, Question)>,
    /// llmman extension: chat template variables, over `enable_thinking: false`.
    pub chat_template_kwargs: Map<String, serde_json::Value>,
}

/// Any other key in a question would answer a different question.
const QUESTION_KEYS: [&str; 3] = ["type", "instructions", "criteria"];
/// `/v1/decisions` fields that would change the answers if honoured or ignored.
const REFUSED: [&str; 3] = [
    "temperature",
    "prompt_format_version",
    "return_prompt_token_ids",
];
const NEEDS_OBJECT: &str = "Input should be a valid dictionary or object to extract fields from";
const NEEDS_TEXT: &str = "Input should be a valid string, object or array";
const EXTRA: &str = "Extra inputs are not permitted";

/// The problems found so far, each located under `at`.
struct Problems {
    found: Vec<Problem>,
    at: Vec<String>,
}

impl Problems {
    fn add(&mut self, kind: &'static str, rest: &[&str], msg: impl Into<String>) {
        let loc = self
            .at
            .iter()
            .map(String::as_str)
            .chain(rest.iter().copied());
        self.found.push(Problem {
            kind,
            loc: loc.map(Into::into).collect(),
            msg: msg.into(),
        });
    }

    /// A problem with level `i` of `criteria`.
    fn add_level(&mut self, kind: &'static str, i: usize, msg: &str) {
        self.add(kind, &["criteria"], msg);
        if let Some(last) = self.found.last_mut() {
            last.loc.push(i.into());
        }
    }

    /// `criteria` is of the wrong type, or missing.
    fn criteria(&mut self, given: Option<&Json>, kind: &'static str, msg: &str) {
        match given {
            None => self.add("missing", &["criteria"], "Field required"),
            Some(_) => self.add(kind, &["criteria"], msg),
        }
    }

    /// Whether `n` is 1 to `max`; a problem if not.
    fn in_range(&mut self, n: usize, max: usize, msg: String) -> bool {
        let fits = (1..=max).contains(&n);
        if !fits {
            self.add(
                if n == 0 { "too_short" } else { "too_long" },
                &["criteria"],
                msg,
            );
        }
        fits
    }
}

fn is_blank(v: &Json) -> bool {
    match v {
        Json::Str(s) => s.trim().is_empty(),
        Json::Arr(a) => a.is_empty(),
        Json::Obj(o) => o.is_empty(),
        _ => false,
    }
}

/// Text, an object or an array (rendered into the prompt), or null.
fn text_field(v: Option<&Json>, loc: &[&str], p: &mut Problems) -> Option<Json> {
    match v {
        None | Some(Json::Null) => None,
        Some(v @ (Json::Str(_) | Json::Arr(_) | Json::Obj(_))) => Some(v.clone()),
        Some(_) => {
            p.add("string_type", loc, NEEDS_TEXT);
            None
        }
    }
}

/// Option names that would make the rendered option lines ambiguous.
fn check_option_names(names: &[&String]) -> Result<(), String> {
    let mut seen = std::collections::HashSet::new();
    for name in names {
        let key = name.trim().to_lowercase();
        if key.is_empty() {
            return Err("option names must be nonempty".into());
        }
        if name
            .chars()
            .any(|c| c.is_control() || c == '\u{2028}' || c == '\u{2029}')
        {
            return Err(format!(
                "option name {name:?} must not contain control or line break characters"
            ));
        }
        if !seen.insert(key) {
            return Err(format!("option name {name:?} repeats another option"));
        }
    }
    Ok(())
}

/// Names and details of a `noul` question: `yes` and `no`, and what makes each true.
fn noul(text: &Option<Json>, criteria: Option<&Json>, p: &mut Problems) -> Option<Details> {
    let (mut yes, mut no) = (None, None);
    match criteria {
        None => {}
        Some(c @ Json::Obj(o)) => {
            for (k, _) in o.iter().filter(|(k, _)| k != "true" && k != "false") {
                p.add("extra_forbidden", &["criteria", k], EXTRA);
            }
            yes = text_field(c.get("true"), &["criteria", "true"], p);
            no = text_field(c.get("false"), &["criteria", "false"], p);
        }
        Some(_) => p.add("model_attributes_type", &["criteria"], NEEDS_OBJECT),
    }
    if ![text, &yes, &no]
        .iter()
        .any(|t| t.as_ref().is_some_and(|t| !is_blank(t)))
    {
        p.add(
            "value_error",
            &[],
            "Value error, a noul question needs instructions or a true or false \
             description to decide on",
        );
        return None;
    }
    Some((vec!["yes".into(), "no".into()], vec![yes, no]))
}

fn choice(criteria: Option<&Json>, p: &mut Problems) -> Option<Details> {
    let Some(Json::Obj(options)) = criteria else {
        p.criteria(criteria, "dict_type", "Input should be a valid dictionary");
        return None;
    };
    let msg = format!("a choice question takes 1 to {MAX_OPTIONS} options");
    let names: Vec<&String> = options.iter().map(|(k, _)| k).collect();
    if !p.in_range(options.len(), MAX_OPTIONS, msg) {
        return None;
    }
    if let Err(e) = check_option_names(&names) {
        p.add("value_error", &["criteria"], format!("Value error, {e}"));
        return None;
    }
    let details = options
        .iter()
        .map(|(k, v)| text_field(Some(v), &["criteria", k], p))
        .collect();
    Some((names.into_iter().cloned().collect(), details))
}

fn score(criteria: Option<&Json>, p: &mut Problems) -> Option<Details> {
    let Some(Json::Arr(levels)) = criteria else {
        p.criteria(criteria, "list_type", "Input should be a valid list");
        return None;
    };
    let msg = format!("a score question takes 1 to {MAX_LEVELS} levels");
    if !p.in_range(levels.len(), MAX_LEVELS, msg) {
        return None;
    }
    let mut ok = true;
    for (i, level) in levels.iter().enumerate() {
        let (kind, msg) = match level {
            Json::Str(_) | Json::Arr(_) | Json::Obj(_) if !is_blank(level) => continue,
            Json::Str(_) | Json::Arr(_) | Json::Obj(_) => {
                ("value_error", "Value error, must not be blank")
            }
            _ => ("string_type", NEEDS_TEXT),
        };
        p.add_level(kind, i, msg);
        ok = false;
    }
    ok.then(|| {
        (
            (0..levels.len()).map(|i| i.to_string()).collect(),
            levels.iter().cloned().map(Some).collect(),
        )
    })
}

/// A question's names and details.
type Details = (Vec<String>, Vec<Option<Json>>);

/// Validates one question; `p.at` locates it.
fn question(v: &Json, p: &mut Problems) -> Option<Question> {
    let Json::Obj(fields) = v else {
        p.add("model_attributes_type", &[], NEEDS_OBJECT);
        return None;
    };
    for (k, _) in fields
        .iter()
        .filter(|(k, _)| !QUESTION_KEYS.contains(&k.as_str()))
    {
        p.add("extra_forbidden", &[k], EXTRA);
    }
    let text = text_field(v.get("instructions"), &["instructions"], p);
    let criteria = v.get("criteria");
    let kind = match v.get("type") {
        Some(Json::Str(t)) if t == "noul" => Kind::Noul,
        Some(Json::Str(t)) if t == "choice" => Kind::Choice,
        Some(Json::Str(t)) if t == "score" => Kind::Score,
        Some(other) => {
            let shown = match other {
                Json::Str(s) => s.clone(),
                o => o.compact(),
            };
            p.add(
                "union_tag_invalid",
                &[],
                format!(
                    "Input tag '{shown}' found using 'type' does not match any of the \
                     expected tags: 'noul', 'choice', 'score'"
                ),
            );
            return None;
        }
        None => {
            p.add(
                "union_tag_not_found",
                &[],
                "Unable to extract tag using discriminator 'type'",
            );
            return None;
        }
    };
    let (names, details) = match kind {
        // only a noul question's criteria are optional, so null is none
        Kind::Noul => noul(&text, criteria.filter(|c| **c != Json::Null), p)?,
        Kind::Choice => choice(criteria, p)?,
        Kind::Score => score(criteria, p)?,
    };
    Some(Question {
        kind,
        text,
        names,
        details,
    })
}

/// Parses and validates a request body; `Err` holds every problem found.
pub fn parse(body: &[u8]) -> Result<Request, Vec<Problem>> {
    let mut p = Problems {
        found: Vec::new(),
        at: vec!["body".into()],
    };
    let root: Json = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            p.add("json_invalid", &[], format!("JSON decode error: {e}"));
            return Err(p.found);
        }
    };
    if !matches!(root, Json::Obj(_)) {
        p.add("model_attributes_type", &[], NEEDS_OBJECT);
        return Err(p.found);
    }
    match root.get("state") {
        None => p.add("missing", &["state"], "Field required"),
        Some(Json::Str(_) | Json::Arr(_) | Json::Obj(_)) => {}
        Some(_) => p.add("string_type", &["state"], NEEDS_TEXT),
    }
    let model = match root.get("model") {
        Some(Json::Str(m)) => Some(m.clone()),
        None => {
            p.add("missing", &["model"], "Field required");
            None
        }
        Some(_) => {
            p.add("string_type", &["model"], "Input should be a valid string");
            None
        }
    };
    let mut questions = Vec::new();
    match root.get("questions") {
        None => p.add("missing", &["questions"], "Field required"),
        Some(Json::Obj(qs)) if qs.is_empty() => p.add(
            "too_short",
            &["questions"],
            "a request needs at least one question",
        ),
        Some(Json::Obj(qs)) if qs.len() > MAX_QUESTIONS => p.add(
            "too_long",
            &["questions"],
            format!("a request takes at most {MAX_QUESTIONS} questions"),
        ),
        Some(Json::Obj(qs)) => {
            for (id, q) in qs {
                p.at.extend(["questions".into(), id.clone()]);
                questions.extend(question(q, &mut p).map(|q| (id.clone(), q)));
                p.at.truncate(1);
            }
        }
        Some(_) => p.add(
            "dict_type",
            &["questions"],
            "Input should be a valid dictionary",
        ),
    }
    for field in REFUSED {
        if root.get(field).is_some_and(|v| *v != Json::Null) {
            p.add(
                "value_error",
                &[field],
                format!("Value error, {field} is not part of this API"),
            );
        }
    }
    let kwargs = match root.get("chat_template_kwargs") {
        None | Some(Json::Null) => Map::new(),
        Some(k @ Json::Obj(_)) => match serde_json::to_value(k) {
            Ok(serde_json::Value::Object(m)) => m,
            _ => Map::new(),
        },
        Some(_) => {
            p.add(
                "dict_type",
                &["chat_template_kwargs"],
                "Input should be a valid dictionary",
            );
            Map::new()
        }
    };
    match (root.get("state"), model) {
        (Some(state), Some(model)) if p.found.is_empty() => Ok(Request {
            state: state.clone(),
            model,
            questions,
            chat_template_kwargs: kwargs,
        }),
        _ => Err(p.found),
    }
}

/// A value as it is written into the prompt: text as is, anything else as compact JSON.
pub fn render_text(v: Option<&Json>) -> String {
    match v {
        None => String::new(),
        Some(Json::Str(s)) => s.clone(),
        Some(v) => v.compact(),
    }
}

/// The prompt wording of one question, with the labels its options go by.
pub fn render_question(state: &str, q: &Question, labels: &[String]) -> String {
    let text = q
        .text
        .as_ref()
        .filter(|t| !is_blank(t))
        .map(|t| render_text(Some(t)))
        .unwrap_or_default();
    let question = (!text.is_empty()).then(|| format!("Question: {text}"));
    let mut lines = Vec::new();
    match q.kind {
        Kind::Choice => {
            lines.extend(question);
            for ((label, name), detail) in labels.iter().zip(&q.names).zip(&q.details) {
                let detail = render_text(detail.as_ref());
                lines.push(match detail.is_empty() {
                    true => format!("{label}: {name}"),
                    false => format!("{label}: {name} - {detail}"),
                });
            }
            lines.push("Answer with the letter of one option only.".into());
        }
        Kind::Score => {
            lines.extend(question);
            for (label, level) in labels.iter().zip(&q.details) {
                lines.push(format!("{label}: {}", render_text(level.as_ref())));
            }
            lines.push("Answer with the number of one level only.".into());
        }
        Kind::Noul => {
            lines.push(match text.is_empty() {
                true => "Is the following true?".into(),
                false => format!("Is the following true? {text}"),
            });
            for (label, detail) in labels.iter().zip(&q.details) {
                let detail = render_text(detail.as_ref());
                if !detail.is_empty() {
                    lines.push(format!("{label}: {detail}"));
                }
            }
            lines.push("Answer with yes or no only.".into());
        }
    }
    format!("{state}\n\n{}", lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(body: &str) -> Request {
        parse(body.as_bytes()).unwrap_or_else(|e| panic!("{e:?}"))
    }

    fn refused(body: &str) -> Vec<Problem> {
        parse(body.as_bytes()).expect_err("should be refused")
    }

    #[test]
    fn a_request_keeps_the_order_the_client_sent() {
        let r = ok(
            r#"{"state": {"z": 1, "a": [2, {"y": 3, "b": 4}]}, "model": "m",
            "questions": {"second": {"type": "choice", "criteria": {"zebra": null, "apple": "x"}},
                          "first": {"type": "noul", "instructions": "q"}}}"#,
        );
        assert_eq!(r.state.compact(), r#"{"z":1,"a":[2,{"y":3,"b":4}]}"#);
        let ids: Vec<_> = r.questions.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, ["second", "first"]);
        assert_eq!(r.questions[0].1.names, ["zebra", "apple"]);
    }

    #[test]
    fn a_repeated_key_keeps_its_first_place_and_last_value() {
        let j: Json = serde_json::from_str(r#"{"a": 1, "b": 2, "a": 3}"#).unwrap();
        assert_eq!(j.compact(), r#"{"a":3,"b":2}"#);
    }

    #[test]
    fn a_large_object_parses_in_linear_time() {
        let keys: Vec<String> = (0..200_000).map(|i| format!("\"k{i}\":1")).collect();
        let body = format!("{{{}}}", keys.join(","));
        let started = std::time::Instant::now();
        let j: Json = serde_json::from_str(&body).unwrap();
        assert!(matches!(&j, Json::Obj(o) if o.len() == 200_000));
        // a scan of the keys seen so far would take minutes
        assert!(started.elapsed().as_secs() < 5, "{:?}", started.elapsed());
    }

    #[test]
    fn compact_json_is_not_ascii_escaped() {
        let j: Json = serde_json::from_str(r#"{"k": "héllo — \"w\"\n"}"#).unwrap();
        assert_eq!(j.compact(), "{\"k\":\"héllo — \\\"w\\\"\\n\"}");
    }

    #[test]
    fn unknown_top_level_fields_are_ignored_but_unknown_question_keys_are_not() {
        ok(r#"{"state": "s", "model": "m", "whatever": 1,
               "questions": {"q": {"type": "noul", "instructions": "i"}}}"#);
        let p = refused(
            r#"{"state": "s", "model": "m",
                "questions": {"q": {"type": "noul", "instructions": "i", "instrucions": "typo"}}}"#,
        );
        assert_eq!(p[0].kind, "extra_forbidden");
        assert_eq!(p[0].loc, ["body", "questions", "q", "instrucions"]);
    }

    #[test]
    fn a_noul_question_needs_something_to_decide_on() {
        let p = refused(
            r#"{"state": "s", "model": "m", "questions": {"q": {"type": "noul", "instructions": "  "}}}"#,
        );
        assert!(p[0].msg.contains("needs instructions"), "{p:?}");
        ok(
            r#"{"state": "s", "model": "m", "questions": {"q": {"type": "noul",
            "criteria": {"true": "it is raining"}}}}"#,
        );
    }

    #[test]
    fn choice_options_must_be_distinct_one_line_names() {
        for (names, want) in [
            (r#"{"a": null, " A ": null}"#, "repeats"),
            (r#"{"": null}"#, "nonempty"),
            (r#"{"two\nlines": null}"#, "control or line break"),
            (r#"{}"#, "1 to 255"),
        ] {
            let body = format!(
                r#"{{"state": "s", "model": "m", "questions": {{"q": {{"type": "choice", "criteria": {names}}}}}}}"#
            );
            let p = refused(&body);
            assert!(p[0].msg.contains(want), "{names}: {p:?}");
        }
    }

    #[test]
    fn a_score_takes_one_to_ten_nonblank_levels() {
        let body = |levels: &str| {
            format!(
                r#"{{"state": "s", "model": "m", "questions": {{"q": {{"type": "score", "criteria": {levels}}}}}}}"#
            )
        };
        ok(&body(r#"["calm", {"label": "angry"}]"#));
        assert!(refused(&body("[]"))[0].msg.contains("1 to 10"));
        assert!(refused(&body(r#"["a", " "]"#))[0].msg.contains("blank"));
        let eleven = format!("[{}]", ["\"x\""; 11].join(","));
        assert!(refused(&body(&eleven))[0].msg.contains("1 to 10"));
    }

    #[test]
    fn null_criteria_is_a_wrong_type_except_for_a_noul_question() {
        let with = |kind: &str| {
            format!(
                r#"{{"state": "s", "model": "m", "questions": {{"q": {{"type": "{kind}",
                    "instructions": "i", "criteria": null}}}}}}"#
            )
        };
        ok(&with("noul"));
        assert_eq!(refused(&with("choice"))[0].kind, "dict_type");
        assert_eq!(refused(&with("score"))[0].kind, "list_type");
    }

    #[test]
    fn a_level_is_located_by_its_index_as_a_number() {
        let p = refused(
            r#"{"state": "s", "model": "m",
                "questions": {"q": {"type": "score", "criteria": ["ok", " ", 5]}}}"#,
        );
        assert_eq!(p[0].loc.last(), Some(&serde_json::json!(1)));
        assert_eq!(
            p[1].to_json()["loc"],
            serde_json::json!(["body", "questions", "q", "criteria", 2])
        );
    }

    #[test]
    fn every_problem_is_reported_with_its_location() {
        let p = refused(r#"{"model": 5, "questions": {"q": {"type": "nope"}, "r": 1}}"#);
        let locs: Vec<_> = p
            .iter()
            .map(|p| {
                p.loc
                    .iter()
                    .map(|c| c.as_str().unwrap())
                    .collect::<Vec<_>>()
                    .join(".")
            })
            .collect();
        assert_eq!(
            locs,
            [
                "body.state",
                "body.model",
                "body.questions.q",
                "body.questions.r"
            ]
        );
        assert_eq!(p[2].kind, "union_tag_invalid");
    }

    #[test]
    fn decisions_fields_are_refused_by_name() {
        let p = refused(
            r#"{"state": "s", "model": "m", "temperature": 0.5,
                "questions": {"q": {"type": "noul", "instructions": "i"}}}"#,
        );
        assert_eq!(p[0].loc, ["body", "temperature"]);
        // null is the same as absent
        ok(r#"{"state": "s", "model": "m", "temperature": null,
               "questions": {"q": {"type": "noul", "instructions": "i"}}}"#);
    }

    #[test]
    fn a_body_that_is_not_an_object_is_refused() {
        assert_eq!(refused("[1]")[0].kind, "model_attributes_type");
        assert_eq!(refused("{nope")[0].kind, "json_invalid");
    }

    #[test]
    fn the_prompt_wording_is_sglangs() {
        let r = ok(r#"{"state": "It broke.", "model": "m", "questions": {
            "team": {"type": "choice", "instructions": "Who?",
                     "criteria": {"billing": null, "technical": "Bugs", "sales": {"k": 1}}},
            "urgent": {"type": "noul", "instructions": "Today?", "criteria": {"true": "yes it is"}},
            "mood": {"type": "score", "criteria": ["Calm", {"label": "Angry"}]}}}"#);
        let labels = |n: &[&str]| n.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let q = |i: usize, l: &[&str]| render_question("It broke.", &r.questions[i].1, &labels(l));
        assert_eq!(
            q(0, &["A", "B", "C"]),
            "It broke.\n\nQuestion: Who?\nA: billing\nB: technical - Bugs\nC: sales - {\"k\":1}\n\
             Answer with the letter of one option only."
        );
        assert_eq!(
            q(1, &["yes", "no"]),
            "It broke.\n\nIs the following true? Today?\nyes: yes it is\nAnswer with yes or no only."
        );
        assert_eq!(
            q(2, &["0", "1"]),
            "It broke.\n\n0: Calm\n1: {\"label\":\"Angry\"}\nAnswer with the number of one level only."
        );
    }
}
