//! `POST /v1/systemone`: the System One decision API. A `state` and typed
//! questions (`noul`, `choice`, `score`) in, a probability per candidate out.
//!
//! A local llama-server model is *read*: each question is one user turn,
//! thinking off, and every answer label is one token at the answer position;
//! its next-token probability, renormalised over the labels, is the answer
//! (`x_source: logprobs`). A hosted model shows no token probabilities, so it
//! is *asked* to state them as JSON, after thinking at the requested variant
//! (`x_source: elicited`). Wording, labels, refusals and formulas follow
//! sglang's `/v1/systemone`.

use std::collections::HashSet;
use std::fmt;
use std::marker::PhantomData;
use std::time::Duration;

use anyhow::anyhow;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use reqwest::Client;
use serde::de::{Deserializer, MapAccess, Visitor};
use serde::ser::{SerializeMap, Serializer};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::openai::local_engine;
use super::refusal::refuse;
use super::relay::proxy;
use super::sched::begin_activity;
use super::{
    ensure_model, provider_catalog, send_chat_completion, AppError, AppState, Engine, RemoteTarget,
    Target,
};
use crate::chat_template::EFFORT_LEVELS;
use crate::providers;
use crate::usage::{Decoder, Dialect, Tokens};

const ROUTE: &str = "/v1/systemone";

/// Limits of the published schema.
const MAX_OPTIONS: usize = 255;
const MAX_LEVELS: usize = 10;
/// Not in the schema: one request is one prefill or one paid call per question.
const MAX_QUESTIONS: usize = 64;
/// Options are labelled `A` to `Z`; sglang goes on with two-letter tokens,
/// llmman refuses rather than answer by unequal labels.
const MAX_LETTER_OPTIONS: usize = 26;

/// `/apply-template` and `/tokenize`: cheap, so a stall is a wedged backend.
const PREPARE_TIMEOUT: Duration = Duration::from_secs(60);
/// One prefill of a context-length prompt, slow on a CPU.
const SCORE_TIMEOUT: Duration = Duration::from_secs(600);
/// Candidates asked of `/completion` beyond the labels, so one a few ranks
/// below the top is still listed.
const TOP_SLACK: usize = 16;
/// Logit added to every label by [`Backend::boosted`].
const LABEL_BOOST: f64 = 100.0;
/// Hosted questions in flight at once.
const HOSTED_CONCURRENCY: usize = 4;

// -- JSON that keeps its key order ------------------------------------------

/// A JSON object in the order sent: `serde_json`'s map sorts, and here the
/// first option of a choice is `A`.
#[derive(Clone, Debug, Default, PartialEq)]
struct Ordered<T>(Vec<(String, T)>);

impl<T> Ordered<T> {
    fn get(&self, key: &str) -> Option<&T> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Ordered<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Entries<T>(PhantomData<T>);
        impl<'de, T: Deserialize<'de>> Visitor<'de> for Entries<T> {
            type Value = Ordered<T>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Ordered<T>, A::Error> {
                let mut entries = Vec::new();
                while let Some(entry) = map.next_entry()? {
                    entries.push(entry);
                }
                Ok(Ordered(entries))
            }
        }
        deserializer.deserialize_map(Entries(PhantomData))
    }
}

impl<T: Serialize> Serialize for Ordered<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (key, value) in &self.0 {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

/// Any JSON value, in the order sent. A state, instruction or description is
/// rendered into the prompt, and a score's levels are echoed, as written.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
enum Node {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<Node>),
    Object(Ordered<Node>),
}

impl Node {
    /// Nothing to read: whitespace, `[]` or `{}`.
    fn is_blank(&self) -> bool {
        match self {
            Node::String(s) => s.trim().is_empty(),
            Node::Array(items) => items.is_empty(),
            Node::Object(entries) => entries.0.is_empty(),
            _ => false,
        }
    }

    /// As shown in a prompt: text as written, an object or array as compact
    /// JSON, nothing as nothing.
    fn render(value: Option<&Node>) -> String {
        match value {
            None | Some(Node::Null) => String::new(),
            Some(Node::String(s)) => s.clone(),
            Some(other) => serde_json::to_string(other).unwrap_or_default(),
        }
    }
}

// -- The request -------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Noul,
    Choice,
    Score,
}

#[derive(Debug)]
struct Question {
    id: String,
    kind: Kind,
    instructions: Option<Node>,
    /// Option names, level indices, or `yes` and `no`, in label order.
    names: Vec<String>,
    /// Option descriptions, the levels, or the `true` and `false` texts.
    details: Vec<Option<Node>>,
}

impl Question {
    /// What the model writes for each candidate: `A`.., a digit, `yes`/`no`.
    fn labels(&self) -> Vec<String> {
        match self.kind {
            Kind::Choice => (b'A'..=b'Z')
                .take(self.names.len())
                .map(|c| char::from(c).to_string())
                .collect(),
            Kind::Noul | Kind::Score => self.names.clone(),
        }
    }
}

#[derive(Debug)]
struct Request {
    state: Node,
    model: String,
    questions: Vec<Question>,
    /// Chat template arguments (sglang's extension), over `enable_thinking: false`.
    template_kwargs: serde_json::Map<String, serde_json::Value>,
    /// The variant: how hard a hosted model thinks (an llmman extension).
    reasoning_effort: Option<String>,
}

/// A variant level: `none` or any effort level.
fn is_effort(level: &str) -> bool {
    level == "none" || EFFORT_LEVELS.contains(&level)
}

/// One entry of FastAPI's 422 `detail` list, which System One documents.
#[derive(Debug, Serialize)]
struct Detail {
    #[serde(rename = "type")]
    kind: &'static str,
    loc: Vec<Loc>,
    msg: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
enum Loc {
    Key(String),
    Index(usize),
}

fn at(path: &[Loc], key: &str) -> Vec<Loc> {
    let mut path = path.to_vec();
    path.push(Loc::Key(key.to_owned()));
    path
}

const NOT_AN_OBJECT: &str = "Input should be a valid dictionary or object to extract fields from";

/// Validation that reports every problem, not the first.
#[derive(Default)]
struct Parser {
    errors: Vec<Detail>,
}

impl Parser {
    fn fail(&mut self, kind: &'static str, path: &[Loc], msg: impl Into<String>) {
        let loc = path.to_vec();
        self.errors.push(Detail {
            kind,
            loc,
            msg: msg.into(),
        });
    }

    fn missing(&mut self, path: &[Loc]) {
        self.fail("missing", path, "Field required");
    }

    fn object<'a>(&mut self, value: &'a Node, path: &[Loc]) -> Option<&'a Ordered<Node>> {
        let Node::Object(fields) = value else {
            self.fail("model_attributes_type", path, NOT_AN_OBJECT);
            return None;
        };
        Some(fields)
    }

    /// A misspelt key would otherwise answer a different question.
    fn only(&mut self, fields: &Ordered<Node>, allowed: &[&str], path: &[Loc]) {
        for (key, _) in &fields.0 {
            if !allowed.contains(&key.as_str()) {
                let msg = "Extra inputs are not permitted";
                self.fail("extra_forbidden", &at(path, key), msg);
            }
        }
    }

    /// `n` items, from 1 to `max`.
    fn size(&mut self, n: usize, max: usize, what: &str, path: &[Loc]) {
        if n == 0 {
            let msg = format!("{what} should have at least 1 item after validation, not 0");
            self.fail("too_short", path, msg);
        } else if n > max {
            let msg = format!("{what} should have at most {max} items after validation, not {n}");
            self.fail("too_long", path, msg);
        }
    }

    /// Text, or an object or array rendered as JSON.
    fn text(&mut self, value: &Node, path: &[Loc]) -> bool {
        let ok = matches!(value, Node::String(_) | Node::Array(_) | Node::Object(_));
        if !ok {
            let msg = "Input should be a valid string, object or array";
            self.fail("string_type", path, msg);
        }
        ok
    }

    fn optional_text(&mut self, value: Option<&Node>, path: &[Loc]) -> Option<Node> {
        match value {
            None | Some(Node::Null) => None,
            Some(value) => self.text(value, path).then(|| value.clone()),
        }
    }

    fn required_text(&mut self, value: &Node, path: &[Loc]) -> Option<Node> {
        if !self.text(value, path) {
            return None;
        }
        if value.is_blank() {
            self.fail("value_error", path, "Value error, must not be blank");
            return None;
        }
        Some(value.clone())
    }

    fn questions(&mut self, map: &Ordered<Node>, path: &[Loc]) -> Vec<Question> {
        let mut seen = HashSet::new();
        let mut questions = Vec::new();
        for (id, value) in &map.0 {
            let here = at(path, id);
            // A JSON parser would keep the last and answer one fewer.
            if seen.insert(id) {
                questions.extend(self.question(id, value, &here));
            } else {
                let msg = "Value error, repeats another question id";
                self.fail("value_error", &here, msg);
            }
        }
        questions
    }

    fn question(&mut self, id: &str, value: &Node, path: &[Loc]) -> Option<Question> {
        let fields = self.object(value, path)?;
        let kind = match fields.get("type") {
            Some(Node::String(t)) if t == "noul" => Kind::Noul,
            Some(Node::String(t)) if t == "choice" => Kind::Choice,
            Some(Node::String(t)) if t == "score" => Kind::Score,
            Some(_) => {
                let msg = "Input should be 'noul', 'choice' or 'score'";
                self.fail("literal_error", &at(path, "type"), msg);
                return None;
            }
            None => {
                self.missing(&at(path, "type"));
                return None;
            }
        };
        let before = self.errors.len();
        self.only(fields, &["type", "instructions", "criteria"], path);
        let instructions =
            self.optional_text(fields.get("instructions"), &at(path, "instructions"));
        let (criteria, criteria_path) = (fields.get("criteria"), at(path, "criteria"));
        let (names, details) = match kind {
            Kind::Noul => {
                let (yes, no) = self.noul_criteria(criteria, &criteria_path);
                let asks = [&instructions, &yes, &no]
                    .into_iter()
                    .any(|t| t.as_ref().is_some_and(|t| !t.is_blank()));
                if !asks {
                    let msg = "Value error, a noul question needs instructions or a true or \
                               false description to decide on";
                    self.fail("value_error", path, msg);
                }
                (vec!["yes".into(), "no".into()], vec![yes, no])
            }
            Kind::Choice => self.choice_criteria(criteria, &criteria_path),
            Kind::Score => self.score_criteria(criteria, &criteria_path),
        };
        (self.errors.len() == before).then(|| Question {
            id: id.to_owned(),
            kind,
            instructions,
            names,
            details,
        })
    }

    fn noul_criteria(
        &mut self,
        criteria: Option<&Node>,
        path: &[Loc],
    ) -> (Option<Node>, Option<Node>) {
        let fields = match criteria {
            None | Some(Node::Null) => None,
            Some(node) => self.object(node, path),
        };
        let Some(fields) = fields else {
            return (None, None);
        };
        self.only(fields, &["true", "false"], path);
        (
            self.optional_text(fields.get("true"), &at(path, "true")),
            self.optional_text(fields.get("false"), &at(path, "false")),
        )
    }

    fn choice_criteria(
        &mut self,
        criteria: Option<&Node>,
        path: &[Loc],
    ) -> (Vec<String>, Vec<Option<Node>>) {
        let (mut names, mut details) = (Vec::new(), Vec::new());
        match criteria {
            Some(Node::Object(options)) => {
                self.size(options.0.len(), MAX_OPTIONS, "Dictionary", path);
                for (name, detail) in &options.0 {
                    names.push(name.clone());
                    details.push(self.optional_text(Some(detail), &at(path, name)));
                }
                if let Err(msg) = check_option_names(&names) {
                    self.fail("value_error", path, format!("Value error, {msg}"));
                }
            }
            Some(_) => self.fail("dict_type", path, "Input should be a valid dictionary"),
            None => self.missing(path),
        }
        (names, details)
    }

    fn score_criteria(
        &mut self,
        criteria: Option<&Node>,
        path: &[Loc],
    ) -> (Vec<String>, Vec<Option<Node>>) {
        let mut details = Vec::new();
        match criteria {
            Some(Node::Array(levels)) => {
                self.size(levels.len(), MAX_LEVELS, "List", path);
                for (i, level) in levels.iter().enumerate() {
                    let mut here = path.to_vec();
                    here.push(Loc::Index(i));
                    details.push(self.required_text(level, &here));
                }
            }
            Some(_) => self.fail("list_type", path, "Input should be a valid list"),
            None => self.missing(path),
        }
        ((0..details.len()).map(|i| i.to_string()).collect(), details)
    }
}

/// Refuses option names that make the option lines ambiguous: blank, more
/// than a line, or equal after trimming and lowercasing.
fn check_option_names(names: &[String]) -> Result<(), String> {
    let mut seen = HashSet::new();
    for name in names {
        let key = name.trim().to_lowercase();
        if key.is_empty() {
            return Err("option names must be nonempty".into());
        }
        if name
            .chars()
            .any(|c| c.is_control() || matches!(c, '\u{2028}' | '\u{2029}'))
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

impl Request {
    fn parse(root: &Node) -> Result<Self, Vec<Detail>> {
        let mut parser = Parser::default();
        let body = [Loc::Key("body".into())];
        let Some(fields) = parser.object(root, &body) else {
            return Err(parser.errors);
        };
        // Unknown top-level fields are ignored, as the published schema allows.
        let state = match fields.get("state") {
            Some(state) => parser
                .text(state, &at(&body, "state"))
                .then(|| state.clone()),
            None => {
                parser.missing(&at(&body, "state"));
                None
            }
        };
        let model = match fields.get("model") {
            Some(Node::String(model)) => Some(model.clone()),
            Some(_) => {
                parser.fail(
                    "string_type",
                    &at(&body, "model"),
                    "Input should be a valid string",
                );
                None
            }
            None => {
                parser.missing(&at(&body, "model"));
                None
            }
        };
        let template_kwargs = match fields.get("chat_template_kwargs") {
            None | Some(Node::Null) => Default::default(),
            Some(kwargs @ Node::Object(_)) => match serde_json::to_value(kwargs) {
                Ok(serde_json::Value::Object(map)) => map,
                _ => Default::default(),
            },
            Some(_) => {
                let msg = "Input should be a valid dictionary";
                parser.fail("dict_type", &at(&body, "chat_template_kwargs"), msg);
                Default::default()
            }
        };
        let reasoning_effort = match fields.get("reasoning_effort") {
            None | Some(Node::Null) => None,
            Some(Node::String(level)) if is_effort(level) => Some(level.clone()),
            Some(_) => {
                let levels = EFFORT_LEVELS
                    .iter()
                    .map(|l| format!("'{l}'"))
                    .collect::<Vec<_>>();
                let msg = format!("Input should be 'none', {}", levels.join(", "));
                parser.fail("literal_error", &at(&body, "reasoning_effort"), msg);
                None
            }
        };
        let path = at(&body, "questions");
        let questions = match fields.get("questions") {
            Some(Node::Object(map)) => {
                parser.size(map.0.len(), MAX_QUESTIONS, "Dictionary", &path);
                parser.questions(map, &path)
            }
            Some(_) => {
                parser.fail("dict_type", &path, "Input should be a valid dictionary");
                Vec::new()
            }
            None => {
                parser.missing(&path);
                Vec::new()
            }
        };
        match (state, model, parser.errors.is_empty()) {
            (Some(state), Some(model), true) => Ok(Request {
                state,
                model,
                questions,
                template_kwargs,
                reasoning_effort,
            }),
            _ => Err(parser.errors),
        }
    }

    /// Valid, but a choice wider than the alphabet cannot be labelled.
    fn unservable(&self) -> Option<String> {
        let wide = self
            .questions
            .iter()
            .find(|q| q.kind == Kind::Choice && q.names.len() > MAX_LETTER_OPTIONS)?;
        Some(format!(
            "question {:?} has {} options, but llmman labels options A to Z, so a choice takes at \
             most {MAX_LETTER_OPTIONS}",
            wide.id,
            wide.names.len()
        ))
    }

    /// Thinking off, then the caller's own arguments.
    fn chat_template_kwargs(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut kwargs = serde_json::Map::new();
        kwargs.insert("enable_thinking".into(), false.into());
        kwargs.extend(self.template_kwargs.clone());
        kwargs
    }
}

fn unprocessable(detail: Vec<Detail>) -> Response {
    let body = Json(json!({ "detail": detail }));
    (StatusCode::UNPROCESSABLE_ENTITY, body).into_response()
}

// -- The prompt --------------------------------------------------------------

/// The question and its candidates, one per line (sglang's prompt format 1).
fn candidate_lines(question: &Question, labels: &[String]) -> Vec<String> {
    let asked = question
        .instructions
        .as_ref()
        .filter(|i| !i.is_blank())
        .map(|i| Node::render(Some(i)))
        .unwrap_or_default();
    let question_line = |prefix: &str| (!asked.is_empty()).then(|| format!("{prefix}{asked}"));
    let mut lines = Vec::new();
    match question.kind {
        Kind::Choice => {
            lines.extend(question_line("Question: "));
            for ((label, name), detail) in labels.iter().zip(&question.names).zip(&question.details)
            {
                let detail = Node::render(detail.as_ref());
                lines.push(if detail.is_empty() {
                    format!("{label}: {name}")
                } else {
                    format!("{label}: {name} - {detail}")
                });
            }
        }
        Kind::Score => {
            lines.extend(question_line("Question: "));
            for (label, level) in labels.iter().zip(&question.details) {
                lines.push(format!("{label}: {}", Node::render(level.as_ref())));
            }
        }
        Kind::Noul => {
            lines.push(
                question_line("Is the following true? ").unwrap_or("Is the following true?".into()),
            );
            for (label, detail) in labels.iter().zip(&question.details) {
                let detail = Node::render(detail.as_ref());
                if !detail.is_empty() {
                    lines.push(format!("{label}: {detail}"));
                }
            }
        }
    }
    lines
}

fn state_and_question(state: &str, question: &Question, labels: &[String]) -> String {
    format!(
        "{state}\n\n{}",
        candidate_lines(question, labels).join("\n")
    )
}

/// What a local model is read at: the question and a closing instruction to
/// answer with one label, so the next token is one.
fn user_turn(state: &str, question: &Question, labels: &[String]) -> String {
    let closing = match question.kind {
        Kind::Choice => "Answer with the letter of one option only.",
        Kind::Score => "Answer with the number of one level only.",
        Kind::Noul => "Answer with yes or no only.",
    };
    format!("{}\n{closing}", state_and_question(state, question, labels))
}

/// Whether the text after the message leaves a `<think>` block open, so the
/// next token is reasoning, not an answer.
fn leaves_thinking_open(prompt: &str, message: &str) -> bool {
    // The message's last line is fixed; what follows it is the generation
    // prompt. `None` orders below `Some`.
    let closing = message.rsplit('\n').next().unwrap_or(message);
    let tail = prompt
        .rfind(closing)
        .map_or(prompt, |at| &prompt[at + closing.len()..]);
    tail.rfind("<think>") > tail.rfind("</think>")
}

// -- Failures and results ----------------------------------------------------

/// Why a question went unanswered: the request's fault (400), the local
/// backend's (500), or a provider's, whose refusal of the caller (a bad key,
/// a rate limit) keeps its status as on every other route.
#[derive(Debug)]
enum Failure {
    Refused(String),
    Backend(String),
    Provider(StatusCode, String),
}

impl Failure {
    fn for_question(self, id: &str) -> Self {
        let named = |m: String| format!("question {id:?}: {m}");
        match self {
            Failure::Refused(m) => Failure::Refused(named(m)),
            Failure::Backend(m) => Failure::Backend(named(m)),
            Failure::Provider(status, m) => Failure::Provider(status, named(m)),
        }
    }

    fn respond(self) -> Result<Response, AppError> {
        match self {
            Failure::Refused(message) => Ok(refuse(StatusCode::BAD_REQUEST, message)),
            Failure::Provider(status, message) if status.is_client_error() => {
                Ok(refuse(status, message))
            }
            Failure::Backend(message) => {
                Err(AppError::status(StatusCode::INTERNAL_SERVER_ERROR, message))
            }
            Failure::Provider(_, message) => {
                Err(AppError::status(StatusCode::BAD_GATEWAY, message))
            }
        }
    }
}

/// A question scored.
#[derive(Debug)]
struct Scored {
    /// One per label, in label order, adding up to 1.
    probabilities: Vec<f64>,
    /// The full-vocabulary probability of the labels, when read.
    mass: Option<f64>,
    source: Source,
    tokens: Tokens,
}

/// Where probabilities come from, reported as `x_source`: a caller setting a
/// threshold on them has to know.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Source {
    /// A local model's own next-token probabilities.
    Logprobs,
    /// Probabilities a hosted model stated.
    Elicited,
}

/// The message in an upstream error body (OpenAI's and Anthropic's shape).
fn upstream_message(body: String) -> String {
    serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().map(str::to_owned))
        .unwrap_or(body)
}

// -- Reading a local llama-server -------------------------------------------

struct Backend<'a> {
    client: &'a Client,
    target: &'a Target,
}

impl Backend<'_> {
    async fn post(
        &self,
        path: &str,
        body: serde_json::Value,
        timeout: Duration,
    ) -> Result<serde_json::Value, Failure> {
        let backend = |what: &str, e: &dyn fmt::Display| {
            Failure::Backend(format!("{what} {path} to the inference backend: {e}"))
        };
        let resp = self
            .client
            .post(self.target.url(path))
            .timeout(timeout)
            .json(&body)
            .send()
            .await
            .map_err(|e| backend("send", &e))?;
        let status = resp.status();
        let text = resp.text().await.map_err(|e| backend("read", &e))?;
        if status.is_success() {
            return serde_json::from_str(&text).map_err(|e| backend("unexpected body from", &e));
        }
        let message = upstream_message(text);
        Err(if status.is_client_error() {
            Failure::Refused(message)
        } else {
            Failure::Backend(format!("inference backend {status}: {message}"))
        })
    }

    /// The chat-templated text of one user turn, as `/v1/chat/completions` renders it.
    async fn apply_template(
        &self,
        message: &str,
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<String, Failure> {
        let body = json!({
            "messages": [{ "role": "user", "content": message }],
            "chat_template_kwargs": kwargs,
        });
        let reply = self
            .post("/apply-template", body, PREPARE_TIMEOUT)
            .await
            .map_err(|f| match f {
                Failure::Refused(m) => Failure::Refused(format!("the chat template failed: {m}")),
                other => other,
            })?;
        reply["prompt"]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| Failure::Backend("/apply-template returned no prompt".into()))
    }

    /// The ids `/completion` evaluates for a text prompt: special tokens
    /// parsed and BOS added, as for a chat request.
    async fn tokenize(&self, text: &str) -> Result<Vec<i64>, Failure> {
        let body = json!({ "content": text, "add_special": true, "parse_special": true });
        let reply = self.post("/tokenize", body, PREPARE_TIMEOUT).await?;
        serde_json::from_value(reply["tokens"].clone())
            .map_err(|_| Failure::Backend("/tokenize returned no tokens".into()))
    }

    /// The token each label adds after the prompt. A label that is not
    /// exactly one distinct token there cannot be scored, so the model is
    /// refused rather than read at a position that is not the answer.
    async fn label_ids(
        &self,
        prompt: &str,
        ids: &[i64],
        labels: &[String],
    ) -> Result<Vec<i64>, Failure> {
        let texts: Vec<String> = labels.iter().map(|l| format!("{prompt}{l}")).collect();
        let ended = futures::future::try_join_all(texts.iter().map(|t| self.tokenize(t))).await?;
        let mut label_ids: Vec<i64> = Vec::new();
        for (label, tokens) in labels.iter().zip(ended) {
            match tokens.split_last() {
                Some((&id, head)) if head == ids && !label_ids.contains(&id) => label_ids.push(id),
                _ => {
                    return Err(Failure::Refused(format!(
                        "the answer label {label:?} is not one distinct token after the chat \
                         prompt for this model, so this model is not supported"
                    )))
                }
            }
        }
        Ok(label_ids)
    }

    /// What the model would write next: one token's worth of `/completion`.
    async fn complete(&self, mut body: serde_json::Value) -> Result<serde_json::Value, Failure> {
        body["n_predict"] = 1.into();
        body["cache_prompt"] = true.into();
        let reply = self.post("/completion", body, SCORE_TIMEOUT).await?;
        if reply["truncated"].as_bool() == Some(true) {
            // The position scored would no longer be the answer's.
            return Err(Failure::Refused(
                "the prompt does not fit the model's context".into(),
            ));
        }
        Ok(reply)
    }

    /// Each label's probability renormalised over the labels, and the
    /// full-vocabulary mass they hold.
    ///
    /// `n_probs` lists the top of the model's own distribution, the exact
    /// softmax over the vocabulary: one request when every label is listed.
    /// A label outside the top is invisible there, so then a second request
    /// reads all of them ([`Backend::boosted`]).
    async fn label_probabilities(
        &self,
        ids: &[i64],
        labels: &[i64],
    ) -> Result<(Vec<f64>, f64), Failure> {
        let n_probs = labels.len() + TOP_SLACK;
        let reply = self
            .complete(json!({ "prompt": ids, "n_probs": n_probs, "temperature": 0 }))
            .await?;
        let top: Vec<_> = candidates(&reply, "top_logprobs", "logprob")?
            .into_iter()
            .map(|(id, logprob)| (id, logprob.exp()))
            .collect();
        if let Some(found) = from_top(&top, labels) {
            return Ok(found);
        }
        let boosted = self.boosted(ids, labels).await?;
        let mass = mass_from(&top, labels, &boosted);
        Ok((boosted, mass))
    }

    /// Every label's share of the labels, however unlikely. A logit of
    /// [`LABEL_BOOST`] added to each puts them ahead of every other token,
    /// and one constant leaves the ratios between them as the model has
    /// them; sampling is cut to a bare temperature so nothing truncates.
    async fn boosted(&self, ids: &[i64], labels: &[i64]) -> Result<Vec<f64>, Failure> {
        let reply = self
            .complete(json!({
                "prompt": ids,
                "n_probs": labels.len(),
                "post_sampling_probs": true,
                "samplers": ["temperature"],
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
                "min_p": 0.0,
                "logit_bias": labels.iter().map(|id| json!([id, LABEL_BOOST])).collect::<Vec<_>>(),
            }))
            .await?;
        let probs = candidates(&reply, "top_probs", "prob")?;
        // A token outside the labels is listed only if the boost did not clear
        // it, which displaces a label: the model puts ~nothing on any of them.
        let total: f64 = probs.iter().map(|(_, p)| p).sum();
        if probs.iter().any(|(id, _)| !labels.contains(id)) || total <= 0.0 {
            return Err(Failure::Refused(format!(
                "the model puts almost no probability on the answer labels (over {LABEL_BOOST} \
                 logits below its top token), so it is not answering with them"
            )));
        }
        // llama-server leaves out a zero, and the rest are shares of `total`.
        let share = |label: &i64| {
            probs
                .iter()
                .find(|(id, _)| id == label)
                .map_or(0.0, |(_, p)| p / total)
        };
        Ok(labels.iter().map(share).collect())
    }

    async fn score(
        &self,
        request: &Request,
        state: &str,
        question: &Question,
    ) -> Result<Scored, Failure> {
        let labels = question.labels();
        let turn = user_turn(state, question, &labels);
        let prompt = self
            .apply_template(&turn, &request.chat_template_kwargs())
            .await?;
        if leaves_thinking_open(&prompt, &turn) {
            return Err(Failure::Refused(
                "the chat template leaves a reasoning block open at the answer position, so this \
                 model is not supported with these chat_template_kwargs"
                    .into(),
            ));
        }
        let ids = self.tokenize(&prompt).await?;
        let label_ids = self.label_ids(&prompt, &ids, &labels).await?;
        let (probabilities, mass) = self.label_probabilities(&ids, &label_ids).await?;
        if !probabilities.iter().chain([&mass]).all(|v| v.is_finite()) {
            return Err(Failure::Backend("scored non-finite values".into()));
        }
        Ok(Scored {
            probabilities,
            mass: Some(mass),
            source: Source::Logprobs,
            tokens: Tokens::new(ids.len() as u64, 0, 0, 0),
        })
    }
}

/// `(token id, value)` of each candidate `/completion` lists for its token;
/// a value JSON cannot hold (`-inf`) reads as negative infinity.
fn candidates(
    reply: &serde_json::Value,
    list: &str,
    key: &str,
) -> Result<Vec<(i64, f64)>, Failure> {
    let unexpected = || {
        Failure::Backend(format!(
            "/completion returned no {list}: the llama-server is too old to report token \
             probabilities this way"
        ))
    };
    reply["completion_probabilities"][0][list]
        .as_array()
        .ok_or_else(unexpected)?
        .iter()
        .map(|c| {
            Some((
                c["id"].as_i64()?,
                c[key].as_f64().unwrap_or(f64::NEG_INFINITY),
            ))
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(unexpected)
}

/// The labels' shares of the labels, and the mass they hold, from the top of
/// the full-vocabulary distribution: only when every label is listed and
/// none has zero probability.
fn from_top(top: &[(i64, f64)], labels: &[i64]) -> Option<(Vec<f64>, f64)> {
    let full: Vec<f64> = labels
        .iter()
        .map(|label| {
            top.iter()
                .find(|(id, _)| id == label)
                .map_or(0.0, |(_, p)| *p)
        })
        .collect();
    let mass: f64 = full.iter().sum();
    full.iter()
        .all(|&p| p > 0.0)
        .then(|| (full.iter().map(|p| p / mass).collect(), mass))
}

/// The labels' mass when some are outside `top`: one label in both lists
/// fixes the scale, its full-vocabulary probability over its share. Zero when
/// none is listed, as for a model that never answers with a label.
fn mass_from(top: &[(i64, f64)], labels: &[i64], shares: &[f64]) -> f64 {
    labels
        .iter()
        .zip(shares)
        .filter_map(|(label, &share)| {
            let full = top.iter().find(|(id, _)| id == label)?.1;
            (share > 0.0 && full > 0.0).then_some((share, full))
        })
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .map_or(0.0, |(share, full)| full / share)
}

// -- Asking a hosted model ---------------------------------------------------

/// What a hosted model is told. Plain JSON in the reply, not a forced tool or
/// `response_format`: Anthropic turns thinking off for a model that thinks on
/// a budget when a tool is forced, which would drop the caller's variant.
fn elicitation_prompt(labels: &[String]) -> String {
    let shape = labels
        .iter()
        .map(|label| format!("{label:?}: <probability>"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "You are a decision model. The user gives a state, then a question with candidate \
         answers, each marked by a label. For every candidate, give the probability, from 0 to \
         1, that it is the right answer for that state. State your honest uncertainty: the \
         probabilities add up to 1, and none is 1 unless nothing else is possible. Reply with \
         one JSON object and nothing else, mapping every label to its probability, like \
         {{{shape}}}."
    )
}

fn reply_text(reply: &serde_json::Value) -> String {
    match &reply["choices"][0]["message"]["content"] {
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Array(parts) => {
            parts.iter().filter_map(|p| p["text"].as_str()).collect()
        }
        _ => String::new(),
    }
}

/// The probabilities a hosted model stated, per label, normalised to add up
/// to 1 (a rounding model may state 0.97). A label left out has none.
fn parse_elicited(reply: &str, labels: &[String]) -> Result<Vec<f64>, String> {
    // The object, whatever surrounds it: a code fence, a sentence.
    let json = reply
        .find('{')
        .zip(reply.rfind('}'))
        .and_then(|(start, end)| reply.get(start..=end))
        .ok_or("the reply holds no JSON object")?;
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("the reply's JSON does not parse: {e}"))?;
    // Some models nest the labels under `probabilities`.
    let stated = value
        .get("probabilities")
        .and_then(|p| p.as_object())
        .or_else(|| value.as_object())
        .ok_or("the reply's JSON is not an object")?;
    if labels.iter().all(|label| !stated.contains_key(label)) {
        return Err("the reply names none of the labels".into());
    }
    let mut probabilities = Vec::new();
    for label in labels {
        let p = match stated.get(label) {
            None => 0.0,
            Some(v) => v
                .as_f64()
                .or_else(|| {
                    v.as_str()
                        .and_then(|s| s.trim().trim_end_matches('%').parse().ok())
                })
                .filter(|p| p.is_finite() && *p >= 0.0)
                .ok_or_else(|| {
                    format!("the probability for {label:?} is not a non-negative number")
                })?,
        };
        probabilities.push(p);
    }
    let total: f64 = probabilities.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        return Err("the probabilities do not add up to a positive number".into());
    }
    Ok(probabilities.iter().map(|p| p / total).collect())
}

/// A hosted model, reached through llmman's chat path so every wire and
/// provider quirk is handled as for any other request.
struct Hosted<'a> {
    client: &'a Client,
    target: &'a Target,
    remote: &'a RemoteTarget,
    /// The variant: `reasoning_effort`, as on every other route.
    effort: Option<&'a str>,
}

impl Hosted<'_> {
    async fn ask(&self, state: &str, question: &Question) -> Result<Scored, Failure> {
        let labels = question.labels();
        let mut req = json!({
            "model": self.remote.model,
            "messages": [
                { "role": "system", "content": elicitation_prompt(&labels) },
                { "role": "user", "content": state_and_question(state, question, &labels) },
            ],
            // Removed on the way out for a model that rejects sampling overrides.
            "temperature": 0,
        });
        if let Some(effort) = self.effort {
            req["reasoning_effort"] = effort.into();
        }
        let upstream = send_chat_completion(self.client, self.target, &req, &self.remote.model)
            .await
            .map_err(|e| Failure::Provider(e.1, format!("{:#}", e.0)))?;
        let status = upstream.status;
        let text = upstream.text().await;
        if !status.is_success() {
            return Err(Failure::Provider(status, upstream_message(text)));
        }
        let gateway = |message: String| Failure::Provider(StatusCode::BAD_GATEWAY, message);
        let reply: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
            gateway(format!(
                "provider {} returned an unexpected body: {e}",
                self.remote.provider
            ))
        })?;
        let probabilities = parse_elicited(&reply_text(&reply), &labels).map_err(|why| {
            let cut = if reply["choices"][0]["finish_reason"] == "length" {
                " (it stopped at its output limit)"
            } else {
                ""
            };
            gateway(format!(
                "{} did not state probabilities for the labels{cut}: {why}",
                self.remote.model
            ))
        })?;
        // Read as the ledger reads any OpenAI-shaped reply: cache and reasoning counts too.
        let mut decoder = Decoder::new(Dialect::OpenAi, Some("application/json"), false);
        decoder.feed(Bytes::from(text));
        decoder.finish();
        Ok(Scored {
            probabilities,
            mass: None,
            source: Source::Elicited,
            tokens: decoder.tokens().unwrap_or_default(),
        })
    }

    async fn ask_all(&self, state: &str, questions: &[Question]) -> Result<Vec<Scored>, Failure> {
        let mut scored = Vec::new();
        for chunk in questions.chunks(HOSTED_CONCURRENCY) {
            let chunk = chunk.iter().map(|q| async move {
                self.ask(state, q).await.map_err(|f| f.for_question(&q.id))
            });
            scored.extend(futures::future::try_join_all(chunk).await?);
        }
        Ok(scored)
    }
}

// -- The answers -------------------------------------------------------------

/// One answer in the published shapes, plus `x_source` and, for a local
/// model, sglang's `x_label_mass`: the probability the model put on the
/// labels at all, low when it wanted to say something else.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Answer {
    Noul {
        noul: f64,
        #[serde(flatten)]
        extra: Extra,
    },
    Choice {
        choice: String,
        confidence: f64,
        probabilities: Ordered<f64>,
        #[serde(flatten)]
        extra: Extra,
    },
    Score {
        score: f64,
        confidence: f64,
        legend: Ordered<Node>,
        probabilities: Ordered<f64>,
        #[serde(flatten)]
        extra: Extra,
    },
}

#[derive(Debug, Serialize)]
struct Extra {
    #[serde(skip_serializing_if = "Option::is_none")]
    x_label_mass: Option<f64>,
    x_source: Source,
}

/// As scored: a softmax can miss a sum of 1 by an ulp, and only confidence normalises.
fn answer(question: &Question, scored: &Scored) -> Answer {
    let p = &scored.probabilities[..];
    let extra = Extra {
        x_label_mass: scored.mass,
        x_source: scored.source,
    };
    let names = question.names.iter().cloned();
    let by_name = || Ordered(names.clone().zip(p.iter().copied()).collect());
    match question.kind {
        Kind::Noul => Answer::Noul { noul: p[0], extra },
        Kind::Choice => Answer::Choice {
            choice: question.names[argmax(p)].clone(),
            confidence: choice_confidence(&normalized(p)),
            probabilities: by_name(),
            extra,
        },
        Kind::Score => {
            let levels = question
                .details
                .iter()
                .map(|d| d.clone().unwrap_or(Node::Null));
            Answer::Score {
                score: p
                    .iter()
                    .enumerate()
                    .map(|(level, p)| level as f64 * p)
                    .sum(),
                confidence: score_confidence(&normalized(p)),
                legend: Ordered(names.clone().zip(levels).collect()),
                probabilities: by_name(),
                extra,
            }
        }
    }
}

/// The first largest, as Python's `index(max(...))` picks it.
fn argmax(values: &[f64]) -> usize {
    (0..values.len()).fold(0, |best, i| if values[i] > values[best] { i } else { best })
}

fn normalized(probabilities: &[f64]) -> Vec<f64> {
    let total: f64 = probabilities.iter().sum();
    if total <= 0.0 {
        return vec![1.0 / probabilities.len() as f64; probabilities.len()];
    }
    probabilities.iter().map(|p| p / total).collect()
}

/// How far the top option stands above a uniform guess, 0 to 1.
fn choice_confidence(q: &[f64]) -> f64 {
    if q.len() == 1 {
        return 1.0;
    }
    let (n, top) = (q.len() as f64, q.iter().copied().fold(f64::MIN, f64::max));
    ((n * top - 1.0) / (n - 1.0)).clamp(0.0, 1.0)
}

/// One minus the spread around the top level relative to a uniform spread, at least 0.
fn score_confidence(q: &[f64]) -> f64 {
    if q.len() == 1 {
        return 1.0;
    }
    let n = q.len() as f64;
    let top = argmax(q) as f64;
    let spread: f64 = q
        .iter()
        .enumerate()
        .map(|(i, p)| p * (i as f64 - top).abs())
        .sum();
    let uniform: f64 = (0..q.len())
        .map(|i| (i as f64 - (n - 1.0) / 2.0).abs())
        .sum::<f64>()
        / n;
    (1.0 - spread / uniform).max(0.0)
}

#[derive(Serialize)]
struct Reply {
    model: String,
    answers: Ordered<Answer>,
    usage: serde_json::Value,
}

/// Every question's prompt counts, so the state is counted once per
/// question. In the Responses API's shape, which `llmman usage` reads:
/// `input_tokens` holds the cached ones, details only when there are any.
fn usage_json(t: &Tokens) -> serde_json::Value {
    let mut usage = json!({ "input_tokens": t.input, "output_tokens": t.output });
    if t.cache_read + t.cache_write > 0 {
        usage["input_tokens_details"] =
            json!({ "cached_tokens": t.cache_read, "cache_write_tokens": t.cache_write });
    }
    if t.reasoning > 0 {
        usage["output_tokens_details"] = json!({ "reasoning_tokens": t.reasoning });
    }
    usage
}

// -- The route ---------------------------------------------------------------

/// `model` and the variant it names: a trailing `/<level>` on a hosted
/// reference, for a client that can only set `model`:
/// `llmman.provider/anthropic/claude-sonnet-5-5/xhigh`, or
/// `anthropic/claude-sonnet-5-5/xhigh` when `anthropic` is a provider. A bare
/// `<org>/<repo>` stays a Hugging Face repository (see
/// [`providers::REMOTE_PREFIX`]); only a known provider and a level make one hosted.
async fn hosted_variant(model: &str) -> Result<Option<(String, String)>, AppError> {
    let Some((head, level)) = model.rsplit_once('/').filter(|(_, level)| is_effort(level)) else {
        return Ok(None);
    };
    if providers::is_remote_ref(head) {
        return Ok(Some((head.to_owned(), level.to_owned())));
    }
    let Some((provider, name)) = head.split_once('/') else {
        return Ok(None);
    };
    let known = provider_catalog().await?.get(provider).is_some();
    Ok(known.then(|| {
        (
            providers::format_remote_ref(provider, name),
            level.to_owned(),
        )
    }))
}

/// `root` with `model` set, every other key where it was.
fn set_model(root: &mut Node, model: String) {
    let Node::Object(fields) = root else {
        unreachable!("parsed as an object");
    };
    match fields.0.iter_mut().find(|(key, _)| key == "model") {
        Some((_, slot)) => *slot = Node::String(model),
        None => fields.0.push(("model".into(), Node::String(model))),
    }
}

/// A local model is read at its first token, with thinking off.
fn local_variant_refusal(effort: &str, model: &str) -> Response {
    let message = format!(
        "reasoning effort {effort:?} needs a hosted model: {model} is read at its first token, \
         with thinking off"
    );
    refuse(StatusCode::BAD_REQUEST, message)
}

pub(super) async fn handle_systemone(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    let mut root: Node = match serde_json::from_slice(&body) {
        Ok(root) => root,
        Err(e) => {
            return Ok(unprocessable(vec![Detail {
                kind: "json_invalid",
                loc: vec![Loc::Key("body".into()), Loc::Index(e.column())],
                msg: "JSON decode error".into(),
            }]))
        }
    };
    let request = match Request::parse(&root) {
        Ok(request) => request,
        Err(detail) => return Ok(unprocessable(detail)),
    };
    if let Some(message) = request.unservable() {
        return Ok(refuse(StatusCode::BAD_REQUEST, message));
    }
    let (model_ref, named) = match hosted_variant(&request.model).await? {
        Some((model, level)) => (model, Some(level)),
        None => (request.model.clone(), None),
    };
    let effort = match (named, &request.reasoning_effort) {
        (Some(named), Some(given)) if named != *given => {
            let message =
                format!("the model names variant {named:?}, but reasoning_effort is {given:?}");
            return Ok(refuse(StatusCode::BAD_REQUEST, message));
        }
        (named, given) => named.or_else(|| given.clone()),
    };
    // A reference that can only be local has no variant: refuse before loading it.
    if let Some(effort) = &effort {
        if !providers::is_remote_ref(&model_ref) && crate::hybrid::split_ref(&model_ref).is_none() {
            return Ok(local_variant_refusal(effort, &model_ref));
        }
    }

    let (model, target, guard) = ensure_model(&state, &model_ref, Some(&headers), None).await?;
    let activity = begin_activity(guard, None).await;
    let state_text = Node::render(Some(&request.state));
    let scored = match &target {
        Target::Local(_) => {
            if let Some(effort) = &effort {
                return Ok(local_variant_refusal(effort, &model));
            }
            // `None`: unloaded since `ensure_model`, which the request below reports.
            if let Some(engine) = local_engine(&state, &target)
                .await
                .filter(|e| *e != Engine::LlamaServer)
            {
                let message = format!(
                    "{model} is served by {}, but {ROUTE} reads token probabilities from \
                     llama-server; use a GGUF model",
                    engine.label()
                );
                return Ok(refuse(StatusCode::NOT_IMPLEMENTED, message));
            }
            let backend = Backend {
                client: &state.0.client,
                target: &target,
            };
            let mut scored = Vec::new();
            // One at a time: the questions share a state, which llama-server's
            // prompt cache then prefills once.
            for question in &request.questions {
                match backend.score(&request, &state_text, question).await {
                    Ok(one) => scored.push(one),
                    Err(failure) => return failure.for_question(&question.id).respond(),
                }
            }
            scored
        }
        // Another llmman, which reads its own backend.
        Target::Peer(_) => {
            set_model(&mut root, model);
            let body = Bytes::from(serde_json::to_vec(&root).map_err(|e| anyhow!(e))?);
            return proxy(&state.0.client, &target, ROUTE, &headers, body, activity).await;
        }
        Target::Remote(remote) => {
            let hosted = Hosted {
                client: &state.0.client,
                target: &target,
                remote,
                effort: effort.as_deref(),
            };
            match hosted.ask_all(&state_text, &request.questions).await {
                Ok(scored) => scored,
                Err(failure) => return failure.respond(),
            }
        }
    };

    let mut tokens = Tokens::default();
    scored.iter().for_each(|s| tokens.add(&s.tokens));
    let answers = request
        .questions
        .iter()
        .zip(&scored)
        .map(|(question, scored)| (question.id.clone(), answer(question, scored)))
        .collect();
    let reply = Reply {
        model,
        answers: Ordered(answers),
        usage: usage_json(&tokens),
    };
    Ok(Json(reply).into_response())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    use std::sync::Arc;
    use std::time::Instant;

    use axum::routing::post;
    use axum::Router;
    use tokio::sync::Mutex;

    use super::super::running_key;
    use super::super::test_support::{running_model_fixture_with_engine, serve_router, test_state};
    use super::*;
    use crate::providers::Wire;

    const STATE: &str = "The integration keeps failing.";
    const CLOSED: &str = "<think>\n\n</think>\n\n";

    fn node(text: &str) -> Node {
        serde_json::from_str(text).unwrap()
    }

    fn parse(text: &str) -> Result<Request, Vec<Detail>> {
        Request::parse(&node(text))
    }

    /// Every problem a bad request has, as `type location`.
    fn problems(text: &str) -> Vec<String> {
        let at = |path: &[Loc]| {
            let parts: Vec<_> = path
                .iter()
                .map(|l| match l {
                    Loc::Key(key) => key.clone(),
                    Loc::Index(i) => i.to_string(),
                })
                .collect();
            parts.join(".")
        };
        let errors = parse(text).unwrap_err();
        errors
            .iter()
            .map(|d| format!("{} {}", d.kind, at(&d.loc)))
            .collect()
    }

    /// The one question of a request around `body`.
    fn question(body: &str) -> Question {
        let text = format!(r#"{{"state":"s","model":"m","questions":{{"q":{body}}}}}"#);
        parse(&text).unwrap().questions.remove(0)
    }

    /// `n` options, or levels, as a request.
    fn wide(kind: &str, n: usize) -> String {
        let items: Vec<_> = (0..n)
            .map(|i| match kind {
                "choice" => format!(r#""option {i}":null"#),
                _ => format!(r#""level {i}""#),
            })
            .collect();
        let criteria = match kind {
            "choice" => format!("{{{}}}", items.join(",")),
            _ => format!("[{}]", items.join(",")),
        };
        format!(
            r#"{{"state":"s","model":"m","questions":{{"q":{{"type":"{kind}","criteria":{criteria}}}}}}}"#
        )
    }

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    fn near(got: f64, want: f64) {
        assert!((got - want).abs() < 1e-9, "{got} vs {want}");
    }

    // -- the request ---------------------------------------------------------

    /// The first option is `A`, so order is part of the question; `serde_json`'s map sorts.
    #[test]
    fn options_and_questions_keep_the_order_sent() {
        let request = parse(
            r#"{"state":"s","model":"m","questions":{
                "zeta":{"type":"choice","criteria":{"zulu":null,"alpha":"first","mid":null}},
                "alpha":{"type":"score","criteria":["low","high"]}}}"#,
        )
        .unwrap();
        let ids: Vec<_> = request.questions.iter().map(|q| q.id.as_str()).collect();
        assert_eq!(ids, ["zeta", "alpha"]);
        assert_eq!(
            request.questions[0].names,
            strings(&["zulu", "alpha", "mid"])
        );
        assert_eq!(request.questions[0].labels(), strings(&["A", "B", "C"]));
        assert_eq!(request.questions[1].labels(), strings(&["0", "1"]));
    }

    /// Objects render as compact JSON in the order sent, escapes written out.
    #[test]
    fn structured_text_renders_as_written() {
        let request = parse(
            r#"{"state":{"z":1,"a":["Jos\u00e9", {"k": true}]},"model":"m",
                "questions":{"q":{"type":"noul","instructions":"x"}}}"#,
        )
        .unwrap();
        assert_eq!(
            Node::render(Some(&request.state)),
            r#"{"z":1,"a":["José",{"k":true}]}"#
        );
        assert_eq!(Node::render(Some(&Node::String("as is".into()))), "as is");
        assert_eq!(Node::render(None), "");
        assert_eq!(serde_json::to_string(&Node::Null).unwrap(), "null");
    }

    /// sglang's prompt format 1.
    #[test]
    fn the_prompt_follows_prompt_format_one() {
        let turn = |body: &str| {
            let q = question(body);
            user_turn("s", &q, &q.labels())
        };
        let choice = r#"{"type":"choice","instructions":"Which team?","criteria":{"billing":"Payments","sales":null}}"#;
        assert_eq!(
            turn(choice),
            "s\n\nQuestion: Which team?\nA: billing - Payments\nB: sales\nAnswer with the letter of one option only."
        );
        assert_eq!(
            turn(r#"{"type":"choice","criteria":{"billing":"Payments","sales":null}}"#),
            "s\n\nA: billing - Payments\nB: sales\nAnswer with the letter of one option only."
        );
        assert_eq!(
            turn(r#"{"type":"score","instructions":"Mood?","criteria":["Calm",{"label":"Angry"}]}"#),
            "s\n\nQuestion: Mood?\n0: Calm\n1: {\"label\":\"Angry\"}\nAnswer with the number of one level only."
        );
        assert_eq!(
            turn(r#"{"type":"noul","instructions":"Needs an answer today."}"#),
            "s\n\nIs the following true? Needs an answer today.\nAnswer with yes or no only."
        );
        assert_eq!(
            turn(r#"{"type":"noul","criteria":{"true":"Reply today","false":"Can wait"}}"#),
            "s\n\nIs the following true?\nyes: Reply today\nno: Can wait\nAnswer with yes or no only."
        );
        // A hosted model is told the format by its system message instead.
        let q = question(r#"{"type":"noul","instructions":"Needs an answer today."}"#);
        assert_eq!(
            state_and_question("s", &q, &q.labels()),
            "s\n\nIs the following true? Needs an answer today."
        );
    }

    #[test]
    fn every_problem_is_reported_with_its_location() {
        assert_eq!(
            problems("{}"),
            [
                "missing body.state",
                "missing body.model",
                "missing body.questions"
            ]
        );
        assert_eq!(problems("[]"), ["model_attributes_type body"]);
        assert_eq!(
            problems(r#"{"state":1,"model":2,"questions":{}}"#),
            [
                "string_type body.state",
                "string_type body.model",
                "too_short body.questions"
            ]
        );
        assert_eq!(
            problems(
                r#"{"state":"s","model":"m","questions":{
                    "a":{"type":"maybe"},
                    "b":{"type":"noul","instrucshuns":"x"},
                    "c":{"type":"noul"},
                    "d":{"type":"choice","criteria":{"X":null,"x ":null}},
                    "e":{"type":"score","criteria":["a","  "]},
                    "f":{"type":"choice"},
                    "g":{"type":"score","criteria":[]},
                    "h":{"type":"noul","instructions":"x","criteria":{"maybe":"y"}},
                    "i":7,
                    "j":{}}}"#
            ),
            [
                "literal_error body.questions.a.type",
                "extra_forbidden body.questions.b.instrucshuns",
                "value_error body.questions.b",
                "value_error body.questions.c",
                "value_error body.questions.d.criteria",
                "value_error body.questions.e.criteria.1",
                "missing body.questions.f.criteria",
                "too_short body.questions.g.criteria",
                "extra_forbidden body.questions.h.criteria.maybe",
                "model_attributes_type body.questions.i",
                "missing body.questions.j.type",
            ]
        );
        let twice = r#"{"state":"s","model":"m","questions":{
            "q":{"type":"noul","instructions":"x"},"q":{"type":"noul","instructions":"y"}}}"#;
        assert_eq!(problems(twice), ["value_error body.questions.q"]);
    }

    /// The published limits, plus a cap on questions; bad option names refused as sglang does.
    #[test]
    fn limits_and_option_names_are_enforced() {
        assert!(parse(&wide("choice", 255)).is_ok());
        assert_eq!(
            problems(&wide("choice", 256)),
            ["too_long body.questions.q.criteria"]
        );
        assert_eq!(
            problems(&wide("choice", 0)),
            ["too_short body.questions.q.criteria"]
        );
        assert!(parse(&wide("score", 1)).is_ok() && parse(&wide("score", 10)).is_ok());
        assert_eq!(
            problems(&wide("score", 11)),
            ["too_long body.questions.q.criteria"]
        );

        let many = |n: usize| {
            let qs: Vec<_> = (0..n)
                .map(|i| format!(r#""q{i}":{{"type":"noul","instructions":"x"}}"#))
                .collect();
            format!(
                r#"{{"state":"s","model":"m","questions":{{{}}}}}"#,
                qs.join(",")
            )
        };
        assert!(parse(&many(MAX_QUESTIONS)).is_ok());
        assert_eq!(
            problems(&many(MAX_QUESTIONS + 1)),
            ["too_long body.questions"]
        );

        for bad in [
            r#""  ""#,
            r#""two\nlines""#,
            r#""tab\there""#,
            r#""a\u2028b""#,
        ] {
            let text = format!(
                r#"{{"state":"s","model":"m","questions":{{"q":{{"type":"choice","criteria":{{{bad}:null}}}}}}}}"#
            );
            assert_eq!(
                problems(&text),
                ["value_error body.questions.q.criteria"],
                "{bad}"
            );
        }
        // Allowed: an empty state, an empty id, a blank description, unknown top-level fields.
        let lenient = r#"{"state":"","model":"m","images":[],"questions":{"":{"type":"choice","criteria":{"a":"  "}}}}"#;
        assert!(parse(lenient).is_ok());
    }

    /// Valid for the schema, but the alphabet is the most a choice is labelled with.
    #[test]
    fn a_choice_wider_than_the_alphabet_is_unservable() {
        let message = parse(&wide("choice", 27)).unwrap().unservable().unwrap();
        assert!(
            message.contains("27 options") && message.contains("A to Z"),
            "{message}"
        );
        assert!(parse(&wide("choice", 26)).unwrap().unservable().is_none());
    }

    #[test]
    fn a_variant_is_a_known_level() {
        let with = |level: &str| {
            format!(
                r#"{{"state":"s","model":"m","reasoning_effort":{level},"questions":{{"q":{{"type":"noul","instructions":"x"}}}}}}"#
            )
        };
        for level in ["none", "minimal", "low", "medium", "high", "xhigh", "max"] {
            let request = parse(&with(&format!("\"{level}\""))).unwrap();
            assert_eq!(request.reasoning_effort.as_deref(), Some(level));
        }
        assert_eq!(parse(&with("null")).unwrap().reasoning_effort, None);
        for bad in ["\"turbo\"", "3"] {
            assert_eq!(
                problems(&with(bad)),
                ["literal_error body.reasoning_effort"]
            );
        }
    }

    /// A client that can only set `model` names the variant as a last segment
    /// of a hosted reference; two-part names are left alone.
    #[tokio::test]
    async fn a_variant_is_a_trailing_level_on_a_hosted_reference() {
        let split = |model: &'static str| async move { hosted_variant(model).await.ok().flatten() };
        let sonnet = "llmman.provider/anthropic/claude-sonnet-5-5";
        assert_eq!(
            split("llmman.provider/anthropic/claude-sonnet-5-5/xhigh").await,
            Some((sonnet.into(), "xhigh".into()))
        );
        assert_eq!(
            split("llmman.provider/openrouter/qwen/qwen3-coder/none").await,
            Some((
                "llmman.provider/openrouter/qwen/qwen3-coder".into(),
                "none".into()
            ))
        );
        for plain in [
            sonnet,
            "qwen3.5:0.8b",
            "gemma4",
            "openai/gpt-oss-120b",
            "hf.co/unsloth/Qwen3.5-0.8B-GGUF",
            "llmman.provider/anthropic/claude-sonnet-5-5/fast",
        ] {
            assert_eq!(split(plain).await, None, "{plain}");
        }
    }

    /// The peer gets the request as sent, option order included.
    #[test]
    fn a_peer_gets_the_request_with_only_its_model_changed() {
        let text = r#"{"state":"s","questions":{"q":{"type":"choice","criteria":{"z":null,"a":null}}},"model":"qwen3.5:0.8b"}"#;
        let mut root = node(text);
        set_model(&mut root, "docker.io/ai/qwen3.5:0.8b".into());
        assert_eq!(
            serde_json::to_string(&root).unwrap(),
            text.replace("qwen3.5:0.8b", "docker.io/ai/qwen3.5:0.8b")
        );
        let mut bare = node(r#"{"state":"s"}"#);
        set_model(&mut bare, "m".into());
        assert_eq!(
            serde_json::to_string(&bare).unwrap(),
            r#"{"state":"s","model":"m"}"#
        );
    }

    // -- reading a model ------------------------------------------------------

    #[test]
    fn a_thinking_block_left_open_is_found() {
        let turn = "state\n\nQuestion\nAnswer with yes or no only.";
        let chat = |tail: &str| format!("<|user|>{turn}<|end|><|assistant|>{tail}");
        assert!(leaves_thinking_open(&chat("<think>\n"), turn));
        assert!(!leaves_thinking_open(&chat(CLOSED), turn));
        assert!(!leaves_thinking_open(&chat(""), turn));
        // A `<think>` the caller wrote in the state is not the template's.
        let quoted = "<|user|>x <think> y\n\nAnswer with yes or no only.<|end|><|assistant|>";
        assert!(!leaves_thinking_open(
            quoted,
            "x <think> y\n\nAnswer with yes or no only."
        ));
    }

    #[test]
    fn label_probabilities_come_from_the_top_when_every_label_is_listed() {
        let top = [(1, 0.6), (9, 0.1), (2, 0.2), (3, 0.1)];
        let (p, mass) = from_top(&top, &[1, 2, 3]).unwrap();
        near(mass, 0.9);
        [0.6 / 0.9, 0.2 / 0.9, 0.1 / 0.9]
            .iter()
            .zip(&p)
            .for_each(|(w, g)| near(*g, *w));
        // One missing, or at zero, goes to the boosted read.
        assert!(from_top(&top, &[1, 2, 4]).is_none());
        assert!(from_top(&[(1, 0.5), (2, 0.0)], &[1, 2]).is_none());
        // The mass of labels outside the top follows from one inside: label 1
        // holds 0.6 of the vocabulary and 2/3 of the labels, so they hold 0.9.
        near(
            mass_from(
                &[(1, 0.6), (2, 0.3)],
                &[1, 2, 3],
                &[0.6 / 0.9, 0.3 / 0.9, 0.0],
            ),
            0.9,
        );
        // None of them listed: a model that never answers with a label.
        assert_eq!(mass_from(&[(9, 0.9)], &[1, 2], &[0.5, 0.5]), 0.0);
    }

    /// sglang's published formulas and worked examples.
    #[test]
    fn confidence_matches_the_published_formulas() {
        near(
            score_confidence(&[0.0, 0.14, 0.86, 0.0, 0.0]),
            1.0 - 0.14 / 1.2,
        );
        near(score_confidence(&[0.0, 0.0, 0.48, 0.52]), 0.52);
        near(score_confidence(&[0.5, 0.0, 0.0, 0.0, 0.5]), 0.0);
        near(choice_confidence(&[0.25; 4]), 0.0);
        near(choice_confidence(&[0.1, 0.9]), 0.8);
        near(choice_confidence(&[1.0]), 1.0);
        near(score_confidence(&[1.0]), 1.0);
        assert_eq!(normalized(&[2.0, 2.0]), [0.5, 0.5]);
        assert_eq!(normalized(&[0.0, 0.0]), [0.5, 0.5]);
        assert_eq!(argmax(&[0.3, 0.4, 0.4]), 1);
    }

    fn scored(probabilities: &[f64], mass: Option<f64>, source: Source) -> Scored {
        Scored {
            probabilities: probabilities.to_vec(),
            mass,
            source,
            tokens: Tokens::default(),
        }
    }

    /// Published shapes, extensions after them, probabilities in option order, levels echoed.
    #[test]
    fn answers_take_the_published_shapes() {
        let json = |q: &Question, s: &Scored| serde_json::to_string(&answer(q, s)).unwrap();
        let q = question(r#"{"type":"noul","instructions":"x"}"#);
        assert_eq!(
            json(&q, &scored(&[0.75, 0.25], Some(0.5), Source::Logprobs)),
            r#"{"type":"noul","noul":0.75,"x_label_mass":0.5,"x_source":"logprobs"}"#
        );
        let q = question(r#"{"type":"choice","criteria":{"zulu":null,"alpha":null,"mid":null}}"#);
        assert_eq!(
            json(&q, &scored(&[0.25, 0.5, 0.25], None, Source::Elicited)),
            r#"{"type":"choice","choice":"alpha","confidence":0.25,"probabilities":{"zulu":0.25,"alpha":0.5,"mid":0.25},"x_source":"elicited"}"#
        );
        let q = question(r#"{"type":"score","criteria":["Calm",{"label":"Angry","b":1}]}"#);
        assert_eq!(
            json(&q, &scored(&[0.25, 0.75], Some(1.0), Source::Logprobs)),
            r#"{"type":"score","score":0.75,"confidence":0.5,"legend":{"0":"Calm","1":{"label":"Angry","b":1}},"probabilities":{"0":0.25,"1":0.75},"x_label_mass":1.0,"x_source":"logprobs"}"#
        );
    }

    /// Usage in the Responses shape the ledger reads; details only when there are some.
    #[test]
    fn usage_carries_cache_and_reasoning_counts() {
        assert_eq!(
            usage_json(&Tokens::new(10, 0, 0, 0)),
            json!({ "input_tokens": 10, "output_tokens": 0 })
        );
        let tokens = Tokens::new(10, 6, 2, 5).with_reasoning(3);
        assert_eq!(
            usage_json(&tokens),
            json!({
                "input_tokens": 10, "output_tokens": 5,
                "input_tokens_details": { "cached_tokens": 6, "cache_write_tokens": 2 },
                "output_tokens_details": { "reasoning_tokens": 3 },
            })
        );
        // What `llmman usage` reads back from it.
        let mut decoder = Decoder::new(Dialect::Responses, Some("application/json"), false);
        decoder.feed(Bytes::from(
            json!({ "usage": usage_json(&tokens) }).to_string(),
        ));
        decoder.finish();
        assert_eq!(decoder.tokens(), Some(tokens));
    }

    // -- a fake llama-server, through the real router -------------------------

    /// The path and JSON body of each request a fake saw.
    type Seen = Arc<Mutex<Vec<(String, serde_json::Value)>>>;

    async fn serve(app: Router) -> u16 {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        port
    }

    /// A token per character, except a `yes` or `no` after a newline: one each.
    fn fake_tokens(text: &str) -> Vec<i64> {
        for (word, id) in [("yes", 1001), ("no", 1002)] {
            if let Some(head) = text.strip_suffix(word).filter(|h| h.ends_with('\n')) {
                return [fake_tokens(head), vec![id]].concat();
            }
        }
        text.chars().map(|c| c as i64).collect()
    }

    /// What a fake llama-server gets wrong.
    #[derive(Clone, Copy, Default)]
    struct Flaws {
        glued_labels: bool,
        same_token: bool,
        overflow: bool,
        /// The boosted read lists a non-label first.
        displaced: bool,
    }

    /// A fake llama-server: a chat renders as `<|user|>..<|end|><|assistant|>tail`,
    /// text tokenizes by [`fake_tokens`], and `/completion` lists the largest
    /// of `dist` (token, probability); a boosted request gets just its labels.
    async fn fake_llama(
        tail: &'static str,
        mut dist: Vec<(i64, f64)>,
        flaws: Flaws,
    ) -> (u16, Seen) {
        dist.sort_by(|a, b| b.1.total_cmp(&a.1));
        let seen = Seen::default();
        let record = |seen: &Seen, path: &'static str, body: &serde_json::Value| {
            let (seen, body) = (seen.clone(), body.clone());
            async move { seen.lock().await.push((path.into(), body)) }
        };
        let (s1, s2, s3) = (seen.clone(), seen.clone(), seen.clone());
        let app = Router::new()
            .route(
                "/apply-template",
                post(move |Json(body): Json<serde_json::Value>| async move {
                    record(&s1, "/apply-template", &body).await;
                    let turn = body["messages"][0]["content"].as_str().unwrap();
                    Json(json!({ "prompt": format!("<|user|>{turn}<|end|><|assistant|>{tail}") }))
                }),
            )
            .route(
                "/tokenize",
                post(move |Json(body): Json<serde_json::Value>| async move {
                    record(&s2, "/tokenize", &body).await;
                    let text = body["content"].as_str().unwrap();
                    let mut tokens = fake_tokens(text);
                    if flaws.same_token {
                        tokens.iter_mut().for_each(|t| *t = if *t == 1002 { 1001 } else { *t });
                    }
                    if flaws.glued_labels && !text.ends_with('\n') {
                        tokens.truncate(tokens.len() - 2);
                        tokens.push(9999);
                    }
                    Json(json!({ "tokens": tokens }))
                }),
            )
            .route(
                "/completion",
                post(move |Json(body): Json<serde_json::Value>| async move {
                    record(&s3, "/completion", &body).await;
                    if flaws.overflow {
                        let error = json!({ "error": { "code": 400, "type": "exceed_context_size_error",
                            "message": "request (9 tokens) exceeds the available context size (8 tokens), try increasing it" }});
                        return (StatusCode::BAD_REQUEST, Json(error)).into_response();
                    }
                    let candidates = if body["post_sampling_probs"] == true {
                        let boosted: Vec<i64> = body["logit_bias"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|entry| entry[0].as_i64().unwrap())
                            .collect();
                        let labels: Vec<_> = dist.iter().filter(|(id, _)| boosted.contains(id)).collect();
                        let total: f64 = labels.iter().map(|(_, p)| p).sum();
                        let mut top: Vec<_> =
                            labels.iter().map(|(id, p)| json!({ "id": id, "prob": p / total })).collect();
                        if flaws.displaced {
                            top.insert(0, json!({ "id": 5000, "prob": 0.9 }));
                            top.pop();
                        }
                        json!({ "top_probs": top })
                    } else {
                        let n = body["n_probs"].as_u64().unwrap() as usize;
                        let top: Vec<_> =
                            dist.iter().take(n).map(|(id, p)| json!({ "id": id, "logprob": p.ln() })).collect();
                        json!({ "top_logprobs": top })
                    };
                    Json(json!({ "truncated": false, "completion_probabilities": [candidates] })).into_response()
                }),
            );
        (serve(app).await, seen)
    }

    /// The labelled probabilities, then thirty tokens of 0.001 that outrank
    /// any label the model puts almost nothing on.
    fn distribution(labelled: &[(i64, f64)]) -> Vec<(i64, f64)> {
        let fillers = (0..30).map(|i| (5000 + i, 0.001));
        labelled.iter().copied().chain(fillers).collect()
    }

    /// The router over a model loaded as `qwen3.5:0.8b` on `port`.
    async fn serve_model(port: u16, engine: Engine) -> String {
        let state = test_state();
        let mut loaded = running_model_fixture_with_engine(engine, None);
        loaded.port = port;
        let key = running_key(&state, "qwen3.5:0.8b");
        state.0.manager.lock().await.running.insert(key, loaded);
        serve_router(state).await
    }

    /// `body` posted as given: `json!` would sort its keys.
    async fn ask_text(url: &str, body: String) -> (StatusCode, String) {
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{url}/v1/systemone"))
            .body(body)
            .send()
            .await
            .unwrap();
        (resp.status(), resp.text().await.unwrap())
    }

    async fn ask(url: &str, body: serde_json::Value) -> (StatusCode, String) {
        ask_text(url, body.to_string()).await
    }

    fn request(questions: serde_json::Value) -> serde_json::Value {
        json!({ "state": STATE, "model": "qwen3.5:0.8b", "questions": questions })
    }

    async fn completions(seen: &Seen) -> Vec<serde_json::Value> {
        let seen = seen.lock().await;
        seen.iter()
            .filter(|(p, _)| p == "/completion")
            .map(|(_, b)| b.clone())
            .collect()
    }

    /// One `/completion` per question, at its labels' probabilities renormalised,
    /// every request shaped as a chat would be.
    #[tokio::test]
    async fn a_local_model_is_read_at_its_first_token() {
        let dist = distribution(&[
            (65, 0.10),
            (66, 0.60),
            (67, 0.20), // A B C
            (48, 0.05),
            (49, 0.15),
            (50, 0.30), // 0 1 2
            (1001, 0.30),
            (1002, 0.10), // yes no
        ]);
        let (port, seen) = fake_llama(CLOSED, dist, Flaws::default()).await;
        let url = serve_model(port, Engine::LlamaServer).await;
        let (status, text) = ask_text(
            &url,
            format!(
                r#"{{"state":"{STATE}","model":"qwen3.5:0.8b","questions":{{
                    "team":{{"type":"choice","instructions":"Which team?",
                             "criteria":{{"billing":null,"technical":"Bugs","sales":null}}}},
                    "urgent":{{"type":"noul","instructions":"Needs an answer today."}},
                    "mood":{{"type":"score","criteria":["Calm","Civil","Angry"]}}}}}}"#
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{text}");

        // In the order asked.
        let Node::Object(reply) = node(&text) else {
            panic!("{text}")
        };
        let Some(Node::Object(answers)) = reply.get("answers") else {
            panic!("{text}")
        };
        let order: Vec<_> = answers.0.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(order, ["team", "urgent", "mood"]);

        let body: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(body["model"], "docker.io/ai/qwen3.5:0.8b");
        let team = &body["answers"]["team"];
        assert_eq!(team["choice"], "technical");
        near(
            team["probabilities"]["billing"].as_f64().unwrap(),
            0.1 / 0.9,
        );
        near(
            team["probabilities"]["technical"].as_f64().unwrap(),
            0.6 / 0.9,
        );
        near(team["probabilities"]["sales"].as_f64().unwrap(), 0.2 / 0.9);
        near(team["x_label_mass"].as_f64().unwrap(), 0.9);
        assert_eq!(team["x_source"], "logprobs");
        near(body["answers"]["urgent"]["noul"].as_f64().unwrap(), 0.75);
        near(
            body["answers"]["urgent"]["x_label_mass"].as_f64().unwrap(),
            0.4,
        );
        let mood = &body["answers"]["mood"];
        near(mood["score"].as_f64().unwrap(), 0.3 + 2.0 * 0.6);
        assert_eq!(
            mood["legend"],
            json!({ "0": "Calm", "1": "Civil", "2": "Angry" })
        );

        // Nothing is generated, and every prompt is counted.
        let sent = completions(&seen).await;
        assert_eq!(sent.len(), 3, "{sent:?}");
        let tokens: u64 = sent
            .iter()
            .map(|b| b["prompt"].as_array().unwrap().len() as u64)
            .sum();
        assert_eq!(
            body["usage"],
            json!({ "input_tokens": tokens, "output_tokens": 0 })
        );
        for read in &sent {
            assert_eq!(
                (&read["n_predict"], &read["cache_prompt"]),
                (&json!(1), &json!(true))
            );
            assert!(read.get("post_sampling_probs").is_none(), "{read}");
        }
        assert_eq!(sent[0]["n_probs"], 3 + TOP_SLACK);

        // Thinking off, the question as one user turn, labels checked as `prompt + label`.
        let seen = seen.lock().await;
        let (path, template) = &seen[0];
        assert_eq!(path, "/apply-template");
        assert_eq!(
            template["chat_template_kwargs"],
            json!({ "enable_thinking": false })
        );
        let turn = format!(
            "{STATE}\n\nQuestion: Which team?\nA: billing\nB: technical - Bugs\nC: sales\n\
             Answer with the letter of one option only."
        );
        assert_eq!(
            template["messages"],
            json!([{ "role": "user", "content": turn }])
        );
        let label_check = |(p, b): &&(String, serde_json::Value)| {
            p == "/tokenize" && b["content"].as_str().unwrap().ends_with('A')
        };
        let (_, check) = seen.iter().find(label_check).unwrap();
        assert_eq!(
            (&check["add_special"], &check["parse_special"]),
            (&json!(true), &json!(true))
        );
    }

    /// A label below the top is invisible to `n_probs`, so it is read with every label boosted.
    #[tokio::test]
    async fn a_label_outside_the_top_is_read_with_a_boost() {
        // `C` holds a billionth: below thirty tokens of a thousandth.
        let dist = distribution(&[(65, 0.30), (66, 0.60), (67, 1e-9)]);
        let (port, seen) = fake_llama(CLOSED, dist, Flaws::default()).await;
        let url = serve_model(port, Engine::LlamaServer).await;
        let choice = json!({ "team": { "type": "choice", "criteria": { "a": null, "b": null, "c": null } } });
        let (status, text) = ask(&url, request(choice)).await;
        assert_eq!(status, StatusCode::OK, "{text}");

        let sent = completions(&seen).await;
        assert_eq!(sent.len(), 2, "{sent:?}");
        let boost = &sent[1];
        assert_eq!(boost["post_sampling_probs"], true);
        assert_eq!(boost["samplers"], json!(["temperature"]));
        assert_eq!(boost["top_k"], 0);
        assert_eq!(
            boost["logit_bias"],
            json!([[65, 100.0], [66, 100.0], [67, 100.0]])
        );

        let body: serde_json::Value = serde_json::from_str(&text).unwrap();
        let team = &body["answers"]["team"];
        assert_eq!(team["choice"], "b");
        let p = &team["probabilities"];
        near(p["a"].as_f64().unwrap(), 1.0 / 3.0);
        near(p["b"].as_f64().unwrap(), 2.0 / 3.0);
        // Not zero, which is all the top alone would say; the mass is the labels' real one.
        assert!(
            (p["c"].as_f64().unwrap() - 1e-9 / 0.900000001).abs() < 1e-15,
            "{p}"
        );
        near(team["x_label_mass"].as_f64().unwrap(), 0.900000001);
    }

    /// Each way a local model cannot be read is a 400 that says why.
    #[tokio::test]
    async fn a_local_model_that_cannot_be_read_is_refused() {
        let noul = json!({ "q": { "type": "noul", "instructions": "x" } });
        let ab = json!({ "q": { "type": "choice", "criteria": { "a": null, "b": null } } });
        let abc =
            json!({ "q": { "type": "choice", "criteria": { "a": null, "b": null, "c": null } } });
        let on = |f: fn(&mut Flaws)| {
            let mut flaws = Flaws::default();
            f(&mut flaws);
            flaws
        };
        let cases = [
            (
                "<think>\n",
                Flaws::default(),
                &noul,
                "reasoning block",
                false,
            ),
            (
                CLOSED,
                on(|f| f.glued_labels = true),
                &ab,
                "not one distinct token",
                false,
            ),
            (
                CLOSED,
                on(|f| f.same_token = true),
                &noul,
                "not one distinct token",
                false,
            ),
            (
                CLOSED,
                on(|f| f.overflow = true),
                &ab,
                "exceeds the available context size (8 tokens)",
                true,
            ),
            (
                CLOSED,
                on(|f| f.displaced = true),
                &abc,
                "almost no probability",
                true,
            ),
        ];
        for (tail, flaws, questions, why, reads) in cases {
            // `c` is crowded out of the top, so the boosted read is needed.
            let dist = distribution(&[(65, 0.5), (66, 0.5), (67, 1e-9)]);
            let (port, seen) = fake_llama(tail, dist, flaws).await;
            let url = serve_model(port, Engine::LlamaServer).await;
            let (status, text) = ask(&url, request(questions.clone())).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{why}: {text}");
            assert!(
                text.contains(why) && text.contains(r#"question \"q\""#),
                "{why}: {text}"
            );
            assert_eq!(!completions(&seen).await.is_empty(), reads, "{why}");
        }
    }

    /// Only llama-server gives token probabilities the way this reads them.
    #[tokio::test]
    async fn another_local_engine_is_turned_down_by_name() {
        let url = serve_model(1, Engine::Vllm).await;
        let (status, text) = ask(
            &url,
            request(json!({ "q": { "type": "noul", "instructions": "x" } })),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{text}");
        assert!(
            text.contains("served by vllm") && text.contains("llama-server"),
            "{text}"
        );
    }

    /// Refused before any model is loaded for them.
    #[tokio::test]
    async fn requests_that_cannot_be_served_never_reach_a_model() {
        let url = serve_router(test_state()).await;
        let (status, text) = ask(&url, json!({ "state": "s" })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{text}");
        let body: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            body["detail"][0],
            json!({ "loc": ["body", "model"], "msg": "Field required", "type": "missing" })
        );
        let (status, text) = ask_text(&url, "{".into()).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(text.contains("json_invalid"), "{text}");

        let q = json!({ "q": { "type": "noul", "instructions": "x" } });
        let options: serde_json::Map<_, _> =
            (0..27).map(|i| (format!("o{i}"), json!(null))).collect();
        let too_wide = request(json!({ "q": { "type": "choice", "criteria": options } }));
        let with = |key: &str, value: &str| {
            let mut body = request(q.clone());
            body[key] = json!(value);
            body
        };
        let cases = [
            (too_wide, "A to Z"),
            // A local model is read before it thinks, so has no variant.
            (with("reasoning_effort", "high"), "needs a hosted model"),
            (
                {
                    let mut conflict =
                        with("model", "llmman.provider/anthropic/claude-sonnet-5-5/max");
                    conflict["reasoning_effort"] = json!("low");
                    conflict
                },
                r#"names variant \"max\""#,
            ),
        ];
        for (body, why) in cases {
            let (status, text) = ask(&url, body).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{why}: {text}");
            assert!(text.contains(why), "{why}: {text}");
        }
    }

    // -- hosted providers -----------------------------------------------------

    fn hosted_target(base: &str, wire: Wire, model: &str) -> Target {
        Target::Remote(Arc::new(RemoteTarget {
            provider: "mockprov".into(),
            base_url: base.into(),
            wire,
            model: model.into(),
            max_output: None,
            cost: None,
            api_key: Some("sk-test".into()),
        }))
    }

    /// A fake provider: what it saw, and the most it had in flight at once.
    struct Provider {
        base: String,
        seen: Seen,
        peak: Arc<AtomicUsize>,
    }

    /// What a fake provider says to a question, from the system message it got.
    type Say = fn(&str) -> (StatusCode, String);

    /// The labels a system message asks for: `A`, `B`, ...
    fn asked_labels(system: &str) -> Vec<String> {
        let n = system.matches("<probability>").count();
        (b'A'..)
            .take(n)
            .map(|c| char::from(c).to_string())
            .collect()
    }

    /// All the probability on the last label.
    fn last_label(system: &str) -> (StatusCode, String) {
        let labels = asked_labels(system);
        let stated: serde_json::Map<_, _> = (0..labels.len())
            .map(|i| (labels[i].clone(), json!(u8::from(i + 1 == labels.len()))))
            .collect();
        (
            StatusCode::OK,
            serde_json::Value::Object(stated).to_string(),
        )
    }

    /// Holds a request until `gate` have arrived, so a batch is in flight
    /// together however slow the machine, then a moment more so one that went
    /// past the bound would show.
    async fn in_flight(gate: usize, started: &AtomicUsize, now: &AtomicUsize, peak: &AtomicUsize) {
        started.fetch_add(1, SeqCst);
        peak.fetch_max(now.fetch_add(1, SeqCst) + 1, SeqCst);
        let deadline = Instant::now() + Duration::from_secs(10);
        while started.load(SeqCst) < gate && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        if gate > 1 {
            tokio::time::sleep(Duration::from_millis(30)).await;
        }
        now.fetch_sub(1, SeqCst);
    }

    async fn fake_openai(say: Say) -> Provider {
        fake_gated_openai(say, 1).await
    }

    /// An OpenAI-wire provider answering `/v1/chat/completions` with `say`,
    /// holding requests until `gate` are in flight.
    async fn fake_gated_openai(say: Say, gate: usize) -> Provider {
        let (seen, peak) = (Seen::default(), Arc::new(AtomicUsize::new(0)));
        let (record, peak_in) = (seen.clone(), peak.clone());
        let (started, now) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move |Json(body): Json<serde_json::Value>| async move {
                in_flight(gate, &started, &now, &peak_in).await;
                let system = body["messages"][0]["content"].as_str().unwrap().to_owned();
                record.lock().await.push(("/v1/chat/completions".into(), body));
                let (status, text) = say(&system);
                let reply = if status.is_success() {
                    json!({
                        "choices": [{ "message": { "role": "assistant", "content": text }, "finish_reason": "stop" }],
                        "usage": { "prompt_tokens": 7, "completion_tokens": 5,
                                   "prompt_tokens_details": { "cached_tokens": 3 },
                                   "completion_tokens_details": { "reasoning_tokens": 2 } },
                    })
                } else {
                    json!({ "error": { "message": text } })
                };
                (status, Json(reply))
            }),
        );
        let port = serve(app).await;
        Provider {
            base: format!("http://127.0.0.1:{port}/v1"),
            seen,
            peak,
        }
    }

    /// A Messages-API provider answering `/v1/messages` with a stream.
    async fn fake_anthropic(say: Say) -> Provider {
        let seen = Seen::default();
        let record = seen.clone();
        let app = Router::new().route(
            "/v1/messages",
            post(move |Json(body): Json<serde_json::Value>| async move {
                let system = body["system"][0]["text"].as_str().unwrap().to_owned();
                record.lock().await.push(("/v1/messages".into(), body));
                let (status, text) = say(&system);
                if !status.is_success() {
                    let error = json!({ "type": "error", "error": { "type": "x", "message": text } });
                    return (status, Json(error)).into_response();
                }
                let event = |name: &str, data: serde_json::Value| format!("event: {name}\ndata: {data}\n\n");
                let stream = [
                    event("message_start", json!({ "type": "message_start", "message": {
                        "id": "m", "model": "m", "usage": { "input_tokens": 111, "output_tokens": 1 } } })),
                    event("content_block_start", json!({ "type": "content_block_start", "index": 0,
                        "content_block": { "type": "text", "text": "" } })),
                    event("content_block_delta", json!({ "type": "content_block_delta", "index": 0,
                        "delta": { "type": "text_delta", "text": text } })),
                    event("message_delta", json!({ "type": "message_delta",
                        "delta": { "stop_reason": "end_turn" }, "usage": { "output_tokens": 22 } })),
                    event("message_stop", json!({ "type": "message_stop" })),
                ]
                .concat();
                ([("content-type", "text/event-stream")], stream).into_response()
            }),
        );
        let port = serve(app).await;
        Provider {
            base: format!("http://127.0.0.1:{port}/v1"),
            seen,
            peak: Arc::default(),
        }
    }

    fn choice_question(options: usize) -> Question {
        let criteria: Vec<_> = (0..options)
            .map(|i| format!(r#""option {i}":null"#))
            .collect();
        question(&format!(
            r#"{{"type":"choice","instructions":"Pick","criteria":{{{}}}}}"#,
            criteria.join(",")
        ))
    }

    fn hosted<'a>(client: &'a Client, target: &'a Target, effort: Option<&'a str>) -> Hosted<'a> {
        let Target::Remote(remote) = target else {
            panic!("not hosted: {target:?}")
        };
        Hosted {
            client,
            target,
            remote,
            effort,
        }
    }

    async fn ask_from(
        target: &Target,
        effort: Option<&str>,
        q: &Question,
    ) -> Result<Scored, Failure> {
        hosted(&Client::new(), target, effort).ask(STATE, q).await
    }

    /// The model is shown the question, told to reply with its probabilities as
    /// JSON, and its reply is the answer; the variant is `reasoning_effort`.
    #[tokio::test]
    async fn a_hosted_model_is_asked_for_its_probabilities_in_json() {
        let say: Say = |_| {
            (
                StatusCode::OK,
                "```json\n{\"A\": 0.2, \"B\": 0.8}\n```".into(),
            )
        };
        let provider = fake_openai(say).await;
        let target = hosted_target(&provider.base, Wire::OpenAi, "mock-model");
        let question = choice_question(2);

        let got = ask_from(&target, Some("high"), &question).await.unwrap();
        assert_eq!(got.probabilities, [0.2, 0.8]);
        assert_eq!((got.mass, got.source), (None, Source::Elicited));
        // Cache and reasoning counts reach the ledger too.
        assert_eq!(got.tokens, Tokens::new(7, 3, 0, 5).with_reasoning(2));

        let seen = provider.seen.lock().await;
        let sent = &seen[0].1;
        assert_eq!(
            (
                &sent["model"],
                &sent["reasoning_effort"],
                &sent["temperature"]
            ),
            (&json!("mock-model"), &json!("high"), &json!(0))
        );
        for key in ["response_format", "tools", "max_tokens"] {
            assert!(sent.get(key).is_none(), "{key} in {sent}");
        }
        assert_eq!(sent["messages"][0]["role"], "system");
        let system = sent["messages"][0]["content"].as_str().unwrap();
        assert!(
            system.contains(r#"like {"A": <probability>, "B": <probability>}."#),
            "{system}"
        );
        let user = state_and_question(STATE, &question, &question.labels());
        assert_eq!(
            sent["messages"][1],
            json!({ "role": "user", "content": user })
        );
        drop(seen);

        // Without a variant the model's own default applies.
        ask_from(&target, None, &question).await.unwrap();
        assert!(provider.seen.lock().await[1]
            .1
            .get("reasoning_effort")
            .is_none());
    }

    /// `anthropic/claude-sonnet-5-5/xhigh` on the wire: adaptive thinking at
    /// that effort, which a forced tool would have turned off.
    #[tokio::test]
    async fn an_anthropic_model_thinks_at_the_variant_asked_for() {
        let provider = fake_anthropic(last_label).await;
        let target = hosted_target(&provider.base, Wire::Anthropic, "claude-sonnet-5-5");
        let question = choice_question(3);

        let got = ask_from(&target, Some("xhigh"), &question).await.unwrap();
        assert_eq!(got.probabilities, [0.0, 0.0, 1.0]);
        assert_eq!(got.tokens, Tokens::new(111, 0, 0, 22));
        ask_from(&target, None, &question).await.unwrap();
        ask_from(&target, Some("none"), &question).await.unwrap();

        let seen = provider.seen.lock().await;
        let xhigh = &seen[0].1;
        assert_eq!(xhigh["thinking"]["type"], "adaptive");
        assert_eq!(xhigh["output_config"]["effort"], "xhigh");
        assert_eq!(xhigh["model"], "claude-sonnet-5-5");
        for key in ["tools", "tool_choice", "temperature"] {
            assert!(xhigh.get(key).is_none(), "{key} in {xhigh}");
        }
        assert!(seen[1].1.get("thinking").is_none(), "the default applies");
        assert_eq!(seen[2].1["thinking"], json!({ "type": "disabled" }));
    }

    /// A provider's refusal of the caller keeps its status; its own failure is a bad gateway.
    #[tokio::test]
    async fn a_providers_refusal_keeps_its_status() {
        let denied = fake_openai(|_| (StatusCode::UNAUTHORIZED, "invalid x-api-key".into())).await;
        let target = hosted_target(&denied.base, Wire::OpenAi, "mock-model");
        let failure = ask_from(&target, None, &choice_question(2))
            .await
            .unwrap_err();
        assert!(
            matches!(&failure, Failure::Provider(StatusCode::UNAUTHORIZED, m) if m == "invalid x-api-key"),
            "{failure:?}"
        );
        let response = failure.for_question("q").respond().unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            body["error"]["message"],
            "question \"q\": invalid x-api-key"
        );

        let broken =
            fake_openai(|_| (StatusCode::INTERNAL_SERVER_ERROR, "overloaded".into())).await;
        let target = hosted_target(&broken.base, Wire::OpenAi, "mock-model");
        let failure = ask_from(&target, None, &choice_question(2))
            .await
            .unwrap_err();
        assert_eq!(
            failure.respond().expect_err("a 502, not a relayed 500").1,
            StatusCode::BAD_GATEWAY
        );

        // The Messages API's error shape reads the same.
        let slow = fake_anthropic(|_| (StatusCode::TOO_MANY_REQUESTS, "slow down".into())).await;
        let target = hosted_target(&slow.base, Wire::Anthropic, "claude-sonnet-5-5");
        let failure = ask_from(&target, None, &choice_question(2))
            .await
            .unwrap_err();
        assert!(
            matches!(&failure, Failure::Provider(StatusCode::TOO_MANY_REQUESTS, m) if m == "slow down"),
            "{failure:?}"
        );
    }

    #[tokio::test]
    async fn a_reply_with_no_probabilities_is_a_bad_gateway() {
        let provider = fake_openai(|_| (StatusCode::OK, "I would say billing.".into())).await;
        let target = hosted_target(&provider.base, Wire::OpenAi, "mock-model");
        let failure = ask_from(&target, None, &choice_question(2))
            .await
            .unwrap_err();
        let Failure::Provider(StatusCode::BAD_GATEWAY, message) = &failure else {
            panic!("{failure:?}")
        };
        assert!(
            message.contains("mock-model did not state probabilities")
                && message.contains("no JSON object"),
            "{message}"
        );
    }

    /// Questions go out a few at a time, come back in the order asked, and are named when they fail.
    #[tokio::test]
    async fn hosted_questions_are_asked_a_few_at_a_time_and_answered_in_order() {
        let provider = fake_gated_openai(last_label, HOSTED_CONCURRENCY).await;
        let target = hosted_target(&provider.base, Wire::OpenAi, "mock-model");
        let questions: Vec<_> = (2..8).map(choice_question).collect();
        let client = Client::new();
        let scored = hosted(&client, &target, None)
            .ask_all(STATE, &questions)
            .await
            .unwrap();
        let widths: Vec<_> = scored.iter().map(|s| s.probabilities.len()).collect();
        assert_eq!(widths, [2, 3, 4, 5, 6, 7]);
        assert!(scored.iter().all(|s| s.probabilities.last() == Some(&1.0)));
        assert_eq!(provider.seen.lock().await.len(), 6);
        assert_eq!(provider.peak.load(SeqCst), HOSTED_CONCURRENCY);

        let flaky = fake_openai(|system| match asked_labels(system).len() {
            3 => (StatusCode::OK, "no idea".into()),
            _ => last_label(system),
        })
        .await;
        let target = hosted_target(&flaky.base, Wire::OpenAi, "mock-model");
        let mut named = vec![choice_question(2), choice_question(3)];
        named[1].id = "third".into();
        let failure = hosted(&client, &target, None)
            .ask_all(STATE, &named)
            .await
            .unwrap_err();
        assert!(
            matches!(&failure, Failure::Provider(_, m) if m.starts_with("question \"third\": ")),
            "{failure:?}"
        );
    }

    /// However a hosted model words its probabilities.
    #[test]
    fn stated_probabilities_are_read_however_they_are_worded() {
        let labels = strings(&["A", "B", "C"]);
        let read = |reply: &str| parse_elicited(reply, &labels);
        let want =
            |got: Vec<f64>, want: &[f64]| got.iter().zip(want).for_each(|(g, w)| near(*g, *w));
        want(
            read(r#"{"A": 0.5, "B": 0.25, "C": 0.25}"#).unwrap(),
            &[0.5, 0.25, 0.25],
        );
        // Fenced, introduced, nested, as text or percentages, rounded below 1: all normalised.
        want(
            read("Here you go:\n```json\n{\"A\":0.1,\"B\":0.7,\"C\":0.1}\n```\nDone.").unwrap(),
            &[1.0 / 9.0, 7.0 / 9.0, 1.0 / 9.0],
        );
        want(
            read(r#"{"probabilities": {"A": "20%", "B": "60%", "C": "20%"}}"#).unwrap(),
            &[0.2, 0.6, 0.2],
        );
        want(read(r#"{"A": 0, "B": 1}"#).unwrap(), &[0.0, 1.0, 0.0]);

        let positive = "do not add up to a positive number";
        for (reply, why) in [
            ("billing", "no JSON object"),
            ("{A: 1}", "does not parse"),
            ("[0.5, 0.5]", "no JSON object"),
            (r#"{"x": 1}"#, "none of the labels"),
            (r#"{"A": 0, "B": 0, "C": 0}"#, positive),
            (r#"{"A": 1e308, "B": 1e308}"#, positive),
            (r#"{"A": -0.5, "B": 1}"#, "not a non-negative number"),
            (r#"{"A": "likely", "B": 1}"#, "not a non-negative number"),
            (r#"{"A": null, "B": 1}"#, "not a non-negative number"),
        ] {
            let err = read(reply).unwrap_err();
            assert!(err.contains(why), "{reply:?}: {err}");
        }
        assert_eq!(
            reply_text(&json!({ "choices": [{ "message": { "content": "hi" } }] })),
            "hi"
        );
        let parts = json!({ "choices": [{ "message": { "content": [{ "text": "a" }, { "text": "b" }] } }] });
        assert_eq!(reply_text(&parts), "ab");
        assert_eq!(reply_text(&json!({})), "");
        let prompt = elicitation_prompt(&strings(&["yes", "no"]));
        assert!(
            prompt.contains(r#"like {"yes": <probability>, "no": <probability>}."#),
            "{prompt}"
        );
    }
}
