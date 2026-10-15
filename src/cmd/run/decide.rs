//! `llmman run` for a decision model (Clef, Nimble, ...), after ollama's
//! `cmd/decide.go`. It generates no text: it answers typed questions about a
//! text or image through llama-server's `/v1/systemone`, with a probability
//! for each answer.
//!
//! `--ask` is a question, yes/no unless `--choice` labels or `--level`
//! scale follow it. The text is the words after the model, or stdin.
//! Interactively, the first message is the text and each line after it a
//! yes/no question; with `--ask`, each message is a new text.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::io::{self, Read};
use std::time::Instant;

use anyhow::Context;
use base64::Engine as _;
use clap::{Arg, ArgAction, ArgMatches, Args as _, Command, FromArgMatches};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{chat_client, extract_files, start_spinner, MediaFile, RunArgs, SpinnerGuard, Turns};
use crate::daemon;
use crate::ordered::Ordered;

// ---------------------------------------------------------------------------
// Questions: --ask, --choice, --level
// ---------------------------------------------------------------------------

/// What a question is answered with.
#[derive(Debug, Clone, PartialEq)]
enum Kind {
    /// Yes or no.
    YesNo,
    /// One of these labels, each with an optional description.
    Choice(Vec<(String, Option<String>)>),
    /// A level of this scale, lowest first.
    Level(Vec<String>),
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Question {
    text: String,
    kind: Kind,
}

impl Question {
    fn yes_no(text: &str) -> Self {
        Self {
            text: text.to_string(),
            kind: Kind::YesNo,
        }
    }
}

/// The `--ask`, `--choice` and `--level` flags in the order given: a
/// `--choice` or `--level` answers the `--ask` before it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Asked(Vec<(Flag, String)>);

impl Asked {
    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The questions asked; a `--choice` or `--level` with no `--ask`
    /// before it, or both on one question, is an error.
    pub(super) fn questions(&self) -> anyhow::Result<Vec<Question>> {
        let mut questions: Vec<Question> = Vec::new();
        for (flag, value) in &self.0 {
            if *flag == Flag::Ask {
                questions.push(Question::yes_no(value));
                continue;
            }
            let question = questions.last_mut().ok_or_else(|| {
                anyhow::anyhow!("--{} must come after the --ask it answers", flag.name())
            })?;
            match (&mut question.kind, flag) {
                (kind @ Kind::YesNo, Flag::Choice) => *kind = Kind::Choice(vec![label(value)]),
                (Kind::Choice(labels), Flag::Choice) => labels.push(label(value)),
                (kind @ Kind::YesNo, _) => *kind = Kind::Level(vec![value.clone()]),
                (Kind::Level(levels), Flag::Level) => levels.push(value.clone()),
                _ => anyhow::bail!("a question takes --choice or --level, not both"),
            }
        }
        Ok(questions)
    }
}

/// A `--choice`: `label`, or `label=description`.
fn label(value: &str) -> (String, Option<String>) {
    match value.split_once('=') {
        Some((label, description)) => (label.to_string(), Some(description.to_string())),
        None => (value.to_string(), None),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Flag {
    Ask,
    Choice,
    Level,
}

impl Flag {
    const ALL: [Flag; 3] = [Flag::Ask, Flag::Choice, Flag::Level];

    fn name(self) -> &'static str {
        match self {
            Flag::Ask => "ask",
            Flag::Choice => "choice",
            Flag::Level => "level",
        }
    }
}

impl clap::Args for Asked {
    fn augment_args(cmd: Command) -> Command {
        let flag = |flag: Flag, value_name: &'static str, help: &'static str| {
            Arg::new(flag.name())
                .long(flag.name())
                .value_name(value_name)
                .action(ArgAction::Append)
                .value_parser(clap::value_parser!(String))
                .help(help)
        };
        cmd.arg(flag(
            Flag::Ask,
            "QUESTION",
            "Decision models only: ask a question about the text, answered yes or no \
             unless --choice or --level follow it. Repeatable",
        ))
        .arg(flag(
            Flag::Choice,
            "LABEL[=DESCRIPTION]",
            "A label to answer the --ask before it with. Repeatable",
        ))
        .arg(flag(
            Flag::Level,
            "LEVEL",
            "A level of the scale to answer the --ask before it on, lowest first. Repeatable",
        ))
    }

    fn augment_args_for_update(cmd: Command) -> Command {
        Self::augment_args(cmd)
    }
}

impl FromArgMatches for Asked {
    fn from_arg_matches(matches: &ArgMatches) -> Result<Self, clap::Error> {
        let mut flags: Vec<(usize, Flag, String)> = Vec::new();
        for flag in Flag::ALL {
            if let (Some(indices), Some(values)) = (
                matches.indices_of(flag.name()),
                matches.get_many::<String>(flag.name()),
            ) {
                flags.extend(indices.zip(values).map(|(i, v)| (i, flag, v.clone())));
            }
        }
        flags.sort_by_key(|(index, ..)| *index);
        Ok(Self(
            flags
                .into_iter()
                .map(|(_, flag, value)| (flag, value))
                .collect(),
        ))
    }

    fn update_from_arg_matches(&mut self, matches: &ArgMatches) -> Result<(), clap::Error> {
        *self = Self::from_arg_matches(matches)?;
        Ok(())
    }
}

/// Other models fail on a decision model's flags rather than ignore them.
pub(super) fn reject_questions(args: &RunArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        args.ask.is_empty() && args.format.is_none() && !args.verbose,
        "{} is not a decision model, so --ask, --choice, --level, --format and --verbose \
         don't apply to it",
        args.model
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Wire types of /v1/systemone
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    state: Value,
    /// Data URLs, shared by every question.
    #[serde(skip_serializing_if = "<[String]>::is_empty")]
    images: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<&'a str>,
    questions: Ordered<QuestionBody<'a>>,
}

#[derive(Serialize)]
struct QuestionBody<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    instructions: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    criteria: Option<Criteria<'a>>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Criteria<'a> {
    /// Label to description, `null` for a label without one.
    Labels(Ordered<Option<&'a str>>),
    Levels(&'a [String]),
}

impl<'a> QuestionBody<'a> {
    fn new(question: &'a Question) -> Self {
        let (kind, criteria) = match &question.kind {
            Kind::YesNo => ("noul", None),
            Kind::Choice(labels) => {
                let labels = labels
                    .iter()
                    .map(|(label, description)| (label.clone(), description.as_deref()))
                    .collect();
                ("choice", Some(Criteria::Labels(Ordered(labels))))
            }
            Kind::Level(levels) => ("score", Some(Criteria::Levels(levels))),
        };
        Self {
            kind,
            instructions: &question.text,
            criteria,
        }
    }
}

/// The text to judge: a string, or a JSON object or array given as one,
/// which the model reads as JSON.
fn state_value(text: &str) -> Value {
    let trimmed = text.trim();
    if trimmed.starts_with(['{', '[']) {
        if let Ok(value) = serde_json::from_str(trimmed) {
            return value;
        }
    }
    Value::String(text.to_string())
}

#[derive(Serialize, Deserialize)]
struct Response {
    #[serde(default)]
    model: String,
    answers: Ordered<Value>,
    #[serde(default)]
    usage: Value,
}

/// One answer. Which fields it has depends on its `type`.
#[derive(Deserialize)]
struct Answer {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    noul: f64,
    #[serde(default)]
    choice: String,
    #[serde(default)]
    score: f64,
    /// Level index to description.
    #[serde(default)]
    legend: BTreeMap<String, Value>,
    #[serde(default)]
    probabilities: BTreeMap<String, f64>,
}

fn percent(probability: f64) -> String {
    format!("{:.0}%", probability * 100.0)
}

impl Answer {
    /// The answer as the cells of a line: the likelier answer to a yes/no
    /// question, the label chosen, or the most likely level of a scale,
    /// each with its probability, and for a scale the score itself.
    fn cells(&self) -> Result<Vec<String>, String> {
        match self.kind.as_str() {
            "noul" if self.noul < 0.5 => Ok(vec!["no".into(), percent(1.0 - self.noul)]),
            "noul" => Ok(vec!["yes".into(), percent(self.noul)]),
            "choice" => {
                let probability = self.probabilities.get(&self.choice).copied();
                Ok(vec![
                    self.choice.clone(),
                    percent(probability.unwrap_or(0.0)),
                ])
            }
            "score" => {
                // Not the level nearest the score: an answer split between
                // the ends of the scale scores in the middle.
                let probability = |i: usize| {
                    self.probabilities
                        .get(&i.to_string())
                        .copied()
                        .unwrap_or(0.0)
                };
                let mut level = 0;
                for i in 1..self.legend.len() {
                    if probability(i) > probability(level) {
                        level = i;
                    }
                }
                let name = match self.legend.get(&level.to_string()) {
                    Some(Value::String(name)) => name.clone(),
                    Some(other) => other.to_string(),
                    None => String::new(),
                };
                Ok(vec![
                    name,
                    percent(probability(level)),
                    format!(
                        "{:.1} / {}",
                        self.score,
                        self.legend.len().saturating_sub(1)
                    ),
                ])
            }
            other => Err(format!("unknown type {other:?}")),
        }
    }
}

/// `questions` answered from `response`, a line each, with the question
/// first when there is more than one.
fn render(questions: &[&Question], response: &Response) -> anyhow::Result<String> {
    let mut rows = Vec::new();
    for question in questions {
        let (_, raw) = response
            .answers
            .0
            .iter()
            .find(|(id, _)| *id == question.text)
            .with_context(|| format!("no answer to {:?}", question.text))?;
        let answer: Answer = serde_json::from_value(raw.clone())
            .with_context(|| format!("answer to {:?}", question.text))?;
        let mut row = Vec::new();
        if questions.len() > 1 {
            row.push(question.text.clone());
        }
        row.extend(
            answer
                .cells()
                .map_err(|e| anyhow::anyhow!("answer to {:?} has {e}", question.text))?,
        );
        rows.push(row);
    }
    Ok(crate::fmt::columns(&rows, 2))
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

/// Text, and the images in it, to answer questions about.
struct Input {
    text: String,
    /// Data URLs.
    images: Vec<String>,
}

impl Input {
    fn is_empty(&self) -> bool {
        self.text.is_empty() && self.images.is_empty()
    }
}

fn data_url(file: &MediaFile) -> String {
    let data = base64::engine::general_purpose::STANDARD.encode(&file.data);
    format!("data:{};base64,{data}", file.content_type)
}

/// A conversation with a decision model over `/v1/systemone`.
struct Session<'a> {
    // One runtime for every turn, as for a chat (see `chat_submit`).
    rt: tokio::runtime::Runtime,
    client: Client,
    model: &'a str,
    args: &'a RunArgs,
    /// From `--ask`; with none, they are typed.
    questions: Vec<Question>,
    multimodal: bool,
    /// Interactive, without `--ask`: the text the typed questions are about.
    state: Option<Input>,
}

/// `args` parsed again from the command line. clap gives `run`'s prompt
/// everything from its first word on, so a chat can be asked about `--think`
/// unquoted; a decision model's flags may follow its text, as in ollama
/// (cobra), and `--` marks the rest as text.
fn reparse(args: &RunArgs) -> anyhow::Result<RunArgs> {
    let argv: Vec<OsString> = std::env::args_os()
        .skip_while(|a| a != "run")
        .skip(1)
        .collect();
    if argv.is_empty() {
        return Ok(args.clone());
    }
    parse_argv(argv)
}

fn parse_argv(argv: Vec<OsString>) -> anyhow::Result<RunArgs> {
    let command = RunArgs::augment_args(
        Command::new("llmman run").override_usage("llmman run [OPTIONS] <MODEL> [PROMPT]..."),
    )
    .mut_arg("prompt", |prompt| {
        prompt.trailing_var_arg(false).allow_hyphen_values(false)
    });
    command
        .try_get_matches_from(std::iter::once("run".into()).chain(argv))
        .and_then(|matches| RunArgs::from_arg_matches(&matches))
        .map_err(|e| anyhow::anyhow!("{}", e.to_string().trim_start_matches("error: ").trim_end()))
}

/// `model` asked the questions of `args` about its text or stdin, or
/// interactively.
pub(super) fn run(
    model: &str,
    args: &RunArgs,
    multimodal: bool,
    interactive: bool,
) -> anyhow::Result<()> {
    let args = reparse(args)?;
    super::check_keepalive(args.keepalive.as_deref())?;
    let mut session = Session {
        rt: tokio::runtime::Runtime::new().context("start tokio runtime")?,
        client: chat_client(None)?,
        model,
        questions: args.ask.questions()?,
        args: &args,
        multimodal,
        state: None,
    };
    if interactive {
        eprintln!("{}", session.hint());
        return super::run_interactive_tty(&mut session);
    }

    let mut text = args.prompt.join(" ");
    if text.is_empty() {
        io::stdin().read_to_string(&mut text)?;
    }
    let input = session.input(&text)?;
    if session.questions.is_empty() || input.is_empty() {
        let name = &args.model;
        anyhow::bail!(
            "{name} answers questions about text, e.g. \
             llmman run {name} --ask \"Is a refund being requested?\" < ticket.txt"
        );
    }
    session.decide(&input, &session.questions)
}

impl Session<'_> {
    /// What to type next (ollama's prompt placeholder).
    fn hint(&self) -> &'static str {
        if !self.questions.is_empty() {
            "Enter text to evaluate."
        } else if self.state.is_some() {
            "Ask yes/no questions about the text, one per line."
        } else {
            "Enter text to evaluate, then ask yes/no questions about it, one per line."
        }
    }

    /// `content`, with the image files it names read if the model takes images.
    fn input(&self, content: &str) -> anyhow::Result<Input> {
        let (text, images) = if self.multimodal {
            let (text, files) = extract_files(content)?;
            (text, files.iter().map(data_url).collect())
        } else {
            (content.to_string(), Vec::new())
        };
        Ok(Input {
            text: text.trim().to_string(),
            images,
        })
    }

    /// Asks `questions` about `input` and prints the answers. A repeated
    /// question is asked once: its text is its name.
    fn decide(&self, input: &Input, questions: &[Question]) -> anyhow::Result<()> {
        let mut unique: Vec<&Question> = Vec::new();
        for question in questions {
            match unique.iter().position(|q| q.text == question.text) {
                Some(i) => unique[i] = question,
                None => unique.push(question),
            }
        }
        let request = Request {
            model: self.model,
            state: state_value(&input.text),
            images: &input.images,
            keep_alive: self.args.keepalive.as_deref(),
            questions: Ordered(
                unique
                    .iter()
                    .map(|q| (q.text.clone(), QuestionBody::new(q)))
                    .collect(),
            ),
        };
        let started = Instant::now();
        let Some(body) = self.rt.block_on(self.post(&request))? else {
            return Ok(());
        };
        let elapsed = started.elapsed();
        let mut response: Response =
            serde_json::from_str(&body).context("decode the decision response")?;
        if self.args.format.is_some() {
            response.model = self.model.to_string();
            println!("{}", serde_json::to_string(&response)?);
        } else {
            print!("{}", render(&unique, &response)?);
        }
        if self.args.verbose {
            eprintln!("total duration:       {elapsed:?}");
            let tokens = response.usage["input_tokens"].as_u64().unwrap_or(0);
            if tokens > 0 {
                eprintln!("prompt eval count:    {tokens} token(s)");
            }
        }
        Ok(())
    }

    /// The response body, or `None` when Ctrl-C ended the request first.
    async fn post(&self, request: &Request<'_>) -> anyhow::Result<Option<String>> {
        let mut spinner = SpinnerGuard(start_spinner());
        let ctrl_c = tokio::signal::ctrl_c();
        tokio::pin!(ctrl_c);
        let send = async {
            let resp = self
                .client
                .post(format!("{}/v1/systemone", daemon::server()))
                .json(request)
                .send()
                .await
                .context("connect to llmman serve")?;
            let status = resp.status();
            let body = resp.text().await.context("read response")?;
            Ok::<_, anyhow::Error>((status, body))
        };
        let (status, body) = tokio::select! {
            result = send => result?,
            _ = &mut ctrl_c => return Ok(None),
        };
        spinner.stop();
        anyhow::ensure!(
            status.is_success(),
            "{}",
            daemon::api_error(&body).unwrap_or(body)
        );
        Ok(Some(body))
    }
}

impl Turns for Session<'_> {
    fn multimodal(&self) -> bool {
        self.multimodal
    }

    fn submit(&mut self, content: String) -> anyhow::Result<()> {
        let input = self.input(&content)?;
        if input.is_empty() {
            return Ok(());
        }
        // A failed question does not end the session.
        let report = |result: anyhow::Result<()>| {
            if let Err(e) = result {
                eprintln!("error: {e:#}");
            }
        };
        if !self.questions.is_empty() {
            // With --ask, each message is a text to answer them about.
            report(self.decide(&input, &self.questions));
        } else if let Some(state) = &self.state {
            // Each line is a question; pasted lines are asked together.
            let questions: Vec<Question> = input
                .text
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(Question::yes_no)
                .collect();
            if !questions.is_empty() {
                report(self.decide(state, &questions));
            }
        } else {
            self.state = Some(input);
            eprintln!("{}", self.hint());
        }
        Ok(())
    }

    fn clear(&mut self) {
        self.state = None;
        eprintln!("Cleared. {}", self.hint());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(argv: &[&str]) -> Asked {
        #[derive(clap::Parser)]
        struct Cli {
            #[command(flatten)]
            asked: Asked,
        }
        <Cli as clap::Parser>::try_parse_from(["run"].iter().chain(argv))
            .unwrap()
            .asked
    }

    fn asked(argv: &[&str]) -> Vec<Question> {
        parse(argv).questions().unwrap()
    }

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| v.to_string()).collect()
    }

    /// Ollama's `TestRunDecisionModel` questions: each `--choice` and
    /// `--level` answers the `--ask` before it, in any mix.
    #[test]
    fn choices_and_levels_answer_the_ask_before_them() {
        assert_eq!(
            asked(&[
                "--ask",
                "Is a refund being requested?",
                "--ask",
                "Which team?",
                "--choice",
                "billing=Payments and refunds",
                "--choice",
                "technical",
                "--ask",
                "How urgent?",
                "--level",
                "routine",
                "--level",
                "soon",
            ]),
            [
                Question::yes_no("Is a refund being requested?"),
                Question {
                    text: "Which team?".into(),
                    kind: Kind::Choice(vec![
                        ("billing".into(), Some("Payments and refunds".into())),
                        ("technical".into(), None),
                    ]),
                },
                Question {
                    text: "How urgent?".into(),
                    kind: Kind::Level(strings(&["routine", "soon"])),
                },
            ]
        );
        assert_eq!(asked(&[]), []);
    }

    /// Ollama's `TestRunDecisionModelErrors` flag errors.
    #[test]
    fn a_choice_or_level_needs_its_own_ask_and_only_one_kind() {
        let message = |argv: &[&str]| parse(argv).questions().unwrap_err().to_string();
        assert_eq!(
            message(&["--choice", "billing", "--ask", "Which team?"]),
            "--choice must come after the --ask it answers"
        );
        assert_eq!(
            message(&["--level", "low"]),
            "--level must come after the --ask it answers"
        );
        let both = "a question takes --choice or --level, not both";
        assert_eq!(
            message(&["--ask", "How urgent?", "--choice", "a", "--level", "low"]),
            both
        );
        assert_eq!(
            message(&["--ask", "How urgent?", "--level", "low", "--choice", "a"]),
            both
        );
    }

    /// `llmman run clef-flash ARGV...`, as [`reparse`] reads it.
    fn reparsed(argv: &[&str]) -> anyhow::Result<RunArgs> {
        parse_argv(argv.iter().map(OsString::from).collect())
    }

    fn reparsed_ok(argv: &[&str]) -> (String, Vec<Question>, RunArgs) {
        let args = reparsed(&[&["clef-flash"], argv].concat()).unwrap();
        (args.prompt.join(" "), args.ask.questions().unwrap(), args)
    }

    /// Cobra takes flags on either side of the text, so ollama's
    /// `run nimble "text" --ask "..."` works; so does this.
    #[test]
    fn flags_may_follow_the_text() {
        let (prompt, questions, args) = reparsed_ok(&[
            "I was charged twice",
            "--ask",
            "Which team?",
            "--choice",
            "billing=Payments",
            "--choice",
            "technical",
            "--format",
            "json",
            "--verbose",
            "--keepalive",
            "1h",
        ]);
        assert_eq!(prompt, "I was charged twice");
        assert!(args.format.is_some() && args.verbose);
        assert_eq!(args.keepalive.as_deref(), Some("1h"));
        assert_eq!(
            questions,
            [Question {
                text: "Which team?".into(),
                kind: Kind::Choice(vec![
                    ("billing".into(), Some("Payments".into())),
                    ("technical".into(), None),
                ]),
            }]
        );

        // Both sides at once: a flag after the text carries on from one before.
        let (prompt, questions, _) = reparsed_ok(&[
            "--ask",
            "How urgent?",
            "Please",
            "help",
            "--level",
            "Routine",
            "--level",
            "Soon",
            "--ask",
            "Is it spam?",
        ]);
        assert_eq!(prompt, "Please help");
        assert_eq!(
            questions,
            [
                Question {
                    text: "How urgent?".into(),
                    kind: Kind::Level(strings(&["Routine", "Soon"])),
                },
                Question::yes_no("Is it spam?"),
            ]
        );
    }

    /// Other flags of `run` are not text after it either; one it does not
    /// have is an error, as in cobra. Order still counts.
    #[test]
    fn other_flags_after_the_text_are_not_text() {
        let (prompt, _, args) = reparsed_ok(&["Win a cruise", "--think", "false", "--nowordwrap"]);
        assert_eq!(prompt, "Win a cruise");
        assert!(args.nowordwrap);
        let err = reparsed(&["clef-flash", "Win", "--nonsense"]).unwrap_err();
        assert!(err.to_string().contains("--nonsense"), "{err}");
        let args = reparsed(&["clef-flash", "Win", "--choice", "a"]).unwrap();
        let err = args.ask.questions().unwrap_err();
        assert!(
            err.to_string().contains("must come after the --ask"),
            "{err}"
        );
    }

    /// Flags before a `--` count, whatever the text before it; everything
    /// after it is text.
    #[test]
    fn flags_before_a_separator_count_and_nothing_after_it_does() {
        let (prompt, questions, _) = reparsed_ok(&[
            "Win a cruise",
            "--ask",
            "Is it spam?",
            "--",
            "--ask",
            "is",
            "text",
        ]);
        assert_eq!(prompt, "Win a cruise --ask is text");
        assert_eq!(questions, [Question::yes_no("Is it spam?")]);
        let (prompt, ..) = reparsed_ok(&["--ask", "q", "--", "--literal"]);
        assert_eq!(prompt, "--literal");
    }

    /// Negative `--keepalive` values keep the model loaded.
    #[test]
    fn keepalive_takes_negative_values() {
        for value in ["-1", "-5m"] {
            let (_, _, args) = reparsed_ok(&["--keepalive", value, "text"]);
            assert_eq!(args.keepalive.as_deref(), Some(value));
            assert!(super::super::check_keepalive(Some(value)).is_ok());
        }
        assert!(super::super::check_keepalive(Some("banana")).is_err());
    }

    fn body(question: &Question) -> Value {
        serde_json::to_value(QuestionBody::new(question)).unwrap()
    }

    #[test]
    fn the_questions_are_sent_as_typed_in_the_order_given() {
        let questions = [
            Question {
                text: "zebra?".into(),
                kind: Kind::Choice(vec![("b".into(), Some("Bee".into())), ("a".into(), None)]),
            },
            Question::yes_no("apple?"),
            Question {
                text: "How urgent?".into(),
                kind: Kind::Level(strings(&["Routine", "Soon"])),
            },
        ];
        let request = Request {
            model: "m",
            state: state_value("hi"),
            images: &[],
            keep_alive: None,
            questions: Ordered(
                questions
                    .iter()
                    .map(|q| (q.text.clone(), QuestionBody::new(q)))
                    .collect(),
            ),
        };
        // Not sorted by serde_json's map: Clef reads them in this order.
        assert_eq!(
            serde_json::to_string(&request).unwrap(),
            r#"{"model":"m","state":"hi","questions":{"zebra?":{"type":"choice","instructions":"zebra?","criteria":{"b":"Bee","a":null}},"apple?":{"type":"noul","instructions":"apple?"},"How urgent?":{"type":"score","instructions":"How urgent?","criteria":["Routine","Soon"]}}}"#
        );
        assert_eq!(body(&questions[1])["type"], "noul");
    }

    #[test]
    fn images_ride_along_as_data_urls() {
        let images = vec!["data:image/png;base64,AAAA".to_string()];
        let request = Request {
            model: "m",
            state: Value::Null,
            images: &images,
            keep_alive: Some("5m"),
            questions: Ordered(Vec::new()),
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["images"], serde_json::json!(images));
        assert_eq!(json["keep_alive"], "5m");
        let file = MediaFile {
            content_type: "image/png",
            data: vec![0, 0, 0],
        };
        assert_eq!(data_url(&file), "data:image/png;base64,AAAA");
    }

    /// Ollama's `decide`: JSON in the text is the state as JSON, anything
    /// else a string.
    #[test]
    fn json_text_is_the_state_as_json() {
        assert_eq!(
            state_value(r#"  {"ticket": "I was charged twice"}  "#),
            serde_json::json!({"ticket": "I was charged twice"})
        );
        assert_eq!(state_value("[1, 2]"), serde_json::json!([1, 2]));
        assert_eq!(state_value("{not json"), Value::String("{not json".into()));
        assert_eq!(state_value("hello"), Value::String("hello".into()));
    }

    fn response(answers: &str) -> Response {
        serde_json::from_str(&format!(r#"{{"answers":{answers}}}"#)).unwrap()
    }

    /// The lines ollama's own tests expect: the likelier answer and its
    /// probability, the label, or the likeliest level with the score.
    #[test]
    fn answers_print_as_the_likelier_answer_with_its_probability() {
        let q = |text: &str| Question::yes_no(text);
        let response = response(
            r#"{
              "yes?": {"type": "noul", "noul": 0.97},
              "no?": {"type": "noul", "noul": 0.2},
              "team?": {"type": "choice", "choice": "billing",
                        "probabilities": {"billing": 0.91, "technical": 0.09}},
              "urgent?": {"type": "score", "score": 1.2,
                          "legend": {"0": "Routine", "1": "Soon", "2": "Immediate"},
                          "probabilities": {"0": 0.2, "1": 0.64, "2": 0.16}}
            }"#,
        );
        let line = |text: &str| render(&[&q(text)], &response).unwrap();
        assert_eq!(line("yes?"), "yes  97%\n");
        assert_eq!(line("no?"), "no  80%\n");
        assert_eq!(line("team?"), "billing  91%\n");
        assert_eq!(line("urgent?"), "Soon  64%  1.2 / 2\n");

        // Several: the question first, and the columns line up.
        let all = [&q("yes?"), &q("team?"), &q("urgent?")];
        assert_eq!(
            render(&all, &response).unwrap(),
            "yes?     yes      97%\n\
             team?    billing  91%\n\
             urgent?  Soon     64%  1.2 / 2\n"
        );
    }

    /// A scale answered between its ends is the likeliest level, not the one
    /// nearest the score.
    #[test]
    fn a_split_scale_reports_the_likeliest_level() {
        let answer: Answer = serde_json::from_value(serde_json::json!({
            "type": "score", "score": 1.0,
            "legend": {"0": "Low", "1": "Mid", "2": "High"},
            "probabilities": {"0": 0.45, "1": 0.1, "2": 0.45},
        }))
        .unwrap();
        assert_eq!(answer.cells().unwrap(), ["Low", "45%", "1.0 / 2"]);
    }

    #[test]
    fn a_missing_or_unknown_answer_is_an_error() {
        let response = response(r#"{"a?": {"type": "mystery"}}"#);
        let err = render(&[&Question::yes_no("b?")], &response).unwrap_err();
        assert!(err.to_string().contains("no answer to \"b?\""), "{err}");
        let err = render(&[&Question::yes_no("a?")], &response).unwrap_err();
        assert!(err.to_string().contains("unknown type"), "{err}");
    }

    #[test]
    fn answers_keep_the_order_the_backend_sent() {
        let response = response(r#"{"z": 1, "a": 2, "m": 3}"#);
        let ids: Vec<_> = response
            .answers
            .0
            .iter()
            .map(|(id, _)| id.as_str())
            .collect();
        assert_eq!(ids, ["z", "a", "m"]);
        assert_eq!(
            serde_json::to_string(&response).unwrap(),
            r#"{"model":"","answers":{"z":1,"a":2,"m":3},"usage":null}"#
        );
    }
}
