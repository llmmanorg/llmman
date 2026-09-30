//! The System One decision API on ggml, like `crate::mediagen`. Each
//! question is one user turn through the model's chat template, thinking
//! off, and the model is read at the answer position: its probability of
//! each label (`A`..`Z`, a level digit, `yes`/`no`). Nothing is generated.

pub mod answer;
pub mod engine;
pub mod request;
pub mod server;
pub mod template;

use std::collections::HashSet;

use serde_json::{Map, Value};

use request::{render_question, render_text, Json, Kind, Question, Request};

/// A request refused (400), or one the server could not answer (500).
#[derive(Debug)]
pub enum Error {
    Refused(String),
    Failed(String),
}

impl Error {
    fn context(self, prefix: String) -> Error {
        match self {
            Error::Refused(m) => Error::Refused(prefix + &m),
            Error::Failed(m) => Error::Failed(prefix + &m),
        }
    }
}

/// The model, as the answers need it.
pub trait Reader {
    /// One user turn through the chat template, with the assistant turn opener.
    fn render(&self, content: &str, kwargs: &Map<String, Value>) -> Result<String, Error>;
    /// The token ids of `text`, special tokens parsed.
    fn encode(&self, text: &str) -> Result<Vec<i32>, Error>;
    /// The token the vocabulary starts prompts with, if it adds one.
    fn bos(&self) -> Option<i32>;
    /// The text of `token` if the tokenizer splits text at it.
    fn special(&self, token: i32) -> Option<String>;
    /// The most tokens a prompt may have.
    fn context(&self) -> usize;
    /// The log probability, over the whole vocabulary, of each of `tokens`
    /// as the token after `prompt`.
    fn read(&mut self, prompt: &[i32], tokens: &[i32]) -> Result<Vec<f64>, Error>;
}

pub struct Decision {
    /// `(question id, answer)` in request order.
    pub answers: Vec<(String, Json)>,
    /// Every question's prompt: the state counts once per question.
    pub input_tokens: usize,
}

/// Reasoning blocks a generation prompt may leave open, where label
/// probabilities would mean nothing.
const REASONING_BLOCKS: [(&str, &str); 2] =
    [("<think>", "</think>"), ("<|channel>thought", "<channel|>")];

/// Up to `N_LETTERS` options are labelled `A`..`Z`; past that, with two letters.
const LETTERS: std::ops::RangeInclusive<char> = 'A'..='Z';
const N_LETTERS: usize = 26;

pub fn decide(reader: &mut dyn Reader, req: &Request) -> Result<Decision, Error> {
    let kwargs = template_kwargs(req)?;
    let state = render_text(Some(&req.state));
    let mut pairs = None;
    let (mut answers, mut input_tokens) = (Vec::new(), 0);
    for (id, q) in &req.questions {
        let fail = |e: Error| e.context(format!("question {id:?}: "));
        let (prompt, options) = ask(reader, &state, q, &kwargs, &mut pairs).map_err(fail)?;
        let flat: Vec<i32> = options.concat();
        let logprobs = reader.read(&prompt, &flat).map_err(fail)?;
        if logprobs.len() != flat.len() {
            return Err(fail(Error::Failed(
                "the read returned the wrong number of scores".into(),
            )));
        }
        let mut rest = &logprobs[..];
        let per_option: Vec<f64> = options
            .iter()
            .map(|tokens| {
                let (own, tail) = rest.split_at(tokens.len());
                rest = tail;
                answer::log_sum_exp(own)
            })
            .collect();
        let json =
            answer::answer(q, &answer::score(&per_option)).map_err(|m| fail(Error::Failed(m)))?;
        answers.push((id.clone(), json));
        input_tokens += prompt.len();
    }
    Ok(Decision {
        answers,
        input_tokens,
    })
}

/// Template variables: thinking off, then the request's. Turning it on is
/// refused, as the read would be inside the reasoning.
pub fn template_kwargs(req: &Request) -> Result<Map<String, Value>, Error> {
    let mut kwargs = Map::new();
    kwargs.insert("enable_thinking".into(), false.into());
    if let Some(v) = req.chat_template_kwargs.get("enable_thinking") {
        if *v != Value::Bool(false) {
            return Err(Error::Refused(format!(
                "chat_template_kwargs sets \"enable_thinking\" to {v}, but decisions need it \
                 false or unset"
            )));
        }
    }
    kwargs.extend(req.chat_template_kwargs.clone());
    Ok(kwargs)
}

/// The labels of `q`'s options: letters, two-letter ones past 26, level digits, or `yes`/`no`.
fn labels(
    reader: &dyn Reader,
    q: &Question,
    kwargs: &Map<String, Value>,
    pairs: &mut Option<Vec<String>>,
) -> Result<Vec<String>, Error> {
    let n = q.names.len();
    if q.kind != Kind::Choice {
        return Ok(q.names.clone());
    }
    if n <= N_LETTERS {
        return Ok(LETTERS.take(n).map(String::from).collect());
    }
    if pairs.is_none() {
        *pairs = Some(pair_labels(reader, kwargs)?);
    }
    let pairs = pairs.as_ref().map(Vec::as_slice).unwrap_or_default();
    if n > pairs.len() {
        return Err(Error::Refused(format!(
            "it has {n} options, but the served tokenizer and chat template can label at \
             most {}",
            pairs.len().max(N_LETTERS)
        )));
    }
    Ok(pairs[..n].to_vec())
}

/// Two-letter labels that are distinct single tokens at the answer position.
fn pair_labels(reader: &dyn Reader, kwargs: &Map<String, Value>) -> Result<Vec<String>, Error> {
    let prompt = reader.render("x", kwargs)?;
    let at = Labels::new(reader, &prompt, &tokenize(reader, &prompt)?)?;
    let mut seen = HashSet::new();
    let mut labels = Vec::new();
    for a in LETTERS {
        for b in LETTERS {
            let label = format!("{a}{b}");
            if at.token(reader, &label)?.is_some_and(|t| seen.insert(t)) {
                labels.push(label);
            }
        }
    }
    Ok(labels)
}

/// The token ids of a prompt, with the `<bos>` the vocabulary adds unless the template wrote it.
fn tokenize(reader: &dyn Reader, prompt: &str) -> Result<Vec<i32>, Error> {
    let mut ids = reader.encode(prompt)?;
    if let Some(bos) = reader.bos().filter(|b| ids.first() != Some(b)) {
        ids.insert(0, bos);
    }
    Ok(ids)
}

/// What a label is checked against to be one token: the text after the
/// prompt's last special token, which tokenizes on its own (text is split
/// at special tokens), so the state is not tokenized again per label. When
/// the tail does not tokenize alone to the same tokens, the whole prompt.
struct Labels {
    text: String,
    ids: Vec<i32>,
    whole: bool,
}

impl Labels {
    fn new(reader: &dyn Reader, prompt: &str, ids: &[i32]) -> Result<Labels, Error> {
        let last = ids
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, &t)| Some((i, reader.special(t)?)));
        if let Some((at, token)) = last {
            if let Some(start) = prompt.rfind(&token) {
                let text = &prompt[start + token.len()..];
                let tail = reader.encode(text)?;
                if tail == ids[at + 1..] {
                    return Ok(Labels {
                        text: text.into(),
                        ids: tail,
                        whole: false,
                    });
                }
            }
        }
        Ok(Labels {
            text: prompt.into(),
            ids: ids.to_vec(),
            whole: true,
        })
    }

    /// The token `label` adds after the text, when it is exactly one.
    fn token(&self, reader: &dyn Reader, label: &str) -> Result<Option<i32>, Error> {
        let text = format!("{}{label}", self.text);
        let with = if self.whole {
            tokenize(reader, &text)?
        } else {
            reader.encode(&text)?
        };
        let n = self.ids.len();
        Ok((with.len() == n + 1 && with[..n] == *self.ids).then(|| with[n]))
    }
}

/// Spellings of an answer: `Yes` is how Gemma 4 says it.
fn spellings(kind: Kind, label: &str) -> Vec<String> {
    let mut all = vec![label.to_string()];
    if kind == Kind::Noul {
        let mut c = label.chars();
        if let Some(first) = c.next() {
            all.push(first.to_uppercase().chain(c).collect());
        }
    }
    all
}

/// The prompt of `q` and, per option, the tokens that answer it.
fn ask(
    reader: &dyn Reader,
    state: &str,
    q: &Question,
    kwargs: &Map<String, Value>,
    pairs: &mut Option<Vec<String>>,
) -> Result<(Vec<i32>, Vec<Vec<i32>>), Error> {
    let labels = labels(reader, q, kwargs, pairs)?;
    let content = render_question(state, q, &labels);
    let prompt = reader.render(&content, kwargs)?;
    check_reasoning(&prompt, &content)?;
    let ids = tokenize(reader, &prompt)?;
    if ids.len() >= reader.context() {
        return Err(Error::Refused(format!(
            "the prompt has {} tokens, which does not fit the context length of {} tokens",
            ids.len(),
            reader.context()
        )));
    }
    let after = Labels::new(reader, &prompt, &ids)?;
    let mut seen = HashSet::new();
    let mut options = Vec::new();
    for label in &labels {
        let mut tokens = Vec::new();
        for spelling in spellings(q.kind, label) {
            if let Some(t) = after.token(reader, &spelling)? {
                if !tokens.contains(&t) {
                    tokens.push(t);
                }
            }
        }
        if tokens.is_empty() || tokens.iter().any(|t| !seen.insert(*t)) {
            return Err(Error::Refused(format!(
                "the answer label {label:?} is not one distinct token after the chat prompt \
                 for this tokenizer, so this model is not supported"
            )));
        }
        options.push(tokens);
    }
    Ok((ids, options))
}

/// Refuses a prompt that leaves a reasoning block open at the answer position.
fn check_reasoning(prompt: &str, content: &str) -> Result<(), Error> {
    // after the message only: its last line is fixed text
    let closing = content.rsplit('\n').next().unwrap_or_default();
    let tail = prompt
        .rfind(closing)
        .map_or(prompt, |at| &prompt[at + closing.len()..]);
    for (open, close) in REASONING_BLOCKS {
        if tail.rfind(open) > tail.rfind(close) {
            return Err(Error::Refused(
                "the chat template leaves a reasoning block open at the answer position, so \
                 this model is not supported with these chat_template_kwargs"
                    .into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::collections::HashMap;

    /// A model with a character tokenizer, `yes`/`no` words and `AA`..`ZZ` as
    /// single tokens at the answer position, and canned log probabilities.
    struct Fake {
        lp: HashMap<i32, f64>,
        opener: &'static str,
        reads: Vec<(usize, Vec<i32>)>,
        /// `<a>` is one special token, which text is split at.
        special: bool,
        /// A fragment tokenized alone gets a token a prompt's does not, as with
        /// a tokenizer that marks the start of a text.
        marks_the_start: bool,
        /// Characters tokenized so far.
        encoded: Cell<usize>,
        limit: usize,
    }

    impl Fake {
        fn new(lp: &[(&str, f64)]) -> Fake {
            let mut f = Fake {
                lp: HashMap::new(),
                opener: "<a>",
                reads: Vec::new(),
                special: false,
                marks_the_start: false,
                encoded: Cell::new(0),
                limit: 400,
            };
            for (t, p) in lp {
                let id = f.encode(&format!("<a>{t}")).unwrap().pop().unwrap();
                f.lp.insert(id, p.ln());
            }
            f
        }

        /// The fragment after the last `<a>`, or all of a text that has none in special mode.
        fn answer_token(&self, tail: &str) -> Option<i32> {
            let word = ["yes", "Yes", "no", "No"].iter().position(|x| *x == tail);
            match word {
                Some(w) => Some(1_000_000 + w as i32),
                // `AA`..`ZZ` are one token each
                None if tail.len() == 2 && tail.chars().all(|c| c.is_ascii_uppercase()) => {
                    Some(2_000_000 + tail.bytes().fold(0, |a, b| a * 256 + b as i32))
                }
                None => None,
            }
        }
    }

    impl Reader for Fake {
        fn render(&self, content: &str, _: &Map<String, Value>) -> Result<String, Error> {
            Ok(format!("<u>{content}</u>{}", self.opener))
        }
        fn encode(&self, text: &str) -> Result<Vec<i32>, Error> {
            self.encoded.set(self.encoded.get() + text.len());
            let (head, tail) = match text.rfind("<a>") {
                Some(at) => (&text[..at], Some(&text[at + 3..])),
                None => (text, None),
            };
            let chars = |t: &str| t.chars().map(|c| c as i32).collect::<Vec<_>>();
            let mut ids = if self.special && tail.is_none() {
                // nothing before it: what follows a special token, tokenized alone
                return Ok(match self.answer_token(head) {
                    Some(t) => vec![t],
                    None => [vec![7; self.marks_the_start as usize], chars(head)].concat(),
                });
            } else {
                chars(head)
            };
            let Some(tail) = tail else { return Ok(ids) };
            match self.special {
                true => ids.push(SPECIAL),
                false => ids.extend(chars("<a>")),
            }
            match self.answer_token(tail) {
                Some(t) => ids.push(t),
                None => ids.extend(chars(tail)),
            }
            Ok(ids)
        }
        fn bos(&self) -> Option<i32> {
            None
        }
        fn special(&self, token: i32) -> Option<String> {
            (self.special && token == SPECIAL).then(|| "<a>".to_string())
        }
        fn context(&self) -> usize {
            self.limit
        }
        fn read(&mut self, prompt: &[i32], tokens: &[i32]) -> Result<Vec<f64>, Error> {
            self.reads.push((prompt.len(), tokens.to_vec()));
            Ok(tokens
                .iter()
                .map(|t| self.lp.get(t).copied().unwrap_or(f64::NEG_INFINITY))
                .collect())
        }
    }

    const SPECIAL: i32 = 9999;

    fn request(questions: &str) -> Request {
        request::parse(
            format!(r#"{{"state": "It broke.", "model": "m", "questions": {questions}}}"#)
                .as_bytes(),
        )
        .unwrap_or_else(|e| panic!("{e:?}"))
    }

    fn num(a: &Json, k: &str) -> f64 {
        match a.get(k) {
            Some(Json::Num(n)) => n.as_f64().unwrap(),
            other => panic!("{k}: {other:?}"),
        }
    }

    #[test]
    fn questions_are_read_in_order_and_answered_by_their_labels() {
        let mut m = Fake::new(&[
            ("A", 0.1),
            ("B", 0.6),
            ("C", 0.1),
            ("0", 0.2),
            ("1", 0.6),
            ("yes", 0.3),
            ("no", 0.1),
        ]);
        let req = request(
            r#"{"team": {"type": "choice", "criteria": {"billing": null, "technical": null, "sales": null}},
                "mood": {"type": "score", "criteria": ["Calm", "Angry"]},
                "urgent": {"type": "noul", "instructions": "Today?"}}"#,
        );
        let d = decide(&mut m, &req).unwrap();
        let ids: Vec<_> = d.answers.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, ["team", "mood", "urgent"]);
        let team = &d.answers[0].1;
        assert_eq!(team.get("choice"), Some(&Json::from("technical")));
        assert!((num(team, "x_label_mass") - 0.8).abs() < 1e-12);
        // 0.3 of 0.4
        assert!((num(&d.answers[2].1, "noul") - 0.75).abs() < 1e-12);
        assert_eq!(m.reads.len(), 3);
        assert_eq!(d.input_tokens, m.reads.iter().map(|r| r.0).sum::<usize>());
    }

    #[test]
    fn a_capitalised_yes_counts_as_yes() {
        let mut m = Fake::new(&[("yes", 0.05), ("Yes", 0.45), ("no", 0.10), ("No", 0.30)]);
        let req = request(r#"{"q": {"type": "noul", "instructions": "Today?"}}"#);
        let a = &decide(&mut m, &req).unwrap().answers[0].1;
        assert!((num(a, "noul") - 0.5 / 0.9).abs() < 1e-12);
        assert!((num(a, "x_label_mass") - 0.9).abs() < 1e-12);
    }

    #[test]
    fn a_label_that_is_not_one_token_refuses_the_model() {
        struct Merges(Fake);
        impl Reader for Merges {
            fn render(&self, c: &str, k: &Map<String, Value>) -> Result<String, Error> {
                self.0.render(c, k)
            }
            // `B` merges into the newline before it: two labels cannot be told apart
            fn encode(&self, text: &str) -> Result<Vec<i32>, Error> {
                let mut ids = self.0.encode(text)?;
                if text.ends_with("<a>B") {
                    ids.pop();
                    ids.pop();
                    ids.push(7);
                }
                Ok(ids)
            }
            fn bos(&self) -> Option<i32> {
                self.0.bos()
            }
            fn special(&self, t: i32) -> Option<String> {
                self.0.special(t)
            }
            fn context(&self) -> usize {
                self.0.context()
            }
            fn read(&mut self, p: &[i32], t: &[i32]) -> Result<Vec<f64>, Error> {
                self.0.read(p, t)
            }
        }
        let mut m = Merges(Fake::new(&[("A", 0.5)]));
        let req = request(r#"{"q": {"type": "choice", "criteria": {"a": null, "b": null}}}"#);
        let Err(Error::Refused(msg)) = decide(&mut m, &req) else {
            panic!()
        };
        assert!(
            msg.starts_with("question \"q\": the answer label \"B\" is not one distinct token"),
            "{msg}"
        );
    }

    #[test]
    fn a_prompt_past_the_context_is_refused_before_reading() {
        let mut m = Fake::new(&[]);
        let big = "x".repeat(500);
        let req = request(&format!(
            r#"{{"q": {{"type": "noul", "instructions": "{big}"}}}}"#
        ));
        let Err(Error::Refused(msg)) = decide(&mut m, &req) else {
            panic!()
        };
        assert!(
            msg.contains("does not fit the context length of 400 tokens"),
            "{msg}"
        );
        assert!(m.reads.is_empty());
    }

    #[test]
    fn a_template_that_opens_a_reasoning_block_is_refused() {
        let mut m = Fake::new(&[]);
        m.opener = "<a><think>\n";
        let req = request(r#"{"q": {"type": "noul", "instructions": "Today?"}}"#);
        let Err(Error::Refused(msg)) = decide(&mut m, &req) else {
            panic!()
        };
        assert!(msg.contains("reasoning block open"), "{msg}");
        // closed again, it is fine
        assert!(check_reasoning(
            "...only.<a><think>\n\n</think>\n\n",
            "x\nAnswer with yes or no only."
        )
        .is_ok());
        // a reasoning block in the message itself is not the template's
        assert!(check_reasoning("<think>a</think>...\nAnswer.<a>", "<think>a\nAnswer.").is_ok());
    }

    #[test]
    fn thinking_can_be_left_off_or_set_false_but_not_on() {
        let with = |k: &str| {
            request::parse(
                format!(
                    r#"{{"state": "s", "model": "m", "chat_template_kwargs": {k},
                    "questions": {{"q": {{"type": "noul", "instructions": "i"}}}}}}"#
                )
                .as_bytes(),
            )
            .unwrap()
        };
        assert_eq!(
            template_kwargs(&with("{}")).unwrap()["enable_thinking"],
            false
        );
        assert!(template_kwargs(&with(r#"{"enable_thinking": false}"#)).is_ok());
        assert!(template_kwargs(&with(r#"{"enable_thinking": true}"#)).is_err());
        assert_eq!(template_kwargs(&with(r#"{"x": 1}"#)).unwrap()["x"], 1);
    }

    #[test]
    fn options_past_26_get_two_letter_labels() {
        // every two-letter label is one token in this fake
        let opts: Vec<String> = (0..30).map(|i| format!("\"o{i}\": null")).collect();
        let req = request(&format!(
            r#"{{"q": {{"type": "choice", "criteria": {{{}}}}}}}"#,
            opts.join(",")
        ));
        let mut m = Fake::new(&[("AA", 0.9)]);
        let (_, options) = ask(&m, "s", &req.questions[0].1, &Map::new(), &mut None).unwrap();
        assert_eq!(options.len(), 30);
        // `AA` labels the first option
        let a = &decide(&mut m, &req).unwrap().answers[0].1;
        assert_eq!(a.get("choice"), Some(&Json::from("o0")));
    }

    /// Reads a question of `n` options over a long state: the tokens that
    /// answer it, and the characters tokenized to find them.
    fn many_options(m: &mut Fake, n: usize) -> (Vec<i32>, usize) {
        let options: Vec<String> = (0..n).map(|i| format!("\"o{i}\": null")).collect();
        let body = format!(
            "{{\"state\": \"{}\", \"model\": \"m\", \"questions\": \
             {{\"q\": {{\"type\": \"choice\", \"criteria\": {{{}}}}}}}}}",
            "long state. ".repeat(40),
            options.join(",")
        );
        let req = request::parse(body.as_bytes()).unwrap_or_else(|e| panic!("{e:?}"));
        m.limit = 100_000;
        m.encoded.set(0);
        decide(m, &req).unwrap();
        (m.reads[0].1.clone(), m.encoded.get())
    }

    #[test]
    fn labels_are_checked_after_the_last_special_token_not_over_the_whole_prompt() {
        let mut whole = Fake::new(&[("B", 0.9)]);
        let mut tail = Fake::new(&[("B", 0.9)]);
        tail.special = true;
        let (tokens_whole, cost_whole) = many_options(&mut whole, 20);
        let (tokens_tail, cost_tail) = many_options(&mut tail, 20);
        assert_eq!(tokens_whole.len(), 20);
        // the same labels, found for a fraction of the tokenizing
        assert_eq!(tokens_whole, tokens_tail);
        assert!(cost_tail * 8 < cost_whole, "{cost_tail} vs {cost_whole}");
    }

    #[test]
    fn a_tail_that_does_not_tokenize_alone_falls_back_to_the_whole_prompt() {
        let mut m = Fake::new(&[("B", 0.9)]);
        m.special = true;
        m.marks_the_start = true;
        let (tokens, cost) = many_options(&mut m, 20);
        assert_eq!(tokens.len(), 20);
        // every label was checked against the whole prompt, as without special tokens
        assert!(cost > 20 * "long state. ".len() * 40, "{cost}");
    }
}
