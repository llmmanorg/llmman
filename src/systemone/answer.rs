//! From the model's probabilities of the answer labels to System One's
//! answers: sglang's formulas, `probabilities` normalised over the labels
//! and `x_label_mass` the probability the model put on the labels at all.

use super::request::{Json, Kind, Question};

/// What one question's read comes to.
#[derive(Debug, PartialEq)]
pub struct Scored {
    /// Per option, normalised over the options.
    pub probabilities: Vec<f64>,
    /// Probability of all the options at the answer position, out of the whole vocabulary.
    pub mass: f64,
}

/// `logprobs` are each option's log probability out of the whole vocabulary.
pub fn score(logprobs: &[f64]) -> Scored {
    let max = logprobs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logprobs.iter().map(|l| (l - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    Scored {
        probabilities: weights.iter().map(|w| w / total).collect(),
        mass: logprobs.iter().map(|l| l.exp()).sum(),
    }
}

/// Log of the sum of `exp` of `logprobs`: the probability of any one of several tokens.
pub fn log_sum_exp(logprobs: &[f64]) -> f64 {
    let max = logprobs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return max;
    }
    max + logprobs.iter().map(|l| (l - max).exp()).sum::<f64>().ln()
}

/// The answer to `q`, or why its scores are unusable (a server fault).
pub fn answer(q: &Question, s: &Scored) -> Result<Json, String> {
    if !s.mass.is_finite() || !s.probabilities.iter().all(|p| p.is_finite()) {
        return Err("a question scored non-finite values".into());
    }
    let p = &s.probabilities;
    let by_name = || {
        Json::Obj(
            q.names
                .iter()
                .zip(p)
                .map(|(n, p)| (n.clone(), Json::f64(*p)))
                .collect(),
        )
    };
    let entries = match q.kind {
        Kind::Noul => vec![("type", "noul".into()), ("noul", Json::f64(p[0]))],
        Kind::Choice => {
            let top = p
                .iter()
                .enumerate()
                .fold(0, |t, (i, v)| if *v > p[t] { i } else { t });
            vec![
                ("type", "choice".into()),
                ("choice", Json::Str(q.names[top].clone())),
                ("confidence", Json::f64(choice_confidence(&normalized(p)))),
                ("probabilities", by_name()),
            ]
        }
        Kind::Score => {
            let legend = q
                .names
                .iter()
                .zip(&q.details)
                .map(|(n, d)| (n.clone(), d.clone().unwrap_or(Json::Null)))
                .collect();
            vec![
                ("type", "score".into()),
                (
                    "score",
                    Json::f64(p.iter().enumerate().map(|(i, v)| i as f64 * v).sum()),
                ),
                ("confidence", Json::f64(score_confidence(&normalized(p)))),
                ("legend", Json::Obj(legend)),
                ("probabilities", by_name()),
            ]
        }
    };
    let mut entries: Vec<(String, Json)> = entries
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    entries.push(("x_label_mass".into(), Json::f64(s.mass)));
    Ok(Json::Obj(entries))
}

fn normalized(p: &[f64]) -> Vec<f64> {
    let total: f64 = p.iter().sum();
    if total <= 0.0 {
        return vec![1.0 / p.len() as f64; p.len()];
    }
    p.iter().map(|v| v / total).collect()
}

/// How far the top option stands above a uniform guess, from 0 to 1.
fn choice_confidence(q: &[f64]) -> f64 {
    let n = q.len() as f64;
    if q.len() == 1 {
        return 1.0;
    }
    let top = q.iter().copied().fold(0.0, f64::max);
    ((n * top - 1.0) / (n - 1.0)).clamp(0.0, 1.0)
}

/// One minus the spread around the top level relative to a uniform spread, floored at 0.
fn score_confidence(q: &[f64]) -> f64 {
    if q.len() == 1 {
        return 1.0;
    }
    let top = q
        .iter()
        .enumerate()
        .fold(0, |t, (i, v)| if *v > q[t] { i } else { t });
    let spread: f64 = q
        .iter()
        .enumerate()
        .map(|(i, p)| p * (i as f64 - top as f64).abs())
        .sum();
    let mid = (q.len() - 1) as f64 / 2.0;
    let uniform = (0..q.len()).map(|i| (i as f64 - mid).abs()).sum::<f64>() / q.len() as f64;
    (1.0 - spread / uniform).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn question(kind: Kind, names: &[&str], details: Vec<Option<Json>>) -> Question {
        Question {
            kind,
            text: None,
            names: names.iter().map(|s| s.to_string()).collect(),
            details,
        }
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    #[test]
    fn probabilities_are_normalised_and_the_mass_is_not() {
        let s = score(&[0.5f64.ln(), 0.25f64.ln()]);
        close(s.probabilities[0], 2.0 / 3.0);
        close(s.probabilities[1], 1.0 / 3.0);
        close(s.mass, 0.75);
    }

    #[test]
    fn several_tokens_of_one_option_add_their_probabilities() {
        close(log_sum_exp(&[0.25f64.ln(), 0.5f64.ln()]), 0.75f64.ln());
        assert_eq!(log_sum_exp(&[f64::NEG_INFINITY]), f64::NEG_INFINITY);
    }

    #[test]
    fn an_option_with_no_probability_anywhere_is_a_server_fault() {
        let q = question(Kind::Noul, &["yes", "no"], vec![None, None]);
        assert!(answer(&q, &score(&[f64::NEG_INFINITY, f64::NEG_INFINITY])).is_err());
    }

    #[test]
    fn a_noul_answer_is_the_probability_of_yes() {
        let q = question(Kind::Noul, &["yes", "no"], vec![None, None]);
        let a = answer(&q, &score(&[0.6f64.ln(), 0.2f64.ln()])).unwrap();
        let num = |k: &str| match a.get(k) {
            Some(Json::Num(n)) => n.as_f64().unwrap(),
            other => panic!("{k}: {other:?}"),
        };
        close(num("noul"), 0.75);
        close(num("x_label_mass"), 0.8);
        assert_eq!(a.get("type"), Some(&Json::from("noul")));
    }

    #[test]
    fn a_choice_answer_names_the_top_option_in_the_order_sent() {
        let q = question(
            Kind::Choice,
            &["zebra", "apple", "mango"],
            vec![None, None, None],
        );
        let a = answer(&q, &score(&[0.1f64.ln(), 0.6f64.ln(), 0.1f64.ln()])).unwrap();
        assert_eq!(a.get("choice"), Some(&Json::from("apple")));
        let Some(Json::Obj(p)) = a.get("probabilities") else {
            panic!()
        };
        let order: Vec<_> = p.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(order, ["zebra", "apple", "mango"]);
        // 1/3 is the floor of a uniform guess
        close(choice_confidence(&[1.0 / 3.0; 3]), 0.0);
        close(choice_confidence(&[1.0, 0.0, 0.0]), 1.0);
    }

    #[test]
    fn a_score_answer_is_the_mean_level_with_the_levels_as_legend() {
        let level = Json::Obj(vec![("label".into(), "Angry".into())]);
        let q = question(
            Kind::Score,
            &["0", "1", "2"],
            vec![Some("Calm".into()), Some("Civil".into()), Some(level)],
        );
        let a = answer(&q, &score(&[0.2f64.ln(), 0.3f64.ln(), 0.5f64.ln()])).unwrap();
        let Some(Json::Num(n)) = a.get("score") else {
            panic!()
        };
        close(n.as_f64().unwrap(), 0.3 + 2.0 * 0.5);
        assert_eq!(
            a.get("legend").unwrap().compact(),
            r#"{"0":"Calm","1":"Civil","2":{"label":"Angry"}}"#
        );
        // all on the top level is certain; uniform is not
        close(score_confidence(&[0.0, 1.0, 0.0]), 1.0);
        close(score_confidence(&[1.0 / 3.0; 3]), 0.0);
        close(score_confidence(&[1.0]), 1.0);
    }
}
