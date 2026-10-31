//! The canary2 decoder prompt (NeMo's `canary2` prompt format, as
//! transcribe.cpp's `build_prompt_canary2` renders it at
//! `model.cpp:625-692`, db678a0). For transcription with an empty decoder
//! context the prompt is ten ids on a single-SentencePiece vocabulary
//! (canary-1b-v2) and nine on an aggregate one, which drops the empty
//! context slot: `\u{2581}` (single-SP only), `<|startofcontext|>`,
//! `<|startoftranscript|>`, `<|emo:undefined|>`, the source language, the
//! target language (the source again), `<|pnc|>` or `<|nopnc|>`,
//! `<|noitn|>`, `<|notimestamp|>`, `<|nodiarize|>`. Named specials come
//! from [`CanaryHParams::special`]; `<|emo:undefined|>` and `\u{2581}`
//! have no named id and are looked up in [`Vocab`] by string.

use super::model::{CanaryHParams, Vocab};

/// The prompt format this builder renders.
const FORMAT: &str = "canary2";

/// The SentencePiece word marker NeMo renders for the empty decoder
/// context slot.
const SP_SPACE: &str = "\u{2581}";

/// The emotion slot of a transcription prompt.
const EMO_UNDEFINED: &str = "<|emo:undefined|>";

/// Why [`build`] could not render a prompt. Every variant is decided from
/// the hyperparameters and the vocabulary alone, before any graph exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptError {
    /// The language code is not in the file's `general.languages`.
    UnsupportedLanguage(String),
    /// A named special id (`stt.canary.special.<name>_id`) is absent.
    MissingSpecial(String),
    /// A piece looked up by string is absent from the vocabulary.
    MissingPiece(String),
    /// `stt.canary.tokenizer.prompt_format` is not `canary2`.
    UnsupportedFormat(String),
}

impl std::fmt::Display for PromptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedLanguage(lang) => {
                write!(
                    f,
                    "language {lang:?} is not in the model's general.languages"
                )
            }
            Self::MissingSpecial(name) => {
                write!(f, "the model has no stt.canary.special.{name}_id")
            }
            Self::MissingPiece(piece) => {
                write!(f, "the vocabulary has no piece {piece:?}")
            }
            Self::UnsupportedFormat(format) => {
                write!(
                    f,
                    "prompt format {format:?} is not supported, only {FORMAT:?} is"
                )
            }
        }
    }
}

impl std::error::Error for PromptError {}

/// The prompt ids for transcribing `lang` with (`pnc`) or without
/// punctuation and capitalisation. `lang` must be in `hp.languages` and
/// have a `lang.<code>` special; `hp.prompt_format` must be `canary2`.
pub fn build(
    hp: &CanaryHParams,
    vocab: &Vocab,
    lang: &str,
    pnc: bool,
) -> Result<Vec<i32>, PromptError> {
    if hp.prompt_format != FORMAT {
        return Err(PromptError::UnsupportedFormat(hp.prompt_format.clone()));
    }
    if !hp.languages.iter().any(|l| l == lang) {
        return Err(PromptError::UnsupportedLanguage(lang.to_owned()));
    }
    let special = |name: &str| {
        hp.special
            .get(name)
            .map(|&id| id as i32)
            .ok_or_else(|| PromptError::MissingSpecial(name.to_owned()))
    };
    let piece = |text: &str| {
        vocab
            .id_of(text)
            .map(|id| id as i32)
            .ok_or_else(|| PromptError::MissingPiece(text.to_owned()))
    };
    let lang_id = special(&format!("lang.{lang}"))?;

    let mut ids = Vec::with_capacity(10);
    if hp.single_sp {
        ids.push(piece(SP_SPACE)?);
    }
    ids.push(special("startofcontext")?);
    ids.push(special("startoftranscript")?);
    ids.push(piece(EMO_UNDEFINED)?);
    ids.push(lang_id);
    ids.push(lang_id);
    ids.push(special(if pnc { "pnc" } else { "nopnc" })?);
    ids.push(special("noitn")?);
    ids.push(special("notimestamp")?);
    ids.push(special("nodiarize")?);
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speech::model::phase1_hparams;

    /// The Phase 1 fixture plus the prompt specials of the pinned file
    /// (loader test output, `stt.canary.special.*_id`).
    fn hparams() -> CanaryHParams {
        let mut hp = phase1_hparams();
        hp.special.extend(
            [
                ("pnc", 5),
                ("nopnc", 6),
                ("noitn", 9),
                ("notimestamp", 11),
                ("nodiarize", 13),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v)),
        );
        hp
    }

    /// A small table with the string-looked-up pieces at known ids:
    /// `\u{2581}` at 20, `<|emo:undefined|>` at 21.
    fn vocab() -> Vocab {
        let mut pieces: Vec<String> = (0..20).map(|i| format!("<|special{i}|>")).collect();
        pieces[0] = "<unk>".into();
        pieces[2] = "<pad>".into();
        pieces.extend(
            [
                "\u{2581}",
                "<|emo:undefined|>",
                "\u{2581}Hello",
                ",",
                "\u{2581}world",
                ".",
            ]
            .map(str::to_owned),
        );
        Vocab { pieces }
    }

    #[test]
    fn build_renders_ten_ids_on_a_single_sp_vocabulary() {
        let ids = build(&hparams(), &vocab(), "en", true).unwrap();
        assert_eq!(ids, [20, 7, 4, 21, 64, 64, 5, 9, 11, 13]);
    }

    #[test]
    fn build_fills_both_language_slots_with_the_requested_language() {
        let mut hp = hparams();
        hp.special.insert("lang.de".to_string(), 78);
        let ids = build(&hp, &vocab(), "de", true).unwrap();
        assert_eq!(ids, [20, 7, 4, 21, 78, 78, 5, 9, 11, 13]);
    }

    #[test]
    fn build_drops_the_context_slot_on_an_aggregate_vocabulary() {
        let mut hp = hparams();
        hp.single_sp = false;
        let ids = build(&hp, &vocab(), "en", true).unwrap();
        assert_eq!(ids, [7, 4, 21, 64, 64, 5, 9, 11, 13]);
    }

    #[test]
    fn build_uses_nopnc_when_punctuation_is_off() {
        let ids = build(&hparams(), &vocab(), "en", false).unwrap();
        assert_eq!(ids[6], 6);
        assert_eq!(ids.len(), 10);
    }

    #[test]
    fn build_rejects_a_language_outside_general_languages() {
        assert_eq!(
            build(&hparams(), &vocab(), "xx", true),
            Err(PromptError::UnsupportedLanguage("xx".into()))
        );
    }

    #[test]
    fn build_rejects_a_listed_language_without_a_special_id() {
        // `de` is in the fixture's `languages` but has no `lang.de` id.
        assert_eq!(
            build(&hparams(), &vocab(), "de", true),
            Err(PromptError::MissingSpecial("lang.de".into()))
        );
    }

    #[test]
    fn build_rejects_a_missing_piece_and_another_format() {
        let mut vocab = vocab();
        vocab.pieces.truncate(21);
        assert_eq!(
            build(&hparams(), &vocab, "en", true),
            Err(PromptError::MissingPiece("<|emo:undefined|>".into()))
        );
        let mut hp = hparams();
        hp.prompt_format = "canary1".into();
        assert_eq!(
            build(&hp, &vocab, "en", true),
            Err(PromptError::UnsupportedFormat("canary1".into()))
        );
    }

    #[test]
    fn detokenize_drops_specials_maps_the_marker_and_trims() {
        let v = vocab();
        // `<|en|>`-style control pieces, `<unk>`, `<pad>`, an id past the
        // table and a negative id all vanish; the leading marker trims off.
        let ids = [4, 22, 23, 0, 24, 2, 25, 3, 99, -1];
        assert_eq!(v.detokenize(&ids), "Hello, world.");
        assert_eq!(v.detokenize(&[20, 20]), "");
        assert_eq!(v.detokenize(&[]), "");
    }

    #[test]
    fn prompt_error_displays_its_cause() {
        assert_eq!(
            PromptError::UnsupportedLanguage("xx".into()).to_string(),
            "language \"xx\" is not in the model's general.languages"
        );
    }
}
