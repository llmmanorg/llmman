//! Canary 1B v2 as a GGUF describes it: hyperparameters under `stt.*`,
//! the vocabulary, and the tensor names and shapes transcribe.cpp's loader
//! expects (`src/arch/canary/weights.cpp` at db678a0). Metadata comes from
//! [`crate::gguf::Info`], the tensor index from
//! [`crate::mediagen::weights::Index`]; [`load`] uploads the checked file
//! through [`Weights`] and hands out typed handles.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{anyhow, bail, Context as _, Result};

use crate::gguf::{ggml_type_name, Info, Value};
use crate::mediagen::backend::Backend;
use crate::mediagen::ffi::{ty, Tensor};
use crate::mediagen::weights::{self, Index, LoadOpts, Weights};

/// Layer counts past this are a corrupt or hostile header, not a model;
/// the contract set grows with them.
const MAX_LAYERS: u64 = 1024;

/// Dimensions past this are a corrupt or hostile header, not a model;
/// [`expected_shape`] multiplies them as `i64`, and two values below this
/// bound cannot wrap.
const MAX_DIM: u64 = 1 << 24;

/// Mel front end settings, `stt.frontend.*`.
#[derive(Debug, Clone, PartialEq)]
pub struct Frontend {
    /// `stt.frontend.type`, `mel` on the pinned file.
    pub kind: String,
    pub num_mels: u64,
    pub sample_rate: u64,
    pub n_fft: u64,
    pub win_length: u64,
    pub hop_length: u64,
    pub window: String,
    pub normalize: String,
    pub pre_emphasis: f32,
    pub f_min: f32,
    pub f_max: f32,
}

/// Canary hyperparameters, read by [`CanaryHParams::from_info`] from the
/// keys `weights.cpp:40-319` reads. Every key is required here; the
/// converter writes all of them.
#[derive(Debug, Clone, PartialEq)]
pub struct CanaryHParams {
    pub enc_n_layers: u64,
    pub enc_d_model: u64,
    pub enc_n_heads: u64,
    pub enc_d_ff: u64,
    pub enc_conv_kernel: u64,
    pub enc_subsampling_factor: u64,
    pub enc_subsampling_channels: u64,
    pub enc_pos_emb_max_len: u64,
    pub enc_use_bias: bool,
    pub dec_n_layers: u64,
    pub dec_d_model: u64,
    pub dec_n_heads: u64,
    pub dec_d_ff: u64,
    pub dec_max_position: u64,
    pub dec_vocab_size: u64,
    pub dec_activation: String,
    pub dec_pre_ln: bool,
    pub dec_learn_positional_encodings: bool,
    /// `stt.canary.decoder.encoder_decoder_proj`: the file carries
    /// `enc.proj.*` (`weights.cpp:509-512`).
    pub encoder_decoder_proj: bool,
    pub frontend: Frontend,
    /// `stt.canary.tokenizer.prompt_format`, `canary2` on the pinned file.
    pub prompt_format: String,
    pub single_sp: bool,
    /// `general.languages`.
    pub languages: Vec<String>,
    /// Every `stt.canary.special.<name>_id`, keyed by `<name>`: `endoftext`
    /// is 3, `lang.en` is 64.
    pub special: BTreeMap<String, u32>,
    /// `<|endoftext|>`, the stop token.
    pub eos: u32,
    /// `<|startoftranscript|>`.
    pub bos: u32,
    pub pad: u32,
    /// `<|startofcontext|>`, the first prompt token.
    pub startofcontext: u32,
}

/// A required key: `None` from `pick` is missing or the wrong type.
fn kv<T>(info: &Info, key: &str, what: &str, pick: impl FnOnce(&Value) -> Option<T>) -> Result<T> {
    info.metadata
        .get(key)
        .and_then(pick)
        .ok_or_else(|| anyhow!("{key}: missing or not {what}"))
}

fn kv_u64(info: &Info, key: &str) -> Result<u64> {
    info.u64(key)
        .ok_or_else(|| anyhow!("{key}: missing or not an integer"))
}

fn kv_bool(info: &Info, key: &str) -> Result<bool> {
    kv(info, key, "a bool", Value::as_bool)
}

fn kv_string(info: &Info, key: &str) -> Result<String> {
    kv(info, key, "a string", |v| v.as_str().map(str::to_owned))
}

/// `f_min`, `f_max` and `pre_emphasis` are F32 in the file.
fn kv_f32(info: &Info, key: &str) -> Result<f32> {
    kv(info, key, "an F32", |v| match v {
        Value::F32(f) => Some(*f),
        _ => None,
    })
}

fn kv_strings(info: &Info, key: &str) -> Result<Vec<String>> {
    kv(info, key, "a string array", |v| match v {
        Value::Array(items) => items
            .iter()
            .map(|i| i.as_str().map(str::to_owned))
            .collect(),
        _ => None,
    })
}

impl CanaryHParams {
    pub fn from_info(info: &Info) -> Result<Self> {
        match info.architecture() {
            Some("canary") => {}
            other => bail!("general.architecture is {other:?}, expected \"canary\""),
        }
        let u = |key| kv_u64(info, key);
        let b = |key| kv_bool(info, key);
        let s = |key| kv_string(info, key);
        let f = |key| kv_f32(info, key);

        let mut special = BTreeMap::new();
        for (key, value) in &info.metadata {
            let Some(name) = key
                .strip_prefix("stt.canary.special.")
                .and_then(|k| k.strip_suffix("_id"))
            else {
                continue;
            };
            let id = value
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .ok_or_else(|| anyhow!("{key}: not a token id, found {value:?}"))?;
            special.insert(name.to_owned(), id);
        }
        let required = |name: &str| {
            special
                .get(name)
                .copied()
                .ok_or_else(|| anyhow!("stt.canary.special.{name}_id: missing"))
        };

        let hp = CanaryHParams {
            enc_n_layers: u("stt.canary.encoder.n_layers")?,
            enc_d_model: u("stt.canary.encoder.d_model")?,
            enc_n_heads: u("stt.canary.encoder.n_heads")?,
            enc_d_ff: u("stt.canary.encoder.d_ff")?,
            enc_conv_kernel: u("stt.canary.encoder.conv_kernel")?,
            enc_subsampling_factor: u("stt.canary.encoder.subsampling_factor")?,
            enc_subsampling_channels: u("stt.canary.encoder.subsampling_channels")?,
            enc_pos_emb_max_len: u("stt.canary.encoder.pos_emb_max_len")?,
            enc_use_bias: b("stt.canary.encoder.use_bias")?,
            dec_n_layers: u("stt.canary.decoder.n_layers")?,
            dec_d_model: u("stt.canary.decoder.d_model")?,
            dec_n_heads: u("stt.canary.decoder.n_heads")?,
            dec_d_ff: u("stt.canary.decoder.d_ff")?,
            dec_max_position: u("stt.canary.decoder.max_position")?,
            dec_vocab_size: u("stt.canary.decoder.vocab_size")?,
            dec_activation: s("stt.canary.decoder.activation")?,
            dec_pre_ln: b("stt.canary.decoder.pre_ln")?,
            dec_learn_positional_encodings: b("stt.canary.decoder.learn_positional_encodings")?,
            encoder_decoder_proj: b("stt.canary.decoder.encoder_decoder_proj")?,
            frontend: Frontend {
                kind: s("stt.frontend.type")?,
                num_mels: u("stt.frontend.num_mels")?,
                sample_rate: u("stt.frontend.sample_rate")?,
                n_fft: u("stt.frontend.n_fft")?,
                win_length: u("stt.frontend.win_length")?,
                hop_length: u("stt.frontend.hop_length")?,
                window: s("stt.frontend.window")?,
                normalize: s("stt.frontend.normalize")?,
                pre_emphasis: f("stt.frontend.pre_emphasis")?,
                f_min: f("stt.frontend.f_min")?,
                f_max: f("stt.frontend.f_max")?,
            },
            prompt_format: s("stt.canary.tokenizer.prompt_format")?,
            single_sp: b("stt.canary.tokenizer.single_sp")?,
            languages: kv_strings(info, "general.languages")?,
            eos: required("endoftext")?,
            bos: required("startoftranscript")?,
            pad: required("pad")?,
            startofcontext: required("startofcontext")?,
            special,
        };

        // The shape rules divide and multiply by these; a zero, a remainder
        // or a value that wraps the product is not a Canary file.
        for (key, n, max) in [
            ("stt.canary.encoder.n_layers", hp.enc_n_layers, MAX_LAYERS),
            ("stt.canary.decoder.n_layers", hp.dec_n_layers, MAX_LAYERS),
            ("stt.canary.encoder.d_model", hp.enc_d_model, MAX_DIM),
            ("stt.canary.encoder.n_heads", hp.enc_n_heads, MAX_DIM),
            ("stt.canary.decoder.n_heads", hp.dec_n_heads, MAX_DIM),
            ("stt.canary.encoder.d_ff", hp.enc_d_ff, MAX_DIM),
            (
                "stt.canary.encoder.conv_kernel",
                hp.enc_conv_kernel,
                MAX_DIM,
            ),
            (
                "stt.canary.encoder.subsampling_factor",
                hp.enc_subsampling_factor,
                MAX_DIM,
            ),
            (
                "stt.canary.encoder.subsampling_channels",
                hp.enc_subsampling_channels,
                MAX_DIM,
            ),
            ("stt.canary.decoder.d_model", hp.dec_d_model, MAX_DIM),
            ("stt.canary.decoder.d_ff", hp.dec_d_ff, MAX_DIM),
            (
                "stt.canary.decoder.max_position",
                hp.dec_max_position,
                MAX_DIM,
            ),
            ("stt.canary.decoder.vocab_size", hp.dec_vocab_size, MAX_DIM),
            ("stt.frontend.num_mels", hp.frontend.num_mels, MAX_DIM),
        ] {
            if n == 0 || n > max {
                bail!("{key}: implausible value {n}");
            }
        }
        if hp.enc_n_heads == 0 || !hp.enc_d_model.is_multiple_of(hp.enc_n_heads) {
            bail!(
                "stt.canary.encoder.n_heads {} does not divide d_model {}",
                hp.enc_n_heads,
                hp.enc_d_model
            );
        }
        if hp.dec_n_heads == 0 || !hp.dec_d_model.is_multiple_of(hp.dec_n_heads) {
            bail!(
                "stt.canary.decoder.n_heads {} does not divide d_model {}",
                hp.dec_n_heads,
                hp.dec_d_model
            );
        }
        if hp.enc_subsampling_factor == 0
            || !hp
                .frontend
                .num_mels
                .is_multiple_of(hp.enc_subsampling_factor)
        {
            bail!(
                "stt.canary.encoder.subsampling_factor {} does not divide stt.frontend.num_mels {}",
                hp.enc_subsampling_factor,
                hp.frontend.num_mels
            );
        }
        // Every special token is looked up in the embedding table.
        for (name, id) in &hp.special {
            if u64::from(*id) >= hp.dec_vocab_size {
                bail!(
                    "stt.canary.special.{name}_id: {id} is not below stt.canary.decoder.vocab_size {}",
                    hp.dec_vocab_size
                );
            }
        }
        Ok(hp)
    }

    /// Encoder head width, `d_model / n_heads`.
    pub fn head_dim(&self) -> u64 {
        self.enc_d_model / self.enc_n_heads
    }
}

/// The SentencePiece pieces from the GGUF key `tokenizer.ggml.tokens`, one
/// per token id (16384 strings on canary-1b-v2; the first five are
/// `<unk>`, `<|nospeech|>`, `<pad>`, `<|endoftext|>`,
/// `<|startoftranscript|>`). The file also carries `tokenizer.ggml.model`,
/// `tokenizer.ggml.scores` and `tokenizer.ggml.token_type`, which the
/// decoder does not need.
#[derive(Debug, Clone)]
pub struct Vocab {
    pub pieces: Vec<String>,
}

impl Vocab {
    /// Fails when the piece count differs from `dec_vocab_size`.
    pub fn from_info(info: &Info, hp: &CanaryHParams) -> Result<Self> {
        let pieces = kv_strings(info, "tokenizer.ggml.tokens")?;
        if pieces.len() as u64 != hp.dec_vocab_size {
            bail!(
                "tokenizer.ggml.tokens has {} pieces, stt.canary.decoder.vocab_size is {}",
                pieces.len(),
                hp.dec_vocab_size
            );
        }
        Ok(Vocab { pieces })
    }
}

/// Tensor names transcribe.cpp's Canary loader reads (`src/arch/canary/
/// weights.cpp:443-573` at db678a0), outside the per-layer blocks.
pub const FIXED_TENSORS: &[&str] = &[
    "enc.pre_encode.conv.0.weight",
    "enc.pre_encode.conv.0.bias",
    "enc.pre_encode.conv.2.weight",
    "enc.pre_encode.conv.2.bias",
    "enc.pre_encode.conv.3.weight",
    "enc.pre_encode.conv.3.bias",
    "enc.pre_encode.conv.5.weight",
    "enc.pre_encode.conv.5.bias",
    "enc.pre_encode.conv.6.weight",
    "enc.pre_encode.conv.6.bias",
    "enc.pre_encode.out.weight",
    "enc.pre_encode.out.bias",
    "dec.embed.token.weight",
    "dec.embed.pos_enc",
    "dec.embed.norm.weight",
    "dec.embed.norm.bias",
    "dec.norm.weight",
    "dec.norm.bias",
    "dec.head.weight",
    "dec.head.bias",
];

/// Per-block suffixes under `enc.blocks.N.` (same source, `:460-506`).
pub const ENC_BLOCK_TENSORS: &[&str] = &[
    "norm_ff1.weight",
    "norm_ff1.bias",
    "ff1.linear1.weight",
    "ff1.linear1.bias",
    "ff1.linear2.weight",
    "ff1.linear2.bias",
    "norm_attn.weight",
    "norm_attn.bias",
    "attn.linear_q.weight",
    "attn.linear_q.bias",
    "attn.linear_k.weight",
    "attn.linear_k.bias",
    "attn.linear_v.weight",
    "attn.linear_v.bias",
    "attn.linear_out.weight",
    "attn.linear_out.bias",
    "attn.linear_pos.weight",
    "attn.pos_bias_u",
    "attn.pos_bias_v",
    "norm_conv.weight",
    "norm_conv.bias",
    "conv.pointwise1.weight",
    "conv.pointwise1.bias",
    "conv.depthwise.weight",
    "conv.depthwise.bias",
    "conv.pointwise2.weight",
    "conv.pointwise2.bias",
    "conv.bn.weight",
    "conv.bn.bias",
    "conv.bn.running_mean",
    "conv.bn.running_var",
    "norm_ff2.weight",
    "norm_ff2.bias",
    "ff2.linear1.weight",
    "ff2.linear1.bias",
    "ff2.linear2.weight",
    "ff2.linear2.bias",
    "norm_out.weight",
    "norm_out.bias",
];

/// Per-layer suffixes under `dec.layer.N.` (same source, `:534-565`).
pub const DEC_LAYER_TENSORS: &[&str] = &[
    "norm1.weight",
    "norm1.bias",
    "self_attn.q.weight",
    "self_attn.q.bias",
    "self_attn.k.weight",
    "self_attn.k.bias",
    "self_attn.v.weight",
    "self_attn.v.bias",
    "self_attn.o.weight",
    "self_attn.o.bias",
    "norm2.weight",
    "norm2.bias",
    "cross_attn.q.weight",
    "cross_attn.q.bias",
    "cross_attn.k.weight",
    "cross_attn.k.bias",
    "cross_attn.v.weight",
    "cross_attn.v.bias",
    "cross_attn.o.weight",
    "cross_attn.o.bias",
    "norm3.weight",
    "norm3.bias",
    "ffn.up.weight",
    "ffn.up.bias",
    "ffn.down.weight",
    "ffn.down.bias",
];

/// The encoder to decoder projection, present when
/// `stt.canary.decoder.encoder_decoder_proj` is true (`weights.cpp:509-512`).
const ENC_PROJ_TENSORS: &[&str] = &["enc.proj.weight", "enc.proj.bias"];

/// The full tensor contract for `n_enc` encoder blocks and `n_dec` decoder
/// layers, with `enc.proj.*` only when `encoder_decoder_proj` is set.
pub fn contract_names(n_enc: u64, n_dec: u64, encoder_decoder_proj: bool) -> BTreeSet<String> {
    let fixed = FIXED_TENSORS.iter().map(|s| s.to_string());
    let proj = ENC_PROJ_TENSORS
        .iter()
        .filter(|_| encoder_decoder_proj)
        .map(|s| s.to_string());
    let enc = (0..n_enc).flat_map(|i| {
        ENC_BLOCK_TENSORS
            .iter()
            .map(move |s| format!("enc.blocks.{i}.{s}"))
    });
    let dec = (0..n_dec).flat_map(|i| {
        DEC_LAYER_TENSORS
            .iter()
            .map(move |s| format!("dec.layer.{i}.{s}"))
    });
    fixed.chain(proj).chain(enc).chain(dec).collect()
}

/// The suffix of `name` under `prefix<i>.` for a layer index `i` below
/// `count`, written as plain decimal digits.
fn layer_suffix<'a>(name: &'a str, prefix: &str, count: u64) -> Option<&'a str> {
    let rest = name.strip_prefix(prefix)?;
    let (index, suffix) = rest.split_once('.')?;
    let i: u64 = index.parse().ok()?;
    (i.to_string() == index && i < count).then_some(suffix)
}

/// The ggml `ne` (`ne[0]` fastest) of a contract tensor, from
/// `weights.cpp:425-573`; `None` for a name outside the contract. The
/// pinned Q8_0 file carries these shapes: conv.0/2/5 `[3, 3, 1, 256]`,
/// conv.3/6 `[1, 1, 256, 256]`, `enc.pre_encode.out.weight` `[4096, 1024]`,
/// pointwise1 `[1, 1024, 2048]`, depthwise `[9, 1, 1024]`, pointwise2
/// `[1, 1024, 1024]`.
pub fn expected_shape(name: &str, hp: &CanaryHParams) -> Option<Vec<i64>> {
    let d = |v: u64| v as i64;
    let channels = d(hp.enc_subsampling_channels);
    let d_enc = d(hp.enc_d_model);
    let d_ff_e = d(hp.enc_d_ff);
    let n_heads = d(hp.enc_n_heads);
    let head_dim = d(hp.head_dim());
    let k = d(hp.enc_conv_kernel);
    let d_dec = d(hp.dec_d_model);
    let d_ff_d = d(hp.dec_d_ff);
    let vocab = d(hp.dec_vocab_size);
    let pre_encode_in = channels * d(hp.frontend.num_mels / hp.enc_subsampling_factor);

    if let Some(suffix) = layer_suffix(name, "enc.blocks.", hp.enc_n_layers) {
        return Some(match suffix {
            "ff1.linear1.weight" | "ff2.linear1.weight" => vec![d_enc, d_ff_e],
            "ff1.linear1.bias" | "ff2.linear1.bias" => vec![d_ff_e],
            "ff1.linear2.weight" | "ff2.linear2.weight" => vec![d_ff_e, d_enc],
            "attn.linear_q.weight"
            | "attn.linear_k.weight"
            | "attn.linear_v.weight"
            | "attn.linear_out.weight"
            | "attn.linear_pos.weight" => vec![d_enc, d_enc],
            "attn.pos_bias_u" | "attn.pos_bias_v" => vec![head_dim, n_heads],
            "conv.pointwise1.weight" => vec![1, d_enc, 2 * d_enc],
            "conv.pointwise1.bias" => vec![2 * d_enc],
            "conv.depthwise.weight" => vec![k, 1, d_enc],
            "conv.pointwise2.weight" => vec![1, d_enc, d_enc],
            // Every other norm, bias and batch-norm vector.
            s if ENC_BLOCK_TENSORS.contains(&s) => vec![d_enc],
            _ => return None,
        });
    }
    if let Some(suffix) = layer_suffix(name, "dec.layer.", hp.dec_n_layers) {
        return Some(match suffix {
            "self_attn.q.weight"
            | "self_attn.k.weight"
            | "self_attn.v.weight"
            | "self_attn.o.weight"
            | "cross_attn.q.weight"
            | "cross_attn.k.weight"
            | "cross_attn.v.weight"
            | "cross_attn.o.weight" => vec![d_dec, d_dec],
            "ffn.up.weight" => vec![d_dec, d_ff_d],
            "ffn.up.bias" => vec![d_ff_d],
            "ffn.down.weight" => vec![d_ff_d, d_dec],
            // Every other norm and bias vector.
            s if DEC_LAYER_TENSORS.contains(&s) => vec![d_dec],
            _ => return None,
        });
    }
    Some(match name {
        "enc.pre_encode.conv.0.weight"
        | "enc.pre_encode.conv.2.weight"
        | "enc.pre_encode.conv.5.weight" => vec![3, 3, 1, channels],
        "enc.pre_encode.conv.3.weight" | "enc.pre_encode.conv.6.weight" => {
            vec![1, 1, channels, channels]
        }
        "enc.pre_encode.conv.0.bias"
        | "enc.pre_encode.conv.2.bias"
        | "enc.pre_encode.conv.3.bias"
        | "enc.pre_encode.conv.5.bias"
        | "enc.pre_encode.conv.6.bias" => vec![channels],
        "enc.pre_encode.out.weight" => vec![pre_encode_in, d_enc],
        "enc.pre_encode.out.bias" => vec![d_enc],
        "enc.proj.weight" if hp.encoder_decoder_proj => vec![d_enc, d_dec],
        "enc.proj.bias" if hp.encoder_decoder_proj => vec![d_dec],
        "dec.embed.token.weight" | "dec.head.weight" => vec![d_dec, vocab],
        "dec.embed.pos_enc" => vec![d_dec, d(hp.dec_max_position)],
        "dec.embed.norm.weight" | "dec.embed.norm.bias" | "dec.norm.weight" | "dec.norm.bias" => {
            vec![d_dec]
        }
        "dec.head.bias" => vec![vocab],
        _ => return None,
    })
}

/// Checks a tensor index against the contract: one error listing every
/// duplicate, unmapped and missing name and every shape mismatch.
pub fn check_index(idx: &Index, hp: &CanaryHParams) -> Result<()> {
    let expected = contract_names(hp.enc_n_layers, hp.dec_n_layers, hp.encoder_decoder_proj);
    let mut problems = Vec::new();
    let mut found: BTreeMap<&str, &[i64]> = BTreeMap::new();
    for t in &idx.tensors {
        if found.insert(&t.name, &t.ne).is_some() {
            problems.push(format!("duplicate: {}", t.name));
        }
    }
    for (name, ne) in &found {
        if !expected.contains(*name) {
            problems.push(format!("unmapped: {name}"));
            continue;
        }
        match expected_shape(name, hp) {
            Some(want) if want.as_slice() == *ne => {}
            Some(want) => problems.push(format!("shape: {name} is {ne:?}, expected {want:?}")),
            None => problems.push(format!("shape: {name} has no rule")),
        }
    }
    for name in &expected {
        if !found.contains_key(name.as_str()) {
            problems.push(format!("missing: {name}"));
        }
    }
    if problems.is_empty() {
        return Ok(());
    }
    bail!(
        "canary tensor contract: {} problems\n  {}",
        problems.len(),
        problems.join("\n  ")
    )
}

/// How a contract tensor is consumed, which bounds its storage type.
/// After transcribe.cpp's quantize buckets
/// (`tools/transcribe-quantize/policy.cpp`): its loader reads vectors as
/// F32, conv kernels as F32 or F16, and feeds every other weight to
/// `mul_mat`. Vectors stay F32 because ggml's CPU add and mul take an
/// F32 second operand only.
#[derive(Debug, Clone, Copy, PartialEq)]
enum TensorClass {
    /// Biases, norm scales, BatchNorm stats, `pos_bias_*`, `pos_enc`.
    Vector,
    /// `enc.pre_encode.conv.*.weight` and `conv.depthwise.weight`.
    Conv,
    /// A `mul_mat` operand.
    Linear,
}

impl TensorClass {
    fn of(name: &str) -> Self {
        let vector = name.ends_with(".bias")
            || name.contains(".bn.")
            || (name.contains("norm") && name.ends_with(".weight"))
            || name.ends_with(".pos_bias_u")
            || name.ends_with(".pos_bias_v")
            || name.ends_with(".pos_enc");
        let conv = name.ends_with(".weight")
            && (name.starts_with("enc.pre_encode.conv.")
                || name.ends_with(".conv.depthwise.weight"));
        if vector {
            Self::Vector
        } else if conv {
            Self::Conv
        } else {
            Self::Linear
        }
    }

    /// The ggml types this class may be stored as: floats for vectors
    /// and kernels; for a linear also the quantized types ggml's CPU
    /// `mul_mat` accepts (`ggml-cpu.c` at b11669).
    fn allowed(self) -> &'static [u32] {
        match self {
            Self::Vector => &[ty::F32],
            Self::Conv => &[ty::F32, ty::F16],
            Self::Linear => &[
                ty::F32,
                ty::F16,
                ty::BF16,
                ty::Q4_0,
                ty::Q4_1,
                ty::Q5_0,
                ty::Q5_1,
                ty::Q8_0,
                ty::Q2_K,
                ty::Q3_K,
                ty::Q4_K,
                ty::Q5_K,
                ty::Q6_K,
                ty::IQ4_NL,
            ],
        }
    }

    fn describe(self) -> &'static str {
        match self {
            Self::Vector => "a bias or norm must be F32",
            Self::Conv => "a conv kernel must be F32 or F16",
            Self::Linear => {
                "a linear weight must be F32, F16, BF16 or a quantized type ggml's CPU mul_mat accepts"
            }
        }
    }
}

/// Checks the storage type of every tensor against its [`TensorClass`]
/// and ggml's block size, before `ggml_new_tensor` can assert on a type
/// it does not know: one error listing every problem. `blck_size` is
/// `api.ggml_blck_size`, passed as a closure so the policy tests run
/// without the libraries; the class check runs first, so it is only
/// called with an id below `GGML_TYPE_COUNT`.
pub fn check_types(idx: &Index, blck_size: &dyn Fn(u32) -> i64) -> Result<()> {
    let mut problems = Vec::new();
    for t in &idx.tensors {
        let class = TensorClass::of(&t.name);
        let stored = format!("{} ({})", ggml_type_name(t.ty), t.ty);
        if !class.allowed().contains(&t.ty) {
            problems.push(format!(
                "type: {} is {stored}, {}",
                t.name,
                class.describe()
            ));
            continue;
        }
        let blck = blck_size(t.ty);
        if blck <= 0 {
            problems.push(format!(
                "type: {} is {stored}, which this ggml does not support",
                t.name
            ));
            continue;
        }
        let width = t.ne.first().copied().unwrap_or(1);
        if width % blck != 0 {
            problems.push(format!(
                "type: {} is {stored} with a row of {width}, not a multiple of its block {blck}",
                t.name
            ));
        }
    }
    if problems.is_empty() {
        return Ok(());
    }
    bail!(
        "canary tensor types: {} problems\n  {}",
        problems.len(),
        problems.join("\n  ")
    )
}

/// `enc.pre_encode.*`: the dw_striding subsampler, three 3x3 convolutions
/// (`conv.0`, `conv.2`, `conv.5`), two 1x1 (`conv.3`, `conv.6`) and the
/// linear onto `d_model`. `_b` is the bias of the tensor before it.
pub struct PreEncode {
    pub conv0: Tensor,
    pub conv0_b: Tensor,
    pub conv2: Tensor,
    pub conv2_b: Tensor,
    pub conv3: Tensor,
    pub conv3_b: Tensor,
    pub conv5: Tensor,
    pub conv5_b: Tensor,
    pub conv6: Tensor,
    pub conv6_b: Tensor,
    pub out: Tensor,
    pub out_b: Tensor,
}

/// One `enc.blocks.N.*` FastConformer block, a field per
/// [`ENC_BLOCK_TENSORS`] suffix.
pub struct EncBlock {
    pub norm_ff1: Tensor,
    pub norm_ff1_b: Tensor,
    pub ff1_linear1: Tensor,
    pub ff1_linear1_b: Tensor,
    pub ff1_linear2: Tensor,
    pub ff1_linear2_b: Tensor,
    pub norm_attn: Tensor,
    pub norm_attn_b: Tensor,
    pub attn_q: Tensor,
    pub attn_q_b: Tensor,
    pub attn_k: Tensor,
    pub attn_k_b: Tensor,
    pub attn_v: Tensor,
    pub attn_v_b: Tensor,
    pub attn_out: Tensor,
    pub attn_out_b: Tensor,
    pub attn_pos: Tensor,
    pub pos_bias_u: Tensor,
    pub pos_bias_v: Tensor,
    pub norm_conv: Tensor,
    pub norm_conv_b: Tensor,
    pub pointwise1: Tensor,
    pub pointwise1_b: Tensor,
    pub depthwise: Tensor,
    pub depthwise_b: Tensor,
    pub pointwise2: Tensor,
    pub pointwise2_b: Tensor,
    pub bn: Tensor,
    pub bn_b: Tensor,
    pub bn_mean: Tensor,
    pub bn_var: Tensor,
    pub norm_ff2: Tensor,
    pub norm_ff2_b: Tensor,
    pub ff2_linear1: Tensor,
    pub ff2_linear1_b: Tensor,
    pub ff2_linear2: Tensor,
    pub ff2_linear2_b: Tensor,
    pub norm_out: Tensor,
    pub norm_out_b: Tensor,
}

/// `dec.embed.*`: token table, the stored sinusoidal position table and
/// the embedding norm.
pub struct DecEmbed {
    pub token: Tensor,
    pub pos_enc: Tensor,
    pub norm: Tensor,
    pub norm_b: Tensor,
}

/// One `dec.layer.N.*` Transformer layer, a field per [`DEC_LAYER_TENSORS`]
/// suffix.
pub struct DecLayer {
    pub norm1: Tensor,
    pub norm1_b: Tensor,
    pub self_q: Tensor,
    pub self_q_b: Tensor,
    pub self_k: Tensor,
    pub self_k_b: Tensor,
    pub self_v: Tensor,
    pub self_v_b: Tensor,
    pub self_o: Tensor,
    pub self_o_b: Tensor,
    pub norm2: Tensor,
    pub norm2_b: Tensor,
    pub cross_q: Tensor,
    pub cross_q_b: Tensor,
    pub cross_k: Tensor,
    pub cross_k_b: Tensor,
    pub cross_v: Tensor,
    pub cross_v_b: Tensor,
    pub cross_o: Tensor,
    pub cross_o_b: Tensor,
    pub norm3: Tensor,
    pub norm3_b: Tensor,
    pub ffn_up: Tensor,
    pub ffn_up_b: Tensor,
    pub ffn_down: Tensor,
    pub ffn_down_b: Tensor,
}

/// Every contract tensor as a handle into the model's weight buffer.
pub struct CanaryTensors {
    pub pre_encode: PreEncode,
    pub enc_blocks: Vec<EncBlock>,
    pub dec_embed: DecEmbed,
    pub dec_layers: Vec<DecLayer>,
    pub dec_norm: Tensor,
    pub dec_norm_b: Tensor,
    pub head: Tensor,
    pub head_b: Tensor,
    /// `enc.proj.weight` and `enc.proj.bias` when `encoder_decoder_proj` is set.
    pub enc_proj: Option<(Tensor, Tensor)>,
}

impl CanaryTensors {
    /// Looks every contract name up through `get`; its error carries the
    /// name (`Weights::get` says `missing tensor "<name>"`).
    fn build(hp: &CanaryHParams, get: &mut dyn FnMut(&str) -> Result<Tensor>) -> Result<Self> {
        let mut t = |name: &str| get(name);
        let pre_encode = PreEncode {
            conv0: t("enc.pre_encode.conv.0.weight")?,
            conv0_b: t("enc.pre_encode.conv.0.bias")?,
            conv2: t("enc.pre_encode.conv.2.weight")?,
            conv2_b: t("enc.pre_encode.conv.2.bias")?,
            conv3: t("enc.pre_encode.conv.3.weight")?,
            conv3_b: t("enc.pre_encode.conv.3.bias")?,
            conv5: t("enc.pre_encode.conv.5.weight")?,
            conv5_b: t("enc.pre_encode.conv.5.bias")?,
            conv6: t("enc.pre_encode.conv.6.weight")?,
            conv6_b: t("enc.pre_encode.conv.6.bias")?,
            out: t("enc.pre_encode.out.weight")?,
            out_b: t("enc.pre_encode.out.bias")?,
        };
        let mut enc_blocks = Vec::with_capacity(hp.enc_n_layers as usize);
        for i in 0..hp.enc_n_layers {
            let mut t = |suffix: &str| get(&format!("enc.blocks.{i}.{suffix}"));
            enc_blocks.push(EncBlock {
                norm_ff1: t("norm_ff1.weight")?,
                norm_ff1_b: t("norm_ff1.bias")?,
                ff1_linear1: t("ff1.linear1.weight")?,
                ff1_linear1_b: t("ff1.linear1.bias")?,
                ff1_linear2: t("ff1.linear2.weight")?,
                ff1_linear2_b: t("ff1.linear2.bias")?,
                norm_attn: t("norm_attn.weight")?,
                norm_attn_b: t("norm_attn.bias")?,
                attn_q: t("attn.linear_q.weight")?,
                attn_q_b: t("attn.linear_q.bias")?,
                attn_k: t("attn.linear_k.weight")?,
                attn_k_b: t("attn.linear_k.bias")?,
                attn_v: t("attn.linear_v.weight")?,
                attn_v_b: t("attn.linear_v.bias")?,
                attn_out: t("attn.linear_out.weight")?,
                attn_out_b: t("attn.linear_out.bias")?,
                attn_pos: t("attn.linear_pos.weight")?,
                pos_bias_u: t("attn.pos_bias_u")?,
                pos_bias_v: t("attn.pos_bias_v")?,
                norm_conv: t("norm_conv.weight")?,
                norm_conv_b: t("norm_conv.bias")?,
                pointwise1: t("conv.pointwise1.weight")?,
                pointwise1_b: t("conv.pointwise1.bias")?,
                depthwise: t("conv.depthwise.weight")?,
                depthwise_b: t("conv.depthwise.bias")?,
                pointwise2: t("conv.pointwise2.weight")?,
                pointwise2_b: t("conv.pointwise2.bias")?,
                bn: t("conv.bn.weight")?,
                bn_b: t("conv.bn.bias")?,
                bn_mean: t("conv.bn.running_mean")?,
                bn_var: t("conv.bn.running_var")?,
                norm_ff2: t("norm_ff2.weight")?,
                norm_ff2_b: t("norm_ff2.bias")?,
                ff2_linear1: t("ff2.linear1.weight")?,
                ff2_linear1_b: t("ff2.linear1.bias")?,
                ff2_linear2: t("ff2.linear2.weight")?,
                ff2_linear2_b: t("ff2.linear2.bias")?,
                norm_out: t("norm_out.weight")?,
                norm_out_b: t("norm_out.bias")?,
            });
        }
        let enc_proj = if hp.encoder_decoder_proj {
            Some((get("enc.proj.weight")?, get("enc.proj.bias")?))
        } else {
            None
        };
        let dec_embed = DecEmbed {
            token: get("dec.embed.token.weight")?,
            pos_enc: get("dec.embed.pos_enc")?,
            norm: get("dec.embed.norm.weight")?,
            norm_b: get("dec.embed.norm.bias")?,
        };
        let mut dec_layers = Vec::with_capacity(hp.dec_n_layers as usize);
        for i in 0..hp.dec_n_layers {
            let mut t = |suffix: &str| get(&format!("dec.layer.{i}.{suffix}"));
            dec_layers.push(DecLayer {
                norm1: t("norm1.weight")?,
                norm1_b: t("norm1.bias")?,
                self_q: t("self_attn.q.weight")?,
                self_q_b: t("self_attn.q.bias")?,
                self_k: t("self_attn.k.weight")?,
                self_k_b: t("self_attn.k.bias")?,
                self_v: t("self_attn.v.weight")?,
                self_v_b: t("self_attn.v.bias")?,
                self_o: t("self_attn.o.weight")?,
                self_o_b: t("self_attn.o.bias")?,
                norm2: t("norm2.weight")?,
                norm2_b: t("norm2.bias")?,
                cross_q: t("cross_attn.q.weight")?,
                cross_q_b: t("cross_attn.q.bias")?,
                cross_k: t("cross_attn.k.weight")?,
                cross_k_b: t("cross_attn.k.bias")?,
                cross_v: t("cross_attn.v.weight")?,
                cross_v_b: t("cross_attn.v.bias")?,
                cross_o: t("cross_attn.o.weight")?,
                cross_o_b: t("cross_attn.o.bias")?,
                norm3: t("norm3.weight")?,
                norm3_b: t("norm3.bias")?,
                ffn_up: t("ffn.up.weight")?,
                ffn_up_b: t("ffn.up.bias")?,
                ffn_down: t("ffn.down.weight")?,
                ffn_down_b: t("ffn.down.bias")?,
            });
        }
        Ok(CanaryTensors {
            pre_encode,
            enc_blocks,
            dec_embed,
            dec_layers,
            dec_norm: get("dec.norm.weight")?,
            dec_norm_b: get("dec.norm.bias")?,
            head: get("dec.head.weight")?,
            head_b: get("dec.head.bias")?,
            enc_proj,
        })
    }

    /// Handles held: the contract size for these layer counts.
    pub fn count(&self) -> usize {
        FIXED_TENSORS.len()
            + self.enc_blocks.len() * ENC_BLOCK_TENSORS.len()
            + self.dec_layers.len() * DEC_LAYER_TENSORS.len()
            + if self.enc_proj.is_some() { 2 } else { 0 }
    }
}

/// A Canary model in one backend buffer. `tensors` point into the buffer
/// `weights` owns, so the fields drop in this order and the buffer goes
/// with the model. The caller drops the model before the [`Backend`] it
/// was loaded on, as `mediagen::Context` orders its fields.
pub struct CanaryModel {
    pub hp: CanaryHParams,
    pub vocab: Vocab,
    pub tensors: CanaryTensors,
    weights: Weights,
}

impl CanaryModel {
    /// Size of the weight buffer in bytes.
    pub fn weight_bytes(&self) -> usize {
        self.weights.size()
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.count()
    }
}

/// Reads, checks and uploads a Canary GGUF onto `be`'s weight buffer type.
/// Types stay as stored: `Weights` converts BF16 alone, so quantized
/// linears and F16/F32 conv kernels arrive byte for byte.
pub fn load(path: &Path, be: &Backend) -> Result<CanaryModel> {
    let info = crate::gguf::read_info(path)?;
    let hp = CanaryHParams::from_info(&info).with_context(|| path.display().to_string())?;
    let vocab = Vocab::from_info(&info, &hp).with_context(|| path.display().to_string())?;
    let idx = weights::read_index(path)?;
    check_index(&idx, &hp).with_context(|| path.display().to_string())?;
    check_types(&idx, &|t| unsafe { (be.api.ggml_blck_size)(t) })
        .with_context(|| path.display().to_string())?;
    let weights = Weights::load(be.api, path, be.weight_buft(), &LoadOpts::default())?;
    let tensors = CanaryTensors::build(&hp, &mut |name| weights.get(name))?;
    Ok(CanaryModel {
        hp,
        vocab,
        tensors,
        weights,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mediagen::weights::TensorDesc;
    use std::collections::HashMap;

    /// The pinned canary-1b-v2 file's values (Phase 1 record).
    fn phase1_hparams() -> CanaryHParams {
        let special: BTreeMap<String, u32> = [
            ("endoftext", 3),
            ("startoftranscript", 4),
            ("pad", 2),
            ("startofcontext", 7),
            ("nospeech", 1),
            ("lang.en", 64),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        CanaryHParams {
            enc_n_layers: 32,
            enc_d_model: 1024,
            enc_n_heads: 8,
            enc_d_ff: 4096,
            enc_conv_kernel: 9,
            enc_subsampling_factor: 8,
            enc_subsampling_channels: 256,
            enc_pos_emb_max_len: 5000,
            enc_use_bias: true,
            dec_n_layers: 8,
            dec_d_model: 1024,
            dec_n_heads: 8,
            dec_d_ff: 4096,
            dec_max_position: 1024,
            dec_vocab_size: 16384,
            dec_activation: "relu".into(),
            dec_pre_ln: true,
            dec_learn_positional_encodings: false,
            encoder_decoder_proj: false,
            frontend: Frontend {
                kind: "mel".into(),
                num_mels: 128,
                sample_rate: 16000,
                n_fft: 512,
                win_length: 400,
                hop_length: 160,
                window: "hann".into(),
                normalize: "per_feature".into(),
                pre_emphasis: 0.97,
                f_min: 0.0,
                f_max: 8000.0,
            },
            prompt_format: "canary2".into(),
            single_sp: true,
            languages: vec!["en".into(), "de".into()],
            special,
            eos: 3,
            bos: 4,
            pad: 2,
            startofcontext: 7,
        }
    }

    /// The same values as GGUF metadata, typed as the file stores them.
    fn phase1_metadata() -> HashMap<String, Value> {
        let u = Value::U32;
        let b = Value::Bool;
        let s = |v: &str| Value::String(v.to_string());
        let strings =
            |v: &[&str]| Value::Array(v.iter().map(|x| Value::String(x.to_string())).collect());
        let entries = vec![
            ("general.architecture", s("canary")),
            ("general.languages", strings(&["en", "de"])),
            ("stt.canary.encoder.n_layers", u(32)),
            ("stt.canary.encoder.d_model", u(1024)),
            ("stt.canary.encoder.n_heads", u(8)),
            ("stt.canary.encoder.d_ff", u(4096)),
            ("stt.canary.encoder.conv_kernel", u(9)),
            ("stt.canary.encoder.subsampling_factor", u(8)),
            ("stt.canary.encoder.subsampling_channels", u(256)),
            ("stt.canary.encoder.pos_emb_max_len", u(5000)),
            ("stt.canary.encoder.use_bias", b(true)),
            ("stt.canary.decoder.n_layers", u(8)),
            ("stt.canary.decoder.d_model", u(1024)),
            ("stt.canary.decoder.n_heads", u(8)),
            ("stt.canary.decoder.d_ff", u(4096)),
            ("stt.canary.decoder.max_position", u(1024)),
            ("stt.canary.decoder.vocab_size", u(16384)),
            ("stt.canary.decoder.activation", s("relu")),
            ("stt.canary.decoder.pre_ln", b(true)),
            ("stt.canary.decoder.learn_positional_encodings", b(false)),
            ("stt.canary.decoder.encoder_decoder_proj", b(false)),
            ("stt.frontend.type", s("mel")),
            ("stt.frontend.num_mels", u(128)),
            ("stt.frontend.sample_rate", u(16000)),
            ("stt.frontend.n_fft", u(512)),
            ("stt.frontend.win_length", u(400)),
            ("stt.frontend.hop_length", u(160)),
            ("stt.frontend.window", s("hann")),
            ("stt.frontend.normalize", s("per_feature")),
            ("stt.frontend.pre_emphasis", Value::F32(0.97)),
            ("stt.frontend.f_min", Value::F32(0.0)),
            ("stt.frontend.f_max", Value::F32(8000.0)),
            ("stt.canary.tokenizer.prompt_format", s("canary2")),
            ("stt.canary.tokenizer.single_sp", b(true)),
            ("stt.canary.special.endoftext_id", u(3)),
            ("stt.canary.special.startoftranscript_id", u(4)),
            ("stt.canary.special.pad_id", u(2)),
            ("stt.canary.special.startofcontext_id", u(7)),
            ("stt.canary.special.nospeech_id", u(1)),
            ("stt.canary.special.lang.en_id", u(64)),
        ];
        entries
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect()
    }

    fn info(metadata: HashMap<String, Value>) -> Info {
        Info {
            metadata,
            ..Default::default()
        }
    }

    fn desc(name: &str, ne: &[i64]) -> TensorDesc {
        typed(name, ne, ty::F32)
    }

    fn typed(name: &str, ne: &[i64], ty: u32) -> TensorDesc {
        TensorDesc {
            name: name.to_string(),
            ty,
            ne: ne.to_vec(),
            offset: 0,
            nbytes: 0,
        }
    }

    /// `ggml_blck_size` of b11669 for the ids a test uses; an id past
    /// `GGML_TYPE_COUNT` would read past ggml's table, so it panics here.
    fn blck_size(ty: u32) -> i64 {
        match ty {
            0 | 1 | 30 => 1,
            2 | 3 | 6 | 7 | 8 | 20 => 32,
            10..=14 => 256,
            4 | 5 => 0,
            t => panic!("ggml_blck_size({t}): not a type the allowlist can reach"),
        }
    }

    #[test]
    fn contract_names_counts_the_pinned_file() {
        let without = contract_names(32, 8, false);
        assert_eq!(without.len(), 1476);
        let with = contract_names(32, 8, true);
        let added: Vec<&String> = with.difference(&without).collect();
        assert_eq!(added, ["enc.proj.bias", "enc.proj.weight"]);
        assert!(without.contains("enc.blocks.31.norm_out.bias"));
        assert!(without.contains("dec.layer.7.ffn.down.bias"));
    }

    #[test]
    fn expected_shape_follows_the_hyperparameters() {
        let hp = phase1_hparams();
        let shape = |name| expected_shape(name, &hp).unwrap();
        assert_eq!(shape("enc.pre_encode.out.weight"), [4096, 1024]);
        assert_eq!(shape("enc.blocks.0.attn.pos_bias_u"), [128, 8]);
        assert_eq!(shape("enc.blocks.3.conv.depthwise.weight"), [9, 1, 1024]);
        assert_eq!(shape("dec.embed.pos_enc"), [1024, 1024]);
        assert_eq!(shape("dec.head.weight"), [1024, 16384]);
        assert_eq!(shape("enc.pre_encode.conv.0.weight"), [3, 3, 1, 256]);
        assert_eq!(shape("enc.pre_encode.conv.3.weight"), [1, 1, 256, 256]);
        assert_eq!(
            shape("enc.blocks.1.conv.pointwise1.weight"),
            [1, 1024, 2048]
        );
        assert_eq!(shape("dec.layer.7.ffn.down.weight"), [4096, 1024]);
    }

    #[test]
    fn expected_shape_covers_the_contract_and_nothing_else() {
        let mut hp = phase1_hparams();
        assert_eq!(expected_shape("enc.proj.weight", &hp), None);
        hp.encoder_decoder_proj = true;
        for name in contract_names(hp.enc_n_layers, hp.dec_n_layers, true) {
            assert!(expected_shape(&name, &hp).is_some(), "{name}");
        }
        assert_eq!(
            expected_shape("enc.proj.weight", &hp),
            Some(vec![1024, 1024])
        );
        assert_eq!(expected_shape("enc.blocks.32.norm_out.bias", &hp), None);
        assert_eq!(expected_shape("enc.blocks.01.norm_out.bias", &hp), None);
        assert_eq!(expected_shape("dec.layer.8.norm1.bias", &hp), None);
        assert_eq!(expected_shape("enc.blocks.0.nope", &hp), None);
        assert_eq!(expected_shape("output.weight", &hp), None);
    }

    #[test]
    fn from_info_reads_the_phase1_keys() {
        let hp = CanaryHParams::from_info(&info(phase1_metadata())).unwrap();
        assert_eq!(hp, phase1_hparams());
        assert_eq!(hp.head_dim(), 128);
        assert_eq!(hp.special["lang.en"], 64);
    }

    #[test]
    fn from_info_names_a_missing_or_mistyped_key() {
        for key in [
            "stt.canary.encoder.conv_kernel",
            "stt.frontend.f_max",
            "stt.canary.tokenizer.single_sp",
            "general.languages",
            "stt.canary.special.startofcontext_id",
        ] {
            let mut m = phase1_metadata();
            m.remove(key).unwrap();
            let err = CanaryHParams::from_info(&info(m)).unwrap_err().to_string();
            assert!(err.contains(key), "{key}: {err}");
        }
        let mut m = phase1_metadata();
        m.insert("stt.frontend.f_min".into(), Value::String("0".into()));
        let err = CanaryHParams::from_info(&info(m)).unwrap_err().to_string();
        assert_eq!(err, "stt.frontend.f_min: missing or not an F32");
        let mut m = phase1_metadata();
        m.insert("stt.canary.special.pad_id".into(), Value::I32(-1));
        let err = CanaryHParams::from_info(&info(m)).unwrap_err().to_string();
        assert!(
            err.starts_with("stt.canary.special.pad_id: not a token id"),
            "{err}"
        );
    }

    #[test]
    fn from_info_rejects_a_special_id_outside_the_vocabulary() {
        for (key, id) in [
            ("stt.canary.special.startoftranscript_id", 16384),
            ("stt.canary.special.lang.en_id", 20000),
        ] {
            let mut m = phase1_metadata();
            m.insert(key.into(), Value::U32(id));
            let err = CanaryHParams::from_info(&info(m)).unwrap_err().to_string();
            assert_eq!(
                err,
                format!("{key}: {id} is not below stt.canary.decoder.vocab_size 16384")
            );
        }
    }

    #[test]
    fn from_info_rejects_another_architecture() {
        let mut m = phase1_metadata();
        m.insert("general.architecture".into(), Value::String("llama".into()));
        let err = CanaryHParams::from_info(&info(m)).unwrap_err().to_string();
        assert_eq!(
            err,
            "general.architecture is Some(\"llama\"), expected \"canary\""
        );
    }

    #[test]
    fn from_info_rejects_heads_that_do_not_divide_d_model() {
        for key in ["stt.canary.encoder.n_heads", "stt.canary.decoder.n_heads"] {
            let mut m = phase1_metadata();
            m.insert(key.into(), Value::U32(7));
            let err = CanaryHParams::from_info(&info(m)).unwrap_err().to_string();
            assert!(
                err.contains(&format!("{key} 7 does not divide d_model 1024")),
                "{err}"
            );
        }
    }

    /// `num_mels = 2^59 + 128` divides by the factor 8, and
    /// `channels * (num_mels / 8)` would wrap to the real file's 4096 in
    /// release; the bound rejects it before `expected_shape` multiplies.
    #[test]
    fn from_info_rejects_a_dimension_that_would_wrap_a_shape() {
        for (key, value) in [
            ("stt.frontend.num_mels", (1u64 << 59) + 128),
            ("stt.canary.encoder.d_model", 1u64 << 62),
            ("stt.canary.decoder.vocab_size", 0),
        ] {
            let mut m = phase1_metadata();
            m.insert(key.into(), Value::U64(value));
            let err = CanaryHParams::from_info(&info(m)).unwrap_err().to_string();
            assert_eq!(err, format!("{key}: implausible value {value}"));
        }
    }

    #[test]
    fn vocab_requires_vocab_size_pieces() {
        let hp = phase1_hparams();
        let pieces =
            |n: usize| Value::Array((0..n).map(|i| Value::String(format!("t{i}"))).collect());
        let mut m = phase1_metadata();
        m.insert("tokenizer.ggml.tokens".into(), pieces(16384));
        let vocab = Vocab::from_info(&info(m), &hp).unwrap();
        assert_eq!(vocab.pieces.len(), 16384);
        assert_eq!(vocab.pieces[3], "t3");
        let mut m = phase1_metadata();
        m.insert("tokenizer.ggml.tokens".into(), pieces(16000));
        let err = Vocab::from_info(&info(m), &hp).unwrap_err().to_string();
        assert_eq!(
            err,
            "tokenizer.ggml.tokens has 16000 pieces, stt.canary.decoder.vocab_size is 16384"
        );
        let err = Vocab::from_info(&info(phase1_metadata()), &hp)
            .unwrap_err()
            .to_string();
        assert_eq!(err, "tokenizer.ggml.tokens: missing or not a string array");
    }

    #[test]
    fn check_index_names_every_problem() {
        let hp = phase1_hparams();
        let mut idx = Index::default();
        for name in contract_names(32, 8, false) {
            idx.tensors
                .push(desc(&name, &expected_shape(&name, &hp).unwrap()));
        }
        check_index(&idx, &hp).unwrap();

        idx.tensors.retain(|t| t.name != "dec.layer.2.norm3.bias");
        idx.tensors.push(desc("enc.proj.weight", &[1024, 1024]));
        idx.tensors.push(desc("dec.head.bias", &[16384]));
        let pos = idx
            .tensors
            .iter()
            .position(|t| t.name == "enc.blocks.5.conv.depthwise.weight")
            .unwrap();
        idx.tensors[pos].ne = vec![31, 1, 1024];
        let err = check_index(&idx, &hp).unwrap_err().to_string();
        assert!(
            err.starts_with("canary tensor contract: 4 problems"),
            "{err}"
        );
        assert!(err.contains("missing: dec.layer.2.norm3.bias"), "{err}");
        assert!(err.contains("unmapped: enc.proj.weight"), "{err}");
        assert!(err.contains("duplicate: dec.head.bias"), "{err}");
        assert!(
            err.contains(
                "shape: enc.blocks.5.conv.depthwise.weight is [31, 1, 1024], expected [9, 1, 1024]"
            ),
            "{err}"
        );
    }

    #[test]
    fn tensor_class_follows_the_quantize_policy() {
        let class = TensorClass::of;
        for name in [
            "enc.pre_encode.conv.0.bias",
            "enc.blocks.0.norm_attn.weight",
            "enc.blocks.0.conv.bn.running_var",
            "enc.blocks.0.attn.pos_bias_u",
            "dec.layer.0.norm1.weight",
            "dec.embed.norm.weight",
            "dec.norm.weight",
            "dec.embed.pos_enc",
            "dec.head.bias",
        ] {
            assert_eq!(class(name), TensorClass::Vector, "{name}");
        }
        for name in [
            "enc.pre_encode.conv.0.weight",
            "enc.pre_encode.conv.3.weight",
            "enc.blocks.0.conv.depthwise.weight",
        ] {
            assert_eq!(class(name), TensorClass::Conv, "{name}");
        }
        for name in [
            "enc.pre_encode.out.weight",
            "enc.blocks.0.attn.linear_q.weight",
            "enc.blocks.0.conv.pointwise1.weight",
            "dec.layer.0.ffn.up.weight",
            "dec.embed.token.weight",
            "dec.head.weight",
        ] {
            assert_eq!(class(name), TensorClass::Linear, "{name}");
        }
    }

    #[test]
    fn check_types_accepts_the_pinned_mix() {
        let mut idx = Index::default();
        idx.tensors
            .push(typed("dec.head.weight", &[1024, 16384], ty::Q8_0));
        idx.tensors
            .push(typed("dec.layer.0.ffn.up.weight", &[1024, 4096], ty::Q6_K));
        idx.tensors
            .push(typed("enc.pre_encode.out.weight", &[4096, 1024], ty::BF16));
        idx.tensors.push(typed(
            "enc.blocks.0.conv.pointwise1.weight",
            &[1, 1024, 2048],
            ty::F16,
        ));
        idx.tensors.push(typed(
            "enc.pre_encode.conv.0.weight",
            &[3, 3, 1, 256],
            ty::F16,
        ));
        idx.tensors
            .push(typed("enc.blocks.0.norm_attn.bias", &[1024], ty::F32));
        check_types(&idx, &blck_size).unwrap();
    }

    #[test]
    fn check_types_names_every_problem() {
        let mut idx = Index::default();
        idx.tensors.push(typed(
            "enc.blocks.0.attn.linear_q.weight",
            &[1024, 1024],
            9999,
        ));
        idx.tensors
            .push(typed("enc.blocks.1.attn.linear_q.weight", &[1024, 1024], 4));
        idx.tensors
            .push(typed("dec.layer.0.ffn.up.weight", &[33, 4096], ty::Q8_0));
        idx.tensors
            .push(typed("enc.blocks.0.norm_attn.bias", &[1024], ty::Q8_0));
        idx.tensors
            .push(typed("dec.embed.pos_enc", &[1024, 1024], ty::BF16));
        idx.tensors.push(typed(
            "enc.blocks.0.conv.depthwise.weight",
            &[9, 1, 1024],
            ty::Q6_K,
        ));
        idx.tensors.push(typed(
            "enc.blocks.0.conv.pointwise1.weight",
            &[1, 1024, 2048],
            ty::Q8_0,
        ));
        let err = check_types(&idx, &blck_size).unwrap_err().to_string();
        assert!(err.starts_with("canary tensor types: 7 problems"), "{err}");
        for line in [
            "type: enc.blocks.0.attn.linear_q.weight is unknown (9999), a linear weight must be",
            "type: enc.blocks.1.attn.linear_q.weight is unknown (4), a linear weight must be",
            "type: dec.layer.0.ffn.up.weight is Q8_0 (8) with a row of 33, not a multiple of its block 32",
            "type: enc.blocks.0.norm_attn.bias is Q8_0 (8), a bias or norm must be F32",
            "type: dec.embed.pos_enc is BF16 (30), a bias or norm must be F32",
            "type: enc.blocks.0.conv.depthwise.weight is Q6_K (14), a conv kernel must be F32 or F16",
            "type: enc.blocks.0.conv.pointwise1.weight is Q8_0 (8) with a row of 1, not a multiple of its block 32",
        ] {
            assert!(err.contains(line), "{line}\n{err}");
        }
    }

    /// `build` over a recording lookup: dangling non-null handles stand in
    /// for ggml tensors, which `build` never dereferences.
    fn build_names(hp: &CanaryHParams) -> (CanaryTensors, Vec<String>) {
        let mut asked = Vec::new();
        let tensors = CanaryTensors::build(hp, &mut |name| {
            asked.push(name.to_string());
            Ok(std::ptr::dangling_mut())
        })
        .unwrap();
        (tensors, asked)
    }

    #[test]
    fn build_asks_for_the_contract_once_each() {
        let mut hp = phase1_hparams();
        for proj in [false, true] {
            hp.encoder_decoder_proj = proj;
            let (tensors, asked) = build_names(&hp);
            let unique: BTreeSet<&String> = asked.iter().collect();
            assert_eq!(unique.len(), asked.len(), "a name asked twice");
            let expected = contract_names(32, 8, proj);
            let asked: BTreeSet<String> = asked.into_iter().collect();
            assert_eq!(asked, expected);
            assert_eq!(tensors.count(), expected.len());
            assert_eq!(tensors.enc_blocks.len(), 32);
            assert_eq!(tensors.dec_layers.len(), 8);
            assert_eq!(tensors.enc_proj.is_some(), proj);
        }
    }

    #[test]
    fn build_surfaces_the_failing_name() {
        let hp = phase1_hparams();
        let err = CanaryTensors::build(&hp, &mut |name| {
            if name == "dec.layer.5.cross_attn.k.bias" {
                bail!("missing tensor {name:?}");
            }
            Ok(std::ptr::dangling_mut())
        })
        .map(|_| ())
        .unwrap_err()
        .to_string();
        assert_eq!(err, "missing tensor \"dec.layer.5.cross_attn.k.bias\"");
    }
}
