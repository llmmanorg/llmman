//! Qwen-Image 2.1 and 2.1 Turbo text-to-image (diffusers `QwenImage21Pipeline`)
//! from their Diffusers safetensors or a transformer GGUF (with a Qwen3-VL
//! GGUF and the Diffusers VAE): Qwen3-VL text encoder, single-stream
//! transformer, flow-matching Euler sampler, Wan-derived RGBA VAE.
//!
//! Prompt tokens are modulated from `t = 0` and attend causally, so their
//! keys and values are computed once per prompt and reused by every step.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;

use super::backend::{Backend, Graph, Resident};
use super::cosmos3::{self, dit as qwen3, read_json};
use super::ffi::{self, Api, Tensor};
use super::ltx::text::Vocab;
use super::qwen3_text;
use super::qwen3_vision;
use super::wan_vae;
use super::weights::{read_index, Index, LoadOpts, Weights};
use super::{dump, GenParams, Output};

pub const PIPELINE: &str = "QwenImage21Pipeline";
/// `QwenImage21Pipeline.__call__` defaults: no guidance, 1024x1024.
const DEFAULT_STEPS: i32 = 40;
/// Qwen-Image-2.1-Turbo's `sample_sigmas` (its `model_index.json`): 8 steps.
const TURBO_SIGMAS: [f32; 8] = [
    1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568,
];
/// `AutoencoderKLQwenImage21` `latents_mean` / `latents_std` (its `vae/config.json`),
/// for a VAE that comes without its config.
#[allow(clippy::approx_constant)]
const LATENTS_MEAN: [f32; 64] = [
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.651,
    0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391,
    -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342,
    -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.368,
    0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275, 4.0162,
    1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
];
const LATENTS_STD: [f32; 64] = [
    3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791, 3.0785, 3.65, 3.9308, 3.0904,
    2.8778, 3.7675, 3.732, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
    3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467, 3.9037, 3.2947, 5.077, 3.5075,
    3.27, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
    3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.408, 3.4959, 3.8747, 3.7608, 3.5735, 3.149,
    3.7662, 3.6746, 3.4563, 3.8161,
];
/// The transformer GGUF's tensor name prefix (ComfyUI's layout).
const GGUF_PREFIX: &str = "model.diffusion_model.";
pub const DEFAULT_SIZE: i64 = 1024;
const SYSTEM: &str = "Comprehend and analyze the provided prompt.";
/// The placeholder each reference image's vision tokens replace.
const IMAGE_PAD: &str = "<|image_pad|>";
/// Pixels per latent token: 16x VAE, no patching.
const SCALE: i64 = 16;
const EPS: f32 = 1e-6;
/// Queries per chunk of dense attention (the prefix's, and the steps' without flash attention).
const ATTN_CHUNK: i64 = 1024;
/// Graph tensors one chunk of attention takes per layer, with room to spare.
const ATTN_CHUNK_NODES: usize = 32;
/// Room left for compute buffers when deciding whether both models fit.
const HEADROOM: u64 = 4 << 30;

/// `QwenImage21Transformer2DModel` config.
struct Hparams {
    layers: i64,
    heads: i64,
    head_dim: i64,
    in_channels: i64,
    /// Rotary dims of the frame, height and width axes.
    axes: [i64; 3],
}

/// `FlowMatchEulerDiscreteScheduler` with dynamic exponential shifting, or
/// a fixed shift (Turbo), and the pipeline's own `sample_sigmas` if any.
struct Sched {
    base_seq_len: f64,
    max_seq_len: f64,
    base_shift: f64,
    max_shift: f64,
    shift_terminal: f64,
    /// `use_dynamic_shifting`; `shift` applies otherwise.
    dynamic: bool,
    shift: f64,
    /// The pipeline's `sample_sigmas`, used for that many steps.
    fixed: Vec<f32>,
}

impl Sched {
    fn from_config(v: &Value) -> Sched {
        let f = |k: &str, d: f64| v.get(k).and_then(Value::as_f64).unwrap_or(d);
        Sched {
            base_seq_len: f("base_image_seq_len", 256.0),
            max_seq_len: f("max_image_seq_len", 8192.0),
            base_shift: f("base_shift", 0.5),
            max_shift: f("max_shift", 0.9),
            // `null` (Turbo) is none
            shift_terminal: match v.get("shift_terminal") {
                Some(Value::Null) => 0.0,
                _ => f("shift_terminal", 0.02),
            },
            dynamic: v
                .get("use_dynamic_shifting")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            shift: f("shift", 1.0),
            fixed: Vec::new(),
        }
    }

    /// Qwen-Image-2.1-Turbo's scheduler config and `sample_sigmas`.
    fn turbo() -> Sched {
        Sched {
            shift_terminal: 0.0,
            dynamic: false,
            fixed: TURBO_SIGMAS.to_vec(),
            ..Sched::from_config(&Value::Null)
        }
    }

    /// `sample_sigmas` when they are `n`, else the pipeline's `linspace(1, 1/n, n)`,
    /// shifted by `calculate_shift(seq_len)` (or the fixed `shift`) and stretched
    /// to end at `shift_terminal`, as `set_timesteps` does with either; then 0.
    fn sigmas(&self, n: usize, seq_len: i64) -> Vec<f32> {
        let m = (self.max_shift - self.base_shift) / (self.max_seq_len - self.base_seq_len);
        let mu = if self.dynamic {
            (seq_len as f64 * m + self.base_shift - m * self.base_seq_len).exp()
        } else {
            self.shift
        };
        let mut s: Vec<f64> = (0..n)
            .map(|i| {
                let t = if n == self.fixed.len() {
                    self.fixed[i] as f64
                } else if n > 1 {
                    1.0 - i as f64 * (1.0 - 1.0 / n as f64) / (n - 1) as f64
                } else {
                    1.0
                };
                mu / (mu + (1.0 / t - 1.0))
            })
            .collect();
        // one step leaves nothing to stretch (diffusers divides by zero there)
        if self.shift_terminal > 0.0 && n > 1 {
            let scale = (1.0 - s[n - 1]) / (1.0 - self.shift_terminal);
            for v in &mut s {
                *v = 1.0 - (1.0 - *v) / scale;
            }
        }
        s.into_iter().map(|v| v as f32).chain([0.0]).collect()
    }
}

/// The prompt's per-layer keys and values, `[hidden, n]` each, on the device.
struct Prefix {
    /// The output image's frame position, after the prompt and references.
    frame: f64,
    kv: Resident,
}

/// The text encoder's hidden states `[T][hidden]` of a prompt, and where each
/// reference image's vision tokens are in them (`(start, len)`).
type EncodedPrompt = (Vec<f32>, Vec<(usize, usize)>);

/// A reference image ready for the prompt: its VAE latents as transformer
/// tokens `[lh * lw][in_channels]`, and its vision tokens.
struct RefImage {
    tokens: Vec<f32>,
    lh: i64,
    lw: i64,
    vision: qwen3_vision::Encoded,
}

/// A run of prefix tokens: prompt tokens `start..start + len` of the text
/// encoder's output, or reference image `index`'s latents.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Seg {
    Text { start: usize, len: usize, at: usize },
    Image { index: usize, len: usize, at: usize },
}

/// `QwenImage21Layout`: the prefix's runs, every token's `(frame, h, w)`
/// position, and the output image's frame position.
struct Layout {
    segments: Vec<Seg>,
    positions: Vec<[f64; 3]>,
    next: f64,
}

/// Queries `start..start + len` of the prefix and the keys `0..keys` they
/// see; a text run's chunk sees them causally under `mask` (`[keys, len]`).
#[derive(Debug, PartialEq)]
struct AttnChunk {
    start: i64,
    len: i64,
    keys: i64,
    mask: Option<Vec<f32>>,
}

impl Layout {
    /// The prefix's attention in chunks of at most [`ATTN_CHUNK`] queries. A
    /// query sees the keys up to the end of its run, causally within a text
    /// run: always the first keys, so only text chunks need a mask, and no
    /// chunk scores keys it cannot see.
    fn attn_chunks(&self) -> Vec<AttnChunk> {
        let mut out = Vec::new();
        for seg in &self.segments {
            let (at, end, text) = match *seg {
                Seg::Text { at, len, .. } => (at as i64, (at + len) as i64, true),
                Seg::Image { at, len, .. } => (at as i64, (at + len) as i64, false),
            };
            for (start, len) in qwen3::query_chunks(at, end, ATTN_CHUNK) {
                let keys = if text { start + len } else { end };
                let mask = text.then(|| {
                    let mut m = vec![0f32; (keys * len) as usize];
                    for (i, row) in m.chunks_mut(keys as usize).enumerate() {
                        row[start as usize + i + 1..].fill(f32::NEG_INFINITY);
                    }
                    m
                });
                out.push(AttnChunk {
                    start,
                    len,
                    keys,
                    mask,
                });
            }
        }
        out
    }
}

/// The prefix of `n_text` prompt tokens whose `slots` (`(start, len)`) hold
/// `images`' vision tokens: each slot's tokens give way to its latents.
fn prefix_layout(n_text: usize, slots: &[(usize, usize)], images: &[RefImage]) -> Result<Layout> {
    let grids: Vec<(i64, i64)> = images.iter().map(|r| (r.lh, r.lw)).collect();
    prefix_layout_grids(n_text, slots, &grids)
}

fn prefix_layout_grids(
    n_text: usize,
    slots: &[(usize, usize)],
    grids: &[(i64, i64)],
) -> Result<Layout> {
    if slots.len() != grids.len() {
        bail!("{} image slots for {} images", slots.len(), grids.len());
    }
    let mut out = Layout {
        segments: Vec::new(),
        positions: Vec::new(),
        next: 0.0,
    };
    let mut pos = 0f64;
    let text = |out: &mut Layout, start: usize, end: usize, pos: &mut f64| {
        if end > start {
            out.segments.push(Seg::Text {
                start,
                len: end - start,
                at: out.positions.len(),
            });
            for _ in start..end {
                out.positions.push([*pos; 3]);
                *pos += 1.0;
            }
        }
    };
    let mut cur = 0;
    for (index, (&(start, len), &(lh, lw))) in slots.iter().zip(grids).enumerate() {
        if start < cur || start + len > n_text || (len as i64) * 4 != lh * lw {
            bail!("vision slots and reference latents must have matching sizes");
        }
        text(&mut out, cur, start, &mut pos);
        out.segments.push(Seg::Image {
            index,
            len: (lh * lw) as usize,
            at: out.positions.len(),
        });
        for p in image_positions(pos, lh, lw) {
            out.positions.push(p);
        }
        pos += lh.max(lw) as f64;
        cur = start + len;
    }
    text(&mut out, cur, n_text, &mut pos);
    out.next = pos;
    Ok(out)
}

/// Text encoder tokens kept for the chat template and the prompt beside the
/// references' vision tokens.
const PROMPT_RESERVE: i64 = 256;

/// Refuses an edit whose references, each a vision token per 32x32 pixels at
/// the output's area, overflow the text encoder's prompt, before any model
/// loads.
pub fn check_refs(p: &GenParams) -> Result<()> {
    let (w, h) = (p.width / 32 * 32, p.height / 32 * 32);
    let mut tokens = 0;
    for f in &p.ref_images {
        super::flux2::check_reference(f)?;
        let (rw, rh) = ref_size(f.width, f.height, (w * h).max(1));
        tokens += (rw / 32) * (rh / 32);
    }
    let max = qwen3::MAX_PROMPT_TOKENS - PROMPT_RESERVE;
    if tokens > max {
        bail!(
            "the reference images make {tokens} vision tokens at {w}x{h}, at most {max}; \
             use a smaller size or fewer references"
        );
    }
    Ok(())
}

/// A reference's size: the output's area `pixels`, aspect kept, sides
/// rounded to multiples of 32.
fn ref_size(w: i64, h: i64, pixels: i64) -> (i64, i64) {
    let side = (pixels as f64 * w as f64 / h as f64).sqrt();
    let r = |v: f64| ((v / 32.0).round() as i64 * 32).max(32);
    (r(side), r(side * h as f64 / w as f64))
}

/// The text encoder's tokenizer: the Diffusers pack's `tokenizer.json`, or
/// the vocabulary of the text encoder GGUF.
enum Tok {
    Hf(Box<tokenizers::Tokenizer>),
    Gguf(Vocab),
}

impl Tok {
    /// Token ids of `text`, special tokens parsed, no BOS.
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            Tok::Hf(t) => encode_ids(t, text),
            Tok::Gguf(t) => Ok(t
                .tokenize(text, false, true)?
                .into_iter()
                .map(|i| i as u32)
                .collect()),
        }
    }
}

type Rename = fn(&str) -> Option<String>;

pub struct Model {
    api: &'static Api,
    hp: Hparams,
    sched: Sched,
    default_steps: i32,
    te_hp: cosmos3::Hparams,
    tok: Tok,
    /// Tokens of the system turn the pipeline drops from the hidden states.
    drop_idx: usize,
    dit_files: Vec<PathBuf>,
    dit_rename: Option<Rename>,
    te_files: Vec<PathBuf>,
    te_rename: Rename,
    dit_bytes: u64,
    te_bytes: u64,
    /// Loaded on demand: they may not fit next to each other.
    dit: Option<Weights>,
    te: Option<Weights>,
    vae: Weights,
    layout: wan_vae::Layout,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
    flash: bool,
    /// The text encoder's vision tower (an mmproj GGUF), for edits; loaded on demand.
    vision_path: Option<PathBuf>,
    vision: Option<qwen3_vision::Vision>,
    cond: Option<(String, Prefix)>,
    ncond: Option<(String, Prefix)>,
}

/// Where the VAE comes from and how its latents are laid out.
struct Vae {
    path: PathBuf,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
    layout: wan_vae::Layout,
}

/// What a Diffusers pack or a GGUF set gives [`Model::assemble`].
struct Parts {
    hp: Hparams,
    sched: Sched,
    te_hp: cosmos3::Hparams,
    tok: Tok,
    dit_files: Vec<PathBuf>,
    dit_rename: Option<Rename>,
    te_files: Vec<PathBuf>,
    te_rename: Rename,
    vae: Vae,
    vision_path: Option<PathBuf>,
    flash: bool,
}

/// Is `path` a Diffusers pipeline index of a pipeline this module runs?
pub fn is_pipeline_index(path: &Path) -> bool {
    pipeline_class(path).as_deref() == Some(PIPELINE)
}

fn pipeline_class(path: &Path) -> Option<String> {
    if path.file_name()? != crate::modelpack::DIFFUSERS_MODEL_INDEX {
        return None;
    }
    let v = read_json(path).ok()?;
    Some(v.get("_class_name")?.as_str()?.to_string())
}

/// The `.safetensors` files of one pipeline component, in shard order.
fn shards(files: &BTreeMap<String, PathBuf>, dir: &str) -> Result<Vec<PathBuf>> {
    let v: Vec<PathBuf> = files
        .iter()
        .filter(|(k, _)| {
            k.strip_prefix(dir)
                .and_then(|r| r.strip_prefix('/'))
                .is_some_and(|r| !r.contains('/') && r.ends_with(".safetensors"))
        })
        .map(|(_, p)| p.clone())
        .collect();
    if v.is_empty() {
        bail!("the model pack has no {dir}/*.safetensors");
    }
    Ok(v)
}

fn file<'a>(files: &'a BTreeMap<String, PathBuf>, name: &str) -> Result<&'a PathBuf> {
    files
        .get(name)
        .ok_or_else(|| anyhow!("the model pack has no {name}"))
}

/// Qwen3-VL's language model under the names `cosmos3::dit` reads; its vision
/// tower, final norm and LM head are never loaded.
fn te_name(n: &str) -> Option<String> {
    let n = n.strip_prefix("model.language_model.")?;
    if n == "norm.weight" {
        return None;
    }
    let mut n = n.to_string();
    for (from, to) in [
        ("q_proj", "to_q"),
        ("k_proj", "to_k"),
        ("v_proj", "to_v"),
        ("o_proj", "to_out"),
        ("q_norm", "norm_q"),
        ("k_norm", "norm_k"),
    ] {
        n = n.replace(&format!(".self_attn.{from}."), &format!(".self_attn.{to}."));
    }
    Some(n)
}

/// The GGUF transformer's tensors under the Diffusers names.
fn gguf_dit_name(n: &str) -> Option<String> {
    Some(n.strip_prefix(GGUF_PREFIX).unwrap_or(n).to_string())
}

/// Is this GGUF a Qwen-Image 2.1 transformer? By its tensors: `qwen_image`
/// also names the earlier Qwen-Image models, which this module does not
/// run, and some files (unsloth's base model) carry no architecture at all.
pub fn is_qwen_image21(idx: &Index) -> bool {
    matches!(
        idx.metadata.get("general.architecture").map(String::as_str),
        None | Some("qwen_image")
    ) && ["txt_in.text_norm.weight", "modulation.1.weight"]
        .iter()
        .all(|n| {
            idx.tensors
                .iter()
                .any(|d| gguf_dit_name(&d.name).as_deref() == Some(n))
        })
}

/// The VAE's decoder, and its encoder when `edits` can use it.
fn vae_name(edits: bool) -> impl Fn(&str) -> Option<String> {
    move |n| {
        let decoder = ["decoder.", "post_quant_conv."];
        let encoder = ["encoder.", "quant_conv."];
        (decoder.iter().any(|p| n.starts_with(p))
            || edits && encoder.iter().any(|p| n.starts_with(p)))
        .then(|| n.to_string())
    }
}

/// Bytes the tensors `rename` keeps take in `files`. A GGUF index has no
/// sizes: its tensors are packed, each ends where the next one starts.
fn weight_bytes(files: &[PathBuf], rename: Rename) -> Result<u64> {
    let mut total = 0;
    for f in files {
        let idx = read_index(f)?;
        let mut ends: Vec<u64> = idx.tensors.iter().map(|d| d.offset).collect();
        ends.push(std::fs::metadata(f)?.len());
        ends.sort_unstable();
        for d in idx.tensors.iter().filter(|d| rename(&d.name).is_some()) {
            total += match d.nbytes {
                0 => match ends.get(ends.partition_point(|&e| e <= d.offset)) {
                    Some(end) => end - d.offset,
                    None => bail!("{}: tensor {} starts past the end", f.display(), d.name),
                },
                n => n,
            };
        }
    }
    Ok(total)
}

/// Do `bytes` more weights fit, with room for the compute buffers?
fn fits(be: &Backend, bytes: u64) -> bool {
    be.weight_memory().0 as u64 >= bytes + HEADROOM
}

impl Model {
    /// From a Diffusers pack (`files`: every file by its path in the pack).
    pub fn load(
        api: &'static Api,
        be: &Backend,
        files: &BTreeMap<String, PathBuf>,
        flash: bool,
    ) -> Result<Model> {
        let t0 = Instant::now();
        let tcfg = read_json(file(files, "transformer/config.json")?)?;
        let i = |k: &str, d: i64| tcfg.get(k).and_then(Value::as_i64).unwrap_or(d);
        let axes: Vec<i64> = tcfg
            .get("axes_dims_rope")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_i64).collect())
            .unwrap_or_else(|| vec![16, 56, 56]);
        let head_dim = i("attention_head_dim", 128);
        let [a0, a1, a2] = axes[..] else {
            bail!("unsupported axes_dims_rope {axes:?}");
        };
        if a0 + a1 + a2 != head_dim || i("patch_size", 1) != 1 {
            bail!("unsupported transformer config {tcfg}");
        }
        let hp = Hparams {
            layers: i("num_layers", 32),
            heads: i("num_attention_heads", 32),
            head_dim,
            in_channels: i("in_channels", 64),
            axes: [a0, a1, a2],
        };
        if !tcfg
            .get("causal_condition")
            .and_then(Value::as_bool)
            .unwrap_or(true)
        {
            bail!("only causal_condition transformers are supported");
        }
        let mut sched = files
            .get("scheduler/scheduler_config.json")
            .map(|p| read_json(p))
            .transpose()?
            .map_or_else(
                || Sched::from_config(&Value::Null),
                |v| Sched::from_config(&v),
            );
        // Turbo's own schedule
        if let Some(sigmas) = files
            .get(crate::modelpack::DIFFUSERS_MODEL_INDEX)
            .map(|p| read_json(p))
            .transpose()?
            .and_then(|v| v.get("sample_sigmas").and_then(Value::as_array).cloned())
        {
            sched.fixed = sigmas
                .iter()
                .filter_map(|x| x.as_f64().map(|x| x as f32))
                .collect();
        }
        let te_cfg = read_json(file(files, "text_encoder/config.json")?)?;
        let te_hp = cosmos3::Hparams::from_config(te_cfg.get("text_config").unwrap_or(&te_cfg))?;
        let tok_path = file(files, "processor/tokenizer.json")
            .or_else(|_| file(files, "tokenizer/tokenizer.json"))?;
        let tok = Tok::Hf(Box::new(
            tokenizers::Tokenizer::from_file(tok_path)
                .map_err(|e| anyhow!("{}: {e}", tok_path.display()))?,
        ));

        let vcfg = read_json(file(files, "vae/config.json")?)?;
        let floats = |k: &str| -> Vec<f32> {
            vcfg.get(k)
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_f64().map(|x| x as f32))
                        .collect()
                })
                .unwrap_or_default()
        };
        let mut temporal_up: Vec<bool> = vcfg
            .get("temperal_downsample")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_bool).collect())
            .unwrap_or_default();
        temporal_up.reverse();
        let vae = Vae {
            path: file(files, "vae/diffusion_pytorch_model.safetensors")?.clone(),
            latents_mean: floats("latents_mean"),
            latents_std: floats("latents_std"),
            layout: wan_vae::Layout {
                temporal_up,
                patch: vcfg.get("patch_size").and_then(Value::as_i64).unwrap_or(1) > 1,
            },
        };
        Self::assemble(
            api,
            be,
            t0,
            Parts {
                hp,
                sched,
                te_hp,
                tok,
                dit_files: shards(files, "transformer")?,
                dit_rename: None,
                te_files: shards(files, "text_encoder")?,
                te_rename: te_name,
                vae,
                vision_path: None,
                flash,
            },
        )
    }

    /// From a transformer GGUF (`model`), the Diffusers VAE and a Qwen3-VL GGUF
    /// (and its vision tower for edits); a `turbo` file name selects the Turbo
    /// schedule.
    #[allow(clippy::too_many_arguments)]
    pub fn load_gguf(
        api: &'static Api,
        be: &Backend,
        idx: &Index,
        model: &Path,
        vae: Option<&Path>,
        text: Option<&Path>,
        vision: Option<&Path>,
        flash: bool,
    ) -> Result<Model> {
        let t0 = Instant::now();
        let idx = idx.renamed(gguf_dit_name);
        let ne = |n: &str, i: usize| idx.dim(n, i).with_context(|| model.display().to_string());
        let head_dim = ne("transformer_blocks.0.attn.norm_q.weight", 0)?;
        let img_in = [ne("img_in.weight", 0)?, ne("img_in.weight", 1)?];
        let layers = idx.count("transformer_blocks.");
        let hp = Hparams {
            layers,
            heads: img_in[1] / head_dim,
            head_dim,
            in_channels: img_in[0],
            axes: [16, 56, 56],
        };
        if hp.axes.iter().sum::<i64>() != head_dim {
            bail!("{}: unsupported head dim {head_dim}", model.display());
        }
        let turbo = model
            .file_name()
            .is_some_and(|n| n.to_string_lossy().to_lowercase().contains("turbo"));
        let sched = if turbo {
            Sched::turbo()
        } else {
            Sched::from_config(&Value::Null)
        };
        let text = text.ok_or_else(|| anyhow!("Qwen-Image 2.1 needs its Qwen3-VL text encoder"))?;
        let te_idx = read_index(text)?;
        let te_hp = qwen3_text::hparams(&te_idx, text)?;
        let tok = Tok::Gguf(Vocab::load(api, &text.to_string_lossy())?);
        let vae = Vae {
            path: vae
                .ok_or_else(|| anyhow!("Qwen-Image 2.1 needs its VAE"))?
                .to_path_buf(),
            latents_mean: LATENTS_MEAN.to_vec(),
            latents_std: LATENTS_STD.to_vec(),
            // `temperal_downsample` [false, true, true, true], reversed; no patching
            layout: wan_vae::Layout {
                temporal_up: vec![true, true, true, false],
                patch: false,
            },
        };
        Self::assemble(
            api,
            be,
            t0,
            Parts {
                hp,
                sched,
                te_hp,
                tok,
                dit_files: vec![model.to_path_buf()],
                dit_rename: Some(gguf_dit_name),
                te_files: vec![text.to_path_buf()],
                te_rename: qwen3_text::tensor_name,
                vae,
                vision_path: vision.map(Path::to_path_buf),
                flash,
            },
        )
    }

    /// Loads the VAE and sizes the transformer and text encoder, loaded on first use.
    fn assemble(api: &'static Api, be: &Backend, t0: Instant, p: Parts) -> Result<Model> {
        let nch = p.hp.in_channels as usize;
        let v = &p.vae;
        if v.latents_mean.len() != nch
            || v.latents_std.len() != nch
            || v.latents_std.iter().any(|&s| s <= 0.0)
        {
            bail!("the VAE's latents_mean/latents_std must each hold {nch} positive-std entries");
        }
        let vae = Weights::load(
            api,
            &v.path,
            be.weight_buft(),
            &LoadOpts {
                rename: Some(Box::new(vae_name(p.vision_path.is_some()))),
                transform: None,
            },
        )?;
        if !vae.has("decoder.conv_in.weight") {
            bail!("the VAE has no decoder");
        }
        let drop_idx = p
            .tok
            .encode(&format!("<|im_start|>system\n{SYSTEM}<|im_end|>\n"))?
            .len();
        let dit_bytes = weight_bytes(
            &p.dit_files,
            p.dit_rename.unwrap_or(|n| Some(n.to_string())),
        )?;
        let te_bytes = weight_bytes(&p.te_files, p.te_rename)?;
        let default_steps = if p.sched.fixed.is_empty() {
            DEFAULT_STEPS
        } else {
            p.sched.fixed.len() as i32
        };
        eprintln!(
            "[llmman] mediagen: qwen-image model ready in {:.1} s ({} layers, {default_steps} steps, transformer {:.1} GiB, text encoder {:.1} GiB, loaded on first use)",
            t0.elapsed().as_secs_f64(),
            p.hp.layers,
            dit_bytes as f64 / (1u64 << 30) as f64,
            te_bytes as f64 / (1u64 << 30) as f64,
        );
        Ok(Model {
            api,
            hp: p.hp,
            sched: p.sched,
            default_steps,
            te_hp: p.te_hp,
            tok: p.tok,
            drop_idx,
            dit_files: p.dit_files,
            dit_rename: p.dit_rename,
            te_files: p.te_files,
            te_rename: p.te_rename,
            dit_bytes,
            te_bytes,
            dit: None,
            te: None,
            vae,
            layout: p.vae.layout,
            latents_mean: p.vae.latents_mean,
            latents_std: p.vae.latents_std,
            flash: p.flash,
            vision_path: p.vision_path,
            vision: None,
            cond: None,
            ncond: None,
        })
    }

    pub fn default_steps(&self) -> i32 {
        self.default_steps
    }

    /// Edits take reference images: the text encoder has its vision tower and
    /// the VAE its encoder.
    pub fn supports_refs(&self) -> bool {
        self.vision_path.is_some()
            && self.vae.has("encoder.conv_in.weight")
            && self.vae.has("quant_conv.weight")
    }

    fn load_files(
        &self,
        be: &Backend,
        files: &[PathBuf],
        rename: Option<Rename>,
    ) -> Result<Weights> {
        let paths: Vec<&Path> = files.iter().map(PathBuf::as_path).collect();
        Weights::load_files(
            self.api,
            &paths,
            be.weight_buft(),
            &LoadOpts {
                rename: rename.map(|f| Box::new(f) as _),
                transform: None,
            },
        )
    }

    /// The transformer, loaded when it is not; the text encoder makes room for it.
    fn ensure_dit(&mut self, be: &mut Backend) -> Result<()> {
        if self.dit.is_none() {
            if self.te.is_some() && !fits(be, self.dit_bytes) {
                self.te = None;
            }
            self.dit = Some(self.load_files(be, &self.dit_files, self.dit_rename)?);
        }
        Ok(())
    }

    /// The prompt caches of `prompts` (positive, then negative) with the
    /// reference images `refs` (`key` names them), computing the missing ones.
    fn prepare(
        &mut self,
        be: &mut Backend,
        prompts: &[&str],
        refs: &[super::Frames],
        key: &str,
        (w, h): (i64, i64),
    ) -> Result<()> {
        // references are sized to the output's area: an edit's prefix is per size
        let size = if refs.is_empty() {
            String::new()
        } else {
            format!("\0{w}x{h}")
        };
        let keys: Vec<String> = prompts
            .iter()
            .map(|p| format!("{p}\0{key}{size}"))
            .collect();
        let missing: Vec<usize> = (0..prompts.len())
            .filter(|&i| {
                let slot = if i == 0 { &self.cond } else { &self.ncond };
                !matches!(slot, Some((k, _)) if *k == keys[i])
            })
            .collect();
        if missing.is_empty() {
            return self.ensure_dit(be);
        }
        if !refs.is_empty() && !self.supports_refs() {
            bail!(
                "Qwen-Image 2.1 edits need the Qwen3-VL vision tower (an mmproj GGUF) and a VAE \
                 with its encoder"
            );
        }
        // a stale prefix gives its memory up before its replacement is built,
        // and so does a negative one this request does not use
        for &i in &missing {
            *(if i == 0 {
                &mut self.cond
            } else {
                &mut self.ncond
            }) = None;
        }
        if prompts.len() < 2 {
            self.ncond = None;
        }
        if self.te.is_none() {
            if self.dit.is_some() && !fits(be, self.te_bytes) {
                self.dit = None;
                be.release_compute()?;
            }
            self.te = Some(self.load_files(be, &self.te_files, Some(self.te_rename))?);
        }
        let mut images = Vec::new();
        if !refs.is_empty() {
            if self.vision.is_none() {
                let path = self.vision_path.as_ref().expect("checked above");
                if self.dit.is_some() && !fits(be, std::fs::metadata(path)?.len()) {
                    self.dit = None;
                    be.release_compute()?;
                }
                self.vision = Some(qwen3_vision::Vision::load(self.api, be, path, self.flash)?);
            }
            let encoded = refs.iter().try_for_each(|f| {
                images.push(self.reference(be, f, w, h)?);
                be.release_compute()
            });
            // the tower is only for the references: freed even when one fails
            self.vision = None;
            encoded?;
        }
        let mut hidden = Vec::new();
        for &i in &missing {
            let t0 = Instant::now();
            let (hd, slots) = self.encode(be, prompts[i], &images)?;
            dump(
                if i == 0 {
                    "text_hidden"
                } else {
                    "ntext_hidden"
                },
                &hd,
            );
            eprintln!(
                "[llmman] mediagen: encoded {} prompt ({} tokens, {} images) in {:.2} s",
                if i == 0 { "positive" } else { "negative" },
                hd.len() / self.te_hp.hidden as usize,
                images.len(),
                t0.elapsed().as_secs_f64()
            );
            hidden.push((hd, slots));
        }
        let ref_tokens: i64 = images.iter().map(|r| r.lh * r.lw).sum();
        let cache_bytes: u64 = hidden
            .iter()
            .map(|(hd, _)| {
                let n = hd.len() as i64 / self.te_hp.hidden + ref_tokens;
                (2 * self.hp.layers * self.hp.heads * self.hp.head_dim * n * 2) as u64
            })
            .sum();
        let dit_bytes = if self.dit.is_some() {
            0
        } else {
            self.dit_bytes
        };
        // with reference images the prefix graphs are large too: the free
        // memory a backend reports does not cover them (Metal ran out)
        if !images.is_empty() || !fits(be, dit_bytes + cache_bytes) {
            self.te = None;
        }
        be.release_compute()?;
        self.ensure_dit(be)?;
        for (&i, (hd, slots)) in missing.iter().zip(&hidden) {
            let prefix = self.prefix(be, hd, slots, &images)?;
            *(if i == 0 {
                &mut self.cond
            } else {
                &mut self.ncond
            }) = Some((keys[i].clone(), prefix));
        }
        Ok(())
    }

    /// A reference image fitted to the output's area (sides multiples of 32,
    /// aspect kept, stretched like stable-diffusion.cpp's server does), as
    /// VAE latent tokens and vision tokens.
    fn reference(&self, be: &Backend, f: &super::Frames, w: i64, h: i64) -> Result<RefImage> {
        let t0 = Instant::now();
        super::flux2::check_reference(f)?;
        let (rw, rh) = ref_size(f.width, f.height, w * h);
        // as sd-server: [0, 1] floats stretched to the size, then [-1, 1]
        let rgb: Vec<f32> = super::encode::resize_bilinear(
            |i| f.rgb[i] as f32 / 255.0,
            f.width as usize,
            f.height as usize,
            3,
            rw as usize,
            rh as usize,
        )
        .into_iter()
        .map(|v| v * 2.0 - 1.0)
        .collect();
        // opaque RGBA for the VAE
        let rgba: Vec<f32> = rgb
            .chunks(3)
            .flat_map(|p| [p[0], p[1], p[2], 1.0])
            .collect();
        let z = wan_vae::encode(&self.vae, be, &rgba, 1, rh, rw, 4, &self.layout)?;
        let (lh, lw) = (rh / SCALE, rw / SCALE);
        let n = (lh * lw) as usize;
        let mut z = z.into_iter().next().expect("one latent frame");
        if z.len() != n * self.latents_mean.len() {
            bail!(
                "the VAE encoded {} values, expected {} channels x {n} tokens",
                z.len(),
                self.latents_mean.len()
            );
        }
        for (ci, row) in z.chunks_mut(n).enumerate() {
            let (m, s) = (self.latents_mean[ci], self.latents_std[ci]);
            row.iter_mut().for_each(|v| *v = (*v - m) / s);
        }
        let vision = self
            .vision
            .as_ref()
            .expect("vision tower loaded")
            .encode(be, &rgb, rh, rw)?;
        dump("ref_latents", &z);
        dump("ref_vision", &vision.embeds);
        eprintln!(
            "[llmman] mediagen: encoded a {}x{} reference as {rw}x{rh} in {:.2} s",
            f.width,
            f.height,
            t0.elapsed().as_secs_f64()
        );
        Ok(RefImage {
            tokens: transpose(&z, self.hp.in_channels as usize, n),
            lh,
            lw,
            vision,
        })
    }

    /// `_get_qwen_prompt_embeds`: the text model's last hidden states `[T][hidden]`
    /// over the chat template, the system turn dropped, and where each image's
    /// tokens are in them (`(start, len)`).
    fn encode(&self, be: &Backend, prompt: &str, images: &[RefImage]) -> Result<EncodedPrompt> {
        let te = self.te.as_ref().expect("text encoder loaded");
        let mut user = String::new();
        for i in 0..images.len() {
            if i > 0 {
                user.push(' ');
            }
            user += &format!("<image{}><|vision_start|>{IMAGE_PAD}<|vision_end|>", i + 1);
        }
        let text = format!(
            "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}{prompt}<|im_end|>\n<|im_start|>assistant\n"
        );
        let marked = self.tok.encode(&text)?;
        if images.is_empty() {
            let h = qwen3::text_hidden(&self.te_hp, te, be, &marked, None)?;
            return Ok((
                h[self.drop_idx * self.te_hp.hidden as usize..].to_vec(),
                Vec::new(),
            ));
        }
        let pad = match self.tok.encode(IMAGE_PAD)?[..] {
            [id] => id,
            _ => bail!("{IMAGE_PAD} is not one token"),
        };
        // each pad expands to the image's tokens
        let dim = self.te_hp.hidden as usize;
        let (mut ids, mut slots, mut positions) = (Vec::new(), Vec::new(), Vec::new());
        let mut next = images.iter();
        let mut pos = 0f64;
        for &id in &marked {
            if id != pad {
                ids.push(id);
                positions.push([pos; 3]);
                pos += 1.0;
                continue;
            }
            let img = next
                .next()
                .ok_or_else(|| anyhow!("more image markers than images"))?;
            let (gh, gw) = (img.vision.grid_h, img.vision.grid_w);
            slots.push((ids.len(), (gh * gw) as usize));
            for r in 0..gh {
                for c in 0..gw {
                    ids.push(pad);
                    positions.push([pos, pos + r as f64, pos + c as f64]);
                }
            }
            pos += gh.max(gw) as f64;
        }
        let n = ids.len();
        qwen3::check_prompt_len(n)?;
        let mut v = qwen3::VisionInputs {
            embeds: vec![0f32; n * dim],
            keep: vec![1f32; n],
            deepstack: vec![vec![0f32; n * dim]; images[0].vision.deepstack.len()],
            positions,
        };
        for (&(start, len), img) in slots.iter().zip(images) {
            let e = &img.vision;
            if e.embeds.len() != len * dim
                || e.deepstack.len() != v.deepstack.len()
                || e.deepstack.iter().any(|d| d.len() != len * dim)
            {
                bail!(
                    "the vision tower's output ({} values for {len} tokens) does not fit the \
                     text encoder's width {dim}",
                    e.embeds.len()
                );
            }
            v.embeds[start * dim..(start + len) * dim].copy_from_slice(&img.vision.embeds);
            v.keep[start..start + len].fill(0.0);
            for (d, src) in v.deepstack.iter_mut().zip(&img.vision.deepstack) {
                d[start * dim..(start + len) * dim].copy_from_slice(src);
            }
        }
        let h = qwen3::text_hidden(&self.te_hp, te, be, &ids, Some(&v))?;
        let slots = slots
            .into_iter()
            .map(|(s, l)| (s - self.drop_idx, l))
            .collect();
        Ok((h[self.drop_idx * dim..].to_vec(), slots))
    }

    /// The prompt tokens, each reference image's latents in place of its vision
    /// tokens, through every block (`t = 0` modulation; text attends causally,
    /// images to everything up to their end), keeping each block's roped keys
    /// and values.
    fn prefix(
        &self,
        be: &Backend,
        hidden: &[f32],
        slots: &[(usize, usize)],
        images: &[RefImage],
    ) -> Result<Prefix> {
        let w = self.dit.as_ref().expect("transformer loaded");
        let hp = &self.hp;
        let dim = hp.heads * hp.head_dim;
        let ctx_dim = self.te_hp.hidden;
        let n_text = hidden.len() as i64 / ctx_dim;
        let layout = prefix_layout(n_text as usize, slots, images)?;
        let n = layout.positions.len() as i64;
        let shapes: Vec<[i64; 2]> = (0..2 * hp.layers).map(|_| [dim, n]).collect();
        let refs: Vec<&[i64]> = shapes.iter().map(|s| s.as_slice()).collect();
        // F16 like the flash-attention path that reads them: half the memory
        let kv = Resident::new(be, ffi::ty::F16, &refs)?;
        let chunks = layout.attn_chunks();
        let nodes = 4096 + hp.layers as usize * (128 + ATTN_CHUNK_NODES * chunks.len());
        let gr = Graph::new(be.api, nodes)?;
        let ctx_t = gr.input_f32(&[ctx_dim, n_text], hidden, "context");
        let temb = gr.input_f32(&[256], &qwen3::timestep_embedding(0.0), "temb");
        let (cos, sin) = rope_tables(hp, &layout.positions);
        let cos_t = gr.input_f32(&[hp.head_dim / 2, n], &cos, "cos");
        let sin_t = gr.input_f32(&[hp.head_dim / 2, n], &sin, "sin");
        let masks: Vec<Option<Tensor<'_>>> = chunks
            .iter()
            .enumerate()
            .map(|(i, c)| {
                c.mask
                    .as_ref()
                    .map(|m| gr.input_f32(&[c.keys, c.len], m, &format!("mask{i}")))
            })
            .collect();
        let g = &gr;
        let (_, m) = modulation(g, w, temb)?;
        // txt_in: zero-centered RMSNorm, then a GELU MLP
        let y = g.modulate(
            g.rms_norm(ctx_t, EPS),
            w.get("txt_in.text_norm.weight")?,
            None,
        );
        let y = g.gelu(g.linear(y, w.get("txt_in.in_layer.weight")?, None));
        let txt = g.linear(y, w.get("txt_in.out_layer.weight")?, None);
        let mut x: Option<Tensor<'_>> = None;
        for seg in &layout.segments {
            let part = match *seg {
                Seg::Text { start, len, .. } => {
                    let nb = txt.strides()[1];
                    g.view_2d(txt, dim, len as i64, nb, start * nb)
                }
                Seg::Image { index, .. } => {
                    let r = &images[index];
                    let t = gr.input_f32(
                        &[hp.in_channels, r.lh * r.lw],
                        &r.tokens,
                        &format!("ref{index}"),
                    );
                    g.linear(t, w.get("img_in.weight")?, None)
                }
            };
            x = Some(match x {
                Some(prev) => g.concat(prev, g.cont(part), 1),
                None => g.cont(part),
            });
        }
        let mut x = x.ok_or_else(|| anyhow!("empty prompt"))?;
        for l in 0..hp.layers {
            let last = l == hp.layers - 1;
            x = block(g, w, hp, l, x, &m, cos_t, sin_t, |q, k, v| {
                gr.store(k, kv.tensor(2 * l as usize));
                gr.store(v, kv.tensor(2 * l as usize + 1));
                // the last block's output is never read
                (!last).then(|| prefix_attention(g, q, k, v, hp.heads, &chunks, &masks))
            })?;
            if last {
                break;
            }
        }
        gr.compute(be)?;
        Ok(Prefix {
            frame: layout.next,
            kv,
        })
    }

    /// The velocity of the image tokens `x: [n][in_channels]` at timestep `t` (in `[0, 1000]`).
    fn step(
        &self,
        be: &Backend,
        prefix: &Prefix,
        x: &[f32],
        lh: i64,
        lw: i64,
        t: f32,
    ) -> Result<Vec<f32>> {
        let w = self.dit.as_ref().expect("transformer loaded");
        let hp = &self.hp;
        let n = lh * lw;
        // without flash attention, each layer's attention is one subgraph per query chunk
        let chunks = if self.flash {
            0
        } else {
            qwen3::query_chunks(0, n, ATTN_CHUNK).count()
        };
        let nodes = 4096 + hp.layers as usize * (128 + ATTN_CHUNK_NODES * chunks);
        let gr = Graph::new(be.api, nodes)?;
        let x_t = gr.input_f32(&[hp.in_channels, n], x, "latents");
        let temb = gr.input_f32(&[256], &qwen3::timestep_embedding(t), "temb");
        let (cos, sin) = rope_tables(hp, &image_positions(prefix.frame, lh, lw));
        let cos_t = gr.input_f32(&[hp.head_dim / 2, n], &cos, "cos");
        let sin_t = gr.input_f32(&[hp.head_dim / 2, n], &sin, "sin");
        let g = &gr;
        let (tt, m) = modulation(g, w, temb)?;
        let mut x = g.linear(x_t, w.get("img_in.weight")?, None);
        for l in 0..hp.layers {
            let (pk, pv) = (
                prefix.kv.tensor(2 * l as usize),
                prefix.kv.tensor(2 * l as usize + 1),
            );
            x = block(g, w, hp, l, x, &m, cos_t, sin_t, |q, k, v| {
                let kk = g.concat(g.cast(pk, ffi::ty::F32), g.cont(k), 1);
                let vv = g.concat(g.cast(pv, ffi::ty::F32), g.cont(v), 1);
                // flash attention scores in tiles; the dense scores are
                // bounded by query chunks
                Some(if self.flash {
                    qwen3::sdpa_gqa(g, q, kk, vv, hp.heads, hp.heads, None, true)
                } else {
                    let chunks = qwen3::query_chunks(0, q.shape()[1], ATTN_CHUNK);
                    qwen3::attend_in_chunks(g, q, chunks, |qs, _| {
                        qwen3::sdpa_gqa(g, qs, kk, vv, hp.heads, hp.heads, None, false)
                    })
                })
            })?;
        }
        let scale = g.linear(tt, w.get("norm_out.linear.weight")?, None);
        let x = g.modulate(g.norm(x, EPS), scale, None);
        let out = g.cont(g.linear(x, w.get("proj_out.weight")?, None));
        gr.mark_output(out);
        gr.compute(be)?;
        Ok(gr.output_f32(out))
    }

    pub fn generate(
        &mut self,
        be: &mut Backend,
        p: &GenParams,
        mut progress: impl FnMut(i32, i32) -> bool,
    ) -> Result<Output> {
        if p.n_frames > 1 || p.gen_audio {
            bail!("Qwen-Image 2.1 generates images only");
        }
        if p.image.is_some() || p.video.is_some() || p.action.is_some() {
            bail!("Qwen-Image 2.1 takes reference images (\"images\"), not video conditioning");
        }
        // `check_inputs`: multiples of 32
        let (w, h) = (p.width / 32 * 32, p.height / 32 * 32);
        if w <= 0 || h <= 0 {
            bail!("width and height must be at least 32");
        }
        check_refs(p)?;
        let steps = if p.n_steps > 0 {
            p.n_steps
        } else {
            self.default_steps
        };
        let cfg = if p.cfg_scale > 0.0 { p.cfg_scale } else { 1.0 };
        let use_cfg = cfg > 1.0;
        // Qwen has no BOS: an empty prompt leaves the encoder nothing to read
        let nonempty = |s: &str| {
            if s.is_empty() {
                " ".to_string()
            } else {
                s.to_string()
            }
        };
        let (pos, neg) = (nonempty(&p.prompt), nonempty(&p.negative_prompt));
        let mut prompts = vec![pos.as_str()];
        if use_cfg {
            prompts.push(neg.as_str());
        }
        self.prepare(
            be,
            &prompts,
            &p.ref_images,
            &refs_key(&p.ref_images),
            (w, h),
        )?;

        let (lh, lw) = (h / SCALE, w / SCALE);
        let n = lh * lw;
        let c = self.hp.in_channels;
        let seed = p.seed.unwrap_or_else(super::rand_seed);
        // noise in the pipeline's [C][H][W] order, packed to tokens [n][C]
        let mut noise = vec![0f32; (c * n) as usize];
        if !cosmos3::noise_from_env("MEDIAGEN_NOISE", &mut noise)? {
            // stable-diffusion.cpp's default noise: the same seed, the same image
            noise = super::philox_randn(seed as u64, noise.len());
        }
        dump("noise", &noise);
        let mut x = transpose(&noise, c as usize, n as usize);
        let sigmas = self.sched.sigmas(steps as usize, n);
        eprintln!(
            "[llmman] mediagen: generating {w}x{h}, {n} tokens, {steps} steps, cfg {cfg:.1}, seed {seed}"
        );
        let cond = &self.cond.as_ref().expect("prompt encoded").1;
        let ncond = use_cfg.then(|| &self.ncond.as_ref().expect("negative prompt encoded").1);
        for i in 0..steps as usize {
            let t0 = Instant::now();
            let t = sigmas[i] * 1000.0;
            let mut v = self.step(be, cond, &x, lh, lw, t)?;
            if let Some(nc) = ncond {
                let u = self.step(be, nc, &x, lh, lw, t)?;
                for (v, u) in v.iter_mut().zip(&u) {
                    *v = u + cfg * (*v - u);
                }
            }
            if i == 0 {
                dump("pred0", &v);
            }
            let dt = sigmas[i + 1] - sigmas[i];
            for (x, v) in x.iter_mut().zip(&v) {
                *x += dt * v;
            }
            eprintln!(
                "[llmman] mediagen: step {}/{steps} sigma {:.4} -> {:.4} ({:.2} s)",
                i + 1,
                sigmas[i],
                sigmas[i + 1],
                t0.elapsed().as_secs_f64()
            );
            if !progress(i as i32 + 1, steps) {
                bail!("cancelled");
            }
        }
        be.release_compute()?;
        dump("latents", &x);
        // debugging aid: decode given latents instead of the sampled ones
        cosmos3::noise_from_env("MEDIAGEN_LATENTS", &mut x)?;

        let t0 = Instant::now();
        let mut z = transpose(&x, n as usize, c as usize);
        for (ci, row) in z.chunks_mut(n as usize).enumerate() {
            let (m, s) = (self.latents_mean[ci], self.latents_std[ci]);
            row.iter_mut().for_each(|v| *v = *v * s + m);
        }
        let dec = wan_vae::decode(&self.vae, be, &[z], c, lh, lw, &self.layout);
        be.release_compute()?;
        let dec = dec?;
        eprintln!(
            "[llmman] mediagen: decoded {}x{} in {:.2} s",
            dec.width,
            dec.height,
            t0.elapsed().as_secs_f64()
        );
        let to_u8 = |v: f32| (((v + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
        let nc = dec.channels as usize;
        if !(3..=4).contains(&nc) {
            bail!("the VAE decoded {nc} channels, expected RGB or RGBA");
        }
        let px = (dec.width * dec.height) as usize;
        let mut out = Output {
            width: dec.width,
            height: dec.height,
            n_frames: 1,
            fps: p.fps,
            rgb: Vec::with_capacity(px * 3),
            revised_prompt: p.prompt.clone(),
            seed,
            ..Default::default()
        };
        for pix in dec.rgb.chunks(nc).take(px) {
            out.rgb.extend(pix[..3].iter().map(|&v| to_u8(v)));
            if nc == 4 {
                out.alpha.push(to_u8(pix[3]));
            }
        }
        Ok(out)
    }
}

/// Names the reference images of a request, for the prompt caches.
fn refs_key(refs: &[super::Frames]) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for f in refs {
        (f.width, f.height, &f.rgb).hash(&mut h);
    }
    format!("{}:{:016x}", refs.len(), h.finish())
}

fn encode_ids(tok: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tok
        .encode(text, false)
        .map_err(|e| anyhow!("tokenizing the prompt: {e}"))?
        .get_ids()
        .to_vec())
}

/// `[rows][cols]` -> `[cols][rows]`.
pub(super) fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; a.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = a[r * cols + c];
        }
    }
    out
}

/// `QwenImage21Rope` positions of an image's tokens: the frame axis frozen at
/// `frame`, a height/width grid centred on zero.
fn image_positions(frame: f64, lh: i64, lw: i64) -> Vec<[f64; 3]> {
    let mut out = Vec::with_capacity((lh * lw) as usize);
    for y in 0..lh {
        for x in 0..lw {
            out.push([
                frame,
                (y - (lh - lh / 2)) as f64,
                (x - (lw - lw / 2)) as f64,
            ]);
        }
    }
    out
}

/// cos/sin tables `[n][head_dim/2]`: the frame, height and width axes' frequencies
/// (`1 / 10000^(2i/dim)` over each axis' dims) one after the other.
fn rope_tables(hp: &Hparams, positions: &[[f64; 3]]) -> (Vec<f32>, Vec<f32>) {
    super::zimage::rope_tables(&hp.axes, 10000.0, positions)
}

/// The prefix's attention of `q` over `k`/`v` (`[dim, n]` each) by
/// [`Layout::attn_chunks`], whose text chunks' `masks` are on the graph.
fn prefix_attention<'g>(
    g: &'g Graph<'g>,
    q: Tensor<'g>,
    k: Tensor<'g>,
    v: Tensor<'g>,
    heads: i64,
    chunks: &[AttnChunk],
    masks: &[Option<Tensor<'g>>],
) -> Tensor<'g> {
    let (dim, nk, nv) = (k.shape()[0], k.strides()[1], v.strides()[1]);
    let runs = chunks.iter().map(|c| (c.start, c.len));
    qwen3::attend_in_chunks(g, q, runs, |qs, i| {
        let keys = chunks[i].keys;
        let ks = g.view_2d(k, dim, keys, nk, 0);
        let vs = g.view_2d(v, dim, keys, nv, 0);
        qwen3::sdpa_gqa(g, qs, ks, vs, heads, heads, masks[i], false)
    })
}

/// The shared modulation: `silu(temb)` and `[scale1, tanh(gate1), scale2, tanh(gate2)]`.
fn modulation<'g>(
    g: &'g Graph<'g>,
    w: &'g Weights,
    temb: Tensor<'g>,
) -> Result<(Tensor<'g>, [Tensor<'g>; 4])> {
    let te = g.linear(
        g.reshape_2d(temb, 256, 1),
        w.get("time_text_embed.timestep_embedder.linear_1.weight")?,
        None,
    );
    let te = g.linear(
        g.silu(te),
        w.get("time_text_embed.timestep_embedder.linear_2.weight")?,
        None,
    );
    let tt = g.silu(te);
    let m = g.linear(tt, w.get("modulation.1.weight")?, None);
    let dim = m.shape()[0] / 4;
    let part = |i: usize| g.cont(g.view_2d(m, dim, 1, m.strides()[1], i * dim as usize * 4));
    Ok((tt, [part(0), g.tanh(part(1)), part(2), g.tanh(part(3))]))
}

/// Interleaved rotary pairs `(2i, 2i+1)` of each head moved to `(i, i + hd/2)`.
/// Queries and keys only meet in dot products, so they stay permuted.
pub(super) fn halves<'g>(g: &'g Graph<'g>, x: Tensor<'g>, hd: i64) -> Tensor<'g> {
    let [dim, t, ..] = x.shape();
    let x4 = g.reshape_4d(g.cont(x), 2, hd / 2, dim / hd, t);
    g.reshape_2d(g.cont(g.permute(x4, 1, 0, 2, 3)), dim, t)
}

/// One `QwenImage21TransformerBlock`; `attend(q, k, v)` runs the attention
/// (`None` when the block's output is not needed).
fn block<'g>(
    g: &'g Graph<'g>,
    w: &'g Weights,
    hp: &Hparams,
    l: i64,
    x: Tensor<'g>,
    m: &[Tensor<'g>; 4],
    cos: Tensor<'g>,
    sin: Tensor<'g>,
    attend: impl FnOnce(Tensor<'g>, Tensor<'g>, Tensor<'g>) -> Option<Tensor<'g>>,
) -> Result<Tensor<'g>> {
    let p = format!("transformer_blocks.{l}");
    let wt = |s: &str| w.get(&format!("{p}.{s}.weight"));
    let hd = hp.head_dim;
    let h = g.modulate(g.norm(x, EPS), m[0], None);
    let q = qwen3::head_rms(
        g,
        g.linear(h, wt("attn.to_q")?, None),
        hd,
        EPS,
        Some(wt("attn.norm_q")?),
    );
    let k = qwen3::head_rms(
        g,
        g.linear(h, wt("attn.to_k")?, None),
        hd,
        EPS,
        Some(wt("attn.norm_k")?),
    );
    let v = g.linear(h, wt("attn.to_v")?, None);
    let q = qwen3::rope(g, halves(g, q, hd), cos, sin, hp.heads);
    let k = qwen3::rope(g, halves(g, k, hd), cos, sin, hp.heads);
    let Some(attn) = attend(q, k, v) else {
        return Ok(x);
    };
    let x = g.add(x, g.mul(g.linear(attn, wt("attn.to_out.0")?, None), m[1]));
    let h = g.modulate(g.norm(x, EPS), m[2], None);
    let h = match w.opt(&format!("{p}.img_mlp.gate_up.weight")) {
        // ComfyUI's fused `[gate; proj]` (the GGUFs)
        Some(gu) => {
            let gu = g.linear(h, gu, None);
            let [two, t, ..] = gu.shape();
            let nb = gu.strides()[1];
            let gate = g.view_2d(gu, two / 2, t, nb, 0);
            let up = g.view_2d(gu, two / 2, t, nb, (two / 2) as usize * 4);
            g.mul(g.silu(g.cont(gate)), g.cont(up))
        }
        None => {
            let gate = g.silu(g.linear(h, wt("img_mlp.gate_layer")?, None));
            g.mul(gate, g.linear(h, wt("img_mlp.proj")?, None))
        }
    };
    Ok(g.add(x, g.mul(g.linear(h, wt("img_mlp.out")?, None), m[3])))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmas_match_the_scheduler() {
        // FlowMatchEulerDiscreteScheduler.set_timesteps with the pack's
        // scheduler_config.json, as QwenImage21Pipeline calls it
        let sched = Sched::from_config(&Value::Null);
        let cases: [(usize, i64, &[f32]); 2] = [
            (
                8,
                64 * 64,
                &[
                    1.0, 0.916024, 0.820046, 0.709295, 0.580075, 0.427347, 0.244054, 0.02, 0.0,
                ],
            ),
            (5, 32 * 48, &[1.0, 0.824398, 0.612178, 0.350554, 0.02, 0.0]),
        ];
        for (n, seq, want) in cases {
            let got = sched.sigmas(n, seq);
            assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(want) {
                assert!((g - w).abs() < 1e-6, "{got:?} vs {want:?}");
            }
        }
        // one step: nothing to stretch
        assert_eq!(sched.sigmas(1, 4096), [1.0, 0.0]);
    }

    #[test]
    fn image_positions_are_centred() {
        let p = image_positions(7.0, 2, 3);
        assert_eq!(p[0], [7.0, -1.0, -2.0]);
        assert_eq!(p[5], [7.0, 0.0, 0.0]);
        // odd sizes: range(-(n - n//2), n//2)
        let p = image_positions(0.0, 3, 1);
        assert_eq!(
            p.iter().map(|p| p[1]).collect::<Vec<_>>(),
            [-2.0, -1.0, 0.0]
        );
    }

    #[test]
    fn rope_tables_split_the_axes() {
        let hp = Hparams {
            layers: 1,
            heads: 1,
            head_dim: 128,
            in_channels: 64,
            axes: [16, 56, 56],
        };
        let (cos, sin) = rope_tables(&hp, &[[2.0, 3.0, 5.0]]);
        assert_eq!(cos.len(), 64);
        assert!((sin[0] - 2f32.sin()).abs() < 1e-6);
        assert!((sin[8] - 3f32.sin()).abs() < 1e-6);
        assert!((sin[36] - 5f32.sin()).abs() < 1e-6);
        let f = 1.0 / 10000f64.powf(2.0 / 56.0);
        assert!((cos[37] as f64 - (5.0 * f).cos()).abs() < 1e-6);
    }

    #[test]
    fn te_names_map_to_the_qwen3_stack() {
        assert_eq!(
            te_name("model.language_model.layers.3.self_attn.q_proj.weight").as_deref(),
            Some("layers.3.self_attn.to_q.weight")
        );
        assert_eq!(
            te_name("model.language_model.layers.0.self_attn.k_norm.weight").as_deref(),
            Some("layers.0.self_attn.norm_k.weight")
        );
        assert_eq!(
            te_name("model.language_model.embed_tokens.weight").as_deref(),
            Some("embed_tokens.weight")
        );
        assert_eq!(te_name("model.language_model.norm.weight"), None);
        assert_eq!(te_name("model.visual.merger.norm.weight"), None);
        assert_eq!(te_name("lm_head.weight"), None);
    }

    #[test]
    fn prefix_layout_swaps_vision_tokens_for_latents() {
        // 5 prompt tokens, tokens 1..3 an image whose latents are 4x2
        let l = prefix_layout_grids(5, &[(1, 2)], &[(4, 2)]).unwrap();
        assert_eq!(
            l.segments,
            [
                Seg::Text {
                    start: 0,
                    len: 1,
                    at: 0
                },
                Seg::Image {
                    index: 0,
                    len: 8,
                    at: 1
                },
                Seg::Text {
                    start: 3,
                    len: 2,
                    at: 9
                },
            ]
        );
        assert_eq!(l.positions[0], [0.0; 3]);
        // the image at frame 1, centred; the text resumes max(4, 2) later
        assert_eq!(l.positions[1], [1.0, -2.0, -1.0]);
        assert_eq!(l.positions[9], [5.0; 3]);
        assert_eq!(l.next, 7.0);
        let chunks = l.attn_chunks();
        let seen = |q: i64, k: i64| {
            let c = chunks
                .iter()
                .find(|c| (c.start..c.start + c.len).contains(&q))
                .unwrap();
            k < c.keys
                && c.mask
                    .as_ref()
                    .is_none_or(|m| m[((q - c.start) * c.keys + k) as usize] == 0.0)
        };
        assert_eq!(chunks.iter().map(|c| c.len).sum::<i64>(), 11);
        // text is causal, the image sees itself whole but not what follows
        assert!(!seen(0, 1));
        assert!(seen(1, 8) && !seen(1, 9));
        assert!(seen(9, 8) && !seen(9, 10));
        assert!(prefix_layout_grids(5, &[(1, 2)], &[(2, 2)]).is_err());
    }

    #[test]
    fn attn_chunks_of_a_long_image_see_the_whole_image() {
        // a 40x40 latent image (1600 tokens) between two text runs
        let l = prefix_layout_grids(30 + 400 + 5, &[(30, 400)], &[(40, 40)]).unwrap();
        let chunks = l.attn_chunks();
        let image: Vec<_> = chunks
            .iter()
            .filter(|c| (30..1630).contains(&c.start))
            .collect();
        assert_eq!(image.len(), 2);
        assert!(image.iter().all(|c| c.keys == 1630 && c.mask.is_none()));
    }

    #[test]
    fn the_vae_encoder_loads_only_for_edits() {
        let (text, edits) = (vae_name(false), vae_name(true));
        for n in ["decoder.conv_in.weight", "post_quant_conv.weight"] {
            assert!(text(n).is_some() && edits(n).is_some(), "{n}");
        }
        for n in ["encoder.conv_in.weight", "quant_conv.weight"] {
            assert!(text(n).is_none() && edits(n).is_some(), "{n}");
        }
        assert!(edits("other.weight").is_none());
    }

    #[test]
    fn check_refs_bounds_the_vision_tokens() {
        let r = || super::super::Frames {
            width: 64,
            height: 64,
            n_frames: 1,
            rgb: vec![0; 64 * 64 * 3],
        };
        let mut p = GenParams {
            width: 1024,
            height: 1024,
            ref_images: vec![r(), r(), r()],
            ..Default::default()
        };
        // 1024 vision tokens each at 1024x1024
        assert!(check_refs(&p).is_ok());
        p.ref_images.push(r());
        assert!(check_refs(&p).is_err());
        p.ref_images.truncate(1);
        p.ref_images[0].height = 0;
        assert!(check_refs(&p).is_err());
    }

    #[test]
    fn ref_size_keeps_the_output_area() {
        assert_eq!(ref_size(640, 480, 1024 * 1024), (1184, 896));
        assert_eq!(ref_size(100, 100, 256 * 256), (256, 256));
    }

    #[test]
    fn sample_sigmas_are_shifted_and_stretched() {
        // diffusers' set_timesteps(sigmas=[1, 0.5]) with shift_terminal 0.02
        let s = Sched {
            dynamic: false,
            fixed: vec![1.0, 0.5],
            ..Sched::from_config(&Value::Null)
        };
        let got = s.sigmas(2, 4096);
        assert!((got[1] - 0.02).abs() < 1e-6, "{got:?}");
    }

    #[test]
    fn turbo_uses_its_eight_sigmas() {
        let s = Sched::turbo();
        assert_eq!(
            s.sigmas(8, 4096),
            [TURBO_SIGMAS.as_slice(), &[0.0]].concat()
        );
        // other step counts: the static shift of 1, no terminal stretch
        assert_eq!(s.sigmas(4, 4096), [1.0, 0.75, 0.5, 0.25, 0.0]);
    }

    #[test]
    fn transpose_roundtrips() {
        let a: Vec<f32> = (0..6).map(|v| v as f32).collect();
        let t = transpose(&a, 2, 3);
        assert_eq!(t, [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        assert_eq!(transpose(&t, 3, 2), a);
    }
}
