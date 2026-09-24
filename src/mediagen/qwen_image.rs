//! Qwen-Image 2.1 text-to-image (diffusers `QwenImage21Pipeline`) from its
//! Diffusers safetensors: Qwen3-VL text encoder, single-stream transformer,
//! flow-matching Euler sampler, Wan-derived RGBA VAE.
//!
//! Prompt tokens are modulated from `t = 0` and attend causally, so their
//! keys and values are computed once per prompt and reused by every step.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use serde_json::Value;

use super::backend::{Backend, Ctx, Graph, Resident};
use super::cosmos3::{self, dit as qwen3, read_json};
use super::ffi::{self, Api, Tensor};
use super::wan_vae;
use super::weights::{read_index, LoadOpts, Weights};
use super::{dump, GenParams, Mt19937, Output};

pub const PIPELINE: &str = "QwenImage21Pipeline";
/// `QwenImage21Pipeline.__call__` defaults: no guidance, 1024x1024.
pub const DEFAULT_STEPS: i32 = 40;
pub const DEFAULT_SIZE: i64 = 1024;
const SYSTEM: &str = "Comprehend and analyze the provided prompt.";
/// Pixels per latent token: 16x VAE, no patching.
const SCALE: i64 = 16;
const EPS: f32 = 1e-6;
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

/// `FlowMatchEulerDiscreteScheduler` with dynamic exponential shifting.
struct Sched {
    base_seq_len: f64,
    max_seq_len: f64,
    base_shift: f64,
    max_shift: f64,
    shift_terminal: f64,
}

impl Sched {
    fn from_config(v: &Value) -> Sched {
        let f = |k: &str, d: f64| v.get(k).and_then(Value::as_f64).unwrap_or(d);
        Sched {
            base_seq_len: f("base_image_seq_len", 256.0),
            max_seq_len: f("max_image_seq_len", 8192.0),
            base_shift: f("base_shift", 0.5),
            max_shift: f("max_shift", 0.9),
            shift_terminal: f("shift_terminal", 0.02),
        }
    }

    /// The pipeline's `linspace(1, 1/n, n)` shifted by `calculate_shift(seq_len)`,
    /// stretched to end at `shift_terminal`, then 0.
    fn sigmas(&self, n: usize, seq_len: i64) -> Vec<f32> {
        let m = (self.max_shift - self.base_shift) / (self.max_seq_len - self.base_seq_len);
        let mu = (seq_len as f64 * m + self.base_shift - m * self.base_seq_len).exp();
        let mut s: Vec<f64> = (0..n)
            .map(|i| {
                let t = if n > 1 {
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
    n: i64,
    kv: Resident,
}

pub struct Model {
    api: &'static Api,
    hp: Hparams,
    sched: Sched,
    te_hp: cosmos3::Hparams,
    tok: tokenizers::Tokenizer,
    /// Tokens of the system turn the pipeline drops from the hidden states.
    drop_idx: usize,
    dit_files: Vec<PathBuf>,
    te_files: Vec<PathBuf>,
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
    cond: Option<(String, Prefix)>,
    ncond: Option<(String, Prefix)>,
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

fn vae_name(n: &str) -> Option<String> {
    (n.starts_with("decoder.") || n.starts_with("post_quant_conv.")).then(|| n.to_string())
}

/// Bytes the tensors `rename` keeps take in `files`.
fn weight_bytes(files: &[PathBuf], rename: fn(&str) -> Option<String>) -> Result<u64> {
    let mut total = 0;
    for f in files {
        total += read_index(f)?
            .tensors
            .iter()
            .filter(|d| rename(&d.name).is_some())
            .map(|d| d.nbytes)
            .sum::<u64>();
    }
    Ok(total)
}

/// Do `bytes` more weights fit, with room for the compute buffers?
fn fits(be: &Backend, bytes: u64) -> bool {
    be.weight_memory().0 as u64 >= bytes + HEADROOM
}

impl Model {
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
        let sched = files
            .get("scheduler/scheduler_config.json")
            .map(|p| read_json(p))
            .transpose()?
            .map_or_else(
                || Sched::from_config(&Value::Null),
                |v| Sched::from_config(&v),
            );
        let te_cfg = read_json(file(files, "text_encoder/config.json")?)?;
        let te_hp = cosmos3::Hparams::from_config(te_cfg.get("text_config").unwrap_or(&te_cfg))?;
        let tok_path = file(files, "processor/tokenizer.json")
            .or_else(|_| file(files, "tokenizer/tokenizer.json"))?;
        let tok = tokenizers::Tokenizer::from_file(tok_path)
            .map_err(|e| anyhow!("{}: {e}", tok_path.display()))?;
        let drop_idx =
            encode_ids(&tok, &format!("<|im_start|>system\n{SYSTEM}<|im_end|>\n"))?.len();

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
        let (latents_mean, latents_std) = (floats("latents_mean"), floats("latents_std"));
        let nch = hp.in_channels as usize;
        if latents_mean.len() != nch
            || latents_std.len() != nch
            || latents_std.iter().any(|&s| s <= 0.0)
        {
            bail!("vae/config.json: latents_mean/latents_std must each hold {nch} positive-std entries");
        }
        let mut temporal_up: Vec<bool> = vcfg
            .get("temperal_downsample")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_bool).collect())
            .unwrap_or_default();
        temporal_up.reverse();
        let layout = wan_vae::Layout {
            temporal_up,
            patch: vcfg.get("patch_size").and_then(Value::as_i64).unwrap_or(1) > 1,
        };
        let vae_path = file(files, "vae/diffusion_pytorch_model.safetensors")?;
        let vae = Weights::load(
            api,
            vae_path,
            be.weight_buft(),
            &LoadOpts {
                rename: Some(Box::new(vae_name)),
                transform: None,
            },
        )?;
        if !vae.has("decoder.conv_in.weight") {
            bail!("the VAE has no decoder");
        }

        let dit_files = shards(files, "transformer")?;
        let te_files = shards(files, "text_encoder")?;
        let dit_bytes = weight_bytes(&dit_files, |n| Some(n.to_string()))?;
        let te_bytes = weight_bytes(&te_files, te_name)?;
        eprintln!(
            "[llmman] mediagen: qwen-image model ready in {:.1} s ({} layers, transformer {:.1} GiB, text encoder {:.1} GiB, loaded on first use)",
            t0.elapsed().as_secs_f64(),
            hp.layers,
            dit_bytes as f64 / (1u64 << 30) as f64,
            te_bytes as f64 / (1u64 << 30) as f64,
        );
        Ok(Model {
            api,
            hp,
            sched,
            te_hp,
            tok,
            drop_idx,
            dit_files,
            te_files,
            dit_bytes,
            te_bytes,
            dit: None,
            te: None,
            vae,
            layout,
            latents_mean,
            latents_std,
            flash,
            cond: None,
            ncond: None,
        })
    }

    fn load_files(
        &self,
        be: &Backend,
        files: &[PathBuf],
        rename: Option<fn(&str) -> Option<String>>,
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
            self.dit = Some(self.load_files(be, &self.dit_files, None)?);
        }
        Ok(())
    }

    /// The prompt caches of `prompts` (positive, then negative), computing the missing ones.
    fn prepare(&mut self, be: &mut Backend, prompts: &[&str]) -> Result<()> {
        let missing: Vec<usize> = (0..prompts.len())
            .filter(|&i| {
                let slot = if i == 0 { &self.cond } else { &self.ncond };
                !matches!(slot, Some((p, _)) if p == prompts[i])
            })
            .collect();
        if missing.is_empty() {
            return self.ensure_dit(be);
        }
        if self.te.is_none() {
            if self.dit.is_some() && !fits(be, self.te_bytes) {
                self.dit = None;
                be.release_compute()?;
            }
            self.te = Some(self.load_files(be, &self.te_files, Some(te_name))?);
        }
        let mut hidden = Vec::new();
        for &i in &missing {
            let t0 = Instant::now();
            let h = self.encode(be, prompts[i])?;
            dump(
                if i == 0 {
                    "text_hidden"
                } else {
                    "ntext_hidden"
                },
                &h,
            );
            eprintln!(
                "[llmman] mediagen: encoded {} prompt ({} tokens) in {:.2} s",
                if i == 0 { "positive" } else { "negative" },
                h.len() / self.te_hp.hidden as usize,
                t0.elapsed().as_secs_f64()
            );
            hidden.push(h);
        }
        be.release_compute()?;
        self.ensure_dit(be)?;
        for (&i, h) in missing.iter().zip(&hidden) {
            let prefix = self.prefix(be, h)?;
            *(if i == 0 {
                &mut self.cond
            } else {
                &mut self.ncond
            }) = Some((prompts[i].to_string(), prefix));
        }
        Ok(())
    }

    /// `_get_qwen_prompt_embeds`: the text model's last hidden states `[T][hidden]`
    /// over the chat template, the system turn dropped.
    fn encode(&self, be: &Backend, prompt: &str) -> Result<Vec<f32>> {
        let te = self.te.as_ref().expect("text encoder loaded");
        let text = format!(
            "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        );
        let ids = encode_ids(&self.tok, &text)?;
        let h = qwen3::text_hidden(&self.te_hp, te, be, &ids)?;
        Ok(h[self.drop_idx * self.te_hp.hidden as usize..].to_vec())
    }

    /// The prompt tokens through every block (`t = 0` modulation, causal attention),
    /// keeping each block's roped keys and values.
    fn prefix(&self, be: &Backend, hidden: &[f32]) -> Result<Prefix> {
        let w = self.dit.as_ref().expect("transformer loaded");
        let hp = &self.hp;
        let dim = hp.heads * hp.head_dim;
        let ctx_dim = self.te_hp.hidden;
        let n = hidden.len() as i64 / ctx_dim;
        let shapes: Vec<[i64; 2]> = (0..2 * hp.layers).map(|_| [dim, n]).collect();
        let refs: Vec<&[i64]> = shapes.iter().map(|s| s.as_slice()).collect();
        let kv = Resident::new(be, ffi::ty::F32, &refs)?;
        let mut gr = Graph::new(be.api, 4096 + hp.layers as usize * 128)?;
        let ctx_t = gr.input_f32(&[ctx_dim, n], hidden, "context");
        let temb = gr.input_f32(&[256], &qwen3::timestep_embedding(0.0), "temb");
        let pos: Vec<[f64; 3]> = (0..n).map(|i| [i as f64; 3]).collect();
        let (cos, sin) = rope_tables(hp, &pos);
        let cos_t = gr.input_f32(&[hp.head_dim / 2, n], &cos, "cos");
        let sin_t = gr.input_f32(&[hp.head_dim / 2, n], &sin, "sin");
        let mask = gr.input_f32(&[n, n], &qwen3::causal_mask(n), "mask");
        let g = &gr.ctx;
        let (_, m) = modulation(g, w, temb)?;
        // txt_in: zero-centered RMSNorm, then a GELU MLP
        let y = g.modulate(
            g.rms_norm(ctx_t, EPS),
            w.get("txt_in.text_norm.weight")?,
            None,
        );
        let y = g.gelu(g.linear(y, w.get("txt_in.in_layer.weight")?, None));
        let mut x = g.linear(y, w.get("txt_in.out_layer.weight")?, None);
        for l in 0..hp.layers {
            let last = l == hp.layers - 1;
            x = block(g, w, hp, l, x, &m, cos_t, sin_t, |q, k, v| {
                gr.store(k, kv.tensors[2 * l as usize]);
                gr.store(v, kv.tensors[2 * l as usize + 1]);
                // the last block's output is never read
                (!last).then(|| qwen3::sdpa_gqa(g, q, k, v, hp.heads, hp.heads, Some(mask), false))
            })?;
            if last {
                break;
            }
        }
        gr.compute(be)?;
        Ok(Prefix { n, kv })
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
        let mut gr = Graph::new(be.api, 4096 + hp.layers as usize * 128)?;
        let x_t = gr.input_f32(&[hp.in_channels, n], x, "latents");
        let temb = gr.input_f32(&[256], &qwen3::timestep_embedding(t), "temb");
        let (cos, sin) = rope_tables(hp, &image_positions(prefix.n, lh, lw));
        let cos_t = gr.input_f32(&[hp.head_dim / 2, n], &cos, "cos");
        let sin_t = gr.input_f32(&[hp.head_dim / 2, n], &sin, "sin");
        let g = &gr.ctx;
        let (tt, m) = modulation(g, w, temb)?;
        let mut x = g.linear(x_t, w.get("img_in.weight")?, None);
        for l in 0..hp.layers {
            let (pk, pv) = (
                prefix.kv.tensors[2 * l as usize],
                prefix.kv.tensors[2 * l as usize + 1],
            );
            x = block(g, w, hp, l, x, &m, cos_t, sin_t, |q, k, v| {
                let kk = g.concat(pk, g.cont(k), 1);
                let vv = g.concat(pv, g.cont(v), 1);
                Some(qwen3::sdpa_gqa(
                    g, q, kk, vv, hp.heads, hp.heads, None, self.flash,
                ))
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
            bail!("Qwen-Image 2.1 image editing is not supported yet");
        }
        // `check_inputs`: multiples of 32
        let (w, h) = (p.width / 32 * 32, p.height / 32 * 32);
        if w <= 0 || h <= 0 {
            bail!("width and height must be at least 32");
        }
        let steps = if p.n_steps > 0 {
            p.n_steps
        } else {
            DEFAULT_STEPS
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
        self.prepare(be, &prompts)?;

        let (lh, lw) = (h / SCALE, w / SCALE);
        let n = lh * lw;
        let c = self.hp.in_channels;
        let seed = p.seed.unwrap_or_else(super::rand_seed);
        // noise in the pipeline's [C][H][W] order, packed to tokens [n][C]
        let mut noise = vec![0f32; (c * n) as usize];
        if !cosmos3::noise_from_env("MEDIAGEN_NOISE", &mut noise)? {
            let mut rng = Mt19937::new(seed);
            noise.iter_mut().for_each(|v| *v = rng.normal());
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

fn encode_ids(tok: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tok
        .encode(text, false)
        .map_err(|e| anyhow!("tokenizing the prompt: {e}"))?
        .get_ids()
        .to_vec())
}

/// `[rows][cols]` -> `[cols][rows]`.
fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; a.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = a[r * cols + c];
        }
    }
    out
}

/// `QwenImage21Rope` positions of the image tokens after `n_text` prompt tokens:
/// the frame axis frozen at `n_text`, a height/width grid centred on zero.
fn image_positions(n_text: i64, lh: i64, lw: i64) -> Vec<[f64; 3]> {
    let mut out = Vec::with_capacity((lh * lw) as usize);
    for y in 0..lh {
        for x in 0..lw {
            out.push([
                n_text as f64,
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
    let half = (hp.head_dim / 2) as usize;
    let mut cos = Vec::with_capacity(positions.len() * half);
    let mut sin = Vec::with_capacity(positions.len() * half);
    for p in positions {
        for (a, &dim) in hp.axes.iter().enumerate() {
            for i in 0..dim / 2 {
                let angle = p[a] / 10000f64.powf(2.0 * i as f64 / dim as f64);
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
    }
    (cos, sin)
}

/// The shared modulation: `silu(temb)` and `[scale1, tanh(gate1), scale2, tanh(gate2)]`.
fn modulation(g: &Ctx, w: &Weights, temb: Tensor) -> Result<(Tensor, [Tensor; 4])> {
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
    let dim = ffi::shape(m)[0] / 4;
    let part = |i: usize| g.cont(g.view_2d(m, dim, 1, ffi::strides(m)[1], i * dim as usize * 4));
    Ok((tt, [part(0), g.tanh(part(1)), part(2), g.tanh(part(3))]))
}

/// Interleaved rotary pairs `(2i, 2i+1)` of each head moved to `(i, i + hd/2)`.
/// Queries and keys only meet in dot products, so they stay permuted.
fn halves(g: &Ctx, x: Tensor, hd: i64) -> Tensor {
    let [dim, t, ..] = ffi::shape(x);
    let x4 = g.reshape_4d(g.cont(x), 2, hd / 2, dim / hd, t);
    g.reshape_2d(g.cont(g.permute(x4, 1, 0, 2, 3)), dim, t)
}

/// One `QwenImage21TransformerBlock`; `attend(q, k, v)` runs the attention
/// (`None` when the block's output is not needed).
fn block(
    g: &Ctx,
    w: &Weights,
    hp: &Hparams,
    l: i64,
    x: Tensor,
    m: &[Tensor; 4],
    cos: Tensor,
    sin: Tensor,
    attend: impl FnOnce(Tensor, Tensor, Tensor) -> Option<Tensor>,
) -> Result<Tensor> {
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
    let gate = g.silu(g.linear(h, wt("img_mlp.gate_layer")?, None));
    let h = g.mul(gate, g.linear(h, wt("img_mlp.proj")?, None));
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
        let p = image_positions(7, 2, 3);
        assert_eq!(p[0], [7.0, -1.0, -2.0]);
        assert_eq!(p[5], [7.0, 0.0, 0.0]);
        // odd sizes: range(-(n - n//2), n//2)
        let p = image_positions(0, 3, 1);
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
    fn transpose_roundtrips() {
        let a: Vec<f32> = (0..6).map(|v| v as f32).collect();
        let t = transpose(&a, 2, 3);
        assert_eq!(t, [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        assert_eq!(transpose(&t, 3, 2), a);
    }
}
