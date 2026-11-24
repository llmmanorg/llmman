//! Z-Image (Tongyi-MAI) text-to-image from GGUFs: the `lumina2` single-stream
//! transformer (`ZImageTransformer2DModel`, NextDiT), a Qwen3 text encoder
//! read through llama.cpp, the FLUX.1 VAE, and the flow-matching Euler
//! sampler with a static shift.
//!
//! Prompt tokens and image patches share one sequence: two caption refiner
//! blocks (no modulation) and two noise refiner blocks (timestep-modulated)
//! prepare each stream, then every main block attends over both. Both
//! streams are padded to a multiple of 32 tokens with learned pad tokens.

use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, bail, Result};

use super::backend::{Backend, Graph};
use super::cosmos3::dit as qwen3;
use super::ffi::{Api, Tensor};
use super::flux_vae;
use super::ltx::text::TextEncoder;
use super::weights::{Index, LoadOpts, Weights};
use super::{dump, GenParams, Mt19937, Output};

/// `ZImagePipeline` / Z-Image-Turbo defaults: 9 sigmas (8 transformer
/// evaluations, the last step lands on 0), no guidance, 1024x1024.
pub const DEFAULT_STEPS: i32 = 9;
pub const DEFAULT_SIZE: i64 = 1024;
/// `FlowMatchEulerDiscreteScheduler(shift=3.0)`, no dynamic shifting.
const SHIFT: f64 = 3.0;
const PATCH: i64 = 2;
/// Pixels per latent: the FLUX VAE's 8x, then 2x2 patches.
const VAE_SCALE: i64 = 8;
const SEQ_MULTIPLE: i64 = 32;
const THETA: f64 = 256.0;
const AXES: [i64; 3] = [32, 48, 48];
const NORM_EPS: f32 = 1e-5;
const FINAL_EPS: f32 = 1e-6;
const QK_EPS: f32 = 1e-6;
/// `ZImagePipeline(max_sequence_length=512)`.
const MAX_PROMPT_TOKENS: usize = 512;
/// The adaLN input width (`ADALN_EMBED_DIM`).
const ADALN_DIM: i64 = 256;

pub fn is_zimage(arch: &str) -> bool {
    arch == "lumina2"
}

/// Shapes read from the transformer's tensors.
#[derive(Clone, Debug)]
pub struct Hparams {
    pub dim: i64,
    pub heads: i64,
    pub kv_heads: i64,
    pub head_dim: i64,
    pub layers: i64,
    pub refiner_layers: i64,
    pub in_channels: i64,
    pub cap_dim: i64,
    /// One `attention.qkv` (original checkpoints) or `to_q`/`to_k`/`to_v` (Diffusers).
    pub fused_qkv: bool,
}

impl Hparams {
    pub fn from_index(idx: &Index) -> Result<Hparams> {
        let names: Vec<(String, &Vec<i64>)> = idx
            .tensors
            .iter()
            .filter_map(|d| Some((dit_name(&d.name)?, &d.ne)))
            .collect();
        let shape = |n: &str| -> Option<Vec<i64>> {
            names
                .iter()
                .find(|(m, _)| m == n)
                .map(|(_, s)| (*s).clone())
        };
        let count = |prefix: &str| -> i64 {
            names
                .iter()
                .filter_map(|(name, _)| {
                    let rest = name.strip_prefix(prefix)?;
                    rest.split('.').next()?.parse::<i64>().ok()
                })
                .max()
                .map_or(0, |m| m + 1)
        };
        let x_emb = shape("x_embedder.weight").ok_or_else(|| anyhow!("no x_embedder.weight"))?;
        let cap =
            shape("cap_embedder.1.weight").ok_or_else(|| anyhow!("no cap_embedder.1.weight"))?;
        let head_dim = shape("layers.0.attention.q_norm.weight")
            .or_else(|| shape("layers.0.attention.norm_q.weight"))
            .map_or(128, |s| s[0]);
        let (dim, patch_in) = (x_emb[1], x_emb[0]);
        let heads = dim / head_dim;
        let (fused_qkv, kv_heads) = match shape("layers.0.attention.qkv.weight") {
            Some(s) => (true, ((s[1] / head_dim - heads) / 2).max(1)),
            None => (
                false,
                shape("layers.0.attention.to_k.weight").map_or(heads, |s| s[1] / head_dim),
            ),
        };
        let hp = Hparams {
            dim,
            heads,
            kv_heads,
            head_dim,
            layers: count("layers."),
            refiner_layers: count("noise_refiner.").max(count("context_refiner.")),
            in_channels: patch_in / (PATCH * PATCH),
            cap_dim: cap[0],
            fused_qkv,
        };
        if hp.layers == 0
            || AXES.iter().sum::<i64>() != hp.head_dim
            || hp.in_channels * PATCH * PATCH != patch_in
        {
            bail!("unsupported Z-Image transformer {hp:?}");
        }
        Ok(hp)
    }
}

/// Diffusers checkpoints use other names for a few tensors.
fn dit_name(n: &str) -> Option<String> {
    let n = n
        .strip_prefix("model.diffusion_model.")
        .or_else(|| n.strip_prefix("diffusion_model."))
        .unwrap_or(n);
    let mut s = n.to_string();
    for (from, to) in [
        (".attention.norm_q.", ".attention.q_norm."),
        (".attention.norm_k.", ".attention.k_norm."),
        (".attention.to_out.0.", ".attention.out."),
    ] {
        s = s.replace(from, to);
    }
    for (from, to) in [
        ("all_x_embedder.2-1.", "x_embedder."),
        ("all_final_layer.2-1.", "final_layer."),
    ] {
        if let Some(rest) = s.strip_prefix(from) {
            s = format!("{to}{rest}");
        }
    }
    Some(s)
}

/// `[n_steps]` sigmas from 1 to 0, shifted, then the closing 0.
fn sigmas(n: usize) -> Vec<f32> {
    let shift = |s: f64| SHIFT * s / (1.0 + (SHIFT - 1.0) * s);
    let mut v: Vec<f32> = (0..n)
        .map(|i| {
            let s = if n > 1 {
                1.0 - i as f64 / (n - 1) as f64
            } else {
                1.0
            };
            shift(s) as f32
        })
        .collect();
    v.push(0.0);
    v
}

fn pad_to(n: i64, m: i64) -> i64 {
    (n + m - 1) / m * m
}

/// Positions `[t, row, col]` of the padded caption then the padded image
/// tokens: caption token `i` at `(i + 1, 0, 0)`, patch `(r, c)` at
/// `(caption_len + 1, r, c)`, image padding at the origin.
fn positions(n_cap_pad: i64, ph: i64, pw: i64, n_img_pad: i64) -> Vec<[f64; 3]> {
    let mut p = Vec::with_capacity((n_cap_pad + n_img_pad) as usize);
    p.extend((0..n_cap_pad).map(|i| [(i + 1) as f64, 0.0, 0.0]));
    for r in 0..ph {
        for c in 0..pw {
            p.push([(n_cap_pad + 1) as f64, r as f64, c as f64]);
        }
    }
    p.resize((n_cap_pad + n_img_pad) as usize, [0.0; 3]);
    p
}

/// cos/sin tables `[n][head_dim/2]`: each axis' `1 / theta^(2i/dim)` frequencies in turn.
pub(super) fn rope_tables(
    axes: &[i64],
    theta: f64,
    positions: &[[f64; 3]],
) -> (Vec<f32>, Vec<f32>) {
    let half: i64 = axes.iter().sum::<i64>() / 2;
    let mut cos = Vec::with_capacity(positions.len() * half as usize);
    let mut sin = Vec::with_capacity(positions.len() * half as usize);
    for p in positions {
        for (a, &dim) in axes.iter().enumerate() {
            for i in 0..dim / 2 {
                let angle = p[a] / theta.powf(2.0 * i as f64 / dim as f64);
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
    }
    (cos, sin)
}

/// Latents `[C][H][W]` -> patch tokens `[(H/2)(W/2)][2*2*C]`, features `(py, px, c)`.
fn patchify(lat: &[f32], c: i64, h: i64, w: i64) -> Vec<f32> {
    let (ph, pw) = (h / PATCH, w / PATCH);
    let f = PATCH * PATCH * c;
    let mut out = vec![0f32; (ph * pw * f) as usize];
    for ci in 0..c {
        for y in 0..h {
            for x in 0..w {
                let tok = (y / PATCH) * pw + x / PATCH;
                let feat = ((y % PATCH) * PATCH + x % PATCH) * c + ci;
                out[(tok * f + feat) as usize] = lat[((ci * h + y) * w + x) as usize];
            }
        }
    }
    out
}

/// The inverse of [`patchify`].
fn unpatchify(tok: &[f32], c: i64, h: i64, w: i64) -> Vec<f32> {
    let pw = w / PATCH;
    let f = PATCH * PATCH * c;
    let mut out = vec![0f32; (c * h * w) as usize];
    for ci in 0..c {
        for y in 0..h {
            for x in 0..w {
                let t = (y / PATCH) * pw + x / PATCH;
                let feat = ((y % PATCH) * PATCH + x % PATCH) * c + ci;
                out[((ci * h + y) * w + x) as usize] = tok[(t * f + feat) as usize];
            }
        }
    }
    out
}

pub struct Model {
    hp: Hparams,
    dit: Weights,
    te: Box<TextEncoder>,
    vae: Weights,
    flash: bool,
    /// Prompt -> caption hidden states `[T][cap_dim]` and `T`.
    cond: Option<(String, Vec<f32>, i64)>,
    ncond: Option<(String, Vec<f32>, i64)>,
}

impl Model {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        api: &'static Api,
        be: &Backend,
        idx: &Index,
        dit_path: &Path,
        vae_path: Option<&Path>,
        text_path: Option<&Path>,
        text_gpu_layers: i32,
        n_threads: i32,
        flash: bool,
    ) -> Result<Model> {
        let t0 = Instant::now();
        let hp = Hparams::from_index(idx)?;
        let vae_path =
            vae_path.ok_or_else(|| anyhow!("Z-Image needs the FLUX.1 VAE (ae.safetensors)"))?;
        let text_path =
            text_path.ok_or_else(|| anyhow!("Z-Image needs its Qwen3 text encoder GGUF"))?;
        let dit = Weights::load(
            api,
            dit_path,
            be.weight_buft(),
            &LoadOpts {
                rename: Some(Box::new(dit_name)),
                transform: None,
            },
        )?;
        let vae = Weights::load(
            api,
            vae_path,
            be.weight_buft(),
            &LoadOpts {
                rename: Some(Box::new(flux_vae::tensor_name)),
                transform: None,
            },
        )?;
        let te = TextEncoder::load(
            api,
            &text_path.to_string_lossy(),
            text_gpu_layers,
            1024,
            n_threads,
            flash,
        )?;
        eprintln!(
            "[llmman] mediagen: z-image model ready in {:.1} s ({} layers, dim {}, {} heads)",
            t0.elapsed().as_secs_f64(),
            hp.layers,
            hp.dim,
            hp.heads
        );
        Ok(Model {
            hp,
            dit,
            te,
            vae,
            flash,
            cond: None,
            ncond: None,
        })
    }

    /// The caption: Qwen3's second-to-last hidden state over the chat template.
    fn encode(&mut self, prompt: &str) -> Result<(Vec<f32>, i64)> {
        let text = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
        let n_layers = self.te.n_layers();
        let (h, n) = self
            .te
            .hidden_state(&text, n_layers as usize - 1, MAX_PROMPT_TOKENS)?;
        if h.len() as i64 != n * self.hp.cap_dim {
            bail!(
                "the text encoder's hidden size is {}, the transformer expects {}",
                h.len() as i64 / n.max(1),
                self.hp.cap_dim
            );
        }
        Ok((h, n))
    }

    fn prepare(&mut self, prompts: &[&str]) -> Result<()> {
        for (i, &p) in prompts.iter().enumerate() {
            let slot = if i == 0 { &self.cond } else { &self.ncond };
            if matches!(slot, Some((q, _, _)) if q == p) {
                continue;
            }
            let t0 = Instant::now();
            let (h, n) = self.encode(p)?;
            dump(
                if i == 0 {
                    "text_hidden"
                } else {
                    "ntext_hidden"
                },
                &h,
            );
            eprintln!(
                "[llmman] mediagen: encoded {} prompt ({n} tokens) in {:.2} s",
                if i == 0 { "positive" } else { "negative" },
                t0.elapsed().as_secs_f64()
            );
            let v = Some((p.to_string(), h, n));
            if i == 0 {
                self.cond = v;
            } else {
                self.ncond = v;
            }
        }
        self.te.release_context();
        Ok(())
    }

    /// The transformer's prediction for patch tokens `x: [n_img][64]` at
    /// `sigma`, as the velocity `[n_img][64]` (the model predicts its negation).
    fn velocity(
        &self,
        be: &Backend,
        cap: &[f32],
        n_cap: i64,
        x: &[f32],
        ph: i64,
        pw: i64,
        sigma: f32,
    ) -> Result<Vec<f32>> {
        let hp = &self.hp;
        let gr = Graph::new(
            be.api,
            8192 + (hp.layers + 2 * hp.refiner_layers) as usize * 128,
        )?;
        let out = self.forward(&gr, cap, n_cap, x, ph, pw, sigma)?;
        gr.mark_output(out);
        gr.compute(be)?;
        Ok(gr.output_f32(out))
    }

    /// The transformer graph: patch tokens in, the negated model output out.
    #[allow(clippy::too_many_arguments)]
    fn forward<'g>(
        &'g self,
        g: &'g Graph<'g>,
        cap: &[f32],
        n_cap: i64,
        x: &[f32],
        ph: i64,
        pw: i64,
        sigma: f32,
    ) -> Result<Tensor<'g>> {
        let hp = &self.hp;
        let w = &self.dit;
        let n_img = ph * pw;
        let (n_cap_pad, n_img_pad) = (pad_to(n_cap, SEQ_MULTIPLE), pad_to(n_img, SEQ_MULTIPLE));
        let f = hp.in_channels * PATCH * PATCH;
        let x_t = g.input_f32(&[f, n_img], x, "patches");
        let cap_t = g.input_f32(&[hp.cap_dim, n_cap], cap, "caption");
        // the model's timestep runs the other way: 1000 at the clean end
        let temb = g.input_f32(
            &[256],
            &qwen3::timestep_embedding(1000.0 * (1.0 - sigma)),
            "temb",
        );
        let (cos, sin) = rope_tables(&AXES, THETA, &positions(n_cap_pad, ph, pw, n_img_pad));
        let half = hp.head_dim / 2;
        let cos_t = g.input_f32(&[half, n_cap_pad + n_img_pad], &cos, "cos");
        let sin_t = g.input_f32(&[half, n_cap_pad + n_img_pad], &sin, "sin");
        let rows = |t: Tensor<'g>, from: i64, n: i64| -> Tensor<'g> {
            let nb = t.strides()[1];
            g.cont(g.view_2d(t, half, n, nb, from as usize * nb))
        };
        let (cos_c, sin_c) = (rows(cos_t, 0, n_cap_pad), rows(sin_t, 0, n_cap_pad));
        let (cos_i, sin_i) = (
            rows(cos_t, n_cap_pad, n_img_pad),
            rows(sin_t, n_cap_pad, n_img_pad),
        );

        let lin = |x: Tensor<'g>, name: &str| -> Result<Tensor<'g>> {
            Ok(g.linear(
                x,
                w.get(&format!("{name}.weight"))?,
                w.opt(&format!("{name}.bias")),
            ))
        };
        // t_embedder: mlp.0 -> silu -> mlp.2, the adaLN input (no activation)
        let t = lin(
            g.silu(lin(g.reshape_2d(temb, 256, 1), "t_embedder.mlp.0")?),
            "t_embedder.mlp.2",
        )?;
        if t.shape()[0] != ADALN_DIM.min(hp.dim) {
            bail!("unexpected timestep embedding width {}", t.shape()[0]);
        }

        let pad_with = |x: Tensor<'g>, token: &str, to: i64| -> Result<Tensor<'g>> {
            let n = x.shape()[1];
            if n == to {
                return Ok(x);
            }
            let tok = w.get(token)?;
            let tok = if tok.dtype() == super::ffi::ty::F32 {
                tok
            } else {
                g.cast(tok, super::ffi::ty::F32)
            };
            let tok = g.reshape_2d(g.cont(tok), hp.dim, 1);
            let shape = g.new_tensor(super::ffi::ty::F32, &[hp.dim, to - n]);
            Ok(g.concat(x, g.repeat(tok, shape), 1))
        };
        let cap_w = w.get("cap_embedder.0.weight")?;
        let txt = lin(g.mul(g.rms_norm(cap_t, NORM_EPS), cap_w), "cap_embedder.1")?;
        let mut txt = pad_with(txt, "cap_pad_token", n_cap_pad)?;
        let mut img = pad_with(lin(x_t, "x_embedder")?, "x_pad_token", n_img_pad)?;

        for l in 0..hp.refiner_layers {
            txt = self.block(g, &format!("context_refiner.{l}"), txt, None, cos_c, sin_c)?;
        }
        for l in 0..hp.refiner_layers {
            img = self.block(g, &format!("noise_refiner.{l}"), img, Some(t), cos_i, sin_i)?;
        }
        let mut xs = g.concat(txt, img, 1);
        for l in 0..hp.layers {
            xs = self.block(g, &format!("layers.{l}"), xs, Some(t), cos_t, sin_t)?;
        }
        // final_layer: LayerNorm without affine, scaled by silu(t), then the patch projection
        let scale = lin(g.silu(t), "final_layer.adaLN_modulation.1")?;
        let xs = g.modulate(g.norm(xs, FINAL_EPS), scale, None);
        let nb = xs.strides()[1];
        let xs = g.cont(g.view_2d(xs, hp.dim, n_img, nb, n_cap_pad as usize * nb));
        let out = g.cont(g.scale(lin(xs, "final_layer.linear")?, -1.0));
        Ok(out)
    }

    /// One `JointTransformerBlock`; `t` modulates it (the refiners of the caption go without).
    fn block<'g>(
        &'g self,
        g: &'g Graph<'g>,
        p: &str,
        x: Tensor<'g>,
        t: Option<Tensor<'g>>,
        cos: Tensor<'g>,
        sin: Tensor<'g>,
    ) -> Result<Tensor<'g>> {
        let hp = &self.hp;
        let w = &self.dit;
        let wt = |s: &str| w.get(&format!("{p}.{s}.weight"));
        let rms = |x: Tensor<'g>, s: &str| -> Result<Tensor<'g>> {
            Ok(g.mul(g.rms_norm(x, NORM_EPS), wt(s)?))
        };
        // [scale_msa, gate_msa, scale_mlp, gate_mlp]
        let mods = match t {
            Some(t) => {
                let m = g.linear(
                    t,
                    wt("adaLN_modulation.0")?,
                    w.opt(&format!("{p}.adaLN_modulation.0.bias")),
                );
                let dim = hp.dim;
                let part =
                    |i: usize| g.cont(g.view_2d(m, dim, 1, m.strides()[1], i * dim as usize * 4));
                Some([part(0), g.tanh(part(1)), part(2), g.tanh(part(3))])
            }
            None => None,
        };
        let hd = hp.head_dim;
        let mut h = rms(x, "attention_norm1")?;
        if let Some(m) = &mods {
            h = g.modulate(h, m[0], None);
        }
        let (q, k, v) = if hp.fused_qkv {
            let qkv = g.linear(h, wt("attention.qkv")?, None);
            let nb = qkv.strides()[1];
            let es = qkv.strides()[0];
            let n = qkv.shape()[1];
            let (qd, kd) = (hp.heads * hd, hp.kv_heads * hd);
            (
                g.cont(g.view_2d(qkv, qd, n, nb, 0)),
                g.cont(g.view_2d(qkv, kd, n, nb, qd as usize * es)),
                g.cont(g.view_2d(qkv, kd, n, nb, (qd + kd) as usize * es)),
            )
        } else {
            (
                g.linear(h, wt("attention.to_q")?, None),
                g.linear(h, wt("attention.to_k")?, None),
                g.linear(h, wt("attention.to_v")?, None),
            )
        };
        let q = qwen3::head_rms(g, q, hd, QK_EPS, Some(wt("attention.q_norm")?));
        let k = qwen3::head_rms(g, k, hd, QK_EPS, Some(wt("attention.k_norm")?));
        // interleaved rotary pairs, applied as rotate-half on permuted heads
        let q = qwen3::rope(g, super::qwen_image::halves(g, q, hd), cos, sin, hp.heads);
        let k = qwen3::rope(
            g,
            super::qwen_image::halves(g, k, hd),
            cos,
            sin,
            hp.kv_heads,
        );
        let a = qwen3::sdpa_gqa(g, q, k, v, hp.heads, hp.kv_heads, None, self.flash);
        let mut a = rms(g.linear(a, wt("attention.out")?, None), "attention_norm2")?;
        if let Some(m) = &mods {
            a = g.mul(a, m[1]);
        }
        let x = g.add(x, a);
        let mut h = rms(x, "ffn_norm1")?;
        if let Some(m) = &mods {
            h = g.modulate(h, m[2], None);
        }
        let gate = g.silu(g.linear(h, wt("feed_forward.w1")?, None));
        let f = g.linear(
            g.mul(gate, g.linear(h, wt("feed_forward.w3")?, None)),
            wt("feed_forward.w2")?,
            None,
        );
        let mut f = rms(f, "ffn_norm2")?;
        if let Some(m) = &mods {
            f = g.mul(f, m[3]);
        }
        Ok(g.add(x, f))
    }

    pub fn generate(
        &mut self,
        be: &mut Backend,
        p: &GenParams,
        mut progress: impl FnMut(i32, i32) -> bool,
    ) -> Result<Output> {
        if p.n_frames > 1 || p.gen_audio {
            bail!("Z-Image generates images only");
        }
        if p.image.is_some() || p.video.is_some() || p.action.is_some() {
            bail!("Z-Image image editing is not supported");
        }
        let mult = VAE_SCALE * PATCH;
        let (w, h) = (p.width / mult * mult, p.height / mult * mult);
        if w <= 0 || h <= 0 {
            bail!("width and height must be at least {mult}");
        }
        let steps = if p.n_steps > 0 {
            p.n_steps
        } else {
            DEFAULT_STEPS
        };
        let cfg = if p.cfg_scale > 0.0 { p.cfg_scale } else { 1.0 };
        let use_cfg = cfg > 1.0;
        let mut prompts = vec![p.prompt.as_str()];
        if use_cfg {
            prompts.push(p.negative_prompt.as_str());
        }
        self.prepare(&prompts)?;

        let (lh, lw) = (h / VAE_SCALE, w / VAE_SCALE);
        let (ph, pw) = (lh / PATCH, lw / PATCH);
        let c = self.hp.in_channels;
        let seed = p.seed.unwrap_or_else(super::rand_seed);
        let mut noise = vec![0f32; (c * lh * lw) as usize];
        if !super::cosmos3::noise_from_env("MEDIAGEN_NOISE", &mut noise)? {
            let mut rng = Mt19937::new(seed);
            noise.iter_mut().for_each(|v| *v = rng.normal());
        }
        dump("noise", &noise);
        let mut x = patchify(&noise, c, lh, lw);
        let sig = sigmas(steps as usize);
        eprintln!(
            "[llmman] mediagen: generating {w}x{h}, {} tokens, {steps} steps, cfg {cfg:.1}, seed {seed}",
            ph * pw
        );
        let (cap, n_cap) = {
            let c = self.cond.as_ref().expect("prompt encoded");
            (&c.1, c.2)
        };
        for i in 0..steps as usize {
            let dt = sig[i + 1] - sig[i];
            if dt != 0.0 {
                let t0 = Instant::now();
                let mut v = self.velocity(be, cap, n_cap, &x, ph, pw, sig[i])?;
                if use_cfg {
                    let nc = self.ncond.as_ref().expect("negative prompt encoded");
                    let u = self.velocity(be, &nc.1, nc.2, &x, ph, pw, sig[i])?;
                    for (v, u) in v.iter_mut().zip(&u) {
                        *v = u + cfg * (*v - u);
                    }
                }
                if i == 0 {
                    dump("pred0", &v);
                }
                for (x, v) in x.iter_mut().zip(&v) {
                    *x += dt * v;
                }
                eprintln!(
                    "[llmman] mediagen: step {}/{steps} sigma {:.4} -> {:.4} ({:.2} s)",
                    i + 1,
                    sig[i],
                    sig[i + 1],
                    t0.elapsed().as_secs_f64()
                );
            }
            if !progress(i as i32 + 1, steps) {
                bail!("cancelled");
            }
        }
        be.release_compute()?;
        let mut z = unpatchify(&x, c, lh, lw);
        dump("latents", &z);
        super::cosmos3::noise_from_env("MEDIAGEN_LATENTS", &mut z)?;

        let t0 = Instant::now();
        let img = flux_vae::decode(&self.vae, be, &z, lh, lw, self.flash);
        be.release_compute()?;
        let img = img?;
        eprintln!(
            "[llmman] mediagen: decoded {}x{} in {:.2} s",
            img.width,
            img.height,
            t0.elapsed().as_secs_f64()
        );
        let to_u8 = |v: f32| (((v + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
        Ok(Output {
            width: img.width,
            height: img.height,
            n_frames: 1,
            fps: p.fps,
            rgb: img.rgb.iter().map(|&v| to_u8(v)).collect(),
            revised_prompt: p.prompt.clone(),
            seed,
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmas_are_shifted_and_end_at_zero() {
        let s = sigmas(9);
        assert_eq!(s.len(), 10);
        assert_eq!(s[0], 1.0);
        assert_eq!(s[8], 0.0);
        assert_eq!(s[9], 0.0);
        // shift 3 at s = 0.5: 1.5 / 2
        let mid = sigmas(3)[1];
        assert!((mid - 0.75).abs() < 1e-6);
    }

    #[test]
    fn patchify_roundtrips_with_channels_last() {
        let (c, h, w) = (3, 4, 6);
        let lat: Vec<f32> = (0..c * h * w).map(|v| v as f32).collect();
        let tok = patchify(&lat, c, h, w);
        // token 0, feature (py=0, px=1, c=2) is lat[c=2][y=0][x=1]
        assert_eq!(tok[(1 * c + 2) as usize], lat[(2 * h * w + 1) as usize]);
        assert_eq!(unpatchify(&tok, c, h, w), lat);
    }

    #[test]
    fn positions_follow_the_caption() {
        let p = positions(32, 2, 3, 32);
        assert_eq!(p[0], [1.0, 0.0, 0.0]);
        assert_eq!(p[31], [32.0, 0.0, 0.0]);
        assert_eq!(p[32], [33.0, 0.0, 0.0]);
        assert_eq!(p[32 + 5], [33.0, 1.0, 2.0]);
        assert_eq!(p[32 + 6], [0.0, 0.0, 0.0]);
        assert_eq!(p.len(), 64);
    }

    #[test]
    fn diffusers_names_map_to_the_original_ones() {
        assert_eq!(
            dit_name("layers.3.attention.norm_q.weight").as_deref(),
            Some("layers.3.attention.q_norm.weight")
        );
        assert_eq!(
            dit_name("model.diffusion_model.layers.0.attention.to_out.0.weight").as_deref(),
            Some("layers.0.attention.out.weight")
        );
    }
}
