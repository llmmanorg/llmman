//! FLUX.2 [klein] text-to-image from GGUFs (Black Forest Labs; the 4B and
//! 9B transformers share this code): double-stream then single-stream blocks
//! with modulation shared across blocks, no biases, SwiGLU MLPs, 4-axis RoPE
//! (theta 2000), a Qwen3 text encoder (hidden states 9, 18 and 27 side by
//! side over a 512-token padded prompt), and the FLUX.2 VAE (32 channels,
//! 2x2-packed to 128 and batch-normalized).

use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, bail, Result};

use super::backend::{Backend, Graph};
use super::cosmos3::dit as qwen3;
use super::ffi::{Api, Tensor};
use super::flux_vae;
use super::qwen3_text::Qwen3Text;
use super::weights::{Index, LoadOpts, Weights};
use super::{dump, GenParams, Mt19937, Output};

/// The distilled [klein] checkpoints: 4 steps, no guidance; the base ones
/// (`*-base-*` files) are sampled for longer with guidance.
const DISTILLED_STEPS: i32 = 4;
const BASE_STEPS: i32 = 50;
const BASE_CFG: f32 = 4.0;
pub const DEFAULT_SIZE: i64 = 1024;
/// Pixels per transformer token: the VAE's 8x, then 2x2 packing.
const SCALE: i64 = 16;
const THETA: f64 = 2000.0;
const AXES: [i64; 4] = [32, 32, 32, 32];
const EPS: f32 = 1e-6;
/// The prompt is padded (or cut) to this many tokens.
const PROMPT_TOKENS: usize = 512;
/// Hidden states the caption is made of, side by side.
const TEXT_STATES: [usize; 3] = [9, 18, 27];
/// Reference images are scaled down to at most this many pixels.
const MAX_REF_PIXELS: i64 = 1024 * 1024;
/// RoPE time index of reference `k`: `REF_INDEX_SCALE * (k + 1)`.
const REF_INDEX_SCALE: f64 = 10.0;
/// `AutoencoderKLFlux2.batch_norm_eps`.
const BN_EPS: f32 = 1e-4;

/// A FLUX.2 transformer: a `flux` GGUF with the shared modulation layers.
pub fn is_flux2(idx: &Index) -> bool {
    idx.metadata.get("general.architecture").map(String::as_str) == Some("flux")
        && idx
            .tensors
            .iter()
            .any(|d| d.name == "double_stream_modulation_img.lin.weight")
}

#[derive(Clone, Debug)]
pub struct Hparams {
    pub hidden: i64,
    pub heads: i64,
    pub head_dim: i64,
    pub double: i64,
    pub single: i64,
    pub mlp: i64,
    pub in_channels: i64,
    pub ctx_dim: i64,
}

impl Hparams {
    pub fn from_index(idx: &Index) -> Result<Hparams> {
        let ne = |n: &str| -> Result<Vec<i64>> {
            idx.tensors
                .iter()
                .find(|d| d.name == n)
                .map(|d| d.ne.clone())
                .ok_or_else(|| anyhow!("no {n}"))
        };
        let count = |prefix: &str| -> i64 {
            idx.tensors
                .iter()
                .filter_map(|d| {
                    d.name
                        .strip_prefix(prefix)?
                        .split('.')
                        .next()?
                        .parse::<i64>()
                        .ok()
                })
                .max()
                .map_or(0, |m| m + 1)
        };
        let img_in = ne("img_in.weight")?;
        let hidden = img_in[1];
        let head_dim = ne("double_blocks.0.img_attn.norm.key_norm.scale")?[0];
        let hp = Hparams {
            hidden,
            heads: hidden / head_dim,
            head_dim,
            double: count("double_blocks."),
            single: count("single_blocks."),
            mlp: ne("double_blocks.0.img_mlp.0.weight")?[1] / 2,
            in_channels: img_in[0],
            ctx_dim: ne("txt_in.weight")?[0],
        };
        if AXES.iter().sum::<i64>() != hp.head_dim || hp.double == 0 || hp.single == 0 {
            bail!("unsupported FLUX.2 transformer {hp:?}");
        }
        Ok(hp)
    }
}

/// `compute_empirical_mu` of the FLUX.2 sampler: the time shift for
/// `seq_len` image tokens and `steps` steps.
fn empirical_mu(seq_len: i64, steps: usize) -> f64 {
    let (a1, b1, a2, b2) = (8.73809524e-05, 1.89833333, 0.00016927, 0.45666666);
    let s = seq_len as f64;
    if seq_len > 4300 {
        return a2 * s + b2;
    }
    let m200 = a2 * s + b2;
    let m10 = a1 * s + b1;
    let a = (m200 - m10) / 190.0;
    a * steps as f64 + (m200 - 200.0 * a)
}

/// `[steps]` sigmas from 1, time-shifted by [`empirical_mu`], then 0.
fn sigmas(steps: usize, seq_len: i64) -> Vec<f32> {
    let mu = empirical_mu(seq_len, steps).exp();
    (0..steps)
        .map(|i| {
            let t = 1.0 - i as f64 / steps as f64;
            if t >= 1.0 {
                1.0
            } else {
                (mu / (mu + (1.0 / t - 1.0))) as f32
            }
        })
        .chain([0.0])
        .collect()
}

/// Positions `[t, h, w, l]`: caption token `l` at `(0, 0, 0, l)`, the
/// image patches at `(0, r, c, 0)`, reference `k`'s at `(10 (k + 1), r, c, 0)`.
fn positions(n_txt: i64, h: i64, w: i64, refs: &[(i64, i64)]) -> Vec<[f64; 4]> {
    let mut p: Vec<[f64; 4]> = (0..n_txt).map(|l| [0.0, 0.0, 0.0, l as f64]).collect();
    let grid = |p: &mut Vec<[f64; 4]>, t: f64, h: i64, w: i64| {
        for r in 0..h {
            for c in 0..w {
                p.push([t, r as f64, c as f64, 0.0]);
            }
        }
    };
    grid(&mut p, 0.0, h, w);
    for (k, &(rh, rw)) in refs.iter().enumerate() {
        grid(&mut p, REF_INDEX_SCALE * (k + 1) as f64, rh, rw);
    }
    p
}

/// `[C][H][W]` -> `[H*W][C]`.
fn to_tokens(lat: &[f32], c: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; lat.len()];
    for ci in 0..c {
        for i in 0..n {
            out[i * c + ci] = lat[ci * n + i];
        }
    }
    out
}

/// Packed latents `[4C][h][w]` (channel `c * 4 + py * 2 + px`) -> `[C][2h][2w]`.
fn unpack(z: &[f32], c4: usize, h: usize, w: usize) -> Vec<f32> {
    let c = c4 / 4;
    let (hh, ww) = (2 * h, 2 * w);
    let mut out = vec![0f32; z.len()];
    for ci in 0..c {
        for py in 0..2 {
            for px in 0..2 {
                let src = ci * 4 + py * 2 + px;
                for y in 0..h {
                    for x in 0..w {
                        out[(ci * hh + 2 * y + py) * ww + 2 * x + px] = z[(src * h + y) * w + x];
                    }
                }
            }
        }
    }
    out
}

/// The inverse of [`unpack`]: `[C][2h][2w]` -> `[4C][h][w]`.
fn pack(z: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let (hh, ww) = (2 * h, 2 * w);
    let mut out = vec![0f32; z.len()];
    for ci in 0..c {
        for py in 0..2 {
            for px in 0..2 {
                let dst = ci * 4 + py * 2 + px;
                for y in 0..h {
                    for x in 0..w {
                        out[(dst * h + y) * w + x] = z[(ci * hh + 2 * y + py) * ww + 2 * x + px];
                    }
                }
            }
        }
    }
    out
}

/// A reference image's size for encoding: within [`MAX_REF_PIXELS`],
/// sides rounded down to multiples of 16, aspect kept.
fn ref_size(w: i64, h: i64) -> (i64, i64) {
    let scale = ((MAX_REF_PIXELS as f64) / (w * h) as f64).sqrt().min(1.0);
    let fit = |v: i64| ((v as f64 * scale) as i64 / SCALE * SCALE).max(SCALE);
    (fit(w), fit(h))
}

pub struct Model {
    hp: Hparams,
    /// A non-distilled (`base`) checkpoint.
    base: bool,
    dit: Weights,
    te: Qwen3Text,
    vae: Weights,
    bn_mean: Vec<f32>,
    bn_std: Vec<f32>,
    flash: bool,
    /// Prompt -> caption `[PROMPT_TOKENS][ctx_dim]`.
    cond: Option<(String, Vec<f32>)>,
    ncond: Option<(String, Vec<f32>)>,
}

impl Model {
    pub fn load(
        api: &'static Api,
        be: &Backend,
        idx: &Index,
        dit_path: &Path,
        vae_path: Option<&Path>,
        text_path: Option<&Path>,
        flash: bool,
    ) -> Result<Model> {
        let t0 = Instant::now();
        let hp = Hparams::from_index(idx)?;
        let vae_path = vae_path.ok_or_else(|| anyhow!("FLUX.2 needs its VAE"))?;
        let text_path =
            text_path.ok_or_else(|| anyhow!("FLUX.2 needs its Qwen3 text encoder GGUF"))?;
        let dit = Weights::load(api, dit_path, be.weight_buft(), &LoadOpts::default())?;
        let vae = Weights::load(
            api,
            vae_path,
            be.weight_buft(),
            &LoadOpts {
                rename: Some(Box::new(flux_vae::tensor_name)),
                transform: None,
            },
        )?;
        let bn_mean = vae.read_f32("bn.running_mean")?;
        let bn_std: Vec<f32> = vae
            .read_f32("bn.running_var")?
            .iter()
            .map(|v| (v + BN_EPS).sqrt())
            .collect();
        if bn_mean.len() as i64 != hp.in_channels || bn_std.len() as i64 != hp.in_channels {
            bail!(
                "the VAE's batch norm has {} channels, the transformer {}",
                bn_mean.len(),
                hp.in_channels
            );
        }
        let te = Qwen3Text::load(api, be, text_path)?;
        if te.hp.hidden * TEXT_STATES.len() as i64 != hp.ctx_dim {
            bail!(
                "the text encoder's hidden size is {}, the transformer expects {} / {}",
                te.hp.hidden,
                hp.ctx_dim,
                TEXT_STATES.len()
            );
        }
        eprintln!(
            "[llmman] mediagen: flux2 model ready in {:.1} s ({}+{} blocks, hidden {}, text encoder {} layers)",
            t0.elapsed().as_secs_f64(),
            hp.double,
            hp.single,
            hp.hidden,
            te.hp.n_layers
        );
        let base = dit_path
            .file_name()
            .is_some_and(|n| n.to_string_lossy().to_lowercase().contains("base"));
        Ok(Model {
            hp,
            base,
            dit,
            te,
            vae,
            bn_mean,
            bn_std,
            flash,
            cond: None,
            ncond: None,
        })
    }

    pub fn default_steps(&self) -> i32 {
        if self.base {
            BASE_STEPS
        } else {
            DISTILLED_STEPS
        }
    }

    pub fn default_cfg(&self) -> f32 {
        if self.base {
            BASE_CFG
        } else {
            1.0
        }
    }

    /// The caption `[PROMPT_TOKENS][3 * hidden]`: the chat template (thinking
    /// off) padded with `<|endoftext|>`, hidden states 9, 18 and 27 per token.
    fn encode(&self, be: &Backend, prompt: &str) -> Result<Vec<f32>> {
        let text = format!(
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let mut ids = self.te.tokenize(&text)?;
        if ids.len() > PROMPT_TOKENS {
            eprintln!(
                "[llmman] mediagen: prompt has {} tokens, truncating to {PROMPT_TOKENS}",
                ids.len()
            );
            ids.truncate(PROMPT_TOKENS);
        }
        let n_real = ids.len();
        ids.resize(PROMPT_TOKENS, self.te.token("<|endoftext|>")?);
        let states = self.te.hidden_states(be, &ids, n_real, &TEXT_STATES)?;
        let h = self.te.hp.hidden as usize;
        let mut out = vec![0f32; PROMPT_TOKENS * h * states.len()];
        for t in 0..PROMPT_TOKENS {
            for (si, s) in states.iter().enumerate() {
                out[(t * states.len() + si) * h..][..h].copy_from_slice(&s[t * h..(t + 1) * h]);
            }
        }
        Ok(out)
    }

    fn prepare(&mut self, be: &mut Backend, prompts: &[&str]) -> Result<()> {
        for (i, &p) in prompts.iter().enumerate() {
            let slot = if i == 0 { &self.cond } else { &self.ncond };
            if matches!(slot, Some((q, _)) if q == p) {
                continue;
            }
            let t0 = Instant::now();
            let c = self.encode(be, p)?;
            be.release_compute()?;
            dump(
                if i == 0 {
                    "text_hidden"
                } else {
                    "ntext_hidden"
                },
                &c,
            );
            eprintln!(
                "[llmman] mediagen: encoded {} prompt in {:.2} s",
                if i == 0 { "positive" } else { "negative" },
                t0.elapsed().as_secs_f64()
            );
            if i == 0 {
                self.cond = Some((p.to_string(), c));
            } else {
                self.ncond = Some((p.to_string(), c));
            }
        }
        Ok(())
    }

    /// The velocity of image tokens `x: [h*w][in_channels]` at `sigma`.
    /// `refs`: the reference images' tokens `[h*w][in_channels]` with their `(h, w)`.
    #[allow(clippy::too_many_arguments)]
    fn velocity(
        &self,
        be: &Backend,
        cap: &[f32],
        x: &[f32],
        h: i64,
        w: i64,
        refs: &[(Vec<f32>, i64, i64)],
        sigma: f32,
    ) -> Result<Vec<f32>> {
        let hp = &self.hp;
        let gr = Graph::new(be.api, 8192 + (hp.double + hp.single) as usize * 160)?;
        let out = self.forward(&gr, cap, x, h, w, refs, sigma)?;
        gr.mark_output(out);
        gr.compute(be)?;
        Ok(gr.output_f32(out))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward<'g>(
        &'g self,
        g: &'g Graph<'g>,
        cap: &[f32],
        x: &[f32],
        h: i64,
        w: i64,
        refs: &[(Vec<f32>, i64, i64)],
        sigma: f32,
    ) -> Result<Tensor<'g>> {
        let hp = &self.hp;
        let wt = &self.dit;
        let n_txt = PROMPT_TOKENS as i64;
        let n_img = h * w;
        // the image tokens, then every reference's: one stream through the blocks
        let mut tokens = x.to_vec();
        for (t, _, _) in refs {
            tokens.extend_from_slice(t);
        }
        let n_all = tokens.len() as i64 / hp.in_channels;
        let x_t = g.input_f32(&[hp.in_channels, n_all], &tokens, "patches");
        let cap_t = g.input_f32(&[hp.ctx_dim, n_txt], cap, "caption");
        let temb = g.input_f32(&[256], &qwen3::timestep_embedding(1000.0 * sigma), "temb");
        let ref_dims: Vec<(i64, i64)> = refs.iter().map(|&(_, rh, rw)| (rh, rw)).collect();
        let (cos, sin) =
            super::zimage::rope_tables(&AXES, THETA, &positions(n_txt, h, w, &ref_dims));
        let half = hp.head_dim / 2;
        let cos_t = g.input_f32(&[half, n_txt + n_all], &cos, "cos");
        let sin_t = g.input_f32(&[half, n_txt + n_all], &sin, "sin");

        let lin = |x: Tensor<'g>, name: &str| -> Result<Tensor<'g>> {
            Ok(g.linear(
                x,
                wt.get(&format!("{name}.weight"))?,
                wt.opt(&format!("{name}.bias")),
            ))
        };
        let vec = lin(
            g.silu(lin(g.reshape_2d(temb, 256, 1), "time_in.in_layer")?),
            "time_in.out_layer",
        )?;
        let sv = g.silu(vec);
        // the shared modulations, `[shift, scale, gate]` per stage
        let chunks = |name: &str, n: usize| -> Result<Vec<Tensor<'g>>> {
            let m = lin(sv, name)?;
            let nb = m.strides()[1];
            Ok((0..n)
                .map(|i| g.cont(g.view_2d(m, hp.hidden, 1, nb, i * hp.hidden as usize * 4)))
                .collect())
        };
        let dimg = chunks("double_stream_modulation_img.lin", 6)?;
        let dtxt = chunks("double_stream_modulation_txt.lin", 6)?;
        let ss = chunks("single_stream_modulation.lin", 3)?;

        let mut img = lin(x_t, "img_in")?;
        let mut txt = lin(cap_t, "txt_in")?;
        let ln = |x: Tensor<'g>| g.norm(x, EPS);
        for b in 0..hp.double {
            let p = format!("double_blocks.{b}");
            let (iq, ik, iv) = self.qkv(
                g,
                &format!("{p}.img_attn"),
                g.modulate(ln(img), dimg[1], Some(dimg[0])),
            )?;
            let (tq, tk, tv) = self.qkv(
                g,
                &format!("{p}.txt_attn"),
                g.modulate(ln(txt), dtxt[1], Some(dtxt[0])),
            )?;
            let q = g.concat(tq, iq, 1);
            let k = g.concat(tk, ik, 1);
            let v = g.concat(tv, iv, 1);
            let a = self.attend(g, q, k, v, cos_t, sin_t);
            let nb = a.strides()[1];
            let ta = g.cont(g.view_2d(a, hp.hidden, n_txt, nb, 0));
            let ia = g.cont(g.view_2d(a, hp.hidden, n_all, nb, n_txt as usize * nb));
            img = g.add(img, g.mul(lin(ia, &format!("{p}.img_attn.proj"))?, dimg[2]));
            let m = self.mlp(
                g,
                &format!("{p}.img_mlp"),
                g.modulate(ln(img), dimg[4], Some(dimg[3])),
            )?;
            img = g.add(img, g.mul(m, dimg[5]));
            txt = g.add(txt, g.mul(lin(ta, &format!("{p}.txt_attn.proj"))?, dtxt[2]));
            let m = self.mlp(
                g,
                &format!("{p}.txt_mlp"),
                g.modulate(ln(txt), dtxt[4], Some(dtxt[3])),
            )?;
            txt = g.add(txt, g.mul(m, dtxt[5]));
        }
        let mut xs = g.concat(txt, img, 1);
        let n = n_txt + n_all;
        let (hd, hid, mlp) = (hp.hidden, hp.hidden as usize, hp.mlp);
        for b in 0..hp.single {
            let p = format!("single_blocks.{b}");
            let y = lin(
                g.modulate(ln(xs), ss[1], Some(ss[0])),
                &format!("{p}.linear1"),
            )?;
            let nb = y.strides()[1];
            let es = y.strides()[0];
            let part = |off: usize, len: i64| g.cont(g.view_2d(y, len, n, nb, off * es));
            let (q, k, v) = (part(0, hd), part(hid, hd), part(2 * hid, hd));
            let q = qwen3::head_rms(
                g,
                q,
                hp.head_dim,
                EPS,
                Some(wt.get(&format!("{p}.norm.query_norm.scale"))?),
            );
            let k = qwen3::head_rms(
                g,
                k,
                hp.head_dim,
                EPS,
                Some(wt.get(&format!("{p}.norm.key_norm.scale"))?),
            );
            let a = self.attend(g, q, k, v, cos_t, sin_t);
            let m = swiglu(g, part(3 * hid, 2 * mlp));
            let o = lin(g.concat(a, m, 0), &format!("{p}.linear2"))?;
            xs = g.add(xs, g.mul(o, ss[2]));
        }
        let nb = xs.strides()[1];
        let img = g.cont(g.view_2d(xs, hp.hidden, n_img, nb, n_txt as usize * nb));
        let m = lin(g.silu(vec), "final_layer.adaLN_modulation.1")?;
        let mb = m.strides()[1];
        let shift = g.cont(g.view_2d(m, hp.hidden, 1, mb, 0));
        let scale = g.cont(g.view_2d(m, hp.hidden, 1, mb, hp.hidden as usize * 4));
        Ok(g.cont(lin(
            g.modulate(ln(img), scale, Some(shift)),
            "final_layer.linear",
        )?))
    }

    /// A reference image as transformer tokens `[h*w][in_channels]` with
    /// `(h, w)`: resized, encoded (the posterior's mean), packed 2x2 and
    /// batch-normalized, as the sampled latents are before decoding.
    fn reference(&self, be: &Backend, f: &super::Frames) -> Result<(Vec<f32>, i64, i64)> {
        let t0 = Instant::now();
        let (w, h) = ref_size(f.width, f.height);
        let img = image::RgbImage::from_raw(f.width as u32, f.height as u32, f.rgb.clone())
            .ok_or_else(|| anyhow!("reference image: bad pixel buffer"))?;
        let img = if (w, h) == (f.width, f.height) {
            img
        } else {
            image::imageops::resize(
                &img,
                w as u32,
                h as u32,
                image::imageops::FilterType::Lanczos3,
            )
        };
        let z = flux_vae::encode(&self.vae, be, img.as_raw(), h, w)?;
        let (lh, lw) = (h / SCALE, w / SCALE);
        let c = self.hp.in_channels as usize;
        let n = (lh * lw) as usize;
        let mut z = pack(&z, c / 4, lh as usize, lw as usize);
        for (ci, row) in z.chunks_mut(n).enumerate() {
            let (m, s) = (self.bn_mean[ci], self.bn_std[ci]);
            row.iter_mut().for_each(|v| *v = (*v - m) / s);
        }
        eprintln!(
            "[llmman] mediagen: encoded a {}x{} reference as {w}x{h} in {:.2} s",
            f.width,
            f.height,
            t0.elapsed().as_secs_f64()
        );
        Ok((to_tokens(&z, c, n), lh, lw))
    }

    /// `SelfAttention.pre_attention`: q, k (RMS-normed per head) and v.
    fn qkv<'g>(
        &'g self,
        g: &'g Graph<'g>,
        p: &str,
        x: Tensor<'g>,
    ) -> Result<(Tensor<'g>, Tensor<'g>, Tensor<'g>)> {
        let hp = &self.hp;
        let wt = &self.dit;
        let y = g.linear(
            x,
            wt.get(&format!("{p}.qkv.weight"))?,
            wt.opt(&format!("{p}.qkv.bias")),
        );
        let n = y.shape()[1];
        let nb = y.strides()[1];
        let es = y.strides()[0];
        let part = |i: usize| g.cont(g.view_2d(y, hp.hidden, n, nb, i * hp.hidden as usize * es));
        let q = qwen3::head_rms(
            g,
            part(0),
            hp.head_dim,
            EPS,
            Some(wt.get(&format!("{p}.norm.query_norm.scale"))?),
        );
        let k = qwen3::head_rms(
            g,
            part(1),
            hp.head_dim,
            EPS,
            Some(wt.get(&format!("{p}.norm.key_norm.scale"))?),
        );
        Ok((q, k, part(2)))
    }

    /// RoPE (interleaved pairs, as rotate-half on permuted heads) and attention.
    fn attend<'g>(
        &'g self,
        g: &'g Graph<'g>,
        q: Tensor<'g>,
        k: Tensor<'g>,
        v: Tensor<'g>,
        cos: Tensor<'g>,
        sin: Tensor<'g>,
    ) -> Tensor<'g> {
        let (hd, heads) = (self.hp.head_dim, self.hp.heads);
        let q = qwen3::rope(g, super::qwen_image::halves(g, q, hd), cos, sin, heads);
        let k = qwen3::rope(g, super::qwen_image::halves(g, k, hd), cos, sin, heads);
        qwen3::sdpa_gqa(g, q, k, v, heads, heads, None, self.flash)
    }

    fn mlp<'g>(&'g self, g: &'g Graph<'g>, p: &str, x: Tensor<'g>) -> Result<Tensor<'g>> {
        let wt = &self.dit;
        let y = g.linear(
            x,
            wt.get(&format!("{p}.0.weight"))?,
            wt.opt(&format!("{p}.0.bias")),
        );
        Ok(g.linear(
            swiglu(g, y),
            wt.get(&format!("{p}.2.weight"))?,
            wt.opt(&format!("{p}.2.bias")),
        ))
    }

    pub fn generate(
        &mut self,
        be: &mut Backend,
        p: &GenParams,
        mut progress: impl FnMut(i32, i32) -> bool,
    ) -> Result<Output> {
        if p.n_frames > 1 || p.gen_audio {
            bail!("FLUX.2 generates images only");
        }
        if p.image.is_some() || p.video.is_some() || p.action.is_some() {
            bail!("FLUX.2 takes reference images (\"images\"), not video conditioning");
        }
        let (w, h) = (p.width / SCALE * SCALE, p.height / SCALE * SCALE);
        if w <= 0 || h <= 0 {
            bail!("width and height must be at least {SCALE}");
        }
        let steps = if p.n_steps > 0 {
            p.n_steps
        } else {
            self.default_steps()
        };
        let cfg = if p.cfg_scale > 0.0 {
            p.cfg_scale
        } else {
            self.default_cfg()
        };
        let use_cfg = cfg > 1.0;
        let mut prompts = vec![p.prompt.as_str()];
        if use_cfg {
            prompts.push(p.negative_prompt.as_str());
        }
        self.prepare(be, &prompts)?;
        let refs = p
            .ref_images
            .iter()
            .map(|f| self.reference(be, f))
            .collect::<Result<Vec<_>>>()?;
        be.release_compute()?;

        let (lh, lw) = (h / SCALE, w / SCALE);
        let n = (lh * lw) as usize;
        let c = self.hp.in_channels as usize;
        let seed = p.seed.unwrap_or_else(super::rand_seed);
        let mut noise = vec![0f32; c * n];
        if !super::cosmos3::noise_from_env("MEDIAGEN_NOISE", &mut noise)? {
            let mut rng = Mt19937::new(seed);
            noise.iter_mut().for_each(|v| *v = rng.normal());
        }
        dump("noise", &noise);
        let mut x = to_tokens(&noise, c, n);
        let sig = sigmas(steps as usize, n as i64);
        eprintln!("[llmman] mediagen: generating {w}x{h}, {n} tokens, {steps} steps, cfg {cfg:.1}, seed {seed}");
        let cap = &self.cond.as_ref().expect("prompt encoded").1;
        for i in 0..steps as usize {
            let t0 = Instant::now();
            let mut v = self.velocity(be, cap, &x, lh, lw, &refs, sig[i])?;
            if use_cfg {
                let nc = &self.ncond.as_ref().expect("negative prompt encoded").1;
                let u = self.velocity(be, nc, &x, lh, lw, &refs, sig[i])?;
                for (v, u) in v.iter_mut().zip(&u) {
                    *v = u + cfg * (*v - u);
                }
            }
            if i == 0 {
                dump("pred0", &v);
            }
            if v.iter().any(|v| !v.is_finite()) {
                bail!(
                    "the FLUX.2 transformer produced non-finite values at step {}",
                    i + 1
                );
            }
            let dt = sig[i + 1] - sig[i];
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
            if !progress(i as i32 + 1, steps) {
                bail!("cancelled");
            }
        }
        be.release_compute()?;
        // tokens -> [C][h][w], undo the batch norm, unpack 2x2
        let mut z = to_tokens(&x, n, c);
        dump("latents", &z);
        super::cosmos3::noise_from_env("MEDIAGEN_LATENTS", &mut z)?;
        for (ci, row) in z.chunks_mut(n).enumerate() {
            let (m, s) = (self.bn_mean[ci], self.bn_std[ci]);
            row.iter_mut().for_each(|v| *v = *v * s + m);
        }
        let z = unpack(&z, c, lh as usize, lw as usize);
        let t0 = Instant::now();
        let img = flux_vae::decode(&self.vae, be, &z, 2 * lh, 2 * lw);
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

/// `silu(x[..half]) * x[half..]` over the rows of `x: [2m, T]`.
fn swiglu<'g>(g: &'g Graph<'g>, x: Tensor<'g>) -> Tensor<'g> {
    let [d, t, ..] = x.shape();
    let nb = x.strides()[1];
    let es = x.strides()[0];
    let gate = g.view_2d(x, d / 2, t, nb, 0);
    let up = g.view_2d(x, d / 2, t, nb, (d / 2) as usize * es);
    g.mul(g.silu(g.cont(gate)), g.cont(up))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmas_follow_the_empirical_shift() {
        let s = sigmas(4, 4096);
        assert_eq!(s.len(), 5);
        assert_eq!(s[0], 1.0);
        assert_eq!(s[4], 0.0);
        // decreasing, and above the unshifted grid (mu > 0)
        assert!(s.windows(2).all(|w| w[0] > w[1]));
        assert!(s[2] > 0.5);
    }

    #[test]
    fn positions_put_the_caption_on_the_last_axis() {
        let p = positions(2, 2, 3, &[(1, 2), (1, 1)]);
        assert_eq!(p[1], [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(p[2], [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(p[2 + 5], [0.0, 1.0, 2.0, 0.0]);
        // references follow, at time 10, 20, ...
        assert_eq!(p[8], [10.0, 0.0, 0.0, 0.0]);
        assert_eq!(p[9], [10.0, 0.0, 1.0, 0.0]);
        assert_eq!(p[10], [20.0, 0.0, 0.0, 0.0]);
        assert_eq!(p.len(), 11);
    }

    #[test]
    fn unpack_inverts_the_2x2_packing() {
        // channel 1 of the packed latents is (c 0, py 0, px 1)
        let (c4, h, w) = (4, 1, 2);
        let z: Vec<f32> = (0..c4 * h * w).map(|v| v as f32).collect();
        let out = unpack(&z, c4, h, w);
        // out is [1][2][4]: row 0 = (py 0) px 0/1 of x 0, then x 1
        assert_eq!(out, [0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]);
        assert_eq!(pack(&out, c4 / 4, h, w), z);
    }

    #[test]
    fn references_fit_the_pixel_budget_in_multiples_of_16() {
        assert_eq!(ref_size(512, 512), (512, 512));
        assert_eq!(ref_size(500, 300), (496, 288));
        let (w, h) = ref_size(4000, 3000);
        assert!(w * h <= MAX_REF_PIXELS && w % 16 == 0 && h % 16 == 0);
        assert_eq!(ref_size(8, 8), (16, 16));
    }
}
