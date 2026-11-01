//! The Canary FastConformer encoder (NeMo `ConformerEncoder` with
//! `dw_striding` subsampling and relative positional attention) as a
//! ggml graph on [`Backend`], plus the host-side arithmetic the graph
//! consumes as inputs: the subsampled frame count sizes every tensor,
//! the relative-position table feeds `linear_pos`, and the folded
//! BatchNorm replaces the four `conv.bn.*` tensors of each block with a
//! per-channel scale and shift. The host values are computed in `f64`
//! and cast once, so they do not depend on the backend. [`Encoder::run`]
//! builds one graph per call following llama.cpp's `parakeet.cpp`
//! recipe: the subsampler, then the 32 blocks, each a half-step
//! feed-forward, relative-position self-attention, the conv module, a
//! second half-step feed-forward and a LayerNorm.

use std::ffi::c_int;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{anyhow, bail, Result};

use super::model::{CanaryModel, EncBlock};
use crate::mediagen::backend::{Backend, Ctx, Graph};
use crate::mediagen::ffi::{self, ty, Tensor};

/// `nn.BatchNorm1d` default `eps`, the value NeMo's conv module uses.
const BN_EPS: f32 = 1e-5;

/// `nn.LayerNorm` default `eps`, used by every norm of a block.
const LN_EPS: f32 = 1e-5;

/// Intermediate tensor [`Encoder::run`] returns next to its output, in
/// the order requested; each is frame-major `[rows, d_model]` f32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Probe {
    /// The subsampler's linear output, `[t_enc, d_model]`
    /// (NeMo `enc.pre_encode.out`).
    PreEncode,
    /// The relative-position table, `[2 * t_enc - 1, d_model]`
    /// (NeMo `enc.pos_emb`).
    PosEmb,
    /// Output of block `i` after its `norm_out`, `[t_enc, d_model]`.
    Block(usize),
}

/// One encoder pass.
pub struct Run {
    /// Frame-major `[frames, d_model]`: the last block's `norm_out`
    /// output, which is NeMo's `enc.final` (Canary 1B v2 has no
    /// `enc.proj`).
    pub out: Vec<f32>,
    /// Encoder frames, `subsampled_len` of the mel frames.
    pub frames: usize,
    /// One buffer per requested [`Probe`], in request order.
    pub probes: Vec<Vec<f32>>,
}

/// The encoder's host-side state for one loaded model: hyperparameters
/// checked once and the BatchNorm of every block folded once. Weights
/// stay in the [`CanaryModel`] the caller passes to [`Encoder::run`].
pub struct Encoder {
    d_model: usize,
    n_heads: usize,
    n_mels: usize,
    /// `(scale, shift)` per block from [`fold_batch_norm`], applied by
    /// the conv module after the depthwise bias.
    bn: Vec<(Vec<f32>, Vec<f32>)>,
    /// Whether the compute buffer size was logged already (once per
    /// encoder, on the first run).
    logged: AtomicBool,
}

impl Encoder {
    /// Checks the hyperparameters this graph is written for and folds
    /// each block's BatchNorm, reading the four `conv.bn.*` vectors back
    /// from the backend.
    pub fn new(model: &CanaryModel, be: &Backend) -> Result<Encoder> {
        let hp = &model.hp;
        if hp.enc_subsampling_factor != 8 {
            bail!(
                "encoder: subsampling factor {} (the dw_striding graph is 8x)",
                hp.enc_subsampling_factor
            );
        }
        if hp.frontend.num_mels != 128 {
            bail!("encoder: {} mel bins (expected 128)", hp.frontend.num_mels);
        }
        if !hp.enc_d_model.is_multiple_of(hp.enc_n_heads) {
            bail!(
                "encoder: d_model {} is not a multiple of {} heads",
                hp.enc_d_model,
                hp.enc_n_heads
            );
        }
        if be.api.ggml_prec_set_acc.is_none() {
            bail!(
                "encoder: this llama.cpp release has no ggml_prec_set_acc; use the pinned release"
            );
        }
        check_conv_kernel(hp.enc_conv_kernel)?;
        if !hp.enc_d_model.is_multiple_of(2) {
            bail!(
                "encoder: d_model {} is odd; the interleaved position table needs an even width",
                hp.enc_d_model
            );
        }
        let bn = model
            .tensors
            .enc_blocks
            .iter()
            .map(|b| {
                fold_batch_norm(
                    &be.read_f32(b.bn),
                    &be.read_f32(b.bn_b),
                    &be.read_f32(b.bn_mean),
                    &be.read_f32(b.bn_var),
                    BN_EPS,
                )
            })
            .collect();
        Ok(Encoder {
            d_model: hp.enc_d_model as usize,
            n_heads: hp.enc_n_heads as usize,
            n_mels: hp.frontend.num_mels as usize,
            bn,
            logged: AtomicBool::new(false),
        })
    }

    /// Runs the encoder on `mel`, frame-major `[frames, n_mels]` f32 as
    /// `speech::mel::features` returns, and reads back `out` and the
    /// requested `probes`. One graph is built per call.
    ///
    /// The subsampler follows `parakeet.cpp` 14-61: the mel is the
    /// one-channel image ggml `[n_mels, frames, 1, 1]` (frame-major data
    /// is already `[W = mel, H = time]`), then conv.0 3x3 stride 2 plus
    /// ReLU, conv.2 depthwise 3x3 stride 2, conv.3 1x1 plus ReLU, conv.5
    /// depthwise 3x3 stride 2, conv.6 1x1 plus ReLU, each with its bias
    /// broadcast over the channel axis; the `[mel, time, chan]` result is
    /// permuted to `[mel, chan, time]` so the flatten is channel-major
    /// then mel as NeMo's `x.transpose(1, 2).reshape(b, t, -1)`, and the
    /// `out` linear maps the 4096 values of each frame onto `d_model`.
    /// The relative-position table is built on the host and uploaded as
    /// `[d_model, 2 * t_enc - 1]`. Each block then runs [`feed_forward`],
    /// [`attention`], [`conv_module`], [`feed_forward`] and its
    /// `norm_out` on the `[d_model, t_enc]` activations, with that
    /// block's folded BatchNorm uploaded as two `[d_model]` inputs.
    pub fn run(
        &self,
        model: &CanaryModel,
        be: &Backend,
        mel: &[f32],
        probes: &[Probe],
    ) -> Result<Run> {
        if mel.is_empty() || !mel.len().is_multiple_of(self.n_mels) {
            bail!(
                "encoder: {} mel values are not a whole number of {}-bin frames",
                mel.len(),
                self.n_mels
            );
        }
        let frames = mel.len() / self.n_mels;
        let t_enc = subsampled_len(frames);
        let pre = &model.tensors.pre_encode;

        let mut gr = Graph::new(be.api, 1024 + model.tensors.enc_blocks.len() * 128)?;
        let x = gr.input_f32(&[self.n_mels as i64, frames as i64, 1, 1], mel, "mel");
        let pos = gr.input_f32(
            &[self.d_model as i64, 2 * t_enc as i64 - 1],
            &pos_table(t_enc, self.d_model),
            "pos_emb",
        );
        let bn: Vec<(Tensor, Tensor)> = self
            .bn
            .iter()
            .enumerate()
            .map(|(i, (scale, shift))| {
                let n = &[self.d_model as i64];
                (
                    gr.input_f32(n, scale, &format!("bn_scale.{i}")),
                    gr.input_f32(n, shift, &format!("bn_shift.{i}")),
                )
            })
            .collect();
        let g = &gr.ctx;
        let bias = |b: Tensor| g.reshape_4d(b, 1, 1, ffi::shape(b)[0], 1);
        let mut cur = g.conv_2d(pre.conv0, x, 2, 2, 1, 1, 1, 1);
        cur = g.relu(g.add(cur, bias(pre.conv0_b)));
        cur = g.conv_2d_dw_direct(pre.conv2, cur, 2, 2, 1, 1, 1, 1);
        cur = g.add(cur, bias(pre.conv2_b));
        cur = g.conv_2d(pre.conv3, cur, 1, 1, 0, 0, 1, 1);
        cur = g.relu(g.add(cur, bias(pre.conv3_b)));
        cur = g.conv_2d_dw_direct(pre.conv5, cur, 2, 2, 1, 1, 1, 1);
        cur = g.add(cur, bias(pre.conv5_b));
        cur = g.conv_2d(pre.conv6, cur, 1, 1, 0, 0, 1, 1);
        cur = g.relu(g.add(cur, bias(pre.conv6_b)));
        // [mel, time, chan] -> [mel, chan, time]
        cur = g.cont(g.permute(cur, 0, 2, 1, 3));
        let [n_freq, n_chan, n_time, _] = ffi::shape(cur);
        let in_dim = ffi::shape(pre.out)[0];
        if n_time != t_enc as i64 || n_freq * n_chan != in_dim {
            bail!(
                "encoder: subsampler produced [{n_freq}, {n_chan}, {n_time}] for {frames} frames, \
                 expected {in_dim} values per frame over {t_enc} frames"
            );
        }
        cur = g.reshape_2d(cur, in_dim, n_time);
        let pre_out = g.linear(cur, pre.out, Some(pre.out_b));

        let mut blocks = Vec::with_capacity(model.tensors.enc_blocks.len());
        cur = pre_out;
        for (b, &(bn_scale, bn_shift)) in model.tensors.enc_blocks.iter().zip(&bn) {
            cur = feed_forward(
                g,
                cur,
                [b.norm_ff1, b.norm_ff1_b],
                [b.ff1_linear1, b.ff1_linear1_b],
                [b.ff1_linear2, b.ff1_linear2_b],
            );
            cur = attention(g, b, cur, pos, self.n_heads as i64);
            cur = conv_module(g, b, cur, bn_scale, bn_shift);
            cur = feed_forward(
                g,
                cur,
                [b.norm_ff2, b.norm_ff2_b],
                [b.ff2_linear1, b.ff2_linear1_b],
                [b.ff2_linear2, b.ff2_linear2_b],
            );
            cur = layer_norm(g, cur, b.norm_out, b.norm_out_b);
            blocks.push(cur);
        }
        let out = cur;
        gr.mark_output(out);
        let probe_tensors = probes
            .iter()
            .map(|p| match p {
                Probe::PreEncode => Ok(pre_out),
                Probe::PosEmb => Ok(pos),
                Probe::Block(i) => blocks.get(*i).copied().ok_or_else(|| {
                    anyhow!(
                        "encoder: block {i} probe requested of {} blocks",
                        blocks.len()
                    )
                }),
            })
            .collect::<Result<Vec<Tensor>>>()?;
        for &t in &probe_tensors {
            gr.mark_output(t);
        }

        if !self.logged.swap(true, Ordering::Relaxed) {
            let bytes = gr.measure(be)?;
            eprintln!("[llmman] speech: encoder compute buffer {bytes} bytes for {frames} frames");
        }
        gr.compute(be)?;
        Ok(Run {
            out: gr.output_f32(out),
            frames: t_enc,
            probes: probe_tensors.iter().map(|&t| gr.output_f32(t)).collect(),
        })
    }
}

/// `nn.LayerNorm` over `d_model`: `norm(x) * w + b` on `[d_model, t]`.
fn layer_norm(g: &Ctx, x: Tensor, w: Tensor, b: Tensor) -> Tensor {
    g.add(g.mul(g.norm(x, LN_EPS), w), b)
}

/// Macaron half-step feed-forward with its residual:
/// `x + 0.5 * linear2(silu(linear1(norm(x))))`. Each pair is
/// `[weight, bias]`.
fn feed_forward(
    g: &Ctx,
    x: Tensor,
    norm: [Tensor; 2],
    linear1: [Tensor; 2],
    linear2: [Tensor; 2],
) -> Tensor {
    let h = layer_norm(g, x, norm[0], norm[1]);
    let h = g.silu(g.linear(h, linear1[0], Some(linear1[1])));
    let h = g.linear(h, linear2[0], Some(linear2[1]));
    g.add(x, g.scale(h, 0.5))
}

/// NeMo `RelPositionMultiHeadAttention` with its residual, after
/// `parakeet.cpp`'s full-attention branch. `x` is `[d_model, t]` and
/// `pos` the `[d_model, 2t - 1]` table of [`pos_table`], whose row `j`
/// holds relative position `t - 1 - j`.
///
/// Content scores `ac[k, q]` come from `k` against `q + pos_bias_u`;
/// position scores `bd[j, q]` from the projected table against
/// `q + pos_bias_v`. NeMo's `rel_shift` (pad a zero column, view as
/// `[t, 2t]`, drop the first row, view back, keep `t` columns) selects
/// `bd[t - 1 - q + k, q]`, the row of relative position `q - k`; that
/// is one strided view of `bd` with row stride `2t - 2` from offset
/// `t - 1`, so the zero column is never read and is not built. The
/// scores are summed, scaled by `1 / sqrt(d_head)` and softmaxed over
/// the keys; no mask, batch 1 attends to every frame.
fn attention(g: &Ctx, b: &EncBlock, x: Tensor, pos: Tensor, n_heads: i64) -> Tensor {
    let [d_model, t, _, _] = ffi::shape(x);
    let d_head = d_model / n_heads;
    let n_pos = ffi::shape(pos)[1];
    let h = layer_norm(g, x, b.norm_attn, b.norm_attn_b);
    // [d_head, head, t]
    let heads = |y: Tensor| g.reshape_3d(y, d_head, n_heads, t);
    let q = heads(g.linear(h, b.attn_q, Some(b.attn_q_b)));
    let k = heads(g.linear(h, b.attn_k, Some(b.attn_k_b)));
    let v = heads(g.linear(h, b.attn_v, Some(b.attn_v_b)));
    // [d_head, t, head]; the `[d_head, head]` biases broadcast over t
    let k = g.permute(k, 0, 2, 1, 3);
    let q_u = g.permute(g.add(q, b.pos_bias_u), 0, 2, 1, 3);
    let q_v = g.cont(g.permute(g.add(q, b.pos_bias_v), 0, 2, 1, 3));
    // [d_head, 2t - 1, head]
    let p = g.reshape_3d(g.mul_mat(b.attn_pos, pos), d_head, n_heads, n_pos);
    let p = g.cont(g.permute(p, 0, 2, 1, 3));
    // [t_k, t_q, head] and [2t - 1, t_q, head]
    let ac = g.prec_f32(g.mul_mat(k, q_u));
    let bd = g.prec_f32(g.mul_mat(p, q_v));
    let nb = ffi::strides(bd);
    let shifted = g.view_3d(
        bd,
        t,
        t,
        n_heads,
        (n_pos - 1) as usize * nb[0],
        nb[2],
        (t - 1) as usize * nb[0],
    );
    let scores = g.add(ac, g.cont(shifted));
    let probs = g.soft_max_ext(scores, ptr::null_mut(), 1.0 / (d_head as f32).sqrt(), 0.0);
    // [t_k, d_head, head] so that probs . v gives [t_q, d_head, head]
    let v = g.cont(g.permute(v, 1, 2, 0, 3));
    let o = g.prec_f32(g.mul_mat(probs, v));
    // [d_head, head, t_q] flattened head-major, as NeMo's merge
    let o = g.cont_2d(g.permute(o, 2, 0, 1, 3), d_model, t);
    g.add(x, g.linear(o, b.attn_out, Some(b.attn_out_b)))
}

/// NeMo `ConformerConvolution` with its residual, after `parakeet.cpp`
/// 342-388: pointwise1 as a linear to `2 * d_model`, GLU over the
/// channel axis (`signal * sigmoid(gate)` on two views), the depthwise
/// kernel-`k` conv as `ggml_ssm_conv` on the `[t + k - 1, d_model]`
/// zero-padded transpose, its bias, the folded BatchNorm `scale` and
/// `shift`, SiLU and pointwise2. `ggml_ssm_conv` reads an F32 kernel,
/// so an F16 `conv.depthwise.weight` is cast in the graph.
fn conv_module(g: &Ctx, b: &EncBlock, x: Tensor, bn_scale: Tensor, bn_shift: Tensor) -> Tensor {
    let [d_model, t, _, _] = ffi::shape(x);
    let h = layer_norm(g, x, b.norm_conv, b.norm_conv_b);
    let [_, pw_in, pw_out, _] = ffi::shape(b.pointwise1);
    let pw1 = g.reshape_2d(b.pointwise1, pw_in, pw_out);
    // [2 * d_model, t]
    let h = g.linear(h, pw1, Some(b.pointwise1_b));
    let nb = ffi::strides(h);
    let signal = g.view_2d(h, d_model, t, nb[1], 0);
    let gate = g.view_2d(h, d_model, t, nb[1], d_model as usize * nb[0]);
    let h = g.mul(signal, g.sigmoid(gate));
    // [t, d_model], padded on the time axis for a same-length conv
    let h = g.cont(g.transpose(h));
    let [k, _, _, _] = ffi::shape(b.depthwise);
    let pad = ((k - 1) / 2) as c_int;
    let h = g.pad_ext(h, pad, pad, 0, 0, 0, 0, 0, 0);
    let kernel = if ffi::dtype(b.depthwise) == ty::F32 {
        b.depthwise
    } else {
        g.cast(b.depthwise, ty::F32)
    };
    let kernel = g.reshape_2d(kernel, k, d_model);
    // [d_model, t, 1]
    let h = g.ssm_conv(g.reshape_3d(h, t + k - 1, d_model, 1), kernel);
    let h = g.add(g.reshape_2d(h, d_model, t), b.depthwise_b);
    let h = g.add(g.mul(h, bn_scale), bn_shift);
    let h = g.silu(h);
    let [_, pw_in, pw_out, _] = ffi::shape(b.pointwise2);
    let pw2 = g.reshape_2d(b.pointwise2, pw_in, pw_out);
    g.add(x, g.linear(h, pw2, Some(b.pointwise2_b)))
}

/// The depthwise conv in [`conv_module`] pads the time axis by
/// `(k - 1) / 2` on each side and reshapes to `t + k - 1` rows, which
/// only agree for odd `k`; an even or zero kernel would trip
/// `ggml_reshape_3d`'s element-count assert and abort the process. The
/// loader bounds the key only by `MAX_DIM`, so check it here.
fn check_conv_kernel(k: u64) -> Result<()> {
    if k.is_multiple_of(2) {
        bail!("encoder: conv kernel {k} (the depthwise conv needs an odd kernel)");
    }
    Ok(())
}

/// Frames left after the three stride-2 convolutions of the
/// `dw_striding` subsampler, each `floor((len + 2 * 1 - 3) / 2) + 1`
/// for kernel 3, padding 1. `1056 -> 132`, `1057 -> 133` and `0 -> 0`.
pub fn subsampled_len(frames: usize) -> usize {
    (0..3).fold(frames, |len, _| {
        (len + 2).checked_sub(3).map_or(0, |n| n / 2 + 1)
    })
}

/// NeMo `RelPositionalEncoding` table for `t_enc` encoder frames, row
/// major with `2 * t_enc - 1` rows of `d_model` values. Row `r` holds
/// position `p = t_enc - 1 - r`, so positions run from `t_enc - 1` down
/// to `-(t_enc - 1)`. Column `2i` is `sin(p * div_i)` and column
/// `2i + 1` is `cos(p * div_i)` with `div_i = exp(-2i ln(10000) /
/// d_model)`; `d_model` must be even for the interleaving. The buffer is
/// the ggml `[d_model, 2 * t_enc - 1]` input as is.
pub fn pos_table(t_enc: usize, d_model: usize) -> Vec<f32> {
    assert!(
        d_model.is_multiple_of(2),
        "pos_table: d_model {d_model} is odd"
    );
    let rows = 2 * t_enc - 1;
    let div: Vec<f64> = (0..d_model / 2)
        .map(|i| (-((2 * i) as f64) * 10_000f64.ln() / d_model as f64).exp())
        .collect();
    let mut table = Vec::with_capacity(rows * d_model);
    for r in 0..rows {
        let p = (t_enc as f64 - 1.0) - r as f64;
        for d in &div {
            let (sin, cos) = (p * d).sin_cos();
            table.push(sin as f32);
            table.push(cos as f32);
        }
    }
    table
}

/// Fold inference-mode BatchNorm into a per-channel `scale` and `shift`
/// so that `(x - mean) / sqrt(var + eps) * weight + bias` becomes
/// `x * scale + shift`: `scale = weight / sqrt(var + eps)` and
/// `shift = bias - mean * scale`. All four slices share one length.
pub fn fold_batch_norm(
    weight: &[f32],
    bias: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = weight.len();
    assert!(
        bias.len() == n && mean.len() == n && var.len() == n,
        "BatchNorm fold: channel counts differ ({n}, {}, {}, {})",
        bias.len(),
        mean.len(),
        var.len()
    );
    let eps = eps as f64;
    let mut scale = Vec::with_capacity(n);
    let mut shift = Vec::with_capacity(n);
    for (((w, b), m), v) in weight.iter().zip(bias).zip(mean).zip(var) {
        let s = *w as f64 / (*v as f64 + eps).sqrt();
        scale.push(s as f32);
        shift.push((*b as f64 - *m as f64 * s) as f32);
    }
    (scale, shift)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-6
    }

    #[test]
    fn subsampled_len_matches_the_fixture_and_the_dump() {
        assert_eq!(subsampled_len(1056), 132);
        assert_eq!(subsampled_len(1057), 133);
    }

    #[test]
    fn subsampled_len_of_nothing_is_nothing() {
        assert_eq!(subsampled_len(0), 0);
        assert_eq!(subsampled_len(1), 1);
    }

    #[test]
    fn subsampled_len_halves_three_times() {
        let step = |len: usize| (len + 2 - 3) / 2 + 1;
        assert_eq!(step(1056), 528);
        assert_eq!(step(528), 264);
        assert_eq!(step(264), 132);
        assert_eq!(subsampled_len(1056), step(step(step(1056))));
    }

    #[test]
    fn check_conv_kernel_accepts_the_odd_nemo_kernel() {
        check_conv_kernel(9).unwrap();
        check_conv_kernel(31).unwrap();
    }

    #[test]
    fn check_conv_kernel_rejects_even_and_zero_kernels() {
        for k in [0, 8] {
            let err = check_conv_kernel(k).unwrap_err().to_string();
            assert_eq!(
                err,
                format!("encoder: conv kernel {k} (the depthwise conv needs an odd kernel)")
            );
        }
    }

    #[test]
    fn pos_table_middle_row_is_position_zero() {
        let (t, d) = (5, 8);
        let table = pos_table(t, d);
        assert_eq!(table.len(), (2 * t - 1) * d);
        let mid = &table[(t - 1) * d..t * d];
        assert_eq!(mid, &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn pos_table_first_row_is_position_t_minus_one() {
        let (t, d) = (5, 8);
        let table = pos_table(t, d);
        let p = (t - 1) as f64;
        assert!(close(table[0], p.sin() as f32));
        assert!(close(table[1], p.cos() as f32));
        let div_1 = (-2.0 * 10_000f64.ln() / d as f64).exp();
        assert!(close(table[2], (p * div_1).sin() as f32));
        assert!(close(table[3], (p * div_1).cos() as f32));
    }

    #[test]
    fn pos_table_mirrored_rows_negate_sines_and_share_cosines() {
        let (t, d) = (6, 8);
        let table = pos_table(t, d);
        let rows = 2 * t - 1;
        for r in 0..rows {
            let a = &table[r * d..(r + 1) * d];
            let b = &table[(rows - 1 - r) * d..(rows - r) * d];
            for i in 0..d / 2 {
                assert!(close(a[2 * i], -b[2 * i]), "row {r} col {}", 2 * i);
                assert!(
                    close(a[2 * i + 1], b[2 * i + 1]),
                    "row {r} col {}",
                    2 * i + 1
                );
            }
        }
    }

    #[test]
    fn pos_table_rms_matches_the_dump_sidecar() {
        // sin^2 + cos^2 == 1 per pair, so the rms is 1/sqrt(2) for any
        // size; the enc.pos_emb sidecar reports 0.707107.
        let table = pos_table(133, 1024);
        let sum: f64 = table.iter().map(|v| (*v as f64).powi(2)).sum();
        let rms = (sum / table.len() as f64).sqrt();
        assert!(
            (rms - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-6,
            "rms {rms}"
        );
    }

    #[test]
    fn fold_batch_norm_reproduces_the_normalisation() {
        let (w, b, m, v) = (
            [2.0f32, 0.5],
            [0.25f32, -1.0],
            [1.0f32, -3.0],
            [4.0f32, 0.25],
        );
        let eps = 1e-5;
        let (scale, shift) = fold_batch_norm(&w, &b, &m, &v, eps);
        for c in 0..2 {
            for x in [-2.0f32, 0.0, 3.5] {
                let want = (x as f64 - m[c] as f64) / (v[c] as f64 + eps as f64).sqrt()
                    * w[c] as f64
                    + b[c] as f64;
                let got = x * scale[c] + shift[c];
                assert!(
                    (got as f64 - want).abs() < 1e-5,
                    "channel {c} x {x}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn fold_batch_norm_identity_parameters() {
        let eps = 1e-5f32;
        let (scale, shift) =
            fold_batch_norm(&[1.0, 1.0], &[0.0, 0.0], &[0.0, 0.0], &[1.0, 1.0], eps);
        let want = 1.0 / (1.0 + eps as f64).sqrt();
        assert!(scale.iter().all(|s| close(*s, want as f32)));
        assert_eq!(shift, vec![0.0, 0.0]);
    }
}
