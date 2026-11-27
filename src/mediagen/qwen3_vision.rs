//! Qwen3-VL's vision tower from its llama.cpp mmproj GGUF (`qwen3vl_merger`):
//! 16x16 patches with an interpolated learned position embedding, a ViT with
//! 2-D RoPE, then 2x2 patches merged into one language-model token, plus the
//! DeepStack features of a few inner layers that the language model adds to
//! its first layers' image tokens.

use std::path::Path;

use anyhow::{bail, Context, Result};

use super::backend::{Backend, Graph};
use super::cosmos3::dit as qwen3;
use super::ffi::Api;
use super::weights::{read_index, LoadOpts, Weights};

/// Pixels per patch side.
pub const PATCH: i64 = 16;
/// Patches per merged token side.
pub const MERGE: i64 = 2;
const EPS: f32 = 1e-6;
const ROPE_THETA: f64 = 10000.0;

pub struct Vision {
    w: Weights,
    layers: i64,
    heads: i64,
    hidden: i64,
    /// Side of the learned position embedding grid.
    pos_side: i64,
    /// Layers whose output feeds a DeepStack merger, in order.
    deepstack: Vec<i64>,
    flash: bool,
}

/// One image through the tower: `n = (h / 32) * (w / 32)` tokens, row major.
pub struct Encoded {
    /// Merged token grid.
    pub grid_h: i64,
    pub grid_w: i64,
    /// `[n][out_dim]`, the language model's image token embeddings.
    pub embeds: Vec<f32>,
    /// `[n][out_dim]` per DeepStack layer, added after language-model layer `i`.
    pub deepstack: Vec<Vec<f32>>,
}

/// Patch `i` of the block-major order the tower runs in (each 2x2 block's
/// patches together, blocks row major): its `(row, col)` in the patch grid.
fn block_order(gh: i64, gw: i64) -> Vec<(i64, i64)> {
    let mut out = Vec::with_capacity((gh * gw) as usize);
    for bh in 0..gh / MERGE {
        for bw in 0..gw / MERGE {
            for ih in 0..MERGE {
                for iw in 0..MERGE {
                    out.push((bh * MERGE + ih, bw * MERGE + iw));
                }
            }
        }
    }
    out
}

/// Bilinear (align-corners) interpolation of the `side x side` learned grid at
/// each patch: the four corner indexes and weights.
fn pos_taps(order: &[(i64, i64)], gh: i64, gw: i64, side: i64) -> ([Vec<i32>; 4], [Vec<f32>; 4]) {
    let max = (side - 1) as f64;
    let at = |i: i64, n: i64| {
        if n == 1 {
            0.0
        } else {
            max * i as f64 / (n - 1) as f64
        }
    };
    let mut idx: [Vec<i32>; 4] = Default::default();
    let mut wt: [Vec<f32>; 4] = Default::default();
    for &(h, w) in order {
        let (hp, wp) = (at(h, gh), at(w, gw));
        let (h0, w0) = (hp.floor() as i64, wp.floor() as i64);
        let (h1, w1) = ((h0 + 1).min(side - 1), (w0 + 1).min(side - 1));
        let (dh, dw) = (hp - h0 as f64, wp - w0 as f64);
        for (k, (r, c, f)) in [
            (h0, w0, (1.0 - dh) * (1.0 - dw)),
            (h0, w1, (1.0 - dh) * dw),
            (h1, w0, dh * (1.0 - dw)),
            (h1, w1, dh * dw),
        ]
        .into_iter()
        .enumerate()
        {
            idx[k].push((r * side + c) as i32);
            wt[k].push(f as f32);
        }
    }
    (idx, wt)
}

/// 2-D rotary tables `[n][head_dim/2]`: the first half of the frequencies
/// turn with the patch row, the second with its column.
fn rope_tables(order: &[(i64, i64)], head_dim: i64) -> (Vec<f32>, Vec<f32>) {
    let quarter = (head_dim / 4) as usize;
    let mut cos = Vec::with_capacity(order.len() * 2 * quarter);
    let mut sin = Vec::with_capacity(order.len() * 2 * quarter);
    for &(h, w) in order {
        for pos in [h, w] {
            for j in 0..quarter {
                let inv = 1.0 / ROPE_THETA.powf(2.0 * j as f64 / (head_dim / 2) as f64);
                let a = pos as f64 * inv;
                cos.push(a.cos() as f32);
                sin.push(a.sin() as f32);
            }
        }
    }
    (cos, sin)
}

impl Vision {
    pub fn load(api: &'static Api, be: &Backend, path: &Path, flash: bool) -> Result<Vision> {
        let idx = read_index(path)?;
        let projector = idx.metadata.get("clip.projector_type").map(String::as_str);
        if projector != Some("qwen3vl_merger") {
            bail!(
                "{}: expected a Qwen3-VL mmproj, got projector {projector:?}",
                path.display()
            );
        }
        let dim = |i| {
            idx.dim("v.position_embd.weight", i)
                .with_context(|| path.display().to_string())
        };
        let pos = [dim(0)?, dim(1)?];
        let hidden = pos[0];
        let pos_side = (pos[1] as f64).sqrt() as i64;
        let layers = idx.count("v.blk.");
        // head_dim 72 in every released size
        let heads = hidden / 72;
        if pos_side * pos_side != pos[1] || layers == 0 || heads * 72 != hidden {
            bail!("{}: unsupported Qwen3-VL vision tower", path.display());
        }
        let w = Weights::load(
            api,
            path,
            be.weight_buft(),
            &LoadOpts {
                rename: None,
                transform: None,
            },
        )?;
        Ok(Vision {
            w,
            layers,
            heads,
            hidden,
            pos_side,
            deepstack: idx.numbered("v.deepstack."),
            flash,
        })
    }

    /// `rgb`: `[h][w][3]` in `[-1, 1]` (the processor's 0.5 mean and std), sides
    /// multiples of 32.
    pub fn encode(&self, be: &Backend, rgb: &[f32], h: i64, w: i64) -> Result<Encoded> {
        let unit = PATCH * MERGE;
        if h % unit != 0 || w % unit != 0 || rgb.len() as i64 != h * w * 3 {
            bail!("the vision tower takes sides that are multiples of {unit}, got {w}x{h}");
        }
        let (gh, gw) = (h / PATCH, w / PATCH);
        let n = gh * gw;
        let order = block_order(gh, gw);
        // [h][w][3] -> ggml [W, H, 3]
        let mut planar = vec![0f32; rgb.len()];
        let px = (h * w) as usize;
        for (i, p) in rgb.chunks(3).enumerate() {
            for c in 0..3 {
                planar[c * px + i] = p[c];
            }
        }
        let gr = Graph::new(be.api, 8192 + self.layers as usize * 64)?;
        let g = &gr;
        let wt = |n: &str| self.w.get(n);
        let img = gr.input_f32(&[w, h, 3, 1], &planar, "pixels");
        // the two temporal taps see the same frame: one conv with their sum
        let k = g.add(wt("v.patch_embd.weight")?, wt("v.patch_embd.weight.1")?);
        let x = g.conv_2d(k, img, PATCH as i32, PATCH as i32, 0, 0, 1, 1); // [gw, gh, C, 1]
        let x = g.reshape_2d(g.cont(g.permute(x, 1, 2, 0, 3)), self.hidden, n); // raster
        let raster: Vec<i32> = order.iter().map(|&(r, c)| (r * gw + c) as i32).collect();
        let x = g.get_rows(x, gr.input_i32(&[n], &raster, "order"));
        let mut x = g.add(x, wt("v.patch_embd.bias")?);
        let (pidx, pwt) = pos_taps(&order, gh, gw, self.pos_side);
        for k in 0..4 {
            let e = g.get_rows(
                wt("v.position_embd.weight")?,
                gr.input_i32(&[n], &pidx[k], &format!("pos{k}")),
            );
            let f = gr.input_f32(&[1, n], &pwt[k], &format!("posw{k}"));
            x = g.add(x, g.mul(e, f));
        }
        let hd = self.hidden / self.heads;
        let (cos, sin) = rope_tables(&order, hd);
        let cos_t = gr.input_f32(&[hd / 2, n], &cos, "cos");
        let sin_t = gr.input_f32(&[hd / 2, n], &sin, "sin");
        let ln = |x, p: &str| -> Result<_> {
            Ok(g.add(
                g.mul(g.norm(x, EPS), wt(&format!("{p}.weight"))?),
                wt(&format!("{p}.bias"))?,
            ))
        };
        let lin = |x, p: &str| -> Result<_> {
            Ok(g.linear(
                x,
                wt(&format!("{p}.weight"))?,
                Some(wt(&format!("{p}.bias"))?),
            ))
        };
        let merged = |x| g.reshape_2d(g.cont(x), self.hidden * MERGE * MERGE, n / 4);
        let mut stack = Vec::new();
        for l in 0..self.layers {
            let p = format!("v.blk.{l}");
            let qkv = lin(ln(x, &format!("{p}.ln1"))?, &format!("{p}.attn_qkv"))?;
            let nb = qkv.strides()[1];
            let part = |i: usize| g.view_2d(qkv, self.hidden, n, nb, i * self.hidden as usize * 4);
            let q = qwen3::rope(g, part(0), cos_t, sin_t, self.heads);
            let k = qwen3::rope(g, part(1), cos_t, sin_t, self.heads);
            let a = qwen3::sdpa_gqa(
                g,
                q,
                k,
                g.cont(part(2)),
                self.heads,
                self.heads,
                None,
                self.flash,
            );
            x = g.add(x, lin(a, &format!("{p}.attn_out"))?);
            let m = g.gelu(lin(ln(x, &format!("{p}.ln2"))?, &format!("{p}.ffn_up"))?);
            x = g.add(x, lin(m, &format!("{p}.ffn_down"))?);
            if self.deepstack.contains(&l) {
                let d = format!("v.deepstack.{l}");
                let y = ln(merged(x), &format!("{d}.norm"))?;
                let y = lin(
                    g.gelu_erf(lin(y, &format!("{d}.fc1"))?),
                    &format!("{d}.fc2"),
                )?;
                let y = g.cont(y);
                gr.mark_output(y);
                stack.push(y);
            }
        }
        let y = merged(ln(x, "v.post_ln")?);
        let y = g.cont(lin(g.gelu_erf(lin(y, "mm.0")?), "mm.2")?);
        gr.mark_output(y);
        gr.compute(be)?;
        Ok(Encoded {
            grid_h: gh / MERGE,
            grid_w: gw / MERGE,
            embeds: gr.output_f32(y),
            deepstack: stack.iter().map(|&t| gr.output_f32(t)).collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_order_keeps_each_2x2_block_together() {
        let o = block_order(2, 4);
        assert_eq!(&o[..4], [(0, 0), (0, 1), (1, 0), (1, 1)]);
        assert_eq!(&o[4..], [(0, 2), (0, 3), (1, 2), (1, 3)]);
    }

    #[test]
    fn pos_taps_hit_the_grid_corners() {
        let o = block_order(2, 2);
        let (idx, wt) = pos_taps(&o, 2, 2, 48);
        // the first patch is the grid's first cell, the last its last
        assert_eq!((idx[0][0], wt[0][0]), (0, 1.0));
        assert_eq!(idx[3][3], 48 * 48 - 1);
        assert!((wt[0][3] + wt[1][3] + wt[2][3] + wt[3][3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rope_tables_turn_rows_then_columns() {
        let (cos, sin) = rope_tables(&[(3, 5)], 72);
        assert_eq!(cos.len(), 36);
        assert!((sin[0] - 3f32.sin()).abs() < 1e-6);
        assert!((sin[18] - 5f32.sin()).abs() < 1e-6);
        let inv = 1.0 / 10000f64.powf(2.0 / 36.0);
        assert!((cos[19] as f64 - (5.0 * inv).cos()).abs() < 1e-6);
    }
}
