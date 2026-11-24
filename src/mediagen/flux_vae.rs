//! The FLUX.1 autoencoder's decoder (`ae.safetensors`, Black Forest Labs
//! tensor names): 16 latent channels, 8x spatial upsampling. Z-Image decodes
//! with it.

use anyhow::{bail, Result};

use super::backend::{Backend, Graph};
use super::ffi::{self, Tensor};
use super::weights::Weights;

/// `AutoEncoderParams` of FLUX.1: latents are `z / scale + shift` before decoding.
pub const SCALE_FACTOR: f32 = 0.3611;
pub const SHIFT_FACTOR: f32 = 0.1159;
const GROUPS: i32 = 32;
const EPS: f32 = 1e-6;
/// `ch_mult` of the FLUX.1 autoencoder; `ch = 128`, three blocks per level.
const CH_MULT: [i64; 4] = [1, 2, 4, 4];
const CH: i64 = 128;
const BLOCKS: usize = 3;

/// Decoded pixels `[h][w][3]` in about `[-1, 1]`.
pub struct Image {
    pub width: i64,
    pub height: i64,
    pub rgb: Vec<f32>,
}

/// Only the decoder is ever loaded, under the original (`ae.safetensors`)
/// names; a Diffusers `AutoencoderKL` (`vae/diffusion_pytorch_model.safetensors`,
/// whose `up_blocks` run in the opposite order) is renamed to them.
pub fn tensor_name(n: &str) -> Option<String> {
    let n = n.strip_prefix("vae.").unwrap_or(n);
    let rest = n.strip_prefix("decoder.")?;
    let levels = CH_MULT.len();
    let mapped = if let Some(r) = rest.strip_prefix("mid_block.resnets.") {
        let (i, tail) = r.split_once('.')?;
        format!("mid.block_{}.{}", i.parse::<usize>().ok()? + 1, tail)
    } else if let Some(r) = rest.strip_prefix("mid_block.attentions.0.") {
        let r = r
            .replacen("group_norm.", "norm.", 1)
            .replacen("to_q.", "q.", 1)
            .replacen("to_k.", "k.", 1)
            .replacen("to_v.", "v.", 1)
            .replacen("to_out.0.", "proj_out.", 1);
        format!("mid.attn_1.{r}")
    } else if let Some(r) = rest.strip_prefix("up_blocks.") {
        let (i, tail) = r.split_once('.')?;
        let level = levels - 1 - i.parse::<usize>().ok()?;
        if let Some(t) = tail.strip_prefix("resnets.") {
            format!(
                "up.{level}.block.{}",
                t.replacen("conv_shortcut.", "nin_shortcut.", 1)
            )
        } else {
            format!(
                "up.{level}.upsample.{}",
                tail.strip_prefix("upsamplers.0.")?
            )
        }
    } else if let Some(r) = rest.strip_prefix("conv_norm_out.") {
        format!("norm_out.{r}")
    } else {
        rest.to_string()
    };
    Some(format!("decoder.{mapped}"))
}

struct Dec<'g> {
    g: &'g Graph<'g>,
    w: &'g Weights,
    direct: bool,
}

impl<'g> Dec<'g> {
    fn conv(&self, name: &str, x: Tensor<'g>, pad: i32) -> Result<Tensor<'g>> {
        let g = self.g;
        let k = self.w.get(&format!("{name}.weight"))?;
        let y = if self.direct {
            g.conv_2d_direct(k, x, 1, pad)
        } else {
            None
        }
        .unwrap_or_else(|| g.conv_2d(k, x, 1, 1, pad, pad, 1, 1));
        let b = self.w.get(&format!("{name}.bias"))?;
        Ok(g.add(y, g.reshape_4d(b, 1, 1, y.shape()[2], 1)))
    }

    fn group_norm(&self, name: &str, x: Tensor<'g>) -> Result<Tensor<'g>> {
        let g = self.g;
        let c = x.shape()[2];
        let y = g.group_norm(x, GROUPS, EPS);
        let wt = g.reshape_4d(self.w.get(&format!("{name}.weight"))?, 1, 1, c, 1);
        let b = g.reshape_4d(self.w.get(&format!("{name}.bias"))?, 1, 1, c, 1);
        Ok(g.add(g.mul(y, wt), b))
    }

    /// `ResnetBlock`: two norm-swish-conv steps and a (1x1 when the width changes) skip.
    fn resnet(&self, p: &str, x: Tensor<'g>) -> Result<Tensor<'g>> {
        let g = self.g;
        let h = self.conv(
            &format!("{p}.conv1"),
            g.silu(self.group_norm(&format!("{p}.norm1"), x)?),
            1,
        )?;
        let h = self.conv(
            &format!("{p}.conv2"),
            g.silu(self.group_norm(&format!("{p}.norm2"), h)?),
            1,
        )?;
        let skip = if self.w.has(&format!("{p}.nin_shortcut.weight")) {
            self.conv(&format!("{p}.nin_shortcut"), x, 0)?
        } else {
            x
        };
        Ok(g.add(skip, h))
    }

    /// `AttnBlock`: one head over every pixel.
    fn attn(&self, p: &str, x: Tensor<'g>) -> Result<Tensor<'g>> {
        let g = self.g;
        let [w, h, c, _] = x.shape();
        let hn = self.group_norm(&format!("{p}.norm"), x)?;
        // [W, H, C] -> tokens [C, W*H]
        let t = g.reshape_2d(g.cont(g.permute(hn, 1, 2, 0, 3)), c, w * h);
        let proj = |name: &str| -> Result<Tensor<'g>> {
            let k = self.w.get(&format!("{p}.{name}.weight"))?;
            let b = self.w.get(&format!("{p}.{name}.bias"))?;
            Ok(g.linear(t, g.reshape_2d(k, c, c), Some(b)))
        };
        let (q, k, v) = (proj("q")?, proj("k")?, proj("v")?);
        // F32 attention: flash attention's F16 keys/values can overflow here
        let a = super::cosmos3::dit::sdpa_gqa(g, q, k, v, 1, 1, None, false);
        let ko = self.w.get(&format!("{p}.proj_out.weight"))?;
        let bo = self.w.get(&format!("{p}.proj_out.bias"))?;
        let o = g.linear(a, g.reshape_2d(ko, c, c), Some(bo));
        // tokens [C, W*H] -> [W, H, C]
        let o = g.cont(g.permute(g.reshape_3d(o, c, w, h), 2, 0, 1, 3));
        Ok(g.add(x, g.reshape_4d(o, w, h, c, 1)))
    }

    fn decode(&self, z: Tensor<'g>) -> Result<Tensor<'g>> {
        let g = self.g;
        let mut x = self.conv("decoder.conv_in", z, 1)?;
        x = self.resnet("decoder.mid.block_1", x)?;
        x = self.attn("decoder.mid.attn_1", x)?;
        x = self.resnet("decoder.mid.block_2", x)?;
        for level in (0..CH_MULT.len()).rev() {
            for b in 0..BLOCKS {
                x = self.resnet(&format!("decoder.up.{level}.block.{b}"), x)?;
            }
            if level != 0 {
                let [w, h, c, n] = x.shape();
                let up = g.interpolate(x, 2 * w, 2 * h, c, n, ffi::GGML_SCALE_MODE_NEAREST);
                x = self.conv(&format!("decoder.up.{level}.upsample.conv"), up, 1)?;
            }
        }
        let x = g.silu(self.group_norm("decoder.norm_out", x)?);
        self.conv("decoder.conv_out", x, 1)
    }
}

/// Decodes latents `[16][h][w]` (as the transformer sees them, before
/// `SCALE_FACTOR`/`SHIFT_FACTOR`) to an `8h x 8w` image.
pub fn decode(w: &Weights, be: &Backend, z: &[f32], h: i64, wd: i64) -> Result<Image> {
    let c = w.get("decoder.conv_in.weight")?.shape()[2];
    if z.len() as i64 != c * h * wd {
        bail!("expected {c}x{h}x{wd} latents, got {}", z.len());
    }
    if w.get("decoder.conv_out.weight")?.shape()[3] != 3
        || w.get("decoder.up.0.block.0.conv1.weight")?.shape()[3] != CH * CH_MULT[0]
    {
        bail!("not a FLUX.1 autoencoder decoder");
    }
    // [C][H][W] is ggml's [W, H, C, 1] already
    let x: Vec<f32> = z.iter().map(|v| v / SCALE_FACTOR + SHIFT_FACTOR).collect();
    let gr = Graph::new(be.api, 16 * 1024)?;
    let zt = gr.input_f32(&[wd, h, c, 1], &x, "latents");
    let dec = Dec {
        g: &gr,
        w,
        // the same choice as the Wan VAE: im2col unless asked, or on Vulkan
        direct: super::wan_vae::direct_conv(be),
    };
    let out = gr.cont(dec.decode(zt)?);
    gr.mark_output(out);
    gr.compute(be)?;
    let planar = gr.output_f32(out);
    if planar.iter().any(|v| !v.is_finite()) {
        bail!("the VAE decoder produced non-finite values");
    }
    let [ow, oh, oc, _] = out.shape();
    if oc != 3 {
        bail!("the decoder produced {oc} channels");
    }
    let px = (ow * oh) as usize;
    let mut rgb = vec![0f32; px * 3];
    for ch in 0..3 {
        for i in 0..px {
            rgb[i * 3 + ch] = planar[ch * px + i];
        }
    }
    Ok(Image {
        width: ow,
        height: oh,
        rgb,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diffusers_names_map_to_the_original_ones() {
        let m = |n: &str| tensor_name(n);
        assert_eq!(m("encoder.conv_in.weight"), None);
        assert_eq!(
            m("decoder.conv_in.weight").as_deref(),
            Some("decoder.conv_in.weight")
        );
        assert_eq!(
            m("decoder.mid_block.resnets.1.conv2.bias").as_deref(),
            Some("decoder.mid.block_2.conv2.bias")
        );
        assert_eq!(
            m("decoder.mid_block.attentions.0.to_out.0.weight").as_deref(),
            Some("decoder.mid.attn_1.proj_out.weight")
        );
        assert_eq!(
            m("decoder.mid_block.attentions.0.group_norm.bias").as_deref(),
            Some("decoder.mid.attn_1.norm.bias")
        );
        assert_eq!(
            m("decoder.up_blocks.0.resnets.2.norm1.weight").as_deref(),
            Some("decoder.up.3.block.2.norm1.weight")
        );
        assert_eq!(
            m("decoder.up_blocks.2.resnets.0.conv_shortcut.weight").as_deref(),
            Some("decoder.up.1.block.0.nin_shortcut.weight")
        );
        assert_eq!(
            m("decoder.up_blocks.1.upsamplers.0.conv.weight").as_deref(),
            Some("decoder.up.2.upsample.conv.weight")
        );
        assert_eq!(
            m("decoder.conv_norm_out.weight").as_deref(),
            Some("decoder.norm_out.weight")
        );
        // the original names pass through
        assert_eq!(
            m("decoder.up.3.block.0.conv1.weight").as_deref(),
            Some("decoder.up.3.block.0.conv1.weight")
        );
    }
}
