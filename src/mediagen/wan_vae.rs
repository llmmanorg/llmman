//! Wan 2.2 TI2V video VAE (diffusers `AutoencoderKLWan`): 48 latent channels,
//! 16x spatial and 4x temporal compression. Runs one latent frame at a time as
//! the reference does, with each causal conv caching the two frames before its
//! chunk; the first latent frame is never resampled in time.
//! Also decodes Qwen-Image 2.1's image specialization (2-D kernels, RGBA).
//! Activations are `[W, H, C, T]` in ggml order.

use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::mediagen::backend::{Backend, Ctx, Graph, Resident};
use crate::mediagen::ffi::{self, Tensor};
use crate::mediagen::weights::{read_safetensors_raw, write_safetensors_raw, Weights};

/// im2col scratch per `conv_2d` call; frames are batched under this.
const IM2COL_BUDGET: i64 = 256 << 20;
const NORM_EPS: f32 = 1e-12;

/// A frame-major decoded chunk.
pub struct Chunk {
    pub frames: i64,
    pub height: i64,
    pub width: i64,
    /// 3 (RGB) or 4 (RGBA).
    pub channels: i64,
    /// `[f][h][w][channels]`, roughly in `[-1, 1]`.
    pub rgb: Vec<f32>,
}

/// What sets the decoders this module runs apart.
pub struct Layout {
    /// Per up block with an upsampler: does it double the frames (`temperal_upsample`)?
    pub temporal_up: Vec<bool>,
    /// `conv_out` emits 2x2 pixel patches folded into the channels (`patch_size = 2`).
    pub patch: bool,
}

impl Layout {
    /// Wan 2.2 TI2V.
    pub fn wan22() -> Layout {
        Layout {
            temporal_up: vec![true, true, false],
            patch: true,
        }
    }
}

/// The cached input frames of every causal conv, in traversal order, on the
/// device: two tensors per slot, one read and one written (`ggml_cpy`) by each
/// chunk's graph, swapped between chunks. Unfilled slots read as zeros.
struct Caches<'b> {
    be: &'b Backend,
    slots: Vec<Option<CacheSlot>>,
    /// Which of the two tensors of each slot the next chunk reads.
    read: usize,
    /// Slots the last chunk left holding data (vs zeros).
    filled: Vec<bool>,
}

struct CacheSlot {
    pair: Resident,
    ne: [i64; 4],
}

impl<'b> Caches<'b> {
    fn new(be: &'b Backend) -> Caches<'b> {
        Caches {
            be,
            slots: Vec::new(),
            read: 0,
            filled: Vec::new(),
        }
    }

    /// The tensor holding `slot`'s frames for the chunk being built, if any.
    fn get(&self, slot: usize) -> Option<Tensor> {
        match (self.slots.get(slot), self.filled.get(slot)) {
            (Some(Some(s)), Some(true)) => Some(s.pair.tensors[self.read]),
            _ => None,
        }
    }

    /// The tensor the chunk being built writes `slot` into, allocated on first use.
    fn target(&mut self, slot: usize, ne: [i64; 4]) -> Result<Tensor> {
        if self.slots.len() <= slot {
            self.slots.resize_with(slot + 1, || None);
        }
        if let Some(s) = &self.slots[slot] {
            if s.ne != ne {
                bail!("cache slot {slot}: shape {ne:?} does not match {:?}", s.ne);
            }
        } else {
            let pair = Resident::new(self.be, ffi::ty::F32, &[&ne, &ne])
                .with_context(|| format!("cache slot {slot} {ne:?}"))?;
            self.slots[slot] = Some(CacheSlot { pair, ne });
        }
        Ok(self.slots[slot].as_ref().unwrap().pair.tensors[1 - self.read])
    }

    /// After a chunk's graph ran: the written set becomes the one to read.
    fn advance(&mut self, filled: Vec<bool>) {
        self.filled = filled;
        self.read = 1 - self.read;
    }
}

struct Build<'a, 'b> {
    gr: &'a mut Graph,
    w: &'a Weights,
    /// Use `ggml_conv_2d_direct` instead of im2col + matmul.
    direct: bool,
    caches: &'a mut Caches<'b>,
    slot: usize,
    /// Per slot: does this chunk leave data (vs zeros) for the next one?
    filled: Vec<bool>,
    first_chunk: bool,
}

impl Build<'_, '_> {
    fn g(&self) -> &Ctx {
        &self.gr.ctx
    }

    fn zeros(&mut self, ne: [i64; 4], name: &str) -> Tensor {
        let n = ne.iter().product::<i64>() as usize;
        self.gr.input_f32(&ne, &vec![0f32; n], name)
    }

    /// The `[C]` affine weight as F32.
    fn vec_f32(&self, t: Tensor) -> Tensor {
        let g = self.g();
        let n: i64 = ffi::shape(t).iter().product();
        let t = if unsafe { (*t).type_ } == ffi::ty::F32 {
            t
        } else {
            g.cast(t, ffi::ty::F32)
        };
        g.reshape_4d(g.cont(t), n, 1, 1, 1)
    }

    /// `WanRMS_norm`: normalize over the channels, times gamma.
    fn norm(&self, x: Tensor, gamma: &str) -> Result<Tensor> {
        let gm = self.vec_f32(self.w.get(gamma)?);
        let g = self.g();
        let p = g.cont(g.permute(x, 1, 2, 0, 3)); // [C, W, H, T]
        let p = g.mul(g.rms_norm(p, NORM_EPS), gm);
        Ok(g.cont(g.permute(p, 2, 0, 1, 3)))
    }

    fn bias(&self, y: Tensor, name: &str) -> Result<Tensor> {
        let g = self.g();
        let b = self.w.get(name)?;
        let oc = ffi::shape(y)[2];
        Ok(g.add(y, g.reshape_4d(b, 1, 1, oc, 1)))
    }

    /// 1x1 conv (shortcuts, quant convs, attention projections): no padding, no cache.
    fn conv1(&self, prefix: &str, x: Tensor) -> Result<Tensor> {
        let k = self.w.get(&format!("{prefix}.weight"))?;
        let y = self.conv2(k, x, 1, 0);
        self.bias(y, &format!("{prefix}.bias"))
    }

    /// 2-D 3x3 conv (`resample.1`) over every frame.
    fn conv2d(&self, prefix: &str, x: Tensor) -> Result<Tensor> {
        let k = self.w.get(&format!("{prefix}.weight"))?;
        let y = self.conv_frames(k, x, 1, ffi::shape(x)[3])?;
        self.bias(y, &format!("{prefix}.bias"))
    }

    /// A 2-D conv: direct when the backend has it, else im2col + matmul.
    fn conv2(&self, k: Tensor, x: Tensor, s: i32, pad: i32) -> Tensor {
        match (self.direct, self.g().api.ggml_conv_2d_direct) {
            (true, Some(f)) => unsafe { f(self.g().raw, k, x, s, s, pad, pad, 1, 1) },
            _ => self.g().conv_2d(k, x, s, s, pad, pad, 1, 1),
        }
    }

    /// `conv_2d` over the first `nframes` frames, batched so im2col stays under budget.
    fn conv_frames(&self, k: Tensor, x: Tensor, pad: i32, nframes: i64) -> Result<Tensor> {
        let g = self.g();
        let [w, h, c, _] = ffi::shape(x);
        let kw = ffi::shape(k)[0];
        let per_frame = kw * kw * c * w * h * 2;
        if !self.direct && pad > 0 && per_frame > IM2COL_BUDGET {
            return Ok(self.conv_bands(k, x, pad as i64, nframes));
        }
        let group = if self.direct {
            nframes
        } else {
            (IM2COL_BUDGET / per_frame.max(1)).clamp(1, nframes)
        };
        let nb = ffi::strides(x);
        let mut out: Option<Tensor> = None;
        let mut f0 = 0;
        while f0 < nframes {
            let nf = group.min(nframes - f0);
            let xv = g.view_4d(x, w, h, c, nf, nb[1], nb[2], nb[3], (f0 as usize) * nb[3]);
            let y = self.conv2(k, xv, 1, pad);
            out = Some(match out {
                None => y,
                Some(o) => g.concat(o, y, 3),
            });
            f0 += nf;
        }
        Ok(out.expect("at least one frame"))
    }

    /// [`Self::conv_frames`] in bands of rows (plus `pad` halo rows) when one
    /// frame's im2col is over budget; same result as one conv.
    fn conv_bands(&self, k: Tensor, x: Tensor, pad: i64, nframes: i64) -> Tensor {
        let g = self.g();
        let [w, h, c, _] = ffi::shape(x);
        let kw = ffi::shape(k)[0];
        let rows = (IM2COL_BUDGET / (kw * kw * c * w * 2)).clamp(1, h);
        let nb = ffi::strides(x);
        let mut out: Option<Tensor> = None;
        for f in 0..nframes {
            let mut frame: Option<Tensor> = None;
            let mut y0 = 0;
            while y0 < h {
                let y1 = (y0 + rows).min(h);
                let (a, b) = ((y0 - pad).max(0), (y1 + pad).min(h));
                let off = a as usize * nb[1] + f as usize * nb[3];
                let band = g.cont(g.view_4d(x, w, b - a, c, 1, nb[1], nb[2], nb[3], off));
                let (top, bottom) = (a - (y0 - pad), y1 + pad - b);
                let band = if top > 0 || bottom > 0 {
                    g.pad_ext(band, 0, 0, top as i32, bottom as i32, 0, 0, 0, 0)
                } else {
                    band
                };
                let y = g.conv_2d(k, band, 1, 1, pad as i32, 0, 1, 1);
                frame = Some(frame.map_or(y, |o| g.concat(o, y, 1)));
                y0 = y1;
            }
            let frame = frame.expect("at least one row");
            out = Some(out.map_or(frame, |o| g.concat(o, frame, 3)));
        }
        out.expect("at least one frame")
    }

    /// Takes the next cache slot: its frames of shape `ne`, or zeros when unfilled.
    fn take_cache(&mut self, prefix: &str, ne: [i64; 4]) -> Result<(usize, Tensor)> {
        let slot = self.slot;
        self.slot += 1;
        let t = match self.caches.get(slot) {
            Some(t) => {
                if ffi::shape(t) != ne {
                    bail!(
                        "{prefix}: cache shape {:?} does not match {ne:?}",
                        ffi::shape(t)
                    );
                }
                t
            }
            None => self.zeros(ne, &format!("zcache{slot}")),
        };
        Ok((slot, t))
    }

    /// Records `t` (or zeros when `None`) as what `slot` holds for the next chunk:
    /// a copy into the slot's write tensor, part of this chunk's graph.
    fn keep_cache(&mut self, slot: usize, t: Option<Tensor>) -> Result<()> {
        if self.filled.len() <= slot {
            self.filled.resize(slot + 1, false);
        }
        if let Some(t) = t {
            let dst = self.caches.target(slot, ffi::shape(t))?;
            self.gr.store(t, dst);
        }
        self.filled[slot] = t.is_some();
        Ok(())
    }

    /// The chunk's frames with the two before each stacked along the channels:
    /// `[w, h, 3c, n]` where channel block `k` holds frame `t + k - 2` of the stream.
    /// A 3x3x3 causal conv is then one 2-D conv with a `[3, 3, 3c, oc]` kernel.
    fn stack_taps(&self, xx: Tensor, n: i64) -> Tensor {
        let g = self.g();
        let [w, h, c, _] = ffi::shape(xx);
        let nb = ffi::strides(xx);
        let tap = |k: usize| g.view_4d(xx, w, h, c, n, nb[1], nb[2], nb[3], k * nb[3]);
        g.concat(g.concat(tap(0), tap(1), 2), tap(2), 2)
    }

    /// Causal 3x3x3 (or 3x1x1) conv over the chunk with the cached two frames
    /// before it; a 2-D kernel is a plain per-frame conv.
    fn cconv(&mut self, prefix: &str, x: Tensor, spatial_pad: i32) -> Result<Tensor> {
        let [w, h, c, n] = ffi::shape(x);
        let k = self.w.get(&format!("{prefix}.weight"))?;
        if ffi::shape(k)[2] == c {
            let y = self.conv_frames(k, x, spatial_pad, n)?;
            return self.bias(y, &format!("{prefix}.bias"));
        }
        let (slot, cache) = self.take_cache(prefix, [w, h, c, 2])?;
        let g = self.g();
        let xx = g.concat(cache, x, 3); // [w, h, c, n + 2]
        let nb = ffi::strides(xx);
        let keep = g.view_4d(xx, w, h, c, 2, nb[1], nb[2], nb[3], (n as usize) * nb[3]);
        self.keep_cache(slot, Some(keep))?;
        let stacked = self.stack_taps(xx, n);
        let y = self.conv_frames(k, stacked, spatial_pad, n)?; // k: [kw, kh, 3c, oc]
        self.bias(y, &format!("{prefix}.bias"))
    }

    /// `WanResidualBlock`
    fn resnet(&mut self, prefix: &str, x: Tensor) -> Result<Tensor> {
        let short = if self.w.has(&format!("{prefix}.conv_shortcut.weight")) {
            self.conv1(&format!("{prefix}.conv_shortcut"), x)?
        } else {
            x
        };
        let h = self.norm(x, &format!("{prefix}.norm1.gamma"))?;
        let h = self.cconv(&format!("{prefix}.conv1"), self.g().silu(h), 1)?;
        let h = self.norm(h, &format!("{prefix}.norm2.gamma"))?;
        let h = self.cconv(&format!("{prefix}.conv2"), self.g().silu(h), 1)?;
        Ok(self.g().add(h, short))
    }

    /// `WanAttentionBlock`: single-head spatial attention within each frame.
    fn attention(&mut self, prefix: &str, x: Tensor) -> Result<Tensor> {
        let [w, h, c, n] = ffi::shape(x);
        let hn = self.norm(x, &format!("{prefix}.norm.gamma"))?;
        let qkv = self.conv1(&format!("{prefix}.to_qkv"), hn)?; // [w, h, 3c, n]
        let g = self.g();
        let hw = w * h;
        let qkv = g.reshape_3d(qkv, hw, 3 * c, n);
        let nb = ffi::strides(qkv);
        let q = g.view_3d(qkv, hw, c, n, nb[1], nb[2], 0);
        let k = g.view_3d(qkv, hw, c, n, nb[1], nb[2], (c as usize) * nb[1]);
        let v = g.view_3d(qkv, hw, c, n, nb[1], nb[2], (2 * c as usize) * nb[1]);
        // tokens along ne1 for q/k, along ne0 for v
        let qc = g.cont(g.permute(q, 1, 0, 2, 3)); // [c, hw, n]
        let kc = g.cont(g.permute(k, 1, 0, 2, 3));
        let vc = g.cont(v); // [hw, c, n]
        let scores = g.mul_mat(kc, qc); // [hw_k, hw_q, n]
        let scores = g.soft_max_ext(scores, std::ptr::null_mut(), 1.0 / (c as f32).sqrt(), 0.0);
        let o = g.mul_mat(vc, scores); // [c, hw_q, n]
        let o = g.cont(g.permute(o, 1, 0, 2, 3)); // [hw, c, n]
        let o = g.reshape_4d(o, w, h, c, n);
        let o = self.conv1(&format!("{prefix}.proj"), o)?;
        Ok(self.g().add(o, x))
    }

    /// `WanResample` upsample2d/upsample3d: optional causal time conv that
    /// doubles the frames (never the first latent frame), then nearest 2x
    /// spatial upsampling and a 3x3 conv.
    fn upsample(&mut self, prefix: &str, x: Tensor, time: bool) -> Result<Tensor> {
        let mut x = x;
        if time {
            let [w, h, c, n] = ffi::shape(x);
            if self.first_chunk {
                // `Rep`: frame 0 passes through; the next chunk starts from zero padding
                let slot = self.slot;
                self.slot += 1;
                self.keep_cache(slot, None)?;
            } else {
                let y = self.cconv(&format!("{prefix}.time_conv"), x, 0)?; // [w, h, 2c, n]
                                                                           // channels [0, c) are frame 2k, [c, 2c) frame 2k+1: a plain reshape interleaves
                x = self.g().reshape_4d(y, w, h, c, 2 * n);
            }
        }
        let [w, h, c, n] = ffi::shape(x);
        let g = self.g();
        let up = g.interpolate(x, 2 * w, 2 * h, c, n, ffi::GGML_SCALE_MODE_NEAREST);
        self.conv2d(&format!("{prefix}.resample.1"), up)
    }

    /// `DupUp3D` with `factor_t = factor_s = 2`: nearest duplication in t, h and w
    /// (`repeats == factor`), dropping the first frame on the first chunk.
    fn dup_up_3d(&self, x: Tensor) -> Tensor {
        let g = self.g();
        let [w, h, c, n] = ffi::shape(x);
        let up = g.interpolate(x, 2 * w, 2 * h, c, 2 * n, ffi::GGML_SCALE_MODE_NEAREST);
        if self.first_chunk {
            let nb = ffi::strides(up);
            g.cont(g.view_4d(up, 2 * w, 2 * h, c, 2 * n - 1, nb[1], nb[2], nb[3], nb[3]))
        } else {
            up
        }
    }

    /// `DupUp3D` with `factor_t = 1, factor_s = 2, repeats = 2`:
    /// `out[oc, 2h + j, 2w + k] = x[2oc + j, h, w]`.
    fn dup_up_2d(&self, x: Tensor) -> Tensor {
        let g = self.g();
        let [w, h, c, n] = ffi::shape(x);
        let up = g.interpolate(x, 2 * w, h, c, n, ffi::GGML_SCALE_MODE_NEAREST);
        // channel = j + 2*oc: split off j and move it under h
        let s = g.reshape_4d(up, 2 * w, h, 2, (c / 2) * n);
        let s = g.cont(g.permute(s, 0, 2, 1, 3)); // [2w, 2(j), h, ...]
        g.reshape_4d(s, 2 * w, 2 * h, c / 2, n)
    }

    /// `DupUp3D` (`factor_s = 2`) from `x`'s channels to `out_c`:
    /// `repeats = out_c * factor / in_c` picks the variant.
    fn dup_up(&self, x: Tensor, time: bool, out_c: i64) -> Result<Tensor> {
        let c = ffi::shape(x)[2];
        let factor = if time { 8 } else { 4 };
        match (time, out_c * factor / c) {
            (true, 8) => Ok(self.dup_up_3d(x)),
            (true, 4) if self.first_chunk => Ok(self.dup_up_3d_halve(x)),
            (false, 2) => Ok(self.dup_up_2d(x)),
            (_, r) => bail!("unsupported DupUp3D: {c} -> {out_c} channels, repeats {r}"),
        }
    }

    /// `DupUp3D` with `factor_t = factor_s = 2, repeats = 4` on the first chunk
    /// (its first frame dropped): `out[oc, 2h + j, 2w + k] = x[2oc + 1, h, w]`.
    fn dup_up_3d_halve(&self, x: Tensor) -> Tensor {
        let g = self.g();
        let [w, h, c, n] = ffi::shape(x);
        let s = g.reshape_4d(x, w * h, 2, c / 2, n);
        let nb = ffi::strides(s);
        let odd = g.cont(g.view_4d(s, w * h, 1, c / 2, n, nb[1], nb[2], nb[3], nb[1]));
        let odd = g.reshape_4d(odd, w, h, c / 2, n);
        g.interpolate(odd, 2 * w, 2 * h, c / 2, n, ffi::GGML_SCALE_MODE_NEAREST)
    }

    /// `[W, H, 12, F]` -> `[2W, 2H, 3, F]`: `out[c, 2h + pb, 2w + pa] = x[4c + 2pa + pb]`.
    fn unpatchify(&self, x: Tensor) -> Result<Tensor> {
        let g = self.g();
        let [w, h, c12, f] = ffi::shape(x);
        if c12 != 12 {
            bail!("decoder.conv_out has {c12} channels, expected 12");
        }
        // pb (fastest channel index) under h
        let s = g.reshape_4d(x, w, h, 2, 6 * f);
        let s = g.cont(g.permute(s, 0, 2, 1, 3)); // [w, 2(pb), h, 6f]
                                                  // then pa under w
        let s = g.reshape_4d(s, w, 2 * h, 2, 3 * f);
        let s = g.cont(g.permute(s, 1, 2, 0, 3)); // [2(pa), w, 2h, 3f]
        Ok(g.reshape_4d(s, 2 * w, 2 * h, 3, f))
    }

    /// `[2W, 2H, 3, F]` -> `[W, H, 12, F]`, the inverse of [`Self::unpatchify`].
    fn patchify(&self, img: Tensor) -> Tensor {
        let g = self.g();
        let [w2, h2, c, f] = ffi::shape(img);
        debug_assert_eq!(c, 3);
        let (w, h) = (w2 / 2, h2 / 2);
        let s = g.reshape_4d(img, 2, w, h2, 3 * f); // [pa, w, h', 3f]
        let s = g.cont(g.permute(s, 2, 0, 1, 3)); // [w, h', pa, 3f]
        let s = g.reshape_4d(s, w, 2, h, 6 * f); // [w, pb, h, (pa, c, f)]
        let s = g.cont(g.permute(s, 0, 2, 1, 3)); // [w, h, pb, ...]
        g.reshape_4d(s, w, h, 12, f)
    }

    /// 2x2 average pooling of every frame.
    fn pool2(&self, x: Tensor) -> Tensor {
        let g = self.g();
        let [w, h, c, n] = ffi::shape(x);
        let s = g.reshape_4d(x, 2, w / 2, h, c * n);
        let nb = ffi::strides(s);
        let a = g.view_4d(s, 1, w / 2, h, c * n, nb[1], nb[2], nb[3], 0);
        let b = g.view_4d(s, 1, w / 2, h, c * n, nb[1], nb[2], nb[3], nb[0]);
        let s = g.reshape_4d(g.add(a, b), w / 2, 2, h / 2, c * n);
        let nb = ffi::strides(s);
        let a = g.view_4d(s, w / 2, 1, h / 2, c * n, nb[1], nb[2], nb[3], 0);
        let b = g.view_4d(s, w / 2, 1, h / 2, c * n, nb[1], nb[2], nb[3], nb[1]);
        g.scale(g.reshape_4d(g.add(a, b), w / 2, h / 2, c, n), 0.25)
    }

    /// `AvgDown3D` with `factor_s = 2` and `factor_t` 1 or 2: pooled frames, and for
    /// `factor_t = 2` frame pairs `(2t', 2t'+1)` stacked into channels `2c + i` (the
    /// first chunk's single frame is paired with a zero frame in front of it).
    fn avg_down(&mut self, x: Tensor, time: bool) -> Tensor {
        let p = self.pool2(x);
        if !time {
            return p;
        }
        let [w, h, c, n] = ffi::shape(p);
        let (p, n) = if n % 2 == 1 {
            let z = self.zeros([w, h, c, 1], "avgpad");
            (self.g().concat(z, p, 3), n + 1)
        } else {
            (p, n)
        };
        let g = self.g();
        let s = g.reshape_4d(p, w * h, c, 2, n / 2); // [hw, c, i, t']
        let s = g.cont(g.permute(s, 0, 2, 1, 3)); // [hw, i, c, t']
        g.reshape_4d(s, w, h, 2 * c, n / 2)
    }

    /// `WanResample` downsample2d/downsample3d: zero-pad right/bottom by one and a
    /// stride-2 3x3 conv, then (3-D) a stride-2 causal time conv that skips the
    /// first latent frame and halves the rest.
    fn downsample(&mut self, prefix: &str, x: Tensor, time: bool) -> Result<Tensor> {
        let g = self.g();
        let padded = g.pad(x, 1, 1, 0, 0);
        let k = self.w.get(&format!("{prefix}.resample.1.weight"))?;
        let y = self.conv2(k, padded, 2, 0);
        let mut x = self.bias(y, &format!("{prefix}.resample.1.bias"))?;
        if time {
            let [w, h, c, n] = ffi::shape(x);
            if self.first_chunk {
                // the first frame passes through and becomes the next chunk's context
                let slot = self.slot;
                self.slot += 1;
                self.keep_cache(slot, Some(x))?;
            } else {
                let (slot, cache) = self.take_cache(prefix, [w, h, c, 1])?;
                let g = self.g();
                let xx = g.concat(cache, x, 3); // [w, h, c, n + 1]
                let nb = ffi::strides(xx);
                let last = g.view_4d(xx, w, h, c, 1, nb[1], nb[2], nb[3], (n as usize) * nb[3]);
                self.keep_cache(slot, Some(last))?;
                let g = self.g();
                // frames k, k+2, k+4, ... of the concatenation, stacked along the channels
                let tap =
                    |k: usize| g.view_4d(xx, w, h, c, n / 2, nb[1], nb[2], 2 * nb[3], k * nb[3]);
                let stacked = g.concat(g.concat(tap(0), tap(1), 2), tap(2), 2);
                let kt = self.w.get(&format!("{prefix}.time_conv.weight"))?; // [1, 1, 3c, c]
                let y = self.conv2(kt, stacked, 1, 0);
                x = self.bias(y, &format!("{prefix}.time_conv.bias"))?;
            }
        }
        Ok(x)
    }

    /// The encoder over one chunk of pixel frames `[W, H, 3, F]` (F = 1, then 4):
    /// the mean of the posterior, `[W/16, H/16, 48, T]`.
    fn encoder(&mut self, img: Tensor) -> Result<Tensor> {
        let mut x = self.patchify(img);
        x = self.cconv("encoder.conv_in", x, 1)?;
        for b in 0..4 {
            let p = format!("encoder.down_blocks.{b}");
            let x_copy = x;
            for r in 0..2 {
                x = self.resnet(&format!("{p}.resnets.{r}"), x)?;
            }
            match b {
                0 => {
                    x = self.downsample(&format!("{p}.downsampler"), x, false)?;
                    let short = self.avg_down(x_copy, false);
                    x = self.g().add(x, short);
                }
                1 | 2 => {
                    x = self.downsample(&format!("{p}.downsampler"), x, true)?;
                    let short = self.avg_down(x_copy, true);
                    x = self.g().add(x, short);
                }
                _ => x = self.g().add(x, x_copy),
            }
        }
        x = self.resnet("encoder.mid_block.resnets.0", x)?;
        x = self.attention("encoder.mid_block.attentions.0", x)?;
        x = self.resnet("encoder.mid_block.resnets.1", x)?;
        let x = self.norm(x, "encoder.norm_out.gamma")?;
        let x = self.cconv("encoder.conv_out", self.g().silu(x), 1)?;
        let x = self.conv1("quant_conv", x)?; // [w, h, 96, t]: mean, then log-variance
        let g = self.g();
        let [w, h, _, t] = ffi::shape(x);
        let nb = ffi::strides(x);
        Ok(g.cont(g.view_4d(x, w, h, 48, t, nb[1], nb[2], nb[3], 0)))
    }

    fn decoder(&mut self, z: Tensor, layout: &Layout) -> Result<Tensor> {
        let mut x = self.conv1("post_quant_conv", z)?;
        x = self.cconv("decoder.conv_in", x, 1)?;
        x = self.resnet("decoder.mid_block.resnets.0", x)?;
        x = self.attention("decoder.mid_block.attentions.0", x)?;
        x = self.resnet("decoder.mid_block.resnets.1", x)?;
        for b in 0.. {
            let p = format!("decoder.up_blocks.{b}");
            if !self.w.has(&format!("{p}.resnets.0.norm1.gamma")) {
                break;
            }
            let x_copy = x;
            for r in 0..3 {
                x = self.resnet(&format!("{p}.resnets.{r}"), x)?;
            }
            if self.w.has(&format!("{p}.upsampler.resample.1.weight")) {
                let time = layout.temporal_up.get(b).copied().unwrap_or(false);
                x = self.upsample(&format!("{p}.upsampler"), x, time)?;
                let short = self.dup_up(x_copy, time, ffi::shape(x)[2])?;
                x = self.g().add(x, short);
            }
        }
        let x = self.norm(x, "decoder.norm_out.gamma")?;
        let x = self.cconv("decoder.conv_out", self.g().silu(x), 1)?;
        if layout.patch {
            self.unpatchify(x)
        } else {
            Ok(x)
        }
    }
}

/// `ggml_conv_2d_direct` is 4x faster than im2col on Vulkan and 10-30x slower
/// on Metal and CUDA; `MEDIAGEN_DIRECT_CONV=0|1` overrides.
fn direct_conv(be: &Backend) -> bool {
    let wanted = match std::env::var("MEDIAGEN_DIRECT_CONV").as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        _ => be.gpu_name().starts_with("Vulkan"),
    };
    wanted && be.supports_direct_conv()
}

/// Decodes latent frames `z[i]` (`[C][H][W]` each, VAE-space, already denormalized)
/// into pixels, one latent frame per graph.
pub fn decode(
    w: &Weights,
    be: &Backend,
    z: &[Vec<f32>],
    c: i64,
    h: i64,
    wd: i64,
    layout: &Layout,
) -> Result<Chunk> {
    let want = ffi::shape(w.get("post_quant_conv.weight")?)[2];
    if c != want {
        bail!("expected {want} latent channels, got {c}");
    }
    let mut caches = Caches::new(be);
    let mut out = Chunk {
        frames: 0,
        height: 0,
        width: 0,
        channels: 0,
        rgb: Vec::new(),
    };
    let timing = std::env::var_os("MEDIAGEN_TIMING").is_some();
    let direct = direct_conv(be);
    for (i, frame) in z.iter().enumerate() {
        let t_start = std::time::Instant::now();
        let mut gr = Graph::new(be.api, 16 * 1024)?;
        // torch [C][H][W] -> ggml [W, H, C, 1]
        let mut x = vec![0f32; (wd * h * c) as usize];
        for ci in 0..c as usize {
            for hi in 0..h as usize {
                for wi in 0..wd as usize {
                    x[wi + (wd as usize) * (hi + (h as usize) * ci)] =
                        frame[(ci * h as usize + hi) * wd as usize + wi];
                }
            }
        }
        let zt = gr.input_f32(&[wd, h, c, 1], &x, "z");
        let mut b = Build {
            gr: &mut gr,
            w,
            direct,
            caches: &mut caches,
            slot: 0,
            filled: Vec::new(),
            first_chunk: i == 0,
        };
        let rgb = b.decoder(zt, layout)?;
        let rgb = b.g().cont(rgb);
        b.gr.mark_output(rgb);
        let filled = std::mem::take(&mut b.filled);
        drop(b);
        let t_built = t_start.elapsed();
        gr.compute(be)?;
        let t_computed = t_start.elapsed();
        let [ow, oh, nc, of] = ffi::shape(rgb);
        let data = gr.output_f32(rgb);
        if out.frames == 0 {
            out.width = ow;
            out.height = oh;
            out.channels = nc;
        }
        // [w, h, c, f] -> [f][h][w][c]
        let (ow, oh, nc) = (ow as usize, oh as usize, nc as usize);
        for f in 0..of as usize {
            for y in 0..oh {
                for xx in 0..ow {
                    for ch in 0..nc {
                        out.rgb.push(data[xx + ow * (y + oh * (ch + nc * f))]);
                    }
                }
            }
        }
        out.frames += of;
        caches.advance(filled);
        if timing {
            eprintln!(
                "[llmman] mediagen: vae chunk {i}: build {:.2} s, compute {:.2} s, readback {:.2} s",
                t_built.as_secs_f64(),
                (t_computed - t_built).as_secs_f64(),
                (t_start.elapsed() - t_computed).as_secs_f64()
            );
        }
    }
    Ok(out)
}

/// Encodes RGB frames (`[f][h][w][3]`, in `[-1, 1]`, `F = 4k + 1`) to the posterior
/// mean, one latent frame `[48][H/16][W/16]` per 4 pixel frames after the first,
/// chunked as the reference does (frame 0 alone, then groups of 4).
pub fn encode(
    w: &Weights,
    be: &Backend,
    rgb: &[f32],
    frames: i64,
    h: i64,
    wd: i64,
) -> Result<Vec<Vec<f32>>> {
    if frames < 1 || (frames - 1) % 4 != 0 {
        bail!("the encoder takes 4k + 1 frames, got {frames}");
    }
    let mut caches = Caches::new(be);
    let mut out = Vec::new();
    let direct = direct_conv(be);
    let (hu, wu) = (h as usize, wd as usize);
    let mut f0 = 0usize;
    let mut first = true;
    while (f0 as i64) < frames {
        let n = if first { 1 } else { 4 };
        // [f][h][w][3] -> ggml [W, H, 3, n]
        let mut x = vec![0f32; wu * hu * 3 * n];
        for fi in 0..n {
            for y in 0..hu {
                for xx in 0..wu {
                    for ch in 0..3 {
                        x[xx + wu * (y + hu * (ch + 3 * fi))] =
                            rgb[(((f0 + fi) * hu + y) * wu + xx) * 3 + ch];
                    }
                }
            }
        }
        let mut gr = Graph::new(be.api, 16 * 1024)?;
        let img = gr.input_f32(&[wd, h, 3, n as i64], &x, "img");
        let mut b = Build {
            gr: &mut gr,
            w,
            direct,
            caches: &mut caches,
            slot: 0,
            filled: Vec::new(),
            first_chunk: first,
        };
        let z = b.encoder(img)?;
        b.gr.mark_output(z);
        let filled = std::mem::take(&mut b.filled);
        drop(b);
        gr.compute(be)?;
        let [lw, lh, c, lt] = ffi::shape(z);
        let data = gr.output_f32(z);
        let (lw, lh, c) = (lw as usize, lh as usize, c as usize);
        for t in 0..lt as usize {
            let mut frame = vec![0f32; c * lh * lw];
            for ci in 0..c {
                for y in 0..lh {
                    for xx in 0..lw {
                        frame[(ci * lh + y) * lw + xx] = data[xx + lw * (y + lh * (ci + c * t))];
                    }
                }
            }
            out.push(frame);
        }
        caches.advance(filled);
        f0 += n;
        first = false;
    }
    Ok(out)
}

/// Rewrites the VAE as `dst` with every 5-D conv weight `[oc, ic, kt, kh, kw]`
/// merged to `[oc, kt*ic, kh, kw]` (tap major), so a causal 3-D conv is one 2-D
/// conv over the frames stacked along the channels ([`Build::stack_taps`]).
pub fn merge_time_taps(src: &Path, dst: &Path) -> Result<()> {
    let mut tensors = read_safetensors_raw(src)?;
    for t in &mut tensors {
        if t.shape.len() != 5 {
            continue;
        }
        let esz = match t.dtype.as_str() {
            "BF16" | "F16" => 2usize,
            "F32" => 4,
            other => bail!("{}: unsupported dtype {other}", t.name),
        };
        let [oc, ic, kt, kh, kw] = [0, 1, 2, 3, 4].map(|i| t.shape[i] as usize);
        let spatial = kh * kw * esz;
        let mut merged = vec![0u8; t.bytes.len()];
        for o in 0..oc {
            for i in 0..ic {
                for k in 0..kt {
                    let from = ((o * ic + i) * kt + k) * spatial;
                    let to = ((o * kt + k) * ic + i) * spatial;
                    merged[to..to + spatial].copy_from_slice(&t.bytes[from..from + spatial]);
                }
            }
        }
        t.bytes = merged;
        t.shape = vec![oc as i64, (kt * ic) as i64, kh as i64, kw as i64];
    }
    write_safetensors_raw(dst, &tensors, "time taps merged")
}

/// The tap-merged file for `src`, created in `cache_dir` when missing.
pub fn merged_path(src: &Path, cache_dir: &Path) -> Result<std::path::PathBuf> {
    let name = src.file_stem().and_then(|s| s.to_str()).unwrap_or("vae");
    let dst = cache_dir.join(format!("{name}.merged.safetensors"));
    if !dst.exists() {
        std::fs::create_dir_all(cache_dir)?;
        let t0 = std::time::Instant::now();
        merge_time_taps(src, &dst)
            .with_context(|| format!("merging the time taps of {}", src.display()))?;
        eprintln!(
            "[llmman] mediagen: merged the VAE's conv time taps into {} in {:.1} s",
            dst.display(),
            t0.elapsed().as_secs_f64()
        );
    }
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merged_taps_are_tap_major() {
        let dir = std::env::temp_dir().join(format!("llmman-merge-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // [oc=1, ic=2, kt=3, kh=1, kw=1]: value = ic*10 + kt
        let vals: Vec<f32> = (0..2)
            .flat_map(|i| (0..3).map(move |t| (i * 10 + t) as f32))
            .collect();
        let src = dir.join("in.safetensors");
        write_safetensors_raw(
            &src,
            &[crate::mediagen::weights::RawTensor {
                name: "k.weight".into(),
                dtype: "F32".into(),
                shape: vec![1, 2, 3, 1, 1],
                bytes: vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
            }],
            "test",
        )
        .unwrap();
        let dst = dir.join("out.safetensors");
        merge_time_taps(&src, &dst).unwrap();
        let out = read_safetensors_raw(&dst).unwrap();
        assert_eq!(out[0].shape, [1, 6, 1, 1]);
        // channel = kt * ic + i
        assert_eq!(out[0].f32s().unwrap(), [0.0, 10.0, 1.0, 11.0, 2.0, 12.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
