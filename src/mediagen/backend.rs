//! Backend, scheduler and graph building on top of [`ffi::Api`].

use std::cell::Cell;
use std::cell::RefCell;
use std::ffi::{c_int, CString};
use std::marker::PhantomData;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Result};

use super::ffi::{self, Api, GgmlBackend, GgmlBackendBuft, GgmlBackendSched, GgmlContext, Tensor};

/// The compute backends: the GPU when there is one, always the CPU.
pub struct Backend {
    pub api: &'static Api,
    pub gpu: GgmlBackend,
    pub cpu: GgmlBackend,
    /// Buffer type the weights live on: the GPU's when there is one.
    weight_buft: GgmlBackendBuft,
    sched: GgmlBackendSched,
    max_nodes: usize,
    /// Empty graph run after every real one: Metal reports a failed
    /// command buffer (usually out of memory) only on the next compute.
    probe: Option<(Ctx, *mut ffi::GgmlCgraph)>,
    /// The GPU failed and must be recreated by [`Backend::release_compute`].
    gpu_failed: Cell<bool>,
    /// Polled by the CPU backend after every node once
    /// [`Backend::set_abort_flag`] registered it, and read by
    /// [`Backend::cancelled`]. Held here so the pointer ggml keeps stays
    /// valid for the backend's lifetime.
    abort: Option<Arc<AtomicBool>>,
}

unsafe impl Send for Backend {}

impl Backend {
    pub fn new(
        api: &'static Api,
        use_gpu: bool,
        n_threads: i32,
        max_nodes: usize,
    ) -> Result<Backend> {
        unsafe {
            let cpu =
                (api.ggml_backend_init_by_type)(ffi::GGML_BACKEND_DEVICE_TYPE_CPU, ptr::null());
            if cpu.is_null() {
                return Err(anyhow!("failed to initialize the CPU backend"));
            }
            let gpu = if use_gpu {
                init_gpu(api)
            } else {
                ptr::null_mut()
            };
            let mut probe = None;
            let weight_buft = if !gpu.is_null() {
                let name = std::ffi::CStr::from_ptr((api.ggml_backend_name)(gpu)).to_string_lossy();
                eprintln!("[llmman] mediagen: using {name} backend");
                let ctx = Ctx::new(api, (api.ggml_graph_overhead_custom)(1, false) + 1024, true)?;
                let gf = (api.ggml_new_graph_custom)(ctx.raw, 1, false);
                if gf.is_null() {
                    return Err(anyhow!("ggml_new_graph_custom failed"));
                }
                probe = Some((ctx, gf));
                (api.ggml_backend_get_default_buffer_type)(gpu)
            } else {
                eprintln!("[llmman] mediagen: using CPU backend");
                (api.ggml_backend_get_default_buffer_type)(cpu)
            };
            api.set_n_threads(cpu, n_threads);
            let mut be = Backend {
                api,
                gpu,
                cpu,
                weight_buft,
                sched: ptr::null_mut(),
                max_nodes,
                probe,
                gpu_failed: Cell::new(false),
                abort: None,
            };
            be.release_compute()?;
            Ok(be)
        }
    }

    /// Buffer type the weights are allocated on.
    pub fn weight_buft(&self) -> GgmlBackendBuft {
        self.weight_buft
    }

    /// Nodes plus leafs the scheduler can take in one graph.
    pub fn max_nodes(&self) -> usize {
        self.max_nodes
    }

    /// Free and total GPU memory in bytes as the backend reports it (Metal:
    /// the process's working set); `None` without a GPU.
    pub fn gpu_memory(&self) -> Option<(usize, usize)> {
        (!self.gpu.is_null()).then(|| self.dev_memory(self.gpu))
    }

    /// Free and total bytes where the weights live: the GPU's, else host memory.
    pub fn weight_memory(&self) -> (usize, usize) {
        self.dev_memory(if self.gpu.is_null() {
            self.cpu
        } else {
            self.gpu
        })
    }

    fn dev_memory(&self, backend: GgmlBackend) -> (usize, usize) {
        let (mut free, mut total) = (0usize, 0usize);
        unsafe {
            let dev = (self.api.ggml_backend_get_device)(backend);
            (self.api.ggml_backend_dev_memory)(dev, &mut free, &mut total);
        }
        (free, total)
    }

    /// The GPU backend's name (`"Metal"`, `"CUDA0"`, `"Vulkan0"`, ...), `""` without one.
    pub fn gpu_name(&self) -> String {
        if self.gpu.is_null() {
            return String::new();
        }
        unsafe { std::ffi::CStr::from_ptr((self.api.ggml_backend_name)(self.gpu)) }
            .to_string_lossy()
            .into_owned()
    }

    /// Does the GPU run `ggml_conv_2d_direct` (F16 kernel), or would it fall back to CPU?
    pub fn supports_direct_conv(&self) -> bool {
        let (Some(conv), Some(supports)) = (
            self.api.ggml_conv_2d_direct,
            self.api.ggml_backend_supports_op,
        ) else {
            return false;
        };
        if self.gpu.is_null() {
            return false;
        }
        let Ok(ctx) = Ctx::new(self.api, 16 * 1024, true) else {
            return false;
        };
        let k = ctx.new_tensor(ffi::ty::F16, &[3, 3, 8, 8]);
        let x = ctx.new_tensor(ffi::ty::F32, &[16, 16, 8, 2]);
        let y = unsafe { conv(ctx.raw, k.as_ptr(), x.as_ptr(), 1, 1, 1, 1, 1, 1) };
        unsafe { supports(self.gpu, y) }
    }

    /// Recreates the scheduler, freeing its compute buffers, and the GPU
    /// backend if it is in an error state.
    pub fn release_compute(&mut self) -> Result<()> {
        unsafe {
            if !self.sched.is_null() {
                (self.api.ggml_backend_sched_free)(self.sched);
                self.sched = ptr::null_mut();
            }
            if self.gpu_failed.replace(false) && !self.gpu.is_null() {
                (self.api.ggml_backend_free)(self.gpu);
                self.gpu = init_gpu(self.api);
                if self.gpu.is_null() {
                    return Err(anyhow!(
                        "failed to recreate the GPU backend after a failure"
                    ));
                }
                eprintln!("[llmman] mediagen: recreated the GPU backend after a failure");
            }
            let mut ptrs = Vec::new();
            if !self.gpu.is_null() {
                ptrs.push(self.gpu);
            }
            ptrs.push(self.cpu);
            let mut bufts: Vec<GgmlBackendBuft> = ptrs
                .iter()
                .map(|&b| (self.api.ggml_backend_get_default_buffer_type)(b))
                .collect();
            self.sched = (self.api.ggml_backend_sched_new)(
                ptrs.as_mut_ptr(),
                bufts.as_mut_ptr(),
                ptrs.len() as c_int,
                self.max_nodes,
                false,
                true,
            );
        }
        if self.sched.is_null() {
            return Err(anyhow!("ggml_backend_sched_new failed"));
        }
        Ok(())
    }

    /// Reads an F32 tensor allocated on a backend buffer.
    pub fn read_f32(&self, t: Tensor<'_>) -> Vec<f32> {
        read_f32(self.api, t)
    }

    /// Makes every CPU compute stop after the current node while `flag` is
    /// set; [`Graph::compute`] then fails with "graph compute cancelled".
    /// The caller clears the flag before the next run it wants to finish.
    pub fn set_abort_flag(&mut self, flag: Arc<AtomicBool>) {
        let data = Arc::as_ptr(&flag) as *mut std::ffi::c_void;
        self.abort = Some(flag);
        self.api
            .set_abort_callback(self.cpu, Some(abort_requested), data);
    }

    /// Is the flag of [`Backend::set_abort_flag`] set? False without one.
    pub fn cancelled(&self) -> bool {
        self.abort
            .as_ref()
            .is_some_and(|f| f.load(Ordering::Relaxed))
    }

    /// Asks the GPU whether its last compute actually succeeded.
    fn check_gpu(&self) -> Result<()> {
        let Some((_, gf)) = &self.probe else {
            return Ok(());
        };
        if self.gpu.is_null() {
            return Ok(());
        }
        let status = unsafe { (self.api.ggml_backend_graph_compute)(self.gpu, *gf) };
        if status != ffi::GGML_STATUS_SUCCESS {
            self.gpu_failed.set(true);
            return Err(anyhow!(
                "the GPU failed to compute the graph (out of memory?), status {status}"
            ));
        }
        Ok(())
    }
}

/// The [`ffi::AbortCallback`] behind [`Backend::set_abort_flag`]; `data`
/// is the `AtomicBool` the backend keeps alive.
unsafe extern "C" fn abort_requested(data: *mut std::ffi::c_void) -> bool {
    (*(data as *const AtomicBool)).load(Ordering::Relaxed)
}

/// Copies an F32 tensor out of its backend buffer.
fn read_f32(api: &Api, t: Tensor<'_>) -> Vec<f32> {
    let n = unsafe { (api.ggml_nelements)(t.as_ptr()) } as usize;
    let mut out = vec![0f32; n];
    unsafe { (api.ggml_backend_tensor_get)(t.as_ptr(), out.as_mut_ptr() as *mut _, 0, n * 4) };
    out
}

fn init_gpu(api: &Api) -> GgmlBackend {
    unsafe {
        let gpu = (api.ggml_backend_init_by_type)(ffi::GGML_BACKEND_DEVICE_TYPE_GPU, ptr::null());
        if !gpu.is_null() {
            return gpu;
        }
        (api.ggml_backend_init_by_type)(ffi::GGML_BACKEND_DEVICE_TYPE_IGPU, ptr::null())
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        unsafe {
            if !self.sched.is_null() {
                (self.api.ggml_backend_sched_free)(self.sched);
            }
            self.probe = None;
            if !self.gpu.is_null() {
                (self.api.ggml_backend_free)(self.gpu);
            }
            (self.api.ggml_backend_free)(self.cpu);
        }
    }
}

/// A ggml context: the building block under [`Graph`], [`Resident`] and
/// `Weights`, which own one each and hand out its tensors as handles
/// borrowed from themselves. Nothing outside this module reaches it, so
/// no handle can be made without an owner to borrow.
pub struct Ctx {
    pub(super) api: &'static Api,
    pub(super) raw: *mut GgmlContext,
}

impl Drop for Ctx {
    fn drop(&mut self) {
        unsafe { (self.api.ggml_free)(self.raw) }
    }
}

// `.into()` turns a `Tensor` handle into the raw pointer the FFI takes;
// scalar arguments go through the identity conversion, so keep each
// entry's scalar types equal to the `api!` signature.
macro_rules! ops {
    ($($name:ident => $sym:ident($($arg:ident: $ty:ty),*);)*) => {
        impl<'o> Graph<'o> {
            $(pub fn $name(&'o self, $($arg: $ty),*) -> Tensor<'o> {
                unsafe {
                    Tensor::from_raw((self.ctx.api.$sym)(self.ctx.raw, $($arg.into()),*), self)
                }
            })*
        }
    };
}

ops! {
    add => ggml_add(a: Tensor<'o>, b: Tensor<'o>);
    sub => ggml_sub(a: Tensor<'o>, b: Tensor<'o>);
    mul => ggml_mul(a: Tensor<'o>, b: Tensor<'o>);
    scale => ggml_scale(a: Tensor<'o>, s: f32);
    scale_bias => ggml_scale_bias(a: Tensor<'o>, s: f32, b: f32);
    mul_mat => ggml_mul_mat(a: Tensor<'o>, b: Tensor<'o>);
    rms_norm => ggml_rms_norm(a: Tensor<'o>, eps: f32);
    norm => ggml_norm(a: Tensor<'o>, eps: f32);
    silu => ggml_silu(a: Tensor<'o>);
    relu => ggml_relu(a: Tensor<'o>);
    gelu => ggml_gelu(a: Tensor<'o>);
    sigmoid => ggml_sigmoid(a: Tensor<'o>);
    tanh => ggml_tanh(a: Tensor<'o>);
    sin => ggml_sin(a: Tensor<'o>);
    sqr => ggml_sqr(a: Tensor<'o>);
    sqrt => ggml_sqrt(a: Tensor<'o>);
    log => ggml_log(a: Tensor<'o>);
    clamp => ggml_clamp(a: Tensor<'o>, min: f32, max: f32);
    reshape_2d => ggml_reshape_2d(a: Tensor<'o>, ne0: i64, ne1: i64);
    reshape_3d => ggml_reshape_3d(a: Tensor<'o>, ne0: i64, ne1: i64, ne2: i64);
    reshape_4d => ggml_reshape_4d(a: Tensor<'o>, ne0: i64, ne1: i64, ne2: i64, ne3: i64);
    view_2d => ggml_view_2d(a: Tensor<'o>, ne0: i64, ne1: i64, nb1: usize, offset: usize);
    view_3d => ggml_view_3d(a: Tensor<'o>, ne0: i64, ne1: i64, ne2: i64, nb1: usize, nb2: usize, offset: usize);
    view_4d => ggml_view_4d(a: Tensor<'o>, ne0: i64, ne1: i64, ne2: i64, ne3: i64, nb1: usize, nb2: usize, nb3: usize, offset: usize);
    permute => ggml_permute(a: Tensor<'o>, a0: c_int, a1: c_int, a2: c_int, a3: c_int);
    transpose => ggml_transpose(a: Tensor<'o>);
    cont => ggml_cont(a: Tensor<'o>);
    cont_2d => ggml_cont_2d(a: Tensor<'o>, ne0: i64, ne1: i64);
    concat => ggml_concat(a: Tensor<'o>, b: Tensor<'o>, dim: c_int);
    cast => ggml_cast(a: Tensor<'o>, ty: u32);
    get_rows => ggml_get_rows(a: Tensor<'o>, b: Tensor<'o>);
    repeat => ggml_repeat(a: Tensor<'o>, b: Tensor<'o>);
    pad => ggml_pad(a: Tensor<'o>, p0: c_int, p1: c_int, p2: c_int, p3: c_int);
    pad_ext => ggml_pad_ext(a: Tensor<'o>, lp0: c_int, rp0: c_int, lp1: c_int, rp1: c_int, lp2: c_int, rp2: c_int, lp3: c_int, rp3: c_int);
    conv_1d => ggml_conv_1d(a: Tensor<'o>, b: Tensor<'o>, s0: c_int, p0: c_int, d0: c_int);
    conv_2d => ggml_conv_2d(a: Tensor<'o>, b: Tensor<'o>, s0: c_int, s1: c_int, p0: c_int, p1: c_int, d0: c_int, d1: c_int);
    conv_transpose_1d => ggml_conv_transpose_1d(a: Tensor<'o>, b: Tensor<'o>, s0: c_int, p0: c_int, d0: c_int);
    conv_2d_dw_direct => ggml_conv_2d_dw_direct(a: Tensor<'o>, b: Tensor<'o>, stride0: c_int, stride1: c_int, pad0: c_int, pad1: c_int, dilation0: c_int, dilation1: c_int);
    ssm_conv => ggml_ssm_conv(sx: Tensor<'o>, c: Tensor<'o>);
    interpolate => ggml_interpolate(a: Tensor<'o>, ne0: i64, ne1: i64, ne2: i64, ne3: i64, mode: u32);
    timestep_embedding => ggml_timestep_embedding(t: Tensor<'o>, dim: c_int, max_period: c_int);
}

/// `None` is ggml's null, the "no tensor" argument of an optional input.
fn opt_ptr(t: Option<Tensor<'_>>) -> ffi::TensorPtr {
    t.map_or(ptr::null_mut(), Tensor::as_ptr)
}

impl<'o> Graph<'o> {
    pub fn flash_attn_ext(
        &'o self,
        q: Tensor<'o>,
        k: Tensor<'o>,
        v: Tensor<'o>,
        mask: Option<Tensor<'o>>,
        scale: f32,
        max_bias: f32,
        logit_softcap: f32,
    ) -> Tensor<'o> {
        unsafe {
            Tensor::from_raw(
                (self.ctx.api.ggml_flash_attn_ext)(
                    self.ctx.raw,
                    q.as_ptr(),
                    k.as_ptr(),
                    v.as_ptr(),
                    opt_ptr(mask),
                    scale,
                    max_bias,
                    logit_softcap,
                ),
                self,
            )
        }
    }

    pub fn soft_max_ext(
        &'o self,
        a: Tensor<'o>,
        mask: Option<Tensor<'o>>,
        scale: f32,
        max_bias: f32,
    ) -> Tensor<'o> {
        unsafe {
            Tensor::from_raw(
                (self.ctx.api.ggml_soft_max_ext)(
                    self.ctx.raw,
                    a.as_ptr(),
                    opt_ptr(mask),
                    scale,
                    max_bias,
                ),
                self,
            )
        }
    }
}

impl Ctx {
    pub(super) fn new(api: &'static Api, mem_size: usize, no_alloc: bool) -> Result<Ctx> {
        let raw = unsafe {
            (api.ggml_init)(ffi::GgmlInitParams {
                mem_size,
                mem_buffer: ptr::null_mut(),
                no_alloc,
            })
        };
        if raw.is_null() {
            return Err(anyhow!("ggml_init failed"));
        }
        Ok(Ctx { api, raw })
    }

    /// A tensor of this context, borrowed from it; an owner wraps the
    /// result in its own borrow before it leaves the module.
    pub(super) fn new_tensor(&self, ty: u32, ne: &[i64]) -> Tensor<'_> {
        unsafe {
            Tensor::from_raw(
                (self.api.ggml_new_tensor)(self.raw, ty, ne.len() as c_int, ne.as_ptr()),
                self,
            )
        }
    }

    pub(super) fn set_name(&self, t: Tensor<'_>, name: &str) {
        let name = CString::new(name).unwrap_or_default();
        unsafe { (self.api.ggml_set_name)(t.as_ptr(), name.as_ptr()) };
    }
}

impl<'o> Graph<'o> {
    /// The ggml API the graph was built with, for calls that take a
    /// node's pointer and nothing of the context.
    pub fn api(&'o self) -> &'static Api {
        self.ctx.api
    }

    /// A new tensor of the graph's context, for a shape another op
    /// repeats into; it carries no data.
    pub fn new_tensor(&'o self, ty: u32, ne: &[i64]) -> Tensor<'o> {
        // SAFETY: the tensor is a node of `ctx`, which lives as long as self.
        unsafe { Tensor::from_raw(self.ctx.new_tensor(ty, ne).as_ptr(), self) }
    }

    pub fn set_name(&'o self, t: Tensor<'o>, name: &str) {
        self.ctx.set_name(t, name);
    }

    /// x * (1 + scale) + shift; not `x * scale + x`, which Metal's
    /// norm + mul + add fusion miscomputes when `x` is a norm's output.
    pub fn modulate(
        &'o self,
        x: Tensor<'o>,
        scale: Tensor<'o>,
        shift: Option<Tensor<'o>>,
    ) -> Tensor<'o> {
        let y = self.mul(x, self.scale_bias(scale, 1.0, 1.0));
        match shift {
            Some(s) => self.add(y, s),
            None => y,
        }
    }

    pub fn linear(&'o self, x: Tensor<'o>, w: Tensor<'o>, b: Option<Tensor<'o>>) -> Tensor<'o> {
        let y = self.mul_mat(w, x);
        match b {
            Some(b) => self.add(y, b),
            None => y,
        }
    }

    /// Asks for F32 accumulation in the matmul (or flash attention) `t`
    /// and returns it; ggml ignores the hint on any other op. A release
    /// without `ggml_prec_set_acc` leaves `t` as it is: the speech encoder
    /// checks for the symbol before it builds a graph.
    pub fn prec_f32(&'o self, t: Tensor<'o>) -> Tensor<'o> {
        if let Some(set_acc) = self.ctx.api.ggml_prec_set_acc {
            unsafe { set_acc(t.as_ptr(), ffi::GGML_PREC_F32) };
        }
        t
    }

    /// `ggml_set_rows` when this release has the symbol, else None:
    /// writes the rows of `b` into `a` at the row indices `c`.
    pub fn set_rows(&'o self, a: Tensor<'o>, b: Tensor<'o>, c: Tensor<'o>) -> Option<Tensor<'o>> {
        let f = self.ctx.api.ggml_set_rows?;
        Some(unsafe { Tensor::from_raw(f(self.ctx.raw, a.as_ptr(), b.as_ptr(), c.as_ptr()), self) })
    }

    /// `ggml_conv_2d_direct` when this release has the symbol, else None;
    /// the kernel `k` over `x` with stride `s` and padding `pad` on both
    /// axes.
    pub fn conv_2d_direct(
        &'o self,
        k: Tensor<'o>,
        x: Tensor<'o>,
        s: i32,
        pad: i32,
    ) -> Option<Tensor<'o>> {
        let f = self.ctx.api.ggml_conv_2d_direct?;
        Some(unsafe {
            Tensor::from_raw(
                f(self.ctx.raw, k.as_ptr(), x.as_ptr(), s, s, pad, pad, 1, 1),
                self,
            )
        })
    }
}

/// Tensors in their own backend buffer, outliving the graphs that use them:
/// caches one graph writes ([`Graph::store`]) and the next ones read.
pub struct Resident {
    ctx: Ctx,
    buf: ffi::GgmlBackendBuffer,
    /// Raw, since a handle would borrow `ctx` inside the same struct;
    /// [`Resident::tensor`] hands them out borrowed from `self`.
    tensors: Vec<ffi::TensorPtr>,
}

unsafe impl Send for Resident {}

impl Resident {
    pub fn new(be: &Backend, ty: u32, shapes: &[&[i64]]) -> Result<Resident> {
        let api = be.api;
        let ctx = Ctx::new(
            api,
            unsafe { (api.ggml_tensor_overhead)() } * (shapes.len() + 1),
            true,
        )?;
        let tensors = shapes
            .iter()
            .map(|ne| ctx.new_tensor(ty, ne).as_ptr())
            .collect();
        let buf =
            unsafe { (api.ggml_backend_alloc_ctx_tensors_from_buft)(ctx.raw, be.weight_buft) };
        if buf.is_null() {
            return Err(anyhow!(
                "allocating {} resident tensors failed",
                shapes.len()
            ));
        }
        Ok(Resident { ctx, buf, tensors })
    }

    /// Zeroes every tensor, so a graph reading a window past the rows
    /// written so far (a decoder step over its full cache) meets zeros,
    /// not whatever the allocation held.
    pub fn clear(&self) {
        unsafe { (self.ctx.api.ggml_backend_buffer_clear)(self.buf, 0) }
    }

    /// The `i`th tensor, in the order of the shapes given to [`Resident::new`].
    pub fn tensor(&self, i: usize) -> Tensor<'_> {
        // SAFETY: the pointer came from `ctx`, which lives as long as self.
        unsafe { Tensor::from_raw(self.tensors[i], self) }
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

impl Drop for Resident {
    fn drop(&mut self) {
        unsafe { (self.ctx.api.ggml_backend_buffer_free)(self.buf) }
    }
}

/// One graph: inputs uploaded after allocation, outputs read after compute.
/// Inputs are numbered in the order they are added; [`Graph::input`]
/// reads one back and the `update_input` methods replace its bytes.
///
/// `'o` is the graph's own scope, and every owner whose tensors enter
/// the graph must outlive it. Every method takes `&'o self` and tensors
/// borrowed for `'o`: a node an op returns cannot outlive the graph, an
/// operand from a `Resident` or `Weights` must outlive the graph's last
/// use, compute included, and so must anything [`Graph::mark_output`] is
/// given, so the graph never computes over or reads back freed memory.
/// The context is private: only these methods create tensors in it.
///
/// An operand's owner cannot go before the graph that uses it:
///
/// ```compile_fail,E0505
/// use llmman::mediagen::backend::{Backend, Graph};
/// use llmman::mediagen::ffi::Api;
/// use llmman::mediagen::weights::Weights;
///
/// fn f(api: &'static Api, be: &Backend, w: Weights) -> anyhow::Result<()> {
///     let g = Graph::new(api, 16)?;
///     let v = g.view_2d(w.get("a")?, 1, 1, 4, 0);
///     drop(w);
///     g.mark_output(v);
///     g.compute(be)
/// }
/// ```
///
/// Nor can an output come from an owner that goes before the graph:
///
/// ```compile_fail,E0597
/// use llmman::mediagen::backend::{Backend, Graph, Resident};
/// use llmman::mediagen::ffi::{ty, Api};
///
/// fn f(api: &'static Api, be: &Backend) -> anyhow::Result<usize> {
///     let g = Graph::new(api, 16)?;
///     let i = {
///         let r = Resident::new(be, ty::F32, &[&[1]])?;
///         g.mark_output(r.tensor(0))
///     };
///     g.compute(be)?;
///     Ok(i)
/// }
/// ```
///
/// And the context behind the graph is not reachable:
///
/// ```compile_fail,E0616
/// use llmman::mediagen::backend::Graph;
///
/// fn f(g: &Graph<'_>) {
///     let _ = &g.ctx;
/// }
/// ```
///
/// With every owner outliving the graph, the same shapes compile:
///
/// ```
/// use llmman::mediagen::backend::{Backend, Graph, Resident};
/// use llmman::mediagen::ffi::{ty, Api};
/// use llmman::mediagen::weights::Weights;
///
/// fn f(api: &'static Api, be: &Backend, w: &Weights) -> anyhow::Result<Vec<f32>> {
///     let r = Resident::new(be, ty::F32, &[&[1]])?;
///     let g = Graph::new(api, 16)?;
///     let v = g.view_2d(w.get("a")?, 1, 1, 4, 0);
///     let i = g.mark_output(v);
///     g.store(v, r.tensor(0));
///     g.compute(be)?;
///     Ok(g.output_f32(g.output(i)))
/// }
/// ```
pub struct Graph<'o> {
    ctx: Ctx,
    gf: *mut ffi::GgmlCgraph,
    /// Raw, since a handle would borrow `ctx` inside the same struct.
    /// Behind a `RefCell` so adding an input borrows the graph like
    /// building a node does, and the handles it returns stay usable.
    inputs: RefCell<Vec<(ffi::TensorPtr, Vec<u8>)>>,
    /// The tensors [`Graph::mark_output`] was given, in order, so a
    /// builder can hand its graph back and name an output by index.
    outputs: RefCell<Vec<ffi::TensorPtr>>,
    /// The scope the owners of every tensor in the graph outlive.
    /// Invariant in `'o`, so a method call cannot shrink it to its own
    /// borrow: one `'o` spans every use of the graph, compute included.
    owners: PhantomData<fn(&'o ()) -> &'o ()>,
}

impl<'o> Graph<'o> {
    pub fn new(api: &'static Api, max_nodes: usize) -> Result<Graph<'o>> {
        let mem = unsafe {
            (api.ggml_tensor_overhead)() * max_nodes
                + (api.ggml_graph_overhead_custom)(max_nodes, false)
        };
        let ctx = Ctx::new(api, mem, true)?;
        let gf = unsafe { (api.ggml_new_graph_custom)(ctx.raw, max_nodes, false) };
        if gf.is_null() {
            return Err(anyhow!("ggml_new_graph_custom failed"));
        }
        Ok(Graph {
            ctx,
            gf,
            inputs: RefCell::new(Vec::new()),
            outputs: RefCell::new(Vec::new()),
            owners: PhantomData,
        })
    }

    /// Adds an F32 input holding `data`; the handle and its index, for
    /// [`Graph::update_input_f32`], count up from zero.
    pub fn input_f32(&'o self, ne: &[i64], data: &[f32], name: &str) -> Tensor<'o> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_input(ffi::ty::F32, ne, bytes, name)
    }

    /// Adds an I32 input holding `data`; see [`Graph::input_f32`].
    pub fn input_i32(&'o self, ne: &[i64], data: &[i32], name: &str) -> Tensor<'o> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_input(ffi::ty::I32, ne, bytes, name)
    }

    fn add_input(&'o self, ty: u32, ne: &[i64], bytes: Vec<u8>, name: &str) -> Tensor<'o> {
        let t = self.new_tensor(ty, ne);
        self.set_name(t, name);
        unsafe { (self.ctx.api.ggml_set_input)(t.as_ptr()) };
        debug_assert_eq!(
            unsafe { (self.ctx.api.ggml_nbytes)(t.as_ptr()) },
            bytes.len()
        );
        self.inputs.borrow_mut().push((t.as_ptr(), bytes));
        t
    }

    /// The `i`th input added, counting from zero.
    pub fn input(&'o self, i: usize) -> Tensor<'o> {
        let ptr = self.inputs.borrow()[i].0;
        // SAFETY: the pointer came from `ctx`, which lives as long as self.
        unsafe { Tensor::from_raw(ptr, self) }
    }

    /// How many inputs the graph has.
    pub fn n_inputs(&'o self) -> usize {
        self.inputs.borrow().len()
    }

    /// Replaces the bytes of input `i` for the next [`Graph::compute`] in
    /// place: a graph built once runs again on new data without
    /// allocating. Panics when there is no such input or the lengths
    /// differ.
    pub fn update_input(&'o self, i: usize, bytes: &[u8]) {
        self.fill_input(i, bytes.len(), |slot| slot.copy_from_slice(bytes));
    }

    /// The one lookup behind every input update: borrows the inputs,
    /// finds slot `i`, checks it holds `len` bytes and hands it to `fill`
    /// to write in place. Panics on a missing input first, then on a
    /// length that differs.
    fn fill_input(&'o self, i: usize, len: usize, fill: impl FnOnce(&mut [u8])) {
        let mut inputs = self.inputs.borrow_mut();
        let slot = &mut inputs.get_mut(i).expect("no such graph input").1;
        assert_eq!(slot.len(), len, "input size changed");
        fill(slot);
    }

    /// [`Graph::update_input`] for an F32 input.
    pub fn update_input_f32(&'o self, i: usize, data: &[f32]) {
        self.update_words(i, data.iter().map(|v| v.to_le_bytes()));
    }

    /// [`Graph::update_input`] for an I32 input.
    pub fn update_input_i32(&'o self, i: usize, data: &[i32]) {
        self.update_words(i, data.iter().map(|v| v.to_le_bytes()));
    }

    /// [`Graph::fill_input`] with four-byte words written in place.
    fn update_words(&'o self, i: usize, words: impl ExactSizeIterator<Item = [u8; 4]>) {
        self.fill_input(i, words.len() * 4, |slot| {
            for (dst, w) in slot.as_chunks_mut::<4>().0.iter_mut().zip(words) {
                *dst = w;
            }
        });
    }

    /// Copies `src` into the [`Resident`] tensor `dst` when the graph runs.
    pub fn store(&'o self, src: Tensor<'o>, dst: Tensor<'o>) {
        self.mark_output(unsafe {
            Tensor::from_raw(
                (self.ctx.api.ggml_cpy)(self.ctx.raw, src.as_ptr(), dst.as_ptr()),
                self,
            )
        });
    }

    /// Marks `t` as an output the graph computes, and returns its index
    /// for [`Graph::output`]. `t` is a node of this graph or a tensor of
    /// an owner that outlives it, so reading it back is always sound.
    pub fn mark_output(&'o self, t: Tensor<'o>) -> usize {
        unsafe {
            (self.ctx.api.ggml_set_output)(t.as_ptr());
            (self.ctx.api.ggml_build_forward_expand)(self.gf, t.as_ptr());
        }
        let mut outputs = self.outputs.borrow_mut();
        outputs.push(t.as_ptr());
        outputs.len() - 1
    }

    /// The `i`th tensor given to [`Graph::mark_output`].
    pub fn output(&'o self, i: usize) -> Tensor<'o> {
        let ptr = self.outputs.borrow()[i];
        // SAFETY: the pointer came from a tensor borrowed for `'o` when
        // it was marked, so its owner outlives this graph.
        unsafe { Tensor::from_raw(ptr, self) }
    }

    /// Compute buffer bytes this graph needs, planned by a CPU graph
    /// allocator: nothing is allocated on the GPU, and the host memory is
    /// only reserved, not touched.
    pub fn measure(&'o self, be: &Backend) -> Result<usize> {
        let api = self.ctx.api;
        unsafe {
            let buft = (api.ggml_backend_get_default_buffer_type)(be.cpu);
            let galloc = (api.ggml_gallocr_new)(buft);
            if galloc.is_null() {
                return Err(anyhow!("ggml_gallocr_new failed"));
            }
            let ok = (api.ggml_gallocr_reserve)(galloc, self.gf);
            let size = if ok {
                (api.ggml_gallocr_get_buffer_size)(galloc, 0)
            } else {
                usize::MAX
            };
            (api.ggml_gallocr_free)(galloc);
            Ok(size)
        }
    }

    pub fn compute(&'o self, be: &Backend) -> Result<()> {
        let api = self.ctx.api;
        unsafe {
            (api.ggml_backend_sched_reset)(be.sched);
            if !(api.ggml_backend_sched_alloc_graph)(be.sched, self.gf) {
                return Err(anyhow!(
                    "failed to allocate a graph of {} nodes",
                    (api.ggml_graph_n_nodes)(self.gf)
                ));
            }
            for (t, bytes) in self.inputs.borrow().iter() {
                (api.ggml_backend_tensor_set)(*t, bytes.as_ptr() as *const _, 0, bytes.len());
            }
            let status = (api.ggml_backend_sched_graph_compute)(be.sched, self.gf);
            if status == ffi::GGML_STATUS_ABORTED {
                return Err(anyhow!("graph compute cancelled"));
            }
            if status != ffi::GGML_STATUS_SUCCESS {
                be.gpu_failed.set(!be.gpu.is_null());
                return Err(anyhow!("graph compute failed with status {status}"));
            }
        }
        be.check_gpu()
    }

    pub fn output_f32(&'o self, t: Tensor<'o>) -> Vec<f32> {
        read_f32(self.ctx.api, t)
    }
}
