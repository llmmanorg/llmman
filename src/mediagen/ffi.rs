//! `dlopen` bindings to the ggml and llama libraries of a llama.cpp
//! release; the GPU backend module is picked at runtime like llama-server
//! does. Only what the diffusion pipeline needs is bound.

#![allow(non_camel_case_types, dead_code)]

use std::ffi::{c_char, c_int, c_void, CStr};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

use anyhow::{Context, Result};
use libloading::Library;

use crate::llama_release::{PinnedGgml, PINNED_GGML};

pub type GgmlContext = c_void;
pub type GgmlCgraph = c_void;
pub type GgmlBackend = *mut c_void;
pub type GgmlBackendBuft = *mut c_void;
pub type GgmlBackendBuffer = *mut c_void;
pub type GgmlBackendSched = *mut c_void;
pub type GgmlBackendDev = *mut c_void;
pub type GgmlBackendReg = *mut c_void;
pub type GgmlGallocr = *mut c_void;
pub type LlamaModel = c_void;
pub type LlamaContextT = c_void;
pub type LlamaVocab = c_void;
pub type LlamaMemory = *mut c_void;
pub type LlamaSampler = c_void;
pub type LlamaToken = i32;

pub const GGML_MAX_DIMS: usize = 4;
pub const GGML_MAX_NAME: usize = 64;

/// `enum ggml_type` values used here (`ggml.h` at the release in
/// `LLAMA_CPP_RELEASE`); ids 4 and 5 are removed types whose block size
/// ggml reports as 0.
pub mod ty {
    pub const F32: u32 = 0;
    pub const F16: u32 = 1;
    pub const Q4_0: u32 = 2;
    pub const Q4_1: u32 = 3;
    pub const Q5_0: u32 = 6;
    pub const Q5_1: u32 = 7;
    pub const Q8_0: u32 = 8;
    pub const Q2_K: u32 = 10;
    pub const Q3_K: u32 = 11;
    pub const Q4_K: u32 = 12;
    pub const Q5_K: u32 = 13;
    pub const Q6_K: u32 = 14;
    pub const IQ4_NL: u32 = 20;
    pub const I32: u32 = 26;
    pub const BF16: u32 = 30;
    /// `GGML_TYPE_COUNT`; `ggml_blck_size` and `ggml_type_size` index a
    /// table of this length and only assert the bound in debug builds.
    pub const COUNT: u32 = 44;
}

pub const GGML_BACKEND_DEVICE_TYPE_CPU: c_int = 0;
pub const GGML_BACKEND_DEVICE_TYPE_GPU: c_int = 1;
pub const GGML_BACKEND_DEVICE_TYPE_IGPU: c_int = 2;
pub const GGML_BACKEND_BUFFER_USAGE_WEIGHTS: c_int = 1;
pub const GGML_PREC_F32: c_int = 10;
pub const GGML_SCALE_MODE_NEAREST: u32 = 0;
pub const GGML_STATUS_SUCCESS: c_int = 0;
/// A CPU compute stopped by the abort callback (`enum ggml_status`).
pub const GGML_STATUS_ABORTED: c_int = 1;
pub const LLAMA_POOLING_TYPE_NONE: c_int = 0;
pub const LLAMA_FLASH_ATTN_TYPE_AUTO: c_int = -1;
pub const LLAMA_FLASH_ATTN_TYPE_DISABLED: c_int = 0;

/// `struct ggml_tensor`, read only (shape, type, strides, name).
#[repr(C)]
pub struct GgmlTensor {
    pub type_: u32,
    pub buffer: *mut c_void,
    pub ne: [i64; GGML_MAX_DIMS],
    pub nb: [usize; GGML_MAX_DIMS],
    pub op: u32,
    pub op_params: [i32; 16],
    pub flags: i32,
    pub src: [*mut GgmlTensor; 10],
    pub view_src: *mut GgmlTensor,
    pub view_offs: usize,
    pub data: *mut c_void,
    pub name: [c_char; GGML_MAX_NAME],
    pub extra: *mut c_void,
    pub padding: [c_char; 8],
}

pub type TensorPtr = *mut GgmlTensor;

/// A tensor ggml owns: a node of a `ggml_context` or graph, or a weight
/// in its buffer. The handle borrows its owner for `'a`: the `Graph`,
/// `Resident` or `Weights` that keeps the tensor alive, so a handle
/// cannot outlive the memory it points at, and a graph's op builders only
/// ever receive tensors whose owners outlive the graph (see `Graph`).
/// The shape, strides and type are copied out when the handle is made, so
/// the accessors never touch ggml's memory; only [`Tensor::as_ptr`] hands
/// the pointer back to ggml. ggml never resizes a tensor in place (a
/// reshape or view is a new tensor), so the copy stays current.
///
/// A node of one graph cannot be used once that graph is gone:
///
/// ```compile_fail,E0597
/// use llmman::mediagen::backend::Graph;
/// use llmman::mediagen::ffi::{ty, Api, Tensor};
///
/// fn keep<'o>(g: &'o Graph<'o>) -> Tensor<'o> {
///     g.new_tensor(ty::F32, &[1])
/// }
///
/// fn later<'o>(api: &'static Api, other: &'o Graph<'o>) {
///     let t;
///     {
///         let g = Graph::new(api, 8).unwrap();
///         t = keep(&g);
///     }
///     other.mark_output(t);
/// }
/// ```
///
/// A node can be marked as its own graph's output. A node may also enter
/// another graph, but only while its own graph outlives every use of
/// that other graph; the borrow checker holds the first graph alive for
/// as long as the second one uses the node:
///
/// ```
/// use llmman::mediagen::backend::Graph;
/// use llmman::mediagen::ffi::{ty, Api, Tensor};
///
/// fn keep<'o>(g: &'o Graph<'o>) -> Tensor<'o> {
///     g.new_tensor(ty::F32, &[1])
/// }
///
/// fn later(api: &'static Api) {
///     let g = Graph::new(api, 8).unwrap();
///     let t = keep(&g);
///     g.mark_output(t);
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tensor<'a> {
    ptr: NonNull<GgmlTensor>,
    ne: [i64; GGML_MAX_DIMS],
    nb: [usize; GGML_MAX_DIMS],
    ty: u32,
    owner: PhantomData<&'a ()>,
}

impl<'a> Tensor<'a> {
    /// Wraps a pointer ggml returned, copying its shape, strides and
    /// type, as a handle that borrows `owner`. Panics on null.
    ///
    /// # Safety
    /// `p` must point at a live ggml tensor that `owner` keeps alive for
    /// as long as `owner` is borrowed: one of its `ggml_context`, graph
    /// or weight buffer.
    pub unsafe fn from_raw<O: ?Sized>(p: TensorPtr, owner: &'a O) -> Tensor<'a> {
        let _ = owner;
        let ptr = NonNull::new(p).expect("null ggml tensor");
        // SAFETY: the caller guarantees `p` points at a live tensor.
        let t = unsafe { ptr.as_ref() };
        Tensor {
            ptr,
            ne: t.ne,
            nb: t.nb,
            ty: t.type_,
            owner: PhantomData,
        }
    }

    pub fn as_ptr(self) -> TensorPtr {
        self.ptr.as_ptr()
    }

    /// Shape in ggml order.
    pub fn shape(self) -> [i64; 4] {
        self.ne
    }

    /// Strides in bytes, ggml order.
    pub fn strides(self) -> [usize; 4] {
        self.nb
    }

    /// Storage type, one of [`ty`].
    pub fn dtype(self) -> u32 {
        self.ty
    }

    /// A placeholder for tests: an empty shape and a pointer at no
    /// tensor, which only `as_ptr` could misuse.
    #[cfg(test)]
    pub(crate) fn dangling() -> Tensor<'static> {
        Tensor {
            ptr: NonNull::dangling(),
            ne: [0; GGML_MAX_DIMS],
            nb: [0; GGML_MAX_DIMS],
            ty: 0,
            owner: PhantomData,
        }
    }
}

impl From<Tensor<'_>> for TensorPtr {
    fn from(t: Tensor<'_>) -> TensorPtr {
        t.as_ptr()
    }
}

impl GgmlTensor {
    pub fn name(&self) -> String {
        unsafe { CStr::from_ptr(self.name.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }
}

#[repr(C)]
pub struct GgmlInitParams {
    pub mem_size: usize,
    pub mem_buffer: *mut c_void,
    pub no_alloc: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlamaBatch {
    pub n_tokens: i32,
    pub token: *mut LlamaToken,
    pub embd: *mut f32,
    pub pos: *mut i32,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut i32,
    pub logits: *mut i8,
}

#[repr(C)]
pub struct LlamaChatMessage {
    pub role: *const c_char,
    pub content: *const c_char,
}

#[repr(C)]
pub struct LlamaSamplerChainParams {
    pub no_perf: bool,
}

/// By-value parameter structs travel as this oversized blob (aggregates
/// this big go through memory on every ABI we target); known fields are
/// patched through the prefix mirrors below.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct ParamBlob(pub [u8; 1024]);

/// Prefix of `struct llama_model_params`.
#[repr(C)]
pub struct LlamaModelParams {
    pub devices: *mut c_void,
    pub tensor_buft_overrides: *const c_void,
    pub n_gpu_layers: i32,
    pub split_mode: c_int,
    pub load_mode: c_int,
    pub lazy_mode: c_int,
    pub main_gpu: i32,
    pub tensor_split: *const f32,
    pub progress_callback: *const c_void,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const c_void,
    pub vocab_only: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
    pub load_mtp: bool,
}

pub type EvalCallback =
    unsafe extern "C" fn(t: TensorPtr, ask: bool, user_data: *mut c_void) -> bool;

/// `ggml_abort_callback`: polled by the CPU backend after each node; a
/// `true` return stops the graph with [`GGML_STATUS_ABORTED`].
pub type AbortCallback = unsafe extern "C" fn(data: *mut c_void) -> bool;

/// Prefix of `struct llama_context_params`.
#[repr(C)]
pub struct LlamaContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub n_outputs_max: u32,
    pub n_outputs_max_per_seq: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub ctx_type: c_int,
    pub rope_scaling_type: c_int,
    pub pooling_type: c_int,
    pub attention_type: c_int,
    pub flash_attn_type: c_int,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,
    pub cb_eval: Option<EvalCallback>,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: u32,
    pub type_v: u32,
    /// Added mid-struct at b11669.
    pub moe_cache_size: usize,
    pub abort_callback: *const c_void,
    pub abort_callback_data: *mut c_void,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
}

impl ParamBlob {
    pub fn view_mut<T>(&mut self) -> &mut T {
        assert!(std::mem::size_of::<T>() <= 1024);
        unsafe { &mut *(self.0.as_mut_ptr() as *mut T) }
    }
}

pub type LogCallback =
    unsafe extern "C" fn(level: c_int, text: *const c_char, user_data: *mut c_void);

macro_rules! api {
    ($(fn $name:ident($($arg:ident: $ty:ty),*) $(-> $ret:ty)?;)*
     optional: $(fn $oname:ident($($oarg:ident: $oty:ty),*) $(-> $oret:ty)?;)*) => {
        /// The bound function pointers; `optional` ones may be absent from older releases.
        pub struct Api {
            _libs: Vec<Library>,
            $(pub $name: unsafe extern "C" fn($($arg: $ty),*) $(-> $ret)?,)*
            $(pub $oname: Option<unsafe extern "C" fn($($oarg: $oty),*) $(-> $oret)?>,)*
        }
        impl Api {
            fn bind(libs: Vec<Library>) -> Result<Self> {
                $(let $name = lookup(&libs, stringify!($name))?;)*
                $(let $oname = lookup(&libs, stringify!($oname)).ok();)*
                Ok(Self { _libs: libs, $($name,)* $($oname,)* })
            }
        }
    };
}

api! {
    // ggml-base
    fn ggml_init(params: GgmlInitParams) -> *mut GgmlContext;
    fn ggml_free(ctx: *mut GgmlContext);
    fn ggml_tensor_overhead() -> usize;
    fn ggml_graph_overhead_custom(size: usize, grads: bool) -> usize;
    fn ggml_new_tensor(ctx: *mut GgmlContext, type_: u32, n_dims: c_int, ne: *const i64) -> TensorPtr;
    fn ggml_set_name(t: TensorPtr, name: *const c_char) -> TensorPtr;
    fn ggml_set_input(t: TensorPtr);
    fn ggml_set_output(t: TensorPtr);
    fn ggml_nbytes(t: *const GgmlTensor) -> usize;
    fn ggml_nelements(t: *const GgmlTensor) -> i64;
    fn ggml_type_size(type_: u32) -> usize;
    fn ggml_blck_size(type_: u32) -> i64;
    fn ggml_row_size(type_: u32, ne: i64) -> usize;
    fn ggml_is_contiguous(t: *const GgmlTensor) -> bool;
    fn ggml_new_graph_custom(ctx: *mut GgmlContext, size: usize, grads: bool) -> *mut GgmlCgraph;
    fn ggml_build_forward_expand(gf: *mut GgmlCgraph, t: TensorPtr);
    fn ggml_graph_n_nodes(gf: *mut GgmlCgraph) -> c_int;
    fn ggml_fp32_to_fp16_row(x: *const f32, y: *mut u16, n: i64);
    fn ggml_fp16_to_fp32_row(x: *const u16, y: *mut f32, n: i64);
    fn ggml_bf16_to_fp32_row(x: *const u16, y: *mut f32, n: i64);
    // ops
    fn ggml_add(ctx: *mut GgmlContext, a: TensorPtr, b: TensorPtr) -> TensorPtr;
    fn ggml_sub(ctx: *mut GgmlContext, a: TensorPtr, b: TensorPtr) -> TensorPtr;
    fn ggml_mul(ctx: *mut GgmlContext, a: TensorPtr, b: TensorPtr) -> TensorPtr;
    fn ggml_scale(ctx: *mut GgmlContext, a: TensorPtr, s: f32) -> TensorPtr;
    fn ggml_scale_bias(ctx: *mut GgmlContext, a: TensorPtr, s: f32, b: f32) -> TensorPtr;
    fn ggml_mul_mat(ctx: *mut GgmlContext, a: TensorPtr, b: TensorPtr) -> TensorPtr;
    fn ggml_rms_norm(ctx: *mut GgmlContext, a: TensorPtr, eps: f32) -> TensorPtr;
    fn ggml_norm(ctx: *mut GgmlContext, a: TensorPtr, eps: f32) -> TensorPtr;
    fn ggml_silu(ctx: *mut GgmlContext, a: TensorPtr) -> TensorPtr;
    fn ggml_relu(ctx: *mut GgmlContext, a: TensorPtr) -> TensorPtr;
    fn ggml_gelu(ctx: *mut GgmlContext, a: TensorPtr) -> TensorPtr;
    fn ggml_sigmoid(ctx: *mut GgmlContext, a: TensorPtr) -> TensorPtr;
    fn ggml_tanh(ctx: *mut GgmlContext, a: TensorPtr) -> TensorPtr;
    fn ggml_sin(ctx: *mut GgmlContext, a: TensorPtr) -> TensorPtr;
    fn ggml_sqr(ctx: *mut GgmlContext, a: TensorPtr) -> TensorPtr;
    fn ggml_sqrt(ctx: *mut GgmlContext, a: TensorPtr) -> TensorPtr;
    fn ggml_log(ctx: *mut GgmlContext, a: TensorPtr) -> TensorPtr;
    fn ggml_clamp(ctx: *mut GgmlContext, a: TensorPtr, min: f32, max: f32) -> TensorPtr;
    fn ggml_cpy(ctx: *mut GgmlContext, a: TensorPtr, b: TensorPtr) -> TensorPtr;
    fn ggml_reshape_2d(ctx: *mut GgmlContext, a: TensorPtr, ne0: i64, ne1: i64) -> TensorPtr;
    fn ggml_reshape_3d(ctx: *mut GgmlContext, a: TensorPtr, ne0: i64, ne1: i64, ne2: i64) -> TensorPtr;
    fn ggml_reshape_4d(ctx: *mut GgmlContext, a: TensorPtr, ne0: i64, ne1: i64, ne2: i64, ne3: i64) -> TensorPtr;
    fn ggml_view_2d(ctx: *mut GgmlContext, a: TensorPtr, ne0: i64, ne1: i64, nb1: usize, offset: usize) -> TensorPtr;
    fn ggml_view_3d(ctx: *mut GgmlContext, a: TensorPtr, ne0: i64, ne1: i64, ne2: i64, nb1: usize, nb2: usize, offset: usize) -> TensorPtr;
    fn ggml_view_4d(ctx: *mut GgmlContext, a: TensorPtr, ne0: i64, ne1: i64, ne2: i64, ne3: i64, nb1: usize, nb2: usize, nb3: usize, offset: usize) -> TensorPtr;
    fn ggml_permute(ctx: *mut GgmlContext, a: TensorPtr, a0: c_int, a1: c_int, a2: c_int, a3: c_int) -> TensorPtr;
    fn ggml_transpose(ctx: *mut GgmlContext, a: TensorPtr) -> TensorPtr;
    fn ggml_cont(ctx: *mut GgmlContext, a: TensorPtr) -> TensorPtr;
    fn ggml_cont_2d(ctx: *mut GgmlContext, a: TensorPtr, ne0: i64, ne1: i64) -> TensorPtr;
    fn ggml_concat(ctx: *mut GgmlContext, a: TensorPtr, b: TensorPtr, dim: c_int) -> TensorPtr;
    fn ggml_cast(ctx: *mut GgmlContext, a: TensorPtr, type_: u32) -> TensorPtr;
    fn ggml_get_rows(ctx: *mut GgmlContext, a: TensorPtr, b: TensorPtr) -> TensorPtr;
    fn ggml_repeat(ctx: *mut GgmlContext, a: TensorPtr, b: TensorPtr) -> TensorPtr;
    fn ggml_pad(ctx: *mut GgmlContext, a: TensorPtr, p0: c_int, p1: c_int, p2: c_int, p3: c_int) -> TensorPtr;
    fn ggml_pad_ext(ctx: *mut GgmlContext, a: TensorPtr, lp0: c_int, rp0: c_int, lp1: c_int, rp1: c_int, lp2: c_int, rp2: c_int, lp3: c_int, rp3: c_int) -> TensorPtr;
    fn ggml_conv_1d(ctx: *mut GgmlContext, a: TensorPtr, b: TensorPtr, s0: c_int, p0: c_int, d0: c_int) -> TensorPtr;
    fn ggml_conv_2d(ctx: *mut GgmlContext, a: TensorPtr, b: TensorPtr, s0: c_int, s1: c_int, p0: c_int, p1: c_int, d0: c_int, d1: c_int) -> TensorPtr;
    fn ggml_conv_transpose_1d(ctx: *mut GgmlContext, a: TensorPtr, b: TensorPtr, s0: c_int, p0: c_int, d0: c_int) -> TensorPtr;
    fn ggml_conv_2d_dw_direct(ctx: *mut GgmlContext, a: TensorPtr, b: TensorPtr, stride0: c_int, stride1: c_int, pad0: c_int, pad1: c_int, dilation0: c_int, dilation1: c_int) -> TensorPtr;
    fn ggml_ssm_conv(ctx: *mut GgmlContext, sx: TensorPtr, c: TensorPtr) -> TensorPtr;
    fn ggml_interpolate(ctx: *mut GgmlContext, a: TensorPtr, ne0: i64, ne1: i64, ne2: i64, ne3: i64, mode: u32) -> TensorPtr;
    fn ggml_timestep_embedding(ctx: *mut GgmlContext, timesteps: TensorPtr, dim: c_int, max_period: c_int) -> TensorPtr;
    fn ggml_flash_attn_ext(ctx: *mut GgmlContext, q: TensorPtr, k: TensorPtr, v: TensorPtr, mask: TensorPtr, scale: f32, max_bias: f32, logit_softcap: f32) -> TensorPtr;
    fn ggml_flash_attn_ext_set_prec(a: TensorPtr, prec: c_int);
    fn ggml_soft_max_ext(ctx: *mut GgmlContext, a: TensorPtr, mask: TensorPtr, scale: f32, max_bias: f32) -> TensorPtr;
    // backend
    fn ggml_backend_load_all_from_path(dir: *const c_char);
    fn ggml_backend_init_by_type(type_: c_int, params: *const c_char) -> GgmlBackend;
    fn ggml_backend_free(backend: GgmlBackend);
    fn ggml_backend_name(backend: GgmlBackend) -> *const c_char;
    fn ggml_backend_get_default_buffer_type(backend: GgmlBackend) -> GgmlBackendBuft;
    fn ggml_backend_graph_compute(backend: GgmlBackend, gf: *mut GgmlCgraph) -> c_int;
    fn ggml_backend_get_device(backend: GgmlBackend) -> GgmlBackendDev;
    fn ggml_backend_dev_memory(dev: GgmlBackendDev, free: *mut usize, total: *mut usize);
    fn ggml_gallocr_new(buft: GgmlBackendBuft) -> GgmlGallocr;
    fn ggml_gallocr_free(galloc: GgmlGallocr);
    fn ggml_gallocr_reserve(galloc: GgmlGallocr, gf: *mut GgmlCgraph) -> bool;
    fn ggml_gallocr_get_buffer_size(galloc: GgmlGallocr, buffer_id: c_int) -> usize;
    fn ggml_backend_dev_backend_reg(dev: GgmlBackendDev) -> GgmlBackendReg;
    fn ggml_backend_reg_get_proc_address(reg: GgmlBackendReg, name: *const c_char) -> *mut c_void;
    fn ggml_backend_alloc_ctx_tensors_from_buft(ctx: *mut GgmlContext, buft: GgmlBackendBuft) -> GgmlBackendBuffer;
    fn ggml_backend_buffer_set_usage(buf: GgmlBackendBuffer, usage: c_int);
    fn ggml_backend_buffer_free(buf: GgmlBackendBuffer);
    fn ggml_backend_buffer_clear(buf: GgmlBackendBuffer, value: u8);
    fn ggml_backend_buffer_get_size(buf: GgmlBackendBuffer) -> usize;
    fn ggml_backend_tensor_set(t: TensorPtr, data: *const c_void, offset: usize, size: usize);
    fn ggml_backend_tensor_get(t: *const GgmlTensor, data: *mut c_void, offset: usize, size: usize);
    fn ggml_backend_sched_new(backends: *mut GgmlBackend, bufts: *mut GgmlBackendBuft, n: c_int, graph_size: usize, parallel: bool, op_offload: bool) -> GgmlBackendSched;
    fn ggml_backend_sched_free(sched: GgmlBackendSched);
    fn ggml_backend_sched_reset(sched: GgmlBackendSched);
    fn ggml_backend_sched_alloc_graph(sched: GgmlBackendSched, gf: *mut GgmlCgraph) -> bool;
    fn ggml_backend_sched_graph_compute(sched: GgmlBackendSched, gf: *mut GgmlCgraph) -> c_int;
    // llama
    fn llama_backend_init();
    fn llama_log_set(cb: Option<LogCallback>, user_data: *mut c_void);
    fn llama_model_default_params() -> ParamBlob;
    fn llama_model_load_from_file(path: *const c_char, params: ParamBlob) -> *mut LlamaModel;
    fn llama_model_free(model: *mut LlamaModel);
    fn llama_model_get_vocab(model: *const LlamaModel) -> *const LlamaVocab;
    fn llama_model_n_layer(model: *const LlamaModel) -> i32;
    fn llama_model_n_embd(model: *const LlamaModel) -> i32;
    fn llama_model_chat_template(model: *const LlamaModel, name: *const c_char) -> *const c_char;
    fn llama_context_default_params() -> ParamBlob;
    fn llama_init_from_model(model: *mut LlamaModel, params: ParamBlob) -> *mut LlamaContextT;
    fn llama_free(ctx: *mut LlamaContextT);
    fn llama_get_memory(ctx: *const LlamaContextT) -> LlamaMemory;
    fn llama_memory_clear(mem: LlamaMemory, data: bool);
    fn llama_set_embeddings(ctx: *mut LlamaContextT, embeddings: bool);
    fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> LlamaBatch;
    fn llama_batch_free(batch: LlamaBatch);
    fn llama_decode(ctx: *mut LlamaContextT, batch: LlamaBatch) -> i32;
    fn llama_get_embeddings_ith(ctx: *mut LlamaContextT, i: i32) -> *mut f32;
    fn llama_tokenize(vocab: *const LlamaVocab, text: *const c_char, text_len: i32, tokens: *mut LlamaToken, n_max: i32, add_special: bool, parse_special: bool) -> i32;
    fn llama_token_to_piece(vocab: *const LlamaVocab, token: LlamaToken, buf: *mut c_char, len: i32, lstrip: i32, special: bool) -> i32;
    fn llama_vocab_is_eog(vocab: *const LlamaVocab, token: LlamaToken) -> bool;
    fn llama_chat_apply_template(tmpl: *const c_char, chat: *const LlamaChatMessage, n: usize, add_ass: bool, buf: *mut c_char, len: i32) -> i32;
    fn llama_sampler_chain_default_params() -> LlamaSamplerChainParams;
    fn llama_sampler_chain_init(params: LlamaSamplerChainParams) -> *mut LlamaSampler;
    fn llama_sampler_chain_add(chain: *mut LlamaSampler, smpl: *mut LlamaSampler);
    fn llama_sampler_init_temp(t: f32) -> *mut LlamaSampler;
    fn llama_sampler_init_dist(seed: u32) -> *mut LlamaSampler;
    fn llama_sampler_sample(smpl: *mut LlamaSampler, ctx: *mut LlamaContextT, idx: i32) -> LlamaToken;
    fn llama_sampler_free(smpl: *mut LlamaSampler);

    optional:
    // direct 2-D convolution (no im2col), ggml >= mid-2025
    fn ggml_conv_2d_direct(ctx: *mut GgmlContext, a: TensorPtr, b: TensorPtr, s0: c_int, s1: c_int, p0: c_int, p1: c_int, d0: c_int, d1: c_int) -> TensorPtr;
    fn ggml_backend_supports_op(backend: GgmlBackend, op: TensorPtr) -> bool;
    // library identity, ggml >= 2025-07-02
    fn ggml_version() -> *const c_char;
    fn ggml_commit() -> *const c_char;
    // F32 accumulation hint for mul_mat / flash attention, only the speech encoder needs it
    fn ggml_prec_set_acc(a: TensorPtr, prec: c_int) -> bool;
    // row writes into a KV cache, only the speech decoder needs it
    fn ggml_set_rows(ctx: *mut GgmlContext, a: TensorPtr, b: TensorPtr, c: TensorPtr) -> TensorPtr;
}

fn lookup<T: Copy>(libs: &[Library], name: &str) -> Result<T> {
    let cname = format!("{name}\0");
    for lib in libs {
        if let Ok(sym) = unsafe { lib.get::<T>(cname.as_bytes()) } {
            return Ok(*sym);
        }
    }
    anyhow::bail!("symbol {name} not found in the ggml/llama libraries")
}

fn lib_name(stem: &str) -> String {
    if cfg!(target_os = "macos") {
        format!("lib{stem}.dylib")
    } else if cfg!(target_os = "windows") {
        format!("{stem}.dll")
    } else {
        format!("lib{stem}.so")
    }
}

/// Opens the library with the symbols exported globally, so the modules
/// dlopened next (`libggml`, `libllama`, the backends) resolve their
/// `ggml-base` imports even when their runpath does not point here.
fn open(path: &Path) -> Result<Library> {
    #[cfg(unix)]
    {
        use libloading::os::unix::Library as UnixLibrary;
        let lib = unsafe { UnixLibrary::open(Some(path), libc::RTLD_NOW | libc::RTLD_GLOBAL) }
            .with_context(|| format!("dlopen {}", path.display()))?;
        Ok(Library::from(lib))
    }
    #[cfg(not(unix))]
    {
        unsafe { Library::new(path) }.with_context(|| format!("load {}", path.display()))
    }
}

/// True if `dir` holds the ggml and llama libraries.
pub fn has_libs(dir: &Path) -> bool {
    ["ggml-base", "ggml", "llama"]
        .iter()
        .all(|stem| dir.join(lib_name(stem)).exists())
}

/// The ggml/llama libraries of the `llama-server` at `bin`: next to it in
/// a release archive or image, in `../lib` for an installed build.
pub fn lib_dir_of(bin: &Path) -> Option<PathBuf> {
    let dir = bin.parent()?;
    [dir.to_path_buf(), dir.join("../lib"), dir.join("../lib64")]
        .into_iter()
        .find(|d| has_libs(d))
}

/// The mirrored [`LlamaModelParams`] prefix holds for this libllama if the
/// defaults it fills in land at the mirrored offsets.
pub fn check_model_params(p: &LlamaModelParams) -> Result<()> {
    if p.split_mode != 1 || p.main_gpu != 0 || p.vocab_only || p.check_tensors || !p.use_extra_bufts
    {
        anyhow::bail!("llama_model_params layout does not match this libllama");
    }
    Ok(())
}

/// [`check_model_params`] for [`LlamaContextParams`].
pub fn check_context_params(p: &LlamaContextParams) -> Result<()> {
    if p.n_batch != 2048
        || p.n_ubatch != 512
        || p.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_AUTO
        || p.type_k != ty::F16
        || !p.offload_kqv
        || !p.op_offload
        || !p.swa_full
        || p.kv_unified
    {
        anyhow::bail!("llama_context_params layout does not match this libllama");
    }
    Ok(())
}

/// What [`Api::load_with`] does with the `ggml_version()` and
/// `ggml_commit()` the libraries report.
pub enum IdentityCheck {
    /// Fail unless they match this pin; for the release llmman was tested
    /// against.
    Strict(&'static PinnedGgml),
    /// Print them and go on; for a user-chosen release or `latest`.
    Log,
}

/// The identity the libraries in `dir` report against `expected`: `None`
/// is an absent symbol, the version must match exactly, the commit after
/// a `-dirty` suffix is dropped must start with the pinned prefix.
pub fn check_identity(
    dir: &Path,
    expected: &PinnedGgml,
    version: Option<&str>,
    commit: Option<&str>,
) -> Result<()> {
    let (Some(version), Some(commit)) = (version, commit) else {
        let missing = if version.is_none() {
            "ggml_version"
        } else {
            "ggml_commit"
        };
        anyhow::bail!(
            "unsupported llama.cpp build (no {missing}) in {}",
            dir.display()
        );
    };
    let clean = commit.strip_suffix("-dirty").unwrap_or(commit);
    if version != expected.version || clean.get(..expected.commit.len()) != Some(expected.commit) {
        anyhow::bail!(
            "the llama.cpp libraries in {} are ggml {version} {commit}, expected {} {} \
             (release {}); an old build? remove it from PATH or pass --llama-cpp-version \
             to use a release",
            dir.display(),
            expected.version,
            expected.commit,
            expected.tag
        );
    }
    Ok(())
}

/// The string an optional `const char *` getter returns; `None` when the
/// symbol is absent or it returns null.
fn c_string(f: Option<unsafe extern "C" fn() -> *const c_char>) -> Option<String> {
    let p = unsafe { f?() };
    if p.is_null() {
        return None;
    }
    Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
}

impl Api {
    /// [`Api::load_with`] held to the release in `LLAMA_CPP_RELEASE`.
    pub fn load(dir: &Path) -> Result<Api> {
        Api::load_with(dir, IdentityCheck::Strict(&PINNED_GGML))
    }

    /// Loads `libggml-base`, `libggml` and `libllama` from `dir`, checks
    /// their identity per `check`, then loads the backend modules found
    /// there (`libggml-cuda`, `libggml-metal`, …).
    pub fn load_with(dir: &Path, check: IdentityCheck) -> Result<Api> {
        let mut libs = Vec::new();
        for stem in ["ggml-base", "ggml", "llama"] {
            libs.push(open(&dir.join(lib_name(stem)))?);
        }
        let api = Api::bind_checked(dir, libs, check)?;
        let cdir = std::ffi::CString::new(dir.to_string_lossy().as_bytes())
            .context("library directory path")?;
        unsafe {
            (api.ggml_backend_load_all_from_path)(cdir.as_ptr());
            (api.llama_backend_init)();
        }
        Ok(api)
    }

    /// Checks the identity of `libs` per `check`, then binds the API. The
    /// identity goes first so that a mismatched build which also lacks a
    /// symbol is reported as the wrong build, not as the missing symbol.
    fn bind_checked(dir: &Path, libs: Vec<Library>, check: IdentityCheck) -> Result<Api> {
        let version = c_string(lookup(&libs, "ggml_version").ok());
        let commit = c_string(lookup(&libs, "ggml_commit").ok());
        match check {
            IdentityCheck::Strict(expected) => {
                check_identity(dir, expected, version.as_deref(), commit.as_deref())?
            }
            IdentityCheck::Log => eprintln!(
                "[llmman] mediagen: llama.cpp libraries in {}: ggml {} {}",
                dir.display(),
                version.as_deref().unwrap_or("(no ggml_version)"),
                commit.as_deref().unwrap_or("(no ggml_commit)")
            ),
        }
        Api::bind(libs)
    }

    /// Everything the pipeline assumes about this llama.cpp that binding
    /// the symbols did not already prove: the two mirrored param structs.
    /// No model needed, so CI can run it against the pinned release.
    pub fn check_layout(&self) -> Result<()> {
        unsafe {
            let mut mp = (self.llama_model_default_params)();
            check_model_params(mp.view_mut::<LlamaModelParams>())?;
            let mut cp = (self.llama_context_default_params)();
            check_context_params(cp.view_mut::<LlamaContextParams>())?;
        }
        Ok(())
    }

    pub fn set_n_threads(&self, backend: GgmlBackend, n: i32) {
        unsafe {
            let reg = (self.ggml_backend_dev_backend_reg)((self.ggml_backend_get_device)(backend));
            let p = (self.ggml_backend_reg_get_proc_address)(
                reg,
                c"ggml_backend_set_n_threads".as_ptr(),
            );
            if !p.is_null() {
                let f: unsafe extern "C" fn(GgmlBackend, c_int) = std::mem::transmute(p);
                f(backend, n);
            }
        }
    }

    /// Registers `cb` with `data` on the CPU `backend` through the
    /// registry's `ggml_backend_set_abort_callback` entry (the CPU module
    /// maps it to `ggml_backend_cpu_set_abort_callback`); a backend without
    /// that entry is left as it is. `data` must outlive the backend.
    pub fn set_abort_callback(
        &self,
        backend: GgmlBackend,
        cb: Option<AbortCallback>,
        data: *mut c_void,
    ) {
        unsafe {
            let reg = (self.ggml_backend_dev_backend_reg)((self.ggml_backend_get_device)(backend));
            let p = (self.ggml_backend_reg_get_proc_address)(
                reg,
                c"ggml_backend_set_abort_callback".as_ptr(),
            );
            if !p.is_null() {
                let f: unsafe extern "C" fn(GgmlBackend, Option<AbortCallback>, *mut c_void) =
                    std::mem::transmute(p);
                f(backend, cb, data);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    /// The accessors read the copy taken at construction, not the
    /// tensor: the box that owns it is the handle's owner here and lives
    /// through the reads, as `from_raw` requires.
    #[test]
    fn a_handle_answers_from_its_copy_of_the_tensor() {
        use super::{ty, GgmlTensor, Tensor};
        // SAFETY: an all-zero GgmlTensor is a valid value (integers and
        // null pointers).
        let mut raw: Box<GgmlTensor> = unsafe { Box::new(std::mem::zeroed()) };
        raw.type_ = ty::F16;
        raw.ne = [3, 5, 1, 1];
        raw.nb = [2, 6, 30, 30];
        let ptr: *mut GgmlTensor = &mut *raw;
        // SAFETY: `ptr` points at the live tensor `raw` owns for as long
        // as `raw` is borrowed.
        let t = unsafe { Tensor::from_raw(ptr, &raw) };
        assert_eq!(t.shape(), [3, 5, 1, 1]);
        assert_eq!(t.strides(), [2, 6, 30, 30]);
        assert_eq!(t.dtype(), ty::F16);
        drop(raw);
    }

    use super::*;

    const PIN: PinnedGgml = PinnedGgml {
        tag: "b11669",
        version: "0.27.0",
        commit: "620d2225c",
    };

    fn check(version: Option<&str>, commit: Option<&str>) -> Result<()> {
        check_identity(Path::new("/opt/llama"), &PIN, version, commit)
    }

    #[test]
    fn check_identity_accepts_the_pinned_version_and_commit() {
        check(Some("0.27.0"), Some("620d2225c")).unwrap();
    }

    #[test]
    fn check_identity_ignores_a_dirty_suffix_on_the_commit() {
        check(Some("0.27.0"), Some("620d2225c-dirty")).unwrap();
    }

    #[test]
    fn check_identity_rejects_an_absent_version() {
        let err = check(None, Some("620d2225c")).unwrap_err().to_string();
        assert_eq!(
            err,
            "unsupported llama.cpp build (no ggml_version) in /opt/llama"
        );
    }

    #[test]
    fn check_identity_rejects_an_absent_commit() {
        let err = check(Some("0.27.0"), None).unwrap_err().to_string();
        assert_eq!(
            err,
            "unsupported llama.cpp build (no ggml_commit) in /opt/llama"
        );
    }

    #[test]
    fn check_identity_rejects_another_version() {
        let err = check(Some("0.26.0"), Some("620d2225c"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("/opt/llama"), "{err}");
        assert!(err.contains("ggml 0.26.0 620d2225c"), "{err}");
        assert!(err.contains("expected 0.27.0 620d2225c"), "{err}");
        assert!(
            err.ends_with("pass --llama-cpp-version to use a release"),
            "{err}"
        );
    }

    #[test]
    fn check_identity_rejects_another_commit() {
        let err = check(Some("0.27.0"), Some("abcdef012-dirty"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("/opt/llama"), "{err}");
        assert!(err.contains("abcdef012-dirty"), "{err}");
        assert!(err.contains("620d2225c"), "{err}");
    }

    #[test]
    fn check_identity_rejects_a_commit_shorter_than_the_pin() {
        let err = check(Some("0.27.0"), Some("620d222"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("ggml 0.27.0 620d222"), "{err}");
    }

    /// The test binary itself stands in for a build without ggml: it has
    /// neither `ggml_version` nor `ggml_init`, and the strict check must
    /// name the former before `bind` can fail on the latter.
    #[cfg(unix)]
    #[test]
    fn bind_checked_reports_the_identity_before_a_missing_symbol() {
        let this: Library = libloading::os::unix::Library::this().into();
        let err = match Api::bind_checked(
            Path::new("/opt/llama"),
            vec![this],
            IdentityCheck::Strict(&PIN),
        ) {
            Ok(_) => panic!("the test binary bound as a llama.cpp build"),
            Err(e) => e.to_string(),
        };
        assert_eq!(
            err,
            "unsupported llama.cpp build (no ggml_version) in /opt/llama"
        );
    }
}
