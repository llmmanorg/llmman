//! libllama behind [`Reader`]: one model and context, with the KV cache
//! kept between questions so a shared state is decoded once.

use std::ffi::CStr;
use std::path::Path;
use std::ptr;

use anyhow::{anyhow, Context as _, Result};
use serde_json::{Map, Value};

use super::template::Template;
use super::{Error, Reader};
use crate::mediagen::ffi::{self, Api, LlamaContextParams, LlamaContextT, LlamaModel, LlamaVocab};

/// Tokens per decode call: libllama's default `n_batch`.
const BATCH: usize = 2048;
/// The context starts here and doubles as prompts need it.
const MIN_CTX: u32 = 2048;

pub struct Options {
    pub n_gpu_layers: i32,
    pub n_threads: i32,
    pub flash_attn: bool,
    /// The most context to allocate; `None` for the model's own.
    pub max_ctx: Option<u32>,
}

pub struct Engine {
    api: &'static Api,
    model: *mut LlamaModel,
    ctx: *mut LlamaContextT,
    vocab: *const LlamaVocab,
    n_vocab: usize,
    /// Size of `ctx`; 0 before the first read.
    n_ctx: u32,
    max_ctx: u32,
    opts: Options,
    template: Template,
    bos: i32,
    add_bos: bool,
    /// The tokens in the KV cache, at positions `0..`.
    cached: Vec<i32>,
}

unsafe impl Send for Engine {}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                (self.api.llama_free)(self.ctx);
            }
            (self.api.llama_model_free)(self.model);
        }
    }
}

/// The text of `token`; `None` for no token (`-1`, which libllama aborts on) or none.
fn token_text(api: &Api, vocab: *const LlamaVocab, token: i32) -> Option<String> {
    if token < 0 {
        return None;
    }
    let text = unsafe { (api.llama_vocab_get_text)(vocab, token) };
    (!text.is_null()).then(|| {
        unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned()
    })
}

impl Engine {
    pub fn load(api: &'static Api, path: &Path, opts: Options) -> Result<Engine> {
        let model = api.load_model(&path.to_string_lossy(), opts.n_gpu_layers)?;
        let vocab = unsafe { (api.llama_model_get_vocab)(model) };
        let (bos, eos) = unsafe { ((api.llama_vocab_bos)(vocab), (api.llama_vocab_eos)(vocab)) };
        let text = |t| token_text(api, vocab, t).unwrap_or_default();
        let source = unsafe { (api.llama_model_chat_template)(model, ptr::null()) };
        let template = if source.is_null() {
            Err(anyhow!("the model has no chat template"))
        } else {
            let source = unsafe { CStr::from_ptr(source) }.to_string_lossy();
            Template::new(&source, &text(bos), &text(eos))
        };
        let template = template.inspect_err(|_| unsafe { (api.llama_model_free)(model) })?;
        let trained = unsafe { (api.llama_model_n_ctx_train)(model) }.max(0) as u32;
        let max_ctx = match (opts.max_ctx.filter(|n| *n > 0), trained) {
            (Some(n), 0) => n,
            (Some(n), t) => n.min(t),
            (None, 0) => MIN_CTX,
            (None, t) => t,
        };
        Ok(Engine {
            api,
            model,
            ctx: ptr::null_mut(),
            vocab,
            n_vocab: unsafe { (api.llama_vocab_n_tokens)(vocab) }.max(0) as usize,
            n_ctx: 0,
            max_ctx,
            opts,
            template,
            bos,
            add_bos: unsafe { (api.llama_vocab_get_add_bos)(vocab) },
            cached: Vec::new(),
        })
    }

    /// A context of at least `need` tokens; growing it drops the KV cache.
    fn ensure_context(&mut self, need: usize) -> Result<(), Error> {
        if !self.ctx.is_null() && need <= self.n_ctx as usize {
            return Ok(());
        }
        let n_ctx = (need.next_power_of_two() as u32)
            .max(MIN_CTX)
            .min(self.max_ctx);
        self.free_context();
        let api = self.api;
        let ctx = unsafe {
            let mut cp = (api.llama_context_default_params)();
            {
                let p = cp.view_mut::<LlamaContextParams>();
                ffi::check_context_params(p).map_err(|e| Error::Failed(e.to_string()))?;
                p.n_ctx = n_ctx;
                p.n_seq_max = 1;
                p.n_threads = self.opts.n_threads;
                p.n_threads_batch = self.opts.n_threads;
                p.no_perf = true;
                p.flash_attn_type = if self.opts.flash_attn {
                    ffi::LLAMA_FLASH_ATTN_TYPE_AUTO
                } else {
                    ffi::LLAMA_FLASH_ATTN_TYPE_DISABLED
                };
            }
            (api.llama_init_from_model)(self.model, cp)
        };
        if ctx.is_null() {
            return Err(Error::Failed(format!(
                "failed to create a context of {n_ctx} tokens (out of memory? \
                 LLMMAN_CONTEXT_LENGTH caps it)"
            )));
        }
        self.ctx = ctx;
        self.n_ctx = n_ctx;
        Ok(())
    }

    fn free_context(&mut self) {
        if !self.ctx.is_null() {
            unsafe { (self.api.llama_free)(self.ctx) };
            self.ctx = ptr::null_mut();
        }
        self.cached.clear();
    }

    fn reset(&mut self) {
        unsafe { (self.api.llama_memory_clear)((self.api.llama_get_memory)(self.ctx), true) };
        self.cached.clear();
    }

    /// Brings the KV cache to exactly `prompt`, decoding only what it lacks.
    fn prefill(&mut self, prompt: &[i32]) -> Result<()> {
        let shared = self
            .cached
            .iter()
            .zip(prompt)
            .take_while(|(a, b)| a == b)
            .count();
        // the last token is always decoded, for its logits
        let mut keep = shared.min(prompt.len() - 1);
        let mem = unsafe { (self.api.llama_get_memory)(self.ctx) };
        // recurrent and hybrid memory cannot roll back to a position
        if keep < self.cached.len()
            && !unsafe { (self.api.llama_memory_seq_rm)(mem, 0, keep as i32, -1) }
        {
            self.reset();
            keep = 0;
        }
        self.cached.truncate(keep);
        for chunk in prompt[keep..].chunks(BATCH) {
            if let Err(e) = self.api.decode(self.ctx, chunk, self.cached.len(), false) {
                self.reset();
                return Err(e);
            }
            self.cached.extend_from_slice(chunk);
        }
        Ok(())
    }
}

impl Reader for Engine {
    fn render(&self, content: &str, kwargs: &Map<String, Value>) -> Result<String, Error> {
        self.template
            .render(content, kwargs)
            .map_err(|e| Error::Refused(e.to_string()))
    }

    fn encode(&self, text: &str) -> Result<Vec<i32>, Error> {
        self.api
            .tokenize(self.vocab, text, false, true)
            .map_err(|e| Error::Failed(e.to_string()))
    }

    fn bos(&self) -> Option<i32> {
        (self.add_bos && self.bos >= 0).then_some(self.bos)
    }

    fn special(&self, token: i32) -> Option<String> {
        let attr = unsafe { (self.api.llama_vocab_get_attr)(self.vocab, token) };
        let splits =
            attr & (ffi::LLAMA_TOKEN_ATTR_CONTROL | ffi::LLAMA_TOKEN_ATTR_USER_DEFINED) != 0;
        token_text(self.api, self.vocab, token).filter(|_| splits)
    }

    fn context(&self) -> usize {
        self.max_ctx as usize
    }

    fn read(&mut self, prompt: &[i32], tokens: &[i32]) -> Result<Vec<f64>, Error> {
        let failed = |e: anyhow::Error| Error::Failed(format!("{e:#}"));
        if prompt.is_empty() {
            return Err(Error::Refused("the prompt has no tokens".into()));
        }
        self.ensure_context(prompt.len() + 1)?;
        self.prefill(prompt).map_err(failed)?;
        let logits = unsafe { (self.api.llama_get_logits_ith)(self.ctx, -1) };
        if logits.is_null() {
            return Err(failed(anyhow!("libllama returned no logits")));
        }
        let logits = unsafe { std::slice::from_raw_parts(logits, self.n_vocab) };
        // log softmax over the whole vocabulary
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let total: f64 = logits.iter().map(|&l| (l as f64 - max).exp()).sum();
        let log_total = max + total.ln();
        tokens
            .iter()
            .map(|&t| {
                logits
                    .get(t as usize)
                    .map(|&l| l as f64 - log_total)
                    .with_context(|| format!("token {t} is not in the vocabulary"))
                    .map_err(failed)
            })
            .collect()
    }
}
