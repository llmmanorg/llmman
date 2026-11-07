//! The Canary Transformer decoder (NeMo `TransformerDecoder`, pre-LN,
//! ReLU feed-forward, untied head) as ggml graphs on [`Backend`], after
//! transcribe.cpp's `canary/decoder.cpp` at db678a0. A [`Session`] holds
//! the per-request state: the encoder output projected once into the
//! cross-attention keys and values of every layer, the self-attention
//! KV cache the prefill and the steps write through `set_rows` and read
//! through strided views, and the step graph built once over the full
//! cache window. [`Session::prefill`] runs the prompt tokens in one
//! graph and returns the logits of the last position with the host
//! argmax; [`StepGraph::step`] feeds one token and returns the next.
//! [`transcribe`] runs the whole greedy loop from the encoder output to
//! the text.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};

use super::encoder::layer_norm;
use super::graph_nodes;
use super::model::{CanaryModel, CanaryTensors, DecLayer};
use super::prompt;
use crate::mediagen::backend::{Backend, Graph, Resident};
use crate::mediagen::ffi::{ty, Tensor};

/// Decoder steps a request may take beyond its encoder frames before it
/// is cut off, as transcribe.cpp bounds its loop.
pub const STEP_SLACK: usize = 512;

/// Intermediate tensor [`Session::prefill`] returns next to the logits,
/// in the order requested; each is token-major `[n_tokens, d_model]` f32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Probe {
    /// The embedding after its LayerNorm (NeMo `dec.embed.in`).
    EmbedIn,
    /// Layer `l`'s self-attention output before the residual add
    /// (NeMo `dec.layer.l.self_attn.out`).
    SelfAttn(usize),
    /// Layer `l`'s cross-attention output before the residual add.
    CrossAttn(usize),
    /// Layer `l`'s feed-forward output before the residual add.
    Ffn(usize),
}

/// One prefill pass over the prompt.
pub struct Prefill {
    /// `[vocab]` logits of the last prompt position, before softmax.
    pub logits: Vec<f32>,
    /// Host argmax of `logits`: the greedy first output token.
    pub next: i32,
    /// One buffer per requested [`Probe`], in request order.
    pub probes: Vec<Vec<f32>>,
}

/// One greedy transcription, as [`transcribe`] returns it.
pub struct Transcript {
    /// The output token ids, without the end-of-sequence token.
    pub ids: Vec<i32>,
    /// `ids` through [`super::model::Vocab::detokenize`].
    pub text: String,
    /// Wall time of the prompt prefill graph.
    pub prefill: Duration,
    /// Wall time of every step together.
    pub steps: Duration,
    /// Steps run: one per token fed back after the prefill.
    pub n_steps: usize,
}

/// The decoder's hyperparameters, checked once per loaded model.
/// Weights stay in the [`CanaryModel`] a [`Session`] borrows.
pub struct Decoder {
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
    n_layers: usize,
    vocab: usize,
    /// `dec_max_position`: rows of the stored position table and the
    /// largest KV cache a session may ask for.
    max_ctx: usize,
    /// Whether the compute buffer sizes were logged already (once per
    /// decoder: the step graph on the first session, the prefill on
    /// its first run).
    logged_step: AtomicBool,
    logged_prefill: AtomicBool,
}

impl Decoder {
    /// Checks the hyperparameters this graph is written for.
    pub fn new(model: &CanaryModel) -> Result<Decoder> {
        let hp = model.hp();
        if hp.dec_activation != "relu" {
            bail!(
                "decoder: activation {:?} (the graph applies relu)",
                hp.dec_activation
            );
        }
        if !hp.dec_pre_ln {
            bail!("decoder: post-LN layers (the graph is pre-LN)");
        }
        if hp.encoder_decoder_proj {
            bail!("decoder: files with an encoder-decoder projection are not supported yet");
        }
        if hp.dec_n_heads == 0 || !hp.dec_d_model.is_multiple_of(hp.dec_n_heads) {
            bail!(
                "decoder: d_model {} is not a multiple of {} heads",
                hp.dec_d_model,
                hp.dec_n_heads
            );
        }
        Ok(Decoder {
            d_model: hp.dec_d_model as usize,
            n_heads: hp.dec_n_heads as usize,
            head_dim: (hp.dec_d_model / hp.dec_n_heads) as usize,
            n_layers: hp.dec_n_layers as usize,
            vocab: hp.dec_vocab_size as usize,
            max_ctx: hp.dec_max_position as usize,
            logged_step: AtomicBool::new(false),
            logged_prefill: AtomicBool::new(false),
        })
    }
}

/// The step graph of a [`Session`] on a graph the caller owns, built by
/// [`Session::step_graph`] with fixed shapes and recomputed by
/// [`StepGraph::step`] on updated inputs. Its inputs, in order: `[1]` I32
/// the token fed in; `[1]` I32 its position, also the cache row it is
/// written to; `[max_kv, 1]` F32 the mask over the full cache window
/// ([`STEP_TOKENS`], [`STEP_POS`], [`STEP_MASK`]). It keeps the session
/// that built it, whose caches it reads and writes, so a step can never
/// run against another session.
pub struct StepGraph<'o> {
    session: &'o Session<'o>,
    gr: &'o Graph<'o>,
    /// Output index of the `[vocab]` F32 logits.
    logits: usize,
    /// Host copy of the mask the loop rewrites in place.
    mask_buf: RefCell<Vec<f32>>,
}

/// Input indices of the step graph, in the order [`Session::step_graph`]
/// adds them.
const STEP_TOKENS: usize = 0;
const STEP_POS: usize = 1;
const STEP_MASK: usize = 2;

/// The decoder body on a graph: the logits of the last token and the
/// tensors the probes read.
struct Body<'g> {
    logits: Tensor<'g>,
    embed_in: Tensor<'g>,
    /// Per layer: self-attention, cross-attention and feed-forward
    /// outputs before their residual add.
    sub: Vec<[Tensor<'g>; 3]>,
}

/// One transcription request: the cross-attention memory and the
/// self-attention cache of every layer, resident on the backend. The
/// step graph lives outside, on a graph the caller owns
/// ([`Session::step_graph`]), so nothing here borrows itself.
pub struct Session<'a> {
    dec: &'a Decoder,
    model: &'a CanaryModel,
    be: &'a Backend,
    /// Encoder frames in the cross-attention memory.
    frames: usize,
    /// Rows of each self-attention cache tensor.
    max_kv: usize,
    /// Tokens already written to the cache; the next token's position.
    n_past: Cell<usize>,
    /// `2 * n_layers` F32 `[d_model, frames]`: layer `l`'s cross keys at
    /// `2l` and values at `2l + 1`.
    cross: Resident,
    /// `2 * n_layers` F32 `[d_model, max_kv]`: layer `l`'s self keys at
    /// `2l` and values at `2l + 1`, row `p` holding position `p`; zero
    /// past `n_past`.
    cache: Resident,
}

impl<'a> Session<'a> {
    /// Projects `enc_out`, frame-major `[frames, d_model]` as
    /// `speech::encoder::Run::out` holds it, into every layer's cross
    /// keys and values in one graph, allocates a zeroed self-attention
    /// cache of `max_kv` positions (at most the model's `max_position`)
    /// (see [`Session::step_graph`] for the step graph over it).
    pub fn new(
        dec: &'a Decoder,
        model: &'a CanaryModel,
        be: &'a Backend,
        enc_out: &[f32],
        frames: usize,
        max_kv: usize,
    ) -> Result<Session<'a>> {
        if be.api.ggml_set_rows.is_none() {
            bail!("decoder: this llama.cpp release has no ggml_set_rows; use the pinned release");
        }
        if frames == 0 || enc_out.len() != frames * dec.d_model {
            bail!(
                "decoder: {} encoder values are not [{frames}, {}]",
                enc_out.len(),
                dec.d_model
            );
        }
        if max_kv == 0 || max_kv > dec.max_ctx {
            bail!(
                "decoder: a cache of {max_kv} positions (the position table has {})",
                dec.max_ctx
            );
        }
        let tensors = model.tensors()?;
        let layers = &tensors.dec_layers;
        if layers.len() != dec.n_layers {
            bail!(
                "decoder: {} layer tensors for {} layers",
                layers.len(),
                dec.n_layers
            );
        }
        let cross = resident_pairs(be, dec.n_layers, [dec.d_model as i64, frames as i64])?;
        let cache = resident_pairs(be, dec.n_layers, [dec.d_model as i64, max_kv as i64])?;
        // The step graph reads the whole window, so the rows no token
        // has written yet must be finite for their masked scores to
        // drop out of the softmax.
        cache.clear();

        let gr = Graph::new(be.api, 64 + dec.n_layers * 16)?;
        let enc = gr.input_f32(&[dec.d_model as i64, frames as i64], enc_out, "enc_out");
        let g = &gr;
        for (l, layer) in layers.iter().enumerate() {
            gr.store(
                g.linear(enc, layer.cross_k, Some(layer.cross_k_b)),
                cross.tensor(2 * l),
            );
            gr.store(
                g.linear(enc, layer.cross_v, Some(layer.cross_v_b)),
                cross.tensor(2 * l + 1),
            );
        }
        gr.compute(be)?;

        Ok(Session {
            dec,
            model,
            be,
            frames,
            max_kv,
            n_past: Cell::new(0),
            cross,
            cache,
        })
    }

    /// Positions written to the self-attention cache so far.
    pub fn n_past(&self) -> usize {
        self.n_past.get()
    }

    /// The step graph over this session's caches on `gr`, built once
    /// with fixed shapes and recomputed by [`StepGraph::step`] on updated
    /// inputs. `gr` and the session outlive it, so the graph never reads
    /// a cache that is gone. `gr` must be fresh: the step's three inputs
    /// are addressed by index, so a graph that already holds inputs is
    /// refused.
    pub fn step_graph<'o>(&'o self, gr: &'o Graph<'o>) -> Result<StepGraph<'o>> {
        if gr.n_inputs() != 0 {
            bail!(
                "decoder step graph: the graph already holds {} inputs, expected a fresh one",
                gr.n_inputs()
            );
        }
        let dec = self.dec;
        let tensors = self.model.tensors()?;
        let mask_buf = causal_mask(self.max_kv, 1, 0);
        let tokens = gr.input_i32(&[1], &[0], "tokens");
        let pos = gr.input_i32(&[1], &[0], "positions");
        let mask = gr.input_f32(&[self.max_kv as i64, 1], &mask_buf, "mask");
        let body = body(
            dec,
            &tensors,
            &self.cross,
            &self.cache,
            self.frames,
            gr,
            tokens,
            pos,
            mask,
            1,
            self.max_kv,
        );
        let logits = gr.mark_output(body.logits);
        if !dec.logged_step.swap(true, Ordering::Relaxed) {
            let bytes = gr.measure(self.be)?;
            eprintln!(
                "[llmman] speech: decoder step compute buffer {bytes} bytes over a cache of \
                 {} positions and {} frames",
                self.max_kv, self.frames
            );
        }
        Ok(StepGraph {
            session: self,
            gr,
            logits,
            mask_buf: RefCell::new(mask_buf),
        })
    }

    /// Runs `ids` at positions `n_past..n_past + n` in one graph, writes
    /// their keys and values into the cache and returns the logits of
    /// the last one with the requested `probes`.
    ///
    /// The embedding is `get_rows(token, ids) + get_rows(pos_enc,
    /// positions)` through the embedding LayerNorm; NeMo applies no
    /// `sqrt(d_model)` scale. Each layer is `x += self_attn(norm1(x))`
    /// over the cache window `[0, n_past + n)` under the causal mask of
    /// [`causal_mask`], `x += cross_attn(norm2(x))` over the encoder
    /// memory without a mask, `x += ffn_down(relu(ffn_up(norm3(x))))`.
    /// The final LayerNorm and the head run on the last position only.
    pub fn prefill(&self, ids: &[i32], probes: &[Probe]) -> Result<Prefill> {
        let dec = self.dec;
        let n = ids.len();
        if n == 0 {
            bail!("decoder: prefill of no tokens");
        }
        let n_past = self.n_past.get();
        if n_past + n > self.max_kv {
            bail!(
                "decoder: {n} tokens after {n_past} exceed the cache of {} positions",
                self.max_kv
            );
        }
        for &id in ids {
            self.check_token(id)?;
        }
        let n_kv = n_past + n;

        let tensors = self.model.tensors()?;
        let gr = Graph::new(self.be.api, graph_nodes(dec.n_layers))?;
        let tokens = gr.input_i32(&[n as i64], ids, "tokens");
        // Positions double as the cache rows the keys and values go to.
        let positions: Vec<i32> = (n_past..n_kv).map(|p| p as i32).collect();
        let pos = gr.input_i32(&[n as i64], &positions, "positions");
        let mask = gr.input_f32(
            &[n_kv as i64, n as i64],
            &causal_mask(n_kv, n, n_past),
            "mask",
        );
        let body = body(
            dec,
            &tensors,
            &self.cross,
            &self.cache,
            self.frames,
            &gr,
            tokens,
            pos,
            mask,
            n,
            n_kv,
        );
        gr.mark_output(body.logits);
        let probe_tensors = probes
            .iter()
            .map(|p| {
                let (l, i) = match *p {
                    Probe::EmbedIn => return Ok(body.embed_in),
                    Probe::SelfAttn(l) => (l, 0),
                    Probe::CrossAttn(l) => (l, 1),
                    Probe::Ffn(l) => (l, 2),
                };
                body.sub.get(l).map(|s| s[i]).ok_or_else(|| {
                    anyhow!(
                        "decoder: layer {l} probe requested of {} layers",
                        body.sub.len()
                    )
                })
            })
            .collect::<Result<Vec<Tensor>>>()?;
        for &t in &probe_tensors {
            gr.mark_output(t);
        }

        if !dec.logged_prefill.swap(true, Ordering::Relaxed) {
            let bytes = gr.measure(self.be)?;
            eprintln!(
                "[llmman] speech: decoder prefill compute buffer {bytes} bytes for {n} tokens \
                 over {} frames",
                self.frames
            );
        }
        gr.compute(self.be)?;
        self.n_past.set(n_kv);
        let logits = gr.output_f32(body.logits);
        Ok(Prefill {
            next: argmax(&logits),
            logits,
            probes: probe_tensors.iter().map(|&t| gr.output_f32(t)).collect(),
        })
    }

    fn check_token(&self, id: i32) -> Result<()> {
        if id < 0 || id as usize >= self.dec.vocab {
            bail!(
                "decoder: token id {id} is outside the vocabulary of {}",
                self.dec.vocab
            );
        }
        Ok(())
    }
}

impl StepGraph<'_> {
    /// Writes `token` at the session's `n_past` and returns the greedy
    /// next token, recomputing the graph on its updated inputs without
    /// allocating. Fails once the cache is full.
    pub fn step(&self, token: i32) -> Result<i32> {
        let session = self.session;
        session.check_token(token)?;
        let p = session.n_past.get();
        if p >= session.max_kv {
            bail!("decoder: the cache of {} positions is full", session.max_kv);
        }
        let gr = self.gr;
        {
            let mut mask_buf = self.mask_buf.borrow_mut();
            step_mask(&mut mask_buf, p);
            gr.update_input_f32(STEP_MASK, &mask_buf);
        }
        gr.update_input_i32(STEP_TOKENS, &[token]);
        gr.update_input_i32(STEP_POS, &[p as i32]);
        gr.compute(session.be)?;
        let logits = gr.output_f32(gr.output(self.logits));
        session.n_past.set(p + 1);
        Ok(argmax(&logits))
    }
}

/// Transcribes the encoder output `enc_out`, frame-major `[frames,
/// d_model]`, in `lang` with (`pnc`) or without punctuation: the canary2
/// prompt (a typed [`prompt::PromptError`] before any graph runs), one
/// prefill, then greedy steps until `endoftext`, the cache of
/// `min(max_ctx, prompt + frames + STEP_SLACK)` positions, or the
/// backend's cancellation flag, which is an error.
pub fn transcribe(
    dec: &Decoder,
    model: &CanaryModel,
    be: &Backend,
    enc_out: &[f32],
    frames: usize,
    lang: &str,
    pnc: bool,
) -> Result<Transcript> {
    let prompt = prompt::build(model.hp(), &model.vocab, lang, pnc)?;
    let max_kv = cache_size(dec.max_ctx, prompt.len(), frames);
    let session = Session::new(dec, model, be, enc_out, frames, max_kv)?;
    let started = Instant::now();
    let first = session.prefill(&prompt, &[])?.next;
    let prefill = started.elapsed();
    // Each step writes one cache row; the prefill left max_kv - prompt.
    let max_steps = max_kv - prompt.len();
    let gr = Graph::new(be.api, graph_nodes(dec.n_layers))?;
    let st = session.step_graph(&gr)?;
    let started = Instant::now();
    let (ids, n_steps) = greedy(
        first,
        model.hp().eos as i32,
        max_steps,
        || be.cancelled(),
        |token| st.step(token),
    )?;
    Ok(Transcript {
        text: model.vocab.detokenize(&ids),
        ids,
        prefill,
        steps: started.elapsed(),
        n_steps,
    })
}

/// Cache rows for a prompt of `prompt_len` tokens over `frames` encoder
/// frames: room for `frames + STEP_SLACK` steps, clamped to `max_ctx`.
pub fn cache_size(max_ctx: usize, prompt_len: usize, frames: usize) -> usize {
    max_ctx.min(prompt_len + frames + STEP_SLACK)
}

/// The greedy loop: `first` is the prefill's prediction, `step(token)`
/// writes a token and returns the next. Appends tokens until `eos`
/// (not appended) or `max_steps` steps, failing when `cancelled()`
/// before a step. Returns the ids and the steps run. The ids vector is
/// the loop's only allocation.
fn greedy(
    first: i32,
    eos: i32,
    max_steps: usize,
    mut cancelled: impl FnMut() -> bool,
    mut step: impl FnMut(i32) -> Result<i32>,
) -> Result<(Vec<i32>, usize)> {
    let mut ids = Vec::with_capacity(max_steps + 1);
    let mut next = first;
    let mut n_steps = 0;
    while next != eos {
        ids.push(next);
        if n_steps == max_steps {
            break;
        }
        if cancelled() {
            bail!("decoder: cancelled after {} tokens", ids.len());
        }
        next = step(next)?;
        n_steps += 1;
    }
    Ok((ids, n_steps))
}

/// The decoder on the graph `gr` for `n` tokens: `tokens` and `pos` are
/// `[n]` I32 inputs (the positions double as the cache rows the keys and
/// values are written to), `mask` is `[n_kv, n]` F32 over the cache
/// window of `n_kv` rows read by the self-attention. See
/// [`Session::prefill`] for the recipe.
#[allow(clippy::too_many_arguments)]
fn body<'g>(
    dec: &Decoder,
    t: &CanaryTensors<'g>,
    cross: &'g Resident,
    cache: &'g Resident,
    frames: usize,
    gr: &'g Graph<'g>,
    tokens: Tensor<'g>,
    pos: Tensor<'g>,
    mask: Tensor<'g>,
    n: usize,
    n_kv: usize,
) -> Body<'g> {
    let d_model = dec.d_model as i64;
    let (head_dim, n_heads) = (dec.head_dim as i64, dec.n_heads as i64);
    let g = gr;
    let embed = &t.dec_embed;
    let x = g.add(
        g.get_rows(embed.token, tokens),
        g.get_rows(embed.pos_enc, pos),
    );
    let embed_in = layer_norm(g, x, embed.norm, embed.norm_b);
    let mut x = embed_in;
    // Strided view of a resident `[d_model, rows]` tensor as
    // `[head_dim, n_heads, window]`.
    let window = |cache, rows: i64| {
        g.view_3d(
            cache,
            head_dim,
            n_heads,
            rows,
            dec.head_dim * 4,
            dec.d_model * 4,
            0,
        )
    };
    let mut sub = Vec::with_capacity(dec.n_layers);
    for (l, layer) in t.dec_layers.iter().enumerate() {
        let (k_cache, v_cache) = (cache.tensor(2 * l), cache.tensor(2 * l + 1));
        let h = layer_norm(g, x, layer.norm1, layer.norm1_b);
        let q = g.linear(h, layer.self_q, Some(layer.self_q_b));
        let k = g.linear(h, layer.self_k, Some(layer.self_k_b));
        let v = g.linear(h, layer.self_v, Some(layer.self_v_b));
        // The writes go into the graph before the reads of the same
        // cache below, so they run first.
        // `Session::new` refuses a release without `ggml_set_rows`.
        let set_rows = |a, b| {
            g.set_rows(a, b, pos)
                .expect("ggml_set_rows checked at session start")
        };
        gr.mark_output(set_rows(k_cache, k));
        gr.mark_output(set_rows(v_cache, v));
        let o = attention(
            g,
            q,
            window(k_cache, n_kv as i64),
            window(v_cache, n_kv as i64),
            Some(mask),
        );
        let self_out = g.linear(o, layer.self_o, Some(layer.self_o_b));
        x = g.add(x, self_out);

        let h = layer_norm(g, x, layer.norm2, layer.norm2_b);
        let q = g.linear(h, layer.cross_q, Some(layer.cross_q_b));
        let o = attention(
            g,
            q,
            window(cross.tensor(2 * l), frames as i64),
            window(cross.tensor(2 * l + 1), frames as i64),
            None,
        );
        let cross_out = g.linear(o, layer.cross_o, Some(layer.cross_o_b));
        x = g.add(x, cross_out);

        let ffn_out = feed_forward(g, layer, x);
        x = g.add(x, ffn_out);
        sub.push([self_out, cross_out, ffn_out]);
    }
    let x = layer_norm(g, x, t.dec_norm, t.dec_norm_b);
    let nb = x.strides();
    let last = g.view_2d(x, d_model, 1, nb[1], (n - 1) * nb[1]);
    let logits = g.linear(last, t.head, Some(t.head_b));
    Body {
        logits,
        embed_in,
        sub,
    }
}

/// `2 * n_layers` resident F32 tensors of shape `ne`.
fn resident_pairs(be: &Backend, n_layers: usize, ne: [i64; 2]) -> Result<Resident> {
    let shapes = vec![ne.as_slice(); 2 * n_layers];
    Resident::new(be, ty::F32, &shapes)
}

/// Multi-head attention of the queries `q`, `[d_model, n]`, over the key
/// and value windows `[head_dim, n_heads, n_kv]`, with `mask` `[n_kv, n]`
/// F32 or `None`; returns `[d_model, n]` with the heads merged head-major
/// as NeMo does. Scores accumulate in F32 and are scaled by
/// `1 / sqrt(head_dim)` inside the softmax.
fn attention<'g>(
    g: &'g Graph<'g>,
    q: Tensor<'g>,
    k: Tensor<'g>,
    v: Tensor<'g>,
    mask: Option<Tensor<'g>>,
) -> Tensor<'g> {
    let [d_model, n, _, _] = q.shape();
    let [head_dim, n_heads, _, _] = k.shape();
    // [head_dim, n, head] and [head_dim, n_kv, head]
    let q = g.permute(g.reshape_3d(q, head_dim, n_heads, n), 0, 2, 1, 3);
    let k = g.permute(k, 0, 2, 1, 3);
    // [n_kv, head_dim, head] so that probs . v gives [n, head_dim, head]
    let v = g.cont(g.permute(v, 1, 2, 0, 3));
    // [n_kv, n, head]
    let scores = g.prec_f32(g.mul_mat(k, q));
    let probs = g.soft_max_ext(scores, mask, 1.0 / (head_dim as f32).sqrt(), 0.0);
    let o = g.prec_f32(g.mul_mat(probs, v));
    // [head_dim, head, n] flattened head-major
    g.cont_2d(g.permute(o, 2, 0, 1, 3), d_model, n)
}

/// `ffn_down(relu(ffn_up(norm3(x))))` without its residual.
fn feed_forward<'g>(g: &'g Graph<'g>, layer: &DecLayer<'g>, x: Tensor<'g>) -> Tensor<'g> {
    let h = layer_norm(g, x, layer.norm3, layer.norm3_b);
    let h = g.relu(g.linear(h, layer.ffn_up, Some(layer.ffn_up_b)));
    g.linear(h, layer.ffn_down, Some(layer.ffn_down_b))
}

/// The additive self-attention mask for `n_tokens` queries at positions
/// `n_past..n_past + n_tokens` over a cache window of `n_kv` positions,
/// as the ggml `[n_kv, n_tokens]` F32 input (row-major `[query][key]`):
/// `0` where the key position is at most the query's, `-inf` above it.
/// Positions from `n_past + n_tokens` to `n_kv` are masked for every
/// query, so a step graph over a fixed window gates the unwritten rows.
pub fn causal_mask(n_kv: usize, n_tokens: usize, n_past: usize) -> Vec<f32> {
    assert!(
        n_past + n_tokens <= n_kv,
        "causal_mask: {n_tokens} tokens after {n_past} exceed the window of {n_kv}"
    );
    let mut mask = vec![f32::NEG_INFINITY; n_kv * n_tokens];
    for (q, row) in mask.chunks_exact_mut(n_kv).enumerate() {
        row[..=n_past + q].fill(0.0);
    }
    mask
}

/// Rewrites `mask`, the `[n_kv, 1]` step mask, for the one token at
/// position `n_past`: `0` up to and including its own row, `-inf`
/// above, the same values [`causal_mask`] gives for one token.
pub fn step_mask(mask: &mut [f32], n_past: usize) {
    assert!(
        n_past < mask.len(),
        "step_mask: position {n_past} is outside the window of {}",
        mask.len()
    );
    mask[..=n_past].fill(0.0);
    mask[n_past + 1..].fill(f32::NEG_INFINITY);
}

/// Index of the largest value, the first one on ties.
fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    const INF: f32 = f32::NEG_INFINITY;

    #[test]
    fn causal_mask_prefill_is_lower_triangular() {
        let mask = causal_mask(3, 3, 0);
        assert_eq!(
            mask,
            vec![0.0, INF, INF, 0.0, 0.0, INF, 0.0, 0.0, 0.0],
            "row q keeps keys 0..=q"
        );
    }

    #[test]
    fn causal_mask_offsets_the_queries_by_n_past() {
        // two tokens at positions 2 and 3 over a window of 4
        let mask = causal_mask(4, 2, 2);
        assert_eq!(mask, vec![0.0, 0.0, 0.0, INF, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn causal_mask_step_gates_the_unwritten_window() {
        // one token at position 1 over a fixed window of 5
        let mask = causal_mask(5, 1, 1);
        assert_eq!(mask, vec![0.0, 0.0, INF, INF, INF]);
    }

    #[test]
    #[should_panic(expected = "causal_mask: 3 tokens after 2 exceed the window of 4")]
    fn causal_mask_rejects_tokens_past_the_window() {
        causal_mask(4, 3, 2);
    }

    #[test]
    fn step_mask_matches_causal_mask_for_one_token() {
        let mut mask = vec![1.0; 5];
        for p in 0..5 {
            step_mask(&mut mask, p);
            assert_eq!(mask, causal_mask(5, 1, p), "position {p}");
        }
        // moving back down reopens nothing above the position
        step_mask(&mut mask, 1);
        assert_eq!(mask, vec![0.0, 0.0, INF, INF, INF]);
    }

    #[test]
    #[should_panic(expected = "step_mask: position 5 is outside the window of 5")]
    fn step_mask_rejects_a_position_past_the_window() {
        step_mask(&mut [0.0; 5], 5);
    }

    #[test]
    fn cache_size_clamps_to_max_ctx() {
        assert_eq!(cache_size(1024, 10, 132), 10 + 132 + STEP_SLACK);
        assert_eq!(cache_size(1024, 10, 2000), 1024);
        assert_eq!(cache_size(1024, 10, 1024 - 10 - STEP_SLACK), 1024);
    }

    #[test]
    fn greedy_stops_at_eos_without_appending_it() {
        let mut fed = Vec::new();
        let (ids, n_steps) = greedy(
            7,
            3,
            100,
            || false,
            |t| {
                fed.push(t);
                Ok(if t == 9 { 3 } else { t + 1 })
            },
        )
        .unwrap();
        assert_eq!(ids, vec![7, 8, 9]);
        assert_eq!(n_steps, 3);
        assert_eq!(fed, vec![7, 8, 9]);
    }

    #[test]
    fn greedy_runs_at_most_max_steps() {
        let (ids, n_steps) = greedy(1, 0, 3, || false, |t| Ok(t + 1)).unwrap();
        assert_eq!(
            ids,
            vec![1, 2, 3, 4],
            "the prefill's token plus one per step"
        );
        assert_eq!(n_steps, 3);
        let (ids, n_steps) = greedy(1, 0, 0, || false, |_| unreachable!()).unwrap();
        assert_eq!((ids, n_steps), (vec![1], 0));
    }

    #[test]
    fn greedy_returns_nothing_when_the_prefill_predicts_eos() {
        let (ids, n_steps) = greedy(3, 3, 10, || false, |_| unreachable!()).unwrap();
        assert!(ids.is_empty());
        assert_eq!(n_steps, 0);
    }

    #[test]
    fn greedy_fails_when_cancelled_before_a_step() {
        let mut polls = 0;
        let err = greedy(
            1,
            0,
            10,
            || {
                polls += 1;
                polls == 2
            },
            |t| Ok(t + 1),
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "decoder: cancelled after 2 tokens");
    }

    #[test]
    fn argmax_takes_the_first_largest() {
        assert_eq!(argmax(&[-1.0, 3.5, 3.5, 2.0]), 1);
        assert_eq!(argmax(&[0.0]), 0);
        assert_eq!(argmax(&[INF, INF, 1.0]), 2);
    }
}
