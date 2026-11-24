//! A Qwen3 text model from its llama.cpp GGUF, run in mediagen's own graph
//! (`cosmos3::dit`'s decoder) rather than through llama.cpp, so that any
//! attention mask applies: padded prompts mask their padding the way the
//! Transformers reference does. llama.cpp only tokenizes (vocabulary only).

use std::path::Path;

use anyhow::{anyhow, bail, Result};
use serde_json::json;

use super::backend::Backend;
use super::cosmos3::{self, dit};
use super::ffi::Api;
use super::ltx::text::TextEncoder;
use super::weights::{read_index, LoadOpts, Weights};

/// Qwen3's `rope_theta` (llama.cpp keeps it as a numeric KV, not read here).
const ROPE_THETA: f64 = 1_000_000.0;

pub struct Qwen3Text {
    pub hp: cosmos3::Hparams,
    w: Weights,
    vocab: Box<TextEncoder>,
}

/// llama.cpp's tensor names -> the Transformers ones `cosmos3::dit` reads;
/// the final norm and the LM head are never loaded.
pub fn tensor_name(n: &str) -> Option<String> {
    if n == "token_embd.weight" {
        return Some("embed_tokens.weight".into());
    }
    let rest = n.strip_prefix("blk.")?;
    let (l, t) = rest.split_once('.')?;
    let to = match t {
        "attn_norm.weight" => "input_layernorm.weight",
        "attn_q.weight" => "self_attn.to_q.weight",
        "attn_k.weight" => "self_attn.to_k.weight",
        "attn_v.weight" => "self_attn.to_v.weight",
        "attn_output.weight" => "self_attn.to_out.weight",
        "attn_q_norm.weight" => "self_attn.norm_q.weight",
        "attn_k_norm.weight" => "self_attn.norm_k.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        _ => return None,
    };
    Some(format!("layers.{l}.{to}"))
}

impl Qwen3Text {
    pub fn load(api: &'static Api, be: &Backend, path: &Path) -> Result<Qwen3Text> {
        let idx = read_index(path)?;
        let arch = idx.metadata.get("general.architecture").map(String::as_str);
        if arch != Some("qwen3") {
            bail!("{}: expected a qwen3 GGUF, got {arch:?}", path.display());
        }
        let ne = |n: &str| -> Result<Vec<i64>> {
            idx.tensors
                .iter()
                .find(|d| d.name == n)
                .map(|d| d.ne.clone())
                .ok_or_else(|| anyhow!("{}: no {n}", path.display()))
        };
        let hidden = ne("token_embd.weight")?[0];
        let head_dim = ne("blk.0.attn_q_norm.weight")?[0];
        let n_layers = idx
            .tensors
            .iter()
            .filter_map(|d| {
                d.name
                    .strip_prefix("blk.")?
                    .split('.')
                    .next()?
                    .parse::<i64>()
                    .ok()
            })
            .max()
            .map_or(0, |m| m + 1);
        let hp = cosmos3::Hparams::from_config(&json!({
            "hidden_size": hidden,
            "num_hidden_layers": n_layers,
            "num_attention_heads": ne("blk.0.attn_q.weight")?[1] / head_dim,
            "num_key_value_heads": ne("blk.0.attn_k.weight")?[1] / head_dim,
            "head_dim": head_dim,
            "rms_norm_eps": 1e-6,
            "rope_theta": ROPE_THETA,
            // text only: one position for every axis, plain 1-D RoPE
            "rope_axes_dim": [head_dim / 2, 0, 0],
            "qk_norm_for_text": true,
        }))?;
        let w = Weights::load(
            api,
            path,
            be.weight_buft(),
            &LoadOpts {
                rename: Some(Box::new(tensor_name)),
                transform: None,
            },
        )?;
        let vocab = TextEncoder::load_vocab(api, &path.to_string_lossy())?;
        Ok(Qwen3Text { hp, w, vocab })
    }

    pub fn weights(&self) -> &Weights {
        &self.w
    }

    /// Token ids of `text` as is: special tokens parsed, no BOS.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .vocab
            .tokenize(text, false, true)?
            .into_iter()
            .map(|t| t as u32)
            .collect())
    }

    /// The id of a single special token such as `<|endoftext|>`.
    pub fn token(&self, piece: &str) -> Result<u32> {
        match self.tokenize(piece)?[..] {
            [id] => Ok(id),
            ref ids => bail!("{piece:?} is not one token ({ids:?})"),
        }
    }

    /// Hidden states `states` (`i` = after `i` layers) of `ids`, the first
    /// `n_real` of them the prompt and the rest padding nothing attends to.
    pub fn hidden_states(
        &self,
        be: &Backend,
        ids: &[u32],
        n_real: usize,
        states: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        let mask = dit::padded_causal_mask(ids.len() as i64, n_real as i64);
        dit::text_hidden_states(&self.hp, &self.w, be, ids, &mask, states)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_names_map_to_the_transformers_ones() {
        assert_eq!(
            tensor_name("token_embd.weight").as_deref(),
            Some("embed_tokens.weight")
        );
        assert_eq!(
            tensor_name("blk.12.attn_k_norm.weight").as_deref(),
            Some("layers.12.self_attn.norm_k.weight")
        );
        assert_eq!(
            tensor_name("blk.0.ffn_down.weight").as_deref(),
            Some("layers.0.mlp.down_proj.weight")
        );
        assert_eq!(tensor_name("output_norm.weight"), None);
        assert_eq!(tensor_name("output.weight"), None);
    }
}
