//! Reusable "resolve a local OCI-store reference to servable model files"
//! logic — the CNCF ModelPack (<https://github.com/modelpack/model-spec>)
//! equivalent of `huggingface_hub.snapshot_download`.
//!
//! Originally private to `cmd::serve` (which uses it to decide whether to
//! spawn `llama-server`, `vllm`, or (Apple Silicon macOS) `mlx_lm.server`
//! as its backend for a given model), this
//! module is `pub` so it also backs `cmd::resolve` (`llmman resolve`) — a
//! standalone, scriptable entry point that other tools (e.g. a vLLM plugin
//! that wants vLLM itself, not `llmman`, to be the one serving the model)
//! can shell out to, without needing `llmman serve`'s HTTP daemon or its
//! opinions about which inference backend to launch.
//!
//! [`resolve_model`] finds the one file or directory a server is started on;
//! [`resolve_model_tree`] lays out *every* layer as the model directory, for
//! `llmman resolve --dir` and the Kubernetes CSI driver. Both unpack every
//! layer encoding the ModelPack spec lists (see [`layers`]).
//!
//! Everything here assumes the reference has already been pulled into the
//! local `OciStore` at `store_path` (see `crate::ffi::pull`) — this module
//! only resolves+extracts, it never talks to a registry itself.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context};

use crate::storage::OciStore;

mod layers;
use layers::{Compression, Encoding};

const HF_GGUF_MEDIA_TYPE: &str = "application/vnd.docker.ai.gguf.v3";

/// What kind of model did we find in the OCI store?
pub enum ModelPath {
    /// A GGUF file — serve with llama-server. The second field, when
    /// present, is a companion `--mmproj` projector GGUF (see
    /// [`is_mmproj_layer`]) needed for vision/audio support.
    Gguf(PathBuf, Option<PathBuf>),
    /// A safetensors directory — serve with vllm, or (Apple Silicon
    /// macOS) `mlx_lm.server` — see `cmd::serve::backend::use_mlx_for_safetensors`.
    SafeTensors(PathBuf),
    /// A latent diffusion model (image / video / audio generation) —
    /// served in-process by `crate::mediagen` from the transformer GGUF
    /// plus the sidecars pulled next to it (see
    /// `crate::hf::oci::ANNOTATION_ROLE`), or from a Diffusers pack whose
    /// pipeline it runs (`crate::mediagen::is_native_pipeline`).
    Diffusion(DiffusionPaths),
    /// A Diffusers-layout safetensors directory (a root `model_index.json`,
    /// weights in `transformer/`, `vae/`, ...) such as `nvidia/Cosmos3-Edge`
    /// — served by vLLM-Omni (`vllm serve --omni`); plain vllm cannot.
    Omni(PathBuf),
}

/// The Diffusers pipeline index that marks a [`ModelPath::Omni`] repo.
pub const DIFFUSERS_MODEL_INDEX: &str = "model_index.json";

/// The files of a resolved [`ModelPath::Diffusion`] model.
#[derive(Debug, Clone, Default)]
pub struct DiffusionPaths {
    /// The diffusion transformer GGUF (`--model`), or a Diffusers pack's
    /// `model_index.json` (its components are in `files`).
    pub model: PathBuf,
    /// Video VAE (`--vae`).
    pub vae: Option<PathBuf>,
    /// Audio VAE + vocoder (`--audio-vae`).
    pub audio_vae: Option<PathBuf>,
    /// Text embedding projection / connectors (`--text-proj`).
    pub text_proj: Option<PathBuf>,
    /// Text encoder GGUF (`--text-encoder`).
    pub text_encoder: Option<PathBuf>,
    /// Every raw layer by its `org.cncf.model.filepath` (tokenizer, configs, ...).
    pub files: std::collections::BTreeMap<String, PathBuf>,
}

impl ModelPath {
    /// The local filesystem path this variant resolved to — either a
    /// single `.gguf` file, or the model directory (parent of
    /// `config.json`) for a safetensors checkout.
    pub fn path(&self) -> &Path {
        match self {
            ModelPath::Gguf(p, _) => p,
            ModelPath::SafeTensors(p) | ModelPath::Omni(p) => p,
            ModelPath::Diffusion(d) => &d.model,
        }
    }

    /// The companion `--mmproj` projector file resolved alongside a
    /// `Gguf` model, if any — always `None` for `SafeTensors` (neither
    /// vllm nor mlx_lm.server has an equivalent separate-projector-file
    /// convention).
    pub fn mmproj(&self) -> Option<&Path> {
        match self {
            ModelPath::Gguf(_, mmproj) => mmproj.as_deref(),
            ModelPath::SafeTensors(_) | ModelPath::Diffusion(_) | ModelPath::Omni(_) => None,
        }
    }

    /// A short, stable string identifying which variant this is — used by
    /// `cmd::resolve`'s JSON output and any other consumer that wants to
    /// branch on format without matching the enum directly.
    pub fn format(&self) -> &'static str {
        match self {
            ModelPath::Gguf(..) => "gguf",
            ModelPath::SafeTensors(_) => "safetensors",
            ModelPath::Diffusion(_) => "diffusion",
            ModelPath::Omni(_) => "omni",
        }
    }
}

/// Splits an OCI digest ("sha256:abcd...") down to just its hex portion,
/// which is what the blob store's on-disk layout uses as the filename.
fn digest_hex(digest: &str) -> anyhow::Result<&str> {
    digest
        .split_once(':')
        .map(|(_, hex)| hex)
        .filter(|h| h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit()))
        .ok_or_else(|| anyhow!("malformed digest: {digest}"))
}

fn layer_filepath(l: &crate::storage::oci::Descriptor) -> Option<&str> {
    l.annotations.as_ref().and_then(|a| {
        a.get("org.cncf.model.filepath")
            .or_else(|| a.get("org.opencontainers.image.title"))
            .map(|s| s.as_str())
    })
}

/// The diffusion sidecar role of a layer — see
/// `crate::hf::oci::ANNOTATION_ROLE`.
fn layer_role(l: &crate::storage::oci::Descriptor) -> Option<&str> {
    l.annotations
        .as_ref()
        .and_then(|a| a.get(crate::hf::oci::ANNOTATION_ROLE))
        .map(|s| s.as_str())
}

fn is_gguf_layer(l: &crate::storage::oci::Descriptor) -> bool {
    if l.media_type == HF_GGUF_MEDIA_TYPE {
        return true;
    }
    layer_filepath(l)
        .map(|p| p.to_lowercase().ends_with(".gguf"))
        .unwrap_or(false)
}

fn is_safetensors_layer(l: &crate::storage::oci::Descriptor) -> bool {
    layer_filepath(l)
        .map(|p| p.to_lowercase().ends_with(".safetensors"))
        .unwrap_or(false)
}

/// The layer holding the repo's *root* `model_index.json` (a Diffusers
/// pipeline index; a nested one is some vendored pipeline's, not this
/// repo's).
fn is_diffusers_index_layer(l: &crate::storage::oci::Descriptor) -> bool {
    layer_filepath(l) == Some(DIFFUSERS_MODEL_INDEX)
}

/// A Diffusers-layout repo: a root `model_index.json` next to safetensors.
fn is_diffusers_manifest(manifest: &crate::storage::oci::Manifest) -> bool {
    manifest.layers.iter().any(is_diffusers_index_layer)
        && manifest.layers.iter().any(is_safetensors_layer)
}

/// True if a (already-confirmed-GGUF) layer looks like a multimodal
/// projector rather than the main model — matched by filename containing
/// "mmproj", the de facto convention GGUF repos use, since there's no
/// media-type or metadata distinction to key off instead.
fn is_mmproj_layer(l: &crate::storage::oci::Descriptor) -> bool {
    layer_filepath(l)
        .map(|p| p.to_lowercase().contains("mmproj"))
        .unwrap_or(false)
}

// gguf_architecture/gguf_context_length_override (a GGUF metadata reader
// + --override-kv builder that let --ctx-size force a context above a
// model's own trained length) were tried and removed: llama-server's own
// capping of --ctx-size back down to a model's trained context — see
// cmd::serve::config::context_length_from_env's doc comment — is deliberate, not
// a bug to work around. Defeating that safety net via --override-kv
// produces a real NaN/incoherent-output risk for out-of-distribution
// RoPE positions that llama-server's own warning exists to prevent, for
// a use case (fitting a real agent's system prompt) that a model
// whose trained context is that tight was never going to serve well
// regardless — see docker/sandboxes' own llmmanCtxSize doc comment for the
// model-selection fix that replaced this instead.

/// Extracts a single GGUF layer to a local path, caching under
/// `cache_path` — shared by [`resolve_model`] for both the primary model
/// GGUF and, when present, a companion `--mmproj` GGUF (see
/// [`is_mmproj_layer`]), since both are extracted exactly the same way.
fn extract_gguf_layer(
    store_path: &Path,
    cache_path: &Path,
    layer: &crate::storage::oci::Descriptor,
) -> anyhow::Result<PathBuf> {
    let title = layer_filepath(layer).unwrap_or("model.gguf").to_owned();
    let layer_hex = digest_hex(&layer.digest)?;

    // HF blobs are stored as raw GGUF — use directly. The same goes for
    // the CNCF raw-weight layers `crate::hf::pull` writes: copying a
    // multi-GiB GGUF into the cache would double its disk footprint for
    // nothing.
    if layer.media_type == HF_GGUF_MEDIA_TYPE
        || layer.media_type == crate::hf::oci::MEDIA_TYPE_MODEL_WEIGHT_RAW
    {
        let blob_path = store_path.join("blobs").join("sha256").join(layer_hex);
        if blob_path.exists() && blob_is_gguf(&blob_path) {
            eprintln!("[llmman] using blob directly: {}", blob_path.display());
            return Ok(blob_path);
        }
    }

    // Otherwise extract from tar layer.
    if let Some(p) = cached_gguf(cache_path, layer_hex) {
        return Ok(p);
    }
    let blob_path = raw_blob_path(store_path, layer)?;
    // a GGUF stored whole under a tar media type
    if blob_is_gguf(&blob_path) {
        let name = Path::new(&title)
            .file_name()
            .unwrap_or_else(|| std::ffi::OsStr::new("model.gguf"));
        let dest = cache_path.join(layer_hex).join(name);
        cache_layer_file(&blob_path, &dest, layer.size)?;
        return Ok(dest);
    }
    let dir = layers::extract_tar_layer(
        store_path,
        cache_path,
        layer,
        layers::tar_compression(layer),
    )?;
    find_gguf(&dir).ok_or_else(|| anyhow!("no .gguf in tar layer {}", layer.digest))
}

/// The first `.gguf` file under `dir`, in name order.
fn find_gguf(dir: &Path) -> Option<PathBuf> {
    walkdir::WalkDir::new(dir)
        .sort_by_file_name()
        .into_iter()
        .flatten()
        .find(|e| {
            e.file_type().is_file() && e.path().extension().and_then(|x| x.to_str()) == Some("gguf")
        })
        .map(walkdir::DirEntry::into_path)
}

/// True if the file at `path` starts with the GGUF magic.
fn blob_is_gguf(path: &Path) -> bool {
    use std::io::Read as _;
    let mut magic = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .map(|()| &magic == b"GGUF")
        .unwrap_or(false)
}

/// The content-addressed blob of a layer, as stored.
fn raw_blob_path(
    store_path: &Path,
    layer: &crate::storage::oci::Descriptor,
) -> anyhow::Result<PathBuf> {
    let p = store_path
        .join("blobs")
        .join("sha256")
        .join(digest_hex(&layer.digest)?);
    if !p.exists() {
        anyhow::bail!(
            "missing blob {} for layer {:?}",
            layer.digest,
            layer_filepath(layer)
        );
    }
    Ok(p)
}

/// A raw layer's blob under its original file name as
/// `<cache>/<digest>/<filename>`: the name carries the variant
/// (`distilled` selects the sampling schedule).
fn named_blob_path(
    store_path: &Path,
    cache_path: &Path,
    layer: &crate::storage::oci::Descriptor,
) -> anyhow::Result<PathBuf> {
    let blob = raw_blob_path(store_path, layer)?;
    let Some(name) = layer_filepath(layer).and_then(|p| Path::new(p).file_name()) else {
        return Ok(blob);
    };
    let dir = cache_path.join(digest_hex(&layer.digest)?);
    let link = dir.join(name);
    cache_layer_file(&blob, &link, layer.size)?;
    Ok(link)
}

/// Ollama's `model.Capability` values reported in `/api/show`'s
/// `capabilities` array (`ollama run` reads them to set `opts.MultiModal`).
pub const CAPABILITY_COMPLETION: &str = "completion";
pub const CAPABILITY_VISION: &str = "vision";
/// Produces embeddings instead of text; such a model has no `"completion"`.
pub const CAPABILITY_EMBEDDING: &str = "embedding";
/// Answers typed questions instead of generating text (ollama's
/// `CapabilityDecision`); such a model has no `"completion"`.
pub const CAPABILITY_DECISION: &str = "decision";
/// The chat template renders a request's `tools`.
pub const CAPABILITY_TOOLS: &str = "tools";
/// The chat template handles reasoning.
pub const CAPABILITY_THINKING: &str = "thinking";
/// Generates images (a latent diffusion model) — ollama's
/// `CapabilityImage`. Such a model has no `"completion"`.
pub const CAPABILITY_IMAGE: &str = "image";
pub const CAPABILITY_VIDEO: &str = "video";
pub const CAPABILITY_AUDIO: &str = "audio";

/// The part of ollama's `Model.Capabilities()` (server/images.go) a
/// manifest alone can answer, without extracting or opening a GGUF:
/// `"completion"` always, plus `"vision"` when a companion mmproj layer
/// is present (`projectorCapabilities`) or the CNCF config's
/// `config.capabilities.inputTypes` lists `"image"` (`configCapabilities`).
pub fn capabilities(store: &OciStore, manifest: &crate::storage::oci::Manifest) -> Vec<String> {
    // Shape written by crate::hf::oci::build_cncf_manifest.
    let config: Option<serde_json::Value> = store
        .read_blob(&manifest.config.digest)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok());
    let config_types = |key: &str| -> Vec<String> {
        config
            .as_ref()
            .and_then(|v| {
                v.get("config")?
                    .get("capabilities")?
                    .get(key)?
                    .as_array()
                    .cloned()
            })
            .unwrap_or_default()
            .iter()
            .filter_map(|t| t.as_str().map(str::to_string))
            .collect()
    };

    // A diffusion model generates media instead of text: its capabilities
    // are exactly the output types it was pulled with. A Diffusers-layout
    // repo pulled before those were recorded still generates images.
    let outputs = config_types("outputTypes");
    if outputs.iter().any(|t| t == CAPABILITY_IMAGE)
        || manifest.layers.iter().any(|l| layer_role(l).is_some())
        || is_diffusers_manifest(manifest)
    {
        let mut caps: Vec<String> = outputs
            .into_iter()
            .filter(|t| t == CAPABILITY_IMAGE || t == CAPABILITY_VIDEO || t == CAPABILITY_AUDIO)
            .collect();
        if caps.is_empty() {
            caps.push(CAPABILITY_IMAGE.to_string());
        }
        return caps;
    }

    let mut caps = vec![CAPABILITY_COMPLETION.to_string()];

    let config_says_image = config_types("inputTypes").iter().any(|t| t == "image");

    if has_projector_layer(manifest) || config_says_image {
        caps.push(CAPABILITY_VISION.to_string());
    }
    caps
}

/// `caps` (from [`capabilities`]) plus what only the GGUF header and chat
/// template say, in ollama's order: `"embedding"` for a pooling model,
/// `"decision"` for a decision model or `"audio"` for a speech model, in
/// place of `"completion"`, then `"tools"` and `"thinking"` from the
/// template. A media, embedding, decision or speech model has no chat to
/// equip and keeps what it has.
pub fn model_capabilities(
    mut caps: Vec<String>,
    gguf: Option<&crate::gguf::Info>,
    template: Option<&str>,
) -> Vec<String> {
    let Some(completion) = caps.iter().position(|c| c == CAPABILITY_COMPLETION) else {
        return caps;
    };
    if gguf.is_some_and(|info| info.is_embedding()) {
        caps[completion] = CAPABILITY_EMBEDDING.to_string();
        return caps;
    }
    if gguf.is_some_and(|info| info.is_decision()) {
        caps[completion] = CAPABILITY_DECISION.to_string();
        return caps;
    }
    if gguf.is_some_and(|info| info.is_speech()) {
        caps[completion] = CAPABILITY_AUDIO.to_string();
        return caps;
    }
    let Some(template) = template else {
        return caps;
    };
    if crate::chat_template::handles_tools(template) {
        caps.push(CAPABILITY_TOOLS.to_string());
    }
    if crate::chat_template::thinking_controls(template).thinks {
        caps.push(CAPABILITY_THINKING.to_string());
    }
    caps
}

/// `caps` (from [`capabilities`]) as the projector's own header says: without
/// `"vision"` when it reads no images ([`capabilities`] takes any projector for
/// vision), and with `"audio"` when it reads audio. `None` (no readable header)
/// leaves `caps` as they are.
pub fn with_projector_caps(
    mut caps: Vec<String>,
    projector: Option<&crate::gguf::Info>,
) -> Vec<String> {
    let Some(projector) = projector.map(crate::gguf::Info::projector_caps) else {
        return caps;
    };
    if !projector.vision {
        caps.retain(|c| c != CAPABILITY_VISION);
    }
    if projector.audio && !caps.iter().any(|c| c == CAPABILITY_AUDIO) {
        let at = caps
            .iter()
            .position(|c| c == CAPABILITY_VISION)
            .or_else(|| caps.iter().position(|c| c == CAPABILITY_COMPLETION))
            .map_or(caps.len(), |i| i + 1);
        caps.insert(at, CAPABILITY_AUDIO.to_string());
    }
    caps
}

/// Whether the manifest holds a companion mmproj layer, extracted or not.
pub fn has_projector_layer(manifest: &crate::storage::oci::Manifest) -> bool {
    gguf_layers(manifest).is_some_and(|(_, mmproj)| mmproj.is_some())
}

/// How many primary GGUF layers the manifest holds. More than one is a
/// `gguf-split` set, whose tensors are spread across the shards, so no
/// single header's tensor list describes the whole model.
pub fn gguf_shard_count(manifest: &crate::storage::oci::Manifest) -> usize {
    let layers: Vec<_> = manifest
        .layers
        .iter()
        .filter(|l| is_gguf_layer(l) && layer_role(l).is_none())
        .collect();
    let weights = layers.iter().filter(|l| !is_mmproj_layer(l)).count();
    if weights > 0 {
        return weights;
    }
    // [`gguf_layers`] falls back to the first layer when every one looks
    // like mmproj, so a lone one is the model and its header describes
    // it. Several, and which is the model is a guess — stay plural, so
    // the caller reports nothing rather than one of them.
    layers.len()
}

/// The model's parsed GGUF header: the blob as stored, or a tar layer
/// already extracted. Like [`chat_template`] it extracts nothing itself,
/// so a read-only caller never copies a checkout into the cache. `None`
/// for a checkout-layout model, or a header that will not parse.
pub fn gguf_info(
    store_path: &Path,
    cache_path: &Path,
    manifest: &crate::storage::oci::Manifest,
) -> Option<crate::gguf::Info> {
    let (primary, _) = gguf_layers(manifest)?;
    let path = raw_blob_path(store_path, primary)
        .ok()
        .filter(|p| blob_is_gguf(p))
        .or_else(|| cached_gguf(cache_path, digest_hex(&primary.digest).ok()?))?;
    crate::gguf::read_info(&path).ok()
}

/// The GGUF header of the companion mmproj layer, read like [`gguf_info`].
/// `None` without a projector, for one in a tar layer not extracted yet, or
/// for a header that will not parse.
pub fn projector_info(
    store_path: &Path,
    cache_path: &Path,
    manifest: &crate::storage::oci::Manifest,
) -> Option<crate::gguf::Info> {
    let (_, projector) = gguf_layers(manifest)?;
    let projector = projector?;
    let path = raw_blob_path(store_path, projector)
        .ok()
        .filter(|p| blob_is_gguf(p))
        .or_else(|| cached_gguf(cache_path, digest_hex(&projector.digest).ok()?))?;
    crate::gguf::read_info(&path).ok()
}

/// Ollama's `api.ShowResponse.Template`: the model's chat template, read
/// without extracting anything (a read-only `/api/show` must not copy a
/// checkout into the cache). A GGUF's `tokenizer.chat_template`, from
/// the blob as stored or a tar layer already extracted; else a
/// checkout's `chat_template.jinja`, or `tokenizer_config.json`'s
/// `chat_template` (the `default` of Transformers' named templates).
pub fn chat_template(
    store: &OciStore,
    store_path: &Path,
    cache_path: &Path,
    manifest: &crate::storage::oci::Manifest,
) -> Option<String> {
    if gguf_layers(manifest).is_some() {
        return gguf_info(store_path, cache_path, manifest)?
            .str("tokenizer.chat_template")
            .map(str::to_string);
    }
    let file = |name: &str| {
        manifest
            .layers
            .iter()
            .find(|l| {
                layer_filepath(l)
                    .and_then(|p| Path::new(p).file_name())
                    .is_some_and(|f| f == name)
            })
            .and_then(|l| read_layer_text(store, l).ok())
    };
    if let Some(jinja) = file("chat_template.jinja") {
        return Some(jinja);
    }
    let config: serde_json::Value = serde_json::from_str(&file("tokenizer_config.json")?).ok()?;
    let named_default = |entry: &serde_json::Value| {
        entry.get("name").and_then(serde_json::Value::as_str) == Some("default")
    };
    match config.get("chat_template")? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(named) => named.get("default")?.as_str().map(str::to_string),
        serde_json::Value::Array(named) => named
            .iter()
            .find(|e| named_default(e))
            .or_else(|| named.first())?
            .get("template")?
            .as_str()
            .map(str::to_string),
        _ => None,
    }
}

/// A manifest layer's text: the one file of a single-file tar layer (as
/// `llmman build` writes), else the blob itself (as HuggingFace and cloud
/// pulls store docs and configs).
pub fn read_layer_text(
    store: &OciStore,
    layer: &crate::storage::oci::Descriptor,
) -> anyhow::Result<String> {
    use std::io::Read as _;
    let blob = match layers::tar_compression(layer) {
        Compression::None => store.read_blob(&layer.digest)?,
        compression => {
            // bounded: the layer does not decide how far it inflates
            let mut text = Vec::new();
            layers::open_layer(store.root(), layer, compression)?
                .take(MAX_COMPRESSED_TEXT)
                .read_to_end(&mut text)?;
            text
        }
    };
    if blob.len() >= 512 {
        let mut archive = tar::Archive::new(std::io::Cursor::new(&blob));
        if let Ok(entries) = archive.entries() {
            for mut entry in entries.flatten() {
                let mut s = String::new();
                if entry.read_to_string(&mut s).is_ok() && !s.is_empty() {
                    return Ok(s);
                }
            }
        }
    }
    Ok(String::from_utf8_lossy(&blob).into_owned())
}

/// The most [`read_layer_text`] reads out of a compressed layer.
const MAX_COMPRESSED_TEXT: u64 = 64 << 20;

/// The GGUF a tar layer was already extracted to, if any.
fn cached_gguf(cache_path: &Path, layer_hex: &str) -> Option<PathBuf> {
    let layer_dir = cache_path.join(layer_hex);
    std::fs::read_dir(&layer_dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().and_then(|e| e.to_str()) == Some("gguf"))
        .or_else(|| find_gguf(&layers::extracted_files(&layer_dir)?))
}

/// The manifest's primary GGUF layer and its companion mmproj layer, if
/// any. Prefers a non-mmproj-named layer as primary so an mmproj file
/// that sorts first isn't taken for the model; falls back to the first
/// layer if every one looks like mmproj.
fn gguf_layers(
    manifest: &crate::storage::oci::Manifest,
) -> Option<(
    &crate::storage::oci::Descriptor,
    Option<&crate::storage::oci::Descriptor>,
)> {
    let layers: Vec<&crate::storage::oci::Descriptor> = manifest
        .layers
        .iter()
        .filter(|l| is_gguf_layer(l) && layer_role(l).is_none())
        .collect();
    let primary = *layers
        .iter()
        .find(|l| !is_mmproj_layer(l))
        .or(layers.first())?;
    let mmproj = layers
        .iter()
        .copied()
        .find(|l| is_mmproj_layer(l) && l.digest != primary.digest);
    Some((primary, mmproj))
}

/// Which [`ModelPath`] variant a manifest resolves to — see [`stored_format`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    Gguf,
    SafeTensors,
    Diffusion,
    /// Diffusers-layout safetensors — see [`ModelPath::Omni`].
    Omni,
}

impl ModelFormat {
    /// The same vocabulary [`ModelPath::format`] reports.
    pub fn as_str(self) -> &'static str {
        match self {
            ModelFormat::Gguf => "gguf",
            ModelFormat::SafeTensors => "safetensors",
            ModelFormat::Diffusion => "diffusion",
            ModelFormat::Omni => "omni",
        }
    }
}

/// [`resolve_model`]'s classification: diffusion > GGUF > Diffusers
/// (omni) > safetensors, `None` for "no servable model layer".
pub fn manifest_format(manifest: &crate::storage::oci::Manifest) -> Option<ModelFormat> {
    if manifest.layers.iter().any(|l| layer_role(l).is_some()) {
        Some(ModelFormat::Diffusion)
    } else if gguf_layers(manifest).is_some() {
        Some(ModelFormat::Gguf)
    } else if is_diffusers_manifest(manifest) {
        Some(ModelFormat::Omni)
    } else if manifest.layers.iter().any(is_safetensors_layer) {
        Some(ModelFormat::SafeTensors)
    } else {
        None
    }
}

/// A Diffusers pack whose pipeline `crate::mediagen` runs itself.
fn is_native_diffusers(store: &OciStore, manifest: &crate::storage::oci::Manifest) -> bool {
    manifest
        .layers
        .iter()
        .find(|l| is_diffusers_index_layer(l))
        .and_then(|l| read_layer_text(store, l).ok())
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| {
            v.get("_class_name")?
                .as_str()
                .map(crate::mediagen::is_native_pipeline)
        })
        .unwrap_or(false)
}

/// [`manifest_format`], but a Diffusers pack whose pipeline `crate::mediagen`
/// runs is [`ModelFormat::Diffusion`] rather than [`ModelFormat::Omni`].
pub fn stored_manifest_format(
    store: &OciStore,
    manifest: &crate::storage::oci::Manifest,
) -> Option<ModelFormat> {
    match manifest_format(manifest) {
        Some(ModelFormat::Omni) if is_native_diffusers(store, manifest) => {
            Some(ModelFormat::Diffusion)
        }
        f => f,
    }
}

/// The "nothing servable" error, naming the file extensions that were there.
fn no_servable_layer(model_ref: &str, manifest: &crate::storage::oci::Manifest) -> anyhow::Error {
    let exts: std::collections::HashSet<String> = manifest
        .layers
        .iter()
        .filter_map(|l| layer_filepath(l))
        .filter_map(|p| Path::new(p).extension()?.to_str().map(|e| e.to_lowercase()))
        .collect();
    if exts.is_empty() {
        anyhow!("no servable model layer found in {model_ref}")
    } else {
        anyhow!(
            "no servable model layer in {model_ref} — found {exts:?} files; \
             llmman serve supports GGUF (llama-server/ds4) and safetensors (vllm/vllm-omni/sglang/mlx)"
        )
    }
}

/// What [`resolve_model`] would resolve `model_ref` to, read off its
/// manifest without extracting anything (for `serve --pull-only`, which only
/// needs to know which engine's image to pull).
pub fn stored_format(store_path: &Path, model_ref: &str) -> anyhow::Result<ModelFormat> {
    let store = OciStore::open(store_path)?;
    let desc = store
        .find(model_ref)
        .with_context(|| format!("model not found in store: {model_ref}"))?;
    let manifest = store.read_manifest(&desc.digest)?;
    stored_manifest_format(&store, &manifest).ok_or_else(|| no_servable_layer(model_ref, &manifest))
}

/// Resolve `model_ref` (already present in the `OciStore` at `store_path`)
/// to either a `.gguf` file or an extracted safetensors directory, caching
/// any extraction under `cache_path`.
pub fn resolve_model(
    store_path: &Path,
    cache_path: &Path,
    model_ref: &str,
) -> anyhow::Result<ModelPath> {
    let store = OciStore::open(store_path)?;
    let desc = store
        .find(model_ref)
        .with_context(|| format!("model not found in store: {model_ref}"))?;
    let manifest = store.read_manifest(&desc.digest)?;

    let Some(format) = stored_manifest_format(&store, &manifest) else {
        return Err(no_servable_layer(model_ref, &manifest));
    };

    // ── Diffusers pipeline → crate::mediagen ──────────────────────────────
    if format == ModelFormat::Diffusion && manifest_format(&manifest) == Some(ModelFormat::Omni) {
        let (root, placed) = extract_model_tree(
            store_path,
            cache_path,
            &desc.digest,
            &manifest,
            TreeScope::Servable,
        )?;
        let files: std::collections::BTreeMap<String, PathBuf> = placed
            .into_iter()
            .map(|p| {
                let path = root.join(&p);
                (p, path)
            })
            .collect();
        let model = files
            .get(DIFFUSERS_MODEL_INDEX)
            .cloned()
            .ok_or_else(|| anyhow!("{model_ref}: {DIFFUSERS_MODEL_INDEX} was not extracted"))?;
        return Ok(ModelPath::Diffusion(DiffusionPaths {
            model,
            files,
            ..Default::default()
        }));
    }

    // ── diffusion → crate::mediagen ───────────────────────────────────────
    if format == ModelFormat::Diffusion {
        let (primary, _) = gguf_layers(&manifest)
            .ok_or_else(|| anyhow!("{model_ref}: diffusion model has no transformer GGUF layer"))?;
        // Every file keeps its name (see named_blob_path); a GGUF in a tar
        // layer is extracted.
        let named = |l: &crate::storage::oci::Descriptor| -> anyhow::Result<PathBuf> {
            let raw = raw_blob_path(store_path, l);
            if is_gguf_layer(l) && !raw.as_ref().is_ok_and(|p| blob_is_gguf(p)) {
                extract_gguf_layer(store_path, cache_path, l)
            } else if raw.is_ok()
                && (is_gguf_layer(l) || l.media_type == crate::hf::oci::MEDIA_TYPE_MODEL_WEIGHT_RAW)
            {
                named_blob_path(store_path, cache_path, l)
            } else {
                anyhow::bail!(
                    "{model_ref}: unsupported layer {} ({})",
                    l.digest,
                    l.media_type
                )
            }
        };
        let mut paths = DiffusionPaths {
            model: named(primary)?,
            ..Default::default()
        };
        for l in &manifest.layers {
            // raw layers only: a tar layer's blob is the archive, not the file
            if let (Some(fp), true) = (layer_filepath(l), l.media_type.ends_with(".raw")) {
                paths
                    .files
                    .insert(fp.to_string(), named_blob_path(store_path, cache_path, l)?);
            }
            let Some(role) = layer_role(l) else { continue };
            let p = named(l)?;
            match role {
                "vae" => paths.vae = Some(p),
                "audio_vae" => paths.audio_vae = Some(p),
                "text_proj" => paths.text_proj = Some(p),
                "text_encoder" => paths.text_encoder = Some(p),
                // Cosmos3 sidecars its generation path does not use
                "vision_encoder" | "sound_tokenizer" => {}
                other => {
                    eprintln!("[llmman] {model_ref}: ignoring layer with unknown role {other:?}")
                }
            }
        }
        return Ok(ModelPath::Diffusion(paths));
    }

    // ── GGUF → llama-server ────────────────────────────────────────────────
    if let Some((primary, mmproj)) = gguf_layers(&manifest) {
        let primary_path = extract_gguf_layer(store_path, cache_path, primary)?;
        let mmproj_path = mmproj
            .map(|l| extract_gguf_layer(store_path, cache_path, l))
            .transpose()
            .context("extracting companion mmproj file")?;
        if mmproj_path.is_some() {
            eprintln!("[llmman] {model_ref}: found companion mmproj file");
        }
        return Ok(ModelPath::Gguf(primary_path, mmproj_path));
    }

    // ── safetensors → vllm / vllm-omni / mlx_lm.server ──────────────────
    let model_dir = extract_safetensors_dir(store_path, cache_path, &desc.digest, &manifest)?;
    Ok(match format {
        ModelFormat::Omni => ModelPath::Omni(model_dir),
        _ => {
            debug_assert_eq!(format, ModelFormat::SafeTensors);
            ModelPath::SafeTensors(model_dir)
        }
    })
}

/// The model directory of an extracted checkout: the parent of the
/// shallowest `config.json` or `model_index.json` (a Diffusers repo has a
/// `config.json` per component, so "first in layer order" is wrong), else
/// the checkout root.
fn safetensors_model_dir(cache_dir: &Path, rel_paths: &[&str]) -> PathBuf {
    rel_paths
        .iter()
        .filter(|p| {
            Path::new(p)
                .file_name()
                .is_some_and(|n| n == "config.json" || n == DIFFUSERS_MODEL_INDEX)
        })
        .min_by_key(|p| p.matches('/').count())
        .and_then(|p| Path::new(p).parent())
        // `join("")` would leave a trailing separator on the root case.
        .filter(|parent| !parent.as_os_str().is_empty())
        .map(|parent| cache_dir.join(parent))
        .unwrap_or_else(|| cache_dir.to_path_buf())
}

/// Which of a manifest's layers [`extract_model_tree`] places.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeScope {
    /// Weights and their config files: what serving needs.
    Servable,
    /// Every layer, docs, code and datasets too.
    Full,
}

/// Places `manifest`'s ModelPack layers under `<cache>/<manifest hex>/` and
/// returns that directory with the paths placed, relative to it. A raw layer
/// goes to its `org.cncf.model.filepath`, a tar layer to the paths its entries
/// name; both as hardlinks. Two layers placing one path is an error.
/// Idempotent, and safe to call concurrently.
pub fn extract_model_tree(
    store_path: &Path,
    cache_path: &Path,
    manifest_digest: &str,
    manifest: &crate::storage::oci::Manifest,
    scope: TreeScope,
) -> anyhow::Result<(PathBuf, Vec<String>)> {
    let root = cache_path.join(digest_hex(manifest_digest)?);
    let mut placed = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for layer in &manifest.layers {
        let Some((servable, encoding)) = layers::parse_media_type(&layer.media_type) else {
            continue;
        };
        if scope == TreeScope::Servable && !servable {
            continue;
        }
        let paths = match encoding {
            Encoding::Tar(_) => layers::place_tar_layer(store_path, cache_path, layer, &root)?,
            Encoding::Raw => {
                let Some(rel_path) = layer_filepath(layer) else {
                    continue;
                };
                // The annotation is untrusted: "../../id_rsa" would escape the root.
                if !crate::sources::is_safe_relative_path(rel_path) {
                    eprintln!("[llmman] skipping layer with unsafe filepath {rel_path:?}");
                    continue;
                }
                let dest = root.join(rel_path);
                if !cached_layer_file_matches(&dest, layer.size) {
                    let blob = raw_blob_path(store_path, layer)?;
                    cache_layer_file(&blob, &dest, layer.size)
                        .with_context(|| format!("cache {rel_path} from blob store"))?;
                    eprintln!("[llmman] cached {rel_path}");
                }
                vec![rel_path.to_owned()]
            }
        };
        for path in paths {
            // Which layer wins would otherwise depend on the files' sizes.
            anyhow::ensure!(seen.insert(path.clone()), "two layers place {path}");
            placed.push(path);
        }
    }
    Ok((root, placed))
}

/// Extract CNCF-format safetensors layers to a cache directory and return the
/// model directory (see [`safetensors_model_dir`]).
fn extract_safetensors_dir(
    store_path: &Path,
    cache_path: &Path,
    manifest_digest: &str,
    manifest: &crate::storage::oci::Manifest,
) -> anyhow::Result<PathBuf> {
    let (cache_dir, placed) = extract_model_tree(
        store_path,
        cache_path,
        manifest_digest,
        manifest,
        TreeScope::Servable,
    )?;
    let rel_paths: Vec<&str> = placed.iter().map(String::as_str).collect();
    Ok(safetensors_model_dir(&cache_dir, &rel_paths))
}

/// A model's complete directory, as [`resolve_model_tree`] returns it.
#[derive(Debug)]
pub struct ModelTree {
    /// `<cache>/<manifest hex>`: every layer at its path in the artifact.
    pub root: PathBuf,
    /// The manifest the tree was built from.
    pub digest: String,
    /// What llmman would serve it as; `None` if it cannot (an ONNX export).
    pub format: Option<ModelFormat>,
}

/// Lays out every layer of `model_ref` (already in the store) as the model
/// directory under `cache_path` and returns it. See [`extract_model_tree`].
pub fn resolve_model_tree(
    store_path: &Path,
    cache_path: &Path,
    model_ref: &str,
) -> anyhow::Result<ModelTree> {
    let store = OciStore::open(store_path)?;
    let desc = store
        .find(model_ref)
        .with_context(|| format!("model not found in store: {model_ref}"))?;
    let manifest = store.read_manifest(&desc.digest)?;
    let (root, placed) = extract_model_tree(
        store_path,
        cache_path,
        &desc.digest,
        &manifest,
        TreeScope::Full,
    )?;
    anyhow::ensure!(
        !placed.is_empty(),
        "{model_ref} has no ModelPack file layers to extract"
    );
    let format = stored_manifest_format(&store, &manifest);
    Ok(ModelTree {
        root,
        digest: desc.digest,
        format,
    })
}

fn cached_layer_file_matches(dest: &Path, layer_size: u64) -> bool {
    dest.symlink_metadata()
        .map(|m| m.is_file() && m.len() == layer_size)
        .unwrap_or(false)
}

fn cache_layer_file(src: &Path, dest: &Path, layer_size: u64) -> anyhow::Result<()> {
    if cached_layer_file_matches(dest, layer_size) {
        return Ok(());
    }
    std::fs::create_dir_all(dest.parent().context("no parent")?)?;
    link_or_copy_file(src, dest, layer_size)
}

/// Puts `src` at `dest`, replacing whatever is there: a hardlink, or a copy
/// where the filesystem cannot link. Renamed into place from a temporary
/// name, so a concurrent caller never finds `dest` missing or half done.
fn link_or_copy_file(src: &Path, dest: &Path, layer_size: u64) -> anyhow::Result<()> {
    let tmp = cache_copy_temp_path(dest);
    match std::fs::hard_link(src, &tmp) {
        Ok(()) => {
            let result = (|| {
                let actual_size = tmp
                    .symlink_metadata()
                    .with_context(|| format!("inspect hardlink {}", tmp.display()))?
                    .len();
                if actual_size != layer_size {
                    anyhow::bail!(
                        "blob {} is {actual_size} bytes, expected {layer_size}",
                        src.display()
                    );
                }
                std::fs::rename(&tmp, dest)
                    .with_context(|| format!("rename {} to {}", tmp.display(), dest.display()))
            })();
            // `rename` between two links of one file leaves `tmp` behind
            let _ = std::fs::remove_file(&tmp);
            result
        }
        Err(hardlink_error) => copy_file_atomic(src, dest, layer_size).with_context(|| {
            format!(
                "hardlink {} to {} failed: {hardlink_error}; copy failed",
                src.display(),
                dest.display()
            )
        }),
    }
}

fn copy_file_atomic(src: &Path, dest: &Path, layer_size: u64) -> anyhow::Result<()> {
    let tmp = cache_copy_temp_path(dest);
    let result = (|| {
        let copied = std::fs::copy(src, &tmp)
            .with_context(|| format!("copy {} to {}", src.display(), tmp.display()))?;
        if copied != layer_size {
            anyhow::bail!("copied {copied} bytes, expected {layer_size}");
        }
        match std::fs::rename(&tmp, dest) {
            Ok(()) => Ok(()),
            Err(_) if cached_layer_file_matches(dest, layer_size) => Ok(()),
            Err(e) => {
                Err(e).with_context(|| format!("rename {} to {}", tmp.display(), dest.display()))
            }
        }
    })();
    if result.is_err() || tmp.exists() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

fn cache_copy_temp_path(dest: &Path) -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let mut tmp = dest.to_path_buf().into_os_string();
    tmp.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    PathBuf::from(tmp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_matches_variant() {
        assert_eq!(
            ModelPath::Gguf(PathBuf::from("/x/m.gguf"), None).format(),
            "gguf"
        );
        assert_eq!(
            ModelPath::SafeTensors(PathBuf::from("/x")).format(),
            "safetensors"
        );
    }

    #[test]
    fn path_returns_inner_pathbuf() {
        let p = ModelPath::SafeTensors(PathBuf::from("/models/foo"));
        assert_eq!(p.path(), Path::new("/models/foo"));
    }

    #[test]
    fn mmproj_is_none_unless_explicitly_set() {
        let no_mmproj = ModelPath::Gguf(PathBuf::from("/x/m.gguf"), None);
        assert_eq!(no_mmproj.mmproj(), None);

        let with_mmproj = ModelPath::Gguf(
            PathBuf::from("/x/m.gguf"),
            Some(PathBuf::from("/x/mmproj-f16.gguf")),
        );
        assert_eq!(with_mmproj.mmproj(), Some(Path::new("/x/mmproj-f16.gguf")));

        // SafeTensors (vllm/mlx) has no equivalent separate-projector-file
        // convention.
        assert_eq!(ModelPath::SafeTensors(PathBuf::from("/x")).mmproj(), None);
    }

    fn descriptor(digest: &str, filepath: &str) -> crate::storage::oci::Descriptor {
        let mut ann = std::collections::HashMap::new();
        ann.insert("org.cncf.model.filepath".to_string(), filepath.to_string());
        crate::storage::oci::Descriptor {
            media_type: "application/vnd.cncf.model.weight.v1.tar".into(),
            digest: digest.to_string(),
            size: 123,
            annotations: Some(ann),
        }
    }

    #[test]
    fn is_mmproj_layer_matches_filename_regardless_of_case_or_position() {
        assert!(is_mmproj_layer(&descriptor("sha256:a", "mmproj-F16.gguf")));
        assert!(is_mmproj_layer(&descriptor(
            "sha256:b",
            "Qwen3-VL-mmproj.gguf"
        )));
        assert!(!is_mmproj_layer(&descriptor(
            "sha256:c",
            "model.Q4_K_M.gguf"
        )));
    }

    fn manifest_with(
        layers: Vec<crate::storage::oci::Descriptor>,
    ) -> (OciStore, crate::storage::oci::Manifest) {
        // parallel tests can share a clock tick; the counter keeps their stores apart
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "llmman-modelpack-caps-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let store = OciStore::open(&dir).unwrap();
        let config = store
            .write_blob("application/vnd.cncf.model.config.v1+json", b"{}")
            .unwrap();
        let manifest = crate::storage::oci::Manifest {
            schema_version: 2,
            media_type: "application/vnd.oci.image.manifest.v1+json".into(),
            artifact_type: None,
            config,
            layers,
            annotations: None,
        };
        (store, manifest)
    }

    #[test]
    fn capabilities_is_completion_only_without_a_projector() {
        let (store, m) = manifest_with(vec![descriptor("sha256:a", "model.Q4_K_M.gguf")]);
        assert_eq!(capabilities(&store, &m), vec!["completion"]);
    }

    /// A GGUF header, with `pooling_type` for an embedding model.
    fn gguf_header(pooling: bool) -> crate::gguf::Info {
        use crate::gguf::Value;
        let mut info = crate::gguf::Info::default();
        info.metadata.insert(
            "general.architecture".into(),
            Value::String("gemma3".into()),
        );
        if pooling {
            info.metadata
                .insert("gemma3.pooling_type".into(), Value::U32(1));
        }
        info
    }

    const TOOLS_AND_THINKING: &str =
        "{%- if tools %}{{ tools | tojson }}{%- endif %}{%- if enable_thinking %}<think>{%- endif %}";

    /// `tools` and `thinking` follow `vision`, as in ollama.
    #[test]
    fn model_capabilities_adds_what_the_template_handles() {
        let caps = |template: Option<&str>| {
            model_capabilities(
                vec!["completion".into(), "vision".into()],
                Some(&gguf_header(false)),
                template,
            )
        };
        assert_eq!(
            caps(Some(TOOLS_AND_THINKING)),
            ["completion", "vision", "tools", "thinking"]
        );
        assert_eq!(
            caps(Some("{%- if tools %}{{ tools }}{%- endif %}")),
            ["completion", "vision", "tools"]
        );
        assert_eq!(
            caps(Some("{%- if enable_thinking %}<think>{%- endif %}")),
            ["completion", "vision", "thinking"]
        );
        assert_eq!(
            caps(Some("{{ messages[0].content }}")),
            ["completion", "vision"]
        );
        // No template: what the manifest said.
        assert_eq!(caps(None), ["completion", "vision"]);
        // No GGUF header (a checkout): the template still counts.
        assert_eq!(
            model_capabilities(vec!["completion".into()], None, Some(TOOLS_AND_THINKING)),
            ["completion", "tools", "thinking"]
        );
    }

    /// A pooling model embeds instead of completing, template or not.
    #[test]
    fn model_capabilities_reports_a_pooling_model_as_embedding() {
        assert_eq!(
            model_capabilities(
                vec!["completion".into()],
                Some(&gguf_header(true)),
                Some(TOOLS_AND_THINKING)
            ),
            ["embedding"]
        );
        assert!(gguf_header(true).is_embedding());
        assert!(!gguf_header(false).is_embedding());
    }

    /// A decision model keeps its projector's vision.
    #[test]
    fn model_capabilities_reports_a_decision_model_as_decision() {
        use crate::gguf::Value;
        let mut header = gguf_header(false);
        header
            .metadata
            .insert("gemma3.decision.type".into(), Value::String("clef".into()));
        assert_eq!(
            model_capabilities(
                vec!["completion".into(), "vision".into()],
                Some(&header),
                Some(TOOLS_AND_THINKING)
            ),
            ["decision", "vision"]
        );
        assert!(header.is_decision());
        assert!(!gguf_header(false).is_decision());
    }

    /// A speech model transcribes: `audio` in place of `completion`, so
    /// listings agree with the 400 the daemon gives a chat request.
    #[test]
    fn model_capabilities_reports_a_speech_model_as_audio() {
        use crate::gguf::Value;
        let mut header = crate::gguf::Info::default();
        header.metadata.insert(
            "general.architecture".into(),
            Value::String("canary".into()),
        );
        assert_eq!(
            model_capabilities(
                vec!["completion".into()],
                Some(&header),
                Some(TOOLS_AND_THINKING)
            ),
            ["audio"]
        );
        assert!(header.is_speech());
        assert!(!gguf_header(false).is_speech());
    }

    /// A media model's capabilities are what it generates.
    #[test]
    fn model_capabilities_leaves_a_media_model_alone() {
        let media = vec!["image".to_string(), "video".to_string()];
        assert_eq!(
            model_capabilities(media.clone(), None, Some(TOOLS_AND_THINKING)),
            media
        );
    }

    /// A projector's header with the given `clip.has_<flag>_encoder` flags.
    fn projector_header(flags: &[&str]) -> crate::gguf::Info {
        let mut info = crate::gguf::Info::default();
        for flag in flags {
            info.metadata.insert(
                format!("clip.has_{flag}_encoder"),
                crate::gguf::Value::Bool(true),
            );
        }
        info
    }

    /// A speech-only projector (Soprano) is not vision; LFM3-Omni's is; one that
    /// only hears is audio. An unreadable header changes nothing.
    #[test]
    fn with_projector_caps_follows_the_projectors_own_header() {
        let caps = || vec!["completion".to_string(), "vision".to_string()];
        assert_eq!(
            with_projector_caps(caps(), Some(&projector_header(&["gen_audio"]))),
            ["completion"]
        );
        assert_eq!(
            with_projector_caps(caps(), Some(&projector_header(&["audio"]))),
            ["completion", "audio"]
        );
        assert_eq!(
            with_projector_caps(
                caps(),
                Some(&projector_header(&["vision", "audio", "gen_audio"]))
            ),
            ["completion", "vision", "audio"]
        );
        assert_eq!(with_projector_caps(caps(), None), ["completion", "vision"]);
        // Nothing to take away from a model that never had it.
        assert_eq!(
            with_projector_caps(vec!["completion".into()], Some(&projector_header(&[]))),
            ["completion"]
        );
    }

    #[test]
    fn capabilities_adds_vision_when_a_companion_mmproj_layer_is_present() {
        // Mirrors ollama's projectorCapabilities: any projector ⇒ vision.
        let (store, m) = manifest_with(vec![
            descriptor("sha256:a", "mmproj-F16.gguf"),
            descriptor("sha256:b", "model.Q4_K_M.gguf"),
        ]);
        assert_eq!(capabilities(&store, &m), vec!["completion", "vision"]);
        assert!(has_projector_layer(&m));
    }

    #[test]
    fn capabilities_a_lone_mmproj_named_gguf_is_the_model_not_a_projector() {
        let (store, m) = manifest_with(vec![descriptor("sha256:a", "mmproj-only.gguf")]);
        assert_eq!(capabilities(&store, &m), vec!["completion"]);
        assert!(!has_projector_layer(&m));
    }

    #[test]
    fn capabilities_two_mmproj_named_ggufs_load_with_a_projector_so_report_vision() {
        // resolve_model's fallback: first is primary, second is its mmproj.
        let (store, m) = manifest_with(vec![
            descriptor("sha256:a", "mmproj-model.gguf"),
            descriptor("sha256:b", "mmproj-F16.gguf"),
        ]);
        assert_eq!(capabilities(&store, &m), vec!["completion", "vision"]);
    }

    #[test]
    fn gguf_info_reads_the_header_of_the_stored_primary_layer() {
        let path = crate::gguf::write_test_gguf_with(&[]);
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let (store, mut m) = manifest_with(vec![]);
        let mut d = store.write_blob(HF_GGUF_MEDIA_TYPE, &bytes).unwrap();
        d.annotations.get_or_insert_with(Default::default).insert(
            "org.cncf.model.filepath".to_string(),
            "model.gguf".to_string(),
        );
        m.layers = vec![d];
        let info = gguf_info(store.root(), Path::new("/nonexistent"), &m).expect("header");
        assert_eq!(info.architecture(), Some("llama"));
        assert_eq!(info.context_length(), Some(4096));
    }

    /// A checkout-layout model has no GGUF layer to read.
    #[test]
    fn gguf_info_is_none_without_a_gguf_layer() {
        let (store, m) = manifest_with(vec![descriptor("sha256:a", "model.safetensors")]);
        assert!(gguf_info(store.root(), Path::new("/nonexistent"), &m).is_none());
    }

    #[test]
    fn capabilities_honours_cncf_config_input_types_image() {
        // The exact shape hf::oci::build_cncf_manifest writes.
        let (store, mut m) = manifest_with(vec![descriptor("sha256:a", "model.gguf")]);
        m.config = store
            .write_blob(
                "application/vnd.cncf.model.config.v1+json",
                br#"{"descriptor":{},"modelfs":{"type":"layers","diffIds":[]},"config":{"format":"gguf","capabilities":{"inputTypes":["text","image"],"outputTypes":["text"]}}}"#,
            )
            .unwrap();
        assert_eq!(capabilities(&store, &m), vec!["completion", "vision"]);
    }

    #[test]
    fn capabilities_of_a_diffusion_model_are_its_output_types() {
        // A diffusion pull: the transformer GGUF plus role-annotated
        // sidecars, config listing what it generates. No "completion".
        let mut vae = descriptor("sha256:b", "ltx_video_vae.safetensors");
        vae.annotations.get_or_insert_with(Default::default).insert(
            crate::hf::oci::ANNOTATION_ROLE.to_string(),
            "vae".to_string(),
        );
        let mut te = descriptor("sha256:c", "gemma-3-12b-it-Q4_K_M.gguf");
        te.annotations.get_or_insert_with(Default::default).insert(
            crate::hf::oci::ANNOTATION_ROLE.to_string(),
            "text_encoder".to_string(),
        );
        let (store, mut m) =
            manifest_with(vec![descriptor("sha256:a", "ltx-Q4_K_M.gguf"), vae, te]);
        m.config = store
            .write_blob(
                "application/vnd.cncf.model.config.v1+json",
                br#"{"config":{"format":"gguf","capabilities":{"inputTypes":["text"],"outputTypes":["image","video","audio"]}}}"#,
            )
            .unwrap();
        assert_eq!(capabilities(&store, &m), vec!["image", "video", "audio"]);
        // the text encoder GGUF is a sidecar, not the model
        let (primary, mmproj) = gguf_layers(&m).unwrap();
        assert_eq!(primary.digest, "sha256:a");
        assert!(mmproj.is_none());
    }

    #[test]
    fn capabilities_ignores_input_types_at_the_document_root() {
        let (store, mut m) = manifest_with(vec![descriptor("sha256:a", "model.gguf")]);
        m.config = store
            .write_blob(
                "application/vnd.cncf.model.config.v1+json",
                br#"{"capabilities":{"inputTypes":["text","image"]}}"#,
            )
            .unwrap();
        assert_eq!(capabilities(&store, &m), vec!["completion"]);
    }

    #[test]
    fn manifest_format_follows_resolve_models_precedence() {
        let (_, m) = manifest_with(vec![
            descriptor("sha256:a", "model.Q4_K_M.gguf"),
            descriptor("sha256:b", "extra.safetensors"),
        ]);
        assert_eq!(manifest_format(&m), Some(ModelFormat::Gguf));
        let (_, m) = manifest_with(vec![
            descriptor("sha256:a", "config.json"),
            descriptor("sha256:b", "model-00001-of-00002.safetensors"),
        ]);
        assert_eq!(manifest_format(&m), Some(ModelFormat::SafeTensors));
        let mut vae = descriptor("sha256:b", "vae.safetensors");
        vae.annotations.get_or_insert_with(Default::default).insert(
            crate::hf::oci::ANNOTATION_ROLE.to_string(),
            "vae".to_string(),
        );
        let (_, m) = manifest_with(vec![descriptor("sha256:a", "ltx.gguf"), vae]);
        assert_eq!(manifest_format(&m), Some(ModelFormat::Diffusion));
        let (_, m) = manifest_with(vec![descriptor("sha256:a", "README.md")]);
        assert_eq!(manifest_format(&m), None);
    }

    #[test]
    fn gguf_shard_count_counts_only_the_primary_weights() {
        let (_, m) = manifest_with(vec![
            descriptor("sha256:a", "model-00001-of-00003.gguf"),
            descriptor("sha256:b", "model-00002-of-00003.gguf"),
            descriptor("sha256:c", "model-00003-of-00003.gguf"),
            descriptor("sha256:d", "mmproj-F16.gguf"),
        ]);
        assert_eq!(gguf_shard_count(&m), 3);
        let (_, m) = manifest_with(vec![
            descriptor("sha256:a", "model.Q4_K_M.gguf"),
            descriptor("sha256:b", "mmproj-F16.gguf"),
        ]);
        assert_eq!(gguf_shard_count(&m), 1);
        // A lone mmproj-named GGUF is the model here (see gguf_layers).
        let (_, m) = manifest_with(vec![descriptor("sha256:a", "mmproj-F16.gguf")]);
        assert_eq!(gguf_shard_count(&m), 1);
        // Several, and which one is the model is a guess.
        let (_, m) = manifest_with(vec![
            descriptor("sha256:a", "mmproj-F16.gguf"),
            descriptor("sha256:b", "mmproj-Q8.gguf"),
        ]);
        assert_eq!(gguf_shard_count(&m), 2);
    }

    /// `/api/show` reports these strings, so they are the wire contract.
    #[test]
    fn model_format_as_str_matches_model_paths_vocabulary() {
        assert_eq!(ModelFormat::Gguf.as_str(), "gguf");
        assert_eq!(ModelFormat::SafeTensors.as_str(), "safetensors");
        assert_eq!(ModelFormat::Diffusion.as_str(), "diffusion");
        assert_eq!(ModelFormat::Omni.as_str(), "omni");
    }

    /// The layers `llmman pull nvidia/Cosmos3-Edge` records (the
    /// Diffusers layout: a root pipeline index, per-component subdirs).
    fn cosmos3_layers() -> Vec<crate::storage::oci::Descriptor> {
        vec![
            descriptor("sha256:a", "config.json"),
            descriptor("sha256:b", "model_index.json"),
            descriptor("sha256:c", "scheduler/scheduler_config.json"),
            descriptor("sha256:d", "transformer/config.json"),
            descriptor(
                "sha256:e",
                "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            ),
            descriptor("sha256:f", "vae/config.json"),
            descriptor("sha256:g", "vae/diffusion_pytorch_model.safetensors"),
            descriptor("sha256:h", "vision_encoder/model.safetensors"),
        ]
    }

    #[test]
    fn a_root_model_index_makes_a_safetensors_repo_omni() {
        let (_, m) = manifest_with(cosmos3_layers());
        assert_eq!(manifest_format(&m), Some(ModelFormat::Omni));
        // a GGUF transformer with role-annotated sidecars still wins
        let mut layers = cosmos3_layers();
        let mut vae = descriptor("sha256:z", "vae.safetensors");
        vae.annotations.get_or_insert_with(Default::default).insert(
            crate::hf::oci::ANNOTATION_ROLE.to_string(),
            "vae".to_string(),
        );
        layers.push(descriptor("sha256:y", "ltx.gguf"));
        layers.push(vae);
        let (_, m) = manifest_with(layers);
        assert_eq!(manifest_format(&m), Some(ModelFormat::Diffusion));
    }

    #[test]
    fn a_pipeline_mediagen_runs_is_diffusion_not_omni() {
        let with_index = |class: &str| {
            let (store, mut m) = manifest_with(cosmos3_layers());
            let index = store
                .write_blob(
                    "application/vnd.cncf.model.weight.config.v1.raw",
                    format!(r#"{{"_class_name": "{class}"}}"#).as_bytes(),
                )
                .unwrap();
            m.layers[1].digest = index.digest;
            stored_manifest_format(&store, &m)
        };
        assert_eq!(
            with_index("QwenImage21Pipeline"),
            Some(ModelFormat::Diffusion)
        );
        assert_eq!(with_index("Cosmos3OmniPipeline"), Some(ModelFormat::Omni));
        // the index in a tar layer is read the same, compressed or not
        for suffix in ["tar", "tar+gzip", "tar+zstd"] {
            let (store, mut m) = manifest_with(cosmos3_layers());
            let index = br#"{"_class_name": "QwenImage21Pipeline"}"#;
            m.layers[1] = tar_layer(
                &store,
                "weight.config",
                suffix,
                &[("model_index.json", index)],
            );
            assert_eq!(
                stored_manifest_format(&store, &m),
                Some(ModelFormat::Diffusion),
                "{suffix}"
            );
        }
        // an index missing from the store leaves it to vLLM-Omni
        let (store, m) = manifest_with(cosmos3_layers());
        assert_eq!(stored_manifest_format(&store, &m), Some(ModelFormat::Omni));
    }

    #[test]
    fn a_nested_model_index_or_one_without_weights_is_not_omni() {
        // A pipeline vendored inside an LLM repo does not make it one.
        let (_, m) = manifest_with(vec![
            descriptor("sha256:a", "config.json"),
            descriptor("sha256:b", "model.safetensors"),
            descriptor("sha256:c", "examples/pipeline/model_index.json"),
        ]);
        assert_eq!(manifest_format(&m), Some(ModelFormat::SafeTensors));
        // An index with nothing to serve is still nothing to serve.
        let (_, m) = manifest_with(vec![descriptor("sha256:a", "model_index.json")]);
        assert_eq!(manifest_format(&m), None);
    }

    #[test]
    fn capabilities_of_an_omni_model_are_media_even_without_recorded_outputs() {
        // Pulled before the pull recorded outputTypes: still not a chat model.
        let (store, m) = manifest_with(cosmos3_layers());
        assert_eq!(capabilities(&store, &m), vec!["image"]);
        let (store, mut m) = manifest_with(cosmos3_layers());
        m.config = store
            .write_blob(
                "application/vnd.cncf.model.config.v1+json",
                br#"{"config":{"format":"safetensors","capabilities":{"inputTypes":["text"],"outputTypes":["image","video"]}}}"#,
            )
            .unwrap();
        assert_eq!(capabilities(&store, &m), vec!["image", "video"]);
    }

    #[test]
    fn model_dir_is_the_parent_of_the_shallowest_config_json() {
        let cache = Path::new("/cache/abc");
        // Diffusers: transformer/config.json is listed before the root one.
        let dir = safetensors_model_dir(
            cache,
            &[
                "transformer/config.json",
                "vae/config.json",
                "config.json",
                "model_index.json",
            ],
        );
        assert_eq!(dir, cache);
        // A plain LLM checkout, as before.
        assert_eq!(
            safetensors_model_dir(cache, &["config.json", "model.safetensors"]),
            cache
        );
        // A repo whose only config.json is nested still resolves into it.
        assert_eq!(
            safetensors_model_dir(cache, &["sub/config.json", "sub/model.safetensors"]),
            cache.join("sub")
        );
        // A pure Diffusers repo: no root config.json, only the pipeline
        // index there and a config.json per component.
        assert_eq!(
            safetensors_model_dir(cache, &["vae/config.json", "model_index.json"]),
            cache
        );
        assert_eq!(safetensors_model_dir(cache, &["a.safetensors"]), cache);
    }

    #[test]
    fn omni_variant_reports_its_format_and_path() {
        let p = ModelPath::Omni(PathBuf::from("/cache/cosmos"));
        assert_eq!(p.format(), "omni");
        assert_eq!(p.path(), Path::new("/cache/cosmos"));
        assert_eq!(p.mmproj(), None);
    }

    #[test]
    fn extract_safetensors_dir_replaces_dest_shorter_than_layer_size() {
        let weights = b"complete-weights-bytes";
        let layer_hex = "aa".repeat(32);
        let mut layer = descriptor(&format!("sha256:{layer_hex}"), "model.safetensors");
        layer.media_type = "application/vnd.cncf.model.weight.v1.raw".into();
        layer.size = weights.len() as u64;
        let (store, manifest) = manifest_with(vec![layer]);

        let blob = store.root().join("blobs").join("sha256").join(&layer_hex);
        std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
        std::fs::write(&blob, weights).unwrap();

        let cache = store.root().join("cache");
        let dest = cache.join("bb".repeat(32)).join("model.safetensors");
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        std::fs::write(&dest, b"trunc").unwrap();

        let digest = format!("sha256:{}", "bb".repeat(32));
        extract_safetensors_dir(store.root(), &cache, &digest, &manifest).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), weights);
    }

    #[test]
    fn extract_safetensors_dir_skips_dest_that_already_matches_layer_size() {
        let weights = b"complete-weights-bytes";
        let layer_hex = "aa".repeat(32);
        let mut layer = descriptor(&format!("sha256:{layer_hex}"), "model.safetensors");
        layer.media_type = "application/vnd.cncf.model.weight.v1.raw".into();
        layer.size = weights.len() as u64;
        let (store, manifest) = manifest_with(vec![layer]);

        let cache = store.root().join("cache");
        let dest = cache.join("bb".repeat(32)).join("model.safetensors");
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        std::fs::write(&dest, weights).unwrap();

        let digest = format!("sha256:{}", "bb".repeat(32));
        extract_safetensors_dir(store.root(), &cache, &digest, &manifest).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), weights);
    }

    #[cfg(unix)]
    #[test]
    fn extract_safetensors_dir_links_dest_to_blob() {
        use std::os::unix::fs::MetadataExt;

        let weights = b"complete-weights-bytes";
        let layer_hex = "aa".repeat(32);
        let mut layer = descriptor(&format!("sha256:{layer_hex}"), "model.safetensors");
        layer.media_type = "application/vnd.cncf.model.weight.v1.raw".into();
        layer.size = weights.len() as u64;
        let (store, manifest) = manifest_with(vec![layer]);

        let blob = store.root().join("blobs").join("sha256").join(&layer_hex);
        std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
        std::fs::write(&blob, weights).unwrap();

        let cache = store.root().join("cache");
        let dest = cache.join("bb".repeat(32)).join("model.safetensors");
        let digest = format!("sha256:{}", "bb".repeat(32));
        extract_safetensors_dir(store.root(), &cache, &digest, &manifest).unwrap();

        let dest_meta = std::fs::metadata(&dest).unwrap();
        let blob_meta = std::fs::metadata(&blob).unwrap();
        assert_eq!(
            (dest_meta.dev(), dest_meta.ino()),
            (blob_meta.dev(), blob_meta.ino())
        );
    }

    #[cfg(unix)]
    #[test]
    fn named_blob_path_replaces_symlink_with_hardlink() {
        use std::os::unix::fs::{symlink, MetadataExt};

        let weights = b"complete-weights-bytes";
        let layer_hex = "aa".repeat(32);
        let mut layer = descriptor(&format!("sha256:{layer_hex}"), "model.safetensors");
        layer.media_type = "application/vnd.cncf.model.weight.v1.raw".into();
        layer.size = weights.len() as u64;
        let (store, _) = manifest_with(vec![layer.clone()]);

        let blob = store.root().join("blobs").join("sha256").join(&layer_hex);
        std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
        std::fs::write(&blob, weights).unwrap();

        let cache = store.root().join("cache");
        let dest = cache.join(&layer_hex).join("model.safetensors");
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        symlink(&blob, &dest).unwrap();

        assert_eq!(named_blob_path(store.root(), &cache, &layer).unwrap(), dest);
        assert!(!std::fs::symlink_metadata(&dest)
            .unwrap()
            .file_type()
            .is_symlink());

        let dest_meta = std::fs::metadata(&dest).unwrap();
        let blob_meta = std::fs::metadata(&blob).unwrap();
        assert_eq!(
            (dest_meta.dev(), dest_meta.ino()),
            (blob_meta.dev(), blob_meta.ino())
        );
    }

    #[test]
    fn copy_file_atomic_replaces_dest_through_temp_file() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-modelpack-copy-file-atomic-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("blob");
        let dest = dir.join("model.safetensors");
        let weights = b"complete-weights-bytes";
        std::fs::write(&src, weights).unwrap();
        std::fs::write(&dest, b"trunc").unwrap();

        copy_file_atomic(&src, &dest, weights.len() as u64).unwrap();

        assert_eq!(std::fs::read(&dest).unwrap(), weights);
        assert!(
            std::fs::read_dir(&dir)
                .unwrap()
                .flatten()
                .all(|e| !e.file_name().to_string_lossy().contains(".tmp")),
            "copy temp file should not remain"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Concurrent placers of one file, as two `llmman resolve`s: all succeed.
    #[test]
    fn concurrent_callers_placing_one_file_all_succeed() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("blob");
        std::fs::write(&src, b"complete-weights-bytes").unwrap();
        for round in 0..150 {
            let dest = dir
                .path()
                .join(format!("tree-{round}"))
                .join("model.safetensors");
            std::thread::scope(|s| {
                let handles: Vec<_> = (0..8)
                    .map(|_| s.spawn(|| cache_layer_file(&src, &dest, 22)))
                    .collect();
                for h in handles {
                    h.join().unwrap().unwrap();
                }
            });
            assert_eq!(std::fs::read(&dest).unwrap(), b"complete-weights-bytes");
            let leftovers: Vec<_> = std::fs::read_dir(dest.parent().unwrap())
                .unwrap()
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n != "model.safetensors")
                .collect();
            assert!(leftovers.is_empty(), "{leftovers:?}");
        }
    }

    #[test]
    fn link_or_copy_file_rejects_a_short_blob() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-modelpack-short-blob-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("blob");
        let dest = dir.join("model.safetensors");
        std::fs::write(&src, b"short").unwrap();

        let error = link_or_copy_file(&src, &dest, 22).unwrap_err();
        assert!(
            error.to_string().contains("is 5 bytes, expected 22"),
            "{error:#}"
        );
        assert!(!dest.exists(), "short linked dest should be removed");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // ── ModelPack tar layers (what `llmman build` and modctl write) ────────

    /// A tar archive of `(name, body)` files.
    fn tar_archive(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut builder = tar::Builder::new(Vec::new());
        for (name, body) in files {
            let mut header = tar::Header::new_gnu();
            header.set_size(body.len() as u64);
            header.set_mode(0o644);
            builder.append_data(&mut header, name, *body).unwrap();
        }
        builder.into_inner().unwrap()
    }

    pub(super) fn gzip(data: &[u8]) -> Vec<u8> {
        use std::io::Write as _;
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        enc.write_all(data).unwrap();
        enc.finish().unwrap()
    }

    /// A layer of `store` holding `files` as `<category>.v1.<suffix>`, with
    /// the first file's path as its `org.cncf.model.filepath`.
    fn tar_layer(
        store: &OciStore,
        category: &str,
        suffix: &str,
        files: &[(&str, &[u8])],
    ) -> crate::storage::oci::Descriptor {
        let archive = tar_archive(files);
        let blob = match suffix {
            "tar+gzip" => gzip(&archive),
            "tar+zstd" => zstd::encode_all(&archive[..], 3).unwrap(),
            _ => archive,
        };
        let media = format!("application/vnd.cncf.model.{category}.v1.{suffix}");
        let mut layer = store.write_blob(&media, &blob).unwrap();
        layer.annotations = Some(
            [(
                "org.cncf.model.filepath".to_string(),
                files[0].0.to_string(),
            )]
            .into(),
        );
        layer
    }

    fn raw_layer(
        store: &OciStore,
        media: &str,
        path: &str,
        body: &[u8],
    ) -> crate::storage::oci::Descriptor {
        let mut layer = store.write_blob(media, body).unwrap();
        layer.annotations =
            Some([("org.cncf.model.filepath".to_string(), path.to_string())].into());
        layer
    }

    fn manifest_digest(store: &OciStore, manifest: &crate::storage::oci::Manifest) -> String {
        store.write_manifest(manifest).unwrap().digest
    }

    #[test]
    fn a_built_model_directory_resolves_to_its_files() {
        let src = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(src.path().join("tokenizer")).unwrap();
        std::fs::write(src.path().join("config.json"), br#"{"model_type":"llama"}"#).unwrap();
        std::fs::write(src.path().join("model.safetensors"), b"weights").unwrap();
        std::fs::write(src.path().join("tokenizer/tokenizer.json"), b"{}").unwrap();
        std::fs::write(src.path().join("README.md"), b"# docs").unwrap();
        let (store, _) = manifest_with(vec![]);
        store
            .build(src.path(), "example.invalid/m:v1", &Default::default())
            .unwrap();
        let cache = store.root().join("cache");

        let ModelPath::SafeTensors(dir) =
            resolve_model(store.root(), &cache, "example.invalid/m:v1").unwrap()
        else {
            panic!("not a safetensors model");
        };
        assert_eq!(
            std::fs::read(dir.join("model.safetensors")).unwrap(),
            b"weights"
        );
        assert_eq!(
            std::fs::read(dir.join("tokenizer/tokenizer.json")).unwrap(),
            b"{}"
        );
        assert!(dir.join("config.json").is_file());
        // serving does not need the docs
        assert!(!dir.join("README.md").exists());

        // the whole directory does
        let tree = resolve_model_tree(store.root(), &cache, "example.invalid/m:v1").unwrap();
        assert_eq!(tree.root, dir);
        assert_eq!(tree.format, Some(ModelFormat::SafeTensors));
        assert_eq!(
            std::fs::read(tree.root.join("README.md")).unwrap(),
            b"# docs"
        );
    }

    #[test]
    fn the_tree_places_every_category_and_encoding_at_its_path() {
        let (store, mut m) = manifest_with(vec![]);
        const RAW_W: &str = "application/vnd.cncf.model.weight.v1.raw";
        const RAW_DOC: &str = "application/vnd.cncf.model.doc.v1.raw";
        m.layers = vec![
            tar_layer(&store, "weight", "tar", &[("model-1.safetensors", b"one")]),
            raw_layer(&store, RAW_W, "model-2.safetensors", b"two"),
            tar_layer(
                &store,
                "weight.config",
                "tar+gzip",
                &[("config.json", b"{}")],
            ),
            tar_layer(
                &store,
                "weight.config",
                "tar+zstd",
                &[
                    ("tokenizer/tokenizer.json", b"tok"),
                    ("tokenizer/merges.txt", b"m"),
                ],
            ),
            tar_layer(&store, "doc", "tar", &[("docs/README.md", b"readme")]),
            raw_layer(&store, RAW_DOC, "LICENSE", b"license"),
            tar_layer(&store, "code", "tar+gzip", &[("handler.py", b"print(1)")]),
            tar_layer(&store, "dataset", "tar+zstd", &[("data/a.jsonl", b"{}")]),
        ];
        let digest = manifest_digest(&store, &m);
        let cache = store.root().join("cache");

        let (root, placed) =
            extract_model_tree(store.root(), &cache, &digest, &m, TreeScope::Full).unwrap();
        let read = |p: &str| std::fs::read(root.join(p)).unwrap();
        assert_eq!(read("model-1.safetensors"), b"one");
        assert_eq!(read("model-2.safetensors"), b"two");
        assert_eq!(read("config.json"), b"{}");
        assert_eq!(read("tokenizer/tokenizer.json"), b"tok");
        assert_eq!(read("tokenizer/merges.txt"), b"m");
        assert_eq!(read("docs/README.md"), b"readme");
        assert_eq!(read("LICENSE"), b"license");
        assert_eq!(read("handler.py"), b"print(1)");
        assert_eq!(read("data/a.jsonl"), b"{}");
        assert_eq!(placed.len(), 9);

        // Idempotent: a second call places the same files.
        let (root2, placed2) =
            extract_model_tree(store.root(), &cache, &digest, &m, TreeScope::Full).unwrap();
        assert_eq!((root2, placed2), (root.clone(), placed));

        // Nothing else lives in the tree.
        let mut on_disk: Vec<_> = walkdir::WalkDir::new(&root)
            .min_depth(1)
            .into_iter()
            .flatten()
            .filter(|e| e.file_type().is_file())
            .map(|e| {
                e.path()
                    .strip_prefix(&root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect();
        on_disk.sort();
        assert_eq!(
            on_disk,
            [
                "LICENSE",
                "config.json",
                "data/a.jsonl",
                "docs/README.md",
                "handler.py",
                "model-1.safetensors",
                "model-2.safetensors",
                "tokenizer/merges.txt",
                "tokenizer/tokenizer.json",
            ]
        );
    }

    #[test]
    fn two_layers_placing_one_path_fail_the_tree() {
        const RAW_W: &str = "application/vnd.cncf.model.weight.v1.raw";
        let (store, mut m) = manifest_with(vec![]);
        m.layers = vec![
            raw_layer(&store, RAW_W, "config.json", b"one"),
            tar_layer(&store, "weight.config", "tar", &[("config.json", b"two")]),
        ];
        let digest = manifest_digest(&store, &m);
        let err = extract_model_tree(
            store.root(),
            &store.root().join("cache"),
            &digest,
            &m,
            TreeScope::Full,
        )
        .unwrap_err();
        assert!(err.to_string().contains("two layers place config.json"));
    }

    #[test]
    fn the_serving_directory_is_picked_from_what_was_placed() {
        let (store, mut m) = manifest_with(vec![]);
        m.layers = vec![
            // not placed when serving, so its path must not pick the root
            raw_layer(
                &store,
                "application/vnd.cncf.model.doc.v1.raw",
                "config.json",
                b"doc",
            ),
            tar_layer(
                &store,
                "weight",
                "tar",
                &[("runtime/model.safetensors", b"w")],
            ),
            tar_layer(
                &store,
                "weight.config",
                "tar",
                &[("runtime/config.json", b"{}")],
            ),
        ];
        let digest = manifest_digest(&store, &m);
        let cache = store.root().join("cache");
        let dir = extract_safetensors_dir(store.root(), &cache, &digest, &m).unwrap();
        assert_eq!(
            dir,
            cache.join(digest_hex(&digest).unwrap()).join("runtime")
        );
    }

    #[test]
    fn a_servable_tree_leaves_out_docs_code_and_datasets() {
        let (store, mut m) = manifest_with(vec![]);
        m.layers = vec![
            tar_layer(&store, "weight", "tar", &[("model.safetensors", b"w")]),
            tar_layer(&store, "weight.config", "tar", &[("config.json", b"{}")]),
            tar_layer(&store, "doc", "tar", &[("README.md", b"r")]),
            tar_layer(&store, "code", "tar+gzip", &[("run.py", b"c")]),
            tar_layer(&store, "dataset", "tar", &[("d.csv", b"d")]),
        ];
        let digest = manifest_digest(&store, &m);
        let (root, placed) = extract_model_tree(
            store.root(),
            &store.root().join("cache"),
            &digest,
            &m,
            TreeScope::Servable,
        )
        .unwrap();
        assert_eq!(placed, ["model.safetensors", "config.json"]);
        assert!(!root.join("README.md").exists());
        assert!(!root.join("run.py").exists());
        assert!(!root.join("d.csv").exists());
    }

    #[test]
    fn a_raw_layer_with_an_unsafe_filepath_stays_out_of_the_tree() {
        let (store, mut m) = manifest_with(vec![]);
        m.layers = vec![
            raw_layer(
                &store,
                "application/vnd.cncf.model.weight.v1.raw",
                "../../escape.bin",
                b"x",
            ),
            raw_layer(
                &store,
                "application/vnd.cncf.model.weight.v1.raw",
                "ok.bin",
                b"y",
            ),
        ];
        let digest = manifest_digest(&store, &m);
        let cache = store.root().join("cache");
        let (root, placed) =
            extract_model_tree(store.root(), &cache, &digest, &m, TreeScope::Full).unwrap();
        assert_eq!(placed, ["ok.bin"]);
        assert!(!cache.join("escape.bin").exists());
        assert!(!store.root().join("escape.bin").exists());
        assert!(root.join("ok.bin").is_file());
    }

    #[test]
    fn a_tar_layer_cannot_write_outside_the_tree() {
        let (store, mut m) = manifest_with(vec![]);
        let evil = {
            let mut header = tar::Header::new_gnu();
            header.set_size(1);
            header.set_mode(0o644);
            let name = b"../../escape.bin";
            header.as_old_mut().name[..name.len()].copy_from_slice(name);
            header.set_cksum();
            let mut tar = header.as_bytes().to_vec();
            tar.push(b'x');
            tar.resize(tar.len() + 511 + 1024, 0);
            tar
        };
        let mut layer = store
            .write_blob("application/vnd.cncf.model.weight.v1.tar", &evil)
            .unwrap();
        layer.annotations = Some(
            [(
                "org.cncf.model.filepath".to_string(),
                "model.bin".to_string(),
            )]
            .into(),
        );
        m.layers = vec![layer];
        let digest = manifest_digest(&store, &m);
        let cache = store.root().join("cache");
        let err =
            extract_model_tree(store.root(), &cache, &digest, &m, TreeScope::Full).unwrap_err();
        assert!(format!("{err:#}").contains("unsafe path"), "{err:#}");
        assert!(!cache.join("escape.bin").exists());
        assert!(!store.root().join("escape.bin").exists());
    }

    #[test]
    fn gguf_in_compressed_tar_layers_resolves_with_its_projector() {
        let gguf = |tail: &[u8]| [b"GGUF".as_slice(), &[3, 0, 0, 0], tail].concat();
        let main = gguf(b"model");
        let proj = gguf(b"proj");
        let (store, mut m) = manifest_with(vec![]);
        m.layers = vec![
            tar_layer(
                &store,
                "weight",
                "tar+gzip",
                &[("model.Q4_K_M.gguf", &main)],
            ),
            tar_layer(&store, "weight", "tar+zstd", &[("mmproj-F16.gguf", &proj)]),
        ];
        let desc = store.write_manifest(&m).unwrap();
        store.tag(desc, "example.invalid/gguf:v1").unwrap();
        let cache = store.root().join("cache");
        let ModelPath::Gguf(path, Some(mmproj)) =
            resolve_model(store.root(), &cache, "example.invalid/gguf:v1").unwrap()
        else {
            panic!("not a gguf model with a projector");
        };
        assert_eq!(std::fs::read(&path).unwrap(), main);
        assert_eq!(std::fs::read(&mmproj).unwrap(), proj);
        // resolved again from what is cached
        let ModelPath::Gguf(again, _) =
            resolve_model(store.root(), &cache, "example.invalid/gguf:v1").unwrap()
        else {
            panic!("not a gguf model");
        };
        assert_eq!(again, path);
    }

    #[test]
    fn a_gguf_cached_by_an_earlier_release_is_still_found() {
        let (store, _) = manifest_with(vec![]);
        let layer = tar_layer(&store, "weight", "tar", &[("old.gguf", b"GGUFxxxx")]);
        let hex = digest_hex(&layer.digest).unwrap();
        let cache = store.root().join("cache");
        std::fs::create_dir_all(cache.join(hex)).unwrap();
        std::fs::write(cache.join(hex).join("old.gguf"), b"already here").unwrap();
        let path = extract_gguf_layer(store.root(), &cache, &layer).unwrap();
        assert_eq!(std::fs::read(path).unwrap(), b"already here");
    }

    #[test]
    fn a_gguf_stored_whole_under_a_tar_media_type_is_cached_by_name() {
        let (store, _) = manifest_with(vec![]);
        let mut layer = store
            .write_blob("application/vnd.cncf.model.weight.v1.tar", b"GGUF-whole")
            .unwrap();
        layer.annotations = Some(
            [(
                "org.cncf.model.filepath".to_string(),
                "dir/m.gguf".to_string(),
            )]
            .into(),
        );
        let cache = store.root().join("cache");
        let path = extract_gguf_layer(store.root(), &cache, &layer).unwrap();
        assert_eq!(path.file_name().unwrap(), "m.gguf");
        assert_eq!(std::fs::read(path).unwrap(), b"GGUF-whole");
    }

    #[test]
    fn read_layer_text_reads_a_single_file_layer_in_any_encoding() {
        let (store, _) = manifest_with(vec![]);
        let body = vec![b'x'; 600];
        for suffix in ["tar", "tar+gzip", "tar+zstd"] {
            let layer = tar_layer(&store, "weight.config", suffix, &[("a.json", &body)]);
            assert_eq!(
                read_layer_text(&store, &layer).unwrap().as_bytes(),
                body.as_slice(),
                "{suffix}"
            );
        }
        let raw = raw_layer(
            &store,
            "application/vnd.cncf.model.weight.config.v1.raw",
            "a.json",
            b"{}",
        );
        assert_eq!(read_layer_text(&store, &raw).unwrap(), "{}");
    }

    #[test]
    fn a_tree_needs_layers_it_can_place() {
        let (store, mut m) = manifest_with(vec![]);
        // Docker's own artifact format is not ModelPack: no layers to place.
        let mut layer = store.write_blob(HF_GGUF_MEDIA_TYPE, b"GGUF").unwrap();
        layer.annotations =
            Some([("org.cncf.model.filepath".to_string(), "m.gguf".to_string())].into());
        m.layers = vec![layer];
        let desc = store.write_manifest(&m).unwrap();
        store.tag(desc, "example.invalid/docker:v1").unwrap();
        let err = resolve_model_tree(
            store.root(),
            &store.root().join("cache"),
            "example.invalid/docker:v1",
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("no ModelPack file layers"),
            "{err:#}"
        );
    }

    #[test]
    fn a_tree_of_an_artifact_llmman_cannot_serve_is_still_a_tree() {
        let (store, mut m) = manifest_with(vec![]);
        m.layers = vec![tar_layer(
            &store,
            "weight",
            "tar",
            &[("model.onnx", b"onnx")],
        )];
        let desc = store.write_manifest(&m).unwrap();
        store.tag(desc, "example.invalid/onnx:v1").unwrap();
        let tree = resolve_model_tree(
            store.root(),
            &store.root().join("cache"),
            "example.invalid/onnx:v1",
        )
        .unwrap();
        assert_eq!(tree.format, None);
        assert_eq!(
            std::fs::read(tree.root.join("model.onnx")).unwrap(),
            b"onnx"
        );
    }

    #[test]
    fn is_gguf_layer_matches_both_mmproj_and_primary_model_files() {
        // is_mmproj_layer only narrows *within* the GGUF files a manifest
        // has — is_gguf_layer itself must still say "yes" for an mmproj
        // file, or resolve_model's gguf_layers filter would silently
        // drop it instead of finding a companion.
        assert!(is_gguf_layer(&descriptor("sha256:a", "mmproj-F16.gguf")));
        assert!(is_gguf_layer(&descriptor("sha256:b", "model.Q4_K_M.gguf")));
        assert!(!is_gguf_layer(&descriptor("sha256:c", "model.safetensors")));
    }
}
