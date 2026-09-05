//! Model sources that are neither an OCI registry nor the HuggingFace
//! Hub. Ported from the Go shim's `uri_sources.go`, whose deletion also
//! removed the last caller of `hf.go`.
//!
//! Supported references (the scheme table mirrors NVIDIA NIM's
//! "Model-Free NIM"):
//!
//! | Reference | Source | Auth |
//! |---|---|---|
//! | `ms://owner/repo[:revision]` | ModelScope Hub | `MODELSCOPE_API_TOKEN` |
//! | `modelscope://…` | alias for `ms://` | |
//! | `ngc://org[/team]/model[:version]` | NVIDIA NGC | `NGC_API_KEY` |
//! | `s3://bucket/prefix` | AWS S3 / S3-compatible | the AWS credential chain |
//! | `gs://bucket/prefix` | Google Cloud Storage | `GOOGLE_ACCESS_TOKEN` / ADC |
//! | `/absolute/path` | a local directory | none |
//!
//! `hf://` is deliberately absent: [`crate::hf::classify`] routes it to
//! [`crate::hf`] before this module is reached.
//!
//! Every source writes what the rest of llmman already reads: a CNCF
//! ModelPack manifest in the local OCI layout, one raw blob per file
//! (see [`crate::hf::oci`]). Only [`transfer`] talks to the Go shim, to
//! reuse its registry push.

pub mod gcs;
pub mod local;
pub mod modelscope;
pub mod ngc;
pub mod s3;

use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use bytes::Bytes;
use futures::stream::{Stream, StreamExt as _};

use crate::hf::oci::{self, ModelMeta};
use crate::hf::progress;

/// Every scheme this module claims. `modelscope://` is normally already
/// rewritten to `ms://` by [`crate::shortnames::resolve`]; both are
/// accepted so a caller that skipped that step still routes correctly.
const SCHEMES: &[&str] = &["ms://", "modelscope://", "ngc://", "s3://", "gs://"];

/// Whether `reference` names one of this module's sources.
///
/// A leading `/` is a local directory to import. Deliberately not
/// widened to Windows drive-letter paths: the Go dispatcher this
/// replaces only matched `/`, and widening it would newly capture
/// references that used to reach the registry client.
pub fn handles(reference: &str) -> bool {
    reference.starts_with('/') || SCHEMES.iter().any(|s| reference.starts_with(s))
}

/// Pulls `reference` into `layout_dir`. `progress_key` is the exact,
/// original ref the daemon's `/api/pull` handler was given, so
/// [`crate::hf::progress::poll`] can report byte counts against it —
/// pass `""` when nothing is polling (see that module's doc comment).
pub async fn pull(reference: &str, layout_dir: &Path, progress_key: &str) -> Result<()> {
    let _guard = progress::DoneGuard(progress_key);
    pull_into(
        reference,
        &Target {
            layout_dir,
            progress_key,
            store_as: None,
        },
    )
    .await
}

/// Transfers `reference` straight to an OCI registry `destination`:
/// pull into a throwaway local layout, then hand it to the Go shim's
/// registry push. Streaming into the push instead would need each
/// file's content digest up front, which no generic file store offers —
/// so this stages through disk, as `transferViaStaging` did.
///
/// `changed` is always true on success, like `crate::hf::transfer`'s
/// podman fallback: `ffi::push` doesn't report whether anything changed.
pub async fn transfer(reference: &str, destination: &str) -> Result<crate::ffi::TransferOutcome> {
    let tmp = std::env::temp_dir().join(format!(
        "llmman-source-transfer-{}-{}",
        std::process::id(),
        unique()
    ));
    // store_as: `llmman_push` resolves what to push by an *exact* ref
    // lookup, so a manifest filed under "s3://bucket/x" would not be
    // found when pushing to "docker.io/me/x:v1".
    let result = pull_into(
        reference,
        &Target {
            layout_dir: &tmp,
            progress_key: "",
            store_as: Some(destination),
        },
    )
    .await
    .and_then(|()| {
        crate::ffi::push(
            tmp.to_str()
                .context("temp layout path is not valid UTF-8")?,
            destination,
        )
    })
    // Read the digest back out of the staging layout rather than
    // re-resolving the destination tag afterwards: this is the manifest
    // that was just pushed, so it is what `--sign-key` must sign.
    .and_then(|()| crate::hf::oci::read_manifest_ref(&tmp, destination))
    .map(|desc| crate::ffi::TransferOutcome::new(true, desc.digest));
    let _ = std::fs::remove_dir_all(&tmp);
    result
}

/// Where one source pull writes, and under what name.
pub(crate) struct Target<'a> {
    pub layout_dir: &'a Path,
    pub progress_key: &'a str,
    /// Overrides the manifest ref the pulled model is recorded under.
    /// Set only by [`transfer`] — see its own doc comment.
    pub store_as: Option<&'a str>,
}

impl Target<'_> {
    /// The reference to record this pull under: the destination for a
    /// transfer, otherwise the source's own canonical form.
    fn store_ref<'a>(&'a self, canonical: &'a str) -> &'a str {
        self.store_as.unwrap_or(canonical)
    }

    /// True once `canonical` is fully present in this layout, meaning
    /// the caller must skip all network I/O. Always false for a
    /// transfer's staging directory, which starts empty.
    fn report_cached(&self, canonical: &str, label: &str) -> bool {
        if self.store_as.is_some() {
            return false;
        }
        if oci::cached_layer_name(self.layout_dir, canonical).is_none() {
            return false;
        }
        eprintln!("Cached   {label}");
        true
    }
}

async fn pull_into(reference: &str, target: &Target<'_>) -> Result<()> {
    oci::ensure_layout(target.layout_dir)?;
    progress::set_status(target.progress_key, "pulling");

    if let Some(rest) = strip_any_prefix(reference, &["ms://", "modelscope://"]) {
        return modelscope::pull(reference, rest, target).await;
    }
    if let Some(rest) = reference.strip_prefix("ngc://") {
        return ngc::pull(reference, rest, target).await;
    }
    if let Some(rest) = reference.strip_prefix("s3://") {
        return s3::pull(reference, rest, target).await;
    }
    if let Some(rest) = reference.strip_prefix("gs://") {
        return gcs::pull(reference, rest, target).await;
    }
    if reference.starts_with('/') {
        return local::pull(reference, target);
    }
    anyhow::bail!("{reference}: not a source reference this module handles")
}

fn strip_any_prefix<'a>(s: &'a str, prefixes: &[&str]) -> Option<&'a str> {
    prefixes.iter().find_map(|p| s.strip_prefix(p))
}

/// Splits a `bucket/prefix` remainder (an `s3://`/`gs://` reference with
/// its scheme already removed) into its two halves.
pub(crate) fn split_bucket_prefix<'a>(
    rest: &'a str,
    reference: &str,
) -> Result<(&'a str, &'a str)> {
    rest.split_once('/').with_context(|| {
        let scheme = reference.split_once("://").map(|(s, _)| s).unwrap_or("");
        format!("invalid {scheme} reference {reference:?}: expected {scheme}://bucket/prefix")
    })
}

/// A listing entry's path within the pull: `key` with the requested
/// `prefix` removed. `None` for the prefix entry itself, and for a key
/// that only shares a *textual* prefix — S3 and GCS both return every
/// key starting with the string, so `models/llama` also matches
/// `models/llama-2/config.json`, which belongs to another model.
pub(crate) fn relative_to_prefix<'a>(key: &'a str, prefix: &str) -> Option<&'a str> {
    let rest = key.strip_prefix(prefix)?;
    if !(rest.starts_with('/') || prefix.ends_with('/') || prefix.is_empty()) {
        return None;
    }
    let rel = rest.trim_start_matches('/');
    (!rel.is_empty()).then_some(rel)
}

// ---------------------------------------------------------------------------
// Media-type classification (CNCF ModelPack)
// ---------------------------------------------------------------------------

/// Maps a file extension to the CNCF model layer media type to store it
/// under. The raw (non-tar) variants, since each file becomes its own
/// uncompressed blob.
///
/// Broader than [`crate::hf::api::safetensors_media_type`]: these are
/// arbitrary file stores, so they can carry TensorRT/ONNX engines and
/// inference code a HuggingFace repo's file list never would.
pub fn classify_file(name: &str) -> &'static str {
    let lower = name.rsplit('/').next().unwrap_or(name).to_lowercase();
    let ext = lower.rsplit_once('.').map(|(_, e)| e).unwrap_or("");

    match ext {
        "safetensors" | "bin" | "pt" | "pth" | "gguf" | "ggml" | "gguf_v2" | "ot" | "engine"
        | "trt" | "onnx" => oci::MEDIA_TYPE_MODEL_WEIGHT_RAW,
        // "jinja": a standalone chat_template.jinja file.
        "json" | "yaml" | "yml" | "toml" | "ini" | "cfg" | "conf" | "model" | "tiktoken"
        | "vocab" | "merges" | "spm" | "jinja" => oci::MEDIA_TYPE_MODEL_WEIGHT_CONFIG_RAW,
        // A tokenizer vocab/merges file is config; a README is a doc.
        "txt" if lower.contains("vocab") || lower.contains("merges") => {
            oci::MEDIA_TYPE_MODEL_WEIGHT_CONFIG_RAW
        }
        "py" | "sh" | "js" | "ts" => oci::MEDIA_TYPE_MODEL_CODE_RAW,
        _ => oci::MEDIA_TYPE_MODEL_DOC_RAW,
    }
}

/// True if `rel_path` is safe to record in a layer's
/// `org.cncf.model.filepath` and later join onto a cache directory.
///
/// A listing comes from the remote, so it decides these strings:
/// `modelpack::extract_safetensors_dir` joins them straight onto the
/// cache dir, and `../../id_rsa` would escape it and overwrite whatever
/// it landed on. Absolute paths, any `..`, drive letters and backslashes
/// (a `\` is a legal filename byte on POSIX but a separator on Windows,
/// so a layout written on one is unsafe to extract on the other) are all
/// rejected outright rather than sanitized — silently renaming a file
/// would be its own surprise.
pub fn is_safe_relative_path(rel_path: &str) -> bool {
    !rel_path.is_empty()
        && !rel_path.starts_with('/')
        && !rel_path.contains('\\')
        && !rel_path.contains(':')
        && rel_path
            .split('/')
            .all(|c| !matches!(c, ".." | ".") && !c.is_empty())
}

/// True for a file worth packing out of a generic file store: anything
/// [`classify_file`] has a real media type for, plus the docs a model
/// directory conventionally carries. Dotfiles never are.
///
/// Derived from `classify_file` so the two can't drift. `crate::hf`'s
/// filter is narrower (safetensors only — it selects GGUF in a separate
/// pass), which here would make a `.gguf`-only bucket pull nothing.
pub(crate) fn should_pack(path: &str) -> bool {
    let base = path.rsplit('/').next().unwrap_or(path).to_lowercase();
    is_safe_relative_path(path)
        && !base.starts_with('.')
        && (classify_file(path) != oci::MEDIA_TYPE_MODEL_DOC_RAW
            || base.starts_with("readme")
            || base.starts_with("licen")
            || base.ends_with(".txt")
            || base.ends_with(".md"))
}

// ---------------------------------------------------------------------------
// Generic CNCF ModelPack packager
// ---------------------------------------------------------------------------

/// One file staged on local disk, ready to become a ModelPack layer.
pub(crate) struct PackFile {
    /// Where the bytes are right now.
    pub(crate) local_path: PathBuf,
    /// The path recorded in the layer's `org.cncf.model.filepath`
    /// annotation, and what its media type is classified from.
    pub(crate) relative_path: String,
    /// True for a file this process staged and therefore owns: moved
    /// into the blob store rather than copied, and removed either way.
    /// False for a user's own file, which must be left as it was.
    pub(crate) owned: bool,
}

impl Drop for PackFile {
    fn drop(&mut self) {
        // A no-op once moved into the blob store; on every early return
        // it is what keeps a failed pull from littering blobs/tmp.
        if self.owned {
            let _ = std::fs::remove_file(&self.local_path);
        }
    }
}

/// Streams one file's `body` into the layout's temp area, reporting
/// bytes to [`progress`] as they arrive — the shared inner loop of every
/// remote source below. `tmp_prefix` namespaces the staging file per
/// source (`"ms"`, `"ngc"`, …); `kind` labels errors (`"ModelScope"`).
pub(crate) async fn download_to_pack_file<E>(
    target: &Target<'_>,
    tmp_prefix: &str,
    kind: &str,
    rel_path: &str,
    size: i64,
    body: impl Stream<Item = std::result::Result<Bytes, E>>,
) -> Result<PackFile>
where
    E: std::error::Error + Send + Sync + 'static,
{
    let base = rel_path.rsplit('/').next().unwrap_or(rel_path);
    eprintln!("Pulling  {base}");
    progress::add_total(target.progress_key, size);

    let tmp_dir = target.layout_dir.join("blobs").join("tmp");
    std::fs::create_dir_all(&tmp_dir)?;
    let tmp_path = tmp_dir.join(format!(
        "{tmp_prefix}-{}-{}-{}.part",
        std::process::id(),
        unique(),
        sanitize(rel_path)
    ));
    // Before the first byte is written, so its Drop cleans up the
    // partial file however this function returns.
    let pack = PackFile {
        local_path: tmp_path.clone(),
        relative_path: rel_path.to_string(),
        owned: true,
    };

    let mut file = std::fs::File::create(&tmp_path)
        .with_context(|| format!("{kind}: create staging file for {rel_path}"))?;
    let mut body = std::pin::pin!(body);
    while let Some(chunk) = body.next().await {
        let chunk = chunk.with_context(|| format!("{kind} download {rel_path}"))?;
        file.write_all(&chunk)
            .with_context(|| format!("{kind} write {rel_path}"))?;
        progress::add_completed(target.progress_key, chunk.len() as i64);
    }
    file.flush()
        .with_context(|| format!("{kind} write {rel_path}"))?;
    drop(file);

    eprintln!("Pulled   {base}");
    Ok(pack)
}

/// Writes every file as a raw blob and records a CNCF ModelPack
/// manifest referencing them all. `no_files_err` is returned verbatim
/// when `files` is empty; each source phrases that differently.
pub(crate) fn pack_as_model_pack(
    target: &Target<'_>,
    canonical_ref: &str,
    model_repo: &str,
    files: Vec<PackFile>,
    no_files_err: String,
) -> Result<()> {
    if files.is_empty() {
        anyhow::bail!(no_files_err);
    }

    let mut layers = Vec::with_capacity(files.len());
    for f in &files {
        let mut desc = oci::write_blob_from_file(
            target.layout_dir,
            classify_file(&f.relative_path),
            &f.local_path,
            f.owned,
        )
        .with_context(|| format!("store {}", f.relative_path))?;
        desc.annotations = Some(std::collections::BTreeMap::from([(
            oci::ANNOTATION_FILEPATH.to_string(),
            f.relative_path.clone(),
        )]));
        layers.push(desc);
    }

    // No HuggingFace-style model card to read a license off, so the
    // format label is all that is known — and only when the weights
    // agree on one, or a GGUF bucket would advertise itself as
    // safetensors.
    let meta = ModelMeta {
        format: weight_format(&files).unwrap_or_default().to_string(),
        ..Default::default()
    };
    let manifest_desc = oci::build_cncf_manifest(target.layout_dir, &meta, model_repo, "", layers)?;
    oci::write_manifest_ref(
        target.layout_dir,
        target.store_ref(canonical_ref),
        manifest_desc,
    )
}

/// The CNCF `config.format` for a packed set of files, or `None` when no
/// weight file names one. GGUF wins over safetensors, matching
/// `storage::oci::build`'s own precedence.
fn weight_format(files: &[PackFile]) -> Option<&'static str> {
    let mut format = None;
    for f in files {
        let lower = f.relative_path.to_lowercase();
        if lower.ends_with(".gguf") || lower.ends_with(".ggml") {
            return Some("gguf");
        }
        if lower.ends_with(".safetensors") && format.is_none() {
            format = Some("safetensors");
        }
    }
    format
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// A process-wide counter, so two concurrent pulls that happen to share
/// a filename never collide on a staging path.
fn unique() -> u64 {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_matches_every_documented_scheme_and_local_paths() {
        for r in [
            "ms://owner/repo",
            "modelscope://owner/repo",
            "ngc://org/team/model",
            "s3://bucket/prefix",
            "gs://bucket/prefix",
            "/abs/path/to/model",
        ] {
            assert!(handles(r), "{r} should be handled by crate::sources");
        }
    }

    #[test]
    fn handles_leaves_hf_and_registry_references_alone() {
        for r in [
            "hf://owner/repo",
            "huggingface://owner/repo",
            "hf.co/owner/repo",
            "docker.io/library/alpine:latest",
            "registry.example.com:5000/owner/repo:v1",
        ] {
            assert!(!handles(r), "{r} must not be claimed by crate::sources");
        }
    }

    #[test]
    fn classify_file_maps_weights_config_code_and_docs() {
        assert_eq!(
            classify_file("model-00001-of-00002.safetensors"),
            oci::MEDIA_TYPE_MODEL_WEIGHT_RAW
        );
        assert_eq!(
            classify_file("model.onnx"),
            oci::MEDIA_TYPE_MODEL_WEIGHT_RAW
        );
        assert_eq!(
            classify_file("config.json"),
            oci::MEDIA_TYPE_MODEL_WEIGHT_CONFIG_RAW
        );
        assert_eq!(
            classify_file("chat_template.jinja"),
            oci::MEDIA_TYPE_MODEL_WEIGHT_CONFIG_RAW
        );
        assert_eq!(classify_file("handler.py"), oci::MEDIA_TYPE_MODEL_CODE_RAW);
        assert_eq!(classify_file("README.md"), oci::MEDIA_TYPE_MODEL_DOC_RAW);
    }

    /// A tokenizer's `vocab.txt`/`merges.txt` is configuration the
    /// runtime must load, not documentation — doc-type layers are
    /// dropped before serving, so misclassifying these breaks the model.
    #[test]
    fn classify_file_treats_vocab_and_merges_text_files_as_config() {
        assert_eq!(
            classify_file("vocab.txt"),
            oci::MEDIA_TYPE_MODEL_WEIGHT_CONFIG_RAW
        );
        assert_eq!(
            classify_file("tokenizer/merges.txt"),
            oci::MEDIA_TYPE_MODEL_WEIGHT_CONFIG_RAW
        );
        assert_eq!(classify_file("notes.txt"), oci::MEDIA_TYPE_MODEL_DOC_RAW);
    }

    /// The formats `classify_file` claims to support must actually be
    /// packable: filtering with the HuggingFace path's safetensors-only
    /// predicate made a GGUF/ONNX/TensorRT bucket fail with "no model
    /// files found".
    #[test]
    fn should_pack_accepts_every_weight_format_classify_file_knows() {
        for f in [
            "model.gguf",
            "model.ggml",
            "model.onnx",
            "model.engine",
            "model.trt",
            "model.safetensors",
            "config.json",
            "tokenizer.model",
            "chat_template.jinja",
            "handler.py",
            "README.md",
            "LICENSE",
            "vocab.txt",
        ] {
            assert!(should_pack(f), "{f} should be packed");
        }
    }

    #[test]
    fn should_pack_skips_dotfiles_and_unknown_binaries() {
        for f in [".gitattributes", ".DS_Store", "weights.tflite", "cache.db"] {
            assert!(!should_pack(f), "{f} should not be packed");
        }
    }

    /// A listing comes from the remote, so a hostile or broken one must
    /// not be able to write outside the cache directory these paths are
    /// later joined onto.
    #[test]
    fn is_safe_relative_path_rejects_traversal_and_absolute_paths() {
        for p in [
            "../../id_rsa",
            "a/../../b",
            "/etc/passwd",
            "..",
            "a//b",
            "..\\windows",
            "C:/weights.safetensors",
            "",
        ] {
            assert!(!is_safe_relative_path(p), "{p:?} must be rejected");
            assert!(!should_pack(p), "{p:?} must not be packed");
        }
        for p in ["config.json", "sub/dir/model.safetensors"] {
            assert!(is_safe_relative_path(p), "{p:?} must be accepted");
        }
    }

    #[test]
    fn weight_format_prefers_gguf_and_is_none_without_weights() {
        let f = |p: &str| PackFile {
            local_path: PathBuf::from(p),
            relative_path: p.to_string(),
            owned: false,
        };
        assert_eq!(
            weight_format(&[f("model.safetensors"), f("model.gguf")]),
            Some("gguf")
        );
        assert_eq!(
            weight_format(&[f("config.json"), f("model.safetensors")]),
            Some("safetensors")
        );
        assert_eq!(weight_format(&[f("model.onnx")]), None);
    }

    #[test]
    fn split_bucket_prefix_requires_a_prefix() {
        assert_eq!(
            split_bucket_prefix("bucket/some/prefix", "s3://bucket/some/prefix").unwrap(),
            ("bucket", "some/prefix")
        );
        assert!(split_bucket_prefix("bucket", "s3://bucket").is_err());
    }

    #[test]
    fn relative_to_prefix_strips_the_prefix_and_skips_the_prefix_entry_itself() {
        assert_eq!(
            relative_to_prefix("models/llama/config.json", "models/llama"),
            Some("config.json")
        );
        assert_eq!(
            relative_to_prefix("models/llama/config.json", "models/llama/"),
            Some("config.json")
        );
        assert_eq!(
            relative_to_prefix("models/llama/", "models/llama"),
            None,
            "the prefix \"directory\" entry has nothing left over"
        );
    }

    /// Both object stores match the prefix as a plain string, so a
    /// sibling model's files come back in the same listing. Packing them
    /// would silently mix two models into one manifest.
    #[test]
    fn relative_to_prefix_rejects_a_key_that_only_shares_a_textual_prefix() {
        assert_eq!(
            relative_to_prefix("models/llama-2/config.json", "models/llama"),
            None
        );
        assert_eq!(relative_to_prefix("other/config.json", "models"), None);
    }
}
