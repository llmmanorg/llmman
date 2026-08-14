//! Local OCI Image Layout store.
//!
//! Implements a subset of the OCI Image Layout spec
//! (<https://github.com/opencontainers/image-spec/blob/main/image-layout.md>)
//! sufficient for llmman's local operations: build, list, rm, tag, inspect-local.
//!
//! Layout on disk:
//! ```text
//! <store-root>/
//!   oci-layout             {"imageLayoutVersion":"1.0.0"}
//!   index.json             OCI image index
//!   blobs/
//!     sha256/
//!       <hex>              one file per blob
//! ```

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context};
use serde::{Deserialize, Serialize};
use sha2::{Digest as Sha2Digest, Sha256};

// ---------------------------------------------------------------------------
// Minimal OCI spec types (no external crate needed)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Descriptor {
    pub media_type: String,
    pub digest: String,
    pub size: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Index {
    pub schema_version: u32,
    pub media_type: String,
    pub manifests: Vec<Descriptor>,
}

impl Default for Index {
    fn default() -> Self {
        Self {
            schema_version: 2,
            media_type: "application/vnd.oci.image.index.v1+json".into(),
            manifests: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Manifest {
    pub schema_version: u32,
    pub media_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_type: Option<String>,
    pub config: Descriptor,
    pub layers: Vec<Descriptor>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<std::collections::HashMap<String, String>>,
    /// OCI 1.1 `subject` — set on a signature (or other) manifest to mark
    /// it as a *referrer* of another manifest, identified by digest. See
    /// `cmd::sign`, the only producer of this field today.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject: Option<Descriptor>,
}

/// `application/vnd.cncf.model.config.v1+json` — the CNCF Model Format Spec
/// (<https://github.com/modelpack/model-spec>) config document. Mirrors the
/// `cncfModelConfig` struct in `go-shim/hf.go`, which builds the same shape
/// for HuggingFace/cloud-source pulls; this is the equivalent for `llmman
/// build`'s local-directory packaging path.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CncfConfigDescriptor {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CncfConfigConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CncfModelFs {
    #[serde(rename = "type")]
    pub fs_type: String,
    pub diff_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CncfModelConfig {
    pub descriptor: CncfConfigDescriptor,
    pub config: CncfConfigConfig,
    pub modelfs: CncfModelFs,
}

/// Summary of a locally stored image shown by `list`.
#[derive(Debug)]
pub struct ImageSummary {
    pub reference: String,
    pub digest: String,
    #[allow(dead_code)]
    pub media_type: String,
    pub size: u64,
    pub modified_at: Option<std::time::SystemTime>,
}

// ---------------------------------------------------------------------------
// OciStore
// ---------------------------------------------------------------------------

pub struct OciStore {
    root: PathBuf,
}

impl OciStore {
    /// Open (or create) an OCI layout store at `root`.
    pub fn open(root: impl AsRef<Path>) -> anyhow::Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        fs::create_dir_all(root.join("blobs").join("sha256"))?;

        // Write oci-layout marker if absent
        let marker = root.join("oci-layout");
        if !marker.exists() {
            fs::write(&marker, r#"{"imageLayoutVersion":"1.0.0"}"#)?;
        }
        // Create empty index if absent
        let index_path = root.join("index.json");
        if !index_path.exists() {
            let idx = Index::default();
            fs::write(&index_path, serde_json::to_string_pretty(&idx)?)?;
        }
        Ok(Self { root })
    }

    #[allow(dead_code)]
    pub fn root(&self) -> &Path {
        &self.root
    }

    // ------------------------------------------------------------------
    // Index
    // ------------------------------------------------------------------

    pub fn read_index(&self) -> anyhow::Result<Index> {
        let data = fs::read(self.root.join("index.json"))
            .context("read index.json")?;
        serde_json::from_slice(&data).context("parse index.json")
    }

    fn write_index(&self, idx: &Index) -> anyhow::Result<()> {
        let tmp = self.root.join("index.json.tmp");
        fs::write(&tmp, serde_json::to_string_pretty(idx)?)?;
        fs::rename(tmp, self.root.join("index.json"))?;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Blobs
    // ------------------------------------------------------------------

    fn blob_path(&self, digest: &str) -> anyhow::Result<PathBuf> {
        let (algo, hex) = split_digest(digest)?;
        Ok(self.root.join("blobs").join(algo).join(hex))
    }

    /// Write `data` as a blob.  Returns its `Descriptor`.
    pub fn write_blob(&self, media_type: &str, data: &[u8]) -> anyhow::Result<Descriptor> {
        let hex = hex::encode(Sha256::digest(data));
        let digest = format!("sha256:{}", hex);
        let path = self.root.join("blobs").join("sha256").join(&hex);
        if !path.exists() {
            let tmp = path.with_extension("tmp");
            fs::write(&tmp, data)?;
            fs::rename(tmp, &path)?;
        }
        Ok(Descriptor {
            media_type: media_type.into(),
            digest,
            size: data.len() as u64,
            annotations: None,
        })
    }

    /// Write a large file as a blob, streaming to avoid buffering the whole file.
    #[allow(dead_code)]
    pub fn write_blob_file(
        &self,
        media_type: &str,
        path: impl AsRef<Path>,
    ) -> anyhow::Result<Descriptor> {
        let path = path.as_ref();
        let mut hasher = Sha256::new();
        let mut size = 0u64;
        let tmp = self
            .root
            .join("blobs")
            .join("sha256")
            .join(format!("tmp-{}", std::process::id()));
        {
            let mut src = fs::File::open(path)
                .with_context(|| format!("open {}", path.display()))?;
            let mut dst = fs::File::create(&tmp)?;
            let mut buf = vec![0u8; 1 << 20]; // 1 MiB chunks
            loop {
                let n = src.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                hasher.update(&buf[..n]);
                dst.write_all(&buf[..n])?;
                size += n as u64;
            }
        }
        let hex = hex::encode(hasher.finalize());
        let digest = format!("sha256:{}", hex);
        let dest = self.root.join("blobs").join("sha256").join(&hex);
        if dest.exists() {
            fs::remove_file(&tmp)?;
        } else {
            fs::rename(tmp, &dest)?;
        }
        Ok(Descriptor {
            media_type: media_type.into(),
            digest,
            size,
            annotations: None,
        })
    }

    /// Read a blob's raw bytes.
    pub fn read_blob(&self, digest: &str) -> anyhow::Result<Vec<u8>> {
        fs::read(self.blob_path(digest)?).with_context(|| format!("read blob {}", digest))
    }

    // ------------------------------------------------------------------
    // Manifest helpers
    // ------------------------------------------------------------------

    pub fn write_manifest(&self, manifest: &Manifest) -> anyhow::Result<Descriptor> {
        let data = serde_json::to_vec(manifest)?;
        self.write_blob("application/vnd.oci.image.manifest.v1+json", &data)
    }

    pub fn read_manifest(&self, digest: &str) -> anyhow::Result<Manifest> {
        let data = self.read_blob(digest)?;
        serde_json::from_slice(&data).context("parse manifest")
    }

    // ------------------------------------------------------------------
    // Tag operations
    // ------------------------------------------------------------------

    /// Add a reference to `index.json`, replacing any prior entry with the same ref name.
    /// The full `reference` string is stored in the annotation so `list` shows it verbatim.
    pub fn tag(&self, mut desc: Descriptor, reference: &str) -> anyhow::Result<()> {
        let mut ann = desc.annotations.take().unwrap_or_default();
        ann.insert(
            "org.opencontainers.image.ref.name".into(),
            reference.to_string(),
        );
        desc.annotations = Some(ann);

        let mut idx = self.read_index()?;
        let mut replaced = false;
        for entry in &mut idx.manifests {
            if ref_matches(entry, reference) {
                *entry = desc.clone();
                replaced = true;
                break;
            }
        }
        if !replaced {
            idx.manifests.push(desc);
        }
        self.write_index(&idx)
    }

    /// Find the descriptor for `reference` in the index.
    /// Matches either the full stored reference or (as fallback) just its tag component.
    pub fn find(&self, reference: &str) -> anyhow::Result<Descriptor> {
        let idx = self.read_index()?;
        idx.manifests
            .into_iter()
            .find(|m| ref_matches(m, reference))
            .ok_or_else(|| anyhow!("image not found: {}", reference))
    }

    /// The real total size of an image: the sum of its layer sizes, not
    /// `desc.size` (a `Descriptor`'s own `size` field is the *manifest
    /// blob's* size — just a few hundred bytes of JSON — not the image's
    /// actual content size). Falls back to `desc.size` if the manifest
    /// can't be read, so callers still get *something* rather than an
    /// error over what's only ever used for display.
    pub fn total_size(&self, desc: &Descriptor) -> u64 {
        self.read_manifest(&desc.digest)
            .map(|manifest| manifest.layers.iter().map(|l| l.size).sum())
            .unwrap_or(desc.size)
    }

    // ------------------------------------------------------------------
    // List / Remove
    // ------------------------------------------------------------------

    pub fn list(&self) -> anyhow::Result<Vec<ImageSummary>> {
        let idx = self.read_index()?;
        Ok(idx
            .manifests
            .into_iter()
            .map(|m| {
                let reference = m
                    .annotations
                    .as_ref()
                    .and_then(|a| a.get("org.opencontainers.image.ref.name"))
                    .cloned()
                    .unwrap_or_else(|| m.digest.clone());
                let modified_at = self
                    .blob_path(&m.digest)
                    .ok()
                    .and_then(|p| fs::metadata(p).ok())
                    .and_then(|meta| meta.modified().ok());
                let size = self.total_size(&m);
                ImageSummary {
                    reference,
                    digest: m.digest,
                    media_type: m.media_type,
                    size,
                    modified_at,
                }
            })
            .collect())
    }

    /// Remove a manifest from the index by reference.  Does not GC blobs.
    pub fn remove(&self, reference: &str) -> anyhow::Result<()> {
        let mut idx = self.read_index()?;
        let before = idx.manifests.len();
        idx.manifests.retain(|m| !ref_matches(m, reference));
        if idx.manifests.len() == before {
            return Err(anyhow!("image not found: {}", reference));
        }
        self.write_index(&idx)
    }

    // ------------------------------------------------------------------
    // Build helpers
    // ------------------------------------------------------------------

    /// Package all files in `src_dir` as a CNCF Model Format Spec
    /// (<https://github.com/modelpack/model-spec>) OCI artifact stored in
    /// this layout. Each file becomes one uncompressed tar layer, classified
    /// into the appropriate `application/vnd.cncf.model.*` media type by
    /// extension (mirroring `classifyFile` in `go-shim/uri_sources.go`, the
    /// equivalent classifier used when pulling from HuggingFace/cloud
    /// sources), with `org.cncf.model.filepath` recording its path.
    /// Returns the manifest descriptor.
    pub fn build(
        &self,
        src_dir: impl AsRef<Path>,
        reference: &str,
        labels: &std::collections::HashMap<String, String>,
    ) -> anyhow::Result<Descriptor> {
        use walkdir::WalkDir;

        let src_dir = src_dir.as_ref();
        let mut layers: Vec<Descriptor> = Vec::new();
        let mut format: Option<&'static str> = None;

        // One layer per file (uncompressed tar, filename preserved via annotations)
        for entry in WalkDir::new(src_dir).follow_links(true) {
            let entry = entry?;
            if !entry.file_type().is_file() {
                continue;
            }
            let rel = entry
                .path()
                .strip_prefix(src_dir)
                .unwrap()
                .to_string_lossy()
                .into_owned();

            let media_type = classify_model_layer(&rel);
            if media_type == WEIGHT_TAR_MEDIA_TYPE {
                let lower = rel.to_lowercase();
                if lower.ends_with(".gguf") || lower.ends_with(".ggml") {
                    format = Some("gguf");
                } else if lower.ends_with(".safetensors") && format.is_none() {
                    format = Some("safetensors");
                }
            }

            // Build a minimal tar with a single entry
            let tar_data = make_single_file_tar(entry.path(), &rel)?;
            let mut desc = self.write_blob(media_type, &tar_data)?;
            desc.annotations = Some({
                let mut m = std::collections::HashMap::new();
                m.insert("org.cncf.model.filepath".into(), rel.clone());
                m.insert("org.opencontainers.image.title".into(), rel);
                m
            });
            layers.push(desc);
        }

        if layers.is_empty() {
            return Err(anyhow!("no files found in {}", src_dir.display()));
        }

        // `application/vnd.cncf.model.config.v1+json` config. Since every
        // layer above is an uncompressed tar, each layer's DiffID (hash of
        // its uncompressed content) is simply its own digest.
        let cncf_config = CncfModelConfig {
            descriptor: CncfConfigDescriptor {
                created_at: Some(chrono::Utc::now().to_rfc3339()),
            },
            config: CncfConfigConfig {
                format: format.map(str::to_string),
            },
            modelfs: CncfModelFs {
                fs_type: "layers".into(),
                diff_ids: layers.iter().map(|l| l.digest.clone()).collect(),
            },
        };
        let config_data = serde_json::to_vec(&cncf_config)?;
        let config_desc = self.write_blob(
            "application/vnd.cncf.model.config.v1+json",
            &config_data,
        )?;

        // `--label key=value` pairs have no dedicated slot in the model
        // config schema, so they're carried as manifest annotations instead.
        let manifest_annotations = if labels.is_empty() {
            None
        } else {
            Some(labels.clone())
        };

        // Manifest
        let manifest = Manifest {
            schema_version: 2,
            media_type: "application/vnd.oci.image.manifest.v1+json".into(),
            artifact_type: Some("application/vnd.cncf.model.manifest.v1+json".into()),
            config: config_desc,
            layers,
            annotations: manifest_annotations,
            subject: None,
        };
        let manifest_desc = self.write_manifest(&manifest)?;
        self.tag(manifest_desc.clone(), reference)?;
        Ok(manifest_desc)
    }
}

// ---------------------------------------------------------------------------
// CNCF Model Format Spec layer classification
// ---------------------------------------------------------------------------

const WEIGHT_TAR_MEDIA_TYPE: &str = "application/vnd.cncf.model.weight.v1.tar";
const WEIGHT_CONFIG_TAR_MEDIA_TYPE: &str = "application/vnd.cncf.model.weight.config.v1.tar";
const DOC_TAR_MEDIA_TYPE: &str = "application/vnd.cncf.model.doc.v1.tar";
const CODE_TAR_MEDIA_TYPE: &str = "application/vnd.cncf.model.code.v1.tar";

/// Maps a file's relative path to the appropriate CNCF model layer media
/// type by extension, mirroring `classifyFile` in `go-shim/uri_sources.go`
/// (used for HuggingFace/cloud-source pulls). Uses the `.tar` variants
/// because `build()` wraps each file in its own uncompressed tar archive,
/// rather than the `.raw` variants the Go side uses for un-archived blobs.
fn classify_model_layer(rel_path: &str) -> &'static str {
    let lower = rel_path.to_lowercase();
    let base = Path::new(&lower)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or(lower.as_str());
    let ext = Path::new(base).extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "safetensors" | "bin" | "pt" | "pth" | "gguf" | "ggml" | "ot" | "engine" | "trt"
        | "onnx" => WEIGHT_TAR_MEDIA_TYPE,
        "json" | "yaml" | "yml" | "toml" | "ini" | "cfg" | "conf" | "model" | "tiktoken"
        | "vocab" | "merges" | "spm" => WEIGHT_CONFIG_TAR_MEDIA_TYPE,
        "txt" => {
            if base.contains("vocab") || base.contains("merges") {
                WEIGHT_CONFIG_TAR_MEDIA_TYPE
            } else {
                DOC_TAR_MEDIA_TYPE
            }
        }
        "py" | "sh" | "js" | "ts" => CODE_TAR_MEDIA_TYPE,
        "md" | "rst" | "pdf" => DOC_TAR_MEDIA_TYPE,
        _ => {
            // README, LICENSE, and anything unrecognized default to doc,
            // same as the Go classifier's fallback.
            DOC_TAR_MEDIA_TYPE
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns true if the descriptor's ref annotation matches `reference`.
///
/// Three strategies are tried in order:
/// 1. Exact match (`"reg/repo:tag"` == `"reg/repo:tag"`)
/// 2. Tag-only match (`"latest"` matches `"reg/repo:latest"`)
/// 3. Tagless match: if `reference` has no tag, implicitly append `:latest`
///    so `"reg/repo"` matches `"reg/repo:latest"`
fn ref_matches(desc: &Descriptor, reference: &str) -> bool {
    let stored = match desc
        .annotations
        .as_ref()
        .and_then(|a| a.get("org.opencontainers.image.ref.name"))
    {
        Some(s) => s,
        None => return false,
    };

    if stored == reference || tag_from_ref(stored) == reference {
        return true;
    }

    // The podman backend's own OCI-layout writer can only ever record a
    // bare tag here, never a full reference: go-shim/backend_podman.go's
    // pullToLayout builds the destination as `oci:<layoutDir>:<tag>`
    // (via tagFromRef), and go.podman.io/image's oci transport uses that
    // trailing component verbatim as the stored annotation
    // (oci/layout/oci_dest.go's PutManifest) — there's no repository
    // component in that reference shape at all to also record. This is
    // architecturally different from the docker/containerd backend's own
    // writer (shared_oci.go's updateIndex), which always stores the
    // *full* reference it was called with, matched by the first check
    // above. A `stored` value with no '/' is that shape: match it
    // against the incoming reference's own tag instead of requiring an
    // (impossible, for this backend) full-reference match.
    //
    // Known limitation, not fixed here: two different repositories
    // pulled with the podman backend that happen to share the same bare
    // tag are indistinguishable once stored this way — podman's
    // OCI-layout writer has no way to record more than a tag per entry
    // to begin with, regardless of what this lookup does.
    if !stored.contains('/') && stored.as_str() == tag_from_ref(reference) {
        return true;
    }

    // If reference carries no tag (no ':' after the last '/'), try `:latest`.
    let after_slash = &reference[reference.rfind('/').unwrap_or(0)..];
    if !after_slash.contains(':') && stored.as_str() == format!("{reference}:latest") {
        return true;
    }

    false
}

fn split_digest(digest: &str) -> anyhow::Result<(&str, &str)> {
    let mut parts = digest.splitn(2, ':');
    let algo = parts.next().ok_or_else(|| anyhow!("invalid digest: {}", digest))?;
    let hex = parts.next().ok_or_else(|| anyhow!("invalid digest: {}", digest))?;
    Ok((algo, hex))
}

pub fn tag_from_ref(reference: &str) -> &str {
    if let Some(pos) = reference.rfind(':') {
        if pos > reference.rfind('/').unwrap_or(0) {
            return &reference[pos + 1..];
        }
    }
    "latest"
}

/// Build an in-memory uncompressed tar archive containing a single file.
fn make_single_file_tar(path: &Path, name: &str) -> anyhow::Result<Vec<u8>> {
    let file_data = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut buf = Vec::new();
    {
        let mut archive = tar::Builder::new(&mut buf);
        let mut header = tar::Header::new_gnu();
        header.set_path(name)?;
        header.set_size(file_data.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        archive.append(&header, file_data.as_slice())?;
        archive.finish()?;
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc_with_ref(ref_name: &str) -> Descriptor {
        let mut ann = std::collections::HashMap::new();
        ann.insert("org.opencontainers.image.ref.name".to_string(), ref_name.to_string());
        Descriptor {
            media_type: "application/vnd.oci.image.manifest.v1+json".into(),
            digest: "sha256:deadbeef".into(),
            size: 123,
            annotations: Some(ann),
        }
    }

    /// The docker/containerd backend's own writer (shared_oci.go's
    /// updateIndex) always stores the full reference it was called with.
    #[test]
    fn ref_matches_a_full_reference_stored_verbatim() {
        let d = desc_with_ref("docker.io/ai/qwen3.5:0.8b");
        assert!(ref_matches(&d, "docker.io/ai/qwen3.5:0.8b"));
        assert!(!ref_matches(&d, "docker.io/ai/qwen3.5:1.5b"));
        assert!(!ref_matches(&d, "docker.io/ai/other:0.8b"));
    }

    /// Regression test: the podman backend's OCI-layout writer
    /// (go.podman.io/image, via the `oci:<dir>:<tag>` reference shape —
    /// see backend_podman.go's pullToLayout) can only ever store a bare
    /// tag here, never a full reference — a real pull otherwise
    /// succeeded but the model could never be found again afterward
    /// (`resolve model ...: image not found`) until this matched.
    #[test]
    fn ref_matches_a_bare_tag_stored_by_the_podman_backend() {
        let d = desc_with_ref("0.8b");
        assert!(ref_matches(&d, "docker.io/ai/qwen3.5:0.8b"));
        assert!(ref_matches(&d, "0.8b"));
        assert!(!ref_matches(&d, "docker.io/ai/qwen3.5:1.5b"));
    }

    #[test]
    fn ref_matches_defaults_a_tagless_reference_to_latest() {
        let d = desc_with_ref("docker.io/ai/qwen3.5:latest");
        assert!(ref_matches(&d, "docker.io/ai/qwen3.5"));
    }

    #[test]
    fn ref_matches_returns_false_without_a_ref_name_annotation() {
        let d = Descriptor {
            media_type: "application/vnd.oci.image.manifest.v1+json".into(),
            digest: "sha256:deadbeef".into(),
            size: 123,
            annotations: None,
        };
        assert!(!ref_matches(&d, "docker.io/ai/qwen3.5:0.8b"));
    }

    #[test]
    fn tag_from_ref_extracts_the_tag_after_the_last_slash() {
        assert_eq!(tag_from_ref("docker.io/ai/qwen3.5:0.8b"), "0.8b");
        assert_eq!(tag_from_ref("docker.io/ai/qwen3.5"), "latest");
        // A bare tag with no slash and no colon (podman's own stored
        // shape) has nothing to extract from — falls back to "latest",
        // which is why ref_matches needs its own dedicated bare-tag
        // check above rather than reusing this for both directions.
        assert_eq!(tag_from_ref("0.8b"), "latest");
    }

    /// Ported from ollama's types/model/name_test.go
    /// ("host:port/namespace/model:tag" and the tagless default): a colon
    /// inside the host component must never be mistaken for a tag
    /// separator, and a reference without a tag defaults to "latest".
    #[test]
    fn tag_from_ref_ignores_a_port_in_the_host_component() {
        assert_eq!(tag_from_ref("example.com:5000/ns/model"), "latest");
        assert_eq!(tag_from_ref("example.com:5000/ns/model:tag"), "tag");
        assert_eq!(tag_from_ref("localhost:11434/library/mistral:7b"), "7b");
    }
}
