//! HuggingFace Hub API calls and file-selection logic. Originally
//! ported from the Go shim's `hfFetchModelInfo`/`hfFetchFiles`/
//! `selectGGUF`/`selectMMProj`/`selectLicenseFile`/
//! `safetensorsMediaType`, since deleted — this is the only
//! implementation now.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use serde::Deserialize;

use super::client::HttpStatusError;
use super::oci;

/// One entry from the HuggingFace tree API — mirrors `hfFile`.
#[derive(Debug, Clone, Deserialize)]
pub struct HfFile {
    pub path: String,
    #[serde(default)]
    pub size: i64,
    #[serde(default, rename = "type")]
    pub kind: String, // "file" or "directory"
}

/// Issues an authenticated GET and decodes JSON, retrying transient
/// failures with the shared backoff budget — mirrors `hfGet`.
async fn get_json<T: serde::de::DeserializeOwned>(
    client: &reqwest::Client,
    url: &str,
    token: Option<&str>,
) -> Result<T> {
    let body = super::client::retry(&format!("GET {url}"), || async {
        let mut req = client.get(url);
        if let Some(t) = token {
            req = req.bearer_auth(t);
        }
        let resp = req.send().await.with_context(|| format!("GET {url}"))?;
        let status = resp.status();
        if status != reqwest::StatusCode::OK {
            let headers = resp.headers().clone();
            return Err(
                HttpStatusError::new(format!("GET {url}"), status.as_u16(), &headers).into(),
            );
        }
        resp.bytes()
            .await
            .with_context(|| format!("read body of GET {url}"))
    })
    .await?;
    serde_json::from_slice(&body).with_context(|| format!("decode JSON from {url}"))
}

/// The subset of `GET /api/models/{owner}/{repo}` this needs — mirrors `hfModelInfo`.
#[derive(Debug, Deserialize, Default)]
pub struct ModelInfo {
    #[serde(default)]
    sha: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default, rename = "cardData")]
    card_data: CardData,
}

#[derive(Debug, Deserialize, Default)]
struct CardData {
    #[serde(default)]
    license: String,
}

impl ModelInfo {
    /// The commit SHA to pin resolve URLs to, falling back to "main".
    pub fn commit(&self) -> &str {
        if self.sha.is_empty() {
            "main"
        } else {
            &self.sha
        }
    }

    /// Best-effort SPDX license expression, or `None` if the repo
    /// doesn't declare one usable at all.
    pub fn license(&self) -> Option<String> {
        if !self.card_data.license.is_empty() {
            if let Some(id) = normalize_spdx_license(&self.card_data.license) {
                return Some(id);
            }
        }
        for tag in &self.tags {
            if let Some(slug) = tag.strip_prefix("license:") {
                if !slug.is_empty() {
                    if let Some(id) = normalize_spdx_license(slug) {
                        return Some(id);
                    }
                }
            }
        }
        None
    }
}

/// Maps a HuggingFace license slug to its SPDX expression — mirrors
/// `spdxLicenseIDs`/`normalizeSPDXLicense`. "other"/"unknown" are HF's
/// catch-alls for "not a real license identifier" and report as unusable
/// (`None`) rather than fabricating a bogus SPDX id. Anything else not
/// listed falls through unchanged.
fn normalize_spdx_license(slug: &str) -> Option<String> {
    let slug = slug.trim().to_lowercase();
    match slug.as_str() {
        "apache-2.0" => Some("Apache-2.0".to_string()),
        "mit" => Some("MIT".to_string()),
        "bsd-2-clause" => Some("BSD-2-Clause".to_string()),
        "bsd-3-clause" => Some("BSD-3-Clause".to_string()),
        "gpl-2.0" => Some("GPL-2.0-only".to_string()),
        "gpl-3.0" => Some("GPL-3.0-only".to_string()),
        "lgpl-2.1" => Some("LGPL-2.1-only".to_string()),
        "lgpl-3.0" => Some("LGPL-3.0-only".to_string()),
        "mpl-2.0" => Some("MPL-2.0".to_string()),
        "cc-by-4.0" => Some("CC-BY-4.0".to_string()),
        "cc-by-sa-4.0" => Some("CC-BY-SA-4.0".to_string()),
        "cc0-1.0" => Some("CC0-1.0".to_string()),
        "other" | "unknown" => None,
        other => Some(other.to_string()),
    }
}

pub async fn fetch_model_info(
    client: &reqwest::Client,
    endpoint: &str,
    owner: &str,
    repo: &str,
    token: Option<&str>,
) -> Result<ModelInfo> {
    let url = format!("{endpoint}api/models/{owner}/{repo}");
    get_json(client, &url, token).await.context("HF model info")
}

pub async fn fetch_files(
    client: &reqwest::Client,
    endpoint: &str,
    owner: &str,
    repo: &str,
    commit: &str,
    token: Option<&str>,
) -> Result<Vec<HfFile>> {
    let url = format!("{endpoint}api/models/{owner}/{repo}/tree/{commit}?recursive=true");
    get_json(client, &url, token).await.context("HF file list")
}

// ---------------------------------------------------------------------------
// GGUF file selection (mirrors llama.cpp find_best_model)
// ---------------------------------------------------------------------------

const QUANT_PREFERENCE: &[&str] = &[
    "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0", "Q4_0", "Q6_K", "Q2_K",
];

/// True for GGUF files that are primary model weights (not mmproj
/// projectors or imatrix importance files).
fn is_model_gguf(path: &str) -> bool {
    let lower = path.to_lowercase();
    lower.ends_with(".gguf") && !lower.contains("mmproj") && !lower.contains("imatrix")
}

/// Parses llama.cpp's gguf-split naming convention —
/// "&lt;name&gt;-NNNNN-of-MMMMM.gguf" — without a regex dependency.
/// Returns `(prefix, index, total)`. Case-insensitive (matches
/// `is_model_gguf`'s own lowercasing), so a split isn't silently missed
/// over a `.GGUF`/`-OF-` casing difference.
fn parse_gguf_shard(path: &str) -> Option<(String, u32, u32)> {
    let base = path.rsplit('/').next().unwrap_or(path).to_lowercase();
    let base = base.strip_suffix(".gguf")?;
    let (rest, total_str) = base.rsplit_once("-of-")?;
    let total: u32 = total_str.parse().ok()?;
    let (prefix, index_str) = rest.rsplit_once('-')?;
    let index: u32 = index_str.parse().ok()?;
    Some((prefix.to_string(), index, total))
}

fn dirname(path: &str) -> &str {
    path.rsplit_once('/').map(|(d, _)| d).unwrap_or("")
}

/// Returns every shard of the same multi-part split as `chosen`, in
/// shard order — a manifest built from only some of a split's shards
/// silently produces a model no GGUF-reading runtime can actually load,
/// so this errors rather than return an incomplete set. Returns
/// `[chosen]` unchanged for a file that isn't part of a split.
///
/// Groups by directory as well as name/total: some repos put the same
/// split basenames under multiple per-quantization subdirectories, and
/// grouping by basename alone could merge or miscount across them.
fn gguf_shards(models: &[HfFile], chosen: &HfFile) -> Result<Vec<HfFile>> {
    let Some((prefix, _, total)) = parse_gguf_shard(&chosen.path) else {
        return Ok(vec![chosen.clone()]);
    };
    let chosen_dir = dirname(&chosen.path);
    let mut shards: Vec<(u32, HfFile)> = models
        .iter()
        .filter_map(|f| {
            let (p, idx, t) = parse_gguf_shard(&f.path)?;
            (p == prefix && t == total && dirname(&f.path) == chosen_dir).then(|| (idx, f.clone()))
        })
        .collect();
    shards.sort_by_key(|(idx, _)| *idx);
    shards.dedup_by_key(|(idx, _)| *idx);
    if shards.len() as u32 != total {
        anyhow::bail!(
            "incomplete GGUF split {prefix:?}: found {} of {total} shards",
            shards.len()
        );
    }
    Ok(shards.into_iter().map(|(_, f)| f).collect())
}

/// Picks the best GGUF quant from the file listing, returning every
/// shard of a multi-part split together. `tag` is the user-supplied
/// quantization hint (e.g. "Q4_K_M") or empty for auto.
pub fn select_gguf(files: &[HfFile], tag: &str) -> Result<Vec<HfFile>> {
    let models: Vec<HfFile> = files
        .iter()
        .filter(|f| f.kind == "file" && is_model_gguf(&f.path))
        .cloned()
        .collect();
    if models.is_empty() {
        anyhow::bail!("no GGUF model files found in repository");
    }

    if !tag.is_empty() && tag != "latest" {
        let upper = tag.to_uppercase();
        if let Some(f) = models
            .iter()
            .find(|f| f.path.to_uppercase().contains(&upper))
        {
            return gguf_shards(&models, f);
        }
        let list: String = models.iter().map(|f| format!("  {}\n", f.path)).collect();
        anyhow::bail!("no GGUF file matching {tag:?} found; available:\n{list}");
    }

    for pref in QUANT_PREFERENCE {
        if let Some(f) = models.iter().find(|f| f.path.to_uppercase().contains(pref)) {
            return gguf_shards(&models, f);
        }
    }

    // Fallback: smallest file (most compressed).
    let smallest = models
        .iter()
        .min_by_key(|f| f.size)
        .expect("models is non-empty");
    gguf_shards(&models, smallest)
}

const MMPROJ_PREFERENCE: &[&str] = &["F16", "BF16", "F32"];

/// Returns the repo's multimodal projector file, if it has one.
pub fn select_mmproj(files: &[HfFile]) -> Option<HfFile> {
    let candidates: Vec<&HfFile> = files
        .iter()
        .filter(|f| {
            f.kind == "file"
                && f.path.to_lowercase().contains("mmproj")
                && f.path.to_lowercase().ends_with(".gguf")
        })
        .collect();
    for pref in MMPROJ_PREFERENCE {
        for f in &candidates {
            let base = f.path.rsplit('/').next().unwrap_or(&f.path).to_uppercase();
            if base == format!("{pref}.GGUF") || base.ends_with(&format!("-{pref}.GGUF")) {
                return Some((*f).clone());
            }
        }
    }
    candidates.first().map(|f| (*f).clone())
}

const LICENSE_FILENAMES: &[&str] = &["LICENSE", "LICENSE.txt", "LICENSE.md"];

/// Returns the repo's root-level LICENSE file, if it has one.
pub fn select_license_file(files: &[HfFile]) -> Option<HfFile> {
    for want in LICENSE_FILENAMES {
        if let Some(f) = files
            .iter()
            .find(|f| f.kind == "file" && f.path.eq_ignore_ascii_case(want))
        {
            return Some(f.clone());
        }
    }
    None
}

/// Maps a file extension to the appropriate CNCF model layer media type
/// — mirrors `safetensorsMediaType`. ".jinja" is config, not doc: many
/// repos ship a standalone chat_template.jinja, and doc-type layers are
/// dropped before serving, which would silently hide the chat template.
pub fn safetensors_media_type(path: &str) -> &'static str {
    let ext = path.rsplit('.').next().unwrap_or("").to_lowercase();
    match ext.as_str() {
        "safetensors" | "bin" | "pt" | "pth" => oci::MEDIA_TYPE_MODEL_WEIGHT_RAW,
        "json" | "model" | "txt" | "tiktoken" | "jinja" => oci::MEDIA_TYPE_MODEL_WEIGHT_CONFIG_RAW,
        _ => oci::MEDIA_TYPE_MODEL_DOC_RAW,
    }
}

/// True for files that belong in a local model directory. Deliberately
/// safetensors-only: a GGUF repo is selected by `select_gguf` in a
/// separate pass, so weight formats other than safetensors must not be
/// swept up here. `crate::sources::should_pack` is the wider equivalent
/// for sources that have no such second pass.
fn should_download_safetensors(path: &str) -> bool {
    let base = path.rsplit('/').next().unwrap_or(path).to_lowercase();
    if base.starts_with('.') {
        return false;
    }
    let ext = base.rsplit('.').next().unwrap_or("");
    if matches!(
        ext,
        "safetensors" | "bin" | "pt" | "pth" | "json" | "model" | "txt" | "tiktoken" | "jinja"
    ) {
        return true;
    }
    matches!(
        base.as_str(),
        "readme.md" | "license" | "licence" | "license.txt" | "licence.txt"
    )
}

/// Filters `files` down to the plain files `should_download_safetensors`
/// accepts, ignoring directories.
pub fn select_downloadable_hf_files(files: &[HfFile]) -> Vec<HfFile> {
    files
        .iter()
        .filter(|f| f.kind == "file" && should_download_safetensors(&f.path))
        .cloned()
        .collect()
}

/// Splits a (possibly `:latest`-normalized) HF reference
/// "host/owner/repo[:tag]" into its four components.
pub fn parse_hf_ref(reference: &str) -> Result<(String, String, String, String)> {
    let last_colon = reference.rfind(':').map(|i| i as isize).unwrap_or(-1);
    let last_slash = reference.rfind('/').map(|i| i as isize).unwrap_or(-1);
    let (rest, tag) = if last_colon > last_slash {
        (
            &reference[..last_colon as usize],
            reference[last_colon as usize + 1..].to_string(),
        )
    } else {
        (reference, String::new())
    };
    let parts: Vec<&str> = rest.splitn(3, '/').collect();
    let [host, owner, repo] = parts[..] else {
        anyhow::bail!("invalid HuggingFace reference {reference:?}: expected host/owner/repo");
    };
    Ok((host.to_string(), owner.to_string(), repo.to_string(), tag))
}

#[allow(dead_code)]
pub type Annotations = BTreeMap<String, String>;

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str, size: i64) -> HfFile {
        HfFile {
            path: path.to_string(),
            size,
            kind: "file".to_string(),
        }
    }

    #[test]
    fn select_gguf_prefers_q4_k_m() {
        let files = vec![file("model-Q8_0.gguf", 100), file("model-Q4_K_M.gguf", 50)];
        let got = select_gguf(&files, "").unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].path, "model-Q4_K_M.gguf");
    }

    #[test]
    fn select_gguf_falls_back_to_smallest() {
        let files = vec![
            file("model-weird1.gguf", 100),
            file("model-weird2.gguf", 50),
        ];
        let got = select_gguf(&files, "").unwrap();
        assert_eq!(got[0].path, "model-weird2.gguf");
    }

    #[test]
    fn select_gguf_excludes_mmproj_and_imatrix() {
        let files = vec![
            file("mmproj-F16.gguf", 10),
            file("model.imatrix.gguf", 10),
            file("model-Q4_K_M.gguf", 50),
        ];
        let got = select_gguf(&files, "").unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].path, "model-Q4_K_M.gguf");
    }

    #[test]
    fn select_gguf_by_explicit_tag() {
        let files = vec![file("model-Q4_K_M.gguf", 50), file("model-Q8_0.gguf", 100)];
        let got = select_gguf(&files, "Q8_0").unwrap();
        assert_eq!(got[0].path, "model-Q8_0.gguf");
    }

    #[test]
    fn select_gguf_returns_every_shard_of_a_split_in_order() {
        let files = vec![
            file("model-Q4_K_M-00002-of-00003.gguf", 10),
            file("model-Q4_K_M-00001-of-00003.gguf", 10),
            file("model-Q4_K_M-00003-of-00003.gguf", 10),
            file("model-Q8_0-00001-of-00002.gguf", 10),
        ];
        let got = select_gguf(&files, "Q4_K_M").unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].path, "model-Q4_K_M-00001-of-00003.gguf");
        assert_eq!(got[1].path, "model-Q4_K_M-00002-of-00003.gguf");
        assert_eq!(got[2].path, "model-Q4_K_M-00003-of-00003.gguf");
    }

    #[test]
    fn select_gguf_errors_on_no_gguf_files() {
        let files = vec![file("config.json", 10)];
        assert!(select_gguf(&files, "").is_err());
    }

    #[test]
    fn select_mmproj_prefers_f16() {
        let files = vec![file("mmproj-BF16.gguf", 10), file("mmproj-F16.gguf", 10)];
        assert_eq!(select_mmproj(&files).unwrap().path, "mmproj-F16.gguf");
    }

    #[test]
    fn select_mmproj_absent_returns_none() {
        assert!(select_mmproj(&[file("model.gguf", 10)]).is_none());
    }

    #[test]
    fn select_license_file_case_insensitive() {
        let files = vec![file("license", 10)];
        assert!(select_license_file(&files).is_some());
    }

    #[test]
    fn normalize_spdx_license_maps_known_slugs() {
        assert_eq!(
            normalize_spdx_license("apache-2.0"),
            Some("Apache-2.0".to_string())
        );
        assert_eq!(normalize_spdx_license("other"), None);
        assert_eq!(normalize_spdx_license("unknown"), None);
        assert_eq!(
            normalize_spdx_license("some-custom-license"),
            Some("some-custom-license".to_string())
        );
    }

    #[test]
    fn parse_hf_ref_splits_host_owner_repo_tag() {
        let (host, owner, repo, tag) = parse_hf_ref("huggingface.co/owner/repo:Q4_K_M").unwrap();
        assert_eq!(
            (host.as_str(), owner.as_str(), repo.as_str(), tag.as_str()),
            ("huggingface.co", "owner", "repo", "Q4_K_M")
        );
    }

    #[test]
    fn parse_hf_ref_without_tag() {
        let (host, owner, repo, tag) = parse_hf_ref("huggingface.co/owner/repo").unwrap();
        assert_eq!(
            (host.as_str(), owner.as_str(), repo.as_str(), tag.as_str()),
            ("huggingface.co", "owner", "repo", "")
        );
    }

    #[test]
    fn parse_hf_ref_rejects_malformed_ref() {
        assert!(parse_hf_ref("not-enough-parts").is_err());
    }

    #[test]
    fn safetensors_media_type_classifies_chat_template_jinja_as_config() {
        assert_eq!(
            safetensors_media_type("chat_template.jinja"),
            oci::MEDIA_TYPE_MODEL_WEIGHT_CONFIG_RAW
        );
    }
}
