//! Top-level HuggingFace pull — the Rust port of go-shim/hf.go's
//! `pullHF`/`pullHFSafetensors`/`downloadHFBlob`, now writing straight
//! into the local OCI layout without ever calling into the Go shim.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use sha2::{Digest as _, Sha256};

use super::api::{self, HfFile};
use super::download::{self, FetchRequest};
use super::oci::{self, Descriptor, ModelMeta};
use super::progress;
use crate::xet_fetch::XetFileRef;

/// Pulls `reference` (already stripped of any `hf://`/`huggingface://`
/// scheme prefix) into `layout_dir`. `progress_key` is the exact,
/// original ref the daemon's `/api/pull` handler was given, for
/// `progress::poll` — see that module's own doc comment.
pub async fn pull(reference: &str, layout_dir: &Path, progress_key: &str) -> Result<()> {
    let _guard = progress::DoneGuard(progress_key);
    let (host, owner, repo, tag) = api::parse_hf_ref(reference)?;

    oci::ensure_layout(layout_dir)?;
    if oci::report_cached(layout_dir, reference) {
        return Ok(());
    }

    let endpoint = super::hf_endpoint(&host);
    let token = super::token();
    let api_client = super::api_client()?;
    let dl_client = super::download_client()?;
    let head_client = super::head_client()?;

    progress::set_status(progress_key, "pulling");
    let info = api::fetch_model_info(&api_client, &endpoint, &owner, &repo, token.as_deref())
        .await
        .context("HF model info")?;
    let commit = info.commit().to_string();
    let mut meta = ModelMeta {
        licenses: info.license().into_iter().collect(),
        ..Default::default()
    };

    let files = api::fetch_files(
        &api_client,
        &endpoint,
        &owner,
        &repo,
        &commit,
        token.as_deref(),
    )
    .await
    .context("HF file list")?;

    let manifest_desc = match api::select_gguf(&files, &tag) {
        Ok(shards) => {
            meta.format = "gguf".to_string();
            let filepath_annotation = if shards.len() == 1 {
                basename(&shards[0].path)
            } else {
                String::new()
            };

            let mut layers = Vec::new();
            for f in &shards {
                layers.push(
                    download_layer(
                        &dl_client,
                        &head_client,
                        &endpoint,
                        &owner,
                        &repo,
                        &commit,
                        f,
                        token.as_deref(),
                        layout_dir,
                        progress_key,
                    )
                    .await?,
                );
            }
            if let Some(mmproj) = api::select_mmproj(&files) {
                layers.push(
                    download_layer(
                        &dl_client,
                        &head_client,
                        &endpoint,
                        &owner,
                        &repo,
                        &commit,
                        &mmproj,
                        token.as_deref(),
                        layout_dir,
                        progress_key,
                    )
                    .await?,
                );
                meta.vision = true;
            }
            if let Some(lic) = api::select_license_file(&files) {
                let mut d = download_layer(
                    &dl_client,
                    &head_client,
                    &endpoint,
                    &owner,
                    &repo,
                    &commit,
                    &lic,
                    token.as_deref(),
                    layout_dir,
                    progress_key,
                )
                .await?;
                d.media_type = oci::MEDIA_TYPE_MODEL_DOC_RAW.to_string();
                layers.push(d);
            }
            oci::build_cncf_manifest(
                layout_dir,
                &meta,
                &format!("{owner}/{repo}"),
                &filepath_annotation,
                layers,
            )?
        }
        // Only fall back to safetensors when the repo has no GGUF files
        // at all — a user-requested tag/quant that doesn't exist should
        // be a hard error, not a silent format switch.
        Err(e) if !tag.is_empty() && tag != "latest" => return Err(e),
        Err(_) => {
            meta.format = "safetensors".to_string();
            let to_download = api::select_downloadable_hf_files(&files);
            if to_download.is_empty() {
                anyhow::bail!("no model files found in repository {owner}/{repo}");
            }
            let mut layers = Vec::new();
            for f in &to_download {
                let mut d = download_layer(
                    &dl_client,
                    &head_client,
                    &endpoint,
                    &owner,
                    &repo,
                    &commit,
                    f,
                    token.as_deref(),
                    layout_dir,
                    progress_key,
                )
                .await?;
                d.media_type = api::safetensors_media_type(&f.path).to_string();
                d.annotations = Some(BTreeMap::from([(
                    oci::ANNOTATION_FILEPATH.to_string(),
                    f.path.clone(),
                )]));
                layers.push(d);
            }
            oci::build_cncf_manifest(layout_dir, &meta, &format!("{owner}/{repo}"), "", layers)?
        }
    };

    oci::write_manifest_ref(layout_dir, reference, manifest_desc)
}

fn basename(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_string()
}

/// Downloads one file straight into `layout_dir`'s content-addressed
/// blob store, returning its descriptor — mirrors `downloadHFBlob`.
/// Unlike the Go version, a retry always restarts this file from byte
/// zero rather than resuming a local `.part` file; see this migration's
/// own PR description for why that trade-off was accepted.
#[allow(clippy::too_many_arguments)]
async fn download_layer(
    client: &reqwest::Client,
    head_client: &reqwest::Client,
    endpoint: &str,
    owner: &str,
    repo: &str,
    commit: &str,
    file: &HfFile,
    token: Option<&str>,
    layout_dir: &Path,
    progress_key: &str,
) -> Result<Descriptor> {
    let url = format!("{endpoint}{owner}/{repo}/resolve/{commit}/{}", file.path);
    let label = format!("Pulling {}", basename(&file.path));

    // Tolerates ultimate failure (falls back to a plain, un-Xet'd GET
    // self-hashed after download) but still worth a few retries first —
    // a transient HEAD failure shouldn't cost the Xet fast path.
    let meta = super::client::retry(&format!("HEAD {}", file.path), || {
        download::head_metadata(head_client, url.clone(), token)
    })
    .await
    .unwrap_or_default();
    let xet = meta.xet_hash.map(|hash| XetFileRef {
        endpoint: endpoint.to_string(),
        repo_type: "models".to_string(),
        owner_repo: format!("{owner}/{repo}"),
        revision: commit.to_string(),
        hash,
        size: meta.size.max(file.size) as u64,
        sha256: meta.digest.clone(),
        hf_token: token.map(str::to_string),
    });

    let size = if meta.size > 0 { meta.size } else { file.size };
    progress::add_total(progress_key, size);

    let tmp_dir = layout_dir.join("blobs").join("tmp");
    std::fs::create_dir_all(&tmp_dir)?;
    // pid + a per-call counter, not just pid: two concurrent pulls of
    // different repos that happen to share a filename (e.g. both have a
    // "model.safetensors") would otherwise share this same path.
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp_path = tmp_dir.join(format!(
        "hf-{}-{unique}-{}.part",
        std::process::id(),
        sanitize(&file.path)
    ));

    let req = FetchRequest {
        url,
        token,
        xet,
        label: label.clone(),
    };
    let mut last_err: Option<anyhow::Error> = None;
    let mut next_delay = None;
    let digest = 'attempts: {
        for attempt in 0..super::client::MAX_ATTEMPTS {
            if let Some(delay) = next_delay.take() {
                eprintln!(
                    "\n[llmman] retrying {label} (attempt {}/{}, wait {delay:?})",
                    attempt + 1,
                    super::client::MAX_ATTEMPTS
                );
                tokio::time::sleep(delay).await;
            }
            // A fresh file each attempt, not a reopen: File::create
            // truncates, so a partial write from an earlier failed
            // attempt can never linger underneath this one's bytes.
            // (Progress reporting isn't undone on a retry — a retried
            // file's bar may briefly look ahead of itself; a cosmetic
            // wrinkle only, not worth this migration's added complexity.)
            let f = std::fs::File::create(&tmp_path)?;
            let mut writer = HashingProgressWriter {
                file: f,
                hasher: Sha256::new(),
                progress_key,
            };
            let result = download::fetch_once(client, &req, &mut writer)
                .await
                .and_then(|_| {
                    writer.file.flush()?;
                    let got = format!("sha256:{:x}", writer.hasher.finalize());
                    // meta.digest (from X-Linked-Etag) is a trusted claim
                    // about the content ahead of downloading it — worth
                    // checking the bytes we actually got still match, in
                    // case of transient corruption, same as a failed fetch.
                    match &meta.digest {
                        Some(want) if format!("sha256:{want}") != got => {
                            anyhow::bail!("digest mismatch: expected sha256:{want}, got {got}")
                        }
                        _ => Ok(got),
                    }
                });
            match result {
                Ok(digest) => break 'attempts digest,
                Err(e) => {
                    eprintln!("[llmman] {label} error: {e:#}");
                    match download::should_retry(&e, attempt + 1) {
                        Some(delay) => next_delay = Some(delay),
                        None => {
                            last_err = Some(e);
                            break;
                        }
                    }
                    last_err = Some(e);
                }
            }
        }
        let _ = std::fs::remove_file(&tmp_path);
        return Err(last_err.unwrap()).with_context(|| {
            format!(
                "download {} failed after {} attempts",
                file.path,
                super::client::MAX_ATTEMPTS
            )
        });
    };

    let dest_dir = layout_dir.join("blobs").join("sha256");
    std::fs::create_dir_all(&dest_dir)?;
    let dest = dest_dir.join(digest.trim_start_matches("sha256:"));
    let total = std::fs::metadata(&tmp_path)?.len() as i64;
    if !dest.exists() {
        std::fs::rename(&tmp_path, &dest)?;
    } else {
        let _ = std::fs::remove_file(&tmp_path);
    }

    Ok(Descriptor {
        media_type: oci::MEDIA_TYPE_MODEL_WEIGHT_RAW.to_string(),
        digest,
        size: total,
        annotations: Some(BTreeMap::from([(
            oci::ANNOTATION_FILEPATH.to_string(),
            basename(&file.path),
        )])),
        ..Default::default()
    })
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// Writes to a local file while hashing every byte and reporting it to
/// `progress`.
struct HashingProgressWriter<'a> {
    file: std::fs::File,
    hasher: Sha256,
    progress_key: &'a str,
}

impl Write for HashingProgressWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.file.write(buf)?;
        self.hasher.update(&buf[..n]);
        progress::add_completed(self.progress_key, n as i64);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}
