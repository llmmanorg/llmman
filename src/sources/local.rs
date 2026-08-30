//! A local model directory (`/absolute/path`), imported into the OCI
//! store as a CNCF ModelPack so `llmman serve`/`run`/`push` treat it
//! like anything else pulled. The source files are only ever read.

use std::path::Path;

use anyhow::{Context, Result};

use super::{should_pack, PackFile, Target};

pub(crate) fn pull(local_path: &str, target: &Target<'_>) -> Result<()> {
    let root = Path::new(local_path);
    let meta = std::fs::metadata(root).with_context(|| format!("local path {local_path:?}"))?;
    if !meta.is_dir() {
        anyhow::bail!("local path {local_path:?} is not a directory");
    }

    if target.report_cached(local_path, local_path) {
        return Ok(());
    }

    let mut packed = Vec::new();
    for entry in walkdir::WalkDir::new(root).follow_links(false) {
        let entry = entry.with_context(|| format!("walk {local_path}"))?;
        if !entry.file_type().is_file() {
            continue;
        }
        let rel = entry
            .path()
            .strip_prefix(root)
            .unwrap_or(entry.path())
            .to_string_lossy()
            // Recorded in the manifest, read back on any platform:
            // always the OCI-conventional "/" separator, never "\".
            .replace('\\', "/");
        if !should_pack(&rel) {
            continue;
        }
        packed.push(PackFile {
            local_path: entry.path().to_path_buf(),
            relative_path: rel,
            owned: false,
        });
    }

    if packed.is_empty() {
        anyhow::bail!("no model files found in {local_path}");
    }
    eprintln!("Importing {} files from {local_path}", packed.len());
    // No download progress to show the work happening, so announce each
    // file as it is stored instead.
    for f in &packed {
        if let Ok(m) = std::fs::metadata(&f.local_path) {
            eprintln!("  stored {} ({} bytes)", f.relative_path, m.len());
        }
    }

    let repo = root
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| local_path.to_string());
    super::pack_as_model_pack(
        target,
        local_path,
        &repo,
        packed,
        format!("no model files found in {local_path}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf::oci;

    fn tempdir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "llmman-sources-local-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn pull_imports_a_directory_as_a_model_pack_without_moving_the_originals() {
        let src = tempdir("src");
        let layout = tempdir("layout");
        std::fs::write(src.join("config.json"), b"{}").unwrap();
        std::fs::write(src.join("model.safetensors"), b"weights").unwrap();
        std::fs::create_dir_all(src.join("nested")).unwrap();
        std::fs::write(src.join("nested").join("tokenizer.json"), b"{}").unwrap();
        // Filtered out by should_pack.
        std::fs::write(src.join(".gitattributes"), b"x").unwrap();

        let reference = src.to_str().unwrap().to_string();
        oci::ensure_layout(&layout).unwrap();
        pull(
            &reference,
            &Target {
                layout_dir: &layout,
                progress_key: "",
                store_as: None,
            },
        )
        .unwrap();

        let desc = oci::read_manifest_ref(&layout, &reference).unwrap();
        let manifest: oci::Manifest =
            serde_json::from_slice(&oci::read_blob(&layout, &desc.digest).unwrap()).unwrap();
        let mut paths: Vec<&str> = manifest
            .layers
            .iter()
            .filter_map(|l| l.annotation(oci::ANNOTATION_FILEPATH))
            .collect();
        paths.sort_unstable();
        assert_eq!(
            paths,
            vec!["config.json", "model.safetensors", "nested/tokenizer.json"]
        );
        assert!(
            src.join("model.safetensors").exists(),
            "importing must copy, never move, a user's own files"
        );

        std::fs::remove_dir_all(&src).ok();
        std::fs::remove_dir_all(&layout).ok();
    }

    #[test]
    fn pull_rejects_a_directory_with_nothing_importable_in_it() {
        let src = tempdir("empty");
        let layout = tempdir("empty-layout");
        std::fs::write(src.join(".hidden"), b"x").unwrap();
        let err = pull(
            src.to_str().unwrap(),
            &Target {
                layout_dir: &layout,
                progress_key: "",
                store_as: None,
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("no model files found"));
        std::fs::remove_dir_all(&src).ok();
        std::fs::remove_dir_all(&layout).ok();
    }

    #[test]
    fn pull_rejects_a_path_that_is_not_a_directory() {
        let dir = tempdir("file");
        let file = dir.join("model.gguf");
        std::fs::write(&file, b"x").unwrap();
        let err = pull(
            file.to_str().unwrap(),
            &Target {
                layout_dir: &dir,
                progress_key: "",
                store_as: None,
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("is not a directory"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
