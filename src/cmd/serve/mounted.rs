//! Models served from a directory already on disk (`LLMMAN_MODEL_DIRS`).
//!
//! The engine loads straight on the directory, which is only read: nothing
//! is imported into the store, so store operations do not apply to these.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use anyhow::{anyhow, Context};
use sha2::{Digest, Sha256};

use crate::modelpack::ModelPath;
use crate::storage::oci::{Descriptor, ImageSummary};

pub(crate) const VAR: &str = "LLMMAN_MODEL_DIRS";

#[derive(Debug)]
pub(crate) struct MountedModel {
    /// As configured: what it is listed, loaded and served as.
    pub(crate) name: String,
    pub(crate) dir: PathBuf,
    /// [`reference_key`] of `name`.
    key: String,
    scan: Arc<OnceLock<Scan>>,
}

/// What one walk of the directory finds.
#[derive(Debug)]
struct Scan {
    size: u64,
    /// A symlink whose target is outside the directory, with that target.
    escaping_link: Option<(PathBuf, PathBuf)>,
}

impl Scan {
    fn of(dir: &Path) -> Self {
        let root = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
        let (mut size, mut escaping_link) = (0, None);
        for entry in walkdir::WalkDir::new(dir)
            .into_iter()
            .filter_map(Result::ok)
        {
            if entry.path_is_symlink() && escaping_link.is_none() {
                if let Ok(target) = entry.path().canonicalize() {
                    if !target.starts_with(&root) {
                        escaping_link = Some((entry.path().to_path_buf(), target));
                    }
                }
            }
            if let Ok(meta) = std::fs::metadata(entry.path()) {
                if meta.is_file() {
                    size += meta.len();
                }
            }
        }
        Self {
            size,
            escaping_link,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct MountedModels(Arc<[MountedModel]>);

/// `qwen3`, `qwen3:latest` and `docker.io/ai/qwen3:latest` are one model, as
/// in the store.
fn reference_key(reference: &str) -> Result<String, crate::shortnames::InvalidReference> {
    let resolved = crate::shortnames::resolve_ollama_api(reference)?;
    Ok(crate::storage::default_tag(&resolved))
}

impl MountedModels {
    /// `value` of [`VAR`]: comma-separated `NAME=DIRECTORY`, directory
    /// absolute. Only the syntax is checked.
    pub(crate) fn parse(value: Option<&str>) -> anyhow::Result<Self> {
        let (models, errors) = Self::parse_each(value);
        errors.into_iter().next().map_or(Ok(models), Err)
    }

    /// The valid entries of `value`, and an error for each other one.
    fn parse_each(value: Option<&str>) -> (Self, Vec<anyhow::Error>) {
        let (mut models, mut errors) = (Vec::new(), Vec::new());
        for entry in value.unwrap_or_default().split(',') {
            let entry = entry.trim();
            if entry.is_empty() {
                continue;
            }
            match Self::parse_entry(entry, &models) {
                Ok(model) => models.push(model),
                Err(e) => errors.push(e),
            }
        }
        (Self(models.into()), errors)
    }

    fn parse_entry(entry: &str, seen: &[MountedModel]) -> anyhow::Result<MountedModel> {
        let (name, dir) = entry
            .split_once('=')
            .map(|(n, d)| (n.trim(), d.trim()))
            .filter(|(n, d)| !n.is_empty() && !d.is_empty())
            .ok_or_else(|| anyhow!("{VAR}: {entry:?} is not NAME=DIRECTORY"))?;
        let key = reference_key(name).map_err(|e| anyhow!("{VAR}: {name:?}: {e}"))?;
        anyhow::ensure!(
            Path::new(dir).is_absolute(),
            "{VAR}: {name}: {dir:?} is not an absolute path"
        );
        if let Some(other) = seen.iter().find(|m| m.key == key) {
            anyhow::bail!("{VAR}: {name} and {} are the same model", other.name);
        }
        Ok(MountedModel {
            name: name.to_string(),
            dir: PathBuf::from(dir),
            key,
            scan: Arc::default(),
        })
    }

    /// The valid entries of the environment, for the commands that work on
    /// the store: a malformed one is the daemon's to report.
    pub(crate) fn lenient_from_env() -> Self {
        Self::parse_each(std::env::var(VAR).ok().as_deref()).0
    }

    /// [`parse`](Self::parse) of the environment, and every directory must
    /// exist: a volume is mounted before the container starts.
    pub(crate) fn from_env() -> anyhow::Result<Self> {
        Self::from_value(std::env::var(VAR).ok().as_deref())
    }

    fn from_value(value: Option<&str>) -> anyhow::Result<Self> {
        let models = Self::parse(value)?;
        for m in models.iter() {
            anyhow::ensure!(
                m.dir.is_dir(),
                "{VAR}: {}: {} is not a directory",
                m.name,
                m.dir.display()
            );
        }
        Ok(models)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &MountedModel> {
        self.0.iter()
    }

    /// The model `reference` names, in any spelling of its name.
    pub(crate) fn find(&self, reference: &str) -> Option<&MountedModel> {
        if self.is_empty() {
            return None;
        }
        let key = reference_key(reference).ok()?;
        self.0.iter().find(|m| m.key == key)
    }

    /// Why `action` is not done to the model `reference` names, if it is mounted.
    pub(crate) fn refusal(&self, reference: &str, action: &str) -> Option<String> {
        let m = self.find(reference)?;
        Some(format!(
            "{} is a model directory ({}) served through {VAR}, not a stored model: it \
             cannot be {action}. Its files are never changed; stop serving it by removing \
             it from {VAR}",
            m.name,
            m.dir.display()
        ))
    }
}

impl MountedModel {
    /// What the engine is started on. Blocking.
    pub(crate) fn resolve(&self) -> anyhow::Result<ModelPath> {
        let path = crate::modelpack::resolve_model_dir(&self.dir)
            .with_context(|| format!("resolve model {}", self.name))?;
        // The speech worker finds its model in the store by name.
        if let ModelPath::Gguf(file, _) = &path {
            if crate::gguf::read_info(file).is_ok_and(|info| info.is_speech()) {
                anyhow::bail!(
                    "{} is a speech model, which runs from the store: import it with \
                     `llmman pull {}` instead of mounting it",
                    self.name,
                    self.dir.display()
                );
            }
        }
        Ok(path)
    }

    /// Walks the directory off the runtime's workers, so that later reads of
    /// [`size`](Self::size) do not block.
    pub(crate) async fn scan(&self) {
        let (dir, cell) = (self.dir.clone(), self.scan.clone());
        let _ = tokio::task::spawn_blocking(move || {
            cell.get_or_init(|| Scan::of(&dir));
        })
        .await;
    }

    fn scanned(&self) -> &Scan {
        self.scan.get_or_init(|| Scan::of(&self.dir))
    }

    /// A stand-in for the manifest digest: the content is not hashed, so it
    /// names this mapping.
    pub(crate) fn digest(&self) -> String {
        let mut hash = Sha256::new();
        hash.update(self.name.as_bytes());
        hash.update([0]);
        hash.update(self.dir.to_string_lossy().as_bytes());
        format!("sha256:{:x}", hash.finalize())
    }

    /// The bytes of the files under the directory. Blocks unless scanned.
    pub(crate) fn size(&self) -> u64 {
        self.scanned().size
    }

    /// A container sees only the directory, so a link out of it dangles there.
    pub(crate) fn ensure_visible_to_container(&self) -> anyhow::Result<()> {
        match &self.scanned().escaping_link {
            Some((link, target)) => anyhow::bail!(
                "{} links to {}, outside the model directory, which a container cannot see: \
                 serve it with --runtime path, or put the files in the directory",
                link.display(),
                target.display()
            ),
            None => Ok(()),
        }
    }

    /// What [`OciStore::find`](crate::storage::OciStore::find) answers for a stored model.
    pub(crate) fn descriptor(&self) -> Descriptor {
        Descriptor {
            media_type: String::new(),
            digest: self.digest(),
            size: self.size(),
            annotations: None,
        }
    }

    /// This model as a row of `/api/tags` or `/v1/models`.
    pub(crate) fn summary(&self) -> ImageSummary {
        ImageSummary {
            reference: self.name.clone(),
            digest: self.digest(),
            media_type: String::new(),
            size: self.size(),
            modified_at: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn names(models: &MountedModels) -> HashSet<&str> {
        models.iter().map(|m| m.name.as_str()).collect()
    }

    fn parse(value: &str) -> anyhow::Result<MountedModels> {
        MountedModels::parse(Some(value))
    }

    /// An absolute path on this platform, which need not exist.
    fn abs(root: &tempfile::TempDir, name: &str) -> String {
        root.path().join(name).display().to_string()
    }

    #[test]
    fn nothing_is_mounted_by_default() {
        assert!(MountedModels::parse(None).unwrap().is_empty());
        assert!(parse("").unwrap().is_empty());
        assert!(parse(" , ").unwrap().is_empty());
        assert!(MountedModels::default().find("anything").is_none());
    }

    #[test]
    fn entries_are_name_equals_directory_separated_by_commas() {
        let root = tempfile::tempdir().unwrap();
        let value = format!(
            "qwen3.8:27b-nvfp4={}, small={},",
            abs(&root, "qwen"),
            abs(&root, "a b")
        );
        let models = parse(&value).unwrap();
        assert_eq!(
            names(&models),
            HashSet::from(["qwen3.8:27b-nvfp4", "small"])
        );
        let small = models.find("small").unwrap();
        assert_eq!(small.dir, root.path().join("a b"));
    }

    /// A drive letter's `:` is neither the `=` nor the `,` entries split on.
    #[cfg(windows)]
    #[test]
    fn a_windows_path_with_a_drive_letter_parses() {
        let models = parse(r"qwen3.8:27b-nvfp4=C:\models\qwen, small=D:\my models\small").unwrap();
        assert_eq!(
            models.find("qwen3.8:27b-nvfp4").unwrap().dir,
            Path::new(r"C:\models\qwen")
        );
        assert_eq!(
            models.find("small").unwrap().dir,
            Path::new(r"D:\my models\small")
        );
    }

    #[test]
    fn a_malformed_entry_is_refused_by_name() {
        let root = tempfile::tempdir().unwrap();
        let same = format!("a={},a:latest={}", abs(&root, "x"), abs(&root, "y"));
        for bad in [
            "qwen",
            &format!("={}", abs(&root, "qwen")),
            "qwen=",
            "qwen=models/qwen",
            &same,
        ] {
            let err = parse(bad).unwrap_err().to_string();
            assert!(err.contains(VAR), "{bad}: {err}");
        }
        let err = parse("qwen=models/qwen").unwrap_err().to_string();
        assert!(err.contains("not an absolute path"), "{err}");
        let err = parse(&same).unwrap_err().to_string();
        assert!(err.contains("are the same model"), "{err}");
    }

    #[test]
    fn a_malformed_entry_leaves_the_valid_ones_refusable() {
        let root = tempfile::tempdir().unwrap();
        let value = format!(
            "bad,qwen={},=x,qwen:latest={}",
            abs(&root, "q"),
            abs(&root, "r")
        );
        let (models, errors) = MountedModels::parse_each(Some(&value));
        assert_eq!(errors.len(), 3);
        assert_eq!(names(&models), HashSet::from(["qwen"]));
        let msg = models.refusal("qwen:latest", "removed").unwrap();
        assert!(
            msg.contains("cannot be removed") && msg.contains(VAR),
            "{msg}"
        );
        assert!(models.refusal("other", "removed").is_none());
    }

    #[test]
    fn every_spelling_of_the_name_finds_the_model() {
        let root = tempfile::tempdir().unwrap();
        let models = parse(&format!(
            "qwen3.8:27b-nvfp4={},gemma4={}",
            abs(&root, "qwen"),
            abs(&root, "gemma")
        ))
        .unwrap();
        for spelling in [
            "qwen3.8:27b-nvfp4",
            "docker.io/ai/qwen3.8:27b-nvfp4",
            "gemma4",
            "gemma4:latest",
            "docker.io/ai/gemma4",
        ] {
            assert!(models.find(spelling).is_some(), "{spelling}");
        }
        assert_eq!(models.find("gemma4").unwrap().name, "gemma4");
        for other in [
            "qwen3.8",
            "qwen3.8:27b",
            "docker.io/ai/gemma4:7b",
            "",
            "a b",
        ] {
            assert!(models.find(other).is_none(), "{other}");
        }
    }

    #[test]
    fn the_daemon_needs_every_directory_but_parsing_does_not() {
        let root = tempfile::tempdir().unwrap();
        let missing = abs(&root, "does-not-exist");
        let value = format!("m={missing}");
        assert!(parse(&value).is_ok());
        let err = MountedModels::from_value(Some(&value))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains(&format!("m: {missing} is not a directory")),
            "{err}"
        );
        let value = format!("m={}", root.path().display());
        assert!(MountedModels::from_value(Some(&value)).is_ok());
        assert!(MountedModels::from_value(None).unwrap().is_empty());
    }

    #[test]
    fn the_digest_names_the_mapping_and_the_size_counts_the_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("config.json"), b"12345").unwrap();
        std::fs::write(dir.path().join("sub/w.safetensors"), b"1234567").unwrap();
        let value = format!("m={0},n={0}", dir.path().display());
        let models = parse(&value).unwrap();
        let (m, n) = (models.find("m").unwrap(), models.find("n").unwrap());
        assert_eq!(m.size(), 12);
        let digest = m.digest();
        assert_eq!(digest, m.digest());
        assert_ne!(digest, n.digest());
        crate::storage::oci::split_digest(&digest).unwrap();
        let row = m.summary();
        assert_eq!(
            (row.reference.as_str(), row.size, row.digest),
            ("m", 12, digest)
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_link_out_of_the_directory_is_refused_for_a_container_only() {
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("w.safetensors"), b"weights").unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
        std::fs::write(dir.path().join("own.safetensors"), b"1").unwrap();
        std::os::unix::fs::symlink("own.safetensors", dir.path().join("inside.safetensors"))
            .unwrap();
        let models = parse(&format!("m={}", dir.path().display())).unwrap();
        let m = models.find("m").unwrap();
        m.ensure_visible_to_container().unwrap();

        std::os::unix::fs::symlink(
            outside.path().join("w.safetensors"),
            dir.path().join("model.safetensors"),
        )
        .unwrap();
        let models = parse(&format!("m={}", dir.path().display())).unwrap();
        let err = models
            .find("m")
            .unwrap()
            .ensure_visible_to_container()
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("model.safetensors") && err.contains("--runtime path"),
            "{err}"
        );
    }

    #[test]
    fn a_directory_without_a_model_does_not_resolve_and_a_safetensors_one_does() {
        let dir = tempfile::tempdir().unwrap();
        let models = parse(&format!("m={}", dir.path().display())).unwrap();
        let m = models.find("m").unwrap();
        let err = format!("{:#}", m.resolve().err().unwrap());
        assert!(
            err.contains("resolve model m") && err.contains("holds no model"),
            "{err}"
        );
        std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
        assert_eq!(m.resolve().unwrap().format(), "safetensors");
    }

    #[test]
    fn the_refusal_names_the_model_the_action_and_the_variable() {
        let root = tempfile::tempdir().unwrap();
        let dir = abs(&root, "qwen");
        let models = parse(&format!("qwen={dir}")).unwrap();
        let msg = models.refusal("qwen", "removed").unwrap();
        assert!(msg.contains("qwen") && msg.contains(&dir), "{msg}");
        assert!(
            msg.contains("cannot be removed") && msg.contains(VAR),
            "{msg}"
        );
    }
}
