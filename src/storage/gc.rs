//! Content-addressed garbage collection for the blob store and the
//! extracted-model cache.
//!
//! Because the store is fully content-addressed and every tag is just a
//! small pointer file under `manifests-v2/` or the legacy `manifests/`
//! (see [`crate::storage::oci`]),
//! "what's still needed" can always be recomputed from scratch by reading
//! every *surviving* manifest — there's no refcount or journal to keep in
//! sync (a crash, a manual edit, an interrupted pull can never desync it,
//! because the live set is rebuilt fresh on every sweep).
//!
//! [`sweep`] deletes everything unreferenced but younger than
//! [`GC_GRACE_PERIOD`], since a pull in another process writes blobs before
//! its tag. [`reclaim_removed`] is `rm`'s: it deletes at any age only what
//! the removed tags held, so it needs no grace window.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::{Duration, SystemTime};

use anyhow::Context;

use super::OciStore;

/// Grace window for the startup catch-all sweep — matches
/// [`crate::storage::repair::STALE_TMP_FILE_AGE`], so there's one duration
/// to reason about for "how long might a just-written blob be legitimately
/// unreferenced".
pub const GC_GRACE_PERIOD: Duration = Duration::from_secs(60 * 60);

/// What a sweep freed, for reporting.
#[derive(Debug, Default, Clone, Copy)]
pub struct GcStats {
    pub count: usize,
    pub bytes: u64,
}

/// Whether a prune deletes a blob file or cache directory, given its name
/// (the digest hex) and path.
type Doomed<'a> = &'a dyn Fn(&str, &Path) -> bool;

fn names(digests: &HashSet<String>, hex: &str) -> bool {
    digests.contains(&format!("sha256:{hex}"))
}

/// Doomed: not in `live` and at least `grace` old.
fn unreferenced(live: &HashSet<String>, grace: Duration) -> impl Fn(&str, &Path) -> bool + '_ {
    move |hex, path| !names(live, hex) && is_older_than(path, grace)
}

/// "<size>" and "(<n> blobs, <m> cache entries)" for a prune that freed something.
pub fn freed((blobs, cache): &(GcStats, GcStats)) -> Option<(String, String)> {
    (blobs.count + cache.count > 0).then(|| {
        (
            crate::fmt::human_size(blobs.bytes + cache.bytes),
            format!("({} blobs, {} cache entries)", blobs.count, cache.count),
        )
    })
}

#[cfg(unix)]
type GcFileIdentity = (u64, u64);

#[cfg(windows)]
type GcFileIdentity = same_file::Handle;

#[derive(Default)]
struct GcFileAccount {
    identities: HashSet<GcFileIdentity>,
}

impl GcFileAccount {
    fn remember(&mut self, path: &Path) -> anyhow::Result<()> {
        let Some(meta) = ignore_not_found(std::fs::symlink_metadata(path))
            .with_context(|| format!("inspect {}", path.display()))?
        else {
            return Ok(());
        };
        if !meta.is_file() {
            return Ok(());
        }
        if let Some(identity) =
            file_identity(path, &meta).with_context(|| format!("identify {}", path.display()))?
        {
            self.identities.insert(identity);
        }
        Ok(())
    }

    fn size(&mut self, path: &Path) -> u64 {
        let Ok(meta) = std::fs::symlink_metadata(path) else {
            return 0;
        };
        if !meta.is_file() {
            return 0;
        }
        let size = meta.len();
        match file_identity(path, &meta).ok().flatten() {
            Some(identity) => {
                if self.identities.insert(identity) {
                    size
                } else {
                    0
                }
            }
            None => size,
        }
    }
}

/// Treats a path removed by a concurrent sweep as absent; other I/O errors
/// still abort retained-file accounting.
fn ignore_not_found<T>(result: std::io::Result<T>) -> std::io::Result<Option<T>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

#[cfg(unix)]
fn file_identity(
    _path: &Path,
    meta: &std::fs::Metadata,
) -> std::io::Result<Option<GcFileIdentity>> {
    use std::os::unix::fs::MetadataExt as _;

    Ok(Some((meta.dev(), meta.ino())))
}

#[cfg(windows)]
fn file_identity(
    path: &Path,
    _meta: &std::fs::Metadata,
) -> std::io::Result<Option<GcFileIdentity>> {
    ignore_not_found(same_file::Handle::from_path(path))
}

/// Every blob digest ("sha256:<hex>") still reachable from a surviving
/// tag: each manifest's own digest, its config digest, and every layer
/// digest. Built by walking [`OciStore::list_refs_strict`] and reading each
/// manifest — the same traversal `resolve_model` does per-reference, just
/// over every reference at once.
///
/// This is the authority for a *destructive* sweep, so it fails closed:
/// any reference that can't be enumerated (an unreadable pointer file or
/// subtree) or any manifest that can't be read/parsed aborts the whole
/// computation with an error. The alternative — skipping the unreadable
/// entry, as the display-oriented `list_refs` does — would make that
/// model's config/layers look unreferenced and get deleted, possibly
/// destroying blobs shared with other healthy models. When we can't prove
/// what's still referenced, callers must delete nothing.
pub fn referenced_digests(store: &OciStore) -> anyhow::Result<HashSet<String>> {
    Ok(all_digests(&held_by_manifests(store)?))
}

/// A manifest as [`held_by_manifests`] saw it: when its blob was last
/// written, and its config and layer digests.
#[derive(Debug)]
pub struct Held {
    modified: Option<SystemTime>,
    digests: Vec<String>,
}

fn blob_mtime(store: &OciStore, digest: &str) -> Option<SystemTime> {
    std::fs::metadata(store.blob_path(digest).ok()?)
        .ok()?
        .modified()
        .ok()
}

/// Each manifest tags point at, by digest. Fails closed exactly as
/// [`referenced_digests`] describes.
pub fn held_by_manifests(store: &OciStore) -> anyhow::Result<HashMap<String, Held>> {
    let mut held = HashMap::new();
    for desc in store.list_refs_strict()? {
        let manifest = store
            .read_manifest(&desc.digest)
            .with_context(|| format!("read manifest {} for GC reference scan", desc.digest))?;
        let digests = std::iter::once(manifest.config.digest)
            .chain(manifest.layers.into_iter().map(|l| l.digest))
            .collect();
        let modified = blob_mtime(store, &desc.digest);
        held.insert(desc.digest, Held { modified, digests });
    }
    Ok(held)
}

fn all_digests(held: &HashMap<String, Held>) -> HashSet<String> {
    held.iter()
        .flat_map(|(manifest, h)| std::iter::once(manifest).chain(&h.digests))
        .cloned()
        .collect()
}

/// Deletes every blob file under `blobs/sha256/` whose `sha256:<name>`
/// digest isn't in `live`, skipping in-progress temp writes (`tmp-`/
/// `.tmp`, same as [`crate::storage::repair`]) and anything younger than
/// `grace`. A missing blobs directory is a no-op.
#[cfg(test)]
fn prune_blobs(
    store_root: &Path,
    live: &HashSet<String>,
    grace: Duration,
) -> anyhow::Result<GcStats> {
    let doomed = unreferenced(live, grace);
    let mut account = GcFileAccount::default();
    remember_retained_blobs(store_root, &doomed, &mut account)?;
    prune_blobs_with_account(store_root, &doomed, &mut account)
}

/// Prunes blobs and cache with one file-identity account, so hardlinked
/// blob/cache paths are reported once.
pub fn prune_blobs_and_cache(
    store_root: &Path,
    cache_path: &Path,
    live: &HashSet<String>,
    grace: Duration,
) -> anyhow::Result<(GcStats, GcStats)> {
    prune_doomed(store_root, cache_path, &unreferenced(live, grace), grace)
}

/// [`prune_blobs_and_cache`] for any `doomed` rule. `grace` only ages out
/// stale copy temp files inside cache directories that are kept.
fn prune_doomed(
    store_root: &Path,
    cache_path: &Path,
    doomed: Doomed,
    grace: Duration,
) -> anyhow::Result<(GcStats, GcStats)> {
    let mut account = GcFileAccount::default();
    remember_retained_blobs(store_root, doomed, &mut account)?;
    remember_retained_cache_files(cache_path, doomed, grace, &mut account)?;
    let blob_stats = prune_blobs_with_account(store_root, doomed, &mut account)?;
    let cache_stats = prune_cache_with_account(cache_path, doomed, grace, &mut account)?;
    Ok((blob_stats, cache_stats))
}

fn remember_retained_blobs(
    store_root: &Path,
    doomed: Doomed,
    account: &mut GcFileAccount,
) -> anyhow::Result<()> {
    let blobs_dir = store_root.join("blobs").join("sha256");
    let Some(entries) = ignore_not_found(std::fs::read_dir(&blobs_dir))
        .with_context(|| format!("read {}", blobs_dir.display()))?
    else {
        return Ok(());
    };
    for entry in entries {
        let Some(entry) = ignore_not_found(entry)
            .with_context(|| format!("read entry in {}", blobs_dir.display()))?
        else {
            continue;
        };
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name.starts_with("tmp-") || name.ends_with(".tmp") {
            continue;
        }
        if !doomed(name, &path) {
            account.remember(&path)?;
        }
    }
    Ok(())
}

fn remember_retained_cache_files(
    cache_path: &Path,
    doomed: Doomed,
    grace: Duration,
    account: &mut GcFileAccount,
) -> anyhow::Result<()> {
    let Some(entries) = ignore_not_found(std::fs::read_dir(cache_path))
        .with_context(|| format!("read {}", cache_path.display()))?
    else {
        return Ok(());
    };
    for entry in entries {
        let Some(entry) = ignore_not_found(entry)
            .with_context(|| format!("read entry in {}", cache_path.display()))?
        else {
            continue;
        };
        let path = entry.path();
        let Some(file_type) = ignore_not_found(entry.file_type())
            .with_context(|| format!("inspect {}", path.display()))?
        else {
            continue;
        };
        if !file_type.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !doomed(name, &path) {
            remember_cache_dir_files(&path, grace, account)?;
        }
    }
    Ok(())
}

fn remember_cache_dir_files(
    dir: &Path,
    grace: Duration,
    account: &mut GcFileAccount,
) -> anyhow::Result<()> {
    let Some(entries) = ignore_not_found(std::fs::read_dir(dir))
        .with_context(|| format!("read {}", dir.display()))?
    else {
        return Ok(());
    };
    for entry in entries {
        let Some(entry) =
            ignore_not_found(entry).with_context(|| format!("read entry in {}", dir.display()))?
        else {
            continue;
        };
        let path = entry.path();
        let Some(file_type) = ignore_not_found(entry.file_type())
            .with_context(|| format!("inspect {}", path.display()))?
        else {
            continue;
        };
        if file_type.is_dir() {
            remember_cache_dir_files(&path, grace, account)?;
            continue;
        }
        let is_stale_copy_temp = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| is_cache_copy_temp_name(name) && is_older_than(&path, grace));
        if is_stale_copy_temp {
            continue;
        }
        account.remember(&path)?;
    }
    Ok(())
}

fn prune_blobs_with_account(
    store_root: &Path,
    doomed: Doomed,
    account: &mut GcFileAccount,
) -> anyhow::Result<GcStats> {
    let blobs_dir = store_root.join("blobs").join("sha256");
    let mut stats = GcStats::default();
    let entries = match std::fs::read_dir(&blobs_dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(stats),
        Err(e) => return Err(e).with_context(|| format!("read {}", blobs_dir.display())),
    };
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        // In-progress or abandoned write — left to repair's own sweep.
        if name.starts_with("tmp-") || name.ends_with(".tmp") {
            continue;
        }
        if !doomed(name, &path) {
            continue;
        }
        let size = account.size(&path);
        if let Err(e) = std::fs::remove_file(&path) {
            eprintln!("[llmman] couldn't remove unreferenced blob {name}: {e:#}");
            continue;
        }
        stats.count += 1;
        stats.bytes += size;
    }
    Ok(stats)
}

/// Deletes every cache subdirectory under `cache_path` whose name (a layer
/// hex for GGUF, a manifest hex for safetensors — see
/// `modelpack::extract_gguf_layer` / `extract_safetensors_dir`) doesn't
/// correspond to a live digest, skipping anything younger than `grace`. A
/// missing cache directory is a no-op. Stale `.tmp` files inside kept
/// cache directories are also removed.
#[cfg(test)]
fn prune_cache(
    cache_path: &Path,
    live: &HashSet<String>,
    grace: Duration,
) -> anyhow::Result<GcStats> {
    prune_cache_with_account(
        cache_path,
        &unreferenced(live, grace),
        grace,
        &mut GcFileAccount::default(),
    )
}

fn prune_cache_with_account(
    cache_path: &Path,
    doomed: Doomed,
    grace: Duration,
    account: &mut GcFileAccount,
) -> anyhow::Result<GcStats> {
    let mut stats = GcStats::default();
    let entries = match std::fs::read_dir(cache_path) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(stats),
        Err(e) => return Err(e).with_context(|| format!("read {}", cache_path.display())),
    };
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !doomed(name, &path) {
            let tmp_stats = prune_stale_cache_temps(&path, grace, account);
            stats.count += tmp_stats.count;
            stats.bytes += tmp_stats.bytes;
            continue;
        }
        let size = dir_size(&path, account);
        if let Err(e) = std::fs::remove_dir_all(&path) {
            eprintln!("[llmman] couldn't remove unreferenced cache dir {name}: {e:#}");
            continue;
        }
        stats.count += 1;
        stats.bytes += size;
    }
    Ok(stats)
}

/// True if `path`'s mtime is at least `grace` old. A zero `grace` makes
/// this always true (used only by tests that want to sweep regardless of
/// age). A file whose mtime can't be read is treated as not-yet-old, so
/// it's left alone rather than deleted on a metadata hiccup.
fn is_older_than(path: &Path, grace: Duration) -> bool {
    if grace.is_zero() {
        return true;
    }
    let Ok(modified) = std::fs::metadata(path).and_then(|m| m.modified()) else {
        return false;
    };
    SystemTime::now()
        .duration_since(modified)
        .is_ok_and(|age| age >= grace)
}

/// Total size of files under `dir`. Best-effort: unreadable entries
/// contribute 0.
fn dir_size(dir: &Path, account: &mut GcFileAccount) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|e| {
            let path = e.path();
            if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                dir_size(&path, account)
            } else {
                account.size(&path)
            }
        })
        .sum()
}

fn prune_stale_cache_temps(dir: &Path, grace: Duration, account: &mut GcFileAccount) -> GcStats {
    let mut stats = GcStats::default();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return stats;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            let inner = prune_stale_cache_temps(&path, grace, account);
            stats.count += inner.count;
            stats.bytes += inner.bytes;
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !is_cache_copy_temp_name(name) || !is_older_than(&path, grace) {
            continue;
        }
        let size = account.size(&path);
        if let Err(e) = std::fs::remove_file(&path) {
            eprintln!(
                "[llmman] couldn't remove stale cache temp file {}: {e:#}",
                path.display()
            );
            continue;
        }
        stats.count += 1;
        stats.bytes += size;
    }
    stats
}

fn is_cache_copy_temp_name(name: &str) -> bool {
    let Some(stem) = name.strip_suffix(".tmp") else {
        return false;
    };
    let Some((before_counter, counter)) = stem.rsplit_once('.') else {
        return false;
    };
    let Some((dest, pid)) = before_counter.rsplit_once('.') else {
        return false;
    };
    !dest.is_empty()
        && !pid.is_empty()
        && !counter.is_empty()
        && pid.bytes().all(|b| b.is_ascii_digit())
        && counter.bytes().all(|b| b.is_ascii_digit())
}

/// Sweeps what no surviving tag references, within [`GC_GRACE_PERIOD`].
pub fn sweep(store: &OciStore, cache_path: &Path) -> anyhow::Result<(GcStats, GcStats)> {
    let live = referenced_digests(store)?;
    prune_blobs_and_cache(store.root(), cache_path, &live, GC_GRACE_PERIOD)
}

/// Deletes, at any age, what only the tags removed since `before` (a
/// [`held_by_manifests`] taken ahead of the untag) held, plus cache
/// directories named by those digests. Skips a removed manifest whose blob
/// changed since `before` (republished), and layers touched after it was
/// written (reused by a writer). Deletes nothing if the surviving set can't
/// be computed.
pub fn reclaim_removed(
    store: &OciStore,
    cache_path: &Path,
    before: &HashMap<String, Held>,
) -> anyhow::Result<(GcStats, GcStats)> {
    let live = all_digests(&held_by_manifests(store)?);
    let mut candidates = HashSet::new();
    for (manifest, held) in before.iter().filter(|(m, _)| !live.contains(*m)) {
        let Some(written) = held
            .modified
            .filter(|&at| blob_mtime(store, manifest) == Some(at))
        else {
            continue;
        };
        candidates.insert(manifest.clone());
        candidates.extend(
            held.digests
                .iter()
                .filter(|d| {
                    !live.contains(*d) && blob_mtime(store, d).is_none_or(|at| at <= written)
                })
                .cloned(),
        );
    }
    let doomed = |hex: &str, _: &Path| names(&candidates, hex);
    prune_doomed(store.root(), cache_path, &doomed, GC_GRACE_PERIOD)
}

/// Runs `retag`, which re-points `reference`, then sweeps if that replaced
/// other content: nothing else a tag write does orphans a blob. The tag has
/// landed by then, so a failed sweep is logged, not returned.
pub fn after_retag<T>(
    store: &OciStore,
    cache_path: &Path,
    reference: &str,
    retag: impl FnOnce() -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    if noprune_from_env() {
        return retag();
    }
    let digest = |store: &OciStore| store.find(reference).ok().map(|d| d.digest);
    let before = digest(store);
    let out = retag()?;
    if before.is_some() && digest(store) != before {
        match sweep(store, cache_path) {
            Ok(stats) => {
                if let Some((size, counts)) = freed(&stats) {
                    eprintln!("[llmman] freed {size} replacing {reference} {counts}");
                }
            }
            Err(e) => eprintln!("[llmman] GC sweep after replacing {reference} failed: {e:#}"),
        }
    }
    Ok(out)
}

/// Skips the post-`rm`, post-retag and startup GC sweeps when `LLMMAN_NOPRUNE` is
/// set to anything other than an explicit falsy value — an escape hatch
/// for shared/read-mostly stores or scripts that `rm` in a loop and would
/// rather prune once at the end themselves. Read fresh at each call site,
/// like every other `LLMMAN_*` var.
pub fn noprune_from_env() -> bool {
    crate::env_flag_set("LLMMAN_NOPRUNE")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "llmman-gc-test-{name}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn prune_blobs_removes_only_unreferenced_blobs() {
        let root = temp_dir("prune-blobs");
        let blobs = root.join("blobs").join("sha256");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::write(blobs.join("aaaa"), b"referenced").unwrap();
        std::fs::write(blobs.join("bbbb"), b"orphan").unwrap();
        std::fs::write(blobs.join("tmp-123"), b"in-progress").unwrap();

        let mut live = HashSet::new();
        live.insert("sha256:aaaa".to_string());

        let stats = prune_blobs(&root, &live, Duration::ZERO).unwrap();
        assert_eq!(stats.count, 1);
        assert!(blobs.join("aaaa").exists(), "referenced blob must survive");
        assert!(!blobs.join("bbbb").exists(), "orphan blob must be removed");
        assert!(
            blobs.join("tmp-123").exists(),
            "in-progress temp file must be left to repair's sweep"
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The concurrent-pull safety property: an unreferenced blob younger
    /// than the grace window is left alone (it may be a layer an in-flight
    /// pull has already written but not yet tagged), while an equally
    /// unreferenced but older blob is swept. Without this, a concurrent
    /// `rm` could delete a live pull's just-written layer.
    #[test]
    #[cfg(unix)]
    fn prune_blobs_respects_the_grace_window() {
        let root = temp_dir("prune-blobs-grace");
        let blobs = root.join("blobs").join("sha256");
        std::fs::create_dir_all(&blobs).unwrap();
        let fresh = blobs.join("cccc");
        let stale = blobs.join("dddd");
        std::fs::write(&fresh, b"just written by an in-flight pull").unwrap();
        std::fs::write(&stale, b"long-abandoned orphan").unwrap();
        // Back-date the stale blob past the grace window.
        let old = SystemTime::now() - GC_GRACE_PERIOD - Duration::from_secs(60);
        filetime_set(&stale, old);

        let live = HashSet::new(); // neither is referenced
        let stats = prune_blobs(&root, &live, GC_GRACE_PERIOD).unwrap();

        assert_eq!(stats.count, 1);
        assert_eq!(stats.bytes, b"long-abandoned orphan".len() as u64);
        assert!(
            fresh.exists(),
            "a fresh unreferenced blob must survive — it may be a live pull's untagged layer"
        );
        assert!(!stale.exists(), "a stale unreferenced blob must be swept");

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn prune_cache_removes_only_unreferenced_dirs() {
        let cache = temp_dir("prune-cache");
        std::fs::create_dir_all(cache.join("aaaa")).unwrap();
        std::fs::write(cache.join("aaaa").join("model.gguf"), b"kept").unwrap();
        std::fs::create_dir_all(cache.join("bbbb")).unwrap();
        std::fs::write(cache.join("bbbb").join("model.gguf"), b"orphan").unwrap();

        let mut live = HashSet::new();
        live.insert("sha256:aaaa".to_string());

        let stats = prune_cache(&cache, &live, Duration::ZERO).unwrap();
        assert_eq!(stats.count, 1);
        assert!(cache.join("aaaa").exists(), "referenced cache dir survives");
        assert!(!cache.join("bbbb").exists(), "orphan cache dir removed");

        std::fs::remove_dir_all(&cache).unwrap();
    }

    #[test]
    fn prune_cache_removes_stale_temp_files_inside_live_dirs() {
        let cache = temp_dir("prune-cache-temp");
        std::fs::create_dir_all(cache.join("aaaa").join("sub")).unwrap();
        std::fs::write(cache.join("aaaa").join("model.safetensors"), b"kept").unwrap();
        let legitimate_tmp_layer = cache.join("aaaa").join("sub").join("weights.tmp");
        std::fs::write(&legitimate_tmp_layer, b"real layer").unwrap();
        let tmp = cache
            .join("aaaa")
            .join("sub")
            .join("model.safetensors.123.0.tmp");
        std::fs::write(&tmp, b"partial copy").unwrap();

        let mut live = HashSet::new();
        live.insert("sha256:aaaa".to_string());

        let stats = prune_cache(&cache, &live, Duration::ZERO).unwrap();
        assert_eq!(stats.count, 1);
        assert_eq!(stats.bytes, "partial copy".len() as u64);
        assert!(
            cache.join("aaaa").join("model.safetensors").exists(),
            "live cache file survives"
        );
        assert!(!tmp.exists(), "stale temp file is removed");
        assert!(
            legitimate_tmp_layer.exists(),
            "real cache layer ending in .tmp survives"
        );

        std::fs::remove_dir_all(&cache).unwrap();
    }

    #[test]
    fn cache_copy_temp_name_matches_only_atomic_copy_temps() {
        assert!(is_cache_copy_temp_name("model.safetensors.123.0.tmp"));
        assert!(is_cache_copy_temp_name("config.json.123.45.tmp"));
        assert!(!is_cache_copy_temp_name("weights.tmp"));
        assert!(!is_cache_copy_temp_name("model.safetensors.tmp"));
        assert!(!is_cache_copy_temp_name("model.safetensors.123.tmp"));
        assert!(!is_cache_copy_temp_name("model.safetensors.pid.0.tmp"));
    }

    #[test]
    fn prune_counts_hardlinked_blob_and_cache_bytes_once() {
        let root = temp_dir("prune-hardlinked-cache");
        let blobs = root.join("blobs").join("sha256");
        let cache = root.join("cache");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(cache.join("bbbb")).unwrap();
        let blob = blobs.join("aaaa");
        let cache_file = cache.join("bbbb").join("model.safetensors");
        let weights = b"complete-weights-bytes";
        std::fs::write(&blob, weights).unwrap();
        std::fs::hard_link(&blob, &cache_file).unwrap();

        let live = HashSet::new();
        let (blob_stats, cache_stats) =
            prune_blobs_and_cache(&root, &cache, &live, Duration::ZERO).unwrap();

        assert_eq!(blob_stats.count, 1);
        assert_eq!(cache_stats.count, 1);
        assert_eq!(blob_stats.bytes + cache_stats.bytes, weights.len() as u64);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn prune_counts_duplicate_nested_cache_links_once() {
        let root = temp_dir("prune-duplicate-nested-cache");
        let blobs = root.join("blobs").join("sha256");
        let cache = root.join("cache");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(cache.join("bbbb").join("sub")).unwrap();
        let blob = blobs.join("aaaa");
        let weights = b"complete-weights-bytes";
        std::fs::write(&blob, weights).unwrap();
        std::fs::hard_link(&blob, cache.join("bbbb").join("model.safetensors")).unwrap();
        std::fs::hard_link(
            &blob,
            cache.join("bbbb").join("sub").join("model.safetensors"),
        )
        .unwrap();

        let live = HashSet::new();
        let (blob_stats, cache_stats) =
            prune_blobs_and_cache(&root, &cache, &live, Duration::ZERO).unwrap();

        assert_eq!(blob_stats.count, 1);
        assert_eq!(cache_stats.count, 1);
        assert_eq!(blob_stats.bytes + cache_stats.bytes, weights.len() as u64);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn prune_does_not_count_cache_link_to_live_blob_as_freed_bytes() {
        let root = temp_dir("prune-cache-link-to-live-blob");
        let blobs = root.join("blobs").join("sha256");
        let cache = root.join("cache");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(cache.join("bbbb")).unwrap();
        let blob = blobs.join("aaaa");
        let weights = b"complete-weights-bytes";
        std::fs::write(&blob, weights).unwrap();
        std::fs::hard_link(&blob, cache.join("bbbb").join("model.safetensors")).unwrap();

        let mut live = HashSet::new();
        live.insert("sha256:aaaa".to_string());
        let (blob_stats, cache_stats) =
            prune_blobs_and_cache(&root, &cache, &live, Duration::ZERO).unwrap();

        assert_eq!(blob_stats.count, 0);
        assert_eq!(cache_stats.count, 1);
        assert_eq!(blob_stats.bytes + cache_stats.bytes, 0);
        assert!(blob.exists(), "live blob survives");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn prune_does_not_count_blob_link_retained_by_fresh_cache_dir() {
        let root = temp_dir("prune-blob-link-retained-by-cache");
        let blobs = root.join("blobs").join("sha256");
        let cache = root.join("cache");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(cache.join("bbbb")).unwrap();
        let blob = blobs.join("aaaa");
        let cache_file = cache.join("bbbb").join("model.safetensors");
        let weights = b"complete-weights-bytes";
        std::fs::write(&blob, weights).unwrap();
        std::fs::hard_link(&blob, &cache_file).unwrap();

        let old = SystemTime::now() - GC_GRACE_PERIOD - Duration::from_secs(60);
        filetime_set(&blob, old);

        let live = HashSet::new();
        let (blob_stats, cache_stats) =
            prune_blobs_and_cache(&root, &cache, &live, GC_GRACE_PERIOD).unwrap();

        assert_eq!(blob_stats.count, 1);
        assert_eq!(cache_stats.count, 0);
        assert_eq!(blob_stats.bytes + cache_stats.bytes, 0);
        assert!(!blob.exists(), "old unreferenced blob path is removed");
        assert!(
            cache_file.exists(),
            "fresh cache dir keeps its hardlink to the bytes"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn remember_cache_dir_files_ignores_missing_dirs_but_returns_other_read_errors() {
        let root = temp_dir("remember-cache-read-error");
        std::fs::create_dir_all(&root).unwrap();
        remember_cache_dir_files(
            &root.join("missing"),
            Duration::ZERO,
            &mut GcFileAccount::default(),
        )
        .unwrap();

        let file = root.join("not-a-directory");
        std::fs::write(&file, b"cache entry").unwrap();

        let error = remember_cache_dir_files(&file, Duration::ZERO, &mut GcFileAccount::default())
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains(&format!("read {}", file.display())),
            "unexpected error: {error:#}"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn remember_ignores_missing_paths_but_returns_other_metadata_errors() {
        let root = temp_dir("remember-metadata-error");
        std::fs::create_dir_all(&root).unwrap();
        let mut account = GcFileAccount::default();
        account.remember(&root.join("missing")).unwrap();

        let not_a_dir = root.join("not-a-directory");
        std::fs::write(&not_a_dir, b"cache entry").unwrap();
        let child = not_a_dir.join("child");

        let error = account.remember(&child).unwrap_err();

        assert!(
            error
                .to_string()
                .contains(&format!("inspect {}", child.display())),
            "unexpected error: {error:#}"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn remember_cache_dir_files_includes_non_utf8_names() {
        use std::os::unix::ffi::OsStringExt as _;

        let root = temp_dir("remember-cache-non-utf8");
        std::fs::create_dir_all(&root).unwrap();
        let file = root.join(std::ffi::OsString::from_vec(vec![0xff]));
        std::fs::write(&file, b"cache entry").unwrap();
        let mut account = GcFileAccount::default();

        remember_cache_dir_files(&root, Duration::ZERO, &mut account).unwrap();

        assert_eq!(account.size(&file), 0, "retained file was not remembered");
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A corrupt (unparsable) manifest pointer file must abort the live-set
    /// computation rather than silently omitting that model — otherwise its
    /// blobs would look unreferenced and a destructive sweep would delete
    /// them. Fail closed: an error, so callers prune nothing.
    #[test]
    fn referenced_digests_aborts_on_an_unreadable_reference() {
        let root = temp_dir("referenced-digests-strict");
        let store = OciStore::open(&root).unwrap();

        // A real, healthy tagged model.
        let cfg = store
            .write_blob("application/vnd.cncf.model.config.v1+json", b"{}")
            .unwrap();
        let manifest = crate::storage::oci::Manifest {
            schema_version: 2,
            media_type: "application/vnd.oci.image.manifest.v1+json".into(),
            artifact_type: None,
            config: cfg,
            layers: vec![],
            annotations: None,
        };
        let desc = store.write_manifest(&manifest).unwrap();
        store.tag(desc, "hf.co/ai/healthy:latest").unwrap();

        // A corrupt pointer file elsewhere in manifests/ — enumeration
        // can't parse it, so the whole scan must fail rather than proceed
        // with an incomplete live set.
        let corrupt = root
            .join("manifests")
            .join("hf.co")
            .join("ai")
            .join("broken")
            .join("latest");
        std::fs::create_dir_all(corrupt.parent().unwrap()).unwrap();
        std::fs::write(&corrupt, b"not json").unwrap();

        assert!(
            referenced_digests(&store).is_err(),
            "an unparsable reference must abort the live-set scan, not be skipped"
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn referenced_digests_keeps_differing_legacy_and_v2_manifests() {
        let root = temp_dir("referenced-digests-migration");
        let store = OciStore::open(&root).unwrap();
        let config = store.write_blob("application/json", b"{}").unwrap();
        let old_layer = store
            .write_blob("application/octet-stream", b"old layer")
            .unwrap();
        let new_layer = store
            .write_blob("application/octet-stream", b"new layer")
            .unwrap();
        let make_manifest = |layer| crate::storage::oci::Manifest {
            schema_version: 2,
            media_type: "application/vnd.oci.image.manifest.v1+json".into(),
            artifact_type: None,
            config: config.clone(),
            layers: vec![layer],
            annotations: None,
        };
        let old = store
            .write_manifest(&make_manifest(old_layer.clone()))
            .unwrap();
        let new = store
            .write_manifest(&make_manifest(new_layer.clone()))
            .unwrap();
        let reference = format!("docker.io/ai/model:latest@{}", old.digest);
        store.tag(new.clone(), &reference).unwrap();

        let mut old_ref = old.clone();
        old_ref.annotations = Some(std::collections::HashMap::from([(
            "org.opencontainers.image.ref.name".into(),
            reference.clone(),
        )]));
        let legacy = crate::storage::oci::legacy_manifest_ref_path(&root, &reference);
        std::fs::create_dir_all(legacy.parent().unwrap()).unwrap();
        std::fs::write(&legacy, serde_json::to_vec(&old_ref).unwrap()).unwrap();

        assert_eq!(store.find(&reference).unwrap().digest, old.digest);
        let live = referenced_digests(&store).unwrap();
        for digest in [
            &old.digest,
            &new.digest,
            &old_layer.digest,
            &new_layer.digest,
        ] {
            assert!(live.contains(digest), "GC omitted {digest}");
        }
        assert_eq!(prune_blobs(&root, &live, Duration::ZERO).unwrap().count, 0);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn prune_is_a_no_op_on_missing_directories() {
        let root = temp_dir("missing");
        let live = HashSet::new();
        assert_eq!(prune_blobs(&root, &live, Duration::ZERO).unwrap().count, 0);
        assert_eq!(
            prune_cache(&root.join("cache"), &live, Duration::ZERO)
                .unwrap()
                .count,
            0
        );
    }

    #[cfg(unix)]
    fn blob(store: &OciStore, digest: &str) -> std::path::PathBuf {
        let hex = digest.strip_prefix("sha256:").unwrap();
        store.root().join("blobs").join("sha256").join(hex)
    }

    /// A one-layer model in `store`, just written: layer, then config, then
    /// manifest, the order a pull writes them in.
    #[cfg(unix)]
    fn fresh_model(
        store: &OciStore,
        config: &[u8],
        layer: &[u8],
    ) -> (
        crate::storage::oci::Descriptor,
        crate::storage::oci::Manifest,
    ) {
        let layer = store.write_blob("application/octet-stream", layer).unwrap();
        let cfg = store
            .write_blob("application/vnd.cncf.model.config.v1+json", config)
            .unwrap();
        let manifest = crate::storage::oci::Manifest {
            schema_version: 2,
            media_type: "application/vnd.oci.image.manifest.v1+json".into(),
            artifact_type: None,
            config: cfg,
            layers: vec![layer],
            annotations: None,
        };
        (store.write_manifest(&manifest).unwrap(), manifest)
    }

    /// A one-layer model in `store`, its blobs backdated past the grace window.
    #[cfg(unix)]
    fn aged_model(store: &OciStore, layer: &[u8]) -> crate::storage::oci::Descriptor {
        let (desc, manifest) = fresh_model(store, &[b"config-", layer].concat(), layer);
        let old = SystemTime::now() - GC_GRACE_PERIOD - Duration::from_secs(60);
        for digest in [
            &manifest.layers[0].digest,
            &desc.digest,
            &manifest.config.digest,
        ] {
            filetime_set(&blob(store, digest), old);
        }
        desc
    }

    /// A temp root with a cache directory path and a store.
    #[cfg(unix)]
    fn rm_fixture(label: &str) -> (std::path::PathBuf, std::path::PathBuf, OciStore) {
        let root = temp_dir(label);
        let store = OciStore::open(root.join("store")).unwrap();
        (root.clone(), root.join("cache"), store)
    }

    /// [`fresh_model`], tagged as `reference`.
    #[cfg(unix)]
    fn tagged_model(
        store: &OciStore,
        reference: &str,
        config: &[u8],
        layer: &[u8],
    ) -> (
        crate::storage::oci::Descriptor,
        crate::storage::oci::Manifest,
    ) {
        let (desc, manifest) = fresh_model(store, config, layer);
        store.tag(desc.clone(), reference).unwrap();
        (desc, manifest)
    }

    /// The cache directory for `digest`, created.
    #[cfg(unix)]
    fn cache_dir(cache: &Path, digest: &str) -> std::path::PathBuf {
        let dir = cache.join(digest.strip_prefix("sha256:").unwrap());
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Takes the pre-untag snapshot, untags `reference`, reclaims.
    #[cfg(unix)]
    fn rm_and_reclaim(
        store: &OciStore,
        cache: &Path,
        reference: &str,
    ) -> anyhow::Result<(GcStats, GcStats)> {
        let before = held_by_manifests(store)?;
        store.remove(reference)?;
        reclaim_removed(store, cache, &before)
    }

    #[cfg(unix)]
    #[test]
    fn rm_frees_a_just_written_model_at_once_counting_hardlinks_once() {
        let (root, cache, store) = rm_fixture("rm-fresh");
        let reference = "docker.io/ai/m:latest";
        let (desc, manifest) = tagged_model(&store, reference, b"config", b"fresh weights");
        let layer = manifest.layers[0].digest.clone();
        let layer_cache = cache_dir(&cache, &layer);
        std::fs::hard_link(blob(&store, &layer), layer_cache.join("model.gguf")).unwrap();
        let on_disk: u64 = [&desc.digest, &manifest.config.digest, &layer]
            .iter()
            .map(|d| std::fs::metadata(blob(&store, d)).unwrap().len())
            .sum();

        let (blobs, caches) = rm_and_reclaim(&store, &cache, reference).unwrap();

        assert_eq!((blobs.count, caches.count), (3, 1));
        assert_eq!(blobs.bytes + caches.bytes, on_disk);
        for gone in [&desc.digest, &manifest.config.digest, &layer] {
            assert!(!blob(&store, gone).exists(), "{gone}");
        }
        assert!(!layer_cache.exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn rm_keeps_what_a_surviving_tag_shares() {
        let (root, cache, store) = rm_fixture("rm-shared");
        let (gone, gone_manifest) = tagged_model(
            &store,
            "docker.io/ai/a:latest",
            b"config-a",
            b"shared weights",
        );
        let (kept, _) = tagged_model(
            &store,
            "docker.io/ai/b:latest",
            b"config-b",
            b"shared weights",
        );
        let shared = gone_manifest.layers[0].digest.clone();
        let shared_cache = cache_dir(&cache, &shared);

        let (blobs, caches) = rm_and_reclaim(&store, &cache, "docker.io/ai/a:latest").unwrap();

        assert_eq!((blobs.count, caches.count), (2, 0));
        assert!(!blob(&store, &gone.digest).exists());
        assert!(!blob(&store, &gone_manifest.config.digest).exists());
        assert!(blob(&store, &shared).exists(), "the shared layer survives");
        assert!(shared_cache.exists(), "so does its cache directory");
        assert!(blob(&store, &kept.digest).exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn rm_leaves_unrelated_fresh_orphans_to_the_grace_sweep() {
        let (root, cache, store) = rm_fixture("rm-orphan");
        tagged_model(&store, "docker.io/ai/m:latest", b"config", b"weights");
        // What an in-flight pull in another process has written so far.
        let inflight = store
            .write_blob("application/octet-stream", b"layer of a pull in flight")
            .unwrap();
        let inflight_cache = cache_dir(&cache, &inflight.digest);

        let (blobs, caches) = rm_and_reclaim(&store, &cache, "docker.io/ai/m:latest").unwrap();

        assert_eq!((blobs.count, caches.count), (3, 0));
        assert!(blob(&store, &inflight.digest).exists());
        assert!(inflight_cache.exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn rm_keeps_a_candidate_a_writer_reused_after_the_removed_manifest() {
        let (root, cache, store) = rm_fixture("rm-reused");
        let (desc, manifest) = tagged_model(&store, "docker.io/ai/m:latest", b"config", b"weights");
        // The manifest and config are older than the layer's last touch,
        // as if a concurrent writer reused the layer after the tag landed.
        let now = SystemTime::now();
        filetime_set(&blob(&store, &desc.digest), now - Duration::from_secs(60));
        filetime_set(
            &blob(&store, &manifest.config.digest),
            now - Duration::from_secs(120),
        );

        let (blobs, _) = rm_and_reclaim(&store, &cache, "docker.io/ai/m:latest").unwrap();

        assert_eq!(blobs.count, 2);
        assert!(blob(&store, &manifest.layers[0].digest).exists());
        assert!(!blob(&store, &desc.digest).exists());
        assert!(!blob(&store, &manifest.config.digest).exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn rm_keeps_a_manifest_republished_after_the_snapshot_and_its_layers() {
        let (root, cache, store) = rm_fixture("rm-republished");
        let reference = "docker.io/ai/m:latest";
        let (desc, manifest) = tagged_model(&store, reference, b"config", b"weights");
        let before = held_by_manifests(&store).unwrap();
        store.remove(reference).unwrap();
        // A same-content pull reused the blobs, layer first, between the
        // untag and its own tag write.
        let later = SystemTime::now() + Duration::from_secs(30);
        filetime_set(&blob(&store, &manifest.layers[0].digest), later);
        filetime_set(&blob(&store, &desc.digest), later);

        let (blobs, caches) = reclaim_removed(&store, &cache, &before).unwrap();

        assert_eq!((blobs.count, caches.count), (0, 0));
        for kept in [
            &desc.digest,
            &manifest.config.digest,
            &manifest.layers[0].digest,
        ] {
            assert!(blob(&store, kept).exists(), "{kept}");
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn rm_deletes_nothing_when_the_surviving_set_is_unreadable() {
        let (root, cache, store) = rm_fixture("rm-fail-closed");
        let (desc, manifest) = tagged_model(&store, "docker.io/ai/m:latest", b"config", b"weights");
        let (other, _) = tagged_model(
            &store,
            "docker.io/ai/other:latest",
            b"other config",
            b"other weights",
        );
        let before = held_by_manifests(&store).unwrap();
        store.remove("docker.io/ai/m:latest").unwrap();
        // The surviving manifest becomes unreadable.
        std::fs::write(blob(&store, &other.digest), b"not json").unwrap();

        assert!(reclaim_removed(&store, &cache, &before).is_err());

        for kept in [
            &desc.digest,
            &manifest.config.digest,
            &manifest.layers[0].digest,
        ] {
            assert!(blob(&store, kept).exists(), "{kept}");
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_retag_to_other_content_reclaims_what_only_the_old_content_held() {
        let root = temp_dir("after-retag");
        let cache = root.join("cache");
        let store = OciStore::open(root.join("store")).unwrap();
        let reference = "docker.io/ai/m:latest";
        let old = aged_model(&store, b"old");
        store.tag(old.clone(), reference).unwrap();
        let old_manifest = store.read_manifest(&old.digest).unwrap();
        let old_layer = old_manifest.layers[0].digest.clone();
        let old_cache = cache.join(old_layer.strip_prefix("sha256:").unwrap());
        std::fs::create_dir_all(&old_cache).unwrap();
        filetime_set(
            &old_cache,
            SystemTime::now() - GC_GRACE_PERIOD - Duration::from_secs(60),
        );
        let shared = aged_model(&store, b"shared");
        store.tag(shared, "docker.io/ai/other:latest").unwrap();

        let new = aged_model(&store, b"new");
        after_retag(&store, &cache, reference, || store.tag(new, reference)).unwrap();

        for gone in [&old.digest, &old_layer, &old_manifest.config.digest] {
            assert!(!blob(&store, gone).exists(), "{gone}");
        }
        assert!(!old_cache.exists());
        let live = referenced_digests(&store).unwrap();
        assert_eq!(live.len(), 6);
        assert!(live.iter().all(|d| blob(&store, d).exists()));
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_new_tag_or_the_same_content_sweeps_nothing() {
        let root = temp_dir("after-retag-noop");
        let store = OciStore::open(&root).unwrap();
        let orphan = aged_model(&store, b"orphan");
        let desc = aged_model(&store, b"kept");
        let reference = "docker.io/ai/m:latest";

        after_retag(&store, &root.join("cache"), reference, || {
            store.tag(desc.clone(), reference)
        })
        .unwrap();
        after_retag(&store, &root.join("cache"), reference, || {
            store.tag(desc.clone(), reference)
        })
        .unwrap();

        assert!(blob(&store, &orphan.digest).exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Minimal mtime-backdating helper — mirrors the one in
    /// `storage::repair`'s tests, avoiding a `filetime` dependency just for
    /// this. Unix-only, like that one.
    #[cfg(unix)]
    fn filetime_set(path: &Path, when: SystemTime) {
        use std::os::unix::io::AsRawFd;
        let file = std::fs::File::open(path).unwrap();
        let d = when.duration_since(SystemTime::UNIX_EPOCH).unwrap();
        let tv = libc::timeval {
            tv_sec: d.as_secs() as _,
            tv_usec: d.subsec_micros() as _,
        };
        let times = [tv, tv];
        unsafe {
            libc::futimes(file.as_raw_fd(), times.as_ptr());
        }
    }
}
