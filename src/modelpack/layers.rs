//! How a ModelPack layer's blob holds its file(s), and unpacking it.
//!
//! The spec (<https://github.com/modelpack/model-spec>) gives each layer
//! category (`weight`, `weight.config`, `doc`, `code`, `dataset`) four
//! encodings: `.raw` (the file itself), `.tar`, `.tar+gzip`, `.tar+zstd`.
//!
//! Tar entries are named relative to the model root, as in modctl's `Untar`;
//! the layer's `org.cncf.model.filepath` is not consulted. The archive is
//! untrusted: only regular files and directories are created, and a path
//! that is absolute or leaves the root fails the layer.

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context};

use crate::storage::oci::Descriptor;

use super::{cache_layer_file, digest_hex, raw_blob_path};

/// A tar layer is unpacked into `<cache>/<layer hex>/UNPACKED/FILES`; the
/// completion sentinel sits in `UNPACKED`, out of the archive's reach.
const UNPACKED: &str = "unpacked";
const FILES: &str = "files";

fn completed_files(unpacked: &Path) -> Option<PathBuf> {
    crate::managed::is_complete(unpacked).then(|| unpacked.join(FILES))
}

/// The files of the tar layer in `layer_dir`, if it was fully unpacked.
pub(super) fn extracted_files(layer_dir: &Path) -> Option<PathBuf> {
    completed_files(&layer_dir.join(UNPACKED))
}

/// The compression of a tar layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Compression {
    None,
    Gzip,
    Zstd,
}

/// What a layer's blob is: the file itself, or a tar archive of files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Encoding {
    Raw,
    Tar(Compression),
}

/// A ModelPack layer media type taken apart: whether serving needs the layer
/// (weights or their config) and how its blob is encoded. `None` for others.
pub(super) fn parse_media_type(media_type: &str) -> Option<(bool, Encoding)> {
    let rest = media_type.strip_prefix("application/vnd.cncf.model.")?;
    let (category, suffix) = rest.split_once(".v1.")?;
    let servable = match category {
        "weight" | "weight.config" => true,
        "doc" | "code" | "dataset" => false,
        _ => return None,
    };
    let encoding = match suffix {
        "raw" => Encoding::Raw,
        "tar" => Encoding::Tar(Compression::None),
        "tar+gzip" => Encoding::Tar(Compression::Gzip),
        "tar+zstd" => Encoding::Tar(Compression::Zstd),
        _ => return None,
    };
    Some((servable, encoding))
}

/// The compression of `layer` if it is a tar archive; none if its type does not say.
pub(super) fn tar_compression(layer: &Descriptor) -> Compression {
    match parse_media_type(&layer.media_type) {
        Some((_, Encoding::Tar(compression))) => compression,
        _ => Compression::None,
    }
}

/// The decompressed bytes of `layer`'s blob. Gzip is read as a multi-member
/// stream and zstd frame by frame, as Go's readers do.
pub(super) fn open_layer(
    store_path: &Path,
    layer: &Descriptor,
    compression: Compression,
) -> anyhow::Result<Box<dyn Read>> {
    let path = raw_blob_path(store_path, layer)?;
    let file = std::fs::File::open(&path).with_context(|| format!("open {}", path.display()))?;
    Ok(match compression {
        Compression::None => Box::new(BufReader::with_capacity(1 << 20, file)),
        Compression::Gzip => Box::new(flate2::read::MultiGzDecoder::new(file)),
        Compression::Zstd => Box::new(zstd::stream::read::Decoder::new(file)?),
    })
}

/// Unpacks the tar stream `reader` into `dest`. Files get mode 0755 if the
/// archive sets an executable bit, else 0644, whatever the umask. Symlinks and
/// hard links are skipped (as in modctl's `Untar`); devices and FIFOs are an error.
pub(super) fn unpack_tar(reader: impl Read, dest: &Path) -> anyhow::Result<()> {
    let mut archive = tar::Archive::new(reader);
    for entry in archive.entries().context("read tar archive")? {
        let mut entry = entry.context("read tar entry")?;
        let raw = String::from_utf8(entry.path_bytes().into_owned())
            .map_err(|e| anyhow!("tar entry name {:?} is not UTF-8", e.into_bytes()))?;
        let kind = entry.header().entry_type();
        if kind.is_pax_global_extensions() {
            continue;
        }
        if kind.is_symlink() || kind.is_hard_link() {
            eprintln!("[llmman] skipping link {raw:?} in a model layer");
            continue;
        }
        let is_dir = kind.is_dir() || (kind.is_file() && raw.ends_with('/'));
        if !(is_dir || kind.is_file() || kind.is_contiguous()) {
            bail!("tar entry {raw:?} is not a regular file or directory");
        }
        let Some(name) = entry_name(&raw)? else {
            continue;
        };
        let target = dest.join(&name);
        if is_dir {
            std::fs::create_dir_all(&target)
                .with_context(|| format!("create {}", target.display()))?;
            continue;
        }
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        let executable = entry.header().mode().is_ok_and(|mode| mode & 0o111 != 0);
        write_entry(&mut entry, &target, executable)
            .with_context(|| format!("write {}", target.display()))?;
    }
    Ok(())
}

/// The path of a tar entry relative to the model root, `None` for the root
/// itself. An error, not a skip, if it could land anywhere else.
fn entry_name(raw: &str) -> anyhow::Result<Option<String>> {
    if raw.starts_with('/') {
        bail!("tar entry {raw:?} has an absolute path");
    }
    let mut name = raw;
    while let Some(rest) = name.strip_prefix("./") {
        name = rest;
    }
    let name = name.trim_end_matches('/');
    if name.is_empty() || name == "." {
        return Ok(None);
    }
    if !crate::sources::is_safe_relative_path(name) {
        bail!("tar entry {raw:?} has an unsafe path");
    }
    Ok(Some(name.to_owned()))
}

fn write_entry(entry: &mut impl Read, target: &Path, executable: bool) -> anyhow::Result<()> {
    let file = std::fs::File::create(target)?;
    // Outright, not through the umask: the reader may be another user.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(if executable {
            0o755
        } else {
            0o644
        }))?;
    }
    #[cfg(not(unix))]
    let _ = executable;
    let mut out = BufWriter::with_capacity(1 << 20, file);
    std::io::copy(entry, &mut out)?;
    out.flush()?;
    Ok(())
}

/// Unpacks tar layer `layer` once under `<cache>/<layer hex>/` and returns
/// the directory of its files. Racing callers share the result; a crashed
/// extraction has no sentinel and is redone.
pub(super) fn extract_tar_layer(
    store_path: &Path,
    cache_path: &Path,
    layer: &Descriptor,
    compression: Compression,
) -> anyhow::Result<PathBuf> {
    let hex = digest_hex(&layer.digest)?;
    let root = cache_path.join(hex);
    crate::managed::ensure_dir(
        &root,
        &root.join(UNPACKED),
        &format!("model layer {}", &hex[..12]),
        completed_files,
        |unpacked| {
            eprintln!("[llmman] extracting layer {}", layer.digest);
            let files = unpacked.join(FILES);
            std::fs::create_dir(&files).with_context(|| format!("create {}", files.display()))?;
            unpack_tar(open_layer(store_path, layer, compression)?, &files)
                .with_context(|| format!("extract layer {}", layer.digest))?;
            Ok(files)
        },
    )
}

/// Places tar layer `layer`'s files under `root`, hardlinked from its
/// one extraction. Returns their paths relative to `root`, `/`-separated.
pub(super) fn place_tar_layer(
    store_path: &Path,
    cache_path: &Path,
    layer: &Descriptor,
    root: &Path,
) -> anyhow::Result<Vec<String>> {
    let dir = extract_tar_layer(store_path, cache_path, layer, tar_compression(layer))?;
    let mut placed = Vec::new();
    for entry in walkdir::WalkDir::new(&dir).min_depth(1).sort_by_file_name() {
        let entry = entry?;
        let rel = entry.path().strip_prefix(&dir)?;
        let dest = root.join(rel);
        if entry.file_type().is_dir() {
            std::fs::create_dir_all(&dest).with_context(|| format!("create {}", dest.display()))?;
        } else if entry.file_type().is_file() {
            cache_layer_file(entry.path(), &dest, entry.metadata()?.len())
                .with_context(|| format!("cache {} from layer {}", rel.display(), layer.digest))?;
            placed.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
    Ok(placed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modelpack::tests::gzip;

    /// A tar archive of `(name, type, body)` entries, written with
    /// `tar::Builder` and raw headers so a hostile name survives it.
    fn tar_of(entries: &[(&str, tar::EntryType, &[u8])]) -> Vec<u8> {
        tar_of_links(entries, "/tmp")
    }

    /// [`tar_of`] with the target every link entry points at.
    fn tar_of_links(entries: &[(&str, tar::EntryType, &[u8])], link_to: &str) -> Vec<u8> {
        let mut out = Vec::new();
        for (name, kind, body) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_entry_type(*kind);
            header.set_size(body.len() as u64);
            header.set_mode(0o644);
            // `set_path` refuses `..` and absolute names; the archives under
            // test are exactly the ones that carry them.
            let bytes = name.as_bytes();
            header.as_old_mut().name[..bytes.len()].copy_from_slice(bytes);
            if kind.is_symlink() || kind.is_hard_link() {
                header.set_link_name(link_to).unwrap();
            }
            header.set_cksum();
            out.extend_from_slice(header.as_bytes());
            out.extend_from_slice(body);
            let pad = (512 - body.len() % 512) % 512;
            out.resize(out.len() + pad, 0);
        }
        out.resize(out.len() + 1024, 0);
        out
    }

    use tar::EntryType::{Block, Char, Directory, Fifo, Link, Regular, Symlink};

    fn unpack(
        entries: &[(&str, tar::EntryType, &[u8])],
    ) -> (tempfile::TempDir, anyhow::Result<()>) {
        unpack_links(entries, "/tmp")
    }

    fn unpack_links(
        entries: &[(&str, tar::EntryType, &[u8])],
        link_to: &str,
    ) -> (tempfile::TempDir, anyhow::Result<()>) {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("root");
        std::fs::create_dir_all(&dest).unwrap();
        let result = unpack_tar(std::io::Cursor::new(tar_of_links(entries, link_to)), &dest);
        (dir, result)
    }

    #[test]
    fn media_types_cover_every_category_and_encoding_of_the_spec() {
        let cat = |c: &str, s: &str| format!("application/vnd.cncf.model.{c}.v1.{s}");
        for (category, servable) in [
            ("weight", true),
            ("weight.config", true),
            ("doc", false),
            ("code", false),
            ("dataset", false),
        ] {
            for (suffix, encoding) in [
                ("raw", Encoding::Raw),
                ("tar", Encoding::Tar(Compression::None)),
                ("tar+gzip", Encoding::Tar(Compression::Gzip)),
                ("tar+zstd", Encoding::Tar(Compression::Zstd)),
            ] {
                assert_eq!(
                    parse_media_type(&cat(category, suffix)),
                    Some((servable, encoding)),
                    "{category} {suffix}"
                );
            }
        }
        for other in [
            "application/vnd.cncf.model.config.v1+json",
            "application/vnd.cncf.model.weight.v1.rar",
            "application/vnd.cncf.model.weight.v2.tar",
            "application/vnd.cncf.model.extra.v1.tar",
            "application/vnd.docker.ai.gguf.v3",
            "application/vnd.oci.image.layer.v1.tar",
        ] {
            assert_eq!(parse_media_type(other), None, "{other}");
        }
    }

    #[test]
    fn unpacks_files_and_directories_relative_to_the_root() {
        let (dir, result) = unpack(&[
            ("./", Directory, b""),
            ("config.json", Regular, b"{}"),
            ("./tokenizer/", Directory, b""),
            ("./tokenizer/tokenizer.json", Regular, b"tok"),
            ("empty/", Directory, b""),
            ("deep/er/file.bin", Regular, b"x"),
        ]);
        result.unwrap();
        let root = dir.path().join("root");
        assert_eq!(std::fs::read(root.join("config.json")).unwrap(), b"{}");
        assert_eq!(
            std::fs::read(root.join("tokenizer/tokenizer.json")).unwrap(),
            b"tok"
        );
        assert!(root.join("empty").is_dir());
        assert_eq!(std::fs::read(root.join("deep/er/file.bin")).unwrap(), b"x");
    }

    #[test]
    fn a_path_that_could_leave_the_root_fails_the_layer() {
        for name in [
            "../escape.txt",
            "a/../../escape.txt",
            "a/../b",
            "/tmp/escape.txt",
            "//etc/x",
            "C:evil.txt",
            "..\\escape.txt",
            "a//b",
        ] {
            let (dir, result) = unpack(&[("ok.txt", Regular, b"ok"), (name, Regular, b"pwned")]);
            assert!(result.is_err(), "{name:?} was accepted");
            assert!(!dir.path().join("escape.txt").exists(), "{name:?}");
            assert!(!dir.path().join("root").join("b").exists(), "{name:?}");
        }
    }

    #[test]
    fn links_are_skipped_so_nothing_can_be_written_through_one() {
        let outside = tempfile::tempdir().unwrap();
        let (dir, result) = unpack_links(
            &[
                ("link", Symlink, b""),
                ("hard", Link, b""),
                ("link/pwned.txt", Regular, b"pwned"),
                ("real.txt", Regular, b"real"),
            ],
            outside.path().to_str().unwrap(),
        );
        result.unwrap();
        let root = dir.path().join("root");
        // `link/pwned.txt` became a real directory inside the root; the
        // symlink's target never saw it
        assert!(!std::fs::symlink_metadata(root.join("link"))
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(
            std::fs::read(root.join("link/pwned.txt")).unwrap(),
            b"pwned"
        );
        assert!(!root.join("hard").exists());
        assert_eq!(std::fs::read(root.join("real.txt")).unwrap(), b"real");
        assert!(std::fs::read_dir(outside.path()).unwrap().next().is_none());
    }

    #[test]
    fn devices_and_fifos_fail_the_layer() {
        for kind in [Char, Block, Fifo] {
            let (_dir, result) = unpack(&[("dev/null", kind, b"")]);
            let err = result.unwrap_err().to_string();
            assert!(err.contains("not a regular file"), "{kind:?}: {err}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn files_are_world_readable_and_keep_only_the_executable_bit() {
        use std::os::unix::fs::PermissionsExt;
        let mut out = Vec::new();
        for (name, mode) in [
            ("private.bin", 0o600),
            ("script.sh", 0o750),
            ("setuid", 0o4755),
        ] {
            let mut header = tar::Header::new_gnu();
            header.set_path(name).unwrap();
            header.set_size(1);
            header.set_mode(mode);
            header.set_cksum();
            out.extend_from_slice(header.as_bytes());
            out.extend_from_slice(&[b'x'; 1]);
            out.resize(out.len() + 511, 0);
        }
        out.resize(out.len() + 1024, 0);
        let dir = tempfile::tempdir().unwrap();
        unpack_tar(std::io::Cursor::new(out), dir.path()).unwrap();
        let mode = |n: &str| dir.path().join(n).metadata().unwrap().permissions().mode() & 0o7777;
        // exact, whatever the umask of the process running the test
        assert_eq!(mode("private.bin"), 0o644);
        assert_eq!(mode("script.sh"), 0o755);
        assert_eq!(mode("setuid"), 0o755);
    }

    #[test]
    fn gnu_long_names_are_read() {
        let long = format!("{}/model.safetensors", "d".repeat(150));
        let dir = tempfile::tempdir().unwrap();
        let mut builder = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_gnu();
        header.set_size(3);
        header.set_mode(0o644);
        builder
            .append_data(&mut header, &long, &b"abc"[..])
            .unwrap();
        let bytes = builder.into_inner().unwrap();
        unpack_tar(std::io::Cursor::new(bytes), dir.path()).unwrap();
        assert_eq!(std::fs::read(dir.path().join(&long)).unwrap(), b"abc");
    }

    // ── compressed layers, through the store ───────────────────────────────

    fn store_with_layer(
        media_type: &str,
        blob: &[u8],
    ) -> (tempfile::TempDir, PathBuf, PathBuf, Descriptor) {
        let dir = tempfile::tempdir().unwrap();
        let store = crate::storage::OciStore::open(dir.path().join("store")).unwrap();
        let layer = store.write_blob(media_type, blob).unwrap();
        let store_path = store.root().to_path_buf();
        let cache = dir.path().join("cache");
        (dir, store_path, cache, layer)
    }

    fn one_file_tar() -> Vec<u8> {
        tar_of(&[
            ("tokenizer", Directory, b""),
            ("tokenizer/tokenizer.json", Regular, b"{\"tok\":1}"),
        ])
    }

    #[test]
    fn extracts_every_compression_to_the_same_files() {
        let plain = one_file_tar();
        // zstd as a multi-frame stream with a skippable frame in front, the
        // shape of `zstd:chunked` and of concurrent encoders.
        let (a, b) = plain.split_at(700);
        let mut multi = vec![0x50, 0x2a, 0x4d, 0x18, 4, 0, 0, 0, 1, 2, 3, 4];
        multi.extend(zstd::encode_all(a, 3).unwrap());
        multi.extend(zstd::encode_all(b, 3).unwrap());
        let mut two_members = gzip(a);
        two_members.extend(gzip(b));
        let cases = [
            ("tar", plain.clone()),
            ("tar+gzip", gzip(&plain)),
            ("tar+gzip", two_members),
            ("tar+zstd", zstd::encode_all(&plain[..], 3).unwrap()),
            ("tar+zstd", multi),
        ];
        for (suffix, blob) in cases {
            let media = format!("application/vnd.cncf.model.weight.config.v1.{suffix}");
            let (_dir, store, cache, layer) = store_with_layer(&media, &blob);
            let dir = extract_tar_layer(&store, &cache, &layer, tar_compression(&layer)).unwrap();
            assert_eq!(
                std::fs::read(dir.join("tokenizer/tokenizer.json")).unwrap(),
                b"{\"tok\":1}",
                "{suffix}"
            );
            let root = _dir.path().join("tree");
            assert_eq!(
                place_tar_layer(&store, &cache, &layer, &root).unwrap(),
                ["tokenizer/tokenizer.json"],
                "{suffix}"
            );
        }
    }

    #[test]
    fn a_layer_that_is_not_what_its_media_type_says_fails() {
        let (_dir, store, cache, layer) = store_with_layer(
            "application/vnd.cncf.model.weight.v1.tar+zstd",
            b"this is not zstd",
        );
        assert!(extract_tar_layer(&store, &cache, &layer, tar_compression(&layer)).is_err());
        // and leaves nothing a later call could mistake for a finished layer
        let hex = digest_hex(&layer.digest).unwrap();
        assert!(extracted_files(&cache.join(hex)).is_none());
    }

    #[test]
    fn a_hostile_layer_fails_and_is_not_marked_complete() {
        let blob = tar_of(&[("ok.txt", Regular, b"ok"), ("../escape.txt", Regular, b"x")]);
        let (dir, store, cache, layer) =
            store_with_layer("application/vnd.cncf.model.weight.v1.tar", &blob);
        let err = extract_tar_layer(&store, &cache, &layer, Compression::None).unwrap_err();
        assert!(format!("{err:#}").contains("unsafe path"), "{err:#}");
        assert!(!dir.path().join("escape.txt").exists());
        assert!(!cache.join("escape.txt").exists());
        let hex = digest_hex(&layer.digest).unwrap();
        assert!(extracted_files(&cache.join(hex)).is_none());
    }

    #[test]
    fn an_entry_named_like_the_sentinel_cannot_mark_a_layer_complete() {
        let blob = tar_of(&[
            (".llmman-complete", Regular, b""),
            ("../escape.txt", Regular, b"x"),
        ]);
        let (_dir, store, cache, layer) =
            store_with_layer("application/vnd.cncf.model.weight.v1.tar", &blob);
        assert!(extract_tar_layer(&store, &cache, &layer, Compression::None).is_err());
        assert!(extracted_files(&cache.join(digest_hex(&layer.digest).unwrap())).is_none());
    }

    #[test]
    fn extraction_happens_once_and_a_partial_directory_is_redone() {
        let (_dir, store, cache, layer) = store_with_layer(
            "application/vnd.cncf.model.weight.v1.tar",
            &tar_of(&[("a.bin", Regular, b"aaa")]),
        );
        let hex = digest_hex(&layer.digest).unwrap();
        let unpacked = cache.join(hex).join(UNPACKED);
        let dest = unpacked.join(FILES);

        // a crashed earlier attempt: files, no sentinel
        std::fs::create_dir_all(&dest).unwrap();
        std::fs::write(dest.join("junk"), b"half").unwrap();
        assert_eq!(
            extract_tar_layer(&store, &cache, &layer, Compression::None).unwrap(),
            dest
        );
        assert!(!dest.join("junk").exists());
        assert_eq!(std::fs::read(dest.join("a.bin")).unwrap(), b"aaa");

        // complete: not unpacked again, whatever happened to the blob
        std::fs::write(dest.join("a.bin"), b"kept").unwrap();
        std::fs::remove_file(store.join("blobs/sha256").join(hex)).unwrap();
        extract_tar_layer(&store, &cache, &layer, Compression::None).unwrap();
        assert_eq!(std::fs::read(dest.join("a.bin")).unwrap(), b"kept");
    }

    #[test]
    fn concurrent_callers_get_one_complete_extraction() {
        let big = vec![7u8; 3 << 20];
        let (_dir, store, cache, layer) = store_with_layer(
            "application/vnd.cncf.model.weight.v1.tar+gzip",
            &gzip(&tar_of(&[("w.safetensors", Regular, &big)])),
        );
        let results: Vec<_> = std::thread::scope(|s| {
            (0..8)
                .map(|_| {
                    s.spawn(|| {
                        let dir = extract_tar_layer(&store, &cache, &layer, Compression::Gzip)?;
                        // never a partial file: the directory is only returned complete
                        let len = std::fs::metadata(dir.join("w.safetensors"))?.len();
                        Ok::<u64, anyhow::Error>(len)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect()
        });
        for r in results {
            assert_eq!(r.unwrap(), big.len() as u64);
        }
        let hex = digest_hex(&layer.digest).unwrap();
        let leftovers: Vec<_> = std::fs::read_dir(cache.join(hex))
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n != UNPACKED && n != ".lock")
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    #[cfg(unix)]
    #[test]
    fn placed_files_are_hardlinks_of_the_extracted_ones() {
        use std::os::unix::fs::MetadataExt;
        let (dir, store, cache, layer) = store_with_layer(
            "application/vnd.cncf.model.weight.v1.tar",
            &tar_of(&[("sub/a.bin", Regular, b"aaa")]),
        );
        let root = dir.path().join("tree");
        let placed = place_tar_layer(&store, &cache, &layer, &root).unwrap();
        assert_eq!(placed, ["sub/a.bin"]);
        let extracted = cache
            .join(digest_hex(&layer.digest).unwrap())
            .join(UNPACKED)
            .join(FILES)
            .join("sub/a.bin");
        let (a, b) = (
            root.join("sub/a.bin").metadata().unwrap(),
            extracted.metadata().unwrap(),
        );
        assert_eq!((a.dev(), a.ino()), (b.dev(), b.ino()));
        // idempotent
        assert_eq!(
            place_tar_layer(&store, &cache, &layer, &root).unwrap(),
            placed
        );
    }
}
