//! Short-name alias resolution — loaded from config files at runtime.
//!
//! Mirrors podman's approach: TOML files are read from a priority-ordered set
//! of locations; all files are merged with higher-priority entries winning.
//! Nothing is compiled into the binary.
//!
//! Search order (ascending priority — later files override earlier ones):
//!   1. /usr/share/llmman/shortnames.conf          distro / package default
//!   2. /etc/llmman/shortnames.conf                 system-admin override
//!   3. <binary>/../share/llmman/shortnames.conf    install-tree relative path
//!   4. <binary-dir>/shortnames.conf                development (conf beside binary)
//!   5. ~/.config/llmman/shortnames.conf            per-user aliases

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use serde::Deserialize;

#[derive(Deserialize, Default)]
struct Conf {
    #[serde(default)]
    aliases: HashMap<String, String>,
}

/// Return all candidate config-file paths in ascending priority order.
fn config_paths() -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = vec![
        PathBuf::from("/usr/share/llmman/shortnames.conf"),
        PathBuf::from("/etc/llmman/shortnames.conf"),
    ];

    // Paths relative to the running binary.
    if let Ok(exe) = std::env::current_exe() {
        // <binary>/../share/llmman/shortnames.conf  (standard install layout)
        if let Some(parent) = exe.parent() {
            paths.push(parent.join("../share/llmman/shortnames.conf"));
            // <binary-dir>/shortnames.conf  (development: cargo run / direct exec)
            paths.push(parent.join("shortnames.conf"));
        }
    }

    // ~/.config/llmman/shortnames.conf
    if let Some(cfg) = dirs::config_dir() {
        paths.push(cfg.join("llmman").join("shortnames.conf"));
    }

    paths
}

/// Load and merge aliases from all config files.
/// Higher-priority files (later in the list) override lower-priority ones.
fn load_aliases() -> HashMap<String, String> {
    let mut merged: HashMap<String, String> = HashMap::new();
    for path in config_paths() {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        match toml::from_str::<Conf>(&text) {
            Ok(conf) => {
                for (k, v) in conf.aliases {
                    merged.insert(k, v);
                }
            }
            Err(e) => {
                eprintln!("[llmman] warning: ignoring {}: {e}", path.display());
            }
        }
    }
    merged
}

fn aliases() -> &'static HashMap<String, String> {
    static CACHE: OnceLock<HashMap<String, String>> = OnceLock::new();
    CACHE.get_or_init(load_aliases)
}

/// Returns true if `reference` already carries an explicit registry host:
/// it has a "/" (so there's an actual leading path component to examine),
/// and that first component contains a dot or colon (a "host:port" form,
/// e.g. "localhost:5000" — a real repository name never contains a colon
/// itself, matching docker/distribution's own reference grammar) or
/// equals "localhost" outright. Requiring a "/" here matters: without
/// one, a slash-less reference like "qwen3.5:0.8B" would otherwise be
/// misread as "already has host qwen3.5:0.8B" just because its tag
/// separator is a colon, when there's no host/path structure there at
/// all — leaving it neither hf.co- nor docker.io/ai-prefixed, so it
/// reaches the Go shim raw and dead-ends in its HuggingFace-only parser
/// with a misleading error instead of either default ever being applied.
fn has_host(reference: &str) -> bool {
    match reference.split_once('/') {
        Some((first, _)) => {
            first.contains('.') || first.contains(':') || first.eq_ignore_ascii_case("localhost")
        }
        None => false,
    }
}

/// Resolve `reference` through the short-name alias table, then default the
/// registry to `hf.co` when no host is present.
///
/// URI scheme handling (processed before alias lookup):
///   hf:// huggingface://  → strip scheme, continue as bare owner/repo
///   ms:// modelscope://   → normalise to ms:// (Go shim routes to ModelScope)
///   ngc:// s3:// gs://    → pass through verbatim (Go shim handles natively)
///   /absolute/path        → pass through verbatim (local directory import)
///
/// Resolution order for everything else:
///   1. Exact alias match  → return the mapped value
///   2. Has a registry host → return as-is
///   3. No host            → prepend `hf.co/`
pub fn resolve(reference: &str) -> String {
    // ── URI schemes that bypass alias lookup and hf.co defaulting ─────────
    // Local absolute paths and object-store URIs are forwarded as-is to the
    // Go shim which dispatches them to the appropriate source handler.
    for passthrough in &["ngc://", "s3://", "gs://"] {
        if reference.starts_with(passthrough) {
            return reference.to_owned();
        }
    }
    if reference.starts_with('/') {
        return reference.to_owned();
    }

    // ── Normalise well-known URI schemes to canonical form ─────────────────
    // hf:// and huggingface:// are stripped; the remainder is treated as a
    // bare HuggingFace owner/repo reference through the normal path below.
    let reference = if let Some(r) = reference
        .strip_prefix("hf://")
        .or_else(|| reference.strip_prefix("huggingface://"))
    {
        r
    }
    // ms:// and modelscope:// are normalised to ms:// so the Go shim can
    // detect the scheme and route to the ModelScope download path.
    else if let Some(r) = reference.strip_prefix("modelscope://") {
        return format!("ms://{r}");
    } else if reference.starts_with("ms://") {
        return reference.to_owned();
    } else {
        reference
    };

    // ── Alias lookup → hf.co default ──────────────────────────────────────
    if let Some(mapped) = aliases().get(reference) {
        return mapped.clone();
    }
    if has_host(reference) {
        return reference.to_owned();
    }
    format!("hf.co/{reference}")
}

/// Returns true if `reference` is bare: no "/" at all, i.e. no owner/repo
/// or registry-host structure — just a single path component, optionally
/// with a ":tag". Dots are deliberately *not* checked here (unlike a
/// stricter earlier version of this function): a dotted version number
/// such as "3.5" in "qwen3.5:0.8B" is just part of the name/tag, not a
/// registry host, and real ollama makes the same distinction purely on
/// "/" — it never treats embedded dots specially. Since a single bare
/// component (with or without dots) can never satisfy HuggingFace's
/// required host/owner/repo shape anyway, sending it to `resolve`'s
/// hf.co default would be a guaranteed dead end; docker.io/ai/<reference>
/// below is the only default that's ever actually resolvable for it.
fn is_bare(reference: &str) -> bool {
    !reference.contains('/')
}

/// Resolve `reference` the way every Ollama-API-facing path in `cmd::serve`
/// does (handle_pull, handle_show, handle_delete, ensure_model, and the
/// `--model` preload in serve_async): identical to `resolve`, except a
/// *bare* reference — no "/" anywhere, e.g. "gemma4" or "qwen3.5:0.8B" —
/// defaults to Docker's official curated-model namespace on Docker Hub,
/// `docker.io/ai/<reference>` (e.g. "gemma4" -> "docker.io/ai/gemma4",
/// "qwen3.5:0.8B" -> "docker.io/ai/qwen3.5:0.8B"), instead of `resolve`'s
/// general `hf.co/<reference>` default. Dots in the name or tag don't
/// disqualify this — only a "/" does — since `resolve`'s hf.co default
/// requires a host/owner/repo shape that a single bare component can never
/// satisfy anyway. Any reference with a "/" (an owner/repo path, a URI
/// scheme, an explicit host) is left to `resolve`'s normal rules unchanged.
///
/// CLI subcommands that talk to a local server over the Ollama API (pull,
/// push) go through this same resolution server-side, so the docker.io/ai/
/// default is consistent regardless of whether a bare name reaches llmman
/// via the CLI or directly over HTTP.
pub fn resolve_ollama_api(reference: &str) -> String {
    if is_bare(reference) {
        if let Some(mapped) = aliases().get(reference) {
            return mapped.clone();
        }
        return format!("docker.io/ai/{reference}");
    }
    resolve(reference)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_ollama_api_defaults_bare_names_to_docker_ai() {
        assert_eq!(resolve_ollama_api("gemma4"), "docker.io/ai/gemma4");
        // A tag with no dot is still "bare" by this rule (only "/"
        // disqualifies it) — matches the ai/<name>:<tag> shape on Docker Hub.
        assert_eq!(resolve_ollama_api("gemma4:e4b"), "docker.io/ai/gemma4:e4b");
        // Dots in the name and/or tag don't disqualify "bare" either — only
        // a "/" does. Regression test: this used to fall through unchanged
        // (neither hf.co- nor docker.io/ai-prefixed) because has_host()
        // mistook the dotted version number for an explicit registry host,
        // and dead-ended in the Go shim's HF-only parser with a misleading
        // "invalid HuggingFace reference" error instead.
        assert_eq!(resolve_ollama_api("qwen3.5"), "docker.io/ai/qwen3.5");
        assert_eq!(
            resolve_ollama_api("qwen3.5:0.8B"),
            "docker.io/ai/qwen3.5:0.8B"
        );
    }

    #[test]
    fn resolve_ollama_api_leaves_structured_references_to_resolve() {
        // Owner/repo (has a "/") falls back to resolve()'s hf.co default.
        assert_eq!(
            resolve_ollama_api("unsloth/Qwen3.5-0.8B-GGUF"),
            resolve("unsloth/Qwen3.5-0.8B-GGUF")
        );
        // Already has an explicit host.
        assert_eq!(resolve_ollama_api("hf.co/foo/bar"), "hf.co/foo/bar");
        assert_eq!(
            resolve_ollama_api("docker.io/ai/gemma4"),
            "docker.io/ai/gemma4"
        );
    }

    #[test]
    fn resolve_ollama_api_matches_resolve_for_uri_schemes_and_paths() {
        assert_eq!(
            resolve_ollama_api("hf://unsloth/Qwen3.5-0.8B-GGUF"),
            resolve("hf://unsloth/Qwen3.5-0.8B-GGUF")
        );
        assert_eq!(
            resolve_ollama_api("/abs/path/model.gguf"),
            "/abs/path/model.gguf"
        );
    }

    #[test]
    fn is_bare_rejects_only_slashes() {
        assert!(is_bare("gemma4"));
        assert!(is_bare("gemma4:e4b"));
        assert!(!is_bare("unsloth/gemma4"));
        // Dots alone (no "/") no longer disqualify bareness — see
        // has_host_requires_a_slash below for the corresponding fix.
        assert!(is_bare("qwen3.5"));
        assert!(is_bare("qwen3.5:0.8B"));
        assert!(!is_bare("hf.co/gemma4"));
    }

    /// Ported from ollama's types/model/name_test.go (TestParseNameParts /
    /// TestNameparseNameDefault): ollama fills an unqualified name out to
    /// registry.ollama.ai/library/<model>:latest; llmman's equivalents are
    /// resolve_ollama_api's docker.io/ai/<name> default for bare names and
    /// resolve's hf.co/<owner>/<repo> default for host-less paths, while
    /// anything already carrying a host passes through untouched.
    #[test]
    fn resolve_fills_in_default_registry_like_ollama_parse_name() {
        // Bare model name (ollama: "model" -> registry.ollama.ai/library/model:latest).
        assert_eq!(resolve_ollama_api("mistral"), "docker.io/ai/mistral");
        assert_eq!(resolve_ollama_api("mistral:7b"), "docker.io/ai/mistral:7b");
        // namespace/model (ollama: -> registry.ollama.ai/namespace/model).
        assert_eq!(resolve("namespace/model"), "hf.co/namespace/model");
        // Fully-qualified references pass through untouched...
        assert_eq!(
            resolve("example.com/ns/model:tag"),
            "example.com/ns/model:tag"
        );
        // ...including a host:port first component (ollama's
        // "host:port/namespace/model:tag" case) and localhost.
        assert_eq!(
            resolve("example.com:5000/ns/model:tag"),
            "example.com:5000/ns/model:tag"
        );
        assert_eq!(resolve("localhost/ns/model"), "localhost/ns/model");
    }

    /// Ported from ollama's types/model/name_test.go scheme cases
    /// ("scheme://host/namespace/model:tag" parses with the scheme split
    /// off): llmman likewise never treats a URI scheme as part of the
    /// reference — hf:// and huggingface:// are stripped before the normal
    /// defaulting rules run, modelscope:// is normalised to ms://, and
    /// object-store schemes pass through verbatim.
    #[test]
    fn resolve_splits_uri_schemes_like_ollama_parse_name() {
        assert_eq!(resolve("hf://owner/repo"), "hf.co/owner/repo");
        assert_eq!(resolve("huggingface://owner/repo"), "hf.co/owner/repo");
        assert_eq!(resolve("hf://hf.co/owner/repo"), "hf.co/owner/repo");
        assert_eq!(resolve("modelscope://owner/repo"), "ms://owner/repo");
        assert_eq!(resolve("ms://owner/repo"), "ms://owner/repo");
        assert_eq!(resolve("s3://bucket/key"), "s3://bucket/key");
        assert_eq!(resolve("gs://bucket/key"), "gs://bucket/key");
        assert_eq!(resolve("ngc://org/model"), "ngc://org/model");
    }

    #[test]
    fn has_host_requires_a_slash() {
        // No "/" at all: a dotted version number must not be mistaken for
        // an explicit host, no matter how host-like the dot looks.
        assert!(!has_host("qwen3.5:0.8B"));
        assert!(!has_host("qwen3.5"));
        // With a "/", the first component is genuinely checked for a host.
        assert!(has_host("hf.co/foo/bar"));
        assert!(has_host("localhost/foo"));
        assert!(!has_host("unsloth/Qwen3.5-0.8B-GGUF"));
    }

    #[test]
    // Regression: "localhost:PORT/..." (a local test registry) was
    // mistaken for a host-less reference, since neither the dot check
    // nor the exact "localhost" match recognized it — "resolve" then
    // wrongly prepended "hf.co/", producing "hf.co/localhost:PORT/...".
    fn has_host_recognizes_an_explicit_port() {
        assert!(has_host("localhost:5000/foo/bar"));
        assert!(has_host("registry.example.com:5000/foo"));
        assert_eq!(
            resolve("localhost:5000/foo/bar:tag"),
            "localhost:5000/foo/bar:tag"
        );
    }
}
