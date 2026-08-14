//! `llmman sign` — attach a cosign-compatible [Sigstore](https://www.sigstore.dev/)
//! signature to a model image, as an OCI 1.1 *referrer* of its manifest.
//!
//! There's no "directory kit" concept here, only OCI manifests — either
//! already sitting in the local store, or already pushed to a registry.
//! See `SignArgs::remote` for how those two modes are handled.
//!
//! Two signing modes:
//!
//!   - **Keyless** (the default): obtains a Fulcio short-lived certificate
//!     for an OIDC identity and records the signature in the public Rekor
//!     transparency log. The identity token is resolved in order from
//!     `--identity-token`, `--identity-token-file`, an ambient CI
//!     provider (currently just GitHub Actions), or — if none of those
//!     apply and stdin is a terminal — an interactive browser login.
//!
//!   - **Key-based** (`--key`): signs with an unencrypted ECDSA P-256
//!     private key PEM. No Fulcio, Rekor, or network access at all (aside
//!     from `--remote`'s own registry round-trip).
//!
//! There is currently no `--tlog-upload=false` / private-keyless mode:
//! the `sigstore` crate this is built on
//! (<https://github.com/sigstore/sigstore-rs>) always uploads keyless
//! signatures to the public Rekor log. Use `--key` for signing that must
//! never touch a public log.
//!
//! Signature storage:
//!
//!   - **Local store** (default): the signature bundle and a small
//!     referrer manifest (`subject` = the signed manifest's descriptor)
//!     are written into the same local OCI layout, content-addressed like
//!     everything else there. They are *not* tagged into `index.json`, so
//!     `llmman list` doesn't show them — there is no `llmman push --sign`
//!     yet to carry them along automatically, so for now they sit in the
//!     local store's blobs ready for a future command to pick up.
//!   - **`--remote`**: the signed manifest is resolved directly from the
//!     registry (no local pull needed), and the referrer is pushed there
//!     immediately, tagged with cosign's own fallback naming scheme
//!     (`sha256-<hex>.sig`) for registries that don't yet auto-index OCI
//!     1.1 referrers by `subject` alone.

use std::fs;
use std::io::{Cursor, IsTerminal};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context};
use base64::{engine::general_purpose::STANDARD as base64, Engine as _};
use clap::Args;
use ecdsa::signature::DigestSigner;
use pkcs8::{DecodePrivateKey, EncodePublicKey};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use sigstore::bundle::sign::SigningContext;
use sigstore::oauth::{self, IdentityToken};
use sigstore_protobuf_specs::dev::sigstore::bundle::v1::{
    bundle, verification_material, Bundle, VerificationMaterial,
};
use sigstore_protobuf_specs::dev::sigstore::common::v1::{
    HashAlgorithm, HashOutput, MessageSignature, PublicKeyIdentifier,
};

use crate::ffi;
use crate::storage::oci::{Descriptor, Manifest};
use crate::storage::OciStore;

/// OCI media type of the signature bundle blob (and the referrer
/// manifest's `artifactType`) — the same media type cosign uses for its
/// own OCI 1.1 signature referrers, so other Sigstore-aware tooling
/// recognizes these without needing to know about llmman specifically.
const BUNDLE_MEDIA_TYPE: &str = "application/vnd.dev.sigstore.bundle.v0.3+json";
/// OCI 1.1 "no config" placeholder — see the image-spec's Guidance for
/// Artifact Authors. Its digest is fixed (`sha256` of the literal `{}`).
const EMPTY_MEDIA_TYPE: &str = "application/vnd.oci.empty.v1+json";

#[derive(Args, Debug)]
pub struct SignArgs {
    /// Image reference to sign
    #[arg(value_name = "REFERENCE")]
    pub reference: String,

    /// Sign an image that's already in a registry, fetching its manifest
    /// and attaching the signature there directly instead of using the
    /// local store
    #[arg(long)]
    pub remote: bool,

    /// Private key for key-based signing (unencrypted PEM); omit for
    /// keyless signing (Sigstore Fulcio + Rekor)
    #[arg(long, value_name = "PEM", conflicts_with_all = ["identity_token", "identity_token_file"])]
    pub key: Option<PathBuf>,

    /// OIDC identity token for keyless signing (must carry an `aud:
    /// "sigstore"` claim). Prefer --identity-token-file: this flag's
    /// value is readable by other local users via process arguments and
    /// is recorded in shell history.
    #[arg(long, value_name = "TOKEN")]
    pub identity_token: Option<String>,

    /// File holding the OIDC identity token for keyless signing
    #[arg(long, value_name = "FILE")]
    pub identity_token_file: Option<PathBuf>,

    /// Local store directory (overrides default; ignored with --remote)
    #[arg(long, value_name = "DIR")]
    pub store: Option<PathBuf>,
}

pub fn run(args: &SignArgs) -> anyhow::Result<()> {
    let reference = crate::shortnames::resolve_ollama_api(&args.reference);
    if args.remote {
        sign_remote(args, &reference)
    } else {
        sign_local(args, &reference)
    }
}

// ---------------------------------------------------------------------------
// Local store
// ---------------------------------------------------------------------------

fn sign_local(args: &SignArgs, reference: &str) -> anyhow::Result<()> {
    let store_root = crate::default_store(args.store.as_deref())?;
    let store = OciStore::open(&store_root)?;
    let desc = store.find(reference)?;
    let manifest_bytes = store.read_blob(&desc.digest)?;

    let (bundle_bytes, identity) = sign_bytes(&manifest_bytes, args)?;

    let bundle_desc = store.write_blob(BUNDLE_MEDIA_TYPE, &bundle_bytes)?;
    let empty_desc = store.write_blob(EMPTY_MEDIA_TYPE, b"{}")?;
    let subject = Descriptor {
        media_type: desc.media_type.clone(),
        digest: desc.digest.clone(),
        size: desc.size,
        annotations: None,
    };
    let referrer_desc = store.write_manifest(&referrer_manifest(bundle_desc, empty_desc, subject))?;

    println!("Signed {reference} ({identity})");
    println!("  Signature stored locally as {}", referrer_desc.digest);
    println!(
        "  Not yet pushed — `llmman push` doesn't carry local signatures to a registry yet; \
         re-run `llmman sign --remote {reference}` once it's pushed to attach it there."
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Remote registry
// ---------------------------------------------------------------------------

fn sign_remote(args: &SignArgs, reference: &str) -> anyhow::Result<()> {
    // Exact bytes, not pretty-printed — see go-shim/backend_docker.go's
    // llmman_inspect comment: the digest computed below must match the
    // one the registry already has for this manifest.
    let raw = ffi::inspect_remote(reference)?;
    let bytes = raw.as_bytes();
    let value: serde_json::Value =
        serde_json::from_str(&raw).context("parse remote manifest JSON")?;
    let media_type = value
        .get("mediaType")
        .and_then(|m| m.as_str())
        .ok_or_else(|| {
            anyhow!("remote manifest for {reference} has no \"mediaType\" field; cannot compute an OCI referrer subject for it")
        })?
        .to_string();
    let digest = format!("sha256:{}", hex::encode(Sha256::digest(bytes)));
    let size = bytes.len() as u64;

    let (bundle_bytes, identity) = sign_bytes(bytes, args)?;

    // A throwaway OCI layout holding just the referrer manifest + its two
    // blobs, pushed under cosign's own fallback tag scheme
    // (`sha256-<hex>.sig`) so it lands in the *same repository* as
    // `reference` without ever touching that repository's real tags —
    // see this module's own doc comment for why a distinct tag matters
    // here. Compliant OCI 1.1 registries also index it under `subject`
    // regardless of which tag it was pushed with.
    let tmp_dir = std::env::temp_dir().join(format!(
        "llmman-sign-{}-{}",
        std::process::id(),
        digest.trim_start_matches("sha256:")
    ));
    let _cleanup = TempDirGuard(&tmp_dir);
    let sig_store = OciStore::open(&tmp_dir)?;
    let bundle_desc = sig_store.write_blob(BUNDLE_MEDIA_TYPE, &bundle_bytes)?;
    let empty_desc = sig_store.write_blob(EMPTY_MEDIA_TYPE, b"{}")?;
    let subject = Descriptor {
        media_type,
        digest: digest.clone(),
        size,
        annotations: None,
    };
    let referrer_desc =
        sig_store.write_manifest(&referrer_manifest(bundle_desc, empty_desc, subject))?;
    let sig_ref = format!("{}:{}", repo_of(reference), fallback_sig_tag(&digest));
    sig_store.tag(referrer_desc.clone(), &sig_ref)?;

    let tmp_dir_str = tmp_dir
        .to_str()
        .ok_or_else(|| anyhow!("temp signing directory {} is not valid UTF-8", tmp_dir.display()))?;
    ffi::push(tmp_dir_str, &sig_ref)?;

    println!("Signed {reference} ({identity})");
    println!(
        "  Signature attached: {} (referrer of {digest})",
        referrer_desc.digest
    );
    Ok(())
}

/// Removes the directory on drop, best-effort — signing temp state has no
/// value once the push above either succeeds or fails.
struct TempDirGuard<'a>(&'a Path);

impl Drop for TempDirGuard<'_> {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(self.0);
    }
}

// ---------------------------------------------------------------------------
// Referrer manifest / reference helpers
// ---------------------------------------------------------------------------

fn referrer_manifest(bundle: Descriptor, config: Descriptor, subject: Descriptor) -> Manifest {
    Manifest {
        schema_version: 2,
        media_type: "application/vnd.oci.image.manifest.v1+json".into(),
        artifact_type: Some(BUNDLE_MEDIA_TYPE.into()),
        config,
        layers: vec![bundle],
        annotations: None,
        subject: Some(subject),
    }
}

/// Strips the tag or digest suffix off `reference`, leaving the bare
/// `host/repo` — so a signature reference can be built in the same
/// repository without reusing (and clobbering) the signed image's own
/// tag.
fn repo_of(reference: &str) -> &str {
    if let Some(at) = reference.rfind('@') {
        return &reference[..at];
    }
    if let Some(colon) = reference.rfind(':') {
        if colon > reference.rfind('/').unwrap_or(0) {
            return &reference[..colon];
        }
    }
    reference
}

/// Cosign's own fallback tag scheme for a signature of `sha256:<hex>`:
/// `sha256-<hex>.sig`. Kept for compatibility with tooling that predates
/// (or ignores) the OCI 1.1 Referrers API.
fn fallback_sig_tag(digest: &str) -> String {
    format!("{}.sig", digest.replacen(':', "-", 1))
}

// ---------------------------------------------------------------------------
// Signing
// ---------------------------------------------------------------------------

/// Signs `bytes`, returning the serialized Sigstore bundle JSON and a
/// short human-readable description of who/what signed it.
fn sign_bytes(bytes: &[u8], args: &SignArgs) -> anyhow::Result<(Vec<u8>, String)> {
    match &args.key {
        Some(key_path) => key_based_sign(bytes, key_path),
        None => keyless_sign(bytes, args),
    }
}

/// Key-based signing: ECDSA P-256 over the SHA-256 digest of `bytes`,
/// packaged the same way `cosign sign --key` does — a `MessageSignature`
/// plus a `PublicKeyIdentifier` hint (base64 SHA-256 of the DER
/// SubjectPublicKeyInfo, RFC 6962 §3.2) rather than a certificate, since
/// there's no CA involved.
fn key_based_sign(bytes: &[u8], key_path: &Path) -> anyhow::Result<(Vec<u8>, String)> {
    ensure_private_mode(key_path, "private key")?;
    let pem = fs::read_to_string(key_path)
        .with_context(|| format!("read {}", key_path.display()))?;
    let signing_key = p256::ecdsa::SigningKey::from_pkcs8_pem(&pem).map_err(|e| {
        anyhow!(
            "{}: not an unencrypted PKCS#8 EC private key PEM: {e}",
            key_path.display()
        )
    })?;

    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest_bytes = hasher.clone().finalize().to_vec();
    let signature: p256::ecdsa::Signature = signing_key.sign_digest(hasher);
    let signature_der = signature.to_der().as_bytes().to_vec();

    let public_key_der = signing_key
        .verifying_key()
        .to_public_key_der()
        .context("encode public key")?;
    let hint = base64.encode(Sha256::digest(public_key_der.as_bytes()));

    let bundle = Bundle {
        media_type: BUNDLE_MEDIA_TYPE.to_string(),
        verification_material: Some(VerificationMaterial {
            tlog_entries: vec![],
            timestamp_verification_data: None,
            content: Some(verification_material::Content::PublicKey(
                PublicKeyIdentifier { hint },
            )),
        }),
        content: Some(bundle::Content::MessageSignature(MessageSignature {
            message_digest: Some(HashOutput {
                algorithm: HashAlgorithm::Sha2256.into(),
                digest: digest_bytes,
            }),
            signature: signature_der,
        })),
    };
    let json = serde_json::to_vec(&bundle).context("serialize signature bundle")?;
    Ok((json, format!("key-based, {}", key_path.display())))
}

/// Keyless signing: a Fulcio-issued short-lived certificate for an OIDC
/// identity, with the signature recorded in the public Rekor
/// transparency log — handled end to end by the `sigstore` crate.
fn keyless_sign(bytes: &[u8], args: &SignArgs) -> anyhow::Result<(Vec<u8>, String)> {
    let token = resolve_identity_token(args)?;
    let identity = format!("keyless, {}", token.unverified_claims().email);

    let context = SigningContext::production().context("initialize Sigstore trust root")?;
    let session = context
        .blocking_signer(token)
        .context("obtain a Fulcio signing certificate")?;
    let artifact = session
        .sign(Cursor::new(bytes.to_vec()))
        .context("sign with Sigstore (Fulcio/Rekor)")?;
    let json =
        serde_json::to_vec(&artifact.to_bundle()).context("serialize signature bundle")?;
    Ok((json, identity))
}

/// Resolves an OIDC identity token: an explicit flag first, then an
/// ambient CI provider, then (only if stdin is a terminal) an
/// interactive browser login.
fn resolve_identity_token(args: &SignArgs) -> anyhow::Result<IdentityToken> {
    if let Some(token) = &args.identity_token {
        return IdentityToken::try_from(token.as_str())
            .map_err(|e| anyhow!("--identity-token: {e}"));
    }
    if let Some(path) = &args.identity_token_file {
        ensure_private_mode(path, "identity token file")?;
        let raw =
            fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        return IdentityToken::try_from(raw.trim())
            .map_err(|e| anyhow!("--identity-token-file: {e}"));
    }
    if let Some(raw) = ambient_github_actions_token() {
        return IdentityToken::try_from(raw.as_str())
            .map_err(|e| anyhow!("ambient GitHub Actions OIDC token: {e}"));
    }
    if !std::io::stdin().is_terminal() {
        anyhow::bail!(
            "no OIDC identity token available for keyless signing: pass --identity-token-file, \
             run inside a supported ambient CI provider (currently: GitHub Actions), \
             or use --key for offline key-based signing instead"
        );
    }
    interactive_login()
}

/// GitHub Actions mints an OIDC token scoped to whatever `audience` is
/// requested, from a short-lived per-job token/URL pair the runner
/// exposes as `ACTIONS_ID_TOKEN_REQUEST_TOKEN`/`_URL` (only present when
/// the job has `permissions: id-token: write`). Returns `None` — not an
/// error — whenever these aren't set or the request fails, so callers can
/// fall through to the next resolution method.
fn ambient_github_actions_token() -> Option<String> {
    let url = std::env::var("ACTIONS_ID_TOKEN_REQUEST_URL").ok()?;
    let bearer = std::env::var("ACTIONS_ID_TOKEN_REQUEST_TOKEN").ok()?;
    let sep = if url.contains('?') { "&" } else { "?" };
    let full_url = format!("{url}{sep}audience=sigstore");

    #[derive(Deserialize)]
    struct Response {
        value: String,
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .ok()?;
    let resp = match client.get(&full_url).bearer_auth(bearer).send() {
        Ok(resp) => resp,
        Err(e) => {
            eprintln!("warning: ambient GitHub Actions OIDC token request failed: {e}");
            return None;
        }
    };
    if !resp.status().is_success() {
        eprintln!(
            "warning: ambient GitHub Actions OIDC token request returned {}",
            resp.status()
        );
        return None;
    }
    match resp.json::<Response>() {
        Ok(r) => Some(r.value),
        Err(e) => {
            eprintln!("warning: could not parse ambient GitHub Actions OIDC token response: {e}");
            None
        }
    }
}

/// Interactive fallback: opens a browser against Sigstore's public-good
/// OIDC issuer and waits for the redirect, mirroring `sigstore-rs`'s own
/// `examples/bundle` CLI.
fn interactive_login() -> anyhow::Result<IdentityToken> {
    eprintln!(
        "No OIDC identity token available — opening a browser to sign in with Sigstore's \
         public-good identity provider..."
    );
    let (url, client, nonce, verifier) = oauth::openidflow::OpenIDAuthorize::new(
        "sigstore",
        "",
        "https://oauth2.sigstore.dev/auth",
        "http://localhost:8080",
    )
    .auth_url()
    .context("start Sigstore OIDC authorization")?;

    if webbrowser::open(url.as_ref()).is_err() {
        eprintln!("Could not open a browser automatically — visit this URL to continue:\n  {url}");
    }

    let listener =
        oauth::openidflow::RedirectListener::new("127.0.0.1:8080", client, nonce, verifier);
    let (_, token) = listener.redirect_listener().context(
        "complete Sigstore OIDC login (the browser flow needs to listen on 127.0.0.1:8080; \
         if that port is in use, pass --identity-token-file or --key instead)",
    )?;
    Ok(IdentityToken::from(token))
}

// ---------------------------------------------------------------------------
// File permission checks
// ---------------------------------------------------------------------------

/// Refuses a private key / identity token file that group or others can
/// read — the file's mode is the only thing keeping it secret, since
/// neither is ever accepted via an environment variable. No-op on
/// Windows, where Unix permission bits are synthesized from the
/// read-only attribute and access is governed by ACLs instead.
#[cfg(unix)]
fn ensure_private_mode(path: &Path, what: &str) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mode = fs::metadata(path)
        .with_context(|| format!("stat {}", path.display()))?
        .permissions()
        .mode()
        & 0o777;
    if mode & 0o077 != 0 {
        anyhow::bail!(
            "{what} {:?} is accessible by group or others (mode {:o}); restrict it with: chmod 600 {:?}",
            path,
            mode,
            path
        );
    }
    Ok(())
}

#[cfg(not(unix))]
fn ensure_private_mode(_path: &Path, _what: &str) -> anyhow::Result<()> {
    Ok(())
}
