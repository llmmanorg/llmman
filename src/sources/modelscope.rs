//! ModelScope Hub (`ms://owner/repo[:revision]`).
//!
//! ModelScope's own repo-files API, not the HuggingFace-compatible
//! surface it also exposes — the same model reached as a bare
//! `modelscope.cn/owner/repo` host goes through [`crate::hf`] instead.

use anyhow::{Context, Result};
use serde::Deserialize;

use super::Target;

/// One entry from the ModelScope tree API. The field names are
/// capitalized in the wire format, unlike almost every other JSON API
/// llmman talks to.
#[derive(Deserialize)]
struct MsFile {
    #[serde(rename = "Path")]
    path: String,
    #[serde(rename = "Size")]
    #[serde(default)]
    size: i64,
    /// `"blob"` for a file, `"tree"` for a directory. Note the value:
    /// the Go version matched on `"file"`, which this API never returns,
    /// so every entry was filtered out and every `ms://` pull failed
    /// with "no model files found". Matching on "not a directory" also
    /// survives a third spelling appearing.
    #[serde(rename = "Type")]
    #[serde(default)]
    kind: String,
}

impl MsFile {
    fn is_dir(&self) -> bool {
        self.kind == "tree"
    }
}

#[derive(Deserialize)]
struct MsListing {
    #[serde(rename = "Data")]
    data: MsData,
}

#[derive(Deserialize)]
struct MsData {
    #[serde(rename = "Files")]
    #[serde(default)]
    files: Vec<MsFile>,
}

/// `reference` is the full `ms://`/`modelscope://` reference; `ms_ref`
/// is it with the scheme stripped, i.e. `owner/repo[:revision]`.
/// Revision defaults to ModelScope's own default branch name, `master`
/// (not `main`).
pub(crate) async fn pull(reference: &str, ms_ref: &str, target: &Target<'_>) -> Result<()> {
    let (owner, repo, revision) = parse_ref(ms_ref)?;

    let endpoint = std::env::var("MODELSCOPE_ENDPOINT")
        .ok()
        .filter(|e| !e.is_empty())
        .unwrap_or_else(|| "https://modelscope.cn".to_string());
    let endpoint = endpoint.trim_end_matches('/');
    let token = token_for(endpoint, std::env::var("MODELSCOPE_API_TOKEN").ok());

    // Stored under the reference exactly as given, like every other
    // source. The Go version stored a rebuilt "ms://owner/repo:<rev>",
    // so a successful pull left an entry no later `llmman run
    // ms://owner/repo` could look up — it resolves the tag to "latest",
    // never "master".
    if target.report_cached(reference, &format!("{owner}/{repo}")) {
        return Ok(());
    }

    let api = crate::hf::api_client()?;
    let list_url = format!(
        "{endpoint}/api/v1/models/{owner}/{repo}/repo/files?Revision={revision}&Recursive=true"
    );
    let mut req = api.get(&list_url);
    if let Some(t) = &token {
        req = req.header(reqwest::header::AUTHORIZATION, format!("Token {t}"));
    }
    let resp = req.send().await.context("ModelScope list")?;
    let status = resp.status();
    if !status.is_success() {
        anyhow::bail!("ModelScope list: HTTP {}", status.as_u16());
    }
    let listing: MsListing = resp.json().await.context("ModelScope list decode")?;

    let dl = crate::hf::download_client()?;
    let mut packed = Vec::new();
    for f in &listing.data.files {
        if f.is_dir() || !super::should_pack(&f.path) {
            continue;
        }
        let url = format!("{endpoint}/{owner}/{repo}/resolve/{revision}/{}", f.path);
        let mut req = dl.get(&url);
        if let Some(t) = &token {
            req = req.header(reqwest::header::AUTHORIZATION, format!("Token {t}"));
        }
        let resp = req
            .send()
            .await
            .with_context(|| format!("ModelScope download {}", f.path))?
            .error_for_status()
            .with_context(|| format!("ModelScope download {}", f.path))?;
        packed.push(
            super::download_to_pack_file(
                target,
                "ms",
                "ModelScope",
                &f.path,
                f.size,
                resp.bytes_stream(),
            )
            .await?,
        );
    }

    super::pack_as_model_pack(
        target,
        reference,
        &format!("{owner}/{repo}"),
        packed,
        format!("no model files found in ModelScope repo {owner}/{repo}"),
    )
}

/// `token`, but only for an HTTPS endpoint: a `MODELSCOPE_ENDPOINT`
/// override pointed at plain HTTP would otherwise put the bearer token
/// on the wire in cleartext. Anonymous access to such an endpoint still
/// works, so this warns rather than failing.
fn token_for(endpoint: &str, token: Option<String>) -> Option<String> {
    let token = token.filter(|t| !t.is_empty())?;
    if !endpoint.starts_with("https://") {
        eprintln!(
            "[llmman] warning: MODELSCOPE_ENDPOINT is not HTTPS; \
             continuing anonymously rather than sending MODELSCOPE_API_TOKEN in cleartext"
        );
        return None;
    }
    Some(token)
}

/// Splits `owner/repo[:revision]`. A `:` only starts a revision when it
/// comes after the last `/`, so a revision that itself contains a slash
/// can't be mistaken for one.
fn parse_ref(ms_ref: &str) -> Result<(&str, &str, &str)> {
    let (owner, repo_rev) = ms_ref.split_once('/').with_context(|| {
        format!("invalid ModelScope ref {ms_ref:?}: expected owner/repo[:revision]")
    })?;
    if owner.is_empty() || repo_rev.is_empty() {
        anyhow::bail!("invalid ModelScope ref {ms_ref:?}: expected owner/repo[:revision]");
    }
    Ok(match repo_rev.rsplit_once(':') {
        Some((repo, rev)) if !repo.is_empty() && !rev.is_empty() => (owner, repo, rev),
        _ => (owner, repo_rev, "master"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the bug this port inherited and fixed:
    /// ModelScope labels files `"blob"`, not `"file"`, so filtering on
    /// `"file"` dropped every entry in the repository.
    #[test]
    fn listing_entries_are_files_unless_they_are_trees() {
        let listing: MsListing = serde_json::from_str(
            r#"{"Code":200,"Data":{"Files":[
                 {"Path":"config.json","Size":659,"Type":"blob"},
                 {"Path":"coreml","Size":0,"Type":"tree"}
               ]}}"#,
        )
        .unwrap();
        let files = &listing.data.files;
        assert!(!files[0].is_dir(), "a \"blob\" entry is a file to fetch");
        assert!(files[1].is_dir(), "a \"tree\" entry is a directory to skip");
        assert_eq!(files[0].size, 659);
    }

    #[test]
    fn parse_ref_defaults_the_revision_to_modelscopes_own_master() {
        assert_eq!(
            parse_ref("owner/repo").unwrap(),
            ("owner", "repo", "master")
        );
    }

    #[test]
    fn parse_ref_reads_an_explicit_revision() {
        assert_eq!(
            parse_ref("owner/repo:v1.2").unwrap(),
            ("owner", "repo", "v1.2")
        );
    }

    /// A plain-HTTP endpoint override must not receive the bearer
    /// token — the pull falls back to anonymous instead.
    #[test]
    fn token_for_only_trusts_an_https_endpoint() {
        let secret = || Some("secret".to_string());
        assert_eq!(
            token_for("https://modelscope.cn", secret()).as_deref(),
            Some("secret")
        );
        assert_eq!(token_for("http://127.0.0.1:8731", secret()), None);
        assert_eq!(token_for("https://modelscope.cn", None), None);
    }

    #[test]
    fn parse_ref_rejects_a_reference_with_no_owner_or_repo() {
        assert!(parse_ref("repo").is_err());
        assert!(parse_ref("owner/").is_err());
    }
}
