//! `llmman search` — find models on Docker Hub and Hugging Face.
//!
//! Both registries are queried at once; Docker Hub's rows come first,
//! since `docker.io/ai/` is where a bare `llmman run <name>` already
//! looks. Every NAME printed is fully qualified so it pastes straight
//! into `run`/`pull` (a bare `ai/qwen3` would resolve to hf.co).
//!
//! Docker Hub is queried through its website's search API
//! (`hub.docker.com/api/search/v4`, no credentials). Hub reports the
//! manifest media types pushed to each repo; only repos carrying a CNCF
//! ModelPack manifest are listed, since that is the format `pull` reads.
//! Repos published solely in Docker Model Runner's own format
//! (`application/vnd.docker.ai.model.config.v0.1+json`) are left out.
//! Hugging Face goes through `hf::api::search_models`, with the user's
//! token when configured.
//!
//! The CLI adds a FIT column; it costs a request per row, so
//! `/llmman/search` does not.

use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use futures::{stream, StreamExt};
use serde::{Deserialize, Serialize};

use crate::fmt::{
    columns, human_count, paint, relative_time_rfc3339, stdout_color, GREEN, RED, YELLOW,
};
use crate::{hf, hostgpu};

/// Docker Hub's website search API; not part of the OCI registry
/// protocol, so it doesn't go through the Go shim.
const DOCKER_HUB_SEARCH_URL: &str = "https://hub.docker.com/api/search/v4";

/// Page size requested from Docker Hub, and the cap on `--limit`.
pub const MAX_LIMIT: u32 = 64;

/// Rows per registry when `--limit` is not given.
pub const DEFAULT_LIMIT: u32 = 25;

/// The manifest media type of a CNCF ModelPack artifact.
const MODELPACK_MANIFEST: &str = "application/vnd.cncf.model.manifest.v1+json";

const FIT_HELP: &str = "\
FIT is the percent of this machine's model memory (GPU, else RAM) that a \
row's default download takes, weights only. Green is up to 60%, yellow up \
to 90%, red above; `-` is unknown. Colors show only on a terminal, and not \
with NO_COLOR set.";

/// Size lookups in flight at once for FIT.
const SIZE_LOOKUPS: usize = 16;

/// All of FIT's lookups and the memory probe get this long; rows still
/// unanswered show `-`.
const FIT_BUDGET: Duration = Duration::from_secs(10);

#[derive(Args, Debug)]
#[command(after_help = FIT_HELP)]
pub struct SearchArgs {
    /// Text to match against model names and descriptions
    #[arg(value_name = "QUERY")]
    pub query: String,

    /// Maximum number of results to show from each registry
    #[arg(
        long,
        short = 'n',
        default_value_t = DEFAULT_LIMIT,
        value_parser = clap::value_parser!(u32).range(1..=MAX_LIMIT as i64)
    )]
    pub limit: u32,

    /// Only search one registry instead of both
    #[arg(long, value_enum)]
    pub registry: Option<Registry>,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Registry {
    /// Docker Hub (docker.io)
    #[value(alias = "docker.io", alias = "hub")]
    Docker,
    /// Hugging Face (hf.co)
    #[value(alias = "huggingface", alias = "hf.co")]
    Hf,
}

/// One table row, from either registry; also a `/llmman/search` result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Hit {
    /// Fully qualified reference, ready for `llmman run`/`pull`.
    pub name: String,
    /// Docker Hub pulls; Hugging Face (last-30-day) downloads.
    pub pulls: Option<u64>,
    /// Docker Hub stars; Hugging Face likes.
    pub likes: Option<u64>,
    /// RFC 3339 timestamp of the last update, if reported.
    pub updated: Option<String>,
}

pub fn run(args: &SearchArgs) -> Result<()> {
    let query = args.query.trim();
    anyhow::ensure!(!query.is_empty(), "search query must not be empty");

    let runtime = tokio::runtime::Runtime::new().context("start tokio runtime")?;
    let hits = runtime.block_on(search(query, args.limit, args.registry))?;

    if hits.is_empty() {
        anyhow::bail!("no models found for {query:?}");
    }
    let fits = runtime.block_on(fits(&hits));
    print!("{}", render(&hits, &fits, stdout_color()));
    // Not `drop`, which would wait out a hung memory probe.
    runtime.shutdown_background();
    Ok(())
}

/// Docker Hub's rows first. With both registries selected, one failing
/// is a warning and the other's rows still print; both failing is an
/// error. Also `llmman serve`'s `/llmman/search`.
pub async fn search(query: &str, limit: u32, registry: Option<Registry>) -> Result<Vec<Hit>> {
    let client = hf::api_client()?;
    match registry {
        Some(Registry::Docker) => search_docker_hub(&client, query, limit).await,
        Some(Registry::Hf) => search_hugging_face(&client, query, limit).await,
        None => {
            let (docker, hugging_face) = tokio::join!(
                search_docker_hub(&client, query, limit),
                search_hugging_face(&client, query, limit),
            );
            both(docker, hugging_face)
        }
    }
}

/// What `llmman serve`'s Models page lists before any search: Docker
/// Hub's most pulled models, which Docker's own `ai/` ones lead, then
/// Hugging Face's most downloaded GGUF text-generation repos, the kind
/// llama.cpp serves.
pub async fn popular(limit: u32) -> Result<Vec<Hit>> {
    let client = hf::api_client()?;
    let (docker, hugging_face) = tokio::join!(
        search_docker_hub(&client, "", limit),
        popular_hugging_face(&client, limit),
    );
    both(docker, hugging_face)
}

/// Docker Hub's rows, then Hugging Face's; one registry failing is a
/// warning, both an error.
fn both(docker: Result<Vec<Hit>>, hugging_face: Result<Vec<Hit>>) -> Result<Vec<Hit>> {
    match (docker, hugging_face) {
        (Ok(mut hits), Ok(more)) => {
            hits.extend(more);
            Ok(hits)
        }
        (Ok(hits), Err(e)) => {
            eprintln!("[llmman] Hugging Face search failed, showing Docker Hub only: {e:#}");
            Ok(hits)
        }
        (Err(e), Ok(hits)) => {
            eprintln!("[llmman] Docker Hub search failed, showing Hugging Face only: {e:#}");
            Ok(hits)
        }
        (Err(docker), Err(hugging_face)) => Err(docker
            .context(format!("Hugging Face: {hugging_face:#}"))
            .context("search failed on both registries")),
    }
}

// ---------------------------------------------------------------------------
// Docker Hub
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct HubSearchResponse {
    #[serde(default)]
    results: Vec<HubResult>,
}

#[derive(Debug, Deserialize)]
struct HubResult {
    /// `namespace/repo`, or a bare `repo` for the official `library/`
    /// namespace.
    name: String,
    #[serde(default)]
    star_count: u64,
    /// The exact figure; `pull_count` is a display string like `"500K+"`.
    #[serde(default)]
    raw_pull_count: Option<u64>,
    #[serde(default)]
    updated_at: Option<String>,
    #[serde(default)]
    archived: bool,
    /// Every manifest/config media type pushed to the repo.
    #[serde(default)]
    media_types: Vec<String>,
}

impl HubResult {
    /// A repo can hold several formats (Docker's `ai/` models are
    /// published in both), so this is "has one", not "has only this".
    fn is_modelpack(&self) -> bool {
        self.media_types.iter().any(|t| t == MODELPACK_MANIFEST)
    }
}

/// Two Hub queries, merged: `type=model` only matches repos carrying
/// Model Runner's media type, so it misses ModelPack-only repos like
/// `ai/qwen3.8`; the unfiltered query finds those. Both are sorted by
/// pulls, since Hub's relevance order puts every `<user>/qwen` image
/// ahead of `ai/qwen3.8`. Each fetches a full page since
/// [`docker_hub_hits`] drops most rows before the cut to `limit`. One
/// failing page is tolerated.
async fn search_docker_hub(client: &reqwest::Client, query: &str, limit: u32) -> Result<Vec<Hit>> {
    let page = |kind| async move {
        let url = docker_hub_search_url(query, kind)?;
        hf::api::get_json::<HubSearchResponse>(client, url.as_str(), None).await
    };
    let (typed, untyped) = tokio::join!(page(Some("model")), page(None));
    let pages = match (typed, untyped) {
        (Ok(a), Ok(b)) => vec![a, b],
        (Ok(a), Err(_)) | (Err(_), Ok(a)) => vec![a],
        (Err(e), Err(_)) => return Err(e.context("Docker Hub model search")),
    };
    Ok(docker_hub_hits(pages, limit))
}

/// One page, most pulled first, optionally limited to a Hub `type`.
fn docker_hub_search_url(query: &str, kind: Option<&'static str>) -> Result<reqwest::Url> {
    let size = MAX_LIMIT.to_string();
    let mut params = vec![
        ("query", query),
        ("from", "0"),
        ("size", size.as_str()),
        ("sort", "pull_count"),
        ("order", "desc"),
    ];
    params.extend(kind.map(|k| ("type", k)));
    reqwest::Url::parse_with_params(DOCKER_HUB_SEARCH_URL, params)
        .context("build Docker Hub search URL")
}

/// Filters and merges Hub pages, dropping repeats, most pulled first;
/// stable, so ties keep Hub's order.
fn docker_hub_hits(pages: Vec<HubSearchResponse>, limit: u32) -> Vec<Hit> {
    let mut seen = std::collections::HashSet::new();
    let mut rows: Vec<HubResult> = pages
        .into_iter()
        .flat_map(|r| r.results)
        .filter(|r| !r.archived && r.is_modelpack())
        .filter(|r| seen.insert(r.name.clone()))
        .collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.raw_pull_count.unwrap_or(0)));
    rows.into_iter()
        .take(limit as usize)
        .map(|r| {
            // Hub omits the official namespace; a pullable reference needs it.
            let name = if r.name.contains('/') {
                format!("docker.io/{}", r.name)
            } else {
                format!("docker.io/library/{}", r.name)
            };
            Hit {
                name,
                pulls: r.raw_pull_count,
                likes: Some(r.star_count),
                updated: r.updated_at,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Hugging Face
// ---------------------------------------------------------------------------

async fn search_hugging_face(
    client: &reqwest::Client,
    query: &str,
    limit: u32,
) -> Result<Vec<Hit>> {
    let endpoint = hf::hf_endpoint("hf.co");
    let token = hf::token();
    let models = hf::api::search_models(client, &endpoint, query, limit, token.as_deref()).await?;
    Ok(models.into_iter().map(hugging_face_hit).collect())
}

async fn popular_hugging_face(client: &reqwest::Client, limit: u32) -> Result<Vec<Hit>> {
    let endpoint = hf::hf_endpoint("hf.co");
    let token = hf::token();
    let limit = limit.to_string();
    let url = reqwest::Url::parse_with_params(
        &format!("{endpoint}api/models"),
        [
            ("filter", "gguf"),
            ("pipeline_tag", "text-generation"),
            ("limit", limit.as_str()),
            ("sort", "downloads"),
            ("direction", "-1"),
            ("expand[]", "lastModified"),
            ("expand[]", "downloads"),
            ("expand[]", "likes"),
        ],
    )
    .context("build HF popular-models URL")?;
    let models: Vec<hf::api::SearchHit> = hf::api::get_json(client, url.as_str(), token.as_deref())
        .await
        .context("HF popular models")?;
    Ok(models.into_iter().map(hugging_face_hit).collect())
}

fn hugging_face_hit(model: hf::api::SearchHit) -> Hit {
    Hit {
        name: format!("hf.co/{}", model.id),
        pulls: Some(model.downloads),
        likes: Some(model.likes),
        updated: model.last_modified,
    }
}

// ---------------------------------------------------------------------------
// One result in detail: `llmman serve`'s `/llmman/search/model` and
// `/llmman/search/avatar`, for the web UI's Models page
// ---------------------------------------------------------------------------

/// What a search row expands to: every tag `pull` can take for it, with
/// its size, plus the repo's own facts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelCard {
    pub name: String,
    /// The repo's page on its registry's website.
    pub page: String,
    pub pulls: Option<u64>,
    pub likes: Option<u64>,
    pub updated: Option<String>,
    pub license: Option<String>,
    /// Hugging Face's task tag (`text-generation`, ...).
    pub task: Option<String>,
    pub gated: bool,
    /// Hugging Face's tags, minus the `license:`/`region:` bookkeeping.
    pub tags: Vec<String>,
    pub variants: Vec<Variant>,
    /// The repo's README as its registry serves it (Markdown, often with
    /// HTML in it), cut at [`README_MAX`].
    pub readme: Option<String>,
}

/// Bytes of README kept: model cards run long, and the page shows the top.
const README_MAX: usize = 64 * 1024;

/// `text` cut at [`README_MAX`] bytes, on a character boundary.
fn cap_readme(mut text: String) -> String {
    if text.len() > README_MAX {
        let mut end = README_MAX;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
        text.push_str("\n\n…");
    }
    text
}

/// One pullable tag of a repo.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Variant {
    /// The full reference to pull, tag included.
    pub name: String,
    pub tag: String,
    /// Bytes `pull` downloads for it, when the registry says.
    pub size: Option<u64>,
    /// What `pull` takes when no tag is given.
    pub default: bool,
}

pub async fn model_card(name: &str) -> Result<ModelCard> {
    let client = hf::api_client()?;
    if let Some(repo) = name.strip_prefix("hf.co/") {
        hugging_face_card(&client, name, repo).await
    } else if let Some(repo) = name.strip_prefix("docker.io/") {
        docker_hub_card(&client, name, repo).await
    } else {
        anyhow::bail!("{name:?} is not a Docker Hub or Hugging Face search result")
    }
}

/// The image URL of the repo owner's avatar: `None` when the registry
/// says it has none, an error when the registry could not say.
pub async fn avatar(name: &str) -> Result<Option<String>> {
    let client = hf::api_client()?;
    let (owner, hub) = match (name.strip_prefix("hf.co/"), name.strip_prefix("docker.io/")) {
        (Some(repo), _) => (repo.split('/').next().unwrap_or(repo), false),
        (_, Some(repo)) => (repo.split('/').next().unwrap_or(repo), true),
        _ => return Ok(None),
    };
    // Each registry has separate org and user namespaces; an owner is in one.
    let urls: [String; 2] = if hub {
        ["orgs", "users"].map(|kind| format!("https://hub.docker.com/v2/{kind}/{owner}"))
    } else {
        let endpoint = hf::hf_endpoint("hf.co");
        ["organizations", "users"].map(|kind| format!("{endpoint}api/{kind}/{owner}/avatar"))
    };
    #[derive(Deserialize)]
    struct Account {
        #[serde(default, alias = "avatarUrl")]
        gravatar_url: Option<String>,
    }
    for url in urls {
        match hf::api::get_json::<Account>(&client, &url, None).await {
            Ok(a) => return Ok(a.gravatar_url.filter(|u| !u.is_empty())),
            Err(e) if is_not_found(&e) => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(None)
}

fn is_not_found(e: &anyhow::Error) -> bool {
    e.chain()
        .filter_map(|c| c.downcast_ref::<hf::client::HttpStatusError>())
        .any(|e| e.status == 404)
}

async fn hugging_face_card(client: &reqwest::Client, name: &str, repo: &str) -> Result<ModelCard> {
    let (owner, repo_name) = repo
        .split_once('/')
        .with_context(|| format!("{name:?}: expected hf.co/<owner>/<repo>"))?;
    let endpoint = hf::hf_endpoint("hf.co");
    let token = hf::token();
    let info =
        hf::api::fetch_model_info(client, &endpoint, owner, repo_name, token.as_deref()).await?;
    let readme_url = format!(
        "{endpoint}{owner}/{repo_name}/raw/{}/README.md",
        info.commit()
    );
    let (files, readme) = tokio::join!(
        hf::api::fetch_files(
            client,
            &endpoint,
            owner,
            repo_name,
            info.commit(),
            token.as_deref()
        ),
        // One byte past the cap tells `cap_readme` the file went on.
        hf::api::get_prefix(client, &readme_url, token.as_deref(), README_MAX + 1),
    );
    let files = files?;
    // A repo without a README (404), or one that failed to load, still has a card.
    let readme = readme
        .ok()
        .map(|b| cap_readme(String::from_utf8_lossy(&b).into_owned()));
    Ok(ModelCard {
        name: name.to_owned(),
        page: format!("https://huggingface.co/{repo}"),
        pulls: info.downloads,
        likes: info.likes,
        updated: info.last_modified.clone(),
        license: info.license(),
        task: info.pipeline_tag.clone(),
        gated: info.gated(),
        tags: info
            .tags
            .iter()
            .filter(|t| !t.starts_with("license:") && !t.starts_with("region:"))
            .cloned()
            .collect(),
        variants: hugging_face_variants(name, &files),
        readme,
    })
}

/// A GGUF repo's quantizations, each a tag `hf::api::select_gguf` resolves
/// back to exactly the files it is listed with, plus the projector `pull`
/// adds, so what is shown is what `pull` fetches. A diffusion repo (whose
/// pull picks and adds files its own way) or a safetensors one is its one
/// default download, under the `:latest` a tagless pull is stored as. A
/// repo `pull` would refuse has none.
fn hugging_face_variants(name: &str, files: &[hf::api::HfFile]) -> Vec<Variant> {
    let size =
        |picked: &[hf::api::HfFile]| picked.iter().map(|f| f.size.max(0) as u64).sum::<u64>();
    let only = |size: Option<u64>| {
        vec![Variant {
            name: format!("{name}:latest"),
            tag: "latest".into(),
            size,
            default: true,
        }]
    };
    if hf::api::is_diffusion_repo(files) {
        // Pull refuses a split diffusion transformer, or none at all.
        return match hf::api::select_diffusion_gguf(files, "") {
            Ok(picked) if picked.len() == 1 => only(None),
            _ => Vec::new(),
        };
    }
    let Ok(default) = hf::api::select_gguf(files, "") else {
        let weights = hf::api::select_downloadable_hf_files(files);
        if weights.is_empty() {
            return Vec::new();
        }
        return only(Some(size(&weights)).filter(|&s| s > 0));
    };
    let projector = hf::api::select_mmproj(files).map_or(0, |f| f.size.max(0) as u64);
    let mut seen = std::collections::HashSet::new();
    let mut variants: Vec<Variant> = files
        .iter()
        .filter_map(|f| gguf_quant_label(&f.path))
        .filter(|label| seen.insert(label.to_uppercase()))
        .filter_map(|label| {
            let picked = hf::api::select_gguf(files, &label).ok()?;
            // Only a label that selects its own file: a substring of a
            // longer one (`Q4_K` of `Q4_K_M`) would pull something else.
            gguf_quant_label(&picked.first()?.path).filter(|l| l.eq_ignore_ascii_case(&label))?;
            Some(Variant {
                name: format!("{name}:{label}"),
                default: picked.first().map(|f| &f.path) == default.first().map(|f| &f.path),
                size: Some(size(&picked) + projector),
                tag: label,
            })
        })
        .collect();
    variants.sort_by_key(|v| v.size);
    variants
}

/// The Hub API's URL for one repo; `/tags` and `/tags/<tag>` hang off it.
fn docker_hub_repo_url(namespace: &str, repo_name: &str) -> String {
    format!("https://hub.docker.com/v2/namespaces/{namespace}/repositories/{repo_name}")
}

async fn docker_hub_tags(client: &reqwest::Client, base: &str) -> Result<Vec<HubTag>> {
    let mut url = Some(format!("{base}/tags?page_size=100"));
    let mut tags = Vec::new();
    for _ in 0..HUB_TAG_PAGES {
        let Some(next) = url.take() else { break };
        let page: HubTags = hf::api::get_json(client, &next, None).await?;
        tags.extend(page.results);
        url = page.next;
    }
    Ok(tags)
}

/// `Q4_K_M` from `Model-Q4_K_M.gguf` or `Model-Q4_K_M-00001-of-00002.gguf`;
/// `None` for a projector or importance matrix.
fn gguf_quant_label(path: &str) -> Option<String> {
    let base = path.rsplit('/').next().unwrap_or(path);
    let lower = base.to_lowercase();
    if !lower.ends_with(".gguf") || lower.contains("mmproj") || lower.contains("imatrix") {
        return None;
    }
    let stem = &base[..base.len() - ".gguf".len()];
    let stem = match stem.rsplit_once("-of-") {
        Some((head, total)) if total.chars().all(|c| c.is_ascii_digit()) => {
            head.rsplit_once('-').map(|(h, _)| h).unwrap_or(head)
        }
        _ => stem,
    };
    let label = stem.rsplit(['-', '.']).next()?;
    (!label.is_empty()).then(|| label.to_owned())
}

#[derive(Debug, Deserialize)]
struct HubTags {
    #[serde(default)]
    results: Vec<HubTag>,
    next: Option<String>,
}

/// A bound on the pages of tags read, only against a runaway repo: `ai/`
/// ones run to a couple of hundred tags, well inside it.
const HUB_TAG_PAGES: usize = 20;

#[derive(Debug, Deserialize)]
struct HubTag {
    name: String,
    #[serde(default)]
    full_size: Option<u64>,
    #[serde(default)]
    media_type: String,
}

#[derive(Debug, Deserialize)]
struct HubRepo {
    /// The repo's overview page, in Markdown.
    #[serde(default)]
    full_description: Option<String>,
    #[serde(default)]
    pull_count: Option<u64>,
    #[serde(default)]
    star_count: Option<u64>,
    #[serde(default)]
    last_updated: Option<String>,
}

async fn docker_hub_card(client: &reqwest::Client, name: &str, repo: &str) -> Result<ModelCard> {
    let (namespace, repo_name) = repo
        .split_once('/')
        .with_context(|| format!("{name:?}: expected docker.io/<namespace>/<repo>"))?;
    let base = docker_hub_repo_url(namespace, repo_name);
    let (info, tags) = tokio::join!(
        hf::api::get_json::<HubRepo>(client, &base, None),
        docker_hub_tags(client, &base),
    );
    let info = info.context("Docker Hub repository")?;
    let tags = tags.context("Docker Hub tags")?;
    let page = if namespace == "library" {
        format!("https://hub.docker.com/_/{repo_name}")
    } else {
        format!("https://hub.docker.com/r/{namespace}/{repo_name}")
    };
    Ok(ModelCard {
        name: name.to_owned(),
        page,
        pulls: info.pull_count,
        likes: info.star_count,
        updated: info.last_updated,
        license: None,
        task: None,
        gated: false,
        tags: Vec::new(),
        variants: tags
            .into_iter()
            .filter(|t| t.media_type == MODELPACK_MANIFEST)
            .map(|t| Variant {
                name: format!("{name}:{}", t.name),
                default: t.name == "latest",
                size: t.full_size,
                tag: t.name,
            })
            .collect(),
        readme: info
            .full_description
            .filter(|d| !d.trim().is_empty())
            .map(cap_readme),
    })
}

// ---------------------------------------------------------------------------
// Fit
// ---------------------------------------------------------------------------

/// Up to this percent of model memory is green, up to the next yellow,
/// above red: `fitOf` in `webui/models.js`.
const FIT_OK_MAX: u64 = 60;
const FIT_TIGHT_MAX: u64 = 90;

/// Percent of `memory` that `size` bytes take, rounded up so it never
/// reads under a tier edge it has crossed. `None` if either is unknown.
fn fit_percent(size: u64, memory: u64) -> Option<u64> {
    if size == 0 || memory == 0 {
        return None;
    }
    let percent = (size as u128 * 100).div_ceil(memory as u128);
    Some(u64::try_from(percent).unwrap_or(u64::MAX))
}

fn fit_color(percent: u64) -> &'static str {
    match percent {
        p if p <= FIT_OK_MAX => GREEN,
        p if p <= FIT_TIGHT_MAX => YELLOW,
        _ => RED,
    }
}

/// Each hit's [`fit_percent`] for its default download, in order; `None`
/// where a lookup failed or timed out. Failures are silent.
async fn fits(hits: &[Hit]) -> Vec<Option<u64>> {
    let Ok(client) = hf::api_client() else {
        return vec![None; hits.len()];
    };
    let endpoint = hf::hf_endpoint("hf.co");
    let token = hf::token();
    let (client, endpoint, token) = (&client, &endpoint, token.as_deref());
    let deadline = tokio::time::Instant::now() + FIT_BUDGET;
    // A subprocess on Linux/Windows, so it runs alongside the lookups.
    let probe =
        tokio::task::spawn_blocking(|| hostgpu::memory_bytes(hostgpu::detect_with_vram().1));
    // Unordered, so a stalled lookup does not hold up the rest.
    let found: Vec<(usize, Option<u64>)> = stream::iter(hits.iter().enumerate())
        .map(|(i, hit)| async move {
            let size = default_size(client, endpoint, token, &hit.name);
            (
                i,
                tokio::time::timeout_at(deadline, size).await.ok().flatten(),
            )
        })
        .buffer_unordered(SIZE_LOOKUPS)
        .collect()
        .await;
    let mut sizes = vec![None; hits.len()];
    for (i, size) in found {
        sizes[i] = size;
    }
    // A probe that hangs or fails leaves system RAM.
    let memory = match tokio::time::timeout_at(deadline, probe).await {
        Ok(Ok(bytes)) => bytes,
        _ => hostgpu::memory_bytes(0),
    };
    sizes
        .into_iter()
        .map(|size| fit_percent(size?, memory))
        .collect()
}

/// Bytes `pull` downloads for `name` with no tag: the default variant of
/// its web UI card.
async fn default_size(
    client: &reqwest::Client,
    endpoint: &str,
    token: Option<&str>,
    name: &str,
) -> Option<u64> {
    if let Some(repo) = name.strip_prefix("hf.co/") {
        let (owner, repo_name) = repo.split_once('/')?;
        let url = hf::api::files_url(endpoint, owner, repo_name, "main");
        let files: Vec<hf::api::HfFile> = hf::api::get_json_quiet(client, &url, token).await?;
        hugging_face_variants(name, &files)
            .into_iter()
            .find(|v| v.default)?
            .size
    } else {
        let (namespace, repo_name) = name.strip_prefix("docker.io/")?.split_once('/')?;
        let url = format!("{}/tags/latest", docker_hub_repo_url(namespace, repo_name));
        let tag: HubTag = hf::api::get_json_quiet(client, &url, None).await?;
        tag.full_size
            .filter(|_| tag.media_type == MODELPACK_MANIFEST)
    }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// `list`-style columns; with `color`, each FIT is painted by its tier.
/// `fits` runs alongside `hits`.
fn render(hits: &[Hit], fits: &[Option<u64>], color: bool) -> String {
    let dash = || "-".to_string();
    let header = ["NAME", "PULLS", "LIKES", "UPDATED", "FIT"].map(String::from);
    let mut rows = vec![header.to_vec()];
    for (h, fit) in hits.iter().zip(fits) {
        rows.push(vec![
            h.name.clone(),
            h.pulls.map(human_count).unwrap_or_else(dash),
            h.likes.map(|n| n.to_string()).unwrap_or_else(dash),
            h.updated
                .as_deref()
                .map(relative_time_rfc3339)
                .unwrap_or_else(dash),
            fit.map_or_else(dash, |p| paint(&format!("{p}%"), fit_color(p), color)),
        ]);
    }
    columns(&rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Relevance order buries `ai/qwen3.8` under `<user>/qwen` images;
    /// both pages must ask for pull-count order.
    #[test]
    fn docker_hub_search_url_sorts_by_pulls() {
        for (kind, suffix) in [(Some("model"), "&type=model"), (None, "")] {
            let url = docker_hub_search_url("qwen", kind).unwrap();
            assert_eq!(
                url.as_str(),
                format!(
                    "{DOCKER_HUB_SEARCH_URL}?query=qwen&from=0&size=64\
                     &sort=pull_count&order=desc{suffix}"
                )
            );
        }
    }

    /// A `type=model` page and an unfiltered page, plus rows that must
    /// not print: Model Runner-only, archived, a container image, a raw
    /// Hugging Face mirror, one repeated across pages, and one past
    /// `limit`. ModelPack-only `ai/qwen3.8` appears only on the
    /// unfiltered page and must print, ranked by pulls among the first
    /// page's rows.
    #[test]
    fn docker_hub_rows_become_pullable_references() {
        let typed = r#"{"total": 4, "results": [
            {"name": "ai/qwen3", "short_description": "Qwen3 LLM", "star_count": 210,
             "pull_count": "500K+", "raw_pull_count": 614283,
             "updated_at": "2026-08-17T14:03:18.618971Z", "archived": false,
             "media_types": ["application/vnd.cncf.model.manifest.v1+json",
                             "application/vnd.docker.ai.model.config.v0.1+json"],
             "content_types": ["model"]},
            {"name": "ai/qwen2.5", "star_count": 13, "raw_pull_count": 163532,
             "media_types": ["application/vnd.docker.ai.model.config.v0.1+json"],
             "content_types": ["model"]},
            {"name": "ai/qwen3-embedding", "star_count": 1, "raw_pull_count": 74850,
             "media_types": ["application/vnd.cncf.model.manifest.v1+json",
                             "application/vnd.docker.ai.model.config.v0.1+json"],
             "content_types": ["model"]},
            {"name": "ai/old", "archived": true, "star_count": 0,
             "media_types": ["application/vnd.cncf.model.manifest.v1+json",
                             "application/vnd.docker.ai.model.config.v0.1+json"],
             "content_types": ["model"]}
        ]}"#;
        let untyped = r#"{"total": 6, "results": [
            {"name": "ai/qwen3", "star_count": 210, "raw_pull_count": 614283,
             "media_types": ["application/vnd.cncf.model.manifest.v1+json",
                             "application/vnd.docker.ai.model.config.v0.1+json"],
             "content_types": ["model"]},
            {"name": "library/nginx", "star_count": 99999, "raw_pull_count": 9999999,
             "media_types": ["application/vnd.docker.distribution.manifest.v2+json"],
             "content_types": ["image"]},
            {"name": "ai/qwen3.8", "star_count": 4, "raw_pull_count": 98764,
             "media_types": ["application/vnd.cncf.model.manifest.v1+json"],
             "content_types": ["unrecognized"]},
            {"name": "someone/qwen3.8-27b", "star_count": 0, "raw_pull_count": 90000,
             "media_types": ["application/vnd.huggingface.model.v1"],
             "content_types": ["unrecognized"]},
            {"name": "official-thing", "star_count": 1,
             "media_types": ["application/vnd.cncf.model.manifest.v1+json"]},
            {"name": "ai/third", "star_count": 0,
             "media_types": ["application/vnd.cncf.model.manifest.v1+json"]}
        ]}"#;
        let typed: HubSearchResponse = serde_json::from_str(typed).unwrap();
        let untyped: HubSearchResponse = serde_json::from_str(untyped).unwrap();
        let hits = docker_hub_hits(vec![typed, untyped], 4);
        let names: Vec<&str> = hits.iter().map(|h| h.name.as_str()).collect();
        // A bare name is the official `library/` namespace. Rows with no
        // pull count sort last, keeping Hub's order among themselves.
        assert_eq!(
            names,
            [
                "docker.io/ai/qwen3",
                "docker.io/ai/qwen3.8",
                "docker.io/ai/qwen3-embedding",
                "docker.io/library/official-thing"
            ]
        );
        assert_eq!(
            hits[0],
            Hit {
                name: "docker.io/ai/qwen3".into(),
                pulls: Some(614283),
                likes: Some(210),
                updated: Some("2026-08-17T14:03:18.618971Z".into()),
            }
        );
        assert_eq!(hits[3].pulls, None);
        assert_eq!(hits[3].updated, None);
    }

    #[test]
    fn hugging_face_rows_become_pullable_references() {
        let body = r#"[
            {"id": "unsloth/Qwen3-8B-GGUF", "likes": 989, "downloads": 12853002,
             "lastModified": "2025-07-31T10:27:38.000Z"},
            {"id": "someone/bare"}
        ]"#;
        let models: Vec<hf::api::SearchHit> = serde_json::from_str(body).unwrap();
        let hits: Vec<Hit> = models.into_iter().map(hugging_face_hit).collect();
        assert_eq!(
            hits[0],
            Hit {
                name: "hf.co/unsloth/Qwen3-8B-GGUF".into(),
                pulls: Some(12853002),
                likes: Some(989),
                updated: Some("2025-07-31T10:27:38.000Z".into()),
            }
        );
        assert_eq!(hits[1].pulls, Some(0));
        assert_eq!(hits[1].updated, None);
    }

    #[test]
    fn a_long_readme_is_cut_on_a_character_boundary() {
        assert_eq!(cap_readme("short".into()), "short");
        let long = "é".repeat(README_MAX); // two bytes each
        let cut = cap_readme(long);
        assert!(cut.len() <= README_MAX + "\n\n…".len());
        assert!(cut.ends_with("\n\n…"));
    }

    #[test]
    fn gguf_quant_labels_come_from_the_file_name() {
        assert_eq!(
            gguf_quant_label("Qwen3.5-0.8B-Q4_K_M.gguf").as_deref(),
            Some("Q4_K_M")
        );
        assert_eq!(
            gguf_quant_label("Qwen3.5-0.8B-BF16.gguf").as_deref(),
            Some("BF16")
        );
        assert_eq!(
            gguf_quant_label("Q8_0/Big-Model-Q8_0-00001-of-00003.gguf").as_deref(),
            Some("Q8_0")
        );
        assert_eq!(gguf_quant_label("mmproj-F16.gguf"), None);
        assert_eq!(gguf_quant_label("README.md"), None);
    }

    fn file_of(path: &str, size: i64) -> hf::api::HfFile {
        hf::api::HfFile {
            path: path.into(),
            size,
            kind: "file".into(),
        }
    }

    #[test]
    fn hugging_face_variants_are_tags_that_pull_their_own_file() {
        // Nothing `pull` can take (it would refuse the repo): no variant.
        let none = hugging_face_variants(
            "hf.co/o/N",
            &[file_of("model.onnx", 99), file_of(".gitattributes", 1)],
        );
        assert!(none.is_empty());

        let file = file_of;
        let files = [
            file("M-Q4_K_M.gguf", 500),
            file("M-Q8_0-00001-of-00002.gguf", 400),
            file("M-Q8_0-00002-of-00002.gguf", 400),
            file("mmproj-F16.gguf", 90),
        ];
        let variants = hugging_face_variants("hf.co/o/M", &files);
        let got: Vec<_> = variants
            .iter()
            .map(|v| (v.name.as_str(), v.size, v.default))
            .collect();
        // Each with the projector `pull` adds to it.
        assert_eq!(
            got,
            [
                ("hf.co/o/M:Q4_K_M", Some(590), true),
                ("hf.co/o/M:Q8_0", Some(890), false),
            ]
        );

        // A safetensors repo is one download, named as a tagless pull is stored.
        let weights = [
            file("model-00001-of-00002.safetensors", 300),
            file("model-00002-of-00002.safetensors", 200),
            file("config.json", 1),
        ];
        let only = hugging_face_variants("hf.co/o/S", &weights);
        assert_eq!(only.len(), 1);
        assert_eq!(only[0].name, "hf.co/o/S:latest");
        assert!(only[0].default);
    }

    fn hit(name: &str, pulls: u64, likes: u64) -> Hit {
        Hit {
            name: name.into(),
            pulls: Some(pulls),
            likes: Some(likes),
            updated: None,
        }
    }

    #[test]
    fn render_aligns_columns_and_dashes_missing_values() {
        let hits = vec![
            hit("docker.io/ai/qwen3", 614_283, 210),
            hit("hf.co/unsloth/Qwen3-8B-GGUF", 12_853_002, 989),
        ];
        let out = render(&hits, &[Some(42), None], false);
        assert_eq!(
            out,
            "NAME                           PULLS     LIKES    UPDATED    FIT\n\
             docker.io/ai/qwen3             614.3K    210      -          42%\n\
             hf.co/unsloth/Qwen3-8B-GGUF    12.9M     989      -          -\n"
        );
    }

    /// The web UI's tiers: 60% exactly is still green, 90% still yellow.
    #[test]
    fn fit_percent_rounds_up_so_it_never_reads_under_a_tier_it_has_left() {
        let memory = 1_000;
        let at = |size| fit_percent(size, memory).unwrap();
        assert_eq!(at(600), 60);
        assert_eq!(at(601), 61);
        assert_eq!(at(900), 90);
        assert_eq!(at(901), 91);
        assert_eq!(at(1), 1);
        assert_eq!(at(3_720), 372);

        let color = |size| fit_color(at(size));
        assert_eq!(color(600), GREEN);
        assert_eq!(color(601), YELLOW);
        assert_eq!(color(900), YELLOW);
        assert_eq!(color(901), RED);
        assert_eq!(color(3_720), RED);
    }

    #[test]
    fn fit_is_unknown_without_a_size_or_memory() {
        assert_eq!(fit_percent(0, 16_000_000_000), None);
        assert_eq!(fit_percent(5_000_000_000, 0), None);
        assert_eq!(fit_percent(u64::MAX, 1), Some(u64::MAX));
    }

    /// Color wraps only the last cell, so the columns line up as uncolored.
    #[test]
    fn render_paints_fit_by_tier_and_leaves_the_rest_alone() {
        let hits = vec![
            hit("docker.io/ai/a", 1, 1),
            hit("docker.io/ai/b", 1, 1),
            hit("docker.io/ai/c", 1, 1),
            hit("docker.io/ai/d", 1, 1),
        ];
        let fits = [Some(30), Some(75), Some(140), None];
        let plain = render(&hits, &fits, false);
        let colored = render(&hits, &fits, true);
        assert_eq!(
            colored,
            "NAME              PULLS    LIKES    UPDATED    FIT\n\
             docker.io/ai/a    1        1        -          \x1b[32m30%\x1b[0m\n\
             docker.io/ai/b    1        1        -          \x1b[33m75%\x1b[0m\n\
             docker.io/ai/c    1        1        -          \x1b[31m140%\x1b[0m\n\
             docker.io/ai/d    1        1        -          -\n"
        );
        let stripped = colored
            .replace(GREEN, "")
            .replace(YELLOW, "")
            .replace(RED, "")
            .replace("\x1b[0m", "");
        assert_eq!(stripped, plain);
    }
}
