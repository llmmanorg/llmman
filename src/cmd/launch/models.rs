//! The one list of models every launch offers its agent, and the rule that picks the one it starts
//! on.
//!
//! Built once from the daemon's `GET /llmman/models` (see [`crate::daemon::launch_models`]) and
//! mapped by each integration the same way: the starting model is the first row. Building it loads
//! no model and asks nothing per model or per provider.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::Context;

use super::PORTABLE_THINKING_LEVELS;
use crate::daemon::{LaunchModels, LocalModel, ModelCost, ProviderDetail, ProviderModel};
use crate::providers;

/// One model an agent can switch to.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ModelEntry {
    /// What the agent sends as the model name, which the daemon routes: a
    /// stored model's name, or [`providers::format_remote_ref`].
    pub reference: String,
    /// The plain id to show: the provider's own id for a hosted model, a
    /// stored model's name without the `docker.io/ai/` default.
    pub id: String,
    /// The provider's id and name; `None` for a model stored here.
    pub provider: Option<(String, String)>,
    pub context_window: Option<u64>,
    pub max_output: Option<u32>,
    /// The thinking choices the catalog or chat template names, in cycle
    /// order; empty for a model that does not think, `None` where nothing
    /// says (see [`ModelEntry::variants`]).
    pub thinking: Option<Vec<String>>,
    pub vision: bool,
    /// `None` where nothing says.
    pub tools: Option<bool>,
    /// What the catalog charges per million tokens; `None` for a stored
    /// model, which costs nothing, and where the catalog says nothing.
    pub cost: Option<ModelCost>,
    /// When the catalog says it was released (`YYYY-MM-DD` or `YYYY-MM`).
    pub released: Option<String>,
}

impl ModelEntry {
    pub(super) fn local(model: &LocalModel) -> Self {
        Self {
            reference: model.name.clone(),
            id: local_id(&model.name).to_string(),
            provider: None,
            context_window: model.context,
            max_output: None,
            thinking: model.thinking.clone(),
            vision: model.vision,
            tools: Some(model.tools),
            cost: None,
            released: None,
        }
    }

    pub(super) fn hosted(provider: &ProviderDetail, model: &ProviderModel) -> Self {
        Self {
            reference: providers::format_remote_ref(&provider.id, &model.id),
            id: model.id.clone(),
            provider: Some((provider.id.clone(), provider.name.clone())),
            context_window: model.context,
            max_output: model.output,
            thinking: model.thinking.clone(),
            vision: model.vision,
            tools: model.tools,
            cost: model.cost,
            released: model.released.clone(),
        }
    }

    /// A model the daemon's list does not hold (a provider's model the
    /// catalog predates, a model not yet listed): named, and nothing else
    /// known.
    pub(super) fn unlisted(reference: &str) -> Self {
        let provider = providers::split_remote_ref(reference);
        Self {
            reference: reference.to_string(),
            id: provider
                .map_or_else(|| local_id(reference), |(_, model)| model)
                .to_string(),
            provider: provider.map(|(id, _)| (id.to_string(), id.to_string())),
            context_window: None,
            max_output: None,
            thinking: None,
            vision: false,
            tools: None,
            cost: None,
            released: None,
        }
    }

    /// The variants the model takes, in cycle order: what the catalog or
    /// template says, else [`PORTABLE_THINKING_LEVELS`], which every wire
    /// accepts.
    pub(super) fn variants(&self) -> Vec<String> {
        self.thinking.clone().unwrap_or_else(|| {
            PORTABLE_THINKING_LEVELS
                .iter()
                .map(|level| level.to_string())
                .collect()
        })
    }

    /// The variants `integration` has a word for (`spell_variant`), in
    /// order and without repeats.
    pub(super) fn spelled_variants(&self, integration: &str) -> Vec<String> {
        let mut levels: Vec<String> = Vec::new();
        for variant in self.variants() {
            if let Ok(level) = super::spell_variant(integration, &variant) {
                if !levels.iter().any(|l| l == level) {
                    levels.push(level.to_string());
                }
            }
        }
        levels
    }

    /// `provider/id`, or the id of a stored model: what tells two models of
    /// the same name apart.
    pub(super) fn label(&self) -> String {
        match &self.provider {
            Some((provider, _)) => format!("{provider}/{}", self.id),
            None => self.id.clone(),
        }
    }

    pub(super) fn provider_id(&self) -> Option<&str> {
        self.provider.as_ref().map(|(id, _)| id.as_str())
    }
}

/// A stored model's name as shown: without the `docker.io/ai/` that a bare
/// name defaults to (see [`crate::shortnames::resolve_ollama_api`]).
fn local_id(reference: &str) -> &str {
    reference.strip_prefix("docker.io/ai/").unwrap_or(reference)
}

/// Whether two names are one stored model, as the daemon resolves them.
fn same_local(a: &str, b: &str) -> bool {
    crate::storage::default_tag(a) == crate::storage::default_tag(b)
}

/// The models a launch offers, the one it starts on first.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct ModelList {
    entries: Vec<ModelEntry>,
}

impl ModelList {
    pub(super) fn entries(&self) -> &[ModelEntry] {
        &self.entries
    }

    pub(super) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether there is anything to switch to. An agent given a list of one
    /// needs no file, and is configured for its one model alone: all an
    /// `--sandbox openshell` agent, which cannot see a file, can be given.
    pub(super) fn offers_choice(&self) -> bool {
        self.entries.len() > 1
    }

    /// The model the launch starts on; `None` for a launch with none.
    pub(super) fn start(&self) -> Option<&ModelEntry> {
        self.entries.first()
    }

    /// [`stored_only`] of this list.
    pub(super) fn stored_only(&self, integration: &str, why: &str) -> Self {
        Self::of(stored_only(self.entries.clone(), integration, why))
    }

    /// A list of `entries` as given.
    pub(super) fn of(entries: Vec<ModelEntry>) -> Self {
        Self { entries }
    }

    /// The first `limit` entries, for an agent with a hard limit. The list leads with the start,
    /// its provider's models, the stored ones, then providers a key is held for, so what is left
    /// out is the least usable. Says so once, with `why`.
    pub(super) fn limited(&self, integration: &str, limit: usize, why: &str) -> Self {
        if self.entries.len() <= limit {
            return self.clone();
        }
        eprintln!(
            "[llmman] {integration}: offering the first {limit} of {} models: {why}",
            self.entries.len()
        );
        Self {
            entries: self.entries[..limit].to_vec(),
        }
    }

    /// What an agent's picker calls each row: the plain id, and the provider
    /// too where two rows would otherwise read alike.
    pub(super) fn display_namer(&self) -> impl Fn(&ModelEntry) -> String {
        let mut ids = std::collections::HashMap::<String, usize>::new();
        for entry in &self.entries {
            *ids.entry(entry.id.clone()).or_default() += 1;
        }
        move |entry| match ids.get(&entry.id) {
            Some(1) | None => entry.id.clone(),
            Some(_) => entry.label(),
        }
    }

    /// [`Self::display_namer`] for every row, in order.
    pub(super) fn display_names(&self) -> Vec<String> {
        let name = self.display_namer();
        self.entries.iter().map(name).collect()
    }

    /// `f` over every row with its [`Self::display_namer`] name, in order.
    pub(super) fn map_named<T>(&self, mut f: impl FnMut(&ModelEntry, &str) -> T) -> Vec<T> {
        let name = self.display_namer();
        self.entries
            .iter()
            .map(|entry| f(entry, &name(entry)))
            .collect()
    }

    /// One key per entry for a map an integration keys by name, made from
    /// `base` and told apart (`-2`, `-3`, ...) where two entries of
    /// different providers make the same one.
    pub(super) fn unique_keys(&self, base: impl Fn(&ModelEntry) -> String) -> Vec<String> {
        let mut seen: HashSet<String> = HashSet::new();
        self.entries
            .iter()
            .map(|entry| {
                let base = base(entry);
                let mut key = base.clone();
                let mut n = 1;
                while !seen.insert(key.clone()) {
                    n += 1;
                    key = format!("{base}-{n}");
                }
                key
            })
            .collect()
    }
}

/// Every chat model the daemon can serve, with the stored ones first and
/// the providers after them in the catalog's order: nothing that cannot
/// chat (embedding, speech, transcription, image and video models).
pub(super) fn offered(models: &LaunchModels) -> Vec<ModelEntry> {
    let local = models
        .local
        .iter()
        .filter(|m| m.chat)
        .map(ModelEntry::local);
    let hosted = models.providers.iter().flat_map(|provider| {
        provider
            .models
            .iter()
            .filter(|m| m.answers_chat())
            .map(move |m| ModelEntry::hosted(provider, m))
    });
    local.chain(hosted).collect()
}

/// `entries` without the hosted models, for an agent that cannot take one
/// (see `STORED_ONLY` in the launcher). Says so once, with `why`.
pub(super) fn stored_only(
    mut entries: Vec<ModelEntry>,
    integration: &str,
    why: &str,
) -> Vec<ModelEntry> {
    let before = entries.len();
    entries.retain(|e| e.provider.is_none());
    if entries.len() < before {
        eprintln!("[llmman] {integration}: offering the pulled models only: {why}");
    }
    entries
}

/// What a launch is asked to start on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Want<'a> {
    /// `--provider P --model M`.
    Hosted { provider: &'a str, model: &'a str },
    /// `--provider P`: its default model.
    HostedDefault { provider: &'a str },
    /// `--model M`, as the daemon resolves it.
    Local(&'a str),
    /// Neither flag. `remembered` is the model the agent last used, if it
    /// keeps one.
    Unspecified {
        remembered: Option<&'a str>,
        /// The stored model the daemon says is the one to go on with, if
        /// it says (see [`latest_stored`]).
        latest: Option<(&'a str, Latest)>,
    },
}

/// Why a launch starts on the model it does, for the one that nobody named.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Picked {
    Asked,
    /// `--provider` alone: the provider's default model.
    ProviderDefault,
    Remembered,
    Stored(Latest),
    Provider,
}

/// What made a stored model the one a bare launch starts on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Latest {
    /// It is loaded now.
    Loaded,
    /// It was pulled most recently.
    Pulled,
    /// Nothing told the stored models apart: the first by name.
    FirstByName,
}

/// The stored model a bare launch goes on with, by what the daemon reports at no cost (see
/// [`LocalModel`]): the one loaded now, else the most recently pulled, preferring those that call
/// tools. `None` when the daemon reports neither.
pub(super) fn latest_stored(models: &LaunchModels) -> Option<(&str, Latest)> {
    let chat = || models.local.iter().filter(|m| m.chat);
    let tools: Vec<&LocalModel> = chat().filter(|m| m.tools).collect();
    let pool = if tools.is_empty() {
        chat().collect()
    } else {
        tools
    };
    let by_name = |a: &&LocalModel, b: &&LocalModel| b.name.cmp(&a.name);
    if let Some(m) = pool.iter().copied().filter(|m| m.loaded).max_by(by_name) {
        return Some((m.name.as_str(), Latest::Loaded));
    }
    let newest = pool.iter().filter_map(|m| m.pulled).max()?;
    pool.iter()
        .copied()
        .filter(|m| m.pulled == Some(newest))
        .max_by(by_name)
        .map(|m| (m.name.as_str(), Latest::Pulled))
}

/// A provider's default model, by [`providers::default_model`]: the newest
/// of its tool-calling chat models priced at or below their median. Never an
/// image or embedding row, which are not chat models at all.
pub(super) fn default_model<'a>(
    entries: &'a [ModelEntry],
    provider: &str,
) -> Option<&'a ModelEntry> {
    providers::default_model(
        entries.iter().filter(|e| e.provider_id() == Some(provider)),
        |e| providers::Candidate {
            id: &e.id,
            tools: e.tools,
            input_price: e.cost.map(|c| c.input),
            released: e.released.as_deref(),
        },
    )
}

/// The model a launch starts on, and why. `usable` says whether a provider
/// can be spent on without further ado: a key here, a key the daemon will
/// spend, or none needed.
pub(super) fn resolve_start(
    want: Want<'_>,
    entries: &[ModelEntry],
    usable: impl Fn(&str) -> bool,
) -> anyhow::Result<(ModelEntry, Picked)> {
    let find = |reference: &str| {
        entries
            .iter()
            .find(|e| {
                e.reference == reference
                    || (e.provider.is_none() && same_local(&e.reference, reference))
            })
            .cloned()
    };
    Ok(match want {
        Want::Hosted { provider, model } => {
            let reference = providers::format_remote_ref(provider, model);
            (
                find(&reference).unwrap_or_else(|| ModelEntry::unlisted(&reference)),
                Picked::Asked,
            )
        }
        Want::HostedDefault { provider } => {
            let entry = default_model(entries, provider).with_context(|| {
                format!("{provider} lists no chat model to start on; name one with --model")
            })?;
            (entry.clone(), Picked::ProviderDefault)
        }
        Want::Local(model) => (
            find(model).unwrap_or_else(|| ModelEntry::unlisted(model)),
            Picked::Asked,
        ),
        Want::Unspecified { remembered, latest } => {
            if let Some(entry) = remembered.and_then(|name| {
                find(name).or_else(|| {
                    entries
                        .iter()
                        .find(|e| e.id == name || e.label() == name)
                        .cloned()
                })
            }) {
                // One it has no credential for would fail at its first request.
                if entry.provider_id().is_none_or(&usable) {
                    return Ok((entry, Picked::Remembered));
                }
            }
            if let Some((entry, latest)) = latest.and_then(|(name, latest)| {
                find(name)
                    .filter(|e| e.provider.is_none())
                    .map(|entry| (entry, latest))
            }) {
                return Ok((entry, Picked::Stored(latest)));
            }
            let stored = || entries.iter().filter(|e| e.provider.is_none());
            if let Some(entry) = stored()
                .filter(|e| e.tools == Some(true))
                .min_by(|a, b| a.reference.cmp(&b.reference))
                .or_else(|| stored().min_by(|a, b| a.reference.cmp(&b.reference)))
            {
                return Ok((entry.clone(), Picked::Stored(Latest::FirstByName)));
            }
            let mut seen = HashSet::new();
            let provider = entries
                .iter()
                .filter_map(ModelEntry::provider_id)
                .filter(|p| seen.insert(*p))
                .filter(|p| usable(p))
                .find_map(|p| default_model(entries, p));
            match provider {
                Some(entry) => (entry.clone(), Picked::Provider),
                None => anyhow::bail!(
                    "nothing to start on: no model is pulled and no provider has a key llmman can \
                     use.\nPull one (llmman pull gemma4), or set a provider's key, e.g. \
                     ANTHROPIC_API_KEY or `llmman config set providers.<id>.api_key <key>` \
                     (llmman providers lists them), and launch again."
                ),
            }
        }
    })
}

/// The list the agent is offered with `start` first, if it has one: then `first`'s models
/// when `--provider` named it, the stored models, the providers `usable`
/// says a key is held for, and the rest, each group in the order given.
pub(super) fn ordered(
    start: Option<ModelEntry>,
    mut entries: Vec<ModelEntry>,
    first: Option<&str>,
    usable: impl Fn(&str) -> bool,
) -> ModelList {
    if let Some(start) = &start {
        entries.retain(|e| e.reference != start.reference);
    }
    let group = |e: &ModelEntry| match e.provider_id() {
        Some(p) if Some(p) == first => 0,
        None => 1,
        Some(p) if usable(p) => 2,
        Some(_) => 3,
    };
    entries.sort_by_key(group);
    let mut all: Vec<ModelEntry> = start.into_iter().collect();
    all.extend(entries);
    ModelList::of(all)
}

// ---------------------------------------------------------------------------
// The file the list travels in
// ---------------------------------------------------------------------------

/// The directory `integration` keeps its own state in, which the agent
/// reads and a `--sandbox` mounts: where [`write_transient`] puts what the
/// agent has to read.
pub(super) fn state_dir(integration: &str) -> anyhow::Result<PathBuf> {
    super::sandbox_state(integration)?
        .into_iter()
        .find_map(|state| match state {
            super::sandbox::State::Dir(dir) => Some(dir),
            super::sandbox::State::Files(_) => None,
        })
        .with_context(|| format!("{integration} has no state directory"))
}

/// Where the ids of the rows a launch wrote into `key`'s configuration are
/// kept, beside it: what tells a row llmman wrote from one the user added.
fn rows_file(dir: &Path, key: &str) -> PathBuf {
    dir.join(format!(".llmman-{key}-rows.json"))
}

/// The ids of the rows the last launch wrote into `key`'s configuration
/// ([`remember_rows`]), none when there is no record.
pub(super) fn rows_written(dir: &Path, key: &str) -> HashSet<String> {
    std::fs::read(rows_file(dir, key))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Vec<String>>(&bytes).ok())
        .map(|ids| ids.into_iter().collect())
        .unwrap_or_default()
}

/// Records the ids of `rows`, which a launch has just written into `key`'s
/// configuration, for the next to tell its own rows from the user's.
pub(super) fn remember_rows(
    dir: &Path,
    key: &str,
    rows: &[serde_json::Value],
) -> anyhow::Result<()> {
    let ids: Vec<&str> = rows.iter().filter_map(|r| r["id"].as_str()).collect();
    let path = rows_file(dir, key);
    crate::fsutil::write_atomic(&path, serde_json::to_string(&ids)?.as_bytes())
        .with_context(|| format!("write {}", path.display()))
}

/// What launches of one agent share, such as the rows each wrote into its one configuration file: a
/// shared advisory lock on a file beside it, held while the launch uses them and released by the
/// system should it die. Whoever leaves last takes the rows out ([`SharedUse::release_if_last`]); a
/// launch starting meanwhile waits.
pub(super) struct SharedUse {
    file: std::fs::File,
}

impl SharedUse {
    /// Takes the shared lock on `lock`, making the file if there is none.
    pub(super) fn claim(lock: &Path) -> anyhow::Result<Self> {
        if let Some(dir) = lock.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(lock)
            .with_context(|| format!("open {}", lock.display()))?;
        file.lock_shared()
            .with_context(|| format!("lock {}", lock.display()))?;
        Ok(Self { file })
    }

    /// Lets go of this launch's claim; `Some` while no other launch holds
    /// one, which then cannot start until the result is dropped.
    pub(super) fn release_if_last(self) -> Option<std::fs::File> {
        self.file.unlock().ok()?;
        self.file.try_lock().ok()?;
        Some(self.file)
    }
}

/// Files [`write_transient`] wrote, removed by [`finish`], each with the
/// shared lock the launch holds on it while it runs.
static WRITTEN: Mutex<Vec<(PathBuf, std::fs::File)>> = Mutex::new(Vec::new());

/// A file this old, of the name [`write_transient`] gives, is one a launch
/// that was killed left behind, unless a launch still holds its lock.
const STALE_AFTER: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);

/// Writes `contents` to `llmman-<stem>-<pid>.<ext>` in `dir`, readable by the user alone,
/// atomically, and removed by [`finish`]. `dir` is the integration's own state directory, which a
/// `--sandbox` mounts (a container does not see the host's temp directory).
///
/// The list is thousands of rows, which no variable or argument can carry (`E2BIG` on Linux and
/// macOS, 32k on Windows), and holds no secret.
pub(super) fn write_transient(
    dir: &Path,
    stem: &str,
    ext: &str,
    contents: &str,
) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    sweep(dir, stem, ext);
    let path = dir.join(format!("llmman-{stem}-{}.{ext}", std::process::id()));
    crate::fsutil::write_private(&path, contents.as_bytes())
        .with_context(|| format!("write {}", path.display()))?;
    // A shared lock, which the agent reading the file does not mind and
    // which tells another launch's sweep that this one still runs.
    let held = std::fs::File::open(&path).with_context(|| format!("open {}", path.display()))?;
    held.lock_shared()
        .with_context(|| format!("lock {}", path.display()))?;
    WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((path.clone(), held));
    Ok(path)
}

/// Removes what a killed launch left in `dir`: files of the name
/// [`write_transient`] gives that are old and that no running launch holds
/// the lock of (one that has run for a day is not a stale one).
fn sweep(dir: &Path, stem: &str, ext: &str) {
    let (prefix, suffix) = (format!("llmman-{stem}-"), format!(".{ext}"));
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let old = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|age| age > STALE_AFTER);
        if old && name.starts_with(&prefix) && name.ends_with(&suffix) {
            let unclaimed = std::fs::File::open(entry.path()).is_ok_and(|f| f.try_lock().is_ok());
            if unclaimed {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
}

/// Ends what a launch set up for its agent: the files [`write_transient`]
/// wrote, and the session holding its keys (see [`super::keys`]).
pub(super) fn finish() {
    for (path, held) in WRITTEN.lock().unwrap_or_else(|e| e.into_inner()).drain(..) {
        drop(held);
        let _ = std::fs::remove_file(path);
    }
    super::keys::close();
}

/// [`finish`] for one file, so a test does not end every other test's.
#[cfg(test)]
pub(super) fn finish_file(path: &Path) {
    WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|(p, _)| p != path);
    let _ = std::fs::remove_file(path);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daemon::LaunchModels;

    fn models(json: serde_json::Value) -> LaunchModels {
        serde_json::from_value(json).expect("a model list")
    }

    fn catalog() -> LaunchModels {
        models(serde_json::json!({
            "local": [
                { "name": "docker.io/ai/qwen3.5:0.8b", "chat": true, "vision": false,
                  "tools": true, "context": 32768, "thinking": ["none", "thinking"] },
                { "name": "docker.io/ai/nomic-embed", "chat": false },
                { "name": "hf.co/o/tiny", "chat": true, "tools": false },
            ],
            "providers": [
                { "id": "alpha", "name": "Alpha", "base_url": "https://a", "key_env": "ALPHA_KEY",
                  "key_set": false, "models": [
                    { "id": "alpha-image", "chat": false },
                    { "id": "alpha-plain", "chat": true, "tools": false },
                    { "id": "alpha-agent", "chat": true, "tools": true, "vision": true,
                      "thinking": ["low", "high"], "context": 200000, "output": 8000 },
                    { "id": "text-embedding-3", "chat": false },
                ] },
                { "id": "beta", "name": "Beta", "base_url": "https://b", "key_env": "BETA_KEY",
                  "key_set": true, "models": [
                    { "id": "beta-1", "tools": true },
                ] },
                { "id": "empty", "name": "Empty", "base_url": "https://e", "key_set": false,
                  "models": [ { "id": "only-image", "chat": false } ] },
            ],
        }))
    }

    #[test]
    fn only_chat_models_are_offered() {
        let entries = offered(&catalog());
        let labels: Vec<String> = entries.iter().map(ModelEntry::label).collect();
        assert_eq!(
            labels,
            [
                "qwen3.5:0.8b",
                "hf.co/o/tiny",
                "alpha/alpha-plain",
                "alpha/alpha-agent",
                "beta/beta-1"
            ]
        );
        let agent = &entries[3];
        assert_eq!(agent.reference, "llmman.provider/alpha/alpha-agent");
        assert!(agent.vision);
        assert_eq!(agent.context_window, Some(200_000));
        assert_eq!(agent.max_output, Some(8000));
        assert_eq!(agent.variants(), ["low", "high"]);
        // Nothing said: what every wire accepts.
        assert_eq!(entries[2].variants(), ["none", "low", "medium", "high"]);
        assert_eq!(entries[0].variants(), ["none", "thinking"]);
    }

    #[test]
    fn an_older_daemon_is_judged_by_the_id() {
        let old = models(serde_json::json!({ "local": [], "providers": [
            { "id": "p", "name": "P", "base_url": "https://p", "key_set": false, "models": [
                { "id": "gpt-5" }, { "id": "text-embedding-3-small" }, { "id": "whisper-1" },
                { "id": "gpt-image-1" }, { "id": "omni-tts" },
            ] },
        ] }));
        let ids: Vec<String> = offered(&old).iter().map(|e| e.id.clone()).collect();
        assert_eq!(ids, ["gpt-5"]);
    }

    #[test]
    fn a_providers_default_is_a_tool_calling_chat_model_not_the_first_listed() {
        let entries = offered(&catalog());
        assert_eq!(default_model(&entries, "alpha").unwrap().id, "alpha-agent");
        // Never the image row, though it is listed first.
        assert_eq!(default_model(&entries, "beta").unwrap().id, "beta-1");
        assert!(default_model(&entries, "empty").is_none());
        assert!(default_model(&entries, "missing").is_none());
    }

    fn dated() -> Vec<ModelEntry> {
        let priced = serde_json::json!({ "local": [], "providers": [
            { "id": "p", "name": "P", "base_url": "https://p", "key_set": false, "models": [
                { "id": "a-flagship", "tools": true, "released": "2026-10-01",
                  "cost": { "input": 15.0, "output": 75.0 } },
                { "id": "b-old", "tools": true, "released": "2025-01-01",
                  "cost": { "input": 1.0, "output": 5.0 } },
                { "id": "c-mid", "tools": true, "released": "2026-06-15",
                  "cost": { "input": 3.0, "output": 15.0 } },
                { "id": "d-small", "tools": true, "released": "2026-09",
                  "cost": { "input": 1.0, "output": 5.0 } },
                { "id": "e-unpriced", "tools": true, "released": "2026-08-01" },
                { "id": "f-no-tools", "tools": false, "released": "2026-12-01" },
                { "id": "g-no-date", "tools": true, "cost": { "input": 0.1, "output": 0.2 } },
            ] },
            { "id": "q", "name": "Q", "base_url": "https://q", "key_set": false, "models": [
                { "id": "q-one", "tools": true, "released": "2026-05-01" },
                { "id": "q-two", "tools": true, "released": "2026-05-01" },
            ] },
        ] });
        offered(&models(priced))
    }

    #[test]
    fn the_default_is_the_newest_model_at_or_below_the_median_price() {
        let entries = dated();
        // Prices of the tool-calling models: 0.1, 1, 1, 3, 15 (the unpriced
        // e-unpriced has none): the median is 1, so a-flagship and c-mid are
        // out, and d-small is the newest of what is left.
        assert_eq!(default_model(&entries, "p").unwrap().id, "d-small");
    }

    #[test]
    fn a_tie_in_date_goes_to_the_greater_id_whatever_the_listing_order() {
        let entries = dated();
        assert_eq!(default_model(&entries, "q").unwrap().id, "q-two");
        let mut reversed = entries.clone();
        reversed.reverse();
        assert_eq!(default_model(&reversed, "q").unwrap().id, "q-two");
    }

    fn usable(provider: &str) -> bool {
        provider == "beta"
    }

    #[test]
    fn provider_and_model_start_on_that_model_even_unlisted() {
        let entries = offered(&catalog());
        let (start, picked) = resolve_start(
            Want::Hosted {
                provider: "alpha",
                model: "alpha-agent",
            },
            &entries,
            usable,
        )
        .unwrap();
        assert_eq!(picked, Picked::Asked);
        // Vision and the window come from the catalog on this path too.
        assert!(start.vision);
        assert_eq!(start.context_window, Some(200_000));
        let (start, _) = resolve_start(
            Want::Hosted {
                provider: "alpha",
                model: "brand-new",
            },
            &entries,
            usable,
        )
        .unwrap();
        assert_eq!(start.reference, "llmman.provider/alpha/brand-new");
        assert_eq!(start.id, "brand-new");
        assert!(!start.vision);
    }

    #[test]
    fn a_provider_alone_starts_on_its_default_and_an_empty_one_is_refused() {
        let entries = offered(&catalog());
        let (start, picked) =
            resolve_start(Want::HostedDefault { provider: "alpha" }, &entries, usable).unwrap();
        assert_eq!(
            (start.id.as_str(), picked),
            ("alpha-agent", Picked::ProviderDefault)
        );
        let err = resolve_start(Want::HostedDefault { provider: "empty" }, &entries, usable)
            .unwrap_err()
            .to_string();
        assert!(err.contains("empty lists no chat model"), "{err}");
    }

    #[test]
    fn a_model_alone_starts_on_the_stored_model_however_it_is_spelled() {
        let entries = offered(&catalog());
        let (start, picked) =
            resolve_start(Want::Local("docker.io/ai/qwen3.5:0.8b"), &entries, usable).unwrap();
        assert_eq!(picked, Picked::Asked);
        assert_eq!(start.id, "qwen3.5:0.8b");
        assert_eq!(start.context_window, Some(32768));
        // Untagged is `:latest`, as the daemon resolves it.
        for spelled in ["hf.co/o/tiny", "hf.co/o/tiny:latest"] {
            let (start, _) = resolve_start(Want::Local(spelled), &entries, usable).unwrap();
            assert_eq!(start.id, "hf.co/o/tiny", "{spelled}");
            assert_eq!(start.tools, Some(false));
        }
    }

    #[test]
    fn with_neither_the_remembered_model_comes_first_if_it_is_offered() {
        let entries = offered(&catalog());
        for (remembered, id) in [
            ("alpha-agent", "alpha-agent"),
            ("beta/beta-1", "beta-1"),
            ("llmman.provider/alpha/alpha-plain", "alpha-plain"),
            ("hf.co/o/tiny", "tiny"),
        ] {
            let (start, picked) = resolve_start(
                Want::Unspecified {
                    remembered: Some(remembered),
                    latest: None,
                },
                &entries,
                |_| true,
            )
            .unwrap();
            assert_eq!(picked, Picked::Remembered, "{remembered}");
            assert!(start.id.ends_with(id), "{remembered}: {}", start.id);
        }
    }

    #[test]
    fn with_neither_a_stored_model_then_a_providers_default_are_picked() {
        let entries = offered(&catalog());
        // Not offered: the stored model is the next choice, tool-calling first.
        let (start, picked) = resolve_start(
            Want::Unspecified {
                remembered: Some("gone"),
                latest: None,
            },
            &entries,
            usable,
        )
        .unwrap();
        assert_eq!(
            (start.id.as_str(), picked),
            ("qwen3.5:0.8b", Picked::Stored(Latest::FirstByName))
        );
        // Offered, but its provider has no key: not worth starting on.
        let (start, picked) = resolve_start(
            Want::Unspecified {
                remembered: Some("alpha-agent"),
                latest: None,
            },
            &entries,
            usable,
        )
        .unwrap();
        assert_eq!(
            (start.id.as_str(), picked),
            ("qwen3.5:0.8b", Picked::Stored(Latest::FirstByName))
        );

        let hosted_only: Vec<ModelEntry> = entries
            .iter()
            .filter(|e| e.provider.is_some())
            .cloned()
            .collect();
        let (start, picked) = resolve_start(
            Want::Unspecified {
                remembered: None,
                latest: None,
            },
            &hosted_only,
            usable,
        )
        .unwrap();
        // `alpha` sorts first but has no key.
        assert_eq!((start.id.as_str(), picked), ("beta-1", Picked::Provider));
        assert_eq!(start.provider_id(), Some("beta"));

        let err = resolve_start(
            Want::Unspecified {
                remembered: None,
                latest: None,
            },
            &hosted_only,
            |_| false,
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("llmman pull") && err.contains("api_key"),
            "{err}"
        );
    }

    #[test]
    fn stored_only_drops_every_hosted_row() {
        let entries = offered(&catalog());
        let kept = stored_only(entries.clone(), "x", "a test");
        assert!(kept.iter().all(|e| e.provider.is_none()));
        assert_eq!(kept.len(), 2);
        let list = ModelList::of(entries).stored_only("x", "a test");
        assert_eq!(list.entries().len(), 2);
    }

    fn stored(rows: serde_json::Value) -> LaunchModels {
        models(serde_json::json!({ "local": rows, "providers": [] }))
    }

    #[test]
    fn the_loaded_stored_model_is_preferred_then_the_most_recently_pulled() {
        // The daemon says nothing: no rule, so the first by name stays.
        let silent = stored(serde_json::json!([
            { "name": "a", "chat": true, "tools": true },
            { "name": "b", "chat": true, "tools": true },
        ]));
        assert_eq!(latest_stored(&silent), None);

        let newest = stored(serde_json::json!([
            { "name": "a", "chat": true, "tools": true, "pulled": 100 },
            { "name": "b", "chat": true, "tools": true, "pulled": 300 },
            { "name": "c", "chat": true, "tools": true, "pulled": 200 },
            { "name": "x-embed", "chat": false, "pulled": 900 },
            { "name": "y-notools", "chat": true, "tools": false, "pulled": 800 },
        ]));
        assert_eq!(latest_stored(&newest), Some(("b", Latest::Pulled)));

        let loaded = stored(serde_json::json!([
            { "name": "a", "chat": true, "tools": true, "pulled": 100 },
            { "name": "b", "chat": true, "tools": true, "pulled": 300 },
            { "name": "c", "chat": true, "tools": true, "pulled": 200, "loaded": true },
        ]));
        assert_eq!(latest_stored(&loaded), Some(("c", Latest::Loaded)));

        // Without a tool-calling model, the others are all there is.
        let plain = stored(serde_json::json!([
            { "name": "a", "chat": true, "tools": false, "pulled": 100 },
            { "name": "b", "chat": true, "tools": false, "pulled": 200 },
        ]));
        assert_eq!(latest_stored(&plain), Some(("b", Latest::Pulled)));

        // The same second: the first by name.
        let tie = stored(serde_json::json!([
            { "name": "b", "chat": true, "tools": true, "pulled": 100 },
            { "name": "a", "chat": true, "tools": true, "pulled": 100 },
        ]));
        assert_eq!(latest_stored(&tie), Some(("a", Latest::Pulled)));
        assert_eq!(latest_stored(&stored(serde_json::json!([]))), None);
    }

    #[test]
    fn with_neither_the_daemons_latest_stored_model_beats_the_first_by_name() {
        let catalog = stored(serde_json::json!([
            { "name": "a", "chat": true, "tools": true, "pulled": 100 },
            { "name": "b", "chat": true, "tools": true, "pulled": 300 },
        ]));
        let entries = offered(&catalog);
        let (start, picked) = resolve_start(
            Want::Unspecified {
                remembered: None,
                latest: latest_stored(&catalog),
            },
            &entries,
            usable,
        )
        .unwrap();
        assert_eq!(
            (start.id.as_str(), picked),
            ("b", Picked::Stored(Latest::Pulled))
        );
        // The agent's own remembered model still comes first.
        let (start, picked) = resolve_start(
            Want::Unspecified {
                remembered: Some("a"),
                latest: latest_stored(&catalog),
            },
            &entries,
            usable,
        )
        .unwrap();
        assert_eq!((start.id.as_str(), picked), ("a", Picked::Remembered));
    }

    #[test]
    fn the_list_leads_with_the_start_then_its_provider_then_what_can_be_used() {
        let entries = offered(&catalog());
        let (start, _) = resolve_start(Want::Local("hf.co/o/tiny"), &entries, usable).unwrap();
        let list = ordered(Some(start), entries.clone(), Some("alpha"), usable);
        let labels: Vec<String> = list.entries().iter().map(ModelEntry::label).collect();
        assert_eq!(
            labels,
            [
                "hf.co/o/tiny",
                "alpha/alpha-plain",
                "alpha/alpha-agent",
                "qwen3.5:0.8b",
                "beta/beta-1"
            ]
        );
        assert_eq!(list.start().map(|e| e.id.as_str()), Some("hf.co/o/tiny"));

        let (start, _) = resolve_start(Want::Local("hf.co/o/tiny"), &entries, usable).unwrap();
        let list = ordered(Some(start), entries, None, usable);
        let labels: Vec<String> = list.entries().iter().map(ModelEntry::label).collect();
        // Stored, then the provider with a key, then the one without.
        assert_eq!(
            labels,
            [
                "hf.co/o/tiny",
                "qwen3.5:0.8b",
                "beta/beta-1",
                "alpha/alpha-plain",
                "alpha/alpha-agent"
            ]
        );
    }

    #[test]
    fn a_limit_keeps_the_front_of_the_list_and_keys_stay_unique() {
        let entries = offered(&catalog());
        let list = ModelList::of(entries);
        let limited = list.limited("x", 2, "a test");
        assert_eq!(limited.entries().len(), 2);
        assert_eq!(list.limited("x", 99, "a test"), list);

        let twins = ModelList::of(vec![
            ModelEntry::unlisted("llmman.provider/a/m"),
            ModelEntry::unlisted("llmman.provider/b/m"),
            ModelEntry::unlisted("llmman.provider/c/m"),
        ]);
        assert_eq!(twins.unique_keys(|e| e.id.clone()), ["m", "m-2", "m-3"]);
        assert_eq!(twins.unique_keys(ModelEntry::label), ["a/m", "b/m", "c/m"]);
    }

    /// A file of the name [`write_transient`] gives that is old goes; a
    /// young one and another's file stay.
    fn sweep_removes_only_old_files_of_its_own_name(dir: &Path) -> bool {
        let old = dir.join("llmman-models-1.json");
        let young = dir.join("llmman-models-2.json");
        let other = dir.join("user-models-3.json");
        for path in [&old, &young, &other] {
            std::fs::write(path, "x").unwrap();
        }
        let aged = std::time::SystemTime::now() - STALE_AFTER - std::time::Duration::from_secs(60);
        std::fs::File::options()
            .write(true)
            .open(&old)
            .unwrap()
            .set_modified(aged)
            .unwrap();
        std::fs::File::options()
            .write(true)
            .open(&other)
            .unwrap()
            .set_modified(aged)
            .unwrap();
        sweep(dir, "models", "json");
        !old.exists() && young.exists() && other.exists()
    }

    #[test]
    fn the_file_is_private_atomic_and_swept_when_stale() {
        let dir = super::super::test_temp_dir("models-file");
        let path = write_transient(&dir, "models", "json", "{\"models\":[]}\n").unwrap();
        assert_eq!(path.parent(), Some(dir.as_path()));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{\"models\":[]}\n");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        assert!(WRITTEN.lock().unwrap().iter().any(|(p, _)| *p == path));
        finish_file(&path);
        assert!(!path.exists());
        assert!(sweep_removes_only_old_files_of_its_own_name(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A launch that has run for more than a day still needs its file.
    #[test]
    fn an_old_file_a_running_launch_holds_is_not_swept() {
        let dir = super::super::test_temp_dir("models-held");
        let path = write_transient(&dir, "held", "json", "{}").unwrap();
        let aged = std::time::SystemTime::now() - STALE_AFTER - std::time::Duration::from_secs(60);
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(aged)
            .unwrap();
        sweep(&dir, "held", "json");
        assert!(path.exists(), "swept from under a running launch");
        finish_file(&path);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Only the last of the launches sharing something takes it away.
    #[test]
    fn the_last_launch_out_is_the_one_that_cleans_up() {
        let dir = super::super::test_temp_dir("models-shared");
        let lock = dir.join(".lock");
        let first = SharedUse::claim(&lock).unwrap();
        let second = SharedUse::claim(&lock).unwrap();
        assert!(first.release_if_last().is_none(), "another still uses it");
        let last = second.release_if_last();
        assert!(last.is_some(), "nobody else does");
        drop(last);
        // And it can be claimed again once the cleaner is done.
        assert!(SharedUse::claim(&lock).unwrap().release_if_last().is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
