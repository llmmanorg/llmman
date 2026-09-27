// The model picker (local store + hosted providers with a usable key)
// and the Models page. Nothing here knows a model name;
// everything comes from the daemon.

import * as api from "./api.js";
import * as settings from "./settings.js";
import { toast, formatBytes, $, icon, iconButton, debounce } from "./util.js";
import { render as renderMarkdown } from "./markdown.js";

const state = {
  local: [], // [{id, loaded, capabilities: [..] | null}]
  providers: [], // [{id, name, models: [{id, cost}] | null}]
  selected: settings.get("model") || "",
  loadedAt: 0,
  loading: null,
  listeners: new Set(),
};

export function selected() {
  return state.selected;
}

export function onChange(fn) {
  state.listeners.add(fn);
  return () => state.listeners.delete(fn);
}

function emit() {
  for (const fn of state.listeners) fn(state.selected);
}

export function select(ref) {
  state.selected = ref || "";
  settings.set({ model: state.selected });
  $("#model-btn-name").textContent = displayName(state.selected);
  emit();
}

/** A short label for a ref: `model · Provider` for hosted ones. */
export function displayName(ref) {
  if (!ref) return "Choose a model";
  const remote = api.splitRemoteRef(ref);
  if (remote) {
    const p = state.providers.find((p) => p.id === remote.provider);
    return `${remote.model} · ${p?.name || remote.provider}`;
  }
  return ref;
}

/** Whether `ref` is offered by the picker right now. */
export function isAvailable(ref) {
  const remote = api.splitRemoteRef(ref);
  if (remote) return state.providers.some((p) => p.id === remote.provider);
  return state.local.some((m) => m.id === ref && usable(m));
}

export function isLoaded(ref) {
  return state.local.some((m) => m.id === ref && m.loaded);
}

export function markLoaded(ref, loaded = true) {
  const m = state.local.find((m) => m.id === ref);
  if (m) m.loaded = loaded;
}

/** Reload local models and the provider list (not each provider's models). */
export async function refresh({ quiet = false } = {}) {
  if (state.loading) return state.loading;
  state.loading = (async () => {
    try {
      const [local, providers] = await Promise.all([
        api.listLocal(),
        api.listProviders().catch(() => []),
      ]);
      state.local = local.sort((a, b) => a.id.localeCompare(b.id));
      await annotateCapabilities(state.local);
      const usable = providers.filter((p) => p.key_usable);
      // Keep already-fetched model lists for providers still present.
      state.providers = usable
        .map((p) => ({
          id: p.id,
          name: p.name,
          count: p.models,
          models: state.providers.find((q) => q.id === p.id)?.models ?? null,
        }))
        .sort((a, b) => a.name.localeCompare(b.name));
      state.loadedAt = Date.now();
      // Nothing chosen yet: a loaded chat model, else the first chat model,
      // else a media model, so the composer is usable immediately.
      if (!state.selected || !knownRef(state.selected)) {
        const chat = state.local.filter(chattable);
        const pick = chat.find((m) => m.loaded) || chat[0] || state.local.find(generative);
        if (pick) select(pick.id);
        else if (state.selected && !knownRef(state.selected)) select("");
      } else {
        $("#model-btn-name").textContent = displayName(state.selected);
        emit(); // the capabilities may be new (a restored selection)
      }
    } catch (e) {
      if (!quiet) toast(`Could not list models: ${e.message}`, "error");
    } finally {
      state.loading = null;
    }
  })();
  return state.loading;
}

/**
 * `/api/show` per local model, for its capabilities: chat models and
 * media generation models are offered differently. Cached by reference; a
 * failure leaves `null`, which counts as chattable rather than hiding a
 * model on a hiccup.
 */
const capabilityCache = new Map();
async function annotateCapabilities(local) {
  await Promise.all(
    local.map(async (m) => {
      if (!capabilityCache.has(m.id)) {
        try {
          const info = await api.showModel(m.id);
          capabilityCache.set(m.id, Array.isArray(info.capabilities) ? info.capabilities : null);
        } catch {
          capabilityCache.set(m.id, null);
        }
      }
      m.capabilities = capabilityCache.get(m.id);
    }),
  );
}

/** Whether a local model can take a chat turn. */
export function chattable(m) {
  return !m.capabilities || m.capabilities.includes("completion");
}

/** The media a local model generates, in the composer's order; `[]` for a chat model. */
function mediaOf(m) {
  return ["image", "video", "audio"].filter((k) => m.capabilities?.includes(k));
}

/** Whether a local model generates media rather than text. */
function generative(m) {
  return !chattable(m) && mediaOf(m).length > 0;
}

/** Whether the picker offers a local model at all. */
function usable(m) {
  return chattable(m) || generative(m);
}

/** The media kinds `ref` generates; `[]` for a chat model, a hosted one or an unknown ref. */
export function mediaCapabilities(ref) {
  const m = state.local.find((m) => m.id === ref);
  return m && generative(m) ? mediaOf(m) : [];
}

function knownRef(ref) {
  if (api.splitRemoteRef(ref)) return true; // provider models are checked lazily
  return state.local.some((m) => m.id === ref && usable(m));
}

async function ensureProviderModels() {
  await Promise.all(
    state.providers
      .filter((p) => p.models === null)
      .map(async (p) => {
        try {
          p.models = await api.providerModels(p.id);
        } catch {
          p.models = [];
        }
      }),
  );
}

// ---- Picker menu --------------------------------------------------------

let menuOpen = false;

export function initPicker() {
  const btn = $("#model-btn");
  const menu = $("#model-menu");
  const search = $("#model-search");

  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    if (menuOpen) closeMenu();
    else openMenu();
  });
  search.addEventListener("input", renderMenu);
  search.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeMenu();
    if (e.key === "Enter") {
      const first = menu.querySelector(".menu-item");
      if (first) first.click();
    }
  });
  document.addEventListener("click", (e) => {
    if (menuOpen && !menu.contains(e.target) && e.target !== btn) closeMenu();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && menuOpen) closeMenu();
  });
  $("#models-refresh").addEventListener("click", async () => {
    state.providers.forEach((p) => (p.models = null));
    await refresh();
    await ensureProviderModels();
    renderMenu();
  });
  $("#pull-open").addEventListener("click", () => {
    closeMenu();
    location.hash = "#/models";
  });
  window.addEventListener("resize", () => menuOpen && positionMenu());
}

async function openMenu() {
  const menu = $("#model-menu");
  const btn = $("#model-btn");
  menuOpen = true;
  btn.setAttribute("aria-expanded", "true");
  menu.classList.remove("hidden");
  $("#model-search").value = "";
  renderMenu();
  positionMenu();
  $("#model-search").focus();
  // Refresh in the background if the list is stale; providers' models load lazily.
  if (Date.now() - state.loadedAt > 5000) await refresh({ quiet: true });
  await ensureProviderModels();
  if (menuOpen) renderMenu();
}

function closeMenu() {
  menuOpen = false;
  $("#model-menu").classList.add("hidden");
  $("#model-btn").setAttribute("aria-expanded", "false");
}

function positionMenu() {
  const menu = $("#model-menu");
  const view = $("#view-chat").getBoundingClientRect();
  const rect = $("#model-btn").getBoundingClientRect();
  menu.style.bottom = `${view.bottom - rect.top + 8}px`;
  menu.style.right = `${Math.max(8, view.right - rect.right)}px`;
  menu.style.left = "auto";
  menu.style.top = "auto";
  // Never taller than the room above the button.
  menu.style.maxHeight = `${Math.max(160, rect.top - view.top - 16)}px`;
}

function renderMenu() {
  const body = $("#model-menu-body");
  const q = $("#model-search").value.trim().toLowerCase();
  const match = (s) => !q || s.toLowerCase().includes(q);
  body.replaceChildren();

  const chatModels = state.local.filter(chattable);
  const mediaModels = state.local.filter(generative);
  const local = chatModels.filter((m) => match(m.id));
  body.appendChild(section("i-chip", "Local"));
  if (!local.length) {
    body.appendChild(
      emptyRow(
        chatModels.length
          ? "No local model matches."
          : mediaModels.length
            ? "No local chat models — the store has only media generation models."
            : "No local models yet — pull one below, or `llmman pull <ref>`.",
      ),
    );
  }
  for (const m of local) {
    body.appendChild(
      menuItem({
        ref: m.id,
        name: m.id,
        sub: m.loaded ? "loaded" : "",
        dot: m.loaded ? "ok" : "",
        eject: m.loaded,
      }),
    );
  }

  const media = mediaModels.filter((m) => match(m.id) || mediaOf(m).some(match));
  if (media.length) {
    body.appendChild(section("i-image", "Generate"));
    for (const m of media) {
      body.appendChild(
        menuItem({
          ref: m.id,
          name: m.id,
          sub: [mediaOf(m).join(" · "), m.loaded ? "loaded" : ""].filter(Boolean).join(" · "),
          dot: m.loaded ? "ok" : "",
          eject: m.loaded,
        }),
      );
    }
  }

  for (const p of state.providers) {
    const models = p.models;
    const shown = models ? models.filter((m) => match(m.id) || match(p.name)) : [];
    if (models && !shown.length && q) continue;
    body.appendChild(section("i-cloud", p.name));
    if (models === null) {
      body.appendChild(emptyRow("Loading…"));
      continue;
    }
    if (!shown.length) {
      body.appendChild(emptyRow("No models listed."));
      continue;
    }
    for (const m of shown) {
      body.appendChild(
        menuItem({
          ref: api.remoteRef(p.id, m.id),
          name: m.id,
          sub: m.cost ? `$${trimNumber(m.cost.input)} / $${trimNumber(m.cost.output)} per M` : "",
        }),
      );
    }
  }

  if (!state.providers.length && !q) {
    const hint = emptyRow(
      "Hosted providers appear here when llmman serve has a key for them (e.g. OPENAI_API_KEY or [providers] in llmman.conf) and is bound to loopback.",
    );
    body.appendChild(hint);
  }
}

function trimNumber(n) {
  if (n == null) return "?";
  return Number(n).toString();
}

function section(iconId, label) {
  const d = document.createElement("div");
  d.className = "menu-section";
  d.appendChild(icon(iconId));
  d.appendChild(document.createTextNode(label));
  return d;
}

function emptyRow(text) {
  const d = document.createElement("div");
  d.className = "menu-empty";
  d.textContent = text;
  return d;
}

/** A row: the select button, plus an unload button beside it when loaded. */
function menuItem({ ref, name, sub, dot, eject }) {
  const row = document.createElement("div");
  row.className = "menu-row";
  const b = document.createElement("button");
  b.type = "button";
  b.className = "menu-item" + (ref === state.selected ? " selected" : "");
  b.setAttribute("role", "menuitem");
  b.title = ref;
  if (dot !== undefined) {
    const d = document.createElement("span");
    d.className = "dot status-dot " + (dot || "");
    b.appendChild(d);
  }
  const n = document.createElement("span");
  n.className = "menu-item-name";
  n.textContent = name;
  b.appendChild(n);
  if (sub) {
    const s = document.createElement("span");
    s.className = "menu-item-sub";
    s.textContent = sub;
    b.appendChild(s);
  }
  b.addEventListener("click", () => {
    select(ref);
    closeMenu();
  });
  row.appendChild(b);
  if (eject) {
    row.appendChild(
      iconButton("i-eject", `Unload ${name}`, () => unload(ref).then(renderMenu), "icon-btn eject"),
    );
  }
  return row;
}

async function unload(ref) {
  try {
    await api.unloadModel(ref);
    markLoaded(ref, false);
    toast(`Unloaded ${ref}`);
  } catch (err) {
    toast(err.message, "error");
  }
}

// ---- Search helpers ---------------------------------------------------

/**
 * A pasted reference (a scheme, or a registry host before the first `/`)
 * is pulled as typed; anything else is a search.
 */
function isReference(text) {
  return text.includes("://") || /^[\w-]+(\.[\w-]+)+(:\d+)?\//.test(text);
}

/** Whether any tag of repository `name` is already in the local store. */
function isPulled(name) {
  const repo = name.toLowerCase();
  return state.local.some((m) => {
    const id = m.id.toLowerCase();
    return id === repo || id.startsWith(`${repo}:`);
  });
}

function formatCount(n) {
  return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(n);
}

function relativeTime(iso) {
  const days = Math.round((Date.parse(iso) - Date.now()) / 86_400_000);
  if (!Number.isFinite(days)) return "";
  const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  if (Math.abs(days) < 31) return rtf.format(days, "day");
  if (Math.abs(days) < 365) return rtf.format(Math.round(days / 30), "month");
  return rtf.format(Math.round(days / 365), "year");
}

// ---- Models page (#/models) -------------------------------------------
//
// A list on the left (this machine's models, or search results) and a
// card on the right for the selected one: its tags with sizes and whether
// each fits this machine, and Pull / Use / Delete.

const mb = {
  query: "",
  filter: "all",
  hits: null, // search results, or null for "no search yet"
  popular: null, // what to show before a search: rows, "loading" or an Error
  selected: null, // a repo name, e.g. hf.co/unsloth/Qwen3.5-0.8B-GGUF
  cards: new Map(), // repo → ModelCard | Error
  memory: 0, // bytes of model memory on this machine, 0 if unknown
  choice: new Map(), // repo → the variant name picked on its card
  details: new Map(), // local id (lowercase) → its `/api/tags` row
  searchAbort: null,
  pull: null, // {ref, abort, done, total, status}
};

/** Search rows per registry: a page's worth without scrolling forever. */
const SEARCH_LIMIT = 20;

export function initModelsPage() {
  const query = $("#mb-query");
  query.addEventListener("input", () => mbSearchSoon(query.value.trim()));
  query.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    const text = query.value.trim();
    if (isReference(text)) mbPull(text);
    else mbSearch(text);
  });
  for (const tab of document.querySelectorAll("#mb-tabs [data-filter]")) {
    tab.addEventListener("click", () => {
      mb.filter = tab.dataset.filter;
      for (const t of document.querySelectorAll("#mb-tabs [data-filter]")) {
        t.classList.toggle("active", t === tab);
        t.setAttribute("aria-selected", String(t === tab));
      }
      renderBrowserList();
    });
  }
}

export async function showModelsPage() {
  $("#mb").classList.remove("show-card");
  api
    .nodeInfo()
    .then((n) => {
      mb.memory = n.memory || 0;
      renderCard();
    })
    .catch(() => {});
  loadPopular();
  await reloadLocal();
  $("#mb-query").focus();
}

/** Once per page load: the registries' most popular models change slowly. */
async function loadPopular() {
  if (mb.popular && !(mb.popular instanceof Error)) return;
  mb.popular = "loading";
  try {
    mb.popular = await api.popular({ limit: SEARCH_LIMIT });
  } catch (e) {
    mb.popular = e;
  }
  renderBrowserList();
}

/** The store changed (or may have): re-read it and redraw. */
async function reloadLocal() {
  await refresh({ quiet: true });
  const tags = await api.listLocalDetailed().catch(() => []);
  mb.details = new Map(tags.map((t) => [t.name.toLowerCase(), t]));
  renderBrowserList();
  renderCard();
}

/** Whether the store holds variant `v`: exactly, or as the `:latest` a tagless pull is stored under. */
function holds(v) {
  const want = v.name.toLowerCase();
  const latest = `${repoOf(want)}:latest`;
  return state.local.some((m) => {
    const id = m.id.toLowerCase();
    return id === want || (v.default && id === latest);
  });
}

/** `docker.io/ai/qwen3.5:0.8b` → `docker.io/ai/qwen3.5`. */
function repoOf(ref) {
  const slash = ref.lastIndexOf("/");
  const colon = ref.lastIndexOf(":");
  return colon > slash ? ref.slice(0, colon) : ref;
}

function registryOf(name) {
  if (name.startsWith("hf.co/")) return "hf";
  if (name.startsWith("docker.io/")) return "docker";
  return "other";
}

const mbSearchSoon = debounce((query) => {
  if (query.length < 2 || isReference(query)) {
    mb.searchAbort?.abort();
    mb.query = "";
    mb.hits = null;
    return renderBrowserList();
  }
  mbSearch(query);
}, 300);

async function mbSearch(query) {
  if (!query) return;
  mb.searchAbort?.abort();
  const abort = new AbortController();
  mb.searchAbort = abort;
  mb.query = query;
  mb.hits = "loading";
  renderBrowserList();
  try {
    mb.hits = await api.search(query, { limit: SEARCH_LIMIT, signal: abort.signal });
  } catch (e) {
    if (e.name === "AbortError") return;
    mb.hits = e;
  }
  if (!abort.signal.aborted) renderBrowserList();
}

/** Local models, one row per repo, with the tags this machine holds. */
function localRepos() {
  const byRepo = new Map();
  for (const m of state.local) {
    const repo = repoOf(m.id);
    if (!byRepo.has(repo)) byRepo.set(repo, { name: repo, local: [] });
    byRepo.get(repo).local.push(m);
  }
  return [...byRepo.values()];
}

function renderBrowserList() {
  const list = $("#mb-list");
  list.replaceChildren();
  const filter = mb.filter;
  const shown = (rows) => rows.filter((h) => filter === "all" || registryOf(h.name) === filter);
  if (filter === "local") {
    const repos = localRepos();
    list.appendChild(listHeading("On this machine"));
    if (!repos.length) list.appendChild(emptyRow("Nothing pulled yet. Search above to find a model."));
    for (const r of repos) list.appendChild(browserRow({ name: r.name }));
    return;
  }
  if (mb.hits === null) {
    // No search yet: the popular models, by registry.
    if (mb.popular === "loading" || mb.popular === null) return list.appendChild(emptyRow("Loading popular models…"));
    if (mb.popular instanceof Error) return list.appendChild(emptyRow(mb.popular.message));
    const rows = shown(mb.popular);
    for (const [registry, heading] of [
      ["docker", "Featured on Docker Hub"],
      ["hf", "Popular GGUF on Hugging Face"],
    ]) {
      const group = rows.filter((h) => registryOf(h.name) === registry);
      if (!group.length) continue;
      list.appendChild(listHeading(heading));
      for (const hit of group) list.appendChild(browserRow(hit));
    }
    return;
  }
  if (mb.hits === "loading") return list.appendChild(emptyRow("Searching…"));
  if (mb.hits instanceof Error) return list.appendChild(emptyRow(mb.hits.message));
  const hits = shown(mb.hits);
  list.appendChild(listHeading(`Results for “${mb.query}”`));
  if (!hits.length) list.appendChild(emptyRow("No models found."));
  for (const hit of hits) list.appendChild(browserRow(hit));
}

function listHeading(text) {
  const h = document.createElement("div");
  h.className = "browser-heading";
  h.textContent = text;
  return h;
}

/** A 28px avatar: the owner's picture, or its initials when it has none. */
function avatar(name, size = 28) {
  const box = document.createElement("span");
  box.className = "avatar";
  box.style.setProperty("--size", `${size}px`);
  const owner = name.split("/").at(-2) || "?";
  box.textContent = owner.slice(0, 2).toUpperCase();
  // A stable colour per owner, for the initials.
  box.style.setProperty("--hue", String([...owner].reduce((h, c) => (h * 31 + c.charCodeAt(0)) % 360, 7)));
  const img = document.createElement("img");
  img.alt = "";
  img.loading = "lazy";
  img.src = `llmman/search/avatar?name=${encodeURIComponent(repoOf(name))}`;
  img.addEventListener("load", () => box.classList.add("has-img"));
  img.addEventListener("error", () => img.remove());
  box.appendChild(img);
  return box;
}

function registryBadge(name) {
  const r = registryOf(name);
  const b = document.createElement("span");
  b.className = `registry ${r}`;
  b.textContent = r === "hf" ? "HF" : r === "docker" ? "Hub" : "";
  return b;
}

function browserRow(hit) {
  const name = hit.name;
  const row = document.createElement("button");
  row.type = "button";
  row.className = "browser-row" + (mb.selected === name ? " selected" : "");
  row.setAttribute("role", "option");
  row.setAttribute("aria-selected", String(mb.selected === name));
  row.appendChild(avatar(name));
  const text = document.createElement("span");
  text.className = "browser-row-text";
  const title = document.createElement("span");
  title.className = "browser-row-title";
  const label = document.createElement("span");
  label.className = "label";
  label.textContent = name.split("/").at(-1);
  title.appendChild(label);
  if (isPulled(name)) {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = "pulled";
    title.appendChild(tag);
  }
  text.appendChild(title);
  const sub = document.createElement("span");
  sub.className = "browser-row-sub";
  sub.appendChild(registryBadge(name));
  sub.append(name.split("/").at(-2) || "");
  if (hit.pulls != null) sub.append(stat("i-download", formatCount(hit.pulls)));
  if (hit.likes != null) sub.append(stat("i-heart", formatCount(hit.likes)));
  text.appendChild(sub);
  row.appendChild(text);
  row.addEventListener("click", () => {
    mb.selected = name;
    $("#mb").classList.add("show-card");
    renderBrowserList();
    renderCard();
  });
  return row;
}

function stat(iconId, text) {
  const s = document.createElement("span");
  s.className = "stat";
  s.append(icon(iconId), text);
  return s;
}

// ---- The card ---------------------------------------------------------

async function renderCard() {
  const card = $("#mb-card");
  const name = mb.selected;
  if (!name) {
    card.replaceChildren(placeholder());
    return;
  }
  if (registryOf(name) === "other") return card.replaceChildren(cardView(name, null));
  let info = mb.cards.get(name);
  if (!info) {
    card.replaceChildren(cardView(name, "loading"));
    try {
      info = await api.modelCard(name);
      mb.cards.set(name, info);
    } catch (e) {
      info = e; // not kept: selecting the row again retries
    }
    if (mb.selected !== name) return;
  }
  card.replaceChildren(cardView(name, info));
}

function placeholder() {
  const p = document.createElement("div");
  p.className = "card-empty";
  p.append(icon("i-cube"));
  const h = document.createElement("h2");
  h.textContent = "Select a model";
  const t = document.createElement("p");
  t.className = "muted";
  t.textContent = "Pick one to see its tags and sizes, whether it fits this machine, and pull it.";
  p.append(h, t);
  return p;
}

function cardView(name, info) {
  const wrap = document.createElement("div");
  wrap.className = "card";

  const back = iconButton("i-back", "Back to the list", () => $("#mb").classList.remove("show-card"));
  back.classList.add("card-back");
  wrap.appendChild(back);

  const head = document.createElement("div");
  head.className = "card-head";
  head.appendChild(avatar(name, 48));
  const titles = document.createElement("div");
  const h = document.createElement("h2");
  if (info && !(info instanceof Error) && info !== "loading") {
    const a = document.createElement("a");
    a.href = info.page;
    a.target = "_blank";
    a.rel = "noopener";
    a.textContent = name.split("/").slice(-2).join("/");
    h.appendChild(a);
  } else h.textContent = name.split("/").slice(-2).join("/");
  titles.appendChild(h);
  const sub = document.createElement("div");
  sub.className = "card-stats";
  sub.appendChild(registryBadge(name));
  if (info && typeof info === "object" && !(info instanceof Error)) {
    if (info.pulls != null) sub.append(stat("i-download", `${formatCount(info.pulls)} ${info.pulls === 1 ? "pull" : "pulls"}`));
    if (info.likes != null) sub.append(stat("i-heart", formatCount(info.likes)));
    if (info.updated) sub.append(pill(`updated ${relativeTime(info.updated)}`));
    if (info.license) sub.append(pill(info.license));
    if (info.task) sub.append(pill(info.task));
    if (info.gated) sub.append(pill("gated: needs an HF token"));
  }
  titles.appendChild(sub);
  head.appendChild(titles);
  wrap.appendChild(head);

  // What this machine already has of it.
  const local = state.local.filter((m) => repoOf(m.id).toLowerCase() === name.toLowerCase());
  if (local.length) {
    wrap.appendChild(sectionTitle("On this machine"));
    const box = document.createElement("div");
    box.className = "card-local";
    for (const m of local) box.appendChild(localLine(m));
    wrap.appendChild(box);
  }

  // A pull of this repo, from its card or typed as a reference.
  const pulling = mb.pull && repoOf(mb.pull.ref).toLowerCase() === name.toLowerCase();
  if (pulling) wrap.appendChild(pullProgress());

  if (info === "loading") {
    wrap.appendChild(emptyRow("Loading…"));
    return wrap;
  }
  if (info instanceof Error) {
    wrap.appendChild(emptyRow(info.message));
    return wrap;
  }
  if (!info) return wrap;

  // Tags to pull, with sizes and fit.
  if (info.variants.length) {
    wrap.appendChild(sectionTitle(info.variants.length > 1 ? `Pick a version · ${info.variants.length}` : "Download"));
    const list = document.createElement("div");
    list.className = "variants";
    const chosen =
      info.variants.find((v) => v.name === mb.choice.get(name)) ||
      info.variants.find((v) => v.default) ||
      info.variants[0];
    for (const v of info.variants) {
      const row = document.createElement("label");
      row.className = "variant" + (v === chosen ? " chosen" : "");
      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = "variant";
      radio.checked = v === chosen;
      radio.addEventListener("change", () => {
        mb.choice.set(name, v.name);
        renderCard();
      });
      row.appendChild(radio);
      const tag = document.createElement("span");
      tag.className = "variant-tag";
      tag.textContent = v.tag;
      row.appendChild(tag);
      if (v.default) row.appendChild(pill("default"));
      if (holds(v)) {
        const t = document.createElement("span");
        t.className = "tag";
        t.textContent = "pulled";
        row.appendChild(t);
      }
      const size = document.createElement("span");
      size.className = "variant-size";
      size.textContent = v.size ? formatBytes(v.size) : "";
      row.appendChild(size);
      row.appendChild(fitDot(v.size));
      list.appendChild(row);
    }
    wrap.appendChild(list);
    // The default is rarely first (rows go by size); show it without moving the page.
    requestAnimationFrame(() => {
      const row = list.querySelector(".variant.chosen");
      if (row) list.scrollTop = row.offsetTop - (list.clientHeight - row.offsetHeight) / 2;
    });

    wrap.appendChild(fitLine(chosen));

    const actions = document.createElement("div");
    actions.className = "card-actions";
    if (!pulling) {
      const have = holds(chosen);
      const go = document.createElement("button");
      go.className = "btn primary";
      if (have) go.append(icon("i-check"), ` ${chosen.tag} is on this machine`);
      else go.append(icon("i-download"), ` Pull ${chosen.tag}${chosen.size ? ` · ${formatBytes(chosen.size)}` : ""}`);
      go.disabled = have || !!mb.pull;
      go.title = mb.pull ? `Pulling ${mb.pull.ref}` : chosen.name;
      go.addEventListener("click", () => mbPull(chosen.name));
      actions.appendChild(go);
    }
    wrap.appendChild(actions);
  }

  const tags = (info.tags || []).filter((t) => !t.startsWith("base_model:") && !t.startsWith("dataset:")).slice(0, 14);
  if (tags.length) {
    wrap.appendChild(sectionTitle("Tags"));
    const box = document.createElement("div");
    box.className = "chips";
    for (const t of tags) box.appendChild(pill(t));
    wrap.appendChild(box);
  }

  if (info.readme) {
    wrap.appendChild(sectionTitle("README"));
    const readme = document.createElement("div");
    readme.className = "readme content";
    readme.appendChild(renderMarkdown(readmeMarkdown(info.readme, registryOf(name) === "hf" ? info.page : "")));
    wrap.appendChild(readme);
  }
  return wrap;
}

/**
 * A registry README as Markdown for the page's renderer: front matter and
 * comments dropped, and outside code blocks the HTML model cards are full
 * of reduced to what it says (links, headings, line breaks, text). Images
 * go: most are badges, and the page fetches nothing a README asks for.
 * The renderer still treats all of it as untrusted text. Relative links
 * point into the repo at `page`, when given.
 */
function readmeMarkdown(text, page) {
  const body = text.replace(/^\uFEFF?---\r?\n[\s\S]*?\r?\n---\r?\n/, "");
  const entities = { amp: "&", lt: "<", gt: ">", quot: '"', "#39": "'", nbsp: " " };
  const flat = (t) => t.replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();
  // Split on fenced code blocks; odd pieces are code and stay as written.
  const pieces = body.split(/(^```[\s\S]*?^```[^\n]*$)/m);
  const out = pieces.map((piece, i) => {
    if (i % 2) return piece;
    let s = piece.replace(/<!--[\s\S]*?-->/g, "");
    s = s.replace(/<img\b[^>]*>/gi, "");
    s = s.replace(/<br\s*\/?>/gi, "\n");
    s = s.replace(/<h([1-6])\b[^>]*>([\s\S]*?)<\/h\1>/gi, (_, n, t) => `\n\n${"#".repeat(+n)} ${flat(t)}\n\n`);
    s = s.replace(/<a\b[^>]*?href\s*=\s*["']([^"']+)["'][^>]*>([\s\S]*?)<\/a>/gi, (_, href, t) =>
      flat(t) ? `[${flat(t)}](${href})` : "",
    );
    s = s.replace(/<\/?(p|div|center|table|thead|tbody|tr|details|summary|ul|ol|li)\b[^>]*>/gi, "\n");
    s = s.replace(/<\/?[a-z][^>\n]*>/gi, "");
    s = s.replace(/&(amp|lt|gt|quot|#39|nbsp);/g, (_, e) => entities[e]);
    if (page) {
      s = s.replace(/\]\((?!https?:|mailto:|#)([^)\s]+)\)/g, (_, path) => `](${page}/blob/main/${path.replace(/^\.?\//, "")})`);
    }
    return s.replace(/\n{3,}/g, "\n\n");
  });
  return out.join("").trim();
}

function sectionTitle(text) {
  const h = document.createElement("h3");
  h.textContent = text;
  return h;
}

function pill(text) {
  const p = document.createElement("span");
  p.className = "pill";
  p.textContent = text;
  return p;
}

/** ok / tight / too big, against this machine's model memory. */
function fitOf(size) {
  if (!size || !mb.memory) return null;
  const share = size / mb.memory;
  return share <= 0.6 ? "ok" : share <= 0.9 ? "warn" : "bad";
}

function fitDot(size) {
  const d = document.createElement("span");
  const fit = fitOf(size);
  d.className = "dot " + (fit || "");
  d.title = fit ? { ok: "Fits this machine", warn: "Tight on this machine", bad: "Too big for this machine" }[fit] : "";
  return d;
}

function fitLine(v) {
  const line = document.createElement("div");
  line.className = "fit-line";
  const fit = fitOf(v.size);
  if (!fit) {
    line.textContent = v.size ? "" : "Size unknown until pulled.";
    return line;
  }
  line.appendChild(fitDot(v.size));
  const words = { ok: "Fits this machine", warn: "Tight on this machine", bad: "Too big for this machine" }[fit];
  line.append(` ${words}: ${formatBytes(v.size)} of weights, ${formatBytes(mb.memory)} of model memory here (context and runtime need more).`);
  return line;
}

function localLine(m) {
  const row = document.createElement("div");
  row.className = "local-line";
  const dot = document.createElement("span");
  dot.className = "dot " + (m.loaded ? "ok" : "");
  dot.title = m.loaded ? "loaded" : "not loaded";
  row.appendChild(dot);
  const name = document.createElement("span");
  name.className = "name";
  name.textContent = m.id.split(":").length > 1 && m.id.lastIndexOf(":") > m.id.lastIndexOf("/") ? m.id.slice(m.id.lastIndexOf(":") + 1) : m.id;
  name.title = m.id;
  row.appendChild(name);
  const d = mb.details.get(m.id.toLowerCase());
  if (d) {
    const meta = document.createElement("span");
    meta.className = "meta";
    const x = d.details || {};
    meta.textContent = [x.format, x.parameter_size, x.quantization_level, d.size && formatBytes(d.size)]
      .filter(Boolean)
      .join(" · ");
    row.appendChild(meta);
  }
  const acts = document.createElement("span");
  acts.className = "actions";
  if (usable(m)) {
    const use = document.createElement("button");
    use.className = "btn";
    use.textContent = generative(m) ? "Generate" : "Chat";
    use.addEventListener("click", () => {
      select(m.id);
      location.hash = "#/";
    });
    acts.appendChild(use);
  }
  if (m.loaded) acts.appendChild(iconButton("i-eject", "Unload", () => unload(m.id).then(reloadLocal)));
  acts.appendChild(
    iconButton("i-trash", "Delete from this machine", async () => {
      if (!confirm(`Delete ${m.id} from the local store?`)) return;
      try {
        await api.deleteModel(m.id);
        toast(`Deleted ${m.id}`);
      } catch (e) {
        toast(e.message, "error");
      }
      await reloadLocal();
    }),
  );
  row.appendChild(acts);
  return row;
}

// ---- Pull, from the card ---------------------------------------------

function pullProgress() {
  const box = document.createElement("div");
  box.className = "pull-progress card-pull";
  const p = mb.pull;
  const status = document.createElement("div");
  status.className = "pull-status";
  status.textContent = p.status || `Pulling ${p.ref}`;
  const bar = document.createElement("div");
  bar.className = "bar";
  const fill = document.createElement("div");
  fill.className = "bar-fill" + (p.total ? "" : " indeterminate");
  if (p.total) fill.style.width = `${Math.min(100, (100 * p.done) / p.total).toFixed(1)}%`;
  bar.appendChild(fill);
  const detail = document.createElement("div");
  detail.className = "pull-detail muted";
  detail.textContent = p.total ? `${formatBytes(p.done)} of ${formatBytes(p.total)}` : "";
  const cancel = document.createElement("button");
  cancel.className = "btn";
  cancel.textContent = "Cancel";
  cancel.addEventListener("click", () => p.abort.abort());
  box.append(status, bar, detail, cancel);
  return box;
}

async function mbPull(ref) {
  if (mb.pull) return toast(`Already pulling ${mb.pull.ref}`, "error");
  const abort = new AbortController();
  mb.pull = { ref, abort, done: 0, total: 0, status: "" };
  mb.selected = repoOf(ref);
  $("#mb").classList.add("show-card");
  renderBrowserList();
  renderCard();
  const layers = new Map();
  let frame = 0;
  try {
    await api.pull(
      ref,
      (ev) => {
        if (ev.status) mb.pull.status = ev.status;
        if (ev.total) {
          layers.set(ev.digest || ev.status || "_", { total: ev.total, completed: ev.completed || 0 });
          mb.pull.total = mb.pull.done = 0;
          for (const l of layers.values()) (mb.pull.total += l.total), (mb.pull.done += l.completed);
        }
        cancelAnimationFrame(frame);
        frame = requestAnimationFrame(() => {
          const box = $("#mb-card .card-pull");
          if (box) box.replaceWith(pullProgress());
        });
      },
      abort.signal,
    );
    toast(`Pulled ${ref}`);
  } catch (e) {
    if (e.name !== "AbortError") toast(`Pull failed: ${e.message}`, "error");
  } finally {
    cancelAnimationFrame(frame);
    mb.pull = null;
    await reloadLocal();
  }
}
