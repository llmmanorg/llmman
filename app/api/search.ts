// Model search on the daemon: `/llmman/search`, `/llmman/search/popular`,
// `/llmman/search/model` and `/llmman/search/avatar`. Pulls, `/llmman/node` and the local store are
// `ApiClient` methods (api/client.ts).

import { Platform } from "react-native";

import type { ApiClient, RequestOptions } from "./client";

/** A `/llmman/search` or `/llmman/search/popular` row; `name` pulls as is. */
export interface SearchHit {
  name: string;
  pulls?: number | null;
  likes?: number | null;
  updated?: string | null;
}

/** One tag `pull` can take for a repo, with its size in bytes (0 when unknown). */
export interface Variant {
  name: string;
  tag: string;
  size: number;
  /** The one a tagless pull takes. */
  default: boolean;
}

/** `GET /llmman/search/model`: a search row's tags with sizes, and its repo's facts. */
export interface ModelCard {
  name: string;
  /** The repo's page on its registry. */
  page: string;
  pulls?: number | null;
  likes?: number | null;
  updated?: string | null;
  license?: string | null;
  task?: string | null;
  gated?: boolean;
  tags?: string[] | null;
  variants: Variant[];
  /** The first 64 KiB of the README (Hugging Face) or overview (Docker Hub). */
  readme?: string | null;
}

/** `GET /llmman/node`: `memory` is bytes of model memory, absent when unknown. */
export interface NodeInfo {
  memory?: number;
  loaded?: Record<string, unknown>;
  stored?: Record<string, number>;
}

/** Rows per registry: a screen's worth without scrolling forever. */
export const SEARCH_LIMIT = 20;

/** `GET /llmman/search`: models on Docker Hub and Hugging Face matching `query`, as `llmman search` lists them. */
export async function search(
  client: ApiClient,
  query: string,
  { limit = SEARCH_LIMIT, ...opts }: RequestOptions & { limit?: number } = {},
): Promise<SearchHit[]> {
  const params = `q=${encodeURIComponent(query)}&limit=${limit}`;
  const body = await client.getJson<{ models?: SearchHit[] }>(`llmman/search?${params}`, opts);
  return body.models || [];
}

/** `GET /llmman/search/popular`: what to show before a search, in `search`'s row shape. */
export async function popular(
  client: ApiClient,
  { limit = SEARCH_LIMIT, ...opts }: RequestOptions & { limit?: number } = {},
): Promise<SearchHit[]> {
  const body = await client.getJson<{ models?: SearchHit[] }>(`llmman/search/popular?limit=${limit}`, opts);
  return body.models || [];
}

/** `GET /llmman/search/model`. */
export async function modelCard(client: ApiClient, name: string, opts?: RequestOptions): Promise<ModelCard> {
  const card = await client.getJson<ModelCard>(`llmman/search/model?name=${encodeURIComponent(name)}`, opts);
  return { ...card, variants: card.variants || [] };
}

/**
 * The repo owner's picture on Hugging Face or Gravatar, or null when it has
 * none. `/llmman/search/avatar` answers with a redirect; the app reads its
 * `location` instead of letting an image loader follow it, so the key the
 * daemon needs never goes along to the picture's host. Only an absolute
 * https URL is used.
 *
 * A browser hides a manual redirect's location, so on the web the image
 * loads from the daemon's route itself: an `<img>` follows the redirect and
 * sends no Authorization header. A 404 (no picture) fails the image, which
 * leaves the monogram.
 */
export async function avatarUrl(
  client: ApiClient,
  repo: string,
  opts?: RequestOptions,
): Promise<string | null> {
  const path = `llmman/search/avatar?name=${encodeURIComponent(repo)}`;
  if (Platform.OS === "web") return client.urlFor(path);
  const r = await client.request(path, { redirect: "manual" }, opts);
  if (r.status < 300 || r.status >= 400) return null;
  const location = r.headers.get("location");
  return location && /^https:\/\/[^/]/i.test(location) ? location : null;
}
