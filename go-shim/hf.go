

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	neturl "net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	modelspec "github.com/modelpack/model-spec/specs-go/v1"
	digest "github.com/opencontainers/go-digest"
	specs "github.com/opencontainers/image-spec/specs-go"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/vbauerster/mpb/v8"
)

// hfGGUFMediaType is the standard Docker AI media type for GGUF model layers.
const hfGGUFMediaType = "application/vnd.docker.ai.gguf.v3"

// ---------------------------------------------------------------------------
// Registry detection
// ---------------------------------------------------------------------------

// isKnownOCIHost returns true for registries that are definitely OCI-compliant,
// skipping the network probe entirely.
func isKnownOCIHost(host string) bool {
	switch host {
	case "ghcr.io", "docker.io", "index.docker.io", "registry-1.docker.io",
		"quay.io", "gcr.io", "mcr.microsoft.com", "public.ecr.aws":
		return true
	}
	return false
}

// isKnownHFHost returns true for known HuggingFace-compatible hosts.
func isKnownHFHost(host string) bool {
	switch host {
	case "hf.co", "huggingface.co", "modelscope.cn":
		return true
	}
	return false
}

// isOCIRegistry probes the OCI Distribution /v2/ endpoint and returns true if
// the server advertises itself as an OCI registry via the standard header.
func isOCIRegistry(ctx context.Context, client *http.Client, host string) bool {
	probeCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(probeCtx, "GET", "https://"+host+"/v2/", nil)
	if err != nil {
		return false
	}
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	resp.Body.Close()
	// OCI registries advertise registry/2.0 on both 200 and 401 responses.
	return resp.Header.Get("Docker-Distribution-Api-Version") != ""
}

// isOCIHost reports whether host should be treated as an OCI Distribution
// registry (true) or a HuggingFace-compatible host (false): known-host
// shortcuts first, then a live /v2/ probe as the fallback for anything else.
// This is the single decision table `llmman pull`'s docker and podman
// backends (backend_docker.go, backend_podman.go) and `llmman transfer`'s
// source classification (transfer_common.go) all need to agree on.
func isOCIHost(ctx context.Context, host string) bool {
	if isKnownHFHost(host) {
		return false
	}
	if isKnownOCIHost(host) {
		return true
	}
	probeClient := &http.Client{Timeout: 5 * time.Second}
	return isOCIRegistry(ctx, probeClient, host)
}

// classifyPullRef runs the URI-scheme dispatch and OCI-vs-HuggingFace host
// classification shared by both backends' pullToLayout (backend_docker.go,
// backend_podman.go). If handled is true, dispatchPull has already fully
// processed ref (via one of hf://, ms://, ngc://, s3://, gs://, or a local
// path) and the caller should return dispatchErr immediately without doing
// anything else. Otherwise normalizedRef is ref with a ":latest" tag
// defaulted in, and isOCI reports whether normalizedRef's host should be
// pulled via the OCI registry protocol (true) or the shared HF path (false).
func classifyPullRef(ctx context.Context, ref, layoutDir string) (normalizedRef string, isOCI, handled bool, dispatchErr error) {
	if handled, err := dispatchPull(ctx, ref, layoutDir); handled {
		return ref, false, true, err
	}

	// Normalize: append :latest if reference has no tag or digest.
	if strings.LastIndex(ref, ":") <= strings.LastIndex(ref, "/") {
		ref = ref + ":latest"
	}

	host := strings.SplitN(ref, "/", 2)[0]
	return ref, isOCIHost(ctx, host), false, nil
}

// ---------------------------------------------------------------------------
// HuggingFace API types and helpers
// ---------------------------------------------------------------------------

// hfFile is one entry returned by the HuggingFace tree API.
type hfFile struct {
	Path string `json:"path"`
	Size int64  `json:"size"`
	OID  string `json:"oid"`
	Type string `json:"type"` // "file" or "directory"
}

// hfAPIClient returns the HTTP client used for HuggingFace metadata requests
// (commit lookup, file listing, HEAD digest probes) — a short total timeout
// suffices since these responses are small. Shared by pullHF (this file) and
// dockerTransferHF (transfer_docker.go), which both need one.
func hfAPIClient() *http.Client {
	return &http.Client{Timeout: 120 * time.Second}
}

// hfDownloadClient returns the HTTP client used for actually downloading (or
// streaming) HuggingFace file content: no body read timeout so large files
// can transfer without a deadline, but connection and header timeouts still
// prevent hanging on a stalled server. Mirrors llama.cpp's
// common/download.cpp approach. Shared by pullHF (this file) and
// dockerTransferHF (transfer_docker.go).
func hfDownloadClient() *http.Client {
	return &http.Client{
		Transport: &http.Transport{
			DialContext: (&net.Dialer{
				Timeout:   30 * time.Second,
				KeepAlive: 30 * time.Second,
			}).DialContext,
			TLSHandshakeTimeout:   30 * time.Second,
			ResponseHeaderTimeout: 60 * time.Second,
		},
	}
}

// hfEndpoint returns the HuggingFace API base URL for the host.
// Mirrors llama.cpp's MODEL_ENDPOINT / HF_ENDPOINT override logic.
func hfEndpoint(host string) string {
	for _, env := range []string{"MODEL_ENDPOINT", "HF_ENDPOINT"} {
		if v := os.Getenv(env); v != "" {
			return strings.TrimRight(v, "/") + "/"
		}
	}
	if host == "hf.co" {
		return "https://huggingface.co/"
	}
	return "https://" + host + "/"
}

// hfToken resolves the HuggingFace bearer token to use for authenticated
// requests, mirroring huggingface_hub's own resolution order: the HF_TOKEN
// environment variable (falling back to the legacy
// HUGGING_FACE_HUB_TOKEN), then the on-disk active-token file written by
// `llmman login` — see the Rust `hf` module's `token_path`, which uses the
// exact same path, so either tool's login is honored by the other.
func hfToken() string {
	for _, env := range []string{"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"} {
		if v := strings.TrimSpace(os.Getenv(env)); v != "" {
			return v
		}
	}
	data, err := os.ReadFile(hfTokenPath())
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

// hfTokenPath returns the path to the active HuggingFace token file:
// $HF_TOKEN_PATH if set, else "$HF_HOME/token", else
// "~/.cache/huggingface/token".
func hfTokenPath() string {
	if p := os.Getenv("HF_TOKEN_PATH"); p != "" {
		return p
	}
	if home := os.Getenv("HF_HOME"); home != "" {
		return filepath.Join(home, "token")
	}
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".cache", "huggingface", "token")
	}
	return filepath.Join(".cache", "huggingface", "token")
}

// hfHeadMetadata performs a HEAD request against a HuggingFace file's
// /resolve/ URL and reports the file's real content digest, size, and
// (if present) Xet hash (X-Xet-Hash), without downloading the body —
// mirroring huggingface_hub's own get_hf_file_metadata(). This is what
// makes streaming a HuggingFace file straight into a registry push
// possible at all: containerd/OCI registry
// pushes require the blob's digest to be known *before* any bytes are
// sent (see backend_docker.go's llmman_transfer), and for a large,
// LFS-tracked file (virtually every real GGUF/safetensors weight file)
// the true sha256 of the content is exposed via the X-Linked-Etag header
// on this cheap HEAD request — the same field huggingface_hub prefers
// over the plain ETag for exactly this reason (LFS pointer vs. real
// object). ok is false when the digest can't be determined this way
// (small, non-LFS files, where the ETag is a git blob sha1, not a sha256
// of the content) — callers should fall back to a normal buffered
// download for those; they're tiny (config/tokenizer files), so buffering
// them in memory costs nothing.
// hfHeadMetadataMaxHops bounds how many redirects hfHeadMetadata will
// follow on its own looking for X-Linked-Etag/X-Linked-Size, in case of
// a redirect loop or some other pathological chain — real chains are one
// or two hops (see hfHeadMetadata's own comment), so this is generous
// headroom, not a limit ever expected to actually bind.
const hfHeadMetadataMaxHops = 5

func hfHeadMetadata(ctx context.Context, client *http.Client, target, token string) (dgst digest.Digest, size int64, xetHash string, ok bool, err error) {
	// Do NOT let http.Client itself follow redirects: huggingface.co sets
	// X-Linked-Etag/X-Linked-Size on its own redirecting response
	// (pointing at the real content's sha256/size before it hands off to
	// a CDN); the CDN's own response has neither header and sets an
	// unrelated ETag of its own (its storage object's identifier, not a
	// content hash we can trust) — using that instead silently produces
	// a wrong digest that a registry push then rejects as DIGEST_INVALID
	// after fully uploading the (correct) bytes under the (wrong)
	// declared name. So this follows redirects itself, one hop at a
	// time, stopping the moment a response actually carries those
	// headers rather than assuming that's always the very first hop.
	//
	// It isn't always: a *renamed* repository (an owner or repo name
	// changed after the URL being resolved here was written down, e.g.
	// models/ornith/35b-safetensors's deepreinforce-ai/Ornith-1.0-35B,
	// now ornith-ai/Ornith-1.0-35B) redirects huggingface.co → itself
	// first, to the new owner/repo, carrying neither header, before ever
	// reaching the resolve endpoint's actual CDN-bound redirect that
	// does. Stopping at that first hop unconditionally (an earlier
	// version of this code did) finds neither header on it and
	// concludes — wrongly — that this file has no usable digest at all,
	// which sends a real, multi-gigabyte weight file down
	// streamHFFileToRegistry's small-non-LFS-file fallback instead:
	// buffered entirely in memory rather than streamed, which is exactly
	// how docker/llmman-publisher's ornith:35b-safetensors transfer was
	// running a GitHub-hosted runner out of memory (each of that
	// repository's 16 safetensors shards is itself several gigabytes on
	// its own) rather than actually failing on anything about the
	// transfer itself.
	noRedirect := &http.Client{
		Transport: client.Transport,
		Timeout:   client.Timeout,
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}

	for hop := 0; hop < hfHeadMetadataMaxHops; hop++ {
		req, err := http.NewRequestWithContext(ctx, "HEAD", target, nil)
		if err != nil {
			return "", 0, "", false, err
		}
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		req.Header.Set("Accept-Encoding", "identity") // force the real, uncompressed size

		resp, err := noRedirect.Do(req)
		if err != nil {
			return "", 0, "", false, fmt.Errorf("HEAD %s: %w", target, err)
		}
		resp.Body.Close()
		if resp.StatusCode != 200 && (resp.StatusCode < 300 || resp.StatusCode >= 400) {
			return "", 0, "", false, fmt.Errorf("HEAD %s: HTTP %d", target, resp.StatusCode)
		}

		xLinkedEtag := resp.Header.Get("X-Linked-Etag")
		xLinkedSize := resp.Header.Get("X-Linked-Size")
		xXetHash := resp.Header.Get("X-Xet-Hash")
		isRedirect := resp.StatusCode >= 300 && resp.StatusCode < 400
		if isRedirect && xLinkedEtag == "" && xLinkedSize == "" {
			loc := resp.Header.Get("Location")
			next, parseErr := neturl.Parse(loc)
			if loc == "" || parseErr != nil {
				break // no usable Location to follow — give up, same as before
			}
			target = req.URL.ResolveReference(next).String()
			continue
		}

		// Read size first and independently of digest validity below:
		// callers that fall back to buffering (small, non-LFS files)
		// still want an accurate progress-bar size even though the
		// digest can't be trusted yet — see transfer_docker.go's
		// streamHFFileToRegistry.
		sizeStr := xLinkedSize
		if sizeStr == "" && resp.StatusCode == 200 {
			// Only trust a plain Content-Length when there was no
			// redirect — a redirect response's Content-Length describes
			// its own (tiny) body, not the file being redirected to.
			sizeStr = resp.Header.Get("Content-Length")
		}
		if sizeStr != "" {
			if n, convErr := parseInt64(sizeStr); convErr == nil {
				size = n
			}
		}

		etag := xLinkedEtag
		if etag == "" {
			etag = resp.Header.Get("ETag")
		}
		etag = strings.TrimPrefix(etag, "W/")
		etag = strings.Trim(etag, `"`)
		if len(etag) != 64 {
			return "", size, xXetHash, false, nil // not a sha256 — not LFS, caller should buffer instead
		}

		return digest.NewDigestFromEncoded(digest.SHA256, strings.ToLower(etag)), size, xXetHash, true, nil
	}
	return "", 0, "", false, nil
}

func parseInt64(s string) (int64, error) {
	var n int64
	_, err := fmt.Sscanf(s, "%d", &n)
	return n, err
}

// hfGet issues an authenticated GET and decodes JSON into dst. Transient
// failures — a connection reset mid-handshake, a 5xx, a timeout — are
// retried with the same backoff budget as blob downloads (see
// retryStream); without this, one TCP blip on a cheap metadata call
// fails an entire transfer that the much larger blob traffic would have
// survived. 4xx responses are permanent (isHTTP4xx) and fail
// immediately, exactly as before. The body is buffered and decoded only
// after a fully successful response so a failed attempt can never leave
// dst partially populated.
func hfGet(ctx context.Context, client *http.Client, url, token string, dst any) error {
	var body []byte
	err := retryStream(ctx, "GET "+url, isHTTP4xx, func() error {
		req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
		if err != nil {
			return err
		}
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		resp, err := client.Do(req)
		if err != nil {
			return fmt.Errorf("GET %s: %w", url, err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != 200 {
			return newHTTPStatusError("GET "+url, resp)
		}
		body, err = io.ReadAll(resp.Body)
		return err
	})
	if err != nil {
		return err
	}
	return json.Unmarshal(body, dst)
}

// hfModelInfo is the subset of HuggingFace's GET /api/models/{owner}/{repo}
// response this package needs: the current commit SHA (for pinning
// resolve URLs to an exact revision, same as before) plus enough of the
// model card to populate the CNCF config's descriptor.licenses — see
// license() below. cardData.license is present on essentially every real
// model repo (it's what renders as the license badge on the repo's own
// HuggingFace page); the "license:<slug>" tag is checked as a fallback
// for the rare repo that has tags but no full cardData block.
type hfModelInfo struct {
	SHA      string   `json:"sha"`
	Tags     []string `json:"tags"`
	CardData struct {
		License string `json:"license"`
	} `json:"cardData"`
}

// hfFetchModelInfo fetches hfModelInfo for owner/repo.
func hfFetchModelInfo(ctx context.Context, client *http.Client, endpoint, owner, repo, token string) (hfModelInfo, error) {
	var info hfModelInfo
	url := endpoint + "api/models/" + owner + "/" + repo
	if err := hfGet(ctx, client, url, token, &info); err != nil {
		return hfModelInfo{}, fmt.Errorf("HF model info: %w", err)
	}
	return info, nil
}

// commit returns the commit SHA to pin resolve URLs to, falling back to
// "main" if HuggingFace didn't report one.
func (info hfModelInfo) commit() string {
	if info.SHA == "" {
		return "main"
	}
	return info.SHA
}

// license returns the model's license as a best-effort SPDX license
// expression (see normalizeSPDXLicense), and false if the repo doesn't
// declare one usable at all (no cardData.license/license: tag, or a
// value like "other"/"unknown" that isn't a real license identifier).
func (info hfModelInfo) license() (string, bool) {
	if info.CardData.License != "" {
		if id := normalizeSPDXLicense(info.CardData.License); id != "" {
			return id, true
		}
	}
	for _, t := range info.Tags {
		if slug, ok := strings.CutPrefix(t, "license:"); ok && slug != "" {
			if id := normalizeSPDXLicense(slug); id != "" {
				return id, true
			}
		}
	}
	return "", false
}

// spdxLicenseIDs maps a HuggingFace license slug (cardData.license or a
// "license:<slug>" tag — always lowercase and hyphenated) to its proper
// SPDX license expression, for the licenses these model pipelines
// actually use in practice. "other" and "unknown" are HuggingFace's own
// catch-all slugs for "not a real license identifier" and deliberately
// map to "" so license() reports them as not usable rather than
// fabricating a bogus SPDX expression. Anything else not in this table
// falls through unchanged — not guaranteed to be a valid SPDX identifier,
// but a closer guess than omitting it entirely, and HuggingFace's own
// slugs are already SPDX identifiers lowercased for most of the common
// ones this table doesn't need to list.
var spdxLicenseIDs = map[string]string{
	"apache-2.0":   "Apache-2.0",
	"mit":          "MIT",
	"bsd-2-clause": "BSD-2-Clause",
	"bsd-3-clause": "BSD-3-Clause",
	"gpl-2.0":      "GPL-2.0-only",
	"gpl-3.0":      "GPL-3.0-only",
	"lgpl-2.1":     "LGPL-2.1-only",
	"lgpl-3.0":     "LGPL-3.0-only",
	"mpl-2.0":      "MPL-2.0",
	"cc-by-4.0":    "CC-BY-4.0",
	"cc-by-sa-4.0": "CC-BY-SA-4.0",
	"cc0-1.0":      "CC0-1.0",
	"other":        "",
	"unknown":      "",
}

func normalizeSPDXLicense(slug string) string {
	slug = strings.ToLower(strings.TrimSpace(slug))
	if id, ok := spdxLicenseIDs[slug]; ok {
		return id
	}
	return slug
}

// hfFetchFiles returns the recursive file listing for owner/repo at commit.
func hfFetchFiles(ctx context.Context, client *http.Client, endpoint, owner, repo, commit, token string) ([]hfFile, error) {
	var files []hfFile
	url := endpoint + "api/models/" + owner + "/" + repo + "/tree/" + commit + "?recursive=true"
	if err := hfGet(ctx, client, url, token, &files); err != nil {
		return nil, fmt.Errorf("HF file list: %w", err)
	}
	return files, nil
}

// ---------------------------------------------------------------------------
// GGUF file selection (mirrors llama.cpp find_best_model)
// ---------------------------------------------------------------------------

// quantPreference is the default quantization preference order, matching llama.cpp.
var quantPreference = []string{"Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0", "Q4_0", "Q6_K", "Q2_K"}

// isModelGGUF returns true for GGUF files that are primary model weights
// (not mmproj projectors or imatrix importance files).
func isModelGGUF(path string) bool {
	lower := strings.ToLower(path)
	return strings.HasSuffix(lower, ".gguf") &&
		!strings.Contains(lower, "mmproj") &&
		!strings.Contains(lower, "imatrix")
}

// ggufShardPattern matches llama.cpp's own gguf-split naming convention —
// "<name>-NNNNN-of-MMMMM.gguf" — used to split a model too large for a
// single file into several (very common for large MoE models; e.g.
// unsloth's DeepSeek-V4-Flash-0731-GGUF ships each quant as 5 parts).
// Capture group 1 is the shared prefix every shard of the same split has
// in common, group 2 is this shard's 1-based index, group 3 is the total
// shard count — both fixed-width, but parsed as plain integers rather
// than matched verbatim so a total of e.g. 5 still matches whether it's
// spelled "00005" or "5".
var ggufShardPattern = regexp.MustCompile(`^(.*)-(\d+)-of-(\d+)\.gguf$`)

// parseGGUFShard parses ggufShardPattern out of path's base name. ok is
// false for a file that isn't part of a multi-part split.
func parseGGUFShard(path string) (prefix string, index, total int, ok bool) {
	m := ggufShardPattern.FindStringSubmatch(filepath.Base(path))
	if m == nil {
		return "", 0, 0, false
	}
	idx, err1 := strconv.Atoi(m[2])
	tot, err2 := strconv.Atoi(m[3])
	if err1 != nil || err2 != nil {
		return "", 0, 0, false
	}
	return m[1], idx, tot, true
}

// ggufShards returns every shard of the same multi-part split as chosen,
// in shard order — see ggufShardPattern. A manifest built from only the
// first shard of a split (as an earlier version of selectGGUF did, simply
// returning the first path matching a quant substring — the *first*
// shard's filename matches just as well as any other's) silently produces
// a model no GGUF-reading runtime can actually load, since the remaining
// shards' tensors are just missing. Returns []hfFile{chosen} unchanged
// for a file that isn't part of a split.
func ggufShards(models []hfFile, chosen hfFile) []hfFile {
	prefix, _, total, ok := parseGGUFShard(chosen.Path)
	if !ok {
		return []hfFile{chosen}
	}
	var shards []hfFile
	for _, f := range models {
		if p, _, t, ok := parseGGUFShard(f.Path); ok && p == prefix && t == total {
			shards = append(shards, f)
		}
	}
	sort.Slice(shards, func(i, j int) bool {
		_, ii, _, _ := parseGGUFShard(shards[i].Path)
		_, jj, _, _ := parseGGUFShard(shards[j].Path)
		return ii < jj
	})
	return shards
}

// selectGGUF picks the best GGUF quant from the file listing, returning
// every shard of a multi-part split together (see ggufShards) rather than
// just whichever shard happened to match first.
// tag is the user-supplied quantization hint (e.g. "Q4_K_M") or empty for auto.
func selectGGUF(files []hfFile, tag string) ([]hfFile, error) {
	var models []hfFile
	for _, f := range files {
		if f.Type == "file" && isModelGGUF(f.Path) {
			models = append(models, f)
		}
	}
	if len(models) == 0 {
		return nil, fmt.Errorf("no GGUF model files found in repository")
	}

	// Explicit tag: user asked for a specific quantization.
	if tag != "" && tag != "latest" {
		upper := strings.ToUpper(tag)
		for _, f := range models {
			if strings.Contains(strings.ToUpper(f.Path), upper) {
				return ggufShards(models, f), nil
			}
		}
		return nil, fmt.Errorf("no GGUF file matching %q found; available:\n%s",
			tag, ggufList(models))
	}

	// Auto-select by preference list (Q4_K_M first, then Q8_0, …).
	for _, pref := range quantPreference {
		for _, f := range models {
			if strings.Contains(strings.ToUpper(f.Path), pref) {
				return ggufShards(models, f), nil
			}
		}
	}

	// Fallback: smallest file (most compressed).
	sort.Slice(models, func(i, j int) bool { return models[i].Size < models[j].Size })
	return ggufShards(models, models[0]), nil
}

func ggufList(files []hfFile) string {
	var b strings.Builder
	for _, f := range files {
		b.WriteString("  " + f.Path + "\n")
	}
	return b.String()
}

// ---------------------------------------------------------------------------
// Multimodal projector (mmproj) and LICENSE selection
//
// A GGUF repo's mmproj-*.gguf and LICENSE files are both real files sitting
// right there in the same file listing selectGGUF already has — isModelGGUF
// deliberately excludes mmproj from GGUF weight selection (see its own doc
// comment), and license files were never even file-listing-filtered for at
// all, they just weren't looked for. Both are optional: most repos have
// neither, some vision models have an mmproj, most repos have a LICENSE.
// ---------------------------------------------------------------------------

// mmprojPreference orders which multimodal-projector precision to pick when
// a repo ships several (BF16/F16/F32 side by side, as unsloth's vision
// model GGUF repos generally do) — F16 first, matching what every
// model-publisher JSON config observed in practice consistently points at
// for the same repos.
var mmprojPreference = []string{"F16", "BF16", "F32"}

// selectMMProj returns the repo's multimodal projector file, if it has
// one, for pairing alongside the chosen GGUF weight(s) as an additional
// weight-typed layer (see dockerTransferHF/pullHF) — a model-spec
// manifest has no dedicated media type for this (it's a llama.cpp/GGUF-
// specific concept the spec predates), so it's just another weight layer
// distinguished by its own org.cncf.model.filepath annotation, the same
// way every other layer is.
func selectMMProj(files []hfFile) (hfFile, bool) {
	var candidates []hfFile
	for _, f := range files {
		if f.Type != "file" {
			continue
		}
		lower := strings.ToLower(f.Path)
		if strings.Contains(lower, "mmproj") && strings.HasSuffix(lower, ".gguf") {
			candidates = append(candidates, f)
		}
	}
	for _, pref := range mmprojPreference {
		for _, f := range candidates {
			// Match pref as a whole "-<pref>." component, not merely a
			// substring: "-F16." must not match "mmproj-BF16.gguf" just
			// because "BF16" itself contains the substring "F16".
			base := strings.ToUpper(filepath.Base(f.Path))
			if base == pref+".GGUF" || strings.HasSuffix(base, "-"+pref+".GGUF") {
				return f, true
			}
		}
	}
	if len(candidates) > 0 {
		return candidates[0], true
	}
	return hfFile{}, false
}

// licenseFilenames are the conventional base names a HuggingFace repo's
// plain-text/markdown license file uses, checked in this order.
var licenseFilenames = []string{"LICENSE", "LICENSE.txt", "LICENSE.md"}

// selectLicenseFile returns the repo's root-level LICENSE file, if it has
// one, for attaching as an application/vnd.cncf.model.doc.v1.raw layer —
// spec.md explicitly names LICENSE as a doc-type file example alongside
// README.md.
func selectLicenseFile(files []hfFile) (hfFile, bool) {
	for _, want := range licenseFilenames {
		for _, f := range files {
			if f.Type == "file" && strings.EqualFold(f.Path, want) {
				return f, true
			}
		}
	}
	return hfFile{}, false
}

// ---------------------------------------------------------------------------
// parseHFRef
// ---------------------------------------------------------------------------

// parseHFRef splits a (possibly `:latest`-normalized) HF reference
// "host/owner/repo[:tag]" into its four components.
func parseHFRef(ref string) (host, owner, repo, tag string, err error) {
	if idx := strings.LastIndex(ref, ":"); idx > strings.LastIndex(ref, "/") {
		tag = ref[idx+1:]
		ref = ref[:idx]
	}
	parts := strings.SplitN(ref, "/", 3)
	if len(parts) != 3 {
		return "", "", "", "", fmt.Errorf("invalid HuggingFace reference %q: expected host/owner/repo", ref)
	}
	return parts[0], parts[1], parts[2], tag, nil
}

// ---------------------------------------------------------------------------
// pullHF — top-level HuggingFace pull
// ---------------------------------------------------------------------------

// cachedLayerName returns the GGUF filename for ref if it is fully cached in
// the local OCI store (manifest blob + all layer blobs present), or "" if not.
func cachedLayerName(layoutDir, ref string) string {
	m, err := readManifestRef(layoutDir, ref)
	if err != nil {
		return ""
	}
	if !blobExists(layoutDir, m) {
		return ""
	}
	data, err := readBlob(layoutDir, m.Digest)
	if err != nil {
		return ""
	}
	var manifest ocispec.Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return ""
	}
	for _, layer := range manifest.Layers {
		if !blobExists(layoutDir, layer) {
			return ""
		}
	}
	// All blobs present — return a filename from the first layer annotation.
	if len(manifest.Layers) > 0 {
		ann := manifest.Layers[0].Annotations
		for _, key := range []string{modelspec.AnnotationFilepath, ocispec.AnnotationTitle} {
			if name := ann[key]; name != "" {
				return filepath.Base(name)
			}
		}
	}
	return ref
}

// reportCached prints "Cached <label>" and returns true if ref is already
// fully cached in layoutDir (manifest blob + all layer blobs present) — the
// signal every pull entry point in this package uses to skip all network
// I/O. label defaults to the cached name cachedLayerName itself resolved
// (e.g. a GGUF filename) when the empty string is passed.
func reportCached(layoutDir, ref, label string) bool {
	name := cachedLayerName(layoutDir, ref)
	if name == "" {
		return false
	}
	if label == "" {
		label = name
	}
	fmt.Fprintf(os.Stderr, "Cached   %s\n", label)
	return true
}

// progressKey is the exact, original ref the daemon's /api/pull handler
// was given — normally identical to ref, except when called from
// dispatchPull's "hf://"/"huggingface://" scheme handling, where ref has
// already had that scheme prefix stripped (for use as the storage/index
// key) but the Rust side is still polling llmman_progress with the
// original, prefixed string (see progress_state.go).
func pullHF(ctx context.Context, ref, layoutDir, progressKey string) error {
	host, owner, repo, tag, err := parseHFRef(ref)
	if err != nil {
		return err
	}

	if err := ensureLayout(layoutDir); err != nil {
		return fmt.Errorf("init OCI layout: %w", err)
	}

	// Fast path: skip all network I/O if the ref is fully cached locally.
	if reportCached(layoutDir, ref, "") {
		return nil
	}

	endpoint := hfEndpoint(host)
	token := hfToken()

	apiClient := hfAPIClient()
	dlClient := hfDownloadClient()

	info, err := hfFetchModelInfo(ctx, apiClient, endpoint, owner, repo, token)
	if err != nil {
		return err
	}
	commit := info.commit()
	meta := modelMeta{}
	if license, ok := info.license(); ok {
		meta.Licenses = []string{license}
	}

	files, err := hfFetchFiles(ctx, apiClient, endpoint, owner, repo, commit, token)
	if err != nil {
		return err
	}
	progressSetStatus(progressKey, "pulling")

	// Try GGUF first; fall back to safetensors if the repo has none.
	if shards, err := selectGGUF(files, tag); err == nil {
		meta.Format = "gguf"
		filepathAnnotation := ""
		if len(shards) == 1 {
			filepathAnnotation = filepath.Base(shards[0].Path)
		}
		var layers []ocispec.Descriptor
		for _, f := range shards {
			downloadURL := endpoint + owner + "/" + repo + "/resolve/" + commit + "/" + f.Path
			desc, err := downloadHFBlob(ctx, dlClient, downloadURL, token, layoutDir, owner, repo, commit, f, progressKey)
			if err != nil {
				return err
			}
			layers = append(layers, desc)
		}
		// mmproj: an optional extra weight layer alongside the chosen
		// GGUF shard(s) — see selectMMProj's own doc comment for why
		// this has no dedicated media type of its own.
		if mmproj, ok := selectMMProj(files); ok {
			downloadURL := endpoint + owner + "/" + repo + "/resolve/" + commit + "/" + mmproj.Path
			desc, err := downloadHFBlob(ctx, dlClient, downloadURL, token, layoutDir, owner, repo, commit, mmproj, progressKey)
			if err != nil {
				return fmt.Errorf("download %s: %w", mmproj.Path, err)
			}
			layers = append(layers, desc)
			meta.Vision = true
		}
		// LICENSE: a doc-type layer per spec.md's own example of what
		// application/vnd.cncf.model.doc.v1.raw is for.
		if lic, ok := selectLicenseFile(files); ok {
			downloadURL := endpoint + owner + "/" + repo + "/resolve/" + commit + "/" + lic.Path
			desc, err := downloadHFBlob(ctx, dlClient, downloadURL, token, layoutDir, owner, repo, commit, lic, progressKey)
			if err != nil {
				return fmt.Errorf("download %s: %w", lic.Path, err)
			}
			desc.MediaType = modelspec.MediaTypeModelDocRaw
			layers = append(layers, desc)
		}
		return storeGGUFAsOCI(layoutDir, ref, owner+"/"+repo, meta, filepathAnnotation, layers)
	}

	// No GGUF found — pull safetensors files as a CNCF model-spec image.
	meta.Format = "safetensors"
	return pullHFSafetensors(ctx, dlClient, ref, layoutDir, endpoint, owner, repo, commit, token, files, meta, progressKey)
}

// safetensorsMediaType maps a file extension to the appropriate CNCF layer media type.
//
// ".jinja" is config, not doc: many HF repos ship a standalone
// chat_template.jinja file, and the Rust-side extractor drops "doc"
// layers, so a chat template classified that way never reaches the
// served model directory and vllm refuses every chat request.
func safetensorsMediaType(path string) string {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".safetensors", ".bin", ".pt", ".pth":
		return modelspec.MediaTypeModelWeightRaw
	case ".json", ".model", ".txt", ".tiktoken", ".jinja":
		return modelspec.MediaTypeModelWeightConfigRaw
	default:
		return modelspec.MediaTypeModelDocRaw
	}
}

// shouldDownloadSafetensors returns true for files that belong in a local model directory.
func shouldDownloadSafetensors(path string) bool {
	base := strings.ToLower(filepath.Base(path))
	ext := strings.ToLower(filepath.Ext(path))
	// Skip hidden files, large non-model binaries, and git internals.
	if strings.HasPrefix(base, ".") {
		return false
	}
	switch ext {
	case ".safetensors", ".bin", ".pt", ".pth": // weights
		return true
	// config / tokenizer — ".jinja" is a standalone chat template file,
	// see safetensorsMediaType.
	case ".json", ".model", ".txt", ".tiktoken", ".jinja":
		return true
	}
	// README and licence are useful but optional.
	switch base {
	case "readme.md", "license", "licence", "license.txt", "licence.txt":
		return true
	}
	return false
}

// selectDownloadableHFFiles filters files down to the plain files that
// shouldDownloadSafetensors accepts, ignoring directories. Shared by
// pullHFSafetensors (this file) and dockerTransferHF (transfer_docker.go).
func selectDownloadableHFFiles(files []hfFile) []hfFile {
	var out []hfFile
	for _, f := range files {
		if f.Type == "file" && shouldDownloadSafetensors(f.Path) {
			out = append(out, f)
		}
	}
	return out
}

func pullHFSafetensors(
	ctx context.Context,
	client *http.Client,
	ref, layoutDir, endpoint, owner, repo, commit, token string,
	files []hfFile,
	meta modelMeta,
	progressKey string,
) error {
	toDownload := selectDownloadableHFFiles(files)
	if len(toDownload) == 0 {
		return fmt.Errorf("no model files found in repository %s/%s", owner, repo)
	}

	var layers []ocispec.Descriptor
	for _, f := range toDownload {
		url := endpoint + owner + "/" + repo + "/resolve/" + commit + "/" + f.Path
		desc, err := downloadHFBlob(ctx, client, url, token, layoutDir, owner, repo, commit, f, progressKey)
		if err != nil {
			return fmt.Errorf("download %s: %w", f.Path, err)
		}
		// Override media type and use the full relative path as the filepath annotation.
		desc.MediaType = safetensorsMediaType(f.Path)
		desc.Annotations = map[string]string{
			modelspec.AnnotationFilepath: f.Path,
		}
		layers = append(layers, desc)
	}

	return storeSafetensorsAsOCI(layoutDir, ref, owner+"/"+repo, meta, layers)
}

func storeSafetensorsAsOCI(layoutDir, ref, modelRepo string, meta modelMeta, layers []ocispec.Descriptor) error {
	manifestDesc, err := buildCNCFManifest(layoutBlobSink(layoutDir), meta, modelRepo, "", layers)
	if err != nil {
		return err
	}
	return writeManifestRef(layoutDir, ref, manifestDesc)
}

// ---------------------------------------------------------------------------
// downloadHFBlob — HTTP download with resume, retry, and stall detection.
// Mirrors llama.cpp common/download.cpp: 3 attempts, 2s/4s backoff.
//
// dlMaxAttempts/dlRetryBase/dlStallTimeout/stallReader/isHTTP4xx/retryStream
// now live in shared_oci.go — they're used here for the local-disk pull
// path (which can resume a partial download with a Range request against
// its own .part file) and by transfer_docker.go's streaming push path
// (which, lacking a resumable registry upload — see that file's own
// comment on containerd's docker Pusher — can only retry a failed blob
// from scratch, not resume it, but still benefits from the same
// backoff/stall/permanent-vs-transient logic).
// ---------------------------------------------------------------------------

func downloadHFBlob(ctx context.Context, client *http.Client, url, token, layoutDir, owner, repo, commit string, file hfFile, progressKey string) (ocispec.Descriptor, error) {
	if err := os.MkdirAll(filepath.Join(layoutDir, "blobs"), 0o755); err != nil {
		return ocispec.Descriptor{}, err
	}

	sanitize := strings.NewReplacer("/", "_", ":", "_", ".", "_")
	tmpKey := sanitize.Replace(owner + "_" + repo + "_" + commit[:12] + "_" + filepath.Base(file.Path))
	tmpPath := filepath.Join(layoutDir, "blobs", "hf-"+tmpKey+".part")

	// Deduplicate against any other pull in this process (a different
	// model, running concurrently now that pulls of distinct models are
	// no longer serialized against each other — see blobFetchGroup's own
	// doc comment) that's downloading this exact same source file right
	// now, rather than racing it to write the same deterministic tmpPath.
	return dedupBlobFetch(tmpKey, progressKey, file.Size, func() (ocispec.Descriptor, error) {
		return downloadHFBlobAttempts(ctx, client, url, token, layoutDir, tmpPath, file, progressKey)
	})
}

// downloadHFBlobAttempts is downloadHFBlob's actual retry loop, run at
// most once per tmpKey across every concurrent caller — see
// dedupBlobFetch.
func downloadHFBlobAttempts(ctx context.Context, client *http.Client, url, token, layoutDir, tmpPath string, file hfFile, progressKey string) (ocispec.Descriptor, error) {
	label := "Pulling  " + filepath.Base(file.Path)
	doneLbl := "Pulled   " + filepath.Base(file.Path)
	prog := newProgressPool(80)
	bar := addLayerBar(prog, label, doneLbl, file.Size, progressKey)

	var lastErr error
	// creditedResume tracks how much of the .part file's already-downloaded
	// prefix has already been folded into progressState's completed count
	// (see progress_state.go), so a retry that finds the same partial file
	// it left behind doesn't get double-credited for it — only the delta
	// since the last attempt's startOffset is ever added. (One rare edge
	// case isn't covered: downloadAttempt's own "server ignored Range
	// header, restart from zero" fallback resets the bar but has no way to
	// tell this loop to un-credit what was already added; recovering from
	// a non-range-supporting server mid-retry is rare enough that a
	// possibly-early 100% on the aggregate bar is an acceptable trade-off
	// for not threading extra state through downloadAttempt for it.)
	var creditedResume int64
	for attempt := 0; attempt < dlMaxAttempts; attempt++ {
		if attempt > 0 {
			delay := retryDelay(attempt)
			if ra, ok := retryAfter(lastErr); ok {
				delay = ra
			}
			fmt.Fprintf(os.Stderr, "\n[llmman] retrying %s (attempt %d/%d, wait %v)\n",
				filepath.Base(file.Path), attempt+1, dlMaxAttempts, delay)
			select {
			case <-ctx.Done():
				bar.Abort(false)
				prog.Wait()
				return ocispec.Descriptor{}, ctx.Err()
			case <-time.After(delay):
			}
		}

		// Re-read partial file size in case previous attempt downloaded some bytes.
		startOffset := int64(0)
		if fi, err := os.Stat(tmpPath); err == nil && fi.Size() > 0 && fi.Size() < file.Size {
			startOffset = fi.Size()
		}
		bar.SetCurrent(startOffset)
		if startOffset > creditedResume {
			progressAddCompleted(progressKey, startOffset-creditedResume)
			creditedResume = startOffset
		}

		desc, err := downloadAttempt(ctx, client, url, token, layoutDir, tmpPath, startOffset, file, bar, progressKey)
		if err == nil {
			prog.Wait()
			return desc, nil
		}

		lastErr = err
		// 4xx errors are permanent — no point retrying.
		if isHTTP4xx(err) {
			break
		}
		// Network/5xx error: keep partial file, retry with resume.
		fmt.Fprintf(os.Stderr, "[llmman] download error: %v\n", err)
	}

	bar.Abort(false)
	prog.Wait()
	os.Remove(tmpPath) // exhausted retries
	return ocispec.Descriptor{}, fmt.Errorf("download %s failed after %d attempts: %w",
		filepath.Base(file.Path), dlMaxAttempts, lastErr)
}

// downloadAttempt performs one download attempt with stall detection.
func downloadAttempt(ctx context.Context, client *http.Client, url, token, layoutDir, tmpPath string, startOffset int64, file hfFile, bar *mpb.Bar, progressKey string) (ocispec.Descriptor, error) {
	// Per-attempt context with stall cancellation.
	attemptCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	req, err := http.NewRequestWithContext(attemptCtx, "GET", url, nil)
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	// Always send a Range header, even for a fresh download at byte 0 —
	// some HF CDNs (seen on a CloudFront/S3-fronted Xet-CAS bridge) 400 a
	// full-object GET with no Range at all past a few tens of GB, which
	// isHTTP4xx treats as permanent, so every retry just repeats the same
	// failure. The same origin serves "bytes=0-" fine as a 206.
	req.Header.Set("Range", fmt.Sprintf("bytes=%d-", startOffset))

	resp, err := client.Do(req)
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("download %s: %w", file.Path, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 && resp.StatusCode != 206 {
		return ocispec.Descriptor{}, newHTTPStatusError("download "+file.Path, resp)
	}
	if startOffset > 0 && resp.StatusCode == 200 {
		// Server ignored Range header — restart from zero.
		startOffset = 0
		bar.SetCurrent(0)
	}

	f, digester, startOffset, err := openForResume(tmpPath, startOffset)
	if err != nil {
		return ocispec.Descriptor{}, err
	}

	// Wrap with stall/slow-speed detector: cancel attemptCtx if no bytes
	// for 60s, or if the sustained rate drops far below this process's
	// recent median transfer speed (see stallReader.checkSpeed).
	sr := newStallReader(resp.Body, dlStallTimeout, cancel)
	defer sr.stop()

	proxyRC := proxyOrNop(bar, sr, progressKey)
	written, copyErr := io.Copy(io.MultiWriter(f, digester.Hash()), proxyRC)
	proxyRC.Close()
	f.Close()

	if copyErr != nil {
		// Partial file kept for resume on next attempt — do NOT remove it here.
		return ocispec.Descriptor{}, fmt.Errorf("write %s: %w", file.Path, copyErr)
	}
	// Feed this attempt's throughput into the process-wide speed tracker
	// so future transfers can detect running anomalously slowly by
	// comparison — see globalSpeedTracker.
	globalSpeedTracker.record(sr.finalSpeed())
	total := startOffset + written
	dgst := digester.Digest()

	// Move to content-addressed path.
	dir := filepath.Join(layoutDir, "blobs", dgst.Algorithm().String())
	if err := os.MkdirAll(dir, 0o755); err != nil {
		os.Remove(tmpPath)
		return ocispec.Descriptor{}, err
	}
	dest := filepath.Join(dir, dgst.Hex())
	if fi, err := os.Stat(dest); err == nil && fi.Size() == total {
		os.Remove(tmpPath) // already exists (idempotent)
	} else if err := os.Rename(tmpPath, dest); err != nil {
		os.Remove(tmpPath)
		return ocispec.Descriptor{}, err
	}

	return ocispec.Descriptor{
		// Use the CNCF model-spec weight media type so the stored manifest is
		// spec-compliant.  llmman's serve layer detection falls back to checking
		// the org.cncf.model.filepath annotation for ".gguf", so old manifests
		// (application/vnd.docker.ai.gguf.v3) still work via the other check.
		MediaType: modelspec.MediaTypeModelWeightRaw,
		Digest:    dgst,
		Size:      total,
		Annotations: map[string]string{
			modelspec.AnnotationFilepath: filepath.Base(file.Path),
		},
	}, nil
}

// ---------------------------------------------------------------------------
// storeGGUFAsOCI — wrap the GGUF blob(s) in a CNCF model-spec OCI manifest
// ---------------------------------------------------------------------------

// modelMeta bundles the optional-but-valuable CNCF model-spec metadata
// buildCNCFManifest has enough information to populate beyond the bare
// config.format+modelfs every manifest already needs — see
// https://github.com/modelpack/model-spec/blob/main/docs/config.md. Both
// fields are entirely optional per the spec's own schema and are simply
// omitted when unknown/inapplicable (Model marshals empty sub-fields as
// omitted JSON via their own `,omitempty` tags).
type modelMeta struct {
	// Format is config.format — "gguf" or "safetensors" here.
	Format string

	// Licenses populates descriptor.licenses as SPDX license
	// expressions — see hfModelInfo.license(). Nil if the source repo
	// didn't declare a usable one.
	Licenses []string

	// Vision marks the model as accepting image input in addition to
	// text (config.capabilities.inputTypes/outputTypes) — set when a
	// multimodal projector layer (see selectMMProj) was actually
	// included among layers. The spec has no separate annotation for
	// "this manifest has an mmproj layer"; capabilities is model-spec's
	// own mechanism for signalling multimodal support.
	Vision bool
}

// storeGGUFAsOCI wraps one or more GGUF layers — several for a multi-part
// split (see selectGGUF/ggufShards), just one otherwise, plus an optional
// mmproj layer (see selectMMProj) — in a CNCF model-spec manifest.
// filepathAnnotation only gets set at the manifest level for the
// single-weight-file case: once there's more than one weight layer
// there's no single filename left to describe the model as a whole,
// matching storeSafetensorsAsOCI's own multi-layer convention (each
// layer's own org.cncf.model.filepath annotation — already set by
// downloadHFBlob — is what actually matters; see streamHFFileToRegistry's
// comment on the same annotation in transfer_docker.go).
func storeGGUFAsOCI(layoutDir, ref, modelRepo string, meta modelMeta, filepathAnnotation string, layers []ocispec.Descriptor) error {
	manifestDesc, err := buildCNCFManifest(layoutBlobSink(layoutDir), meta, modelRepo, filepathAnnotation, layers)
	if err != nil {
		return err
	}
	return writeManifestRef(layoutDir, ref, manifestDesc)
}

// ---------------------------------------------------------------------------
// buildCNCFManifest — shared CNCF model-spec manifest+config construction,
// used by both the local-OCI-layout store path above (storeGGUFAsOCI,
// storeSafetensorsAsOCI) and transfer_docker.go's direct-to-registry push
// path (pushCNCFSingleManifest, pushCNCFMultiManifest). The two paths differ
// only in *where* a built blob ends up — a local content-addressed layout
// vs. streamed straight to a registry pusher — which is exactly what the
// cncfBlobSink abstraction below exists to hide.
// ---------------------------------------------------------------------------

// cncfBlobSink stores one marshaled CNCF blob (config or manifest JSON) and
// returns its descriptor.
type cncfBlobSink func(mediaType string, data []byte) (ocispec.Descriptor, error)

// layoutBlobSink is the cncfBlobSink for storing blobs in a local OCI layout
// directory.
func layoutBlobSink(layoutDir string) cncfBlobSink {
	return func(mediaType string, data []byte) (ocispec.Descriptor, error) {
		return writeBlob(layoutDir, mediaType, data)
	}
}

// buildCNCFManifest builds a conformant CNCF model-spec config blob and
// manifest referencing layers, storing each via sink, and returns the
// manifest's descriptor. meta carries the optional descriptor.licenses/
// config.capabilities metadata (see modelMeta's own doc comment).
// filepathAnnotation sets the manifest-level org.cncf.model.filepath
// annotation for the single-weight-file case (GGUF); pass "" for the
// multi-layer safetensors case, which only sets ai.model.repo.
func buildCNCFManifest(sink cncfBlobSink, meta modelMeta, modelRepo, filepathAnnotation string, layers []ocispec.Descriptor) (ocispec.Descriptor, error) {
	model := modelspec.Model{
		ModelFS: modelspec.ModelFS{Type: "layers"},
		Config:  modelspec.ModelConfig{Format: meta.Format},
	}
	if len(meta.Licenses) > 0 {
		model.Descriptor.Licenses = meta.Licenses
	}
	if meta.Vision {
		model.Config.Capabilities = &modelspec.ModelCapabilities{
			InputTypes:  []modelspec.Modality{modelspec.TextModality, modelspec.ImageModality},
			OutputTypes: []modelspec.Modality{modelspec.TextModality},
		}
	}
	for _, l := range layers {
		model.ModelFS.DiffIDs = append(model.ModelFS.DiffIDs, l.Digest)
	}
	cfgData, err := json.Marshal(model)
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("marshal CNCF model config: %w", err)
	}
	configDesc, err := sink(modelspec.MediaTypeModelConfig, cfgData)
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("store CNCF model config: %w", err)
	}

	annotations := map[string]string{"ai.model.repo": modelRepo}
	if filepathAnnotation != "" {
		annotations[modelspec.AnnotationFilepath] = filepathAnnotation
	}
	manifest := ocispec.Manifest{
		Versioned:    specs.Versioned{SchemaVersion: 2},
		MediaType:    ocispec.MediaTypeImageManifest,
		ArtifactType: modelspec.ArtifactTypeModelManifest,
		Config:       configDesc,
		Layers:       layers,
		Annotations:  annotations,
	}
	manifestData, err := json.Marshal(manifest)
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("marshal CNCF manifest: %w", err)
	}
	manifestDesc, err := sink(ocispec.MediaTypeImageManifest, manifestData)
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("store CNCF manifest: %w", err)
	}
	return manifestDesc, nil
}
