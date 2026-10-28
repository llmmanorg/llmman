// OCI 1.1 subject/referrers support shared by both registry backends.
package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	digest "github.com/opencontainers/go-digest"
	specs "github.com/opencontainers/image-spec/specs-go"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

const (
	maxReferrersIndexBytes  = 4 << 20
	maxReferrersPages       = 64
	maxReferrersDescriptors = 10000
	maxReferrersTotalBytes  = 16 << 20
	fallbackPublishAttempts = 3
)

// OCI does not require registries to support conditional tag updates, so the
// fallback index's read-append-PUT sequence must be serialized in-process.
var fallbackPublishMu sync.Mutex

// llmman_push_artifact sets subject on the local manifest and pushes it. A
// registry with the OCI 1.1 referrers API indexes the manifest itself; for an
// older registry we maintain the standard digest-derived fallback index.
//
//export llmman_push_artifact
func llmman_push_artifact(cLayoutDir, cRef, cSubjectJSON *C.char) *C.char {
	ref := C.GoString(cRef)
	progressReset(ref, "retrieving manifest")
	defer progressDone(ref)

	var subject ocispec.Descriptor
	if err := json.Unmarshal([]byte(C.GoString(cSubjectJSON)), &subject); err != nil {
		return errResp(fmt.Errorf("parse subject descriptor: %w", err))
	}
	if err := validateSubject(subject); err != nil {
		return errResp(err)
	}

	artifact, err := setManifestSubject(C.GoString(cLayoutDir), ref, subject)
	if err != nil {
		return errResp(err)
	}
	if _, err := pushDescriptorToRegistry(context.Background(), C.GoString(cLayoutDir), ref, artifact); err != nil {
		return errResp(err)
	}
	if err := publishFallbackReferrer(context.Background(), ref, subject.Digest, artifact); err != nil {
		return errResp(fmt.Errorf("publish referrer index: %w", err))
	}
	return okResp("")
}

// llmman_referrers lists the descriptors attached to digest in ref's
// repository. Docker uses the OCI 1.1 endpoint first and falls back to the
// digest-derived tag only when the registry reports that endpoint absent.
// Podman always reads that tag because its backend has no public authenticated
// native-referrers GET; artifact publication writes the tag on both backends
// when the native endpoint cannot be checked or is absent.
//
//export llmman_referrers
func llmman_referrers(cRef, cDigest *C.char) *C.char {
	dgst, err := digest.Parse(C.GoString(cDigest))
	if err != nil {
		return errResp(fmt.Errorf("parse subject digest: %w", err))
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	refs, _, err := fetchReferrers(ctx, C.GoString(cRef), dgst)
	if err != nil {
		return errResp(err)
	}
	data, err := json.Marshal(refs)
	if err != nil {
		return errResp(fmt.Errorf("encode referrers: %w", err))
	}
	return okResp(string(data))
}

func validateSubject(subject ocispec.Descriptor) error {
	if subject.MediaType == "" {
		return errors.New("subject descriptor has no media type")
	}
	if err := subject.Digest.Validate(); err != nil {
		return fmt.Errorf("invalid subject digest: %w", err)
	}
	if subject.Size < 0 {
		return errors.New("subject descriptor has a negative size")
	}
	return nil
}

// setManifestSubject rewrites only the artifact manifest. Its config and layer
// blobs remain content-addressed and unchanged. The reference is atomically
// advanced to the new manifest descriptor so the ordinary backend push path
// can upload it without learning a second way to read local OCI layouts.
func setManifestSubject(layoutDir, ref string, subject ocispec.Descriptor) (ocispec.Descriptor, error) {
	desc, err := findManifestForPush(layoutDir, ref)
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	raw, err := readBlob(layoutDir, desc.Digest)
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("read artifact manifest: %w", err)
	}
	var manifest ocispec.Manifest
	if err := json.Unmarshal(raw, &manifest); err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("parse artifact manifest: %w", err)
	}
	if manifest.Config.Digest == "" {
		return ocispec.Descriptor{}, errors.New("artifact is not an OCI image manifest")
	}
	// Add the field to the original JSON object instead of serializing the
	// typed Manifest back out. OCI extensions are allowed to add fields a
	// version of image-spec does not know yet; setting subject must not erase
	// those fields from a user's artifact.
	var document map[string]json.RawMessage
	if err := json.Unmarshal(raw, &document); err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("parse artifact document: %w", err)
	}
	subjectJSON, err := json.Marshal(subject)
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("encode subject descriptor: %w", err)
	}
	document["subject"] = subjectJSON
	raw, err = json.Marshal(document)
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("encode artifact manifest: %w", err)
	}
	updated, err := writeBlob(layoutDir, desc.MediaType, raw)
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("write artifact manifest: %w", err)
	}
	updated.Annotations = manifest.Annotations
	updated.ArtifactType = manifest.ArtifactType
	if updated.ArtifactType == "" {
		updated.ArtifactType = manifest.Config.MediaType
	}
	if err := writeManifestRef(layoutDir, ref, updated); err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("update artifact reference: %w", err)
	}
	return updated, nil
}

// splitRepository separates "registry/path" for a fully qualified reference.
// llmman always resolves short names before they reach the shim, so a missing
// registry is a caller error rather than an implied Docker Hub.
func splitRepository(ref string) (registry, path string, err error) {
	repo := repositoryOf(ref)
	registry, path, ok := strings.Cut(repo, "/")
	if !ok || registry == "" || path == "" || (registry != "localhost" && !strings.ContainsAny(registry, ".:")) {
		return "", "", fmt.Errorf("reference %q has no registry component", ref)
	}
	return registry, path, nil
}

func sameRegistryOrigin(origin, target *url.URL) bool {
	return strings.EqualFold(origin.Scheme, target.Scheme) && strings.EqualFold(origin.Host, target.Host)
}

// originTransport keeps a paginated read inside the registry it started from.
// A Link header is registry-controlled, so a cross-origin next page would let
// one registry put descriptors it does not host into another's referrers list.
type originTransport struct {
	origin *url.URL
	next   http.RoundTripper
}

func (t *originTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if !sameRegistryOrigin(t.origin, req.URL) {
		return nil, fmt.Errorf("referrers request leaves registry origin: %s", req.URL)
	}
	return t.next.RoundTrip(req)
}

// originBoundClient refuses both a next-page link and a redirect that leave
// the registry the read started from.
func originBoundClient(origin *url.URL, rt http.RoundTripper) *http.Client {
	return &http.Client{
		Transport: &originTransport{origin: origin, next: rt},
		Timeout:   120 * time.Second,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 10 || !sameRegistryOrigin(origin, req.URL) {
				return fmt.Errorf("referrers redirect leaves registry origin: %s", req.URL)
			}
			return nil
		},
	}
}

func getReferrersIndex(ctx context.Context, client *http.Client, endpoint string) (*ocispec.Index, bool, error) {
	origin, err := url.Parse(endpoint)
	if err != nil || origin.Scheme == "" || origin.Host == "" {
		return nil, false, fmt.Errorf("invalid referrers endpoint %q", endpoint)
	}
	var combined *ocispec.Index
	seen := map[string]bool{}
	totalBytes, totalDescriptors := 0, 0
	for endpoint != "" {
		if len(seen) >= maxReferrersPages {
			return nil, false, fmt.Errorf("referrers pagination exceeds %d pages", maxReferrersPages)
		}
		if seen[endpoint] {
			return nil, false, fmt.Errorf("referrers pagination cycle at %q", endpoint)
		}
		seen[endpoint] = true

		resp, found, err := requestReferrersPage(ctx, client, endpoint)
		if err != nil {
			return nil, false, err
		}
		if !found && combined == nil {
			return nil, false, nil
		}
		if !found {
			return nil, false, fmt.Errorf("referrers pagination endpoint %q returned 404", endpoint)
		}
		responseURL := resp.Request.URL.String()
		linkHeaders := resp.Header.Values("Link")
		raw, err := io.ReadAll(io.LimitReader(resp.Body, maxReferrersIndexBytes+1))
		resp.Body.Close()
		if err != nil {
			return nil, false, err
		}
		if len(raw) > maxReferrersIndexBytes {
			return nil, false, fmt.Errorf("referrers index is over the %d-byte limit", maxReferrersIndexBytes)
		}
		totalBytes += len(raw)
		if totalBytes > maxReferrersTotalBytes {
			return nil, false, fmt.Errorf("referrers pagination exceeds %d bytes", maxReferrersTotalBytes)
		}
		page, err := parseReferrersIndex(raw)
		if err != nil {
			return nil, false, err
		}
		totalDescriptors += len(page.Manifests)
		if totalDescriptors > maxReferrersDescriptors {
			return nil, false, fmt.Errorf("referrers pagination exceeds %d descriptors", maxReferrersDescriptors)
		}
		if combined == nil {
			combined = page
		} else {
			combined.Manifests = append(combined.Manifests, page.Manifests...)
		}
		next, err := nextLink(responseURL, linkHeaders)
		if err != nil {
			return nil, false, err
		}
		if next != "" {
			nextURL, parseErr := url.Parse(next)
			if parseErr != nil || !sameRegistryOrigin(origin, nextURL) {
				return nil, false, fmt.Errorf("referrers next link %q leaves registry origin", next)
			}
		}
		endpoint = next
	}
	return combined, true, nil
}

// checkRegistryStatus turns a non-200 referrers response into an error that
// carries a bounded excerpt of the registry's body, so a 401 or a 429 is
// distinguishable without streaming whatever the registry chose to send.
func checkRegistryStatus(resp *http.Response) error {
	if resp.StatusCode == http.StatusOK {
		return nil
	}
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	detail := strings.TrimSpace(string(body))
	if detail == "" {
		return fmt.Errorf("registry responded %s", resp.Status)
	}
	return fmt.Errorf("registry responded %s: %s", resp.Status, detail)
}

func parseReferrersIndex(raw []byte) (*ocispec.Index, error) {
	var index ocispec.Index
	if err := json.Unmarshal(raw, &index); err != nil {
		return nil, fmt.Errorf("parse referrers index: %w", err)
	}
	if index.SchemaVersion != 2 || (index.MediaType != "" && index.MediaType != ocispec.MediaTypeImageIndex) {
		return nil, fmt.Errorf("invalid referrers index: schemaVersion=%d mediaType=%q", index.SchemaVersion, index.MediaType)
	}
	return &index, nil
}

func nextLink(base string, headers []string) (string, error) {
	for _, header := range headers {
		for _, entry := range splitLinkHeader(header) {
			end := strings.IndexByte(entry, '>')
			if !strings.HasPrefix(entry, "<") || end < 0 {
				continue
			}
			isNext := false
			for _, param := range strings.Split(entry[end+1:], ";") {
				key, value, ok := strings.Cut(strings.TrimSpace(param), "=")
				key, value = strings.TrimSpace(key), strings.TrimSpace(value)
				if ok && strings.EqualFold(key, "rel") {
					for _, rel := range strings.Fields(strings.Trim(value, `"`)) {
						isNext = isNext || strings.EqualFold(rel, "next")
					}
				}
			}
			if !isNext {
				continue
			}
			baseURL, err := url.Parse(base)
			if err != nil {
				return "", err
			}
			target, err := url.Parse(entry[1:end])
			if err != nil {
				return "", fmt.Errorf("parse referrers next link: %w", err)
			}
			return baseURL.ResolveReference(target).String(), nil
		}
	}
	return "", nil
}

func splitLinkHeader(header string) []string {
	var entries []string
	start := 0
	inAngle, inQuote := false, false
	for i, char := range header {
		switch char {
		case '<':
			if !inQuote {
				inAngle = true
			}
		case '>':
			if !inQuote {
				inAngle = false
			}
		case '"':
			if !inAngle {
				inQuote = !inQuote
			}
		case ',':
			if !inAngle && !inQuote {
				entries = append(entries, strings.TrimSpace(header[start:i]))
				start = i + 1
			}
		}
	}
	return append(entries, strings.TrimSpace(header[start:]))
}

func fallbackTag(dgst digest.Digest) string {
	algorithm := dgst.Algorithm().String()
	if len(algorithm) > 32 {
		algorithm = algorithm[:32]
	}
	encoded := dgst.Encoded()
	if len(encoded) > 64 {
		encoded = encoded[:64]
	}
	return algorithm + "-" + encoded
}

// fetchReferrers returns (descriptors, nativeAPI, error). nativeAPI is false
// only for a 404 from the OCI 1.1 endpoint, the signal that selects the
// distribution-spec fallback tag scheme.
func fetchReferrers(ctx context.Context, ref string, subject digest.Digest) ([]ocispec.Descriptor, bool, error) {
	// Like publication: a mirror is not authoritative, and a stale one would
	// hide a freshly published referrer.
	index, native, err := fetchReferrersIndex(withoutMirrors(ctx), ref, subject)
	if index.Manifests == nil {
		index.Manifests = []ocispec.Descriptor{}
	}
	return index.Manifests, native, err
}

func fetchReferrersIndex(ctx context.Context, ref string, subject digest.Digest) (ocispec.Index, bool, error) {
	index, found, err := fetchNativeReferrersIndex(ctx, ref, subject)
	if err != nil || found {
		return index, found, err
	}
	index, err = fetchFallbackReferrersIndex(ctx, ref, subject)
	return index, false, err
}

func fetchNativeReferrersIndex(ctx context.Context, ref string, subject digest.Digest) (ocispec.Index, bool, error) {
	return readNativeReferrers(ctx, ref, subject, false)
}

func probeNativeReferrers(ctx context.Context, ref string, subject digest.Digest) (bool, error) {
	_, found, err := readNativeReferrers(ctx, ref, subject, true)
	return found, err
}

func readNativeReferrers(ctx context.Context, ref string, subject digest.Digest, probeOnly bool) (ocispec.Index, bool, error) {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	rt, root, supported, err := nativeReferrersTransport(ctx, ref)
	if err != nil || !supported {
		return ocispec.Index{}, false, err
	}
	_, path, err := splitRepository(ref)
	if err != nil {
		return ocispec.Index{}, false, err
	}
	client := originBoundClient(&url.URL{Scheme: root.Scheme, Host: root.Host}, rt)
	defer client.CloseIdleConnections()
	endpoint := root.JoinPath(path, "referrers", subject.String()).String()
	if probeOnly {
		found, err := probeReferrersEndpoint(ctx, client, endpoint)
		return ocispec.Index{}, found, err
	}
	index, found, err := getReferrersIndex(ctx, client, endpoint)
	if err != nil {
		return ocispec.Index{}, false, fmt.Errorf("list referrers: %w", err)
	}
	if !found {
		return ocispec.Index{}, false, nil
	}
	return *index, true, nil
}

// requestReferrersPage shares the authenticated GET and status handling used
// by both listing and the bounded capability probe. A 404 selects fallback.
func requestReferrersPage(ctx context.Context, client *http.Client, endpoint string) (*http.Response, bool, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, false, err
	}
	req.Header.Set("Accept", ocispec.MediaTypeImageIndex)
	resp, err := client.Do(req)
	if err != nil {
		return nil, false, err
	}
	if resp.StatusCode == http.StatusNotFound {
		resp.Body.Close()
		return nil, false, nil
	}
	if err := checkRegistryStatus(resp); err != nil {
		resp.Body.Close()
		return nil, false, err
	}
	return resp, true, nil
}

// Publication needs only the endpoint's 200/404 capability signal, not every
// descriptor page. Do not read a potentially large response body here.
func probeReferrersEndpoint(ctx context.Context, client *http.Client, endpoint string) (bool, error) {
	resp, found, err := requestReferrersPage(ctx, client, endpoint)
	if resp != nil {
		resp.Body.Close()
	}
	if err != nil {
		return false, err
	}
	return found, nil
}

func fetchFallbackReferrersIndex(ctx context.Context, ref string, subject digest.Digest) (ocispec.Index, error) {
	fallbackRef := repositoryOf(ref) + ":" + fallbackTag(subject)
	raw, err := fetchManifestRaw(ctx, fallbackRef)
	if err != nil && isNotFoundError(err) {
		empty := ocispec.Index{}
		empty.Versioned = specs.Versioned{SchemaVersion: 2}
		empty.MediaType = ocispec.MediaTypeImageIndex
		empty.Manifests = []ocispec.Descriptor{}
		return empty, nil
	}
	if err != nil {
		return ocispec.Index{}, fmt.Errorf("read fallback referrers index: %w", err)
	}
	index, err := parseReferrersIndex(raw)
	if err != nil {
		return ocispec.Index{}, err
	}
	return *index, nil
}

func publishFallbackReferrer(ctx context.Context, ref string, subject digest.Digest, artifact ocispec.Descriptor) error {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	ctx = withoutMirrors(ctx)
	native, err := probeNativeReferrers(ctx, ref, subject)
	if err != nil || native {
		return err
	}

	fallbackPublishMu.Lock()
	defer fallbackPublishMu.Unlock()

	for attempt := 0; attempt < fallbackPublishAttempts; attempt++ {
		pushed, err := appendFallbackReferrer(ctx, ref, subject, artifact)
		if err != nil {
			return err
		}
		if !pushed {
			return nil
		}
		landed, err := fetchFallbackReferrersIndex(ctx, ref, subject)
		if err != nil {
			return fmt.Errorf("confirm published referrer: %w", err)
		}
		for _, entry := range landed.Manifests {
			if entry.Digest == artifact.Digest {
				return nil
			}
		}
	}
	return fmt.Errorf("referrer %s kept being overwritten by another publisher after %d attempts", artifact.Digest, fallbackPublishAttempts)
}

// appendFallbackReferrer performs one read-append-PUT of the digest-derived
// fallback tag. The caller confirms that its descriptor actually landed.
func appendFallbackReferrer(ctx context.Context, ref string, subject digest.Digest, artifact ocispec.Descriptor) (bool, error) {
	index, err := fetchFallbackReferrersIndex(ctx, ref, subject)
	if err != nil {
		return false, err
	}
	for _, existing := range index.Manifests {
		if existing.Digest == artifact.Digest {
			return false, nil
		}
	}
	index.Manifests = append(index.Manifests, artifact)
	sort.Slice(index.Manifests, func(i, j int) bool {
		return index.Manifests[i].Digest.String() < index.Manifests[j].Digest.String()
	})
	raw, err := json.Marshal(index)
	if err != nil {
		return false, err
	}
	if len(raw) > maxManifestBytes {
		return false, fmt.Errorf("fallback referrers index is over the %d-byte limit", maxManifestBytes)
	}
	staging, err := os.MkdirTemp("", "llmman-referrers-")
	if err != nil {
		return false, err
	}
	defer os.RemoveAll(staging)
	if err := ensureLayout(staging); err != nil {
		return false, err
	}
	fallbackRef := repositoryOf(ref) + ":" + fallbackTag(subject)
	if err := pushStagedManifest(ctx, staging, fallbackRef, ocispec.MediaTypeImageIndex, raw, pushIndexToRegistry); err != nil {
		return false, err
	}
	return true, nil
}
