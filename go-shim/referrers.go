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

	"github.com/google/go-containerregistry/pkg/name"
	"github.com/google/go-containerregistry/pkg/v1/remote/transport"
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

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	artifact, err := setManifestSubject(C.GoString(cLayoutDir), ref, subject)
	if err != nil {
		return errResp(err)
	}
	if _, err := pushDescriptorToRegistry(ctx, C.GoString(cLayoutDir), ref, artifact); err != nil {
		return errResp(err)
	}
	if err := publishFallbackReferrer(ctx, ref, subject.Digest, artifact); err != nil {
		return errResp(fmt.Errorf("publish referrer index: %w", err))
	}
	return okResp("")
}

// llmman_referrers lists the descriptors attached to digest in ref's
// repository. It uses the OCI 1.1 endpoint first and falls back to the
// sha256-<hex> tag only when the registry reports that endpoint absent.
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

func repositoryForRef(ref string) (name.Repository, error) {
	repo, err := name.NewRepository(repositoryOf(ref), name.WeakValidation)
	if err != nil {
		return name.Repository{}, fmt.Errorf("parse repository %q: %w", repositoryOf(ref), err)
	}
	return repo, nil
}

func registryClient(ctx context.Context, repo name.Repository) (*http.Client, error) {
	base, err := registryBaseTransport(ctx, repo)
	if err != nil {
		return nil, fmt.Errorf("load registry transport: %w", err)
	}
	auth, err := registryAuthenticator(ctx, repo)
	if err != nil {
		return nil, fmt.Errorf("load registry credentials: %w", err)
	}
	scopes := []string{repo.Scope(transport.PullScope)}
	origin := &url.URL{Scheme: repo.Scheme(), Host: repo.RegistryStr()}
	roundTripper, err := transport.NewWithContext(ctx, repo.Registry, auth, base, scopes)
	if err != nil {
		return nil, fmt.Errorf("configure registry transport: %w", err)
	}
	return &http.Client{
		Transport: &registryOriginTransport{origin: origin, next: roundTripper},
		Timeout:   120 * time.Second,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 10 || !sameRegistryOrigin(origin, req.URL) {
				return fmt.Errorf("referrers redirect leaves registry origin: %s", req.URL)
			}
			return nil
		},
	}, nil
}

// Restrict the referrers client, not the transport used inside bearer-token
// setup: a registry may legitimately delegate authentication to another host.
type registryOriginTransport struct {
	origin *url.URL
	next   http.RoundTripper
}

func (t *registryOriginTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if !sameRegistryOrigin(t.origin, req.URL) {
		return nil, fmt.Errorf("referrers request leaves registry origin: %s", req.URL)
	}
	return t.next.RoundTrip(req)
}

// Credential helpers and transport setup may block before the first HTTP
// request. Bound that phase as well as the requests made by the client.
func registryClientWithin(ctx context.Context, create func() (*http.Client, error)) (*http.Client, error) {
	type result struct {
		client *http.Client
		err    error
	}
	ready := make(chan result)
	go func() {
		client, err := create()
		select {
		case ready <- result{client, err}:
		case <-ctx.Done():
			if client != nil {
				client.CloseIdleConnections()
			}
		}
	}()
	select {
	case got := <-ready:
		return got.client, got.err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func sameRegistryOrigin(origin, target *url.URL) bool {
	return target.User == nil && strings.EqualFold(target.Scheme, origin.Scheme) && strings.EqualFold(target.Host, origin.Host)
}

func registryURL(repo name.Repository, elems ...string) string {
	path := "/v2/" + repo.RepositoryStr() + "/" + strings.Join(elems, "/")
	return (&url.URL{Scheme: repo.Scheme(), Host: repo.RegistryStr(), Path: path}).String()
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

		req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
		if err != nil {
			return nil, false, err
		}
		req.Header.Set("Accept", ocispec.MediaTypeImageIndex)
		resp, err := client.Do(req)
		if err != nil {
			return nil, false, err
		}
		if resp.StatusCode == http.StatusNotFound && combined == nil {
			resp.Body.Close()
			return nil, false, nil
		}
		if err := transport.CheckError(resp, http.StatusOK); err != nil {
			resp.Body.Close()
			return nil, false, err
		}
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
		next, err := nextLink(endpoint, resp.Header.Values("Link"))
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
	index, native, err := fetchReferrersIndex(ctx, ref, subject)
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
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	repo, err := repositoryForRef(ref)
	if err != nil {
		return ocispec.Index{}, false, err
	}
	client, err := registryClientWithin(ctx, func() (*http.Client, error) {
		return registryClient(ctx, repo)
	})
	if err != nil {
		return ocispec.Index{}, false, err
	}
	index, found, err := getReferrersIndex(ctx, client, registryURL(repo, "referrers", subject.String()))
	if err != nil {
		return ocispec.Index{}, false, fmt.Errorf("list referrers: %w", err)
	}
	if !found {
		return ocispec.Index{}, false, nil
	}
	return *index, true, nil
}

func fetchFallbackReferrersIndex(ctx context.Context, ref string, subject digest.Digest) (ocispec.Index, error) {
	repo, err := repositoryForRef(ref)
	if err != nil {
		return ocispec.Index{}, err
	}
	fallbackRef := repo.String() + ":" + fallbackTag(subject)
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
	_, native, err := fetchNativeReferrersIndex(ctx, ref, subject)
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
	repo, err := repositoryForRef(ref)
	if err != nil {
		return false, err
	}
	fallbackRef := repo.String() + ":" + fallbackTag(subject)
	if err := pushStagedManifest(ctx, staging, fallbackRef, ocispec.MediaTypeImageIndex, raw, pushIndexToRegistry); err != nil {
		return false, err
	}
	return true, nil
}
