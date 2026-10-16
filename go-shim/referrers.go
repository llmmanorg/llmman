// OCI 1.1 subject/referrers support shared by both registry backends.
package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"

	"github.com/google/go-containerregistry/pkg/name"
	"github.com/google/go-containerregistry/pkg/v1/remote"
	"github.com/google/go-containerregistry/pkg/v1/remote/transport"
	digest "github.com/opencontainers/go-digest"
	specs "github.com/opencontainers/image-spec/specs-go"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

const maxReferrersIndexBytes = 4 << 20

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

	ctx := context.Background()
	artifact, err := setManifestSubject(C.GoString(cLayoutDir), ref, subject)
	if err != nil {
		return errResp(err)
	}
	if _, err := pushToRegistry(ctx, C.GoString(cLayoutDir), ref); err != nil {
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
	refs, _, err := fetchReferrers(context.Background(), C.GoString(cRef), dgst)
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

func registryClient(ctx context.Context, repo name.Repository, push bool) (*http.Client, error) {
	auth, err := registryAuthenticator(ctx, repo)
	if err != nil {
		return nil, fmt.Errorf("load registry credentials: %w", err)
	}
	scopes := []string{repo.Scope(transport.PullScope)}
	if push {
		scopes = append(scopes, repo.Scope(transport.PushScope))
	}
	roundTripper, err := transport.NewWithContext(ctx, repo.Registry, auth, remote.DefaultTransport, scopes)
	if err != nil {
		return nil, fmt.Errorf("configure registry transport: %w", err)
	}
	return &http.Client{Transport: roundTripper, Timeout: 120 * time.Second}, nil
}

func registryURL(repo name.Repository, elems ...string) string {
	path := "/v2/" + repo.RepositoryStr() + "/" + strings.Join(elems, "/")
	return (&url.URL{Scheme: repo.Scheme(), Host: repo.RegistryStr(), Path: path}).String()
}

func getReferrersIndex(ctx context.Context, client *http.Client, endpoint string) (*ocispec.Index, bool, error) {
	var combined *ocispec.Index
	seen := map[string]bool{}
	for endpoint != "" {
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
		var page ocispec.Index
		if err := json.Unmarshal(raw, &page); err != nil {
			return nil, false, fmt.Errorf("parse referrers index: %w", err)
		}
		if page.SchemaVersion != 2 || page.MediaType != ocispec.MediaTypeImageIndex {
			return nil, false, fmt.Errorf(
				"invalid referrers index: schemaVersion=%d mediaType=%q",
				page.SchemaVersion, page.MediaType,
			)
		}
		if combined == nil {
			combined = &page
		} else {
			combined.Manifests = append(combined.Manifests, page.Manifests...)
		}
		endpoint, err = nextLink(endpoint, resp.Header.Values("Link"))
		if err != nil {
			return nil, false, err
		}
	}
	return combined, true, nil
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
	return strings.Replace(dgst.String(), ":", "-", 1)
}

// fetchReferrers returns (descriptors, nativeAPI, error). nativeAPI is false
// only for a 404 from the OCI 1.1 endpoint, the signal that selects the
// distribution-spec fallback tag scheme.
func fetchReferrers(ctx context.Context, ref string, subject digest.Digest) ([]ocispec.Descriptor, bool, error) {
	index, native, err := fetchReferrersIndex(ctx, ref, subject)
	return index.Manifests, native, err
}

func fetchReferrersIndex(ctx context.Context, ref string, subject digest.Digest) (ocispec.Index, bool, error) {
	repo, err := repositoryForRef(ref)
	if err != nil {
		return ocispec.Index{}, false, err
	}
	client, err := registryClient(ctx, repo, false)
	if err != nil {
		return ocispec.Index{}, false, err
	}
	index, found, err := getReferrersIndex(ctx, client, registryURL(repo, "referrers", subject.String()))
	if err != nil {
		return ocispec.Index{}, false, fmt.Errorf("list referrers: %w", err)
	}
	if found {
		return *index, true, nil
	}
	index, found, err = getReferrersIndex(ctx, client, registryURL(repo, "manifests", fallbackTag(subject)))
	if err != nil {
		return ocispec.Index{}, false, fmt.Errorf("read fallback referrers index: %w", err)
	}
	if !found {
		empty := ocispec.Index{}
		empty.Versioned = specs.Versioned{SchemaVersion: 2}
		empty.MediaType = ocispec.MediaTypeImageIndex
		empty.Manifests = []ocispec.Descriptor{}
		return empty, false, nil
	}
	return *index, false, nil
}

func publishFallbackReferrer(ctx context.Context, ref string, subject digest.Digest, artifact ocispec.Descriptor) error {
	index, native, err := fetchReferrersIndex(ctx, ref, subject)
	if err != nil || native {
		return err
	}
	for _, existing := range index.Manifests {
		if existing.Digest == artifact.Digest {
			return nil
		}
	}
	annotations := make(map[string]string, len(artifact.Annotations))
	for key, value := range artifact.Annotations {
		annotations[key] = value
	}
	delete(annotations, ocispec.AnnotationRefName) // local ref-name annotations do not belong in the remote index
	artifact.Annotations = annotations
	index.Manifests = append(index.Manifests, artifact)
	sort.Slice(index.Manifests, func(i, j int) bool {
		return index.Manifests[i].Digest.String() < index.Manifests[j].Digest.String()
	})
	raw, err := json.Marshal(index)
	if err != nil {
		return err
	}

	repo, err := repositoryForRef(ref)
	if err != nil {
		return err
	}
	client, err := registryClient(ctx, repo, true)
	if err != nil {
		return err
	}
	endpoint := registryURL(repo, "manifests", fallbackTag(subject))
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, endpoint, bytes.NewReader(raw))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", ocispec.MediaTypeImageIndex)
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return transport.CheckError(resp, http.StatusOK, http.StatusCreated, http.StatusAccepted)
}
