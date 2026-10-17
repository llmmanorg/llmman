package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

func TestFallbackTagTruncatesDigestComponents(t *testing.T) {
	cases := []struct {
		name   string
		digest digest.Digest
		want   string
	}{
		{"sha256 unchanged", digest.Digest("sha256:" + strings.Repeat("a", 64)), "sha256-" + strings.Repeat("a", 64)},
		{"sha512 encoded truncated", digest.Digest("sha512:" + strings.Repeat("b", 128)), "sha512-" + strings.Repeat("b", 64)},
		{"algorithm truncated", digest.Digest(strings.Repeat("c", 40) + ":" + strings.Repeat("d", 64)), strings.Repeat("c", 32) + "-" + strings.Repeat("d", 64)},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			if got := fallbackTag(test.digest); got != test.want {
				t.Fatalf("fallbackTag(%q) = %q, want %q", test.digest, got, test.want)
			}
		})
	}
}

func TestGetReferrersIndexFollowsNextLink(t *testing.T) {
	first := ocispec.Descriptor{Digest: digest.FromString("first")}
	second := ocispec.Descriptor{Digest: digest.FromString("second")}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		page := ocispec.Index{Manifests: []ocispec.Descriptor{first}}
		page.SchemaVersion = 2
		page.MediaType = ocispec.MediaTypeImageIndex
		if r.URL.Query().Get("page") == "2" {
			page.Manifests = []ocispec.Descriptor{second}
		} else {
			w.Header().Set("Link", fmt.Sprintf("<%s?page=2>; rel=\"next\"", r.URL.Path))
		}
		w.Header().Set("Content-Type", ocispec.MediaTypeImageIndex)
		if err := json.NewEncoder(w).Encode(page); err != nil {
			t.Errorf("encode page: %v", err)
		}
	}))
	defer server.Close()

	index, found, err := getReferrersIndex(context.Background(), server.Client(), server.URL+"/v2/org/model/referrers/sha256:subject")
	if err != nil {
		t.Fatalf("get referrers: %v", err)
	}
	if !found {
		t.Fatal("paginated referrers endpoint was not found")
	}
	if len(index.Manifests) != 2 || index.Manifests[0].Digest != first.Digest || index.Manifests[1].Digest != second.Digest {
		t.Fatalf("manifests = %+v, want %s and %s", index.Manifests, first.Digest, second.Digest)
	}
}

func TestParseReferrersIndexAllowsAbsentMediaType(t *testing.T) {
	index, err := parseReferrersIndex([]byte(`{"schemaVersion":2,"manifests":[]}`))
	if err != nil || len(index.Manifests) != 0 {
		t.Fatalf("parse referrers index = %+v, %v", index, err)
	}
}

func TestNextLinkTrimsParameters(t *testing.T) {
	got, err := nextLink("https://registry.example/v2/model/referrers/sha256:abc", []string{`</v2/model/referrers/sha256:abc?page=2>; rel = "next"`})
	if err != nil || got != "https://registry.example/v2/model/referrers/sha256:abc?page=2" {
		t.Fatalf("next link = %q, %v", got, err)
	}
}

func TestRegistryClientRejectsCrossOriginRedirect(t *testing.T) {
	var crossOriginRequests atomic.Int32
	other := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		crossOriginRequests.Add(1)
	}))
	defer other.Close()

	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v2/" {
			w.WriteHeader(http.StatusOK)
			return
		}
		http.Redirect(w, r, other.URL+"/next", http.StatusFound)
	}))
	defer origin.Close()
	u, err := url.Parse(origin.URL)
	if err != nil {
		t.Fatal(err)
	}
	repo, err := repositoryForRef(u.Host + "/org/model:v1")
	if err != nil {
		t.Fatal(err)
	}
	client, err := registryClient(context.Background(), repo)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = getReferrersIndex(context.Background(), client, origin.URL+"/referrers")
	if err == nil || crossOriginRequests.Load() != 0 {
		t.Fatalf("redirect error = %v, cross-origin requests = %d", err, crossOriginRequests.Load())
	}
}

func TestRegistryClientSetupRespectsDeadline(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	blocked := make(chan struct{})
	defer close(blocked)
	_, err := registryClientWithin(ctx, func() (*http.Client, error) {
		<-blocked
		return nil, nil
	})
	if err != context.DeadlineExceeded {
		t.Fatalf("setup error = %v, want deadline exceeded", err)
	}
}

func TestGetReferrersIndexRejectsCrossOriginNextLink(t *testing.T) {
	var crossOriginRequests atomic.Int32
	other := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		crossOriginRequests.Add(1)
	}))
	defer other.Close()

	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		page := ocispec.Index{}
		page.SchemaVersion = 2
		page.MediaType = ocispec.MediaTypeImageIndex
		w.Header().Set("Link", fmt.Sprintf("<%s/next>; rel=\"next\"", other.URL))
		if err := json.NewEncoder(w).Encode(page); err != nil {
			t.Errorf("encode page: %v", err)
		}
	}))
	defer origin.Close()

	_, _, err := getReferrersIndex(context.Background(), origin.Client(), origin.URL+"/referrers")
	if err == nil || !strings.Contains(err.Error(), "leaves registry origin") {
		t.Fatalf("getReferrersIndex error = %v, want cross-origin rejection", err)
	}
	if got := crossOriginRequests.Load(); got != 0 {
		t.Fatalf("cross-origin server received %d requests, want zero", got)
	}
}

func TestSetManifestSubjectPreservesExtensionFields(t *testing.T) {
	dir := t.TempDir()
	ref := "example.invalid/org/artifact:v1"
	if err := ensureLayout(dir); err != nil {
		t.Fatalf("init layout: %v", err)
	}
	raw := []byte(`{"schemaVersion":2,"mediaType":"application/vnd.oci.image.manifest.v1+json","config":{"mediaType":"application/vnd.example.config","digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","size":2},"layers":[],"x-vendor":{"keep":true}}`)
	desc, err := writeBlob(dir, ocispec.MediaTypeImageManifest, raw)
	if err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	if err := writeManifestRef(dir, ref, desc); err != nil {
		t.Fatalf("tag manifest: %v", err)
	}
	subject := ocispec.Descriptor{
		MediaType: ocispec.MediaTypeImageManifest,
		Digest:    digest.FromString("subject"),
		Size:      7,
	}
	updated, err := setManifestSubject(dir, ref, subject)
	if err != nil {
		t.Fatalf("set subject: %v", err)
	}
	got, err := readBlob(dir, updated.Digest)
	if err != nil {
		t.Fatalf("read updated manifest: %v", err)
	}
	var document map[string]json.RawMessage
	if err := json.Unmarshal(got, &document); err != nil {
		t.Fatalf("parse updated manifest: %v", err)
	}
	if string(document["x-vendor"]) != `{"keep":true}` {
		t.Fatalf("extension field changed or disappeared: %s", document["x-vendor"])
	}
}
