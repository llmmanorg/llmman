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

func TestSplitRepositoryAcceptsLocalhost(t *testing.T) {
	registry, path, err := splitRepository("localhost/org/model:latest")
	if err != nil || registry != "localhost" || path != "org/model" {
		t.Fatalf("splitRepository = %q, %q, %v", registry, path, err)
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

func TestGetReferrersIndexResolvesLinkFromRedirectDestination(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/referrers":
			http.Redirect(w, r, "/pages/first", http.StatusFound)
		case "/pages/first":
			if r.URL.Query().Get("page") != "2" {
				w.Header().Set("Link", `<?page=2>; rel="next"`)
			}
			_, _ = w.Write([]byte(`{"schemaVersion":2,"manifests":[]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	_, found, err := getReferrersIndex(context.Background(), server.Client(), server.URL+"/referrers")
	if err != nil || !found {
		t.Fatalf("get referrers after redirect = %t, %v", found, err)
	}
}

func TestGetReferrersIndexLimitsPages(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		page := requests.Add(1)
		w.Header().Set("Link", fmt.Sprintf("<?page=%d>; rel=next", page))
		_, _ = w.Write([]byte(`{"schemaVersion":2,"manifests":[]}`))
	}))
	defer server.Close()

	_, _, err := getReferrersIndex(context.Background(), server.Client(), server.URL+"/referrers")
	if err == nil || !strings.Contains(err.Error(), "pages") {
		t.Fatalf("pagination error = %v, want page limit", err)
	}
	if got := requests.Load(); got != maxReferrersPages {
		t.Fatalf("requested %d pages, want %d", got, maxReferrersPages)
	}
}

func TestGetReferrersIndexLimitsDescriptors(t *testing.T) {
	page := ocispec.Index{Manifests: make([]ocispec.Descriptor, maxReferrersDescriptors+1)}
	page.SchemaVersion = 2
	raw, err := json.Marshal(page)
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) > maxReferrersIndexBytes {
		t.Fatalf("test page exceeds per-page byte limit: %d", len(raw))
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(raw)
	}))
	defer server.Close()

	_, _, err = getReferrersIndex(context.Background(), server.Client(), server.URL+"/referrers")
	if err == nil || !strings.Contains(err.Error(), "descriptors") {
		t.Fatalf("pagination error = %v, want descriptor limit", err)
	}
}

func TestProbeReferrersEndpointStopsAtFirstResponse(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.Header().Set("Link", `<?page=2>; rel=next`)
		_, _ = w.Write([]byte(`not an index`))
	}))
	defer server.Close()
	found, err := probeReferrersEndpoint(context.Background(), server.Client(), server.URL+"/referrers")
	if err != nil || !found || requests.Load() != 1 {
		t.Fatalf("probe = %t, %v, requests = %d; want first response only", found, err, requests.Load())
	}
}

func TestProbeReferrersEndpointDistinguishes404FromOtherErrors(t *testing.T) {
	status := http.StatusNotFound
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(status)
	}))
	defer server.Close()
	found, err := probeReferrersEndpoint(context.Background(), server.Client(), server.URL+"/referrers")
	if err != nil || found {
		t.Fatalf("404 probe = %t, %v; want unsupported", found, err)
	}
	status = http.StatusForbidden
	_, err = probeReferrersEndpoint(context.Background(), server.Client(), server.URL+"/referrers")
	if err == nil {
		t.Fatal("403 must not select fallback publication")
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

// The scheme probe and the bounded client setup moved into each backend's own
// transport (containerd's RegistryHost for Docker), so what referrers.go still
// owns is the origin guard: these cover it on both a redirect and a next link.
func TestOriginBoundClientRejectsCrossOriginRedirect(t *testing.T) {
	var crossOriginRequests atomic.Int32
	other := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		crossOriginRequests.Add(1)
	}))
	defer other.Close()

	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, other.URL+"/next", http.StatusFound)
	}))
	defer origin.Close()
	u, err := url.Parse(origin.URL)
	if err != nil {
		t.Fatal(err)
	}

	client := originBoundClient(&url.URL{Scheme: u.Scheme, Host: u.Host}, origin.Client().Transport)
	_, _, err = getReferrersIndex(context.Background(), client, origin.URL+"/referrers")
	if err == nil || crossOriginRequests.Load() != 0 {
		t.Fatalf("redirect error = %v, cross-origin requests = %d", err, crossOriginRequests.Load())
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

	u, err := url.Parse(origin.URL)
	if err != nil {
		t.Fatal(err)
	}
	client := originBoundClient(&url.URL{Scheme: u.Scheme, Host: u.Host}, origin.Client().Transport)
	_, _, err = getReferrersIndex(context.Background(), client, origin.URL+"/referrers")
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
