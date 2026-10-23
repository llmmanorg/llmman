//go:build podman

package main

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/google/go-containerregistry/pkg/registry"
	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"go.podman.io/image/v5/transports/alltransports"
	"go.podman.io/image/v5/types"
)

// Podman has no native referrers path: go.podman.io/image exposes no
// authenticated GET, so the backend says so and every read falls through to
// the fallback tag, which fetchManifestRaw and pushToRegistry already serve.
func TestPodmanReportsNoNativeReferrersTransport(t *testing.T) {
	rt, root, supported, err := nativeReferrersTransport(context.Background(), "example.test/org/model:v1")
	if err != nil || supported || rt != nil || root != nil {
		t.Fatalf("nativeReferrersTransport = (%v, %v, %t, %v), want unsupported", rt, root, supported, err)
	}
}

func TestPodmanNativeReadFallsThroughToTag(t *testing.T) {
	index, native, err := readNativeReferrers(context.Background(), "example.test/org/model:v1",
		digest.FromString("subject"), false)
	if err != nil || native || len(index.Manifests) != 0 {
		t.Fatalf("readNativeReferrers = (%+v, %t, %v), want an unused native path", index, native, err)
	}
}

func TestPodmanArtifactPushRefusesManifestDigestRewrite(t *testing.T) {
	var ociPuts atomic.Int32
	var convertedPuts atomic.Int32
	reg := registry.New(registry.Logger(log.New(io.Discard, "", 0)))
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPut && strings.Contains(r.URL.Path, "/manifests/") {
			if r.Header.Get("Content-Type") == ocispec.MediaTypeImageManifest {
				ociPuts.Add(1)
				http.Error(w, "OCI manifests rejected for test", http.StatusUnsupportedMediaType)
				return
			}
			convertedPuts.Add(1)
		}
		reg.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}

	ref := u.Host + "/org/model:sbom"
	dir := t.TempDir()
	if err := ensureLayout(dir); err != nil {
		t.Fatal(err)
	}
	config, err := writeBlob(dir, "application/vnd.example.sbom.config.v1+json", []byte(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	layer, err := writeBlob(dir, "application/vnd.example.sbom.v1+json", []byte(`{"name":"test"}`))
	if err != nil {
		t.Fatal(err)
	}
	manifest := ocispec.Manifest{
		MediaType:    ocispec.MediaTypeImageManifest,
		ArtifactType: "application/vnd.example.sbom.v1",
		Config:       config,
		Layers:       []ocispec.Descriptor{layer},
		Subject: &ocispec.Descriptor{
			MediaType: ocispec.MediaTypeImageManifest,
			Digest:    digest.FromString("subject"),
			Size:      123,
		},
	}
	manifest.SchemaVersion = 2
	raw, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	desc, err := writeBlob(dir, ocispec.MediaTypeImageManifest, raw)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeManifestRef(dir, ref, desc); err != nil {
		t.Fatal(err)
	}

	destinationCtx := &types.SystemContext{DockerInsecureSkipTLSVerify: types.OptionalBoolTrue}
	if _, err := pushDescriptorToRegistryWithDestinationContext(context.Background(), dir, ref, desc, destinationCtx, true); err == nil {
		t.Fatal("push succeeded by rewriting an OCI artifact the registry rejected")
	}
	if ociPuts.Load() == 0 {
		t.Fatal("test registry never received the OCI artifact manifest")
	}
	if convertedPuts.Load() != 0 {
		t.Fatalf("push retried with %d digest-changing converted manifest(s)", convertedPuts.Load())
	}
}

func TestPodmanPushesFallbackIndexWithoutArtifactBlobs(t *testing.T) {
	reg := registry.New(registry.Logger(log.New(io.Discard, "", 0)))
	srv := httptest.NewTLSServer(reg)
	defer srv.Close()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	ref := u.Host + "/org/model:sha256-" + strings.Repeat("a", 64)
	sys := &types.SystemContext{DockerInsecureSkipTLSVerify: types.OptionalBoolTrue}
	artifactDir := t.TempDir()
	if err := ensureLayout(artifactDir); err != nil {
		t.Fatal(err)
	}
	config, err := writeBlob(artifactDir, "application/vnd.example.config.v1+json", []byte(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	artifactManifest := ocispec.Manifest{MediaType: ocispec.MediaTypeImageManifest, Config: config}
	artifactManifest.SchemaVersion = 2
	artifactRaw, err := json.Marshal(artifactManifest)
	if err != nil {
		t.Fatal(err)
	}
	artifact, err := writeBlob(artifactDir, ocispec.MediaTypeImageManifest, artifactRaw)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pushDescriptorToRegistryWithDestinationContext(context.Background(), artifactDir, u.Host+"/org/model:artifact", artifact, sys, true); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := ensureLayout(dir); err != nil {
		t.Fatal(err)
	}
	index := ocispec.Index{Manifests: []ocispec.Descriptor{artifact}}
	index.SchemaVersion = 2
	index.MediaType = ocispec.MediaTypeImageIndex
	raw, err := json.Marshal(index)
	if err != nil {
		t.Fatal(err)
	}
	desc, err := writeBlob(dir, ocispec.MediaTypeImageIndex, raw)
	if err != nil {
		t.Fatal(err)
	}
	if err := pushIndexToRegistryWithDestinationContext(context.Background(), dir, ref, desc, sys); err != nil {
		t.Fatal(err)
	}
	srcRef, err := alltransports.ParseImageName("docker://" + ref)
	if err != nil {
		t.Fatal(err)
	}
	src, err := srcRef.NewImageSource(context.Background(), sys)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	got, _, err := src.GetManifest(context.Background(), nil)
	if err != nil || string(got) != string(raw) {
		t.Fatalf("published index = %s, %v; want %s", got, err, raw)
	}
}
