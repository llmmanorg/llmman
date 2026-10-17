//go:build !podman

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-containerregistry/pkg/registry"
	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

func startReferrersRegistry(t *testing.T, native bool) string {
	t.Helper()
	return startRegistryWithOptions(t, nil, registry.WithReferrersSupport(native))
}

func subjectDescriptor(t *testing.T, ref string, dgst digest.Digest) ocispec.Descriptor {
	t.Helper()
	raw, err := fetchManifestRaw(context.Background(), ref)
	if err != nil {
		t.Fatalf("fetch subject manifest: %v", err)
	}
	return ocispec.Descriptor{
		MediaType: ocispec.MediaTypeImageManifest,
		Digest:    dgst,
		Size:      int64(len(raw)),
	}
}

func pushTestArtifact(t *testing.T, ref, payload string, subject ocispec.Descriptor) ocispec.Descriptor {
	t.Helper()
	desc := pushTestArtifactManifest(t, ref, payload, subject)
	if err := publishFallbackReferrer(context.Background(), ref, subject.Digest, desc); err != nil {
		t.Fatalf("publish fallback referrer: %v", err)
	}
	return desc
}

func pushTestArtifactManifest(t *testing.T, ref, payload string, subject ocispec.Descriptor) ocispec.Descriptor {
	t.Helper()
	dir := t.TempDir()
	if err := ensureLayout(dir); err != nil {
		t.Fatalf("init artifact layout: %v", err)
	}
	config, err := writeBlob(dir, "application/vnd.example.sbom.config.v1+json", []byte(`{}`))
	if err != nil {
		t.Fatalf("write artifact config: %v", err)
	}
	layer, err := writeBlob(dir, "application/vnd.example.sbom.v1+json", []byte(payload))
	if err != nil {
		t.Fatalf("write artifact layer: %v", err)
	}
	manifest := ocispec.Manifest{
		MediaType:    ocispec.MediaTypeImageManifest,
		ArtifactType: "application/vnd.example.sbom.v1",
		Config:       config,
		Layers:       []ocispec.Descriptor{layer},
		Annotations: map[string]string{
			ocispec.AnnotationCreated: "2026-10-16T00:00:00Z",
			"example.com/vendor":      payload,
			ocispec.AnnotationRefName: "user-provided",
		},
	}
	manifest.SchemaVersion = 2
	raw, err := json.Marshal(manifest)
	if err != nil {
		t.Fatalf("marshal artifact: %v", err)
	}
	desc, err := writeBlob(dir, ocispec.MediaTypeImageManifest, raw)
	if err != nil {
		t.Fatalf("write artifact manifest: %v", err)
	}
	if err := writeManifestRef(dir, ref, desc); err != nil {
		t.Fatalf("tag artifact: %v", err)
	}

	desc, err = setManifestSubject(dir, ref, subject)
	if err != nil {
		t.Fatalf("set artifact subject: %v", err)
	}
	if _, err := pushToRegistry(context.Background(), dir, ref); err != nil {
		t.Fatalf("push artifact: %v", err)
	}
	return desc
}

func assertRemoteSubject(t *testing.T, ref string, want digest.Digest) {
	t.Helper()
	raw, err := fetchManifestRaw(context.Background(), ref)
	if err != nil {
		t.Fatalf("fetch artifact manifest: %v", err)
	}
	var manifest ocispec.Manifest
	if err := json.Unmarshal(raw, &manifest); err != nil {
		t.Fatalf("parse artifact manifest: %v", err)
	}
	if manifest.Subject == nil || manifest.Subject.Digest != want {
		t.Fatalf("artifact subject = %+v, want %s", manifest.Subject, want)
	}
}

func TestPushArtifactAndListNativeReferrers(t *testing.T) {
	host := startReferrersRegistry(t, true)
	subjectRef := host + "/org/model:v1"
	subjectDigest := publishModel(t, subjectRef)
	subject := subjectDescriptor(t, subjectRef, subjectDigest)
	artifactRef := host + "/org/model:sbom"
	artifact := pushTestArtifact(t, artifactRef, `{"name":"native"}`, subject)

	assertRemoteSubject(t, artifactRef, subjectDigest)
	refs, native, err := fetchReferrers(context.Background(), subjectRef, subjectDigest)
	if err != nil {
		t.Fatalf("list referrers: %v", err)
	}
	if !native {
		t.Fatal("registry's native referrers API was not used")
	}
	if len(refs) != 1 || refs[0].Digest != artifact.Digest {
		t.Fatalf("referrers = %+v, want artifact %s", refs, artifact.Digest)
	}
}

func TestPushDescriptorIgnoresChangedLocalRef(t *testing.T) {
	host := startReferrersRegistry(t, true)
	ref := host + "/org/model:artifact"
	dir := t.TempDir()
	if err := ensureLayout(dir); err != nil {
		t.Fatal(err)
	}
	config, err := writeBlob(dir, "application/vnd.example.config.v1+json", []byte(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	manifest := ocispec.Manifest{MediaType: ocispec.MediaTypeImageManifest, Config: config}
	manifest.SchemaVersion = 2
	manifest.Annotations = map[string]string{"selected": "first"}
	firstRaw, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	first, err := writeBlob(dir, ocispec.MediaTypeImageManifest, firstRaw)
	if err != nil {
		t.Fatal(err)
	}
	manifest.Annotations["selected"] = "second"
	secondRaw, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	second, err := writeBlob(dir, ocispec.MediaTypeImageManifest, secondRaw)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeManifestRef(dir, ref, second); err != nil {
		t.Fatal(err)
	}
	if _, err := pushDescriptorToRegistry(context.Background(), dir, ref, first); err != nil {
		t.Fatal(err)
	}
	got, err := fetchManifestRaw(context.Background(), ref)
	if err != nil || string(got) != string(firstRaw) {
		t.Fatalf("pushed manifest = %s, %v; want first descriptor", got, err)
	}
}

func TestPushArtifactAndListFallbackReferrers(t *testing.T) {
	host := startReferrersRegistry(t, false)
	subjectRef := host + "/org/model:v1"
	subjectDigest := publishModel(t, subjectRef)
	subject := subjectDescriptor(t, subjectRef, subjectDigest)

	first := pushTestArtifact(t, host+"/org/model:sbom-a", `{"name":"a"}`, subject)
	second := pushTestArtifact(t, host+"/org/model:sbom-b", `{"name":"b"}`, subject)

	refs, native, err := fetchReferrers(context.Background(), subjectRef, subjectDigest)
	if err != nil {
		t.Fatalf("list fallback referrers: %v", err)
	}
	if native {
		t.Fatal("registry without the referrers endpoint was reported as native")
	}
	if len(refs) != 2 {
		t.Fatalf("fallback referrers = %+v, want two entries", refs)
	}
	got := map[digest.Digest]bool{}
	for _, ref := range refs {
		got[ref.Digest] = true
		if ref.ArtifactType != "application/vnd.example.sbom.v1" {
			t.Errorf("artifact type = %q", ref.ArtifactType)
		}
		if ref.Annotations[ocispec.AnnotationCreated] != "2026-10-16T00:00:00Z" || ref.Annotations["example.com/vendor"] == "" {
			t.Errorf("artifact annotations = %+v", ref.Annotations)
		}
		if ref.Annotations[ocispec.AnnotationRefName] != "user-provided" {
			t.Errorf("manifest ref-name annotation was changed: %+v", ref.Annotations)
		}
	}
	if !got[first.Digest] || !got[second.Digest] {
		t.Fatalf("fallback referrers = %+v, want %s and %s", refs, first.Digest, second.Digest)
	}
}

func TestFallbackIndexSizeCheckedBeforePush(t *testing.T) {
	var fallbackPuts atomic.Int32
	host := startRegistryWithOptions(t, func(_ http.ResponseWriter, r *http.Request) bool {
		if r.Method == http.MethodPut && strings.Contains(r.URL.Path, "/manifests/sha256-") {
			fallbackPuts.Add(1)
		}
		return false
	}, registry.WithReferrersSupport(false))
	ref := host + "/org/model:artifact"
	artifact := ocispec.Descriptor{
		MediaType:   ocispec.MediaTypeImageManifest,
		Digest:      digest.FromString("artifact"),
		Annotations: map[string]string{"large": strings.Repeat("a", int(maxManifestBytes))},
	}
	err := publishFallbackReferrer(context.Background(), ref, digest.FromString("subject"), artifact)
	if err == nil || !strings.Contains(err.Error(), "over the") {
		t.Fatalf("oversized fallback index error = %v", err)
	}
	if got := fallbackPuts.Load(); got != 0 {
		t.Fatalf("oversized index caused %d manifest PUTs", got)
	}
}

func TestConcurrentFallbackPublicationsKeepBothReferrers(t *testing.T) {
	var mu sync.Mutex
	reads := 0
	bothReading := make(chan struct{})
	host := startRegistryWithOptions(t, func(_ http.ResponseWriter, r *http.Request) bool {
		if r.Method != http.MethodGet || !strings.Contains(r.URL.Path, "/manifests/sha256-") {
			return false
		}
		mu.Lock()
		reads++
		if reads == 2 {
			close(bothReading)
		}
		mu.Unlock()
		select {
		case <-bothReading:
		case <-time.After(100 * time.Millisecond):
		}
		return false
	}, registry.WithReferrersSupport(false))

	subjectRef := host + "/org/model:v1"
	subjectDigest := publishModel(t, subjectRef)
	subject := subjectDescriptor(t, subjectRef, subjectDigest)
	first := pushTestArtifactManifest(t, host+"/org/model:sbom-a", `{"name":"a"}`, subject)
	second := pushTestArtifactManifest(t, host+"/org/model:sbom-b", `{"name":"b"}`, subject)

	errs := make(chan error, 2)
	for _, artifact := range []ocispec.Descriptor{first, second} {
		go func(artifact ocispec.Descriptor) {
			errs <- publishFallbackReferrer(context.Background(), subjectRef, subjectDigest, artifact)
		}(artifact)
	}
	for range 2 {
		if err := <-errs; err != nil {
			t.Fatalf("publish fallback referrer: %v", err)
		}
	}

	refs, native, err := fetchReferrers(context.Background(), subjectRef, subjectDigest)
	if err != nil {
		t.Fatalf("list fallback referrers: %v", err)
	}
	if native || len(refs) != 2 {
		t.Fatalf("fallback referrers = %+v, native = %v; want two fallback entries", refs, native)
	}
}
