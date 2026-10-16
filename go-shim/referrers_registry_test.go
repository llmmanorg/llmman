//go:build !podman

package main

import (
	"context"
	"encoding/json"
	"testing"

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
	if err := publishFallbackReferrer(context.Background(), ref, subject.Digest, desc); err != nil {
		t.Fatalf("publish fallback referrer: %v", err)
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
	}
	if !got[first.Digest] || !got[second.Digest] {
		t.Fatalf("fallback referrers = %+v, want %s and %s", refs, first.Digest, second.Digest)
	}
}
