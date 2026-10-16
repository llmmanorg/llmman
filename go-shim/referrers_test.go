package main

import (
	"encoding/json"
	"testing"

	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

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
