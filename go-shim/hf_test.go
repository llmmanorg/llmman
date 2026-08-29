package main

import (
	"encoding/json"
	"testing"

	modelspec "github.com/modelpack/model-spec/specs-go/v1"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

// TestBuildCNCFManifestPopulatesMetadata exercises buildCNCFManifest
// end-to-end against a local OCI layout (layoutBlobSink), with no
// network involved, to verify the actual JSON shape written for
// descriptor.licenses and config.capabilities matches what model-spec's
// schema expects — see
// https://github.com/modelpack/model-spec/blob/main/docs/config.md.
func TestBuildCNCFManifestPopulatesMetadata(t *testing.T) {
	dir := t.TempDir()
	if err := ensureLayout(dir); err != nil {
		t.Fatalf("ensureLayout: %v", err)
	}

	weightDesc, err := writeBlob(dir, modelspec.MediaTypeModelWeightRaw, []byte("fake gguf weight"))
	if err != nil {
		t.Fatalf("writeBlob weight: %v", err)
	}
	weightDesc.Annotations = map[string]string{modelspec.AnnotationFilepath: "model-Q4_K_M.gguf"}

	mmprojDesc, err := writeBlob(dir, modelspec.MediaTypeModelWeightRaw, []byte("fake mmproj weight"))
	if err != nil {
		t.Fatalf("writeBlob mmproj: %v", err)
	}
	mmprojDesc.Annotations = map[string]string{modelspec.AnnotationFilepath: "mmproj-F16.gguf"}

	meta := modelMeta{
		Format:   "gguf",
		Licenses: []string{"Apache-2.0"},
		Vision:   true,
	}
	layers := []ocispec.Descriptor{weightDesc, mmprojDesc}
	manifestDesc, err := buildCNCFManifest(layoutBlobSink(dir), meta, "unsloth/example-GGUF", "", layers)
	if err != nil {
		t.Fatalf("buildCNCFManifest: %v", err)
	}

	// Read the manifest and config blobs straight back out of the layout
	// and verify the exact JSON shape a real consumer would see.
	manifestData, err := readBlob(dir, manifestDesc.Digest)
	if err != nil {
		t.Fatalf("readBlob manifest: %v", err)
	}
	var manifest ocispec.Manifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		t.Fatalf("unmarshal manifest: %v", err)
	}
	if manifest.ArtifactType != modelspec.ArtifactTypeModelManifest {
		t.Errorf("artifactType = %q, want %q", manifest.ArtifactType, modelspec.ArtifactTypeModelManifest)
	}
	if len(manifest.Layers) != 2 {
		t.Fatalf("expected 2 layers, got %d", len(manifest.Layers))
	}

	configData, err := readBlob(dir, manifest.Config.Digest)
	if err != nil {
		t.Fatalf("readBlob config: %v", err)
	}
	var model modelspec.Model
	if err := json.Unmarshal(configData, &model); err != nil {
		t.Fatalf("unmarshal config: %v", err)
	}
	if len(model.Descriptor.Licenses) != 1 || model.Descriptor.Licenses[0] != "Apache-2.0" {
		t.Errorf("descriptor.licenses = %v, want [Apache-2.0]", model.Descriptor.Licenses)
	}
	if model.Config.Format != "gguf" {
		t.Errorf("config.format = %q, want gguf", model.Config.Format)
	}
	if model.Config.Capabilities == nil {
		t.Fatal("expected config.capabilities to be set for a vision model")
	}
	wantIn := []modelspec.Modality{modelspec.TextModality, modelspec.ImageModality}
	if len(model.Config.Capabilities.InputTypes) != 2 ||
		model.Config.Capabilities.InputTypes[0] != wantIn[0] ||
		model.Config.Capabilities.InputTypes[1] != wantIn[1] {
		t.Errorf("capabilities.inputTypes = %v, want %v", model.Config.Capabilities.InputTypes, wantIn)
	}
	if len(model.ModelFS.DiffIDs) != 2 {
		t.Errorf("modelfs.diffIds has %d entries, want 2 (one per layer)", len(model.ModelFS.DiffIDs))
	}
}

// Regression: a standalone chat_template.jinja file used to be silently
// excluded from a safetensors pull, causing vllm/transformers to refuse
// every chat request with "you must provide a chat template".
func TestShouldDownloadSafetensorsIncludesChatTemplateJinja(t *testing.T) {
	if !shouldDownloadSafetensors("chat_template.jinja") {
		t.Error("chat_template.jinja must be downloaded as part of a safetensors pull")
	}
}
