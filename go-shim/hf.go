// HuggingFace-compatible helpers shared by the non-HF source handlers.
//
// The HuggingFace fetch path itself now lives in Rust (src/hf), so nothing
// here downloads from HuggingFace. What remains is the OCI-vs-HF host
// classification both backends need, plus the API/media-type/manifest
// helpers the ModelScope, NGC, S3 and GCS handlers in uri_sources.go reuse
// — those registries speak the same HuggingFace-compatible API.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	modelspec "github.com/modelpack/model-spec/specs-go/v1"
	specs "github.com/opencontainers/image-spec/specs-go"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

// errHFNotHandledHere reports a HuggingFace reference reaching this shim.
// hf::classify (src/hf/mod.rs) routes every one to Rust before ffi::pull or
// ffi::transfer is called, so this is unreachable unless that routing
// regresses or something calls the C API directly — either of which should
// fail loudly rather than silently take a path that no longer exists.
func errHFNotHandledHere(ref string) error {
	return fmt.Errorf(
		"%s: HuggingFace references are handled natively in Rust (src/hf), not by the Go shim", ref)
}

// ---------------------------------------------------------------------------
// Registry detection
// ---------------------------------------------------------------------------

// isKnownOCIHost returns true for registries that are definitely OCI-compliant,
// skipping the network probe entirely.
func isKnownOCIHost(host string) bool {
	switch host {
	case "ghcr.io", "docker.io", "index.docker.io", "registry-1.docker.io",
		"quay.io", "gcr.io", "mcr.microsoft.com", "public.ecr.aws":
		return true
	}
	return false
}

// isKnownHFHost returns true for known HuggingFace-compatible hosts.
func isKnownHFHost(host string) bool {
	switch host {
	case "hf.co", "huggingface.co", "modelscope.cn":
		return true
	}
	return false
}

// isOCIRegistry probes the OCI Distribution /v2/ endpoint and returns true if
// the server advertises itself as an OCI registry via the standard header.
func isOCIRegistry(ctx context.Context, client *http.Client, host string) bool {
	probeCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(probeCtx, "GET", "https://"+host+"/v2/", nil)
	if err != nil {
		return false
	}
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	resp.Body.Close()
	// OCI registries advertise registry/2.0 on both 200 and 401 responses.
	return resp.Header.Get("Docker-Distribution-Api-Version") != ""
}

// isOCIHost reports whether host should be treated as an OCI Distribution
// registry (true) or a HuggingFace-compatible host (false): known-host
// shortcuts first, then a live /v2/ probe as the fallback for anything else.
// This is the single decision table `llmman pull`'s docker and podman
// backends (backend_docker.go, backend_podman.go) and `llmman transfer`'s
// source classification (transfer_common.go) all need to agree on.
func isOCIHost(ctx context.Context, host string) bool {
	if isKnownHFHost(host) {
		return false
	}
	if isKnownOCIHost(host) {
		return true
	}
	probeClient := &http.Client{Timeout: 5 * time.Second}
	return isOCIRegistry(ctx, probeClient, host)
}

// classifyPullRef runs the URI-scheme dispatch and OCI-vs-HuggingFace host
// classification shared by both backends' pullToLayout (backend_docker.go,
// backend_podman.go). If handled is true, dispatchPull has already fully
// processed ref (via one of hf://, ms://, ngc://, s3://, gs://, or a local
// path) and the caller should return dispatchErr immediately without doing
// anything else. Otherwise normalizedRef is ref with a ":latest" tag
// defaulted in, and isOCI reports whether normalizedRef's host should be
// pulled via the OCI registry protocol (true) or the shared HF path (false).
func classifyPullRef(ctx context.Context, ref, layoutDir string) (normalizedRef string, isOCI, handled bool, dispatchErr error) {
	if handled, err := dispatchPull(ctx, ref, layoutDir); handled {
		return ref, false, true, err
	}

	// Normalize: append :latest if reference has no tag or digest.
	if strings.LastIndex(ref, ":") <= strings.LastIndex(ref, "/") {
		ref = ref + ":latest"
	}

	host := strings.SplitN(ref, "/", 2)[0]
	return ref, isOCIHost(ctx, host), false, nil
}

// ---------------------------------------------------------------------------
// HuggingFace API types and helpers
// ---------------------------------------------------------------------------

// cachedLayerName returns the GGUF filename for ref if it is fully cached in
// the local OCI store (manifest blob + all layer blobs present), or "" if not.
func cachedLayerName(layoutDir, ref string) string {
	m, err := readManifestRef(layoutDir, ref)
	if err != nil {
		return ""
	}
	if !blobExists(layoutDir, m) {
		return ""
	}
	data, err := readBlob(layoutDir, m.Digest)
	if err != nil {
		return ""
	}
	var manifest ocispec.Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return ""
	}
	for _, layer := range manifest.Layers {
		if !blobExists(layoutDir, layer) {
			return ""
		}
	}
	// All blobs present — return a filename from the first layer annotation.
	if len(manifest.Layers) > 0 {
		ann := manifest.Layers[0].Annotations
		for _, key := range []string{modelspec.AnnotationFilepath, ocispec.AnnotationTitle} {
			if name := ann[key]; name != "" {
				return filepath.Base(name)
			}
		}
	}
	return ref
}

// reportCached prints "Cached <label>" and returns true if ref is already
// fully cached in layoutDir (manifest blob + all layer blobs present) — the
// signal every pull entry point in this package uses to skip all network
// I/O. label defaults to the cached name cachedLayerName itself resolved
// (e.g. a GGUF filename) when the empty string is passed.
func reportCached(layoutDir, ref, label string) bool {
	name := cachedLayerName(layoutDir, ref)
	if name == "" {
		return false
	}
	if label == "" {
		label = name
	}
	fmt.Fprintf(os.Stderr, "Cached   %s\n", label)
	return true
}

// shouldDownloadSafetensors returns true for files that belong in a local model directory.
func shouldDownloadSafetensors(path string) bool {
	base := strings.ToLower(filepath.Base(path))
	ext := strings.ToLower(filepath.Ext(path))
	// Skip hidden files, large non-model binaries, and git internals.
	if strings.HasPrefix(base, ".") {
		return false
	}
	switch ext {
	case ".safetensors", ".bin", ".pt", ".pth": // weights
		return true
	// config / tokenizer — ".jinja" is a standalone chat template file,
	// see safetensorsMediaType.
	case ".json", ".model", ".txt", ".tiktoken", ".jinja":
		return true
	}
	// README and licence are useful but optional.
	switch base {
	case "readme.md", "license", "licence", "license.txt", "licence.txt":
		return true
	}
	return false
}

func storeSafetensorsAsOCI(layoutDir, ref, modelRepo string, meta modelMeta, layers []ocispec.Descriptor) error {
	manifestDesc, err := buildCNCFManifest(layoutBlobSink(layoutDir), meta, modelRepo, "", layers)
	if err != nil {
		return err
	}
	return writeManifestRef(layoutDir, ref, manifestDesc)
}

// modelMeta bundles the optional-but-valuable CNCF model-spec metadata
// buildCNCFManifest has enough information to populate beyond the bare
// config.format+modelfs every manifest already needs — see
// https://github.com/modelpack/model-spec/blob/main/docs/config.md. Both
// fields are entirely optional per the spec's own schema and are simply
// omitted when unknown/inapplicable (Model marshals empty sub-fields as
// omitted JSON via their own `,omitempty` tags).
type modelMeta struct {
	// Format is config.format — "gguf" or "safetensors" here.
	Format string

	// Licenses populates descriptor.licenses as SPDX license
	// expressions. Nil if the source repo didn't declare a usable one.
	Licenses []string

	// Vision marks the model as accepting image input in addition to
	// text (config.capabilities.inputTypes/outputTypes) — set when a
	// multimodal projector layer was actually included among layers.
	// The spec has no separate annotation for
	// "this manifest has an mmproj layer"; capabilities is model-spec's
	// own mechanism for signalling multimodal support.
	Vision bool
}

// buildCNCFManifest builds the shared CNCF model-spec manifest+config, for
// storeSafetensorsAsOCI above and the uri_sources.go source handlers. The
// cncfBlobSink below keeps *where* a blob ends up out of that logic.
// cncfBlobSink stores one marshaled CNCF blob (config or manifest JSON) and
// returns its descriptor.
type cncfBlobSink func(mediaType string, data []byte) (ocispec.Descriptor, error)

// layoutBlobSink is the cncfBlobSink for storing blobs in a local OCI layout
// directory.
func layoutBlobSink(layoutDir string) cncfBlobSink {
	return func(mediaType string, data []byte) (ocispec.Descriptor, error) {
		return writeBlob(layoutDir, mediaType, data)
	}
}

// buildCNCFManifest builds a conformant CNCF model-spec config blob and
// manifest referencing layers, storing each via sink, and returns the
// manifest's descriptor. meta carries the optional descriptor.licenses/
// config.capabilities metadata (see modelMeta's own doc comment).
// filepathAnnotation sets the manifest-level org.cncf.model.filepath
// annotation for the single-weight-file case (GGUF); pass "" for the
// multi-layer safetensors case, which only sets ai.model.repo.
func buildCNCFManifest(sink cncfBlobSink, meta modelMeta, modelRepo, filepathAnnotation string, layers []ocispec.Descriptor) (ocispec.Descriptor, error) {
	model := modelspec.Model{
		ModelFS: modelspec.ModelFS{Type: "layers"},
		Config:  modelspec.ModelConfig{Format: meta.Format},
	}
	if len(meta.Licenses) > 0 {
		model.Descriptor.Licenses = meta.Licenses
	}
	if meta.Vision {
		model.Config.Capabilities = &modelspec.ModelCapabilities{
			InputTypes:  []modelspec.Modality{modelspec.TextModality, modelspec.ImageModality},
			OutputTypes: []modelspec.Modality{modelspec.TextModality},
		}
	}
	for _, l := range layers {
		model.ModelFS.DiffIDs = append(model.ModelFS.DiffIDs, l.Digest)
	}
	cfgData, err := json.Marshal(model)
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("marshal CNCF model config: %w", err)
	}
	configDesc, err := sink(modelspec.MediaTypeModelConfig, cfgData)
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("store CNCF model config: %w", err)
	}

	annotations := map[string]string{"ai.model.repo": modelRepo}
	if filepathAnnotation != "" {
		annotations[modelspec.AnnotationFilepath] = filepathAnnotation
	}
	manifest := ocispec.Manifest{
		Versioned:    specs.Versioned{SchemaVersion: 2},
		MediaType:    ocispec.MediaTypeImageManifest,
		ArtifactType: modelspec.ArtifactTypeModelManifest,
		Config:       configDesc,
		Layers:       layers,
		Annotations:  annotations,
	}
	manifestData, err := json.Marshal(manifest)
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("marshal CNCF manifest: %w", err)
	}
	manifestDesc, err := sink(ocispec.MediaTypeImageManifest, manifestData)
	if err != nil {
		return ocispec.Descriptor{}, fmt.Errorf("store CNCF manifest: %w", err)
	}
	return manifestDesc, nil
}
