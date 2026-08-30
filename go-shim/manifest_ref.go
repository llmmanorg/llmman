// manifest_ref.go – per-model manifest reference storage, used by both the
// docker and podman backends. No build tag: compiled for all configurations.
//
// Each pulled/tagged/built model gets its own small JSON file recording
// which manifest blob (content-addressed under blobs/, see writeBlob in
// shared_oci.go) it currently points to — one file per model, replacing
// the old shared index.json every model in the store used to live in
// (breaking change: existing stores need a re-pull, no migration is
// provided). Mirrors Ollama's per-model manifest layout
// (manifest.WriteManifest, named by <host>/<namespace>/<model>/<tag>).
//
// Unlike Ollama, the manifest *content* stays a content-addressed blob
// rather than being inlined under the ref path: llmman must preserve a
// pulled manifest's exact original bytes/digest for registry push
// fidelity and future content-signing support, so each ref file here is
// a small pointer (the manifest's descriptor) rather than the manifest
// itself.
//
// Like Ollama's manifest.WriteManifest, there's no locking around this:
// a torn or racing write can now only ever corrupt its own ref file, not
// every model's, so the old index.json.lock isn't needed. Writes still
// go through a temp-file rename, since that costs nothing extra.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

// manifestsDirName is the top-level directory under a layout root holding
// every ref's manifest-pointer file — the sibling of blobs/.
const manifestsDirName = "manifests"

// refPathSegments splits ref into the path segments used to lay it out
// under manifests/ — one directory per "/"-delimited path segment, with
// an explicit (defaulted, if absent) tag as the final segment. Mirrors
// Ollama's model.Name.Filepath() ({host}/{namespace}/{model}/{tag}),
// generalized to also cover llmman's HuggingFace-style refs (already the
// same shape) and non-registry sources that never carry a tag.
//
// Each segment is sanitized for safe use as a single path component: a
// literal ':' or '\' is replaced with '_' (a Windows path separator or
// drive-letter colon could otherwise land inside a segment — e.g. an
// absolute Windows path, or a URI scheme's "://"), and an empty, "." or
// ".." segment (e.g. an empty tag from "repo:") is replaced with "__" —
// otherwise it would vanish from the joined path entirely (filepath.Join
// drops empty elements), silently writing the manifest file where the
// tag directory belongs and breaking every other tag of that repo.
func refPathSegments(ref string) []string {
	tag := "latest"
	name := ref
	if i := strings.LastIndex(ref, ":"); i > strings.LastIndex(ref, "/") {
		tag = ref[i+1:]
		name = ref[:i]
	}
	var segs []string
	for _, s := range strings.Split(name, "/") {
		if s == "" {
			continue
		}
		segs = append(segs, sanitizeRefSegment(s))
	}
	return append(segs, sanitizeRefSegment(tag))
}

// sanitizeRefSegment neutralizes characters unsafe as a path segment —
// see refPathSegments.
func sanitizeRefSegment(s string) string {
	if s == "" || s == "." || s == ".." {
		return "__"
	}
	return strings.NewReplacer(":", "_", "\\", "_").Replace(s)
}

// manifestRefPath returns the path of ref's manifest-pointer file under
// layoutDir.
func manifestRefPath(layoutDir, ref string) string {
	parts := append([]string{layoutDir, manifestsDirName}, refPathSegments(ref)...)
	return filepath.Join(parts...)
}

// readManifestRef reads the manifest descriptor stored for ref, or an
// error if ref has never been tagged in this store.
func readManifestRef(layoutDir, ref string) (ocispec.Descriptor, error) {
	data, err := os.ReadFile(manifestRefPath(layoutDir, ref))
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	var desc ocispec.Descriptor
	return desc, json.Unmarshal(data, &desc)
}

// writeManifestRef atomically records that ref now points at
// manifestDesc. Every ref's file is independent of every other's, so
// there's no read-modify-write cycle to race here at all.
func writeManifestRef(layoutDir, ref string, manifestDesc ocispec.Descriptor) error {
	// manifestDesc is passed by value, but Annotations is a map: copy it
	// rather than mutate it in place, or this would reach back into
	// whatever map the caller still owns (and race with any concurrent
	// use of that same map for a different ref).
	ann := make(map[string]string, len(manifestDesc.Annotations)+1)
	for k, v := range manifestDesc.Annotations {
		ann[k] = v
	}
	ann[ocispec.AnnotationRefName] = ref
	manifestDesc.Annotations = ann

	path := manifestRefPath(layoutDir, ref)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(manifestDesc, "", "  ")
	if err != nil {
		return err
	}
	// A uniquely-named temp file, not a fixed one: two concurrent writers
	// of the same ref must not truncate/interleave each other's write.
	out, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".*.tmp")
	if err != nil {
		return err
	}
	tmp := out.Name()
	if _, err := out.Write(data); err != nil {
		out.Close()
		os.Remove(tmp)
		return err
	}
	if err := out.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}

// findManifestForPush resolves ref to a manifest descriptor for
// llmman_push: an exact match only. A mistyped ref must error, never
// silently guess at some other locally stored model — which is also why
// the staged-transfer callers (crate::sources::transfer, and
// crate::hf::transfer's podman fallback) record the staged model under
// the *destination* reference before pushing it, rather than relying on
// a "the store has exactly one entry" fallback here.
func findManifestForPush(layoutDir, ref string) (ocispec.Descriptor, error) {
	desc, err := readManifestRef(layoutDir, ref)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return ocispec.Descriptor{}, fmt.Errorf("no manifest found for %q", ref)
		}
		return ocispec.Descriptor{}, fmt.Errorf("read manifest for %q: %w", ref, err)
	}
	return desc, nil
}

// ensureLayout initialises the OCI layout marker files and manifests/
// directory if not present.
func ensureLayout(layoutDir string) error {
	if err := os.MkdirAll(layoutDir, 0o755); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Join(layoutDir, manifestsDirName), 0o755); err != nil {
		return err
	}
	markerPath := filepath.Join(layoutDir, "oci-layout")
	if _, err := os.Stat(markerPath); os.IsNotExist(err) {
		if err := os.WriteFile(markerPath, []byte(`{"imageLayoutVersion":"1.0.0"}`), 0o644); err != nil {
			return err
		}
	}
	return nil
}
