// manifest_ref.go – per-model manifest reference storage, used by both the
// docker and podman backends. No build tag: compiled for all configurations.
//
// Each pulled/tagged/built model gets its own small JSON file recording
// which manifest blob (content-addressed under blobs/, see writeBlob in
// shared_oci.go) it currently points to — one file per model, replacing
// the old shared index.json every model in the store used to live in
// (breaking change: existing index.json stores need a re-pull). The
// pointer path is a readable, percent-escaped repo/tag tree. Legacy
// per-segment pointers remain readable, but normal writes leave them alone.
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
	"syscall"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

// manifestsDirName is the legacy pointer tree; new writes use manifests-v2.
const manifestsDirName = "manifests"
const manifestsV2DirName = "manifests-v2"

func canonicalRefKey(ref string) string {
	// An object-store key or local path is opaque; ":latest" is not a
	// default-tag alias there.
	if strings.HasPrefix(ref, "/") || strings.Contains(ref, "://") {
		return ref
	}
	if strings.LastIndex(ref, ":") > strings.LastIndex(ref, "/") {
		return ref
	}
	return ref + ":latest"
}

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
	key := canonicalRefKey(ref)
	opaque := strings.HasPrefix(ref, "/") || strings.Contains(ref, "://")
	parts := []string{layoutDir, manifestsV2DirName}
	if opaque {
		parts = append(parts, "~") // no encoded segment can be a literal "~"
	}
	name, tag := "", key
	if opaque {
		if i := strings.LastIndexByte(key, '/'); i >= 0 {
			name, tag = key[:i], key[i+1:]
		}
	} else {
		i := strings.LastIndexByte(key, ':')
		name, tag = key[:i], key[i+1:]
	}
	if name != "" || opaque {
		for _, segment := range strings.Split(name, "/") {
			parts = appendRefSegment(parts, segment, true)
		}
	}
	return filepath.Join(appendRefSegment(parts, tag, false)...)
}

// appendRefSegment mirrors Rust's push_ref_segment. The reserved ~ markers
// split unusually long percent-encoded names below filesystem limits without
// aliasing ordinary names or losing any original bytes.
func appendRefSegment(parts []string, segment string, directory bool) []string {
	encoded := escapeRefSegment(segment, directory)
	if len(encoded) <= 200 {
		if directory {
			return append(parts, encoded)
		}
		return append(parts, encoded+".json")
	}
	for len(encoded) > 180 {
		parts = append(parts, "~"+encoded[:180])
		encoded = encoded[180:]
	}
	parts = append(parts, "~"+encoded)
	if directory {
		return append(parts, "~~")
	}
	return append(parts, "~~.json")
}

// escapeRefSegment mirrors Rust's escape_ref_segment. Percent-encoding is
// lossless, and uppercase bytes cannot alias lowercase on case-folding disks.
// Repo directories cannot end in .json, unlike pointer filenames.
func escapeRefSegment(segment string, directory bool) string {
	if segment == "" {
		return "~"
	}
	base, _, _ := strings.Cut(strings.ToLower(segment), ".")
	reserved := map[string]bool{"con": true, "prn": true, "aux": true, "nul": true,
		"com1": true, "com2": true, "com3": true, "com4": true, "com5": true,
		"com6": true, "com7": true, "com8": true, "com9": true,
		"lpt1": true, "lpt2": true, "lpt3": true, "lpt4": true, "lpt5": true,
		"lpt6": true, "lpt7": true, "lpt8": true, "lpt9": true}[base]
	var out strings.Builder
	for i, b := range []byte(segment) {
		safe := b >= 'a' && b <= 'z' || b >= '0' && b <= '9' || b == '-' || b == '_' || b == '.'
		if safe && (!reserved || i != 0) && segment != "." && segment != ".." {
			out.WriteByte(b)
		} else {
			out.WriteString("%")
			out.WriteByte("0123456789ABCDEF"[b>>4])
			out.WriteByte("0123456789ABCDEF"[b&15])
		}
	}
	encoded := out.String()
	if directory && (strings.HasSuffix(encoded, ".json") || strings.HasSuffix(encoded, ".")) {
		i := strings.LastIndexByte(encoded, '.')
		encoded = encoded[:i] + "%2E" + encoded[i+1:]
	}
	return encoded
}

func missingLegacyPointer(err error) bool {
	return errors.Is(err, fs.ErrNotExist) || errors.Is(err, syscall.ENOTDIR)
}

func legacyManifestRefPath(layoutDir, ref string) string {
	parts := append([]string{layoutDir, manifestsDirName}, refPathSegments(ref)...)
	return filepath.Join(parts...)
}

// readManifestRef reads the manifest descriptor stored for ref, or an
// error if ref has never been tagged in this store.
func readManifestRef(layoutDir, ref string) (ocispec.Descriptor, error) {
	for i, path := range []string{manifestRefPath(layoutDir, ref), legacyManifestRefPath(layoutDir, ref)} {
		data, err := os.ReadFile(path)
		if errors.Is(err, fs.ErrNotExist) || (i == 1 && missingLegacyPointer(err)) {
			continue
		}
		if err != nil {
			return ocispec.Descriptor{}, err
		}
		var desc ocispec.Descriptor
		if err := json.Unmarshal(data, &desc); err != nil {
			return ocispec.Descriptor{}, err
		}
		if canonicalRefKey(desc.Annotations[ocispec.AnnotationRefName]) == canonicalRefKey(ref) {
			return desc, nil
		}
	}
	return ocispec.Descriptor{}, fs.ErrNotExist
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

// ensureLayout initialises the OCI layout marker and v2 pointer tree.
// The legacy tree is only read, never created for a new layout.
func ensureLayout(layoutDir string) error {
	if err := os.MkdirAll(layoutDir, 0o755); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Join(layoutDir, manifestsV2DirName), 0o755); err != nil {
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
