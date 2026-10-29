package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

// TestWriteManifestRefIsAtomic checks a per-ref manifest file write never
// leaves a reader able to observe a truncated/partial file, and never
// leaves a stray ".tmp" file behind on success.
func TestWriteManifestRefIsAtomic(t *testing.T) {
	dir := t.TempDir()
	if err := ensureLayout(dir); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, manifestsDirName)); !os.IsNotExist(err) {
		t.Fatalf("new layout created legacy pointer tree: %v", err)
	}
	ref := "docker.io/ai/qwen3.5:0.8b"
	desc := ocispec.Descriptor{
		MediaType: ocispec.MediaTypeImageManifest,
		Digest:    digest.FromString("one"),
		Size:      1,
	}
	if err := writeManifestRef(dir, ref, desc); err != nil {
		t.Fatalf("writeManifestRef: %v", err)
	}

	path := manifestRefPath(dir, ref)
	entries, err := os.ReadDir(filepath.Dir(path))
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".tmp") {
			t.Fatalf("no .tmp file should survive a successful write, found %q", e.Name())
		}
	}

	got, err := readManifestRef(dir, ref)
	if err != nil {
		t.Fatalf("readManifestRef: %v", err)
	}
	if got.Digest != desc.Digest {
		t.Fatalf("readManifestRef returned %+v, want digest %s", got, desc.Digest)
	}
	if got.Annotations[ocispec.AnnotationRefName] != ref {
		t.Fatalf("readManifestRef returned annotations %+v, want ref.name %q", got.Annotations, ref)
	}
}

// TestWriteManifestRefDoesNotMutateTheCallersAnnotations is a regression
// test: manifestDesc is passed by value, but its Annotations map is a
// reference type — writeManifestRef must copy it rather than write
// ref.name into the caller's own map (which could be shared with, and
// so corrupt, a write for a different ref).
func TestWriteManifestRefDoesNotMutateTheCallersAnnotations(t *testing.T) {
	dir := t.TempDir()
	shared := map[string]string{"foo": "bar"}
	desc := ocispec.Descriptor{Digest: digest.FromString("shared"), Annotations: shared}

	if err := writeManifestRef(dir, "docker.io/ai/a:latest", desc); err != nil {
		t.Fatalf("writeManifestRef: %v", err)
	}
	if _, ok := shared[ocispec.AnnotationRefName]; ok {
		t.Fatalf("writeManifestRef mutated the caller's own annotations map: %+v", shared)
	}

	if err := writeManifestRef(dir, "docker.io/ai/b:latest", desc); err != nil {
		t.Fatalf("writeManifestRef: %v", err)
	}
	a, err := readManifestRef(dir, "docker.io/ai/a:latest")
	if err != nil {
		t.Fatalf("readManifestRef(a): %v", err)
	}
	if a.Annotations[ocispec.AnnotationRefName] != "docker.io/ai/a:latest" {
		t.Fatalf("ref a's own ref.name annotation got overwritten: %+v", a.Annotations)
	}
}

// TestManifestRefPathIsOneFilePerModel is the core regression test for
// this change: two different refs must live at two different paths under
// manifests-v2/, so a torn write (or a race between two writers) to one
// model can never affect any other model's own file.
func TestManifestRefPathIsOneFilePerModel(t *testing.T) {
	dir := t.TempDir()
	a := manifestRefPath(dir, "docker.io/ai/qwen3.5:0.8b")
	b := manifestRefPath(dir, "docker.io/ai/gemma4:latest")
	if a == b {
		t.Fatalf("expected distinct paths for distinct refs, got %q for both", a)
	}
	if want := filepath.Join(dir, manifestsV2DirName, "docker.io", "ai", "qwen3.5", "0.8b.json"); a != want {
		t.Fatalf("manifest path = %q, want readable tree %q", a, want)
	}
}

func TestManifestRefPathsMatchSharedGoldenVectors(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "tests", "fixtures", "manifest-ref-paths.json"))
	if err != nil {
		t.Fatal(err)
	}
	var vectors []struct {
		Reference string `json:"reference"`
		Path      string `json:"path"`
	}
	if err := json.Unmarshal(data, &vectors); err != nil {
		t.Fatal(err)
	}
	for _, vector := range vectors {
		got := manifestRefPath("store", vector.Reference)
		want := filepath.Join("store", filepath.FromSlash(vector.Path))
		if got != want {
			t.Errorf("manifestRefPath(%q) = %q, want %q", vector.Reference, got, want)
		}
	}
}

func TestManifestRefPathEscapesCollisionsAndAncestors(t *testing.T) {
	dir := t.TempDir()
	for _, pair := range [][2]string{
		{"reg.example:5000/ns/model:t", "reg.example_5000/ns/model:t"},
		{"reg.example/ns/model:Prod", "reg.example/ns/model:prod"},
		{"reg.example/ns/model:t", "reg.example/ns/model/t:x"},
		{"reg.example/ns/model:t", "reg.example/ns/model/t.json:x"},
	} {
		a, b := manifestRefPath(dir, pair[0]), manifestRefPath(dir, pair[1])
		if a == b || strings.HasPrefix(a, b+string(os.PathSeparator)) || strings.HasPrefix(b, a+string(os.PathSeparator)) {
			t.Fatalf("references %q and %q overlap: %q and %q", pair[0], pair[1], a, b)
		}
	}
	if got, want := manifestRefPath(dir, "reg.example:5000/ns/model:t"),
		filepath.Join(dir, manifestsV2DirName, "reg.example%3A5000", "ns", "model", "t.json"); got != want {
		t.Fatalf("registry path = %q, want %q", got, want)
	}
}

func TestManifestRefPathSplitsLongEncodedTag(t *testing.T) {
	dir := t.TempDir()
	ref := "reg.example/ns/model:" + strings.Repeat("A", 128)
	path := manifestRefPath(dir, ref)
	for _, component := range strings.Split(strings.TrimPrefix(path, dir+string(os.PathSeparator)), string(os.PathSeparator)) {
		if len(component) > 255 {
			t.Fatalf("path component has %d bytes: %q", len(component), component)
		}
	}
	if err := writeManifestRef(dir, ref, ocispec.Descriptor{Digest: digest.FromString("long")}); err != nil {
		t.Fatal(err)
	}
	if got, err := readManifestRef(dir, ref); err != nil || got.Digest != digest.FromString("long") {
		t.Fatalf("long tag read = %+v, %v", got, err)
	}
}

func TestLongOpaqueKeyIgnoresUnrepresentableLegacyPointer(t *testing.T) {
	dir := t.TempDir()
	ref := "s3://bucket/" + strings.Repeat("a", 256)
	legacyParent := filepath.Dir(filepath.Dir(legacyManifestRefPath(dir, ref)))
	if err := os.MkdirAll(legacyParent, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := readManifestRef(dir, ref); !os.IsNotExist(err) {
		t.Fatalf("unrepresentable legacy pointer: got %v, want not-exist", err)
	}
	if err := writeManifestRef(dir, ref, ocispec.Descriptor{Digest: digest.FromString("long")}); err != nil {
		t.Fatal(err)
	}
	if got, err := readManifestRef(dir, ref); err != nil || got.Digest != digest.FromString("long") {
		t.Fatalf("read long opaque key = %+v, %v", got, err)
	}
}

func TestManifestRefLegacyCollisionDoesNotOverwriteOtherReference(t *testing.T) {
	dir := t.TempDir()
	first := "reg.example:5000/ns/model:t"
	second := "reg.example_5000/ns/model:t"
	if legacyManifestRefPath(dir, first) != legacyManifestRefPath(dir, second) {
		t.Fatal("test references must collide in the old layout")
	}
	if manifestRefPath(dir, first) == manifestRefPath(dir, second) {
		t.Fatal("new paths must be distinct")
	}
	old := ocispec.Descriptor{Digest: digest.FromString("old"), Annotations: map[string]string{ocispec.AnnotationRefName: first}}
	data, err := json.Marshal(old)
	if err != nil {
		t.Fatal(err)
	}
	legacy := legacyManifestRefPath(dir, first)
	if err := os.MkdirAll(filepath.Dir(legacy), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(legacy, data, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := writeManifestRef(dir, second, ocispec.Descriptor{Digest: digest.FromString("new")}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(legacy); err != nil {
		t.Fatalf("other reference's legacy pointer was removed: %v", err)
	}
	for ref, want := range map[string]digest.Digest{first: old.Digest, second: digest.FromString("new")} {
		got, err := readManifestRef(dir, ref)
		if err != nil || got.Digest != want {
			t.Fatalf("read %s: digest %s, error %v; want %s", ref, got.Digest, err, want)
		}
	}
}

func TestWriteManifestRefSkipsLegacyDirectory(t *testing.T) {
	dir := t.TempDir()
	ref := "reg.example/ns/model:sub"
	nested := "reg.example/ns/model/sub:t"
	legacy := legacyManifestRefPath(dir, ref)
	nestedPath := legacyManifestRefPath(dir, nested)
	if filepath.Dir(nestedPath) != legacy {
		t.Fatalf("test references do not share the expected legacy directory")
	}
	if err := os.MkdirAll(legacy, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(nestedPath, []byte("nested pointer"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := writeManifestRef(dir, ref, ocispec.Descriptor{Digest: digest.FromString("new")}); err != nil {
		t.Fatalf("a legacy parent directory blocked an independent v2 write: %v", err)
	}
	if _, err := os.Stat(nestedPath); err != nil {
		t.Fatalf("nested legacy pointer was disturbed: %v", err)
	}
	if got, err := readManifestRef(dir, ref); err != nil || got.Digest != digest.FromString("new") {
		t.Fatalf("read v2 ref: %+v, %v", got, err)
	}
}

func TestLegacyAncestorFileDoesNotBlockV2WriteOrRead(t *testing.T) {
	dir := t.TempDir()
	ref := "reg.example/ns/model/sub:t"
	ancestor := legacyManifestRefPath(dir, "reg.example/ns/model:sub")
	if err := os.MkdirAll(filepath.Dir(ancestor), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(ancestor, []byte("another pointer"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := readManifestRef(dir, ref); !os.IsNotExist(err) {
		t.Fatalf("legacy ancestor read = %v, want missing pointer", err)
	}
	want := ocispec.Descriptor{Digest: digest.FromString("new")}
	if err := writeManifestRef(dir, ref, want); err != nil {
		t.Fatalf("legacy ancestor blocked v2 write: %v", err)
	}
	got, err := readManifestRef(dir, ref)
	if err != nil || got.Digest != want.Digest {
		t.Fatalf("v2 read = %+v, %v", got, err)
	}
	if _, err := os.Stat(ancestor); err != nil {
		t.Fatalf("legacy ancestor was changed: %v", err)
	}
}

func TestWriteManifestRefLeavesMatchingLegacyPointerUntouched(t *testing.T) {
	dir := t.TempDir()
	ref := "reg.example/ns/model:t"
	legacy := legacyManifestRefPath(dir, ref)
	if err := os.MkdirAll(filepath.Dir(legacy), 0o755); err != nil {
		t.Fatal(err)
	}
	old := ocispec.Descriptor{Digest: digest.FromString("old"), Annotations: map[string]string{ocispec.AnnotationRefName: ref}}
	raw, err := json.Marshal(old)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(legacy, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := writeManifestRef(dir, ref, ocispec.Descriptor{Digest: digest.FromString("new")}); err != nil {
		t.Fatal(err)
	}
	if got, err := os.ReadFile(legacy); err != nil || string(got) != string(raw) {
		t.Fatalf("legacy pointer changed: %s, %v", got, err)
	}
	if got, err := readManifestRef(dir, ref); err != nil || got.Digest != digest.FromString("new") {
		t.Fatalf("v2 pointer did not take precedence: %+v, %v", got, err)
	}
}

func TestOpaqueReferenceSuffixDoesNotAliasDefaultTag(t *testing.T) {
	dir := t.TempDir()
	for _, pair := range [][2]string{
		{"s3://bucket/key", "s3://bucket/key:latest"},
		{"/models/a", "/models/a:latest"},
	} {
		if manifestRefPath(dir, pair[0]) == manifestRefPath(dir, pair[1]) {
			t.Fatalf("opaque references %q and %q share a path", pair[0], pair[1])
		}
		for i, ref := range pair {
			if err := writeManifestRef(dir, ref, ocispec.Descriptor{Digest: digest.FromString(fmt.Sprintf("%d", i))}); err != nil {
				t.Fatal(err)
			}
		}
		for i, ref := range pair {
			got, err := readManifestRef(dir, ref)
			if err != nil || got.Digest != digest.FromString(fmt.Sprintf("%d", i)) {
				t.Fatalf("read %q: %+v, %v", ref, got, err)
			}
		}
	}
}

// TestRefPathSegmentsSanitizesUnsafeSegments checks a reference can never
// escape the manifests/ tree (a literal "..") or produce a path segment
// that would break on Windows (a literal ":").
func TestRefPathSegmentsSanitizesUnsafeSegments(t *testing.T) {
	cases := []struct {
		ref  string
		want []string
	}{
		{"docker.io/ai/qwen3.5:0.8b", []string{"docker.io", "ai", "qwen3.5", "0.8b"}},
		{"s3://bucket/key", []string{"s3_", "bucket", "key", "latest"}},
		{"../../etc/passwd", []string{"__", "__", "etc", "passwd", "latest"}},
		// An empty tag (a ref ending in a bare ":") must not vanish: an
		// empty final segment would collapse into the repo directory
		// itself, writing the manifest where the tag directory belongs.
		{"docker.io/ai/x:", []string{"docker.io", "ai", "x", "__"}},
	}
	for _, c := range cases {
		got := refPathSegments(c.ref)
		if fmt.Sprint(got) != fmt.Sprint(c.want) {
			t.Errorf("refPathSegments(%q) = %v, want %v", c.ref, got, c.want)
		}
	}
}

// TestConcurrentWriteManifestRefIsSafeAcrossDistinctRefs checks that
// writing many distinct refs concurrently loses none of them: since
// every ref has its own independent file, there's no shared
// read-modify-write cycle to race on, and no locking involved.
func TestConcurrentWriteManifestRefIsSafeAcrossDistinctRefs(t *testing.T) {
	dir := t.TempDir()
	const n = 25

	var wg sync.WaitGroup
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ref := fmt.Sprintf("docker.io/ai/model-%d:latest", i)
			desc := ocispec.Descriptor{
				MediaType: ocispec.MediaTypeImageManifest,
				Digest:    digest.FromString(ref),
				Size:      int64(i),
			}
			if err := writeManifestRef(dir, ref, desc); err != nil {
				errs <- fmt.Errorf("writeManifestRef(%s): %w", ref, err)
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}

	for i := 0; i < n; i++ {
		ref := fmt.Sprintf("docker.io/ai/model-%d:latest", i)
		desc, err := readManifestRef(dir, ref)
		if err != nil {
			t.Errorf("readManifestRef(%s) after %d concurrent writes: %v (lost a concurrent write)", ref, n, err)
			continue
		}
		if want := digest.FromString(ref); desc.Digest != want {
			t.Errorf("readManifestRef(%s) = %s, want %s", ref, desc.Digest, want)
		}
	}
}

// TestConcurrentWriteManifestRefToSameRefNeverCorruptsIt is a regression
// test: two concurrent writers of the *same* ref must each produce a
// valid, parsable file — never a truncated or interleaved one from
// racing on a shared temp file name.
func TestConcurrentWriteManifestRefToSameRefNeverCorruptsIt(t *testing.T) {
	dir := t.TempDir()
	const n = 25

	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			desc := ocispec.Descriptor{
				MediaType: ocispec.MediaTypeImageManifest,
				Digest:    digest.FromString(fmt.Sprintf("v%d", i)),
				Size:      int64(i),
			}
			if err := writeManifestRef(dir, "docker.io/ai/same:latest", desc); err != nil {
				t.Errorf("writeManifestRef: %v", err)
			}
		}(i)
	}
	wg.Wait()

	if _, err := readManifestRef(dir, "docker.io/ai/same:latest"); err != nil {
		t.Fatalf("readManifestRef after concurrent writes: %v", err)
	}
}

func TestFindManifestForPushMatchesExactRefFirst(t *testing.T) {
	dir := t.TempDir()
	ref := "docker.io/ai/qwen3.5:0.8b"
	desc := ocispec.Descriptor{MediaType: ocispec.MediaTypeImageManifest, Digest: digest.FromString(ref), Size: 42}
	if err := writeManifestRef(dir, ref, desc); err != nil {
		t.Fatalf("writeManifestRef: %v", err)
	}
	got, err := findManifestForPush(dir, ref)
	if err != nil {
		t.Fatalf("findManifestForPush: %v", err)
	}
	if got.Digest != desc.Digest {
		t.Fatalf("got digest %s, want %s", got.Digest, desc.Digest)
	}
}

// TestFindManifestForPushRejectsAMismatchedRef checks the push path
// never guesses: a mistyped/mismatched ref must error rather than
// silently pushing whatever else happens to be the only thing in the
// store. A staged transfer, which legitimately pulls under one ref and
// pushes under another, records the staged model under the destination
// ref first (crate::sources::transfer) rather than relying on any
// fallback here.
func TestFindManifestForPushRejectsAMismatchedRef(t *testing.T) {
	dir := t.TempDir()
	ref := "docker.io/ai/qwen3.5:0.8b"
	desc := ocispec.Descriptor{MediaType: ocispec.MediaTypeImageManifest, Digest: digest.FromString(ref), Size: 42}
	if err := writeManifestRef(dir, ref, desc); err != nil {
		t.Fatalf("writeManifestRef: %v", err)
	}
	if _, err := findManifestForPush(dir, "ghcr.io/someone/else:v1"); err == nil {
		t.Fatalf("expected an error for a ref that doesn't match the store's only entry")
	}
}

// TestFindManifestForPushDistinguishesBrokenFromMissing checks a
// corrupt ref file (present, but unparsable) doesn't get the same "no
// manifest found" message as one that was simply never tagged — a user
// needs to tell a broken store apart from a typo.
func TestFindManifestForPushDistinguishesBrokenFromMissing(t *testing.T) {
	dir := t.TempDir()
	ref := "docker.io/ai/broken:latest"
	path := manifestRefPath(dir, ref)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("not json"), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := findManifestForPush(dir, ref)
	if err == nil || strings.Contains(err.Error(), "no manifest found") {
		t.Fatalf("findManifestForPush = %v, want an error distinct from \"no manifest found\"", err)
	}
}

// TestWriteManifestRefWithAnEmptyTagDoesNotBreakOtherTags is a
// regression test: writing a ref with an empty tag ("repo:") must not
// clobber the repo's own directory, or every subsequent tag of that
// repo would fail.
func TestWriteManifestRefWithAnEmptyTagDoesNotBreakOtherTags(t *testing.T) {
	dir := t.TempDir()
	empty := ocispec.Descriptor{MediaType: ocispec.MediaTypeImageManifest, Digest: digest.FromString("empty"), Size: 1}
	if err := writeManifestRef(dir, "docker.io/ai/x:", empty); err != nil {
		t.Fatalf("writeManifestRef with an empty tag: %v", err)
	}

	latest := ocispec.Descriptor{MediaType: ocispec.MediaTypeImageManifest, Digest: digest.FromString("latest"), Size: 2}
	if err := writeManifestRef(dir, "docker.io/ai/x:latest", latest); err != nil {
		t.Fatalf("writeManifestRef for a normal tag of the same repo: %v", err)
	}
	if got, err := readManifestRef(dir, "docker.io/ai/x:latest"); err != nil || got.Digest != latest.Digest {
		t.Fatalf("readManifestRef(docker.io/ai/x:latest) = %+v, %v", got, err)
	}
}
