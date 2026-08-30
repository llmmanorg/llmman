package main

import (
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
// manifests/, so a torn write (or a race between two writers) to one
// model can never affect any other model's own file.
func TestManifestRefPathIsOneFilePerModel(t *testing.T) {
	dir := t.TempDir()
	a := manifestRefPath(dir, "docker.io/ai/qwen3.5:0.8b")
	b := manifestRefPath(dir, "docker.io/ai/gemma4:latest")
	if a == b {
		t.Fatalf("expected distinct paths for distinct refs, got %q for both", a)
	}
	if filepath.Dir(a) == filepath.Dir(b) {
		t.Fatalf("expected distinct parent directories, got %q for both", filepath.Dir(a))
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
