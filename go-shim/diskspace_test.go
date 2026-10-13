package main

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

// Only what isn't stored yet counts, a repeated digest once, and a
// resumable .part only for what it still lacks.
func TestLayersMissingSkipsStoredRepeatedAndResumedBytes(t *testing.T) {
	dir := t.TempDir()
	desc := func(name string, size int64) ocispec.Descriptor {
		return ocispec.Descriptor{Digest: digest.FromString(name), Size: size}
	}
	stored, partial, absent := desc("stored", 100), desc("partial", 300), desc("absent", 50)
	if err := os.MkdirAll(filepath.Dir(blobPath(dir, stored.Digest)), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(blobPath(dir, stored.Digest), make([]byte, 100), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(blobPath(dir, partial.Digest)+".part", make([]byte, 120), 0o644); err != nil {
		t.Fatal(err)
	}
	got := layersMissing(dir, []ocispec.Descriptor{stored, partial, absent, absent, desc("unknown", 0)})
	if want := uint64(300 - 120 + 50); got != want {
		t.Fatalf("layersMissing = %d, want %d", got, want)
	}
}

// Sizes come from the registry, so a sum that wraps must not read as small.
func TestLayersMissingSaturatesOnOverflow(t *testing.T) {
	var layers []ocispec.Descriptor
	for _, name := range []string{"a", "b", "c"} {
		layers = append(layers, ocispec.Descriptor{Digest: digest.FromString(name), Size: math.MaxInt64})
	}
	if got := layersMissing(t.TempDir(), layers); got != math.MaxUint64 {
		t.Fatalf("layersMissing = %d, want saturated", got)
	}
}

func TestEnsureRoomRefusesOnlyWhatDoesNotFit(t *testing.T) {
	defer func(orig func(string) (uint64, bool)) { availableSpace = orig }(availableSpace)

	availableSpace = func(string) (uint64, bool) { return 10, true }
	if err := ensureRoom("m", "/models", 10); err != nil {
		t.Fatalf("an exact fit: %v", err)
	}
	availableSpace = func(string) (uint64, bool) { return 0, false }
	if err := ensureRoom("m", "/models", 11); err != nil {
		t.Fatalf("unknown free space should pass: %v", err)
	}
	availableSpace = func(string) (uint64, bool) { return 1_000_000_000, true }
	err := ensureRoom("docker.io/ai/m", "/models", 3_000_000_000)
	if err == nil {
		t.Fatal("3 GB into 1 GB free should be refused")
	}
	for _, want := range []string{"docker.io/ai/m", "3.0 GB", "1.0 GB", "/models", "LLMMAN_MODELS"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q lacks %q", err, want)
		}
	}
}

func TestDiskFreeReadsTheTempDir(t *testing.T) {
	if free, ok := diskFree(os.TempDir()); !ok || free == 0 {
		t.Fatalf("diskFree(%q) = %d, %v", os.TempDir(), free, ok)
	}
}
