// diskspace.go – refusing a pull the store's disk has no room for, before
// any layer downloads. No build tag: compiled for all configurations.
// The Rust pulls (fsutil::ensure_room) word the error the same way.

package main

import (
	"fmt"
	"math"
	"os"

	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

// availableSpace is diskFree, swapped out by tests.
var availableSpace = diskFree

// layersMissing returns the bytes layers still need in layoutDir: a blob
// already stored at its size needs none, a digest listed twice counts
// once, and a .part left by an interrupted pull only what it lacks,
// since writeBlobStream resumes it.
func layersMissing(layoutDir string, layers []ocispec.Descriptor) uint64 {
	seen := make(map[digest.Digest]bool, len(layers))
	var needed uint64
	for _, layer := range layers {
		if seen[layer.Digest] || layer.Size <= 0 || blobExists(layoutDir, layer) {
			continue
		}
		seen[layer.Digest] = true
		need := layer.Size
		if fi, err := os.Stat(blobPath(layoutDir, layer.Digest) + ".part"); err == nil && fi.Size() < need {
			need -= fi.Size()
		}
		needed += uint64(need)
		if needed < uint64(need) { // wrapped: remote sizes are untrusted
			return math.MaxUint64
		}
	}
	return needed
}

// ensureRoom fails when needed bytes will not fit in what is free at
// layoutDir before ref is pulled there. Free space that could not be
// read passes: running out still fails the write, as it always has.
func ensureRoom(ref, layoutDir string, needed uint64) error {
	free, ok := availableSpace(layoutDir)
	if !ok || needed <= free {
		return nil
	}
	return fmt.Errorf("not enough disk space to pull %s: it needs %s, and %s is free at %s; "+
		"free some space (llmman rm) or set LLMMAN_MODELS to a store on a bigger disk",
		ref, humanSize(needed), humanSize(free), layoutDir)
}

// humanSize is the Rust side's fmt::human_size: decimal GB/MB/kB.
func humanSize(b uint64) string {
	switch {
	case b >= 1_000_000_000:
		return fmt.Sprintf("%.1f GB", float64(b)/1e9)
	case b >= 1_000_000:
		return fmt.Sprintf("%.1f MB", float64(b)/1e6)
	case b >= 1_000:
		return fmt.Sprintf("%.1f kB", float64(b)/1e3)
	default:
		return fmt.Sprintf("%d B", b)
	}
}
