package main

import (
	"context"
	"testing"
)

// Regression check for a renamed HuggingFace repository: deepreinforce-ai
// renamed/moved to ornith-ai, so the old owner/repo now 307-redirects to
// the new one before ever reaching the resolve endpoint's actual
// CDN-bound redirect (the one carrying X-Linked-Etag/X-Linked-Size).
// Requires real network access to huggingface.co.
func TestHFHeadMetadataFollowsRepoRenameRedirect(t *testing.T) {
	ctx := context.Background()
	client := hfAPIClient()
	url := "https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B/resolve/main/model-00001-of-00016.safetensors"

	dgst, size, _, ok, err := hfHeadMetadata(ctx, client, url, "")
	if err != nil {
		t.Fatalf("hfHeadMetadata: %v", err)
	}
	if !ok {
		t.Fatal("expected ok=true (a usable sha256 digest) for a real LFS/Xet-backed file behind a repo-rename redirect, got ok=false — this is exactly the bug: a multi-gigabyte file would silently be sent down the in-memory-buffering fallback instead of streamed")
	}
	if size < 1<<30 { // this specific shard is ~4GB; sanity-check it's not some tiny/wrong value
		t.Errorf("expected a multi-gigabyte size, got %d bytes", size)
	}
	t.Logf("digest=%s size=%d", dgst, size)
}

// Confirms the fix above didn't regress the ordinary case this function
// was already handling fine: a repository that hasn't been renamed,
// where the CDN-bound redirect carrying X-Linked-Etag/X-Linked-Size is
// already the very first hop.
func TestHFHeadMetadataOrdinaryRepoStillWorks(t *testing.T) {
	ctx := context.Background()
	client := hfAPIClient()
	url := "https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q8_0.gguf"

	dgst, size, _, ok, err := hfHeadMetadata(ctx, client, url, "")
	if err != nil {
		t.Fatalf("hfHeadMetadata: %v", err)
	}
	if !ok {
		t.Fatal("expected ok=true for an ordinary, non-renamed LFS repository")
	}
	if size <= 0 {
		t.Errorf("expected a positive size, got %d", size)
	}
	t.Logf("digest=%s size=%d", dgst, size)
}
