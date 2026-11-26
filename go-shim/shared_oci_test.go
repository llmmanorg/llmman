package main

import (
	"os"
	"testing"
	"time"
)

func TestWriteBlobRefreshesTheAgeOfAReusedBlob(t *testing.T) {
	dir := t.TempDir()
	desc, err := writeBlob(dir, "application/octet-stream", []byte("layer"))
	if err != nil {
		t.Fatal(err)
	}
	path := blobPath(dir, desc.Digest)
	old := time.Now().Add(-2 * time.Hour)
	if err := os.Chtimes(path, old, old); err != nil {
		t.Fatal(err)
	}

	if _, err := writeBlob(dir, "application/octet-stream", []byte("layer")); err != nil {
		t.Fatal(err)
	}

	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if age := time.Since(fi.ModTime()); age > time.Minute {
		t.Fatalf("a reused blob kept its age: %v", age)
	}
}
