package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

// cloudFrontLikeServer mimics the real CloudFront/S3-fronted HF Xet-CAS
// bridge behavior for a large file: a GET with no Range header 400s;
// "Range: bytes=0-" (the whole file, phrased as a range) gets a 206.
func cloudFrontLikeServer(t *testing.T, body []byte) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Range") != "bytes=0-" {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write(body)
	}))
}

func TestDownloadAttemptSendsRangeHeaderOnFreshDownload(t *testing.T) {
	want := []byte("pretend this is a multi-gigabyte gguf shard")
	srv := cloudFrontLikeServer(t, want)
	defer srv.Close()

	dir := t.TempDir()
	tmpPath := filepath.Join(dir, "download.part")
	file := hfFile{Path: "model-Q4_K_M.gguf", Size: int64(len(want))}

	// ProxyReader (used by proxyOrNop) panics on a nil receiver, so this
	// needs a real (if throwaway) bar rather than nil.
	prog := newProgressPool(40)
	bar := addLayerBar(prog, "test", "test done", file.Size, "")

	desc, err := downloadAttempt(context.Background(), hfDownloadClient(), srv.URL, "", dir, tmpPath, 0, file, bar, "")
	if err != nil {
		bar.Abort(false) // avoid hanging prog.Wait() below on a never-completed bar
		prog.Wait()
		t.Fatalf("downloadAttempt: %v (fresh downloads must send Range too, not just resumes)", err)
	}
	prog.Wait()

	got, err := os.ReadFile(filepath.Join(dir, "blobs", desc.Digest.Algorithm().String(), desc.Digest.Hex()))
	if err != nil {
		t.Fatalf("reading downloaded blob: %v", err)
	}
	if string(got) != string(want) {
		t.Errorf("downloaded content = %q, want %q", got, want)
	}
}

func TestHFGetBytesSendsRangeHeaderOnFreshDownload(t *testing.T) {
	want := []byte("pretend this is a small config.json fetched through the same bug-prone path")
	srv := cloudFrontLikeServer(t, want)
	defer srv.Close()

	prog := newProgressPool(40)
	bar := addLayerBar(prog, "test", "test done", int64(len(want)), "")

	got, err := hfGetBytes(context.Background(), hfDownloadClient(), srv.URL, "", bar)
	if err != nil {
		bar.Abort(false)
		prog.Wait()
		t.Fatalf("hfGetBytes: %v (must send Range even for a fresh, whole-file request)", err)
	}
	prog.Wait()

	if string(got) != string(want) {
		t.Errorf("hfGetBytes = %q, want %q", got, want)
	}
}
