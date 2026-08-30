// shared_oci.go – OCI layout helpers used by both the docker and podman backends.
// No build tag: compiled for all configurations.

package main

import (
	"context"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/containerd/containerd/v2/core/content"
	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/vbauerster/mpb/v8"
	"github.com/vbauerster/mpb/v8/decor"
	"golang.org/x/sync/singleflight"
)

// blobPath returns the path for a blob in an OCI image layout directory.
func blobPath(layoutDir string, dgst digest.Digest) string {
	return filepath.Join(layoutDir, "blobs", dgst.Algorithm().String(), dgst.Hex())
}

// readBlob reads a blob from an OCI layout directory.
func readBlob(layoutDir string, dgst digest.Digest) ([]byte, error) {
	return os.ReadFile(blobPath(layoutDir, dgst))
}

// writeBlob atomically writes data to the OCI layout blobs directory.
func writeBlob(layoutDir string, mediaType string, data []byte) (ocispec.Descriptor, error) {
	dgst := digest.FromBytes(data)
	dir := filepath.Join(layoutDir, "blobs", dgst.Algorithm().String())
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return ocispec.Descriptor{}, err
	}
	dest := filepath.Join(dir, dgst.Hex())
	if fi, err := os.Stat(dest); err == nil && fi.Size() == int64(len(data)) {
		return ocispec.Descriptor{MediaType: mediaType, Digest: dgst, Size: int64(len(data))}, nil
	}
	tmp := dest + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return ocispec.Descriptor{}, err
	}
	if err := os.Rename(tmp, dest); err != nil {
		return ocispec.Descriptor{}, err
	}
	return ocispec.Descriptor{MediaType: mediaType, Digest: dgst, Size: int64(len(data))}, nil
}

// openForResume opens path for appending if a previously-partial download of
// resumeFrom bytes already exists there, re-hashing those bytes into the
// returned digester so the digest computed over subsequent writes still
// spans the whole file; otherwise (no existing partial, or re-hashing it
// fails) it creates path fresh with a zeroed digester. The returned offset is
// resumeFrom on a successful resume, or 0 if resume wasn't possible. Shared
// by writeBlobStream (OCI-to-OCI transfer) and downloadAttempt (HuggingFace
// pull), which both append to a deterministic ".part" file across retries.
func openForResume(path string, resumeFrom int64) (f *os.File, digester digest.Digester, offset int64, err error) {
	digester = digest.Canonical.Digester()

	if resumeFrom > 0 {
		if pf, openErr := os.Open(path); openErr == nil {
			_, hashErr := io.Copy(digester.Hash(), pf)
			pf.Close()
			if hashErr == nil {
				if af, appendErr := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644); appendErr == nil {
					f = af
					offset = resumeFrom
				}
			}
		}
		if f == nil {
			digester = digest.Canonical.Digester()
		}
	}
	if f == nil {
		if f, err = os.Create(path); err != nil {
			return nil, nil, 0, err
		}
	}
	return f, digester, offset, nil
}

// writeBlobStream writes a large stream to the OCI layout blobs directory with
// resume support via a deterministic .part file.
func writeBlobStream(layoutDir, mediaType string, r io.Reader, size int64, dgst digest.Digest, partOffset int64) (ocispec.Descriptor, error) {
	dir := filepath.Join(layoutDir, "blobs", dgst.Algorithm().String())
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return ocispec.Descriptor{}, err
	}
	dest := filepath.Join(dir, dgst.Hex())
	if fi, err := os.Stat(dest); err == nil && (size <= 0 || fi.Size() == size) {
		return ocispec.Descriptor{MediaType: mediaType, Digest: dgst, Size: fi.Size()}, nil
	}
	tmp := dest + ".part"

	f, digester, startOffset, err := openForResume(tmp, partOffset)
	if err != nil {
		return ocispec.Descriptor{}, err
	}

	written, err := io.Copy(io.MultiWriter(f, digester.Hash()), r)
	f.Close()
	if err != nil {
		os.Remove(tmp)
		return ocispec.Descriptor{}, err
	}
	total := startOffset + written
	if size > 0 && total != size {
		os.Remove(tmp)
		return ocispec.Descriptor{}, fmt.Errorf("size mismatch: expected %d got %d", size, total)
	}
	if got := digester.Digest(); got != dgst {
		os.Remove(tmp)
		return ocispec.Descriptor{}, fmt.Errorf("digest mismatch: expected %s got %s", dgst, got)
	}
	if err := os.Rename(tmp, dest); err != nil {
		os.Remove(tmp)
		return ocispec.Descriptor{}, err
	}
	return ocispec.Descriptor{MediaType: mediaType, Digest: dgst, Size: total}, nil
}

// blobExists reports whether a blob is already fully stored in the layout.
func blobExists(layoutDir string, desc ocispec.Descriptor) bool {
	fi, err := os.Stat(blobPath(layoutDir, desc.Digest))
	return err == nil && fi.Size() == desc.Size
}

// ---------------------------------------------------------------------------
// Retry/stall-detection primitives for the only download and transfer
// paths left in this shim: backend_docker.go's pullToLayout, and
// transfer_docker.go's dockerTransferOCI (streamed straight into a
// registry push, NOT resumable — see transfer_docker.go's own comment on
// why — so it retries a failed blob from scratch instead of resuming it).
// ---------------------------------------------------------------------------

const (
	dlMaxAttempts  = 8               // generous budget: transient blips must never fail a whole transfer
	dlRetryBase    = 1 * time.Second // doubles each retry: 1s, 2s, 4s, 8s, 16s, 32s, 64s
	dlStallTimeout = 60 * time.Second
)

// retryDelay returns the backoff delay before retry attempt i (1-indexed:
// i=1 is the delay before the 2nd overall attempt), doubling each time
// from dlRetryBase and randomized by ±25% so several transfers that all
// hit a transient error around the same moment (e.g. a registry blip
// affecting every blob of one pull at once) don't all wake up and retry
// in the same instant — mirrors Ollama's own jittered backoff for
// redirect resolution and fast-transfer retries (see its server/
// download.go newBackoff and x/transfer/transfer.go backoff).
func retryDelay(attempt int) time.Duration {
	base := dlRetryBase * time.Duration(uint64(1)<<uint(attempt-1))
	// rand.Int63n(base) is uniform in [0, base); subtracting base/2 and
	// halving centers it as a ±25% offset around base.
	offset := time.Duration(rand.Int63n(int64(base))) - base/2
	return base + offset/2
}

// stallReader cancels the context if no bytes arrive within timeout.
// Mirrors llama.cpp's implicit stall detection via cpp-httplib timeouts.
//
// Slow-speed detection (cancel when far below the recent median rate) went
// with the HuggingFace downloader that fed it samples — see src/hf.
type stallReader struct {
	r      io.Reader
	timer  *time.Timer
	cancel context.CancelFunc
}

func newStallReader(r io.Reader, timeout time.Duration, cancel context.CancelFunc) *stallReader {
	sr := &stallReader{r: r, cancel: cancel}
	sr.timer = time.AfterFunc(timeout, cancel)
	return sr
}

func (sr *stallReader) Read(p []byte) (int, error) {
	n, err := sr.r.Read(p)
	if n > 0 {
		sr.timer.Reset(dlStallTimeout) // bytes arrived, reset stall clock
	}
	return n, err
}

func (sr *stallReader) stop() { sr.timer.Stop() }

// stallReadCloser pairs a stallReader with the underlying response body's
// Close, so callers can pass it around as a plain io.ReadCloser and have
// the stall timer stopped automatically whenever the body is closed
// (success, error, or early abandonment all go through Close).
type stallReadCloser struct {
	*stallReader
	body io.Closer
}

func newStallReadCloser(rc io.ReadCloser, timeout time.Duration, cancel context.CancelFunc) *stallReadCloser {
	return &stallReadCloser{stallReader: newStallReader(rc, timeout, cancel), body: rc}
}

func (s *stallReadCloser) Close() error {
	s.stop()
	return s.body.Close()
}

// stallWriter is stallReader's write-side mirror: it cancels the context
// if a Write call goes timeout without returning, rather than if no bytes
// arrive within timeout. content.Copy's read-then-write loop (see
// pushLazy in backend_docker.go, the only caller) has no way on its own
// to notice or escape a destination write that's stopped making progress
// — observed in practice as a registry PUT that goes quiet mid-upload
// following a transient error (a 502 from Docker Hub, in the one that
// prompted this) and then simply never completes or fails on its own.
// stallReader alone doesn't cover this: it only watches the source side
// of that same copy, and a source that's
// still delivering bytes just fine gives it nothing to notice while the
// destination write those bytes are headed into sits blocked. Nor does a
// plain http.Client's own Timeout help — the call actually blocked here
// (an io.Pipe.Write, feeding the goroutine that owns the real HTTP
// request) is on our side of an in-process pipe, not a network round
// trip the http.Client even sees as in progress.
type stallWriter struct {
	content.Writer
	timer  *time.Timer
	cancel context.CancelFunc
}

func newStallWriter(w content.Writer, timeout time.Duration, cancel context.CancelFunc) *stallWriter {
	return &stallWriter{Writer: w, timer: time.AfterFunc(timeout, cancel), cancel: cancel}
}

func (sw *stallWriter) Write(p []byte) (int, error) {
	n, err := sw.Writer.Write(p)
	if n > 0 {
		sw.timer.Reset(dlStallTimeout) // bytes accepted, reset stall clock
	}
	return n, err
}

func (sw *stallWriter) stop() { sw.timer.Stop() }

// isHTTP4xx returns true for permanent HTTP client errors (no point retrying).
func isHTTP4xx(err error) bool {
	if err == nil {
		return false
	}
	s := err.Error()
	for _, code := range []string{"HTTP 400", "HTTP 401", "HTTP 403", "HTTP 404"} {
		if strings.Contains(s, code) {
			return true
		}
	}
	return false
}

// retryStream calls attempt up to dlMaxAttempts times with jittered
// exponential backoff (see retryDelay), stopping immediately once
// isPermanent reports the most recent error isn't worth retrying (e.g. a
// 404 — see isHTTP4xx). Every attempt restarts its work from scratch:
// there's no partial state to pick up from here (see the callers' own
// comments for why).
//
// Retry-After honoring lived here only for HuggingFace downloads, which
// now retry natively in Rust (src/hf/client.rs); the remaining callers are
// OCI registry copies, which never carried one.
func retryStream(ctx context.Context, label string, isPermanent func(error) bool, attempt func() error) error {
	var lastErr error
	for i := 0; i < dlMaxAttempts; i++ {
		if i > 0 {
			delay := retryDelay(i)
			fmt.Fprintf(os.Stderr, "\n[llmman] retrying %s (attempt %d/%d, wait %v)\n", label, i+1, dlMaxAttempts, delay)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(delay):
			}
		}
		err := attempt()
		if err == nil {
			return nil
		}
		lastErr = err
		if isPermanent != nil && isPermanent(err) {
			break
		}
		fmt.Fprintf(os.Stderr, "[llmman] %s error: %v\n", label, err)
	}
	return fmt.Errorf("%s failed after %d attempts: %w", label, dlMaxAttempts, lastErr)
}

// progressTrackingReadCloser wraps an io.ReadCloser, feeding every
// successfully-read byte count into progressAddCompleted (see
// progress_state.go) under key, alongside whatever else it's already
// being used for (usually incrementing an mpb bar, via bar.ProxyReader —
// see proxyOrNop). This is what lets cmd::serve poll real byte-level
// pull/push progress out of the daemon process: every code path in this
// package that reports progress via an mpb bar goes through proxyOrNop,
// so this one wrapper covers all of them. key is the model reference the
// bytes belong to (see progressState's own doc comment on why progress is
// now tracked per key rather than one process-wide total) — pass "" for
// callers (namely `llmman transfer`) that don't want these bytes tracked
// at all.
type progressTrackingReadCloser struct {
	io.ReadCloser
	key string
}

func (p progressTrackingReadCloser) Read(b []byte) (int, error) {
	n, err := p.ReadCloser.Read(b)
	if n > 0 {
		progressAddCompleted(p.key, int64(n))
	}
	return n, err
}

// proxyOrNop wraps r in bar's progress-tracking proxy reader, falling back to
// a plain no-op-Close wrapper around r when the bar declines to proxy (e.g.
// a zero-total spinner bar). Every downloader in this package that reports
// progress via an mpb.Bar needs this same fallback. See
// progressTrackingReadCloser for what key means.
func proxyOrNop(bar *mpb.Bar, r io.Reader, key string) io.ReadCloser {
	if p := bar.ProxyReader(r); p != nil {
		return progressTrackingReadCloser{ReadCloser: p, key: key}
	}
	return io.NopCloser(r)
}

// newProgressPool creates an mpb.Progress bar pool with the output/refresh
// settings shared by every download/transfer path in llmman; only the bar
// width varies by call site (80 for pull, 40 for transfer).
func newProgressPool(width int) *mpb.Progress {
	return mpb.New(mpb.WithWidth(width), mpb.WithOutput(os.Stderr), mpb.WithRefreshRate(180*time.Millisecond))
}

// addLayerBar adds a progress bar into an existing mpb.Progress, and folds
// its size into key's progressState total (see progress_state.go) —
// every call site only ever creates a bar for a blob that's actually going
// to be transferred (see pushLazy's own comment on why), so this is the
// one place that needs to feed progressAddTotal. Pass "" for key to skip
// tracking (see progressTrackingReadCloser).
func addLayerBar(p *mpb.Progress, prefix, onComplete string, size int64, key string) *mpb.Bar {
	progressAddTotal(key, size)
	bar := p.AddBar(size,
		mpb.BarFillerClearOnComplete(),
		mpb.PrependDecorators(
			decor.OnComplete(decor.Name(prefix), onComplete),
		),
		mpb.AppendDecorators(
			decor.OnComplete(decor.CountersKibiByte("% .1f / % .1f"), ""),
			decor.OnComplete(decor.Name("  "), ""),
			decor.OnComplete(decor.AverageSpeed(decor.SizeB1024(0), "% .1f"), ""),
		),
	)
	if size <= 0 {
		bar.SetTotal(0, true)
	}
	return bar
}

// blobFetchGroup deduplicates concurrent fetch-and-write operations for
// the same underlying content (an OCI blob digest, or a source-file key)
// across every in-flight pull in this process.
//
// This became necessary once pulls of *different* models were allowed to
// run concurrently (see the Rust daemon's per-model lock registry in
// serve.rs, replacing what used to be one global PULL_LOCK serializing
// every pull/push in the process): two models can legitimately share an
// underlying blob (e.g. two quantizations of the same base model
// referencing the same tokenizer/config layer, or two tags of the same
// image), so without this, two concurrent pulls could both open and
// append to the exact same deterministic `.part` file at once —
// corrupting each other's output — or at best both redundantly download
// the same multi-gigabyte blob a second time.
var blobFetchGroup singleflight.Group

// dedupBlobFetch runs fetch (the actual network-read-plus-disk-write work
// for a blob/file identified by dedupKey) at most once across every
// concurrent caller sharing that key in this process — see
// blobFetchGroup. Regardless of whether this particular call is the one
// that actually ran fetch or a concurrent call (pulling a different
// model that happens to share this exact blob) got there first, the
// caller's own progressKey is credited size completed bytes exactly
// once: fetch itself is expected to report incremental progress via
// proxyOrNop as it streams (crediting the caller whose fetch actually
// ran), so dedupBlobFetch only tops up the *other* callers who piggybacked
// on that work and would otherwise see their own pull's progress get
// stuck just short of 100%. size is the number of bytes to credit in
// that case.
func dedupBlobFetch(dedupKey, progressKey string, size int64, fetch func() (ocispec.Descriptor, error)) (ocispec.Descriptor, error) {
	var streamed bool
	v, err, _ := blobFetchGroup.Do(dedupKey, func() (interface{}, error) {
		desc, err := fetch()
		if err == nil {
			streamed = true
		}
		return desc, err
	})
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	if !streamed {
		progressAddCompleted(progressKey, size)
	}
	return v.(ocispec.Descriptor), nil
}
