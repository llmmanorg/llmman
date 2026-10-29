//go:build !podman && unix

// Tests for the streamed-push path HuggingFace transfers use: the
// destination must be asked whether it has a blob *before* any source
// bytes are fetched, so an unchanged model costs a HEAD, not a download.
// Run against the real in-memory registry (see startRegistryWithFault),
// counting the requests it actually receives.
//
// unix: the pipe's write end comes back as a raw fd, which these tests
// wrap with os.NewFile; Windows hands back a HANDLE instead.
package main

import (
	"context"
	"errors"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"

	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/vbauerster/mpb/v8"
)

func newTestPushSession(t *testing.T, ref string) *pushSession {
	t.Helper()
	ctx := context.Background()
	pusher, err := newResolver(ctx).Pusher(ctx, normalizeTag(ref))
	if err != nil {
		t.Fatalf("create pusher: %v", err)
	}
	return &pushSession{ctx: ctx, pusher: pusher}
}

func blobDescriptor(data []byte) ocispec.Descriptor {
	return ocispec.Descriptor{
		MediaType: "application/vnd.cncf.model.weight.v1.raw",
		Digest:    digest.FromBytes(data),
		Size:      int64(len(data)),
	}
}

func openStreams() int {
	pushStreamMu.Lock()
	defer pushStreamMu.Unlock()
	return len(pushStreams)
}

func fetchBlob(t *testing.T, host, repo string, dgst digest.Digest) (string, int) {
	t.Helper()
	resp, err := http.Get("http://" + host + "/v2/" + repo + "/blobs/" + dgst.String())
	if err != nil {
		t.Fatalf("GET blob: %v", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read blob: %v", err)
	}
	return string(body), resp.StatusCode
}

// An already-present blob is reported synchronously, with nothing
// created for the caller to write into or wait on, and no upload session
// ever started at the registry.
func TestOpenPushStreamReportsExistingBlob(t *testing.T) {
	log := &requestLog{}
	host := startLoggedRegistry(t, log, nil)
	session := newTestPushSession(t, host+"/org/model:v1")

	data := []byte("weights already in the registry")
	desc := blobDescriptor(data)
	if exists, err := pushBytes(session.ctx, session.pusher, desc, data); err != nil || exists {
		t.Fatalf("seed push = exists %t, err %v; want a fresh upload", exists, err)
	}
	// A new session, as a later transfer run has: containerd's pusher
	// remembers blobs it committed itself and answers those without
	// asking the registry, which would hide the HEAD this test is about.
	session = newTestPushSession(t, host+"/org/model:v1")

	const blobs = "/v2/org/model/blobs/"
	const uploads = "/v2/org/model/blobs/uploads/"
	headsBefore, uploadsBefore := log.count(http.MethodHead, blobs), log.count("", uploads)
	streamsBefore := openStreams()
	result, err := openPushStream(session, desc)
	if err != nil {
		t.Fatalf("openPushStream: %v", err)
	}
	if !result.Exists {
		t.Fatalf("result = %+v, want Exists", result)
	}
	if result.FD != 0 || result.Handle != 0 {
		t.Fatalf("result = %+v: an existing blob must not hand out a pipe or handle", result)
	}
	if n := openStreams(); n != streamsBefore {
		t.Fatalf("%d stream handles tracked, want %d: Exists must not register one", n, streamsBefore)
	}
	if n := log.count(http.MethodHead, blobs) - headsBefore; n != 1 {
		t.Fatalf("saw %d blob HEADs, want 1", n)
	}
	if n := log.count("", uploads) - uploadsBefore; n != 0 {
		t.Fatalf("saw %d upload requests, want 0", n)
	}
}

// A missing blob opens a pipe; what is written to it reaches the
// registry, and the push is reported as a change.
func TestOpenPushStreamUploadsMissingBlob(t *testing.T) {
	log := &requestLog{}
	host := startLoggedRegistry(t, log, nil)
	session := newTestPushSession(t, host+"/org/model:v1")

	data := []byte(strings.Repeat("streamed through the pipe ", 1000))
	desc := blobDescriptor(data)

	result, err := openPushStream(session, desc)
	if err != nil {
		t.Fatalf("openPushStream: %v", err)
	}
	if result.Exists || result.FD == 0 || result.Handle == 0 {
		t.Fatalf("result = %+v, want a pipe and a handle", result)
	}
	w := os.NewFile(uintptr(result.FD), "push-stream-write")
	if _, err := w.Write(data); err != nil {
		t.Fatalf("write to pipe: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close pipe: %v", err)
	}
	changed, err := waitPushStream(result.Handle)
	if err != nil {
		t.Fatalf("waitPushStream: %v", err)
	}
	if !changed {
		t.Fatal("changed = false, want true for a blob the registry did not have")
	}
	if got, status := fetchBlob(t, host, "org/model", desc.Digest); status != http.StatusOK || got != string(data) {
		t.Fatalf("registry blob = %d, %d bytes; want 200 with the %d bytes written", status, len(got), len(data))
	}
	if n := openStreams(); n != 0 {
		t.Fatalf("%d stream handles still tracked after wait", n)
	}
	if n := log.count(http.MethodPost, "/v2/org/model/blobs/uploads/"); n != 1 {
		t.Fatalf("saw %d upload sessions, want 1", n)
	}

	// Pushing the same blob again is now the exists path.
	again, err := openPushStream(session, desc)
	if err != nil || !again.Exists {
		t.Fatalf("second open = %+v, %v; want Exists", again, err)
	}
}

// A zero-byte blob goes through the same stream path (see the
// zero-length write in pushAttempt.copy).
func TestOpenPushStreamUploadsEmptyBlob(t *testing.T) {
	host := startRegistry(t)
	session := newTestPushSession(t, host+"/org/model:v1")
	desc := blobDescriptor(nil)

	result, err := openPushStream(session, desc)
	if err != nil || result.Exists {
		t.Fatalf("openPushStream = %+v, %v; want a pipe", result, err)
	}
	if err := os.NewFile(uintptr(result.FD), "push-stream-write").Close(); err != nil {
		t.Fatalf("close pipe: %v", err)
	}
	if changed, err := waitPushStream(result.Handle); err != nil || !changed {
		t.Fatalf("waitPushStream = %t, %v; want changed", changed, err)
	}
}

// A registry that fails the existence check fails the open itself, with
// no pipe and no handle left behind (previously the error only surfaced
// after the whole body had been written).
func TestOpenPushStreamSurfacesRegistryErrorAtOpen(t *testing.T) {
	host := startRegistryWithFault(t, func(w http.ResponseWriter, r *http.Request) bool {
		if r.Method == http.MethodHead && strings.Contains(r.URL.Path, "/blobs/") {
			w.WriteHeader(http.StatusInternalServerError)
			return true
		}
		return false
	})
	session := newTestPushSession(t, host+"/org/model:v1")

	before := openStreams()
	result, err := openPushStream(session, blobDescriptor([]byte("x")))
	if err == nil {
		t.Fatalf("openPushStream = %+v, want an error", result)
	}
	if n := openStreams(); n != before {
		t.Fatalf("%d stream handles tracked, want %d", n, before)
	}
}

func TestWaitPushStreamUnknownHandle(t *testing.T) {
	if _, err := waitPushStream(-1); err == nil {
		t.Fatal("waitPushStream of an unknown handle succeeded")
	}
}

// pushLazy, now beginPush plus copy, behaves as before for every other
// caller: open is never called for a blob the destination has, and is
// called (and cleaned up) for one it lacks.
func TestPushLazyExistingBlobNeverOpensSource(t *testing.T) {
	log := &requestLog{}
	host := startLoggedRegistry(t, log, nil)
	session := newTestPushSession(t, host+"/org/model:v1")
	data := []byte("already there")
	desc := blobDescriptor(data)
	if _, err := pushBytes(session.ctx, session.pusher, desc, data); err != nil {
		t.Fatalf("seed push: %v", err)
	}
	session = newTestPushSession(t, host+"/org/model:v1") // see TestOpenPushStreamReportsExistingBlob

	uploadsBefore := log.count("", "/v2/org/model/blobs/uploads/")
	opened := false
	exists, err := pushLazy(session.ctx, session.pusher, desc, func() (io.Reader, *mpb.Bar, func(), error) {
		opened = true
		return strings.NewReader(string(data)), nil, nil, nil
	})
	if err != nil || !exists {
		t.Fatalf("pushLazy = exists %t, err %v; want exists", exists, err)
	}
	if opened {
		t.Fatal("open was called for a blob the destination already has")
	}
	if n := log.count("", "/v2/org/model/blobs/uploads/") - uploadsBefore; n != 0 {
		t.Fatalf("saw %d upload requests, want 0", n)
	}
}

func TestPushLazyMissingBlobUploadsAndCleansUp(t *testing.T) {
	host := startRegistry(t)
	session := newTestPushSession(t, host+"/org/model:v1")
	data := []byte("not there yet")
	desc := blobDescriptor(data)

	cleaned := false
	exists, err := pushLazy(session.ctx, session.pusher, desc, func() (io.Reader, *mpb.Bar, func(), error) {
		return strings.NewReader(string(data)), nil, func() { cleaned = true }, nil
	})
	if err != nil || exists {
		t.Fatalf("pushLazy = exists %t, err %v; want an upload", exists, err)
	}
	if !cleaned {
		t.Fatal("cleanup was not called")
	}
	if got, status := fetchBlob(t, host, "org/model", desc.Digest); status != http.StatusOK || got != string(data) {
		t.Fatalf("registry blob = %d %q, want 200 %q", status, got, data)
	}
}

func TestPushLazyOpenErrorIsReturnedAndCleanedUp(t *testing.T) {
	host := startRegistry(t)
	session := newTestPushSession(t, host+"/org/model:v1")
	desc := blobDescriptor([]byte("never opened"))

	boom := errors.New("source unavailable")
	cleaned := false
	exists, err := pushLazy(session.ctx, session.pusher, desc, func() (io.Reader, *mpb.Bar, func(), error) {
		return nil, nil, func() { cleaned = true }, boom
	})
	if exists || !errors.Is(err, boom) {
		t.Fatalf("pushLazy = exists %t, err %v; want %v", exists, err, boom)
	}
	if !cleaned {
		t.Fatal("cleanup was not called after an open error")
	}
}
