//go:build !podman

// push_stream.go – lets Rust push several blobs (plus the final
// manifest) to a registry without ever handing Go a URL to fetch: Rust
// does all HuggingFace/Xet fetching now (src/hf/transfer.rs), and this
// is the one remaining seam where that data reaches containerd's
// registry-push machinery, which only exists in Go.
//
// Shape: llmman_push_session_open resolves destination once, returning
// a session handle. Each llmman_push_stream_open within that session
// first checks whether the destination already has the blob (beginPush,
// backend_docker.go) and, if so, answers {"exists": true}: no pipe, no
// goroutine, and Rust never fetches the source. Otherwise it creates a
// pipe, returns its *write* end's raw fd/HANDLE to Rust, and starts a
// goroutine that uploads from the other end — the registry never knows
// its source was a pipe rather than a file. Rust writes, closes its end, then calls
// llmman_push_stream_wait to block for the outcome.
// llmman_push_session_close releases the session once done.
//
// One resolver/pusher per *session*, not per blob: resolving a
// destination negotiates auth and (for a non-HTTPS-only host)
// containerd's HTTPS-then-HTTP-fallback probe — redoing that per blob
// measurably slowed, and against a slow host could starve, any
// multi-file transfer.
//
// Go creates the pipe, not Rust: Go's own os.Pipe already knows how to
// set one up correctly for its runtime poller on every platform this
// targets, notably Windows. Rust's side only needs a plain blocking
// write into a raw fd/HANDLE, which works the same either way.
package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"sync"

	"github.com/containerd/containerd/v2/core/remotes"
	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/vbauerster/mpb/v8"
)

// pushSession holds the resolver/pusher shared by every blob pushed
// within one llmman_push_session_open/llmman_push_session_close pair.
type pushSession struct {
	ctx    context.Context
	pusher remotes.Pusher
}

// pushStreamHandle tracks one in-flight streamed push between the
// llmman_push_stream_open call that started it and the later
// llmman_push_stream_wait call that collects its result.
type pushStreamHandle struct {
	done    chan struct{}
	changed bool
	err     error
}

var (
	pushStreamMu      sync.Mutex
	pushSessionNextID int64
	pushSessions      = map[int64]*pushSession{}
	pushStreamNextID  int64
	pushStreams       = map[int64]*pushStreamHandle{}
)

// llmman_push_session_open resolves destination once, returning a
// session handle for one or more subsequent llmman_push_stream_open
// calls against it — see this file's own doc comment for why sharing
// one resolver/pusher across a whole transfer matters. Call
// llmman_push_session_close with the returned handle once every blob
// (and the manifest) has been pushed.
//
//export llmman_push_session_open
func llmman_push_session_open(cDestination *C.char) *C.char {
	destination := C.GoString(cDestination)
	ctx := context.Background()
	resolver := newResolver(ctx)
	pusher, err := resolver.Pusher(ctx, normalizeTag(destination))
	if err != nil {
		return errResp(fmt.Errorf("create pusher: %w", err))
	}

	pushStreamMu.Lock()
	pushSessionNextID++
	id := pushSessionNextID
	pushSessions[id] = &pushSession{ctx: ctx, pusher: pusher}
	pushStreamMu.Unlock()

	data, _ := json.Marshal(struct {
		Session int64 `json:"session"`
	}{Session: id})
	return okResp(string(data))
}

// llmman_push_session_close releases a session opened by
// llmman_push_session_open. Safe to call even if some of that session's
// streams were never waited on (e.g. after an earlier error aborted the
// transfer) — it only forgets the session itself, not any handle still
// tracked in pushStreams.
//
//export llmman_push_session_close
func llmman_push_session_close(cSession C.longlong) *C.char {
	id := int64(cSession)
	pushStreamMu.Lock()
	delete(pushSessions, id)
	pushStreamMu.Unlock()
	return okResp("")
}

// pushStreamOpenResult is push_stream_open's data payload. If Exists,
// nothing was opened and FD/Handle must be ignored. Otherwise FD is the
// write end's raw fd (POSIX) or HANDLE (Windows), and Handle is what to
// pass to llmman_push_stream_wait.
type pushStreamOpenResult struct {
	Exists bool   `json:"exists"`
	FD     uint64 `json:"fd"`
	Handle int64  `json:"handle"`
}

// openPushStream is llmman_push_stream_open without the C marshaling
// (cgo is not usable from tests).
func openPushStream(session *pushSession, desc ocispec.Descriptor) (pushStreamOpenResult, error) {
	attempt, exists, err := beginPush(session.ctx, session.pusher, desc)
	if err != nil {
		return pushStreamOpenResult{}, err
	}
	if exists {
		return pushStreamOpenResult{Exists: true}, nil
	}

	readFile, writeFD, err := newRawPipe()
	if err != nil {
		attempt.close()
		return pushStreamOpenResult{}, fmt.Errorf("create pipe: %w", err)
	}

	handle := &pushStreamHandle{done: make(chan struct{})}
	pushStreamMu.Lock()
	pushStreamNextID++
	id := pushStreamNextID
	pushStreams[id] = handle
	pushStreamMu.Unlock()

	go func() {
		defer readFile.Close()
		pushErr := attempt.copy(desc, func() (io.Reader, *mpb.Bar, func(), error) {
			return readFile, nil, nil, nil
		})
		attempt.close()
		// Drain so a failed push gives the writer a clean EOF rather
		// than a broken pipe whose error would mask this one.
		_, _ = io.Copy(io.Discard, readFile)
		handle.changed = true
		handle.err = pushErr
		close(handle.done)
	}()

	return pushStreamOpenResult{FD: writeFD, Handle: id}, nil
}

// llmman_push_stream_open starts pushing one blob (or the final
// manifest — containerd's Pusher tells them apart by MediaType) within
// session. If the destination already has the blob it returns
// {"exists": true} and the caller must not fetch the content. Otherwise
// it returns a pipe's write end and the push runs on a background
// goroutine. Errors from the existence check (auth, network) surface
// here; upload and commit errors surface from llmman_push_stream_wait.
//
// annotationsJSON is an optional JSON object, or "" for none.
//
//export llmman_push_stream_open
func llmman_push_stream_open(cSession C.longlong, cMediaType, cDigest *C.char, cSize C.longlong, cAnnotationsJSON *C.char) *C.char {
	pushStreamMu.Lock()
	session, ok := pushSessions[int64(cSession)]
	pushStreamMu.Unlock()
	if !ok {
		return errMsg(fmt.Sprintf("no push session with handle %d (already closed, or never opened)", int64(cSession)))
	}

	desc := ocispec.Descriptor{
		MediaType: C.GoString(cMediaType),
		Digest:    digest.Digest(C.GoString(cDigest)),
		Size:      int64(cSize),
	}
	if raw := C.GoString(cAnnotationsJSON); raw != "" {
		if err := json.Unmarshal([]byte(raw), &desc.Annotations); err != nil {
			return errResp(fmt.Errorf("parse annotations: %w", err))
		}
	}

	result, err := openPushStream(session, desc)
	if err != nil {
		return errResp(err)
	}
	data, _ := json.Marshal(result)
	return okResp(string(data))
}

// llmman_push_stream_wait blocks until the push started by
// llmman_push_stream_open finishes. The caller must already have closed
// its end of the fd/HANDLE before calling this, or it hangs waiting for
// an EOF that never comes. Returns {"changed": bool}.
//
//export llmman_push_stream_wait
func llmman_push_stream_wait(cHandle C.longlong) *C.char {
	changed, err := waitPushStream(int64(cHandle))
	if err != nil {
		return errResp(err)
	}
	data, _ := json.Marshal(struct {
		Changed bool `json:"changed"`
	}{Changed: changed})
	return okResp(string(data))
}

// waitPushStream is llmman_push_stream_wait without the C marshaling.
func waitPushStream(id int64) (changed bool, err error) {
	pushStreamMu.Lock()
	handle, ok := pushStreams[id]
	if ok {
		delete(pushStreams, id)
	}
	pushStreamMu.Unlock()
	if !ok {
		return false, fmt.Errorf("no push stream with handle %d (already waited on, or never opened)", id)
	}

	<-handle.done
	if handle.err != nil {
		return false, handle.err
	}
	return handle.changed, nil
}
