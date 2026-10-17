// Package main is the CGO entrypoint for the llmman Go shim.
// It is compiled as a C static archive and linked into the Rust binary.
// Build tags select either the Docker (containerd) or Podman backend.
package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"unsafe"

	"github.com/sirupsen/logrus"
)

// LLMMAN_DEBUG=1 (or any non-empty value) surfaces containerd's own
// request/response debug logging — including the exact registry scope
// requested and the token/auth challenge exchange — which is otherwise
// only logged at debug level and invisible by default. Useful for
// diagnosing registry-side auth/scope rejections (401/403/"insufficient_
// scope") without needing to instrument anything by hand.
func init() {
	if os.Getenv("LLMMAN_DEBUG") != "" {
		logrus.SetLevel(logrus.DebugLevel)
	}
	logrus.SetOutput(&filteredLogOutput{out: os.Stderr})
}

// filteredLogOutput drops specific known-benign containerd log lines that
// would otherwise print unconditionally on every push/transfer and look
// like something is wrong when nothing is. Every model layer/config
// llmman pushes uses a custom CNCF ModelPack media type (e.g.
// application/vnd.cncf.model.weight.v1.raw) rather than one of the
// standard OCI media types containerd's remotes.MakeRefKey recognizes
// (layer/config/manifest/index) — MakeRefKey only uses that recognition
// to pick a prefix for an internal upload-tracking key, and falls back to
// "unknown-<digest>" perfectly correctly when it doesn't recognize one,
// but also unconditionally logs a warning on that path
// (containerd/v2/core/remotes/handlers.go's MakeRefKey) with no way to
// suppress it from the caller's side (it's not gated by log level, and
// there's no exported option to opt out). The push/transfer itself is
// entirely unaffected either way; the warning has no diagnostic value.
type filteredLogOutput struct {
	out *os.File
}

func (w *filteredLogOutput) Write(p []byte) (int, error) {
	if bytes.Contains(p, []byte("reference for unknown type")) {
		return len(p), nil
	}
	return w.out.Write(p)
}

// response is the JSON envelope returned by every exported function.
// Rust deserialises this to decide success/failure.
type response struct {
	OK    bool   `json:"ok"`
	Data  string `json:"data,omitempty"`
	Error string `json:"error,omitempty"`
	// NotFound: the registry said the image does not exist (see
	// imageNotFoundError).
	NotFound bool `json:"not_found,omitempty"`
}

// imageNotFoundError marks a failure as the registry's "no such image", so
// the daemon need not match the backend-specific wording.
// Its message is the wrapped error's.
type imageNotFoundError struct{ err error }

func (e imageNotFoundError) Error() string { return e.err.Error() }
func (e imageNotFoundError) Unwrap() error { return e.err }

// isDockerHubRef reports whether ref is on Docker Hub, which answers an
// anonymous pull of a missing repository as it does a private one. Another
// registry's anonymous denial is an auth failure, not a missing image.
func isDockerHubRef(ref string) bool {
	host, _, _ := strings.Cut(ref, "/")
	switch host {
	case "docker.io", "index.docker.io", "registry-1.docker.io":
		return true
	}
	return false
}

// markImageNotFound is err, marked as the registry's "no such image".
func markImageNotFound(err error) error { return imageNotFoundError{err} }

// markIfMissing is err, marked as a missing image if the registry said so
// about ref.
func markIfMissing(err error, ref string) error {
	if isImageNotFound(err, ref) {
		return markImageNotFound(err)
	}
	return err
}

func okResp(data string) *C.char {
	b, _ := json.Marshal(response{OK: true, Data: data})
	return C.CString(string(b))
}

func errResp(err error) *C.char {
	b, _ := json.Marshal(failure(err))
	return C.CString(string(b))
}

// failure is the response for err, which reports not_found if it is marked.
func failure(err error) response {
	var notFound imageNotFoundError
	return response{Error: err.Error(), NotFound: errors.As(err, &notFound)}
}

func errMsg(msg string) *C.char {
	b, _ := json.Marshal(response{OK: false, Error: msg})
	return C.CString(string(b))
}

// llmman_free releases a C string previously returned by this library.
//
//export llmman_free
func llmman_free(s *C.char) {
	C.free(unsafe.Pointer(s))
}

// main is required for -buildmode=c-archive.
func main() {}
