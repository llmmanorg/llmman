//go:build podman

package main

import (
	"errors"
	"fmt"
	"testing"

	"github.com/docker/distribution/registry/api/errcode"
	v2 "github.com/docker/distribution/registry/api/v2"
)

// imageNotFound, not isImageNotFound: whether a login is stored depends on
// the machine's credential files, which is c/image's to test.
func TestPodmanImageNotFound(t *testing.T) {
	const hub, other = "docker.io/ai/nope:1b", "registry.example.com/ai/nope:1b"
	// How go.podman.io/image wraps the registry's first errcode.Error.
	registry := func(code errcode.ErrorCode) error {
		return fmt.Errorf("open image source: reading manifest 1b: %w", code.WithMessage("answer"))
	}
	loggedIn := func() bool { return true }
	anonymous := func() bool { return false }
	neverAsked := func() bool {
		t.Error("asked whether a login is stored for an answer that does not depend on it")
		return false
	}
	denials := []errcode.ErrorCode{errcode.ErrorCodeDenied, errcode.ErrorCodeUnauthorized}

	t.Run("manifest and name unknown are not found whoever asks, wherever", func(t *testing.T) {
		for _, code := range []errcode.ErrorCode{v2.ErrorCodeManifestUnknown, v2.ErrorCodeNameUnknown} {
			for _, ref := range []string{hub, other} {
				if !imageNotFound(registry(code), ref, neverAsked) {
					t.Errorf("%v for %s was not recognised", code, ref)
				}
			}
		}
	})

	t.Run("an anonymous denial from Docker Hub is not found", func(t *testing.T) {
		for _, code := range denials {
			if !imageNotFound(registry(code), hub, anonymous) {
				t.Errorf("an anonymous %v was not recognised", code)
			}
		}
	})

	t.Run("a denial of a stored login is not", func(t *testing.T) {
		for _, code := range denials {
			if imageNotFound(registry(code), hub, loggedIn) {
				t.Errorf("a refused login (%v) was reported as a missing image", code)
			}
		}
	})

	t.Run("an anonymous denial from another registry is not", func(t *testing.T) {
		for _, code := range denials {
			if imageNotFound(registry(code), other, anonymous) {
				t.Errorf("%v from a registry that wants a login was reported as missing the image", code)
			}
		}
	})

	t.Run("anything else is not", func(t *testing.T) {
		for _, err := range []error{
			errors.New("open image source: dial tcp: lookup registry.example.com: no such host"),
			registry(errcode.ErrorCodeTooManyRequests),
			registry(v2.ErrorCodeBlobUnknown),
		} {
			if imageNotFound(err, hub, neverAsked) {
				t.Errorf("%v was reported as a missing image", err)
			}
		}
	})
}
