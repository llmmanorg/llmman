//go:build podman

package main

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/docker/distribution/registry/api/errcode"
	v2 "github.com/docker/distribution/registry/api/v2"
)

func TestPodmanIsImageNotFound(t *testing.T) {
	const ref = "registry.example.com/ai/nope:1b"
	// How go.podman.io/image reports what the registry answered: the
	// first errcode.Error of the body, wrapped (registryHTTPResponseToError).
	registry := func(code errcode.ErrorCode) error {
		return fmt.Errorf("open image source: initializing source docker://%s: reading manifest 1b: %w",
			ref, code.WithMessage("answer"))
	}

	// The pull reads credentials from the usual auth files under $HOME
	// (c/image does not honour REGISTRY_AUTH_FILE), so the test gets a
	// $HOME of its own, with or without a login for the registry.
	home := func(t *testing.T, login bool) {
		dir := t.TempDir()
		t.Setenv("HOME", dir)
		t.Setenv("DOCKER_CONFIG", "")
		t.Setenv("XDG_RUNTIME_DIR", "")
		t.Setenv("XDG_CONFIG_HOME", "")
		if !login {
			return
		}
		body, _ := json.Marshal(map[string]any{"auths": map[string]any{
			"registry.example.com": map[string]string{
				"auth": base64.StdEncoding.EncodeToString([]byte("user:pass")),
			},
		}})
		if err := os.MkdirAll(filepath.Join(dir, ".docker"), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, ".docker", "config.json"), body, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	noCredentials := func(t *testing.T) { home(t, false) }
	withCredentials := func(t *testing.T) { home(t, true) }

	t.Run("manifest and name unknown are not found whoever asks", func(t *testing.T) {
		withCredentials(t)
		for _, code := range []errcode.ErrorCode{v2.ErrorCodeManifestUnknown, v2.ErrorCodeNameUnknown} {
			if !isImageNotFound(registry(code), ref) {
				t.Errorf("%v was not recognised", code)
			}
		}
	})

	t.Run("an anonymous denial is not found", func(t *testing.T) {
		noCredentials(t)
		for _, code := range []errcode.ErrorCode{errcode.ErrorCodeDenied, errcode.ErrorCodeUnauthorized} {
			if !isImageNotFound(registry(code), ref) {
				t.Errorf("an anonymous %v was not recognised", code)
			}
		}
	})

	t.Run("a denial of stored credentials is not", func(t *testing.T) {
		withCredentials(t)
		for _, code := range []errcode.ErrorCode{errcode.ErrorCodeDenied, errcode.ErrorCodeUnauthorized} {
			if isImageNotFound(registry(code), ref) {
				t.Errorf("a refused login (%v) was reported as a missing image", code)
			}
		}
	})

	t.Run("anything else is not", func(t *testing.T) {
		noCredentials(t)
		for _, err := range []error{
			errors.New("open image source: dial tcp: lookup registry.example.com: no such host"),
			registry(errcode.ErrorCodeTooManyRequests),
			registry(v2.ErrorCodeBlobUnknown),
		} {
			if isImageNotFound(err, ref) {
				t.Errorf("%v was reported as a missing image", err)
			}
		}
	})
}
