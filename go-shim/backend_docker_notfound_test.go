//go:build !podman

package main

import (
	"encoding/base64"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/containerd/containerd/v2/core/remotes/docker"
	"github.com/containerd/errdefs"
)

// What containerd's resolver says when Docker Hub refuses an anonymous
// pull of a repository that is not there (resolver.go).
func hubDeniedError() error {
	denied := fmt.Errorf("server message: insufficient_scope: %w", docker.ErrInvalidAuthorization)
	return fmt.Errorf("resolve: %w",
		fmt.Errorf("pull access denied, repository does not exist or may require authorization: %w", denied))
}

// dockerLogins points docker/cli at a config.json holding a login for
// each of registries. Written by hand rather than through dockerLogin: a
// config with no login at all makes docker/cli look for a native
// credential helper, and a developer's machine has one — the keychain
// would be both read and written by the test.
func dockerLogins(t *testing.T, registries ...string) {
	t.Helper()
	dir := isolatedDockerConfigDir(t)
	auth := base64.StdEncoding.EncodeToString([]byte("user:pass"))
	auths := ""
	for i, r := range registries {
		if i > 0 {
			auths += ","
		}
		auths += fmt.Sprintf("%q:{\"auth\":%q}", r, auth)
	}
	body := fmt.Sprintf("{\"auths\":{%s}}", auths)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestDockerIsImageNotFound(t *testing.T) {
	const ref = "registry.example.com/ai/nope:1b"
	missing := fmt.Errorf("resolve: %w", fmt.Errorf("%s: %w", ref, errdefs.ErrNotFound))

	t.Run("a 404 is not found whoever asks", func(t *testing.T) {
		dockerLogins(t, "other.example.com")
		if !isImageNotFound(missing, ref) {
			t.Error("containerd's not-found was not recognised")
		}
		dockerLogins(t, "registry.example.com")
		if !isImageNotFound(missing, ref) {
			t.Error("a 404 stopped being not found once credentials were stored")
		}
	})

	t.Run("a denial with no login for the registry is not found", func(t *testing.T) {
		dockerLogins(t, "other.example.com")
		if !isImageNotFound(hubDeniedError(), ref) {
			t.Error("an anonymous denial was not recognised")
		}
	})

	t.Run("a denial of a stored login is not", func(t *testing.T) {
		dockerLogins(t, "registry.example.com")
		if isImageNotFound(hubDeniedError(), ref) {
			t.Error("a refused login was reported as a missing image")
		}
	})

	t.Run("anything else is not", func(t *testing.T) {
		dockerLogins(t, "other.example.com")
		for _, err := range []error{
			errors.New("resolve: dial tcp: lookup registry.example.com: no such host"),
			fmt.Errorf("resolve: %w", errdefs.ErrUnavailable),
			errors.New("exec: \"docker-credential-desktop\": executable file not found in $PATH"),
		} {
			if isImageNotFound(err, ref) {
				t.Errorf("%v was reported as a missing image", err)
			}
		}
	})
}
