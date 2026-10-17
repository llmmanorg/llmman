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

// What containerd's resolver says when a registry refuses an anonymous pull.
func deniedError() error {
	denied := fmt.Errorf("server message: insufficient_scope: %w", docker.ErrInvalidAuthorization)
	return fmt.Errorf("resolve: %w",
		fmt.Errorf("pull access denied, repository does not exist or may require authorization: %w", denied))
}

// dockerLogins points docker/cli at a config.json holding a login for each
// of registries. Written by hand, not via dockerLogin: with no login in it
// docker/cli uses the machine's native credential helper, and on a
// developer's Mac that is the real keychain.
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
	const hub, other = "docker.io/ai/nope:1b", "registry.example.com/ai/nope:1b"
	missing := func(ref string) error {
		return fmt.Errorf("resolve: %w", fmt.Errorf("%s: %w", ref, errdefs.ErrNotFound))
	}

	t.Run("a 404 is not found whoever asks, wherever", func(t *testing.T) {
		dockerLogins(t, "docker.io")
		for _, ref := range []string{hub, other} {
			if !isImageNotFound(missing(ref), ref) {
				t.Errorf("a 404 for %s was not recognised", ref)
			}
		}
	})

	t.Run("an anonymous denial from Docker Hub is not found", func(t *testing.T) {
		dockerLogins(t, "other.example.com")
		if !isImageNotFound(deniedError(), hub) {
			t.Error("an anonymous denial was not recognised")
		}
	})

	t.Run("a denial of a stored login is not", func(t *testing.T) {
		dockerLogins(t, "docker.io")
		if isImageNotFound(deniedError(), hub) {
			t.Error("a refused login was reported as a missing image")
		}
	})

	t.Run("an anonymous denial from another registry is not", func(t *testing.T) {
		dockerLogins(t, "other.example.com")
		if isImageNotFound(deniedError(), other) {
			t.Error("a registry that wants a login was reported as missing the image")
		}
	})

	t.Run("anything else is not", func(t *testing.T) {
		dockerLogins(t, "other.example.com")
		for _, err := range []error{
			errors.New("resolve: dial tcp: lookup registry.example.com: no such host"),
			fmt.Errorf("resolve: %w", errdefs.ErrUnavailable),
			errors.New("exec: \"docker-credential-desktop\": executable file not found in $PATH"),
		} {
			if isImageNotFound(err, hub) {
				t.Errorf("%v was reported as a missing image", err)
			}
		}
	})
}
