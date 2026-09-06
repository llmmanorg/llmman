//go:build !podman

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	dockercliconfig "github.com/docker/cli/cli/config"
)

// Docker counterpart of backend_podman_auth_test.go. Unlike podman's
// commonauth.Login, dockerLogin never contacts the registry — it only
// writes the credential store — so both halves of the round-trip are
// reachable offline here, not just logout.
//
// A t.TempDir() keeps each test off the developer's real
// ~/.docker/config.json. No credsStore/credHelpers are configured in
// that fresh directory, so docker/cli falls back to its plain-file store
// and the "auths" map is inspectable on disk.

// isolatedDockerConfigDir points docker/cli at a throwaway config dir
// for the duration of one test. SetDir rather than t.Setenv("DOCKER_CONFIG"):
// docker/cli reads that env var exactly once per process (sync.Once in
// config.Dir()), so the second test to set it would silently keep
// writing into the first test's already-deleted TempDir.
func isolatedDockerConfigDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	prev := dockercliconfig.Dir()
	dockercliconfig.SetDir(dir)
	t.Cleanup(func() { dockercliconfig.SetDir(prev) })
	return dir
}

func dockerConfigJSON(t *testing.T, dir string) string {
	t.Helper()
	body, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatalf("read config.json: %v", err)
	}
	return string(body)
}

func TestDockerLoginStoresCredentials(t *testing.T) {
	const registry = "registry.example.com"
	dir := isolatedDockerConfigDir(t)

	if err := dockerLogin(registry, "user", "pass"); err != nil {
		t.Fatalf("dockerLogin(%q): %v", registry, err)
	}

	raw := dockerConfigJSON(t, dir)
	var cfg struct {
		Auths map[string]struct {
			Auth string `json:"auth"`
		} `json:"auths"`
	}
	if err := json.Unmarshal([]byte(raw), &cfg); err != nil {
		t.Fatalf("parse config.json %q: %v", raw, err)
	}
	if _, ok := cfg.Auths[registry]; !ok {
		t.Fatalf("no auths entry for %s after login: %s", registry, raw)
	}
	// docker/cli's file store base64-encodes "user:pass" into "auth";
	// the round-trip below via dockerCredentials is what asserts the
	// value decodes back correctly, so only presence is checked here.

	// The same lookup path pull/push use must see what login stored.
	user, pass, err := dockerCredentials(registry)
	if err != nil {
		t.Fatalf("dockerCredentials(%q): %v", registry, err)
	}
	if user != "user" || pass != "pass" {
		t.Fatalf("dockerCredentials(%q) = (%q, %q), want (user, pass)", registry, user, pass)
	}
}

// TestDockerLogoutRemovesCredentials is the direct equivalent of
// TestPodmanLogoutDoesNotPanicOnSuccess: seed a credential, log out,
// and assert the registry is gone from the store afterwards.
func TestDockerLogoutRemovesCredentials(t *testing.T) {
	const registry = "registry.example.com"
	dir := isolatedDockerConfigDir(t)

	if err := dockerLogin(registry, "user", "pass"); err != nil {
		t.Fatalf("seed via dockerLogin(%q): %v", registry, err)
	}
	if !strings.Contains(dockerConfigJSON(t, dir), registry) {
		t.Fatalf("precondition: %s missing from config.json after login", registry)
	}

	if err := dockerLogout(registry); err != nil {
		t.Fatalf("dockerLogout(%q): %v", registry, err)
	}

	if after := dockerConfigJSON(t, dir); strings.Contains(after, registry) {
		t.Fatalf("credentials for %s still present after logout: %s", registry, after)
	}
	if user, pass, err := dockerCredentials(registry); err != nil || user != "" || pass != "" {
		t.Fatalf("dockerCredentials(%q) after logout = (%q, %q, %v), want empty", registry, user, pass, err)
	}
}

// TestDockerHubLoginIsVisibleUnderConnectionHost pins the normalization
// dockerCredentials exists for: `llmman login docker.io` stores under
// "docker.io", but containerd hands the Credentials callback the actual
// connection host "registry-1.docker.io". Without dockerHubCredentialKeys
// every authenticated Hub push would silently run anonymously.
func TestDockerHubLoginIsVisibleUnderConnectionHost(t *testing.T) {
	isolatedDockerConfigDir(t)

	if err := dockerLogin("docker.io", "hubuser", "hubpass"); err != nil {
		t.Fatalf("dockerLogin(docker.io): %v", err)
	}

	for _, host := range []string{"registry-1.docker.io", "index.docker.io", "docker.io"} {
		user, pass, err := dockerCredentials(host)
		if err != nil {
			t.Fatalf("dockerCredentials(%q): %v", host, err)
		}
		if user != "hubuser" || pass != "hubpass" {
			t.Fatalf("dockerCredentials(%q) = (%q, %q), want (hubuser, hubpass)", host, user, pass)
		}
	}

	// A non-Hub host must not pick up Hub credentials.
	if user, pass, err := dockerCredentials("ghcr.io"); err != nil || user != "" || pass != "" {
		t.Fatalf("dockerCredentials(ghcr.io) = (%q, %q, %v), want empty", user, pass, err)
	}

	if err := dockerLogout("docker.io"); err != nil {
		t.Fatalf("dockerLogout(docker.io): %v", err)
	}
	if user, pass, err := dockerCredentials("registry-1.docker.io"); err != nil || user != "" || pass != "" {
		t.Fatalf("dockerCredentials(registry-1.docker.io) after logout = (%q, %q, %v), want empty", user, pass, err)
	}
}
