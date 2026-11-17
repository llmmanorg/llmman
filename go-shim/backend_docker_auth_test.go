//go:build !podman

package main

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"runtime"
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
// that directory, so docker/cli uses its plain-file store and the
// "auths" map is inspectable on disk.

// isolatedDockerConfigDir points docker/cli at a throwaway config dir
// for the duration of one test. SetDir rather than t.Setenv("DOCKER_CONFIG"):
// docker/cli reads that env var exactly once per process (sync.Once in
// config.Dir()), so the second test to set it would silently keep
// writing into the first test's already-deleted TempDir.
func isolatedDockerConfigDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	// With no auth in config.json docker/cli substitutes the platform's
	// credential store, i.e. the developer's real keychain. Any auth entry
	// keeps it on the plain-file store.
	seed := []byte(`{"auths":{"seed.invalid":{}}}`)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), seed, 0o600); err != nil {
		t.Fatal(err)
	}
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

// TestDockerHubCredentialsSkipOAuthTokens seeds the layout `docker login`'s
// web-based flow leaves: the Hub PAT under https://index.docker.io/v1/ and
// its OAuth access and refresh tokens under sibling keys. The file store
// resolves "index.docker.io" to any of the three in map order, and Hub
// rejects both tokens as registry credentials, so a pull used to fail with
// a 401 from auth.docker.io two runs in three.
func TestDockerHubCredentialsSkipOAuthTokens(t *testing.T) {
	isolatedDockerConfigDir(t)

	for server, password := range map[string]string{
		dockerIndexServer:                   "dckr_pat_hub",
		dockerIndexServer + "access-token":  "expired.jwt.token",
		dockerIndexServer + "refresh-token": "refresh..client",
	} {
		if err := dockerLogin(server, "hubuser", password); err != nil {
			t.Fatalf("dockerLogin(%q): %v", server, err)
		}
	}

	for _, host := range []string{"registry-1.docker.io", "index.docker.io", "docker.io"} {
		for range 50 {
			user, pass, err := dockerCredentials(host)
			if err != nil {
				t.Fatalf("dockerCredentials(%q): %v", host, err)
			}
			if user != "hubuser" || pass != "dckr_pat_hub" {
				t.Fatalf("dockerCredentials(%q) = (%q, %q), want (hubuser, dckr_pat_hub)", host, user, pass)
			}
		}
	}

	if err := dockerLogout(dockerIndexServer); err != nil {
		t.Fatalf("dockerLogout(%q): %v", dockerIndexServer, err)
	}
	for range 50 {
		if user, pass, err := dockerCredentials("registry-1.docker.io"); err != nil || user != "" || pass != "" {
			t.Fatalf("dockerCredentials(registry-1.docker.io) with only OAuth tokens stored = (%q, %q, %v), want empty", user, pass, err)
		}
	}
}

// TestDockerHubCredentialsSkipOAuthTokensFromNativeHelper: like
// docker-credential-osxkeychain, the fake helper matches "index.docker.io" to
// an OAuth entry by host and echoes the lookup name back as ServerURL, so only
// the secret identifies it.
func TestDockerHubCredentialsSkipOAuthTokensFromNativeHelper(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake credential helper is a shell script")
	}
	bin := t.TempDir()
	helper := `#!/bin/sh
[ "$1" = get ] || exit 1
read -r server
case "$server" in
  index.docker.io|https://index.docker.io/v1/access-token) secret=expired.jwt.token ;;
  https://index.docker.io/v1/refresh-token) secret=refresh..client ;;
  *) echo "credentials not found in native keychain"; exit 1 ;;
esac
printf '{"ServerURL":"%s","Username":"hubuser","Secret":"%s"}' "$server" "$secret"
`
	if err := os.WriteFile(filepath.Join(bin, "docker-credential-fake"), []byte(helper), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))

	// credHelpers puts only "index.docker.io" on the helper, so the OAuth keys
	// must be looked up in the store that returned the candidate.
	for _, config := range []string{`{"credsStore":"fake"}`, `{"credHelpers":{"index.docker.io":"fake"}}`} {
		dir := isolatedDockerConfigDir(t)
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o600); err != nil {
			t.Fatal(err)
		}
		// Positive control: the helper must actually return the token.
		got, err := dockercliconfig.LoadDefaultConfigFile(io.Discard).GetCredentialsStore("index.docker.io").Get("index.docker.io")
		if err != nil || got.Password != "expired.jwt.token" {
			t.Fatalf("%s: helper lookup = (%q, %v), want the access token", config, got.Password, err)
		}
		for _, host := range []string{"registry-1.docker.io", "index.docker.io", "docker.io"} {
			if user, pass, err := dockerCredentials(host); err != nil || user != "" || pass != "" {
				t.Fatalf("%s: dockerCredentials(%q) with only OAuth tokens stored = (%q, %q, %v), want empty", config, host, user, pass, err)
			}
		}
	}
}
