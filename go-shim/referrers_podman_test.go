//go:build podman

package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/google/go-containerregistry/pkg/name"
	"github.com/google/go-containerregistry/pkg/registry"
	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"go.podman.io/image/v5/pkg/sysregistriesv2"
	"go.podman.io/image/v5/transports/alltransports"
	"go.podman.io/image/v5/types"
)

func TestPodmanReferrersUsePrimaryRemappedRepository(t *testing.T) {
	seenPaths := make(chan string, 1)
	const username, password = "physical-user", "physical-secret"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v2/" {
			return
		}
		if user, pass, ok := r.BasicAuth(); !ok || user != username || pass != password {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		seenPaths <- r.URL.Path
		_, _ = w.Write([]byte(`{"schemaVersion":2,"manifests":[]}`))
	}))
	defer srv.Close()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	conf := filepath.Join(t.TempDir(), "registries.conf")
	contents := `[[registry]]
prefix = "logical.example/org"
location = "` + u.Host + `/remapped"
insecure = true
`
	if err := os.WriteFile(conf, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("CONTAINERS_REGISTRIES_CONF", conf)
	authDir := t.TempDir()
	authFile := filepath.Join(authDir, "config.json")
	auth := base64.StdEncoding.EncodeToString([]byte(username + ":" + password))
	credentials := `{"auths":{"` + u.Host + `":{"auth":"` + auth + `"}}}`
	if err := os.WriteFile(authFile, []byte(credentials), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("DOCKER_CONFIG", authDir)
	sysregistriesv2.InvalidateCache() // Podman caches config independently of env changes.
	t.Cleanup(sysregistriesv2.InvalidateCache)
	logical, err := name.NewRepository("logical.example/org/model", name.WeakValidation)
	if err != nil {
		t.Fatal(err)
	}
	resolved, allowHTTP, err := registryReadRepository(context.Background(), logical)
	if err != nil || resolved.String() != u.Host+"/remapped/model" || !allowHTTP {
		t.Fatalf("resolved = %s, allowHTTP = %t, %v", resolved, allowHTTP, err)
	}
	client, err := registryClient(context.Background(), logical)
	if err != nil {
		t.Fatal(err)
	}
	endpoint := registryURLWithScheme(client.repo, client.scheme, "referrers", digest.FromString("subject").String())
	if _, found, err := getReferrersIndex(context.Background(), client.client, endpoint); err != nil || !found {
		t.Fatalf("remapped referrers read = %t, %v", found, err)
	}
	seenPath := <-seenPaths
	if !strings.HasPrefix(seenPath, "/v2/remapped/model/referrers/") {
		t.Fatalf("referrers request path = %q, want remapped repository", seenPath)
	}
}

func TestPodmanReferrersLoadImplicitPerHostCA(t *testing.T) {
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {}))
	defer srv.Close()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	certDir := filepath.Join(root, "etc/containers/certs.d", u.Host)
	if err := os.MkdirAll(certDir, 0o755); err != nil {
		t.Fatal(err)
	}
	ca := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: srv.Certificate().Raw})
	if err := os.WriteFile(filepath.Join(certDir, "ca.crt"), ca, 0o644); err != nil {
		t.Fatal(err)
	}
	repo, err := name.NewRepository(u.Host+"/org/model", name.WeakValidation)
	if err != nil {
		t.Fatal(err)
	}
	sys := &types.SystemContext{RootForImplicitAbsolutePaths: root}
	transport, err := podmanRegistryTransport(sys, repo, repo)
	if err != nil {
		t.Fatal(err)
	}
	client := &http.Client{Transport: transport}
	resp, err := client.Get(srv.URL)
	if err != nil {
		t.Fatalf("TLS request with implicit CA: %v", err)
	}
	resp.Body.Close()
}

func TestPodmanReferrersAllowHTTPOnlyWhenInsecure(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v2/" {
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	base := http.DefaultTransport.(*http.Transport).Clone()
	base.Proxy = nil
	base.DialContext = func(ctx context.Context, network, _ string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, network, u.Host)
	}
	repo, err := name.NewRepository("registry.example/org/model", name.WeakValidation)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := registryClientWithTransport(context.Background(), repo, false, base); err == nil {
		t.Fatal("HTTP-only registry accepted without insecure configuration")
	}
	client, err := registryClientWithTransport(context.Background(), repo, true, base)
	if err != nil {
		t.Fatal(err)
	}
	if client.repo.Scheme() != "http" {
		t.Fatalf("selected scheme = %s, want http", client.repo.Scheme())
	}
	if found, err := probeReferrersEndpoint(context.Background(), client.client, registryURLWithScheme(client.repo, client.scheme, "referrers", digest.FromString("subject").String())); err != nil || found {
		t.Fatalf("HTTP referrers probe = %t, %v; want fallback 404", found, err)
	}
}

func TestPodmanArtifactPushRefusesManifestDigestRewrite(t *testing.T) {
	var ociPuts atomic.Int32
	var convertedPuts atomic.Int32
	reg := registry.New(registry.Logger(log.New(io.Discard, "", 0)))
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPut && strings.Contains(r.URL.Path, "/manifests/") {
			if r.Header.Get("Content-Type") == ocispec.MediaTypeImageManifest {
				ociPuts.Add(1)
				http.Error(w, "OCI manifests rejected for test", http.StatusUnsupportedMediaType)
				return
			}
			convertedPuts.Add(1)
		}
		reg.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}

	ref := u.Host + "/org/model:sbom"
	dir := t.TempDir()
	if err := ensureLayout(dir); err != nil {
		t.Fatal(err)
	}
	config, err := writeBlob(dir, "application/vnd.example.sbom.config.v1+json", []byte(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	layer, err := writeBlob(dir, "application/vnd.example.sbom.v1+json", []byte(`{"name":"test"}`))
	if err != nil {
		t.Fatal(err)
	}
	manifest := ocispec.Manifest{
		MediaType:    ocispec.MediaTypeImageManifest,
		ArtifactType: "application/vnd.example.sbom.v1",
		Config:       config,
		Layers:       []ocispec.Descriptor{layer},
		Subject: &ocispec.Descriptor{
			MediaType: ocispec.MediaTypeImageManifest,
			Digest:    digest.FromString("subject"),
			Size:      123,
		},
	}
	manifest.SchemaVersion = 2
	raw, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	desc, err := writeBlob(dir, ocispec.MediaTypeImageManifest, raw)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeManifestRef(dir, ref, desc); err != nil {
		t.Fatal(err)
	}

	destinationCtx := &types.SystemContext{DockerInsecureSkipTLSVerify: types.OptionalBoolTrue}
	if _, err := pushDescriptorToRegistryWithDestinationContext(context.Background(), dir, ref, desc, destinationCtx, true); err == nil {
		t.Fatal("push succeeded by rewriting an OCI artifact the registry rejected")
	}
	if ociPuts.Load() == 0 {
		t.Fatal("test registry never received the OCI artifact manifest")
	}
	if convertedPuts.Load() != 0 {
		t.Fatalf("push retried with %d digest-changing converted manifest(s)", convertedPuts.Load())
	}
}

func TestPodmanPushesFallbackIndexWithoutArtifactBlobs(t *testing.T) {
	reg := registry.New(registry.Logger(log.New(io.Discard, "", 0)))
	srv := httptest.NewTLSServer(reg)
	defer srv.Close()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	ref := u.Host + "/org/model:sha256-" + strings.Repeat("a", 64)
	sys := &types.SystemContext{DockerInsecureSkipTLSVerify: types.OptionalBoolTrue}
	artifactDir := t.TempDir()
	if err := ensureLayout(artifactDir); err != nil {
		t.Fatal(err)
	}
	config, err := writeBlob(artifactDir, "application/vnd.example.config.v1+json", []byte(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	artifactManifest := ocispec.Manifest{MediaType: ocispec.MediaTypeImageManifest, Config: config}
	artifactManifest.SchemaVersion = 2
	artifactRaw, err := json.Marshal(artifactManifest)
	if err != nil {
		t.Fatal(err)
	}
	artifact, err := writeBlob(artifactDir, ocispec.MediaTypeImageManifest, artifactRaw)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pushDescriptorToRegistryWithDestinationContext(context.Background(), artifactDir, u.Host+"/org/model:artifact", artifact, sys, true); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := ensureLayout(dir); err != nil {
		t.Fatal(err)
	}
	index := ocispec.Index{Manifests: []ocispec.Descriptor{artifact}}
	index.SchemaVersion = 2
	index.MediaType = ocispec.MediaTypeImageIndex
	raw, err := json.Marshal(index)
	if err != nil {
		t.Fatal(err)
	}
	desc, err := writeBlob(dir, ocispec.MediaTypeImageIndex, raw)
	if err != nil {
		t.Fatal(err)
	}
	if err := pushIndexToRegistryWithDestinationContext(context.Background(), dir, ref, desc, sys); err != nil {
		t.Fatal(err)
	}
	srcRef, err := alltransports.ParseImageName("docker://" + ref)
	if err != nil {
		t.Fatal(err)
	}
	src, err := srcRef.NewImageSource(context.Background(), sys)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	got, _, err := src.GetManifest(context.Background(), nil)
	if err != nil || string(got) != string(raw) {
		t.Fatalf("published index = %s, %v; want %s", got, err, raw)
	}
}
