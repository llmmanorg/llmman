//go:build podman

package main

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/google/go-containerregistry/pkg/registry"
	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"go.podman.io/image/v5/types"
)

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
	if _, err := pushToRegistryWithDestinationContext(context.Background(), dir, ref, destinationCtx); err == nil {
		t.Fatal("push succeeded by rewriting an OCI artifact the registry rejected")
	}
	if ociPuts.Load() == 0 {
		t.Fatal("test registry never received the OCI artifact manifest")
	}
	if convertedPuts.Load() != 0 {
		t.Fatalf("push retried with %d digest-changing converted manifest(s)", convertedPuts.Load())
	}
}
