//go:build !podman

// End-to-end signing and verification against a real OCI registry
// speaking the real distribution protocol, rather than against mocks.
//
// The unit tests in sigstore_test.go cover the cryptography and the
// claim checks in isolation. What they cannot cover is everything
// between: that a signature is published at the tag cosign's convention
// says it should be, that pullToLayout finds it there, that appending a
// second signature preserves the first, and that a repository with no
// signature reads as "unsigned" rather than as an error. All of that is
// protocol behaviour, so it is tested over the protocol.
//
// !podman: pullToLayout/pushToRegistry exist in both backends with the
// same signatures, and sigstore.go is written against exactly that — but
// the podman backend's copy.Image path wants a policy context and
// registry configuration this harness does not set up. The docker build
// is the authoritative one for `unused` in .golangci.yml for the same
// reason.
package main

import (
	"bytes"
	"context"
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/containerd/errdefs"
	"github.com/google/go-containerregistry/pkg/registry"
	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/sigstore/sigstore/pkg/signature"
)

// startRegistry brings up an in-memory OCI registry and returns its
// host:port. containerd's resolver talks plain HTTP to a loopback host,
// which is what makes this work without TLS.
func startRegistry(t *testing.T) string {
	t.Helper()
	return startRegistryWithFault(t, nil)
}

// startRegistryWithFault is startRegistry with an optional interceptor.
// Returning true from fault means it has written the response itself and
// the registry must not see the request — used to simulate a registry
// that is reachable but failing, as distinct from one that simply has
// nothing at the requested tag.
func startRegistryWithFault(t *testing.T, fault func(http.ResponseWriter, *http.Request) bool) string {
	t.Helper()
	reg := registry.New(registry.Logger(nopLogger(t)))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if fault != nil && fault(w, r) {
			return
		}
		reg.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("parse registry URL: %v", err)
	}
	return u.Host
}

// nopLogger silences the registry's per-request logging, which is
// several hundred lines of noise for a passing test.
func nopLogger(t *testing.T) *log.Logger {
	t.Helper()
	return log.New(io.Discard, "", 0)
}

// writeKeyPair generates a P-256 key pair and writes both halves as PEM,
// returning their paths. Plain (unencrypted) PKCS#8 and PKIX, which is
// what `openssl genpkey` produces and what sigstore's loaders read.
func writeKeyPair(t *testing.T, dir, name string) (privPath, pubPath string) {
	t.Helper()
	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}
	privDER, err := x509.MarshalPKCS8PrivateKey(priv)
	if err != nil {
		t.Fatalf("marshal private key: %v", err)
	}
	pubDER, err := x509.MarshalPKIXPublicKey(&priv.PublicKey)
	if err != nil {
		t.Fatalf("marshal public key: %v", err)
	}
	privPath = filepath.Join(dir, name+".key")
	pubPath = filepath.Join(dir, name+".pub")
	writePEM(t, privPath, "PRIVATE KEY", privDER)
	writePEM(t, pubPath, "PUBLIC KEY", pubDER)
	return privPath, pubPath
}

func writePEM(t *testing.T, path, blockType string, der []byte) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create %s: %v", path, err)
	}
	defer f.Close()
	if err := pem.Encode(f, &pem.Block{Type: blockType, Bytes: der}); err != nil {
		t.Fatalf("encode %s: %v", path, err)
	}
}

// publishModel pushes a minimal but structurally real model artifact to
// ref and returns its manifest digest — the thing a signature covers.
func publishModel(t *testing.T, ref string) digest.Digest {
	t.Helper()
	dir := t.TempDir()
	if err := ensureLayout(dir); err != nil {
		t.Fatalf("init layout: %v", err)
	}
	layer, err := writeBlob(dir, "application/vnd.cncf.model.weight.v1.raw", []byte("pretend GGUF weights"))
	if err != nil {
		t.Fatalf("write layer: %v", err)
	}
	config, err := writeBlob(dir, "application/vnd.cncf.model.config.v1+json", []byte(`{}`))
	if err != nil {
		t.Fatalf("write config: %v", err)
	}
	m := ocispec.Manifest{
		MediaType: ocispec.MediaTypeImageManifest,
		Config:    config,
		Layers:    []ocispec.Descriptor{layer},
	}
	m.SchemaVersion = 2
	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal manifest: %v", err)
	}
	desc, err := writeBlob(dir, ocispec.MediaTypeImageManifest, raw)
	if err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	if err := writeManifestRef(dir, ref, desc); err != nil {
		t.Fatalf("tag manifest: %v", err)
	}
	if _, err := pushToRegistry(context.Background(), dir, ref); err != nil {
		t.Fatalf("push model: %v", err)
	}
	return desc.Digest
}

func TestSignAndVerifyRoundTripAgainstARealRegistry(t *testing.T) {
	ctx := context.Background()
	host := startRegistry(t)
	ref := host + "/org/model:v1"
	target := publishModel(t, ref)

	keys := t.TempDir()
	privPath, pubPath := writeKeyPair(t, keys, "signer")

	// Unsigned to begin with: a repository with no signature artifact is
	// a well-formed "no" (report.Verified false), not an error. Callers
	// need that distinction to tell "unsigned" from "registry down".
	report, err := verifySignatures(ctx, ref, target, []string{pubPath})
	if err != nil {
		t.Fatalf("verify before signing: %v", err)
	}
	if report.Verified || report.SignaturesFound != 0 {
		t.Fatalf("an unsigned model verified: %+v", report)
	}
	if report.Reason == "" {
		t.Error("an unverified report carried no reason")
	}

	if err := signManifest(ctx, ref, target, privPath, nil); err != nil {
		t.Fatalf("sign: %v", err)
	}

	report, err = verifySignatures(ctx, ref, target, []string{pubPath})
	if err != nil {
		t.Fatalf("verify after signing: %v", err)
	}
	if !report.Verified {
		t.Fatalf("a freshly signed model did not verify: %+v", report)
	}
	if report.SignaturesFound != 1 {
		t.Errorf("SignaturesFound = %d, want 1", report.SignaturesFound)
	}
	if len(report.Matches) != 1 || report.Matches[0].KeyPath != pubPath {
		t.Errorf("unexpected matches: %+v", report.Matches)
	}
	if report.Matches[0].Identity != host+"/org/model" {
		t.Errorf("claimed identity = %q, want the repository", report.Matches[0].Identity)
	}
}

func TestVerifyRejectsAStrangersKeyOverTheWire(t *testing.T) {
	ctx := context.Background()
	host := startRegistry(t)
	ref := host + "/org/model:v1"
	target := publishModel(t, ref)

	keys := t.TempDir()
	privPath, _ := writeKeyPair(t, keys, "signer")
	_, strangerPub := writeKeyPair(t, keys, "stranger")

	if err := signManifest(ctx, ref, target, privPath, nil); err != nil {
		t.Fatalf("sign: %v", err)
	}

	report, err := verifySignatures(ctx, ref, target, []string{strangerPub})
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	if report.Verified {
		t.Fatal("a signature by an untrusted key verified")
	}
	// The distinction that matters for a useful error message: something
	// *is* signed here, just not by anyone we trust.
	if report.SignaturesFound != 1 {
		t.Errorf("SignaturesFound = %d, want 1 (signed, but not by us)", report.SignaturesFound)
	}
}

func TestSigningTwiceAppendsRatherThanReplaces(t *testing.T) {
	// Key rotation depends on this: publish under the new key while the
	// old one is still trusted, then withdraw the old key from policy. If
	// the second signature replaced the first, every verifier still on
	// the old key would break the moment the new one was published.
	ctx := context.Background()
	host := startRegistry(t)
	ref := host + "/org/model:v1"
	target := publishModel(t, ref)

	keys := t.TempDir()
	privA, pubA := writeKeyPair(t, keys, "a")
	privB, pubB := writeKeyPair(t, keys, "b")

	if err := signManifest(ctx, ref, target, privA, nil); err != nil {
		t.Fatalf("sign with A: %v", err)
	}
	if err := signManifest(ctx, ref, target, privB, nil); err != nil {
		t.Fatalf("sign with B: %v", err)
	}

	for _, key := range []string{pubA, pubB} {
		report, err := verifySignatures(ctx, ref, target, []string{key})
		if err != nil {
			t.Fatalf("verify with %s: %v", key, err)
		}
		if !report.Verified {
			t.Errorf("%s no longer verifies after a second signature was added: %+v", key, report)
		}
		if report.SignaturesFound != 2 {
			t.Errorf("SignaturesFound = %d, want 2", report.SignaturesFound)
		}
	}
}

func TestSigningIsIdempotent(t *testing.T) {
	// Re-running a sign must not grow the artifact by one layer every
	// time — a CI job that signs on every build would otherwise
	// accumulate them without bound.
	ctx := context.Background()
	host := startRegistry(t)
	ref := host + "/org/model:v1"
	target := publishModel(t, ref)

	keys := t.TempDir()
	privPath, pubPath := writeKeyPair(t, keys, "signer")

	for i := 0; i < 3; i++ {
		if err := signManifest(ctx, ref, target, privPath, nil); err != nil {
			t.Fatalf("sign %d: %v", i, err)
		}
	}
	report, err := verifySignatures(ctx, ref, target, []string{pubPath})
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	if !report.Verified {
		t.Fatalf("did not verify: %+v", report)
	}
	if report.SignaturesFound != 1 {
		t.Errorf("SignaturesFound = %d after signing three times, want 1", report.SignaturesFound)
	}
}

func TestSignaturePublishedAtTheCosignTag(t *testing.T) {
	// The tag convention is the entire interop contract with cosign: get
	// it wrong and `cosign verify` looks in a place nothing was written
	// to. Assert the artifact is reachable at exactly that reference.
	ctx := context.Background()
	host := startRegistry(t)
	ref := host + "/org/model:v1"
	target := publishModel(t, ref)

	keys := t.TempDir()
	privPath, _ := writeKeyPair(t, keys, "signer")
	if err := signManifest(ctx, ref, target, privPath, nil); err != nil {
		t.Fatalf("sign: %v", err)
	}

	sigRef := signatureTagFor(ref, target)
	manifest, dir, err := fetchSignatureManifest(ctx, sigRef)
	if err != nil {
		t.Fatalf("no signature artifact at %s: %v", sigRef, err)
	}
	defer os.RemoveAll(dir)

	if len(manifest.Layers) != 1 {
		t.Fatalf("signature manifest has %d layers, want 1", len(manifest.Layers))
	}
	layer := manifest.Layers[0]
	if layer.MediaType != cosignSignatureMediaType {
		t.Errorf("layer media type = %q, want %q", layer.MediaType, cosignSignatureMediaType)
	}
	if layer.Annotations[cosignSignatureAnnotation] == "" {
		t.Errorf("layer is missing the %s annotation", cosignSignatureAnnotation)
	}

	payload, err := readBlob(dir, layer.Digest)
	if err != nil {
		t.Fatalf("read payload: %v", err)
	}
	var ssp simpleSigningPayload
	if err := json.Unmarshal(payload, &ssp); err != nil {
		t.Fatalf("parse payload: %v", err)
	}
	if ssp.Critical.Image.DockerManifestDigest != target.String() {
		t.Errorf("payload covers %q, want %q", ssp.Critical.Image.DockerManifestDigest, target)
	}
	if ssp.Critical.Type != cosignSignatureType {
		t.Errorf("payload type = %q, want %q", ssp.Critical.Type, cosignSignatureType)
	}
}

func TestFetchSignatureManifestReportsAbsenceDistinctly(t *testing.T) {
	// errNoSignature is what lets verifySignatures answer "unsigned"
	// instead of failing. A registry that is up and simply has nothing
	// there must produce it.
	ctx := context.Background()
	host := startRegistry(t)
	ref := host + "/org/model:v1"
	target := publishModel(t, ref)

	_, _, err := fetchSignatureManifest(ctx, signatureTagFor(ref, target))
	if !errors.Is(err, errNoSignature) {
		t.Fatalf("err = %v, want errNoSignature", err)
	}
}

func TestPullToLayoutHandlesRepeatedLayerDigests(t *testing.T) {
	// Regression: a manifest may legally name one blob twice, and
	// pullToLayout used to create a progress bar per layer entry while
	// singleflighting the shared fetch — leaving one bar never completed
	// and prog.Wait() blocked forever. A two-key signature artifact is
	// the case that surfaced it (identical payload blob, different
	// signature annotations), but nothing about the bug is specific to
	// signatures, so it is pinned here on a plain manifest.
	ctx := context.Background()
	host := startRegistry(t)
	ref := host + "/org/repeated:v1"

	src := t.TempDir()
	if err := ensureLayout(src); err != nil {
		t.Fatalf("init layout: %v", err)
	}
	blob, err := writeBlob(src, "application/vnd.oci.image.layer.v1.tar", []byte("shared blob"))
	if err != nil {
		t.Fatalf("write blob: %v", err)
	}
	config, err := writeBlob(src, "application/vnd.oci.image.config.v1+json", []byte(`{}`))
	if err != nil {
		t.Fatalf("write config: %v", err)
	}
	first := blob
	first.Annotations = map[string]string{"role": "first"}
	second := blob
	second.Annotations = map[string]string{"role": "second"}

	m := ocispec.Manifest{
		MediaType: ocispec.MediaTypeImageManifest,
		Config:    config,
		Layers:    []ocispec.Descriptor{first, second},
	}
	m.SchemaVersion = 2
	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal manifest: %v", err)
	}
	desc, err := writeBlob(src, ocispec.MediaTypeImageManifest, raw)
	if err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	if err := writeManifestRef(src, ref, desc); err != nil {
		t.Fatalf("tag manifest: %v", err)
	}
	if _, err := pushToRegistry(ctx, src, ref); err != nil {
		t.Fatalf("push: %v", err)
	}

	// The assertion is simply that this returns at all: before the fix
	// it blocked in prog.Wait() until the test binary's timeout.
	dst := t.TempDir()
	if err := pullToLayout(ctx, ref, dst); err != nil {
		t.Fatalf("pull: %v", err)
	}
	if got, err := readBlob(dst, blob.Digest); err != nil {
		t.Fatalf("shared blob missing after pull: %v", err)
	} else if string(got) != "shared blob" {
		t.Errorf("shared blob content = %q", got)
	}
}

func TestResolveManifestDigestMatchesWhatWasPushed(t *testing.T) {
	// PullGuard verifies against this digest before downloading
	// anything, then confirms the store landed on the same one. If the
	// two disagreed, every enforced pull would fail.
	ctx := context.Background()
	host := startRegistry(t)
	ref := host + "/org/model:v1"
	target := publishModel(t, ref)

	got, err := resolveManifestDigest(ctx, ref)
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if got != target {
		t.Errorf("resolveManifestDigest = %s, want %s", got, target)
	}
}

func TestIsNotFoundErrorRecognizesTheBackendSentinel(t *testing.T) {
	// containerd's typed answer, which is what a real missing manifest
	// produces. Kept out of sigstore_test.go because the podman build
	// has a different sentinel entirely.
	if !isNotFoundError(fmt.Errorf("resolve: %w", errdefs.ErrNotFound)) {
		t.Error("a wrapped errdefs.ErrNotFound was not recognized as absent")
	}
}

func TestSignRefusesToOverwriteASignatureItCouldNotRead(t *testing.T) {
	// The failure mode this guards: signManifest starts from an empty
	// layer set when the existing artifact is *absent*, so a transport
	// or authorization failure misread as absence would republish a
	// single-layer artifact and destroy every other signer's signature.
	// Here the fetch fails for a reason that is emphatically not
	// absence, and the existing signature must survive untouched.
	ctx := context.Background()

	// Any manifest read fails while the flag is set. Deliberately not
	// scoped to the ".sig" tag: containerd resolves a tag with HEAD and
	// then fetches by digest, so the second request carries no trace of
	// the tag it came from. Nothing else reads a manifest inside the
	// window this test opens.
	var failManifestReads atomic.Bool
	host := startRegistryWithFault(t, func(w http.ResponseWriter, r *http.Request) bool {
		if failManifestReads.Load() && strings.Contains(r.URL.Path, "/manifests/") &&
			(r.Method == http.MethodGet || r.Method == http.MethodHead) {
			w.WriteHeader(http.StatusInternalServerError)
			return true
		}
		return false
	})

	ref := host + "/org/model:v1"
	target := publishModel(t, ref)
	keys := t.TempDir()
	privA, pubA := writeKeyPair(t, keys, "a")
	privB, _ := writeKeyPair(t, keys, "b")

	if err := signManifest(ctx, ref, target, privA, nil); err != nil {
		t.Fatalf("sign with A: %v", err)
	}

	failManifestReads.Store(true)
	if err := signManifest(ctx, ref, target, privB, nil); err == nil {
		t.Fatal("signing succeeded despite being unable to read the existing artifact")
	}
	failManifestReads.Store(false)

	// A's signature is still there.
	report, err := verifySignatures(ctx, ref, target, []string{pubA})
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	if !report.Verified {
		t.Fatalf("the pre-existing signature was destroyed: %+v", report)
	}
}

func TestVerifyRejectsAnOversizedSignatureArtifact(t *testing.T) {
	// The manifest is inspected and bounded before any blob is fetched,
	// so a registry cannot make verification download an artifact of its
	// choosing. The manifest here lies about the payload's size; nothing
	// should try to fetch it.
	ctx := context.Background()
	host := startRegistry(t)
	ref := host + "/org/model:v1"
	target := publishModel(t, ref)

	dir := t.TempDir()
	if err := ensureLayout(dir); err != nil {
		t.Fatalf("init layout: %v", err)
	}
	payload, err := writeBlob(dir, cosignSignatureMediaType, []byte("{}"))
	if err != nil {
		t.Fatalf("write payload: %v", err)
	}
	payload.Size = maxSignatureBytes + 1
	payload.Annotations = map[string]string{cosignSignatureAnnotation: "AA=="}
	config, err := writeBlob(dir, emptyConfigMediaType, emptyConfig)
	if err != nil {
		t.Fatalf("write config: %v", err)
	}
	m := ocispec.Manifest{MediaType: ocispec.MediaTypeImageManifest, Config: config, Layers: []ocispec.Descriptor{payload}}
	m.SchemaVersion = 2
	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	desc, err := writeBlob(dir, ocispec.MediaTypeImageManifest, raw)
	if err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	sigRef := signatureTagFor(ref, target)
	if err := writeManifestRef(dir, sigRef, desc); err != nil {
		t.Fatalf("tag: %v", err)
	}
	if _, err := pushToRegistry(ctx, dir, sigRef); err != nil {
		t.Fatalf("push oversized signature: %v", err)
	}

	keys := t.TempDir()
	_, pub := writeKeyPair(t, keys, "signer")
	if _, err := verifySignatures(ctx, ref, target, []string{pub}); err == nil {
		t.Fatal("an oversized signature artifact was accepted for download")
	}
}

func TestConfirmationRequiresThisKeysOwnSignature(t *testing.T) {
	// The predicate signManifest confirms with. It used to compare layer
	// counts, which cannot distinguish "my signature landed" from
	// "someone else's did": two signers starting from an empty artifact
	// each push one layer, so the loser saw a count of 1 and reported
	// its own lost signature as a success.
	ctx := context.Background()
	host := startRegistry(t)
	ref := host + "/org/model:v1"
	target := publishModel(t, ref)

	keys := t.TempDir()
	privA, pubA := writeKeyPair(t, keys, "a")
	_, pubB := writeKeyPair(t, keys, "b")

	// Only A has signed.
	if err := signManifest(ctx, ref, target, privA, nil); err != nil {
		t.Fatalf("sign with A: %v", err)
	}

	a, err := loadTrustedKeys([]string{pubA})
	if err != nil {
		t.Fatalf("load A: %v", err)
	}
	b, err := loadTrustedKeys([]string{pubB})
	if err != nil {
		t.Fatalf("load B: %v", err)
	}

	landedForA, err := verifyAgainst(ctx, ref, target, a)
	if err != nil {
		t.Fatalf("verifyAgainst A: %v", err)
	}
	if !landedForA.Verified {
		t.Error("A's own signature did not confirm")
	}

	landedForB, err := verifyAgainst(ctx, ref, target, b)
	if err != nil {
		t.Fatalf("verifyAgainst B: %v", err)
	}
	if landedForB.Verified {
		t.Error("A's signature counted as confirmation for B")
	}
	// The count is equal either way — which is exactly why counting was
	// the wrong question.
	if landedForA.SignaturesFound != landedForB.SignaturesFound {
		t.Fatal("test premise broken: the layer count should be identical")
	}
}

func TestSigningRecoversFromAnOverwrittenTag(t *testing.T) {
	// A concurrent signer replaced the artifact with one holding only
	// its own signature. Signing with A again must notice A is missing
	// and re-append, leaving both.
	ctx := context.Background()
	host := startRegistry(t)
	ref := host + "/org/model:v1"
	target := publishModel(t, ref)

	keys := t.TempDir()
	privA, pubA := writeKeyPair(t, keys, "a")
	privB, pubB := writeKeyPair(t, keys, "b")

	if err := signManifest(ctx, ref, target, privA, nil); err != nil {
		t.Fatalf("sign with A: %v", err)
	}
	// B, having read the artifact before A's push, publishes one that
	// knows nothing about A.
	if err := overwriteSignatureWithOnly(ctx, t, ref, target, privB); err != nil {
		t.Fatalf("simulate B overwriting: %v", err)
	}
	for _, key := range []string{pubA, pubB} {
		report, err := verifySignatures(ctx, ref, target, []string{key})
		if err != nil {
			t.Fatalf("verify %s: %v", key, err)
		}
		if key == pubA && report.Verified {
			t.Fatal("test premise broken: A should have been overwritten")
		}
	}

	// A signs again and must end up back in the artifact.
	if err := signManifest(ctx, ref, target, privA, nil); err != nil {
		t.Fatalf("re-sign with A: %v", err)
	}
	for _, key := range []string{pubA, pubB} {
		report, err := verifySignatures(ctx, ref, target, []string{key})
		if err != nil {
			t.Fatalf("verify %s: %v", key, err)
		}
		if !report.Verified {
			t.Errorf("%s does not verify after recovery: %+v", key, report)
		}
	}
}

// overwriteSignatureWithOnly republishes the signature tag holding only
// keyPath's signature, discarding whatever was there — what a signer
// that read the artifact before someone else's push would produce.
func overwriteSignatureWithOnly(
	ctx context.Context,
	t *testing.T,
	ref string,
	target digest.Digest,
	keyPath string,
) error {
	t.Helper()
	signer, err := signature.LoadSignerFromPEMFile(keyPath, crypto.SHA256, staticPassFunc(nil))
	if err != nil {
		return err
	}
	var doc simpleSigningPayload
	doc.Critical.Identity.DockerReference = repositoryOf(ref)
	doc.Critical.Image.DockerManifestDigest = target.String()
	doc.Critical.Type = cosignSignatureType
	payload, err := json.Marshal(doc)
	if err != nil {
		return err
	}
	raw, err := signer.SignMessage(bytes.NewReader(payload))
	if err != nil {
		return err
	}

	dir := t.TempDir()
	if err := ensureLayout(dir); err != nil {
		return err
	}
	payloadDesc, err := writeBlob(dir, cosignSignatureMediaType, payload)
	if err != nil {
		return err
	}
	payloadDesc.Annotations = map[string]string{
		cosignSignatureAnnotation: base64.StdEncoding.EncodeToString(raw),
	}
	config, err := writeBlob(dir, emptyConfigMediaType, emptyConfig)
	if err != nil {
		return err
	}
	m := ocispec.Manifest{
		MediaType: ocispec.MediaTypeImageManifest,
		Config:    config,
		Layers:    []ocispec.Descriptor{payloadDesc},
	}
	m.SchemaVersion = 2
	manifestJSON, err := json.Marshal(m)
	if err != nil {
		return err
	}
	desc, err := writeBlob(dir, ocispec.MediaTypeImageManifest, manifestJSON)
	if err != nil {
		return err
	}
	sigRef := signatureTagFor(ref, target)
	if err := writeManifestRef(dir, sigRef, desc); err != nil {
		return err
	}
	_, err = pushToRegistry(ctx, dir, sigRef)
	return err
}

func TestSignDetectsAnEqualCountOverwriteBetweenPushAndConfirm(t *testing.T) {
	// The exact race the confirm loop exists for, and the one a
	// layer-count check cannot see. A and B both start from an empty
	// artifact, so both push exactly one layer: B's push lands after A's
	// and discards it. A then confirms and sees a count of 1 — its own
	// number — while the signature there is B's.
	//
	// Injected deterministically: B's overwrite is performed the moment
	// A's confirm reads the tag.
	ctx := context.Background()

	var armed, busy atomic.Bool
	var overwrite func()
	host := startRegistryWithFault(t, func(_ http.ResponseWriter, r *http.Request) bool {
		if busy.Load() || !strings.Contains(r.URL.Path, "/manifests/sha256-") ||
			!strings.HasSuffix(r.URL.Path, ".sig") {
			return false
		}
		switch r.Method {
		case http.MethodPut:
			armed.Store(true) // A has just published
		case http.MethodGet, http.MethodHead:
			if armed.CompareAndSwap(true, false) {
				busy.Store(true)
				overwrite() // B lands, discarding A
				busy.Store(false)
			}
		}
		return false // never handled here; always let the registry serve
	})

	ref := host + "/org/model:v1"
	target := publishModel(t, ref)
	keys := t.TempDir()
	privA, pubA := writeKeyPair(t, keys, "a")
	privB, pubB := writeKeyPair(t, keys, "b")
	// One-shot: B races once, then stops. A's retry must recover.
	var once sync.Once
	overwrite = func() {
		once.Do(func() {
			if err := overwriteSignatureWithOnly(ctx, t, ref, target, privB); err != nil {
				t.Errorf("simulate B overwriting: %v", err)
			}
		})
	}

	if err := signManifest(ctx, ref, target, privA, nil); err != nil {
		t.Fatalf("sign with A: %v", err)
	}

	// A reported success, so A's signature must actually be there. The
	// count-based check returned success here with A's signature lost.
	for _, key := range []string{pubA, pubB} {
		report, err := verifySignatures(ctx, ref, target, []string{key})
		if err != nil {
			t.Fatalf("verify %s: %v", key, err)
		}
		if !report.Verified {
			t.Errorf("%s does not verify after signManifest reported success: %+v", key, report)
		}
	}
}

func TestVerifyReportsAFailingRegistryAsAnErrorNotAsUnsigned(t *testing.T) {
	// The negative twin of TestFetchSignatureManifestReportsAbsenceDistinctly,
	// and the distinction the whole policy layer rests on: src/verify.rs
	// treats an *error* from here as indeterminate (refuse to serve, keep
	// any local copy) and a report with Verified false as a verdict
	// against the model (refuse, and drop it). A reachable-but-failing
	// registry must therefore never come back as a well-formed "unsigned".
	ctx := context.Background()

	var failManifestReads atomic.Bool
	host := startRegistryWithFault(t, func(w http.ResponseWriter, r *http.Request) bool {
		if failManifestReads.Load() && strings.Contains(r.URL.Path, "/manifests/") &&
			(r.Method == http.MethodGet || r.Method == http.MethodHead) {
			w.WriteHeader(http.StatusInternalServerError)
			return true
		}
		return false
	})

	ref := host + "/org/model:v1"
	target := publishModel(t, ref)
	keys := t.TempDir()
	priv, pub := writeKeyPair(t, keys, "signer")
	if err := signManifest(ctx, ref, target, priv, nil); err != nil {
		t.Fatalf("sign: %v", err)
	}

	failManifestReads.Store(true)
	report, err := verifySignatures(ctx, ref, target, []string{pub})
	if err == nil {
		t.Fatalf("a failing registry produced a report instead of an error: %+v", report)
	}
	// And it must not be mistaken for absence, which is what would make
	// the policy layer treat it as a verdict.
	if isNotFoundError(err) {
		t.Errorf("a 500 was classified as absence: %v", err)
	}
}

func TestWriteBlobStreamRejectsMoreBytesThanDeclared(t *testing.T) {
	// The size caps on a signature artifact bound its *declared* sizes.
	// A registry that declares 300 bytes and streams gigabytes would
	// defeat them if the write buffered everything before checking, so
	// the copy is bounded by the declared size.
	dir := t.TempDir()
	if err := ensureLayout(dir); err != nil {
		t.Fatalf("init layout: %v", err)
	}
	honest := []byte("the declared content")
	dgst := digest.FromBytes(honest)

	// Far more than declared, as a hostile registry would send.
	liar := bytes.NewReader(append(honest, bytes.Repeat([]byte("x"), 1<<20)...))
	_, err := writeBlobStream(dir, "application/octet-stream", liar, int64(len(honest)), dgst, 0)
	if err == nil {
		t.Fatal("a blob longer than its declared size was accepted")
	}
	if !strings.Contains(err.Error(), "size mismatch") {
		t.Errorf("unexpected error: %v", err)
	}
	// The assertion that pins the *bound* rather than just the
	// rejection: the copy must have stopped a byte past the declared
	// size, leaving the rest of the stream unread. Unbounded, this
	// reader would have been drained to EOF — on a real registry, onto
	// the disk — before the size check noticed.
	if liar.Len() == 0 {
		t.Error("the whole oversized stream was consumed before being rejected")
	}
	// Nothing partial left behind.
	if _, err := readBlob(dir, dgst); err == nil {
		t.Error("the rejected blob was still written to the layout")
	}

	// The honest case still works.
	if _, err := writeBlobStream(dir, "application/octet-stream",
		bytes.NewReader(honest), int64(len(honest)), dgst, 0); err != nil {
		t.Fatalf("an honest blob was rejected: %v", err)
	}
	got, err := readBlob(dir, dgst)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	if string(got) != string(honest) {
		t.Errorf("content = %q", got)
	}
}
