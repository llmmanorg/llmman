package main

import (
	"context"
	"crypto"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/asn1"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"math/big"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cyberphone/json-canonicalization/go/src/webpki.org/jsoncanonicalizer"
	digest "github.com/opencontainers/go-digest"
	"github.com/sigstore/fulcio/pkg/certificate"
	"github.com/sigstore/sigstore/pkg/cryptoutils"
	"github.com/sigstore/sigstore/pkg/signature"
)

// targetA/targetB are two distinct manifest digests, used wherever a
// test needs "the thing being verified" and "something else".
const (
	targetA = "sha256:1111111111111111111111111111111111111111111111111111111111111111"
	targetB = "sha256:2222222222222222222222222222222222222222222222222222222222222222"
)

func TestRepositoryOf(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"docker.io/org/model:v1", "docker.io/org/model"},
		{"docker.io/org/model", "docker.io/org/model"},
		{"docker.io/org/model@" + targetA, "docker.io/org/model"},
		{"docker.io/org/model:v1@" + targetA, "docker.io/org/model"},
		// A registry port is not a tag: the colon comes before the last
		// slash, which is the same rule normalizeTag applies.
		{"registry.example.com:5000/org/model", "registry.example.com:5000/org/model"},
		{"registry.example.com:5000/org/model:v1", "registry.example.com:5000/org/model"},
		{"docker.io/ai/gemma4:latest", "docker.io/ai/gemma4"},
	} {
		if got := repositoryOf(tc.in); got != tc.want {
			t.Errorf("repositoryOf(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestCanonicalRepositoryFoldsDockerHubSpellings(t *testing.T) {
	// Every spelling of the same Docker Hub repository has to land on
	// one string, or a signature cosign wrote (which canonicalizes to
	// index.docker.io/...) would never match a reference llmman resolved
	// (which uses docker.io/...).
	same := []string{
		"docker.io/ai/gemma4",
		"index.docker.io/ai/gemma4",
		"registry-1.docker.io/ai/gemma4",
		"ai/gemma4",
	}
	want := canonicalRepository(same[0])
	for _, s := range same[1:] {
		if got := canonicalRepository(s); got != want {
			t.Errorf("canonicalRepository(%q) = %q, want %q", s, got, want)
		}
	}

	// A different host is left completely alone.
	if got := canonicalRepository("quay.io/org/model"); got != "quay.io/org/model" {
		t.Errorf("canonicalRepository rewrote a non-Docker-Hub reference: %q", got)
	}

	// Regression: stripping the host made this collapse onto the
	// genuinely different "quay.io/org/model", which would have let a
	// signature be transplanted between the two.
	if canonicalRepository("docker.io/quay.io/org/model") == canonicalRepository("quay.io/org/model") {
		t.Error("a docker.io-nested path collided with a real quay.io repository")
	}

	// Official images, both spellings, fold together.
	if canonicalRepository("ubuntu") != canonicalRepository("docker.io/library/ubuntu") {
		t.Error("bare official name and its library/ spelling did not fold together")
	}

	// A bare host, with no repository path at all, still folds — the
	// same alias check must not require a trailing "/" to fire.
	if canonicalRepository("index.docker.io") != canonicalRepository("docker.io") {
		t.Error("a bare index.docker.io host did not fold onto docker.io")
	}
	if canonicalRepository("ubuntu") != canonicalRepository("docker.io/ubuntu") {
		t.Error("docker.io/ubuntu did not fold onto the bare official name")
	}
}

func TestSignatureTagFor(t *testing.T) {
	got := signatureTagFor("docker.io/org/model:v1", digest.Digest(targetA))
	want := "docker.io/org/model:sha256-" + strings.TrimPrefix(targetA, "sha256:") + ".sig"
	if got != want {
		t.Errorf("signatureTagFor = %q, want %q", got, want)
	}

	// The tag on the input is irrelevant: a signature is addressed by
	// the digest it covers, so every tag of one digest resolves to the
	// same signature artifact.
	other := signatureTagFor("docker.io/org/model:someothertag", digest.Digest(targetA))
	if other != got {
		t.Errorf("signature tag depended on the input tag: %q vs %q", other, got)
	}
}

func TestIdentityMatchesRepository(t *testing.T) {
	for _, tc := range []struct {
		name    string
		claimed string
		ref     string
		claims  string
		want    bool
	}{
		{"same repo", "docker.io/org/model", "docker.io/org/model:v1", "", true},
		{"claimed carries a tag", "docker.io/org/model:v9", "docker.io/org/model:v1", "", true},
		{"claimed carries a digest", "docker.io/org/model@" + targetB, "docker.io/org/model:v1", "", true},
		{"docker hub spelling differs", "index.docker.io/ai/gemma4", "docker.io/ai/gemma4:latest", "", true},

		// The copy attack this check exists to stop: a legitimately
		// signed model, moved wholesale into someone else's repository.
		{"different repo", "docker.io/org/model", "docker.io/attacker/model:v1", "", false},
		{"different host", "docker.io/org/model", "quay.io/org/model:v1", "", false},
		{"different namespace", "docker.io/org/model", "docker.io/other/model:v1", "", false},
		// A payload that asserts nothing about where it belongs is not
		// a lenient old-cosign variant, it is unusable.
		{"empty claim", "", "docker.io/org/model:v1", "", false},
		{"blank claim", "   ", "docker.io/org/model:v1", "", false},

		// With a claims glob the pulled reference no longer matters;
		// the claimed repository, canonicalised, must match the glob.
		{"claims accepts a relocated copy", "docker.io/myorg/name", "mirror.corp/myorg/name:v1", "docker.io/myorg/**", true},
		{"claims sees the canonical claimed spelling", "index.docker.io/myorg/name", "mirror.corp/myorg/name:v1", "docker.io/myorg/**", true},
		{"claims strips the claimed tag", "docker.io/myorg/name:v9", "mirror.corp/myorg/name:v1", "docker.io/myorg/**", true},
		{"claims rejects another namespace", "docker.io/myorg/name", "mirror.corp/myorg/name:v1", "docker.io/other/**", false},
		{"claims is not canonicalised", "docker.io/myorg/name", "mirror.corp/myorg/name:v1", "myorg/**", false},
		{"claims single star stays in one segment", "docker.io/myorg/sub/name", "mirror.corp/myorg/name:v1", "docker.io/myorg/*", false},
		{"claims ignores an equal pulled repo", "docker.io/org/model", "docker.io/org/model:v1", "docker.io/other/**", false},
		{"claims still rejects an empty claim", "", "mirror.corp/myorg/name:v1", "**", false},

		// The claimed repository is sized by the registry; the glob is
		// quadratic in it per single star, so host and path are each
		// capped before it reaches the matcher. The path cap is the one
		// distribution/reference applies (255 bytes after the host, the
		// host excluded), so a name both reference parsers accept is
		// never refused here.
		{"claims accepts a 255-byte path", "docker.io/" + strings.Repeat("a", 255-len("myorg/")) + "/myorg", "mirror.corp/myorg/name:v1", "docker.io/**/myorg", true},
		{"claims accepts a 255-byte path behind a host", "docker.io/myorg/" + strings.Repeat("a", 255-len("myorg/")), "mirror.corp/myorg/name:v1", "docker.io/myorg/**", true},
		{"claims refuses a 256-byte path", "docker.io/myorg/" + strings.Repeat("a", 256-len("myorg/")), "mirror.corp/myorg/name:v1", "docker.io/myorg/**", false},
		{"claims accepts a 259-byte host", strings.Repeat("h", 253) + ":65535/myorg/name", "mirror.corp/myorg/name:v1", "**/myorg/name", true},
		{"claims refuses a 260-byte host", strings.Repeat("h", 254) + ":65535/myorg/name", "mirror.corp/myorg/name:v1", "**/myorg/name", false},
		{"equality accepts a 256-byte path", "docker.io/myorg/" + strings.Repeat("a", 256-len("myorg/")), "docker.io/myorg/" + strings.Repeat("a", 256-len("myorg/")) + ":v1", "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := identityMatchesRepository(tc.claimed, tc.ref, tc.claims); got != tc.want {
				t.Errorf("identityMatchesRepository(%q, %q, %q) = %v, want %v", tc.claimed, tc.ref, tc.claims, got, tc.want)
			}
		})
	}
}

// signedPayload builds a simple-signing payload for ref/target and signs
// it, returning the exact bytes and the base64 signature over them —
// i.e. what one signature layer of a cosign artifact holds.
func signedPayload(t *testing.T, signer signature.Signer, ref, target string) ([]byte, string) {
	t.Helper()
	var doc simpleSigningPayload
	doc.Critical.Identity.DockerReference = repositoryOf(ref)
	doc.Critical.Image.DockerManifestDigest = target
	doc.Critical.Type = cosignSignatureType
	payload, err := json.Marshal(doc)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	raw, err := signer.SignMessage(strings.NewReader(string(payload)))
	if err != nil {
		t.Fatalf("sign payload: %v", err)
	}
	return payload, base64.StdEncoding.EncodeToString(raw)
}

func ecdsaSignerVerifier(t *testing.T) (signature.Signer, trustedKey) {
	t.Helper()
	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}
	sv, err := signature.LoadECDSASignerVerifier(priv, crypto.SHA256)
	if err != nil {
		t.Fatalf("load signerverifier: %v", err)
	}
	return sv, trustedKey{path: "test-ecdsa.pub", verifier: sv}
}

// sigAnnotations is the annotation map of a key-signed layer: only the
// signature, as cosign writes for `cosign sign --key`.
func sigAnnotations(b64sig string) map[string]string {
	return map[string]string{cosignSignatureAnnotation: b64sig}
}

func keysOnly(keys []trustedKey) verifyPolicy {
	return verifyPolicy{keys: keys}
}

func TestVerifySignatureLayerIdentityRuleNeedsItsTrustedRoot(t *testing.T) {
	signer, key := ecdsaSignerVerifier(t)
	const ref = "docker.io/org/model:v1"
	payload, b64 := signedPayload(t, signer, ref, targetA)
	identity := &identityPolicy{Issuer: "https://issuer", Subject: "me@example.com", TrustedRoot: filepath.Join(t.TempDir(), "missing.json")}

	_, err := verifySignatureLayer(payload, sigAnnotations(b64), ref, digest.Digest(targetA), verifyPolicy{identity: identity}, "")
	if err == nil || !strings.Contains(err.Error(), "read trusted root") {
		t.Fatalf("identity-only policy with a missing trusted root: err = %v, want read trusted root", err)
	}

	// A key alongside the identity still accepts and labels the signer.
	match, err := verifySignatureLayer(payload, sigAnnotations(b64), ref, digest.Digest(targetA), verifyPolicy{keys: []trustedKey{key}, identity: identity}, "")
	if err != nil {
		t.Fatalf("key beside identity: %v", err)
	}
	if match.Signer != key.path {
		t.Errorf("match.Signer = %q, want %q", match.Signer, key.path)
	}
}

func TestVerifySignatureLayerAcceptsAGoodSignature(t *testing.T) {
	signer, key := ecdsaSignerVerifier(t)
	const ref = "docker.io/org/model:v1"
	payload, b64 := signedPayload(t, signer, ref, targetA)

	match, err := verifySignatureLayer(payload, sigAnnotations(b64), ref, digest.Digest(targetA), keysOnly([]trustedKey{key}), "")
	if err != nil {
		t.Fatalf("verifySignatureLayer: %v", err)
	}
	if match.KeyPath != key.path {
		t.Errorf("match.KeyPath = %q, want %q", match.KeyPath, key.path)
	}
	if match.Identity != "docker.io/org/model" {
		t.Errorf("match.Identity = %q, want the repository", match.Identity)
	}
}

func TestVerifySignatureLayerAcceptsEd25519(t *testing.T) {
	// ed25519 takes a different path through LoadVerifier (the hash
	// argument is ignored and the key type picks the algorithm), so it
	// is worth one test of its own rather than assuming ECDSA covers it.
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}
	signer, err := signature.LoadED25519Signer(priv)
	if err != nil {
		t.Fatalf("load signer: %v", err)
	}
	verifier, err := signature.LoadVerifier(pub, crypto.SHA256)
	if err != nil {
		t.Fatalf("load verifier: %v", err)
	}

	const ref = "docker.io/org/model:v1"
	payload, b64 := signedPayload(t, signer, ref, targetA)
	if _, err := verifySignatureLayer(payload, sigAnnotations(b64), ref, digest.Digest(targetA), keysOnly([]trustedKey{{path: "ed25519.pub", verifier: verifier}}), ""); err != nil {
		t.Fatalf("verifySignatureLayer: %v", err)
	}
}

func TestVerifySignatureLayerRejectsUntrustedKey(t *testing.T) {
	signer, _ := ecdsaSignerVerifier(t)
	_, stranger := ecdsaSignerVerifier(t)
	const ref = "docker.io/org/model:v1"
	payload, b64 := signedPayload(t, signer, ref, targetA)

	_, err := verifySignatureLayer(payload, sigAnnotations(b64), ref, digest.Digest(targetA), keysOnly([]trustedKey{stranger}), "")
	if err == nil {
		t.Fatal("a signature by an untrusted key was accepted")
	}
	if !strings.Contains(err.Error(), "no trusted key") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestVerifySignatureLayerRejectsWrongDigest(t *testing.T) {
	// A validly signed payload, presented as if it covered a different
	// manifest. Without this check a signature over any one model would
	// vouch for every model in the repository.
	signer, key := ecdsaSignerVerifier(t)
	const ref = "docker.io/org/model:v1"
	payload, b64 := signedPayload(t, signer, ref, targetA)

	_, err := verifySignatureLayer(payload, sigAnnotations(b64), ref, digest.Digest(targetB), keysOnly([]trustedKey{key}), "")
	if err == nil {
		t.Fatal("a signature over a different manifest was accepted")
	}
	if !strings.Contains(err.Error(), "payload signs") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestVerifySignatureLayerAcceptsRelocatedRepositoryUnderClaims(t *testing.T) {
	// A model signed as docker.io/myorg/name and copied, signature and
	// all, to mirror.corp/myorg/name. The trust rule's claims glob names
	// the original repository, so the signature verifies and reports the
	// identity it was signed under.
	signer, key := ecdsaSignerVerifier(t)
	payload, b64 := signedPayload(t, signer, "docker.io/myorg/name:v1", targetA)

	match, err := verifySignatureLayer(payload, sigAnnotations(b64), "mirror.corp/myorg/name:v1", digest.Digest(targetA), keysOnly([]trustedKey{key}), "docker.io/myorg/**")
	if err != nil {
		t.Fatalf("relocated copy under matching claims: %v", err)
	}
	if match.Identity != "docker.io/myorg/name" {
		t.Errorf("identity = %q, want docker.io/myorg/name", match.Identity)
	}
}

func TestVerifySignatureLayerRefusesRelocatedRepositoryOutsideClaims(t *testing.T) {
	signer, key := ecdsaSignerVerifier(t)
	payload, b64 := signedPayload(t, signer, "docker.io/myorg/name:v1", targetA)

	for claims, want := range map[string]string{
		// Equality names the pulled repository the identity was held to.
		"": `which is not mirror.corp/myorg/name`,
		// Under claims the pulled repository was not consulted, so the
		// refusal names the glob that did the refusing instead.
		"docker.io/other/**": `which does not match claims "docker.io/other/**"`,
	} {
		_, err := verifySignatureLayer(payload, sigAnnotations(b64), "mirror.corp/myorg/name:v1", digest.Digest(targetA), keysOnly([]trustedKey{key}), claims)
		if err == nil {
			t.Fatalf("claims %q: a relocated copy was accepted", claims)
		}
		if !strings.Contains(err.Error(), "claims identity") || !strings.HasSuffix(err.Error(), want) {
			t.Errorf("claims %q: unexpected error: %v", claims, err)
		}
	}
}

func TestVerifySignatureLayerRefusesOversizedIdentityPromptly(t *testing.T) {
	// An unsigned payload can carry a docker-reference that fills the
	// 64 KiB layer cap; against a claims glob with two single stars
	// globMatch is quadratic in it and would take minutes unless the
	// identity is capped first. The verifier is never even consulted.
	//
	// The identity must defeat the glob at its very end: an `ab...ab`
	// text matches `*a*b` on the first branch tried and returns at once,
	// so only a trailing byte neither star can hand to `b` forces every
	// split to be visited. Uncapped, that run exceeds 20 s.
	_, key := ecdsaSignerVerifier(t)
	const claims = "docker.io/myorg/*a*b"
	var doc simpleSigningPayload
	doc.Critical.Image.DockerManifestDigest = targetA
	doc.Critical.Type = cosignSignatureType
	// Size the identity so the whole payload sits just under the layer
	// cap: the largest text a registry can get past readBlob.
	empty, err := json.Marshal(doc)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	prefix := "docker.io/myorg/"
	room := (maxSignatureBytes - 1 - len(empty) - len(prefix) - len("c")) / len("ab")
	doc.Critical.Identity.DockerReference = prefix + strings.Repeat("ab", room) + "c"
	payload, err := json.Marshal(doc)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	if len(payload) >= maxSignatureBytes || len(payload) < maxSignatureBytes-4 {
		t.Fatalf("payload is %d bytes, want just under %d", len(payload), maxSignatureBytes)
	}

	start := time.Now()
	_, err = verifySignatureLayer(payload, sigAnnotations(base64.StdEncoding.EncodeToString([]byte("not a signature"))), "mirror.corp/myorg/name:v1", digest.Digest(targetA), keysOnly([]trustedKey{key}), claims)
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("an oversized identity was accepted")
	}
	if !strings.Contains(err.Error(), "claims identity") {
		t.Errorf("unexpected error: %v", err)
	}
	if elapsed > time.Second {
		t.Errorf("refusal took %v, want well under a second", elapsed)
	}
}

func TestVerifySignatureLayerRejectsTransplantedSignature(t *testing.T) {
	// The copy attack: a real, correctly signed model and its signature,
	// lifted wholesale into an attacker-controlled repository. The
	// cryptography still checks out — only the claimed identity gives it
	// away.
	signer, key := ecdsaSignerVerifier(t)
	payload, b64 := signedPayload(t, signer, "docker.io/org/model:v1", targetA)

	_, err := verifySignatureLayer(payload, sigAnnotations(b64), "docker.io/attacker/model:v1", digest.Digest(targetA), keysOnly([]trustedKey{key}), "")
	if err == nil {
		t.Fatal("a signature copied into another repository was accepted")
	}
	if !strings.Contains(err.Error(), "claims identity") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestVerifySignatureLayerRejectsTamperedPayload(t *testing.T) {
	signer, key := ecdsaSignerVerifier(t)
	const ref = "docker.io/org/model:v1"
	payload, b64 := signedPayload(t, signer, ref, targetA)

	// Rewrite the digest inside the payload to the one being verified,
	// leaving the signature (over the original bytes) in place. The
	// claim checks now all pass; only the signature can catch this.
	tampered := []byte(strings.Replace(string(payload), targetA, targetB, 1))
	if string(tampered) == string(payload) {
		t.Fatal("test payload did not contain the digest to tamper with")
	}
	_, err := verifySignatureLayer(tampered, sigAnnotations(b64), ref, digest.Digest(targetB), keysOnly([]trustedKey{key}), "")
	if err == nil {
		t.Fatal("a tampered payload was accepted")
	}
	if !strings.Contains(err.Error(), "no trusted key") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestVerifySignatureLayerRejectsWrongPayloadType(t *testing.T) {
	// A document signed by a trusted key for some entirely different
	// purpose must not be replayable as a model signature.
	signer, key := ecdsaSignerVerifier(t)
	const ref = "docker.io/org/model:v1"

	var doc simpleSigningPayload
	doc.Critical.Identity.DockerReference = repositoryOf(ref)
	doc.Critical.Image.DockerManifestDigest = targetA
	doc.Critical.Type = "some other signature scheme"
	payload, err := json.Marshal(doc)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	raw, err := signer.SignMessage(strings.NewReader(string(payload)))
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	b64 := base64.StdEncoding.EncodeToString(raw)

	if _, err := verifySignatureLayer(payload, sigAnnotations(b64), ref, digest.Digest(targetA), keysOnly([]trustedKey{key}), ""); err == nil {
		t.Fatal("a payload of an unrelated signature type was accepted")
	}
}

func TestVerifySignatureLayerRejectsMalformedInput(t *testing.T) {
	_, key := ecdsaSignerVerifier(t)
	keys := []trustedKey{key}

	if _, err := verifySignatureLayer([]byte("{}"), sigAnnotations(""), "docker.io/org/model", digest.Digest(targetA), keysOnly(keys), ""); err == nil {
		t.Error("a layer with no signature annotation was accepted")
	}
	if _, err := verifySignatureLayer([]byte("{}"), sigAnnotations("!!!not base64!!!"), "docker.io/org/model", digest.Digest(targetA), keysOnly(keys), ""); err == nil {
		t.Error("a non-base64 signature was accepted")
	}
	if _, err := verifySignatureLayer([]byte("not json"), sigAnnotations(base64.StdEncoding.EncodeToString([]byte("sig"))), "docker.io/org/model", digest.Digest(targetA), keysOnly(keys), ""); err == nil {
		t.Error("a non-JSON payload was accepted")
	}
}

func TestSimpleSigningPayloadMatchesCosignWireFormat(t *testing.T) {
	// The signature covers these exact bytes, so the shape is part of
	// the interop contract with cosign, not an implementation detail.
	var doc simpleSigningPayload
	doc.Critical.Identity.DockerReference = "docker.io/org/model"
	doc.Critical.Image.DockerManifestDigest = targetA
	doc.Critical.Type = cosignSignatureType

	got, err := json.Marshal(doc)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	want := `{"critical":{"identity":{"docker-reference":"docker.io/org/model"},"image":{"docker-manifest-digest":"` +
		targetA + `"},"type":"cosign container image signature"},"optional":null}`
	if string(got) != want {
		t.Errorf("payload wire format drifted:\n got %s\nwant %s", got, want)
	}
}

func TestIsNotFoundError(t *testing.T) {
	// The typed sentinel each backend produces is covered separately —
	// see TestIsNotFoundErrorRecognizesTheBackendSentinel.
	for _, msg := range []string{
		"fetch manifest: MANIFEST_UNKNOWN: manifest unknown",
		"reading manifest: received unexpected HTTP status 404 Not Found",
		"pinging registry: 404 not found",
	} {
		if !isNotFoundError(errString(msg)) {
			t.Errorf("isNotFoundError(%q) = false, want true", msg)
		}
	}

	// Everything below must be classified as "could not find out", not
	// "absent". signManifest starts from an empty layer set on absent,
	// so a false positive here silently destroys other signers'
	// signatures.
	for _, msg := range []string{
		"dial tcp: connection refused",
		"unauthorized: authentication required",
		// A digest's hex contains "404" for roughly one manifest in
		// seventy, and these messages embed the reference.
		"resolve docker.io/org/model:sha256-404f8a2c9b1d.sig: unauthorized: authentication required",
		// The classic Docker Desktop failure. Bare "not found" matching
		// would call this an absent signature.
		"docker-credential-desktop not found in $PATH",
		"error getting credentials: helper not found",
	} {
		if isNotFoundError(errString(msg)) {
			t.Errorf("isNotFoundError(%q) = true, want false", msg)
		}
	}
}

type errString string

func (e errString) Error() string { return string(e) }

func TestTransferResultJSON(t *testing.T) {
	if got := transferResultJSON(true, digest.Digest(targetA)); got != `{"changed":true,"digest":"`+targetA+`"}` {
		t.Errorf("transferResultJSON = %s", got)
	}
	// The digest is reported even for an unchanged transfer: --sign-key
	// still has to sign a destination that already had the content.
	if got := transferResultJSON(false, digest.Digest(targetA)); got != `{"changed":false,"digest":"`+targetA+`"}` {
		t.Errorf("transferResultJSON = %s", got)
	}
	if got := transferResultJSON(true, ""); got != `{"changed":true}` {
		t.Errorf("transferResultJSON = %s", got)
	}
}

// sigstoreFixture is an in-test Sigstore, modelled on sigstore-go's
// pkg/testing/ca: one ECDSA P-256 root that issues signing certificates
// directly, or through one intermediate after withIntermediate (the
// shape public Fulcio has), and one Rekor log key. It is what a
// TrustedRoot file and a keyless signature layer are both derived from,
// so the two agree by construction and each refusal case breaks exactly
// one thing.
type sigstoreFixture struct {
	root    *x509.Certificate
	rootKey *ecdsa.PrivateKey
	// intermediate, when set, is the CA that issues leaves in the
	// root's place; it is listed before the root in the trusted root's
	// certChain and carried in the chain annotation of every layer.
	intermediate    *x509.Certificate
	intermediateKey *ecdsa.PrivateKey
	rekorKey        *ecdsa.PrivateKey
	// logID is hex(sha256(DER SPKI of rekorKey)), the name a bundle
	// gives its log and the key the trusted root is looked up by.
	logID string
	// integrated is the signing time every bundle and certificate in a
	// test is anchored to, one second resolution like Rekor's.
	integrated time.Time
}

func newSigstoreFixture(t *testing.T) *sigstoreFixture {
	t.Helper()
	now := time.Now().Truncate(time.Second)
	rootKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generate root key: %v", err)
	}
	rootTemplate := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "sigstore", Organization: []string{"sigstore.dev"}},
		NotBefore:             now.Add(-5 * time.Hour),
		NotAfter:              now.Add(5 * time.Hour),
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
		BasicConstraintsValid: true,
		IsCA:                  true,
	}
	rootDER, err := x509.CreateCertificate(rand.Reader, rootTemplate, rootTemplate, &rootKey.PublicKey, rootKey)
	if err != nil {
		t.Fatalf("create root: %v", err)
	}
	root, err := x509.ParseCertificate(rootDER)
	if err != nil {
		t.Fatalf("parse root: %v", err)
	}
	rekorKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generate rekor key: %v", err)
	}
	spki, err := x509.MarshalPKIXPublicKey(&rekorKey.PublicKey)
	if err != nil {
		t.Fatalf("marshal rekor key: %v", err)
	}
	keyID := sha256.Sum256(spki)
	return &sigstoreFixture{
		root:       root,
		rootKey:    rootKey,
		rekorKey:   rekorKey,
		logID:      hex.EncodeToString(keyID[:]),
		integrated: now,
	}
}

// withIntermediate adds an intermediate CA issued by the root, with
// path length zero and no EKU like Fulcio's, and makes it the issuer of
// every leaf from here on.
func (f *sigstoreFixture) withIntermediate(t *testing.T) *sigstoreFixture {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generate intermediate key: %v", err)
	}
	template := &x509.Certificate{
		SerialNumber:          big.NewInt(3),
		Subject:               pkix.Name{CommonName: "sigstore-intermediate", Organization: []string{"sigstore.dev"}},
		NotBefore:             f.root.NotBefore,
		NotAfter:              f.root.NotAfter,
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
		BasicConstraintsValid: true,
		IsCA:                  true,
		MaxPathLen:            0,
		MaxPathLenZero:        true,
	}
	der, err := x509.CreateCertificate(rand.Reader, template, f.root, &key.PublicKey, f.rootKey)
	if err != nil {
		t.Fatalf("create intermediate: %v", err)
	}
	cert, err := x509.ParseCertificate(der)
	if err != nil {
		t.Fatalf("parse intermediate: %v", err)
	}
	f.intermediate, f.intermediateKey = cert, key
	return f
}

// issuer is the CA leaves are issued by: the intermediate when there
// is one, else the root.
func (f *sigstoreFixture) issuer() (*x509.Certificate, *ecdsa.PrivateKey) {
	if f.intermediate != nil {
		return f.intermediate, f.intermediateKey
	}
	return f.root, f.rootKey
}

// leaf issues a short-lived signing certificate the way Fulcio does: the
// code-signing EKU, the OIDC issuer in the non-critical extension
// 1.3.6.1.4.1.57264.1.8 as a DER UTF8String, and the identity as an
// email SAN. Validity is [notBefore, notAfter] so a test can place the
// integrated time outside it.
func (f *sigstoreFixture) leaf(t *testing.T, issuer, email string, notBefore, notAfter time.Time) (*x509.Certificate, signature.Signer) {
	t.Helper()
	return f.issue(t, f.leafTemplate(t, issuer, email, notBefore, notAfter))
}

// leafTemplate is the certificate leaf issues, before signing, so a test
// can alter one field (the EKU, say) and pass it to issue.
func (f *sigstoreFixture) leafTemplate(t *testing.T, issuer, email string, notBefore, notAfter time.Time) *x509.Certificate {
	t.Helper()
	issuerDER, err := asn1.MarshalWithParams(issuer, "utf8")
	if err != nil {
		t.Fatalf("marshal issuer: %v", err)
	}
	return &x509.Certificate{
		SerialNumber:    big.NewInt(2),
		NotBefore:       notBefore,
		NotAfter:        notAfter,
		KeyUsage:        x509.KeyUsageDigitalSignature,
		ExtKeyUsage:     []x509.ExtKeyUsage{x509.ExtKeyUsageCodeSigning},
		EmailAddresses:  []string{email},
		ExtraExtensions: []pkix.Extension{{Id: certificate.OIDIssuerV2, Value: issuerDER}},
	}
}

// issue signs template with the fixture's issuing CA over a fresh P-256
// key and returns the certificate with a signer holding that key.
func (f *sigstoreFixture) issue(t *testing.T, template *x509.Certificate) (*x509.Certificate, signature.Signer) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generate leaf key: %v", err)
	}
	parent, parentKey := f.issuer()
	der, err := x509.CreateCertificate(rand.Reader, template, parent, &key.PublicKey, parentKey)
	if err != nil {
		t.Fatalf("create leaf: %v", err)
	}
	cert, err := x509.ParseCertificate(der)
	if err != nil {
		t.Fatalf("parse leaf: %v", err)
	}
	signer, err := signature.LoadECDSASignerVerifier(key, crypto.SHA256)
	if err != nil {
		t.Fatalf("load leaf signer: %v", err)
	}
	return cert, signer
}

// bundle is the dev.sigstore.cosign/bundle annotation for a signature
// b64sig by cert over payload, logged at f.integrated: a hashedrekord
// body naming the three, and a signed entry timestamp by the Rekor key
// over the RFC 8785 form of {body, integratedTime, logIndex, logID}.
func (f *sigstoreFixture) bundle(t *testing.T, b64sig string, cert *x509.Certificate, payload []byte) string {
	t.Helper()
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert.Raw})
	payloadDigest := sha256.Sum256(payload)
	var body hashedRekordBody
	body.APIVersion = "0.0.1"
	body.Kind = "hashedrekord"
	body.Spec.Data.Hash.Algorithm = "sha256"
	body.Spec.Data.Hash.Value = hex.EncodeToString(payloadDigest[:])
	body.Spec.Signature.Content = b64sig
	body.Spec.Signature.PublicKey.Content = base64.StdEncoding.EncodeToString(certPEM)
	bodyJSON, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal body: %v", err)
	}
	rekorPayload := rekorPayload{
		Body:           base64.StdEncoding.EncodeToString(bodyJSON),
		IntegratedTime: f.integrated.Unix(),
		LogIndex:       42,
		LogID:          f.logID,
	}
	payloadJSON, err := json.Marshal(rekorPayload)
	if err != nil {
		t.Fatalf("marshal rekor payload: %v", err)
	}
	canonical, err := jsoncanonicalizer.Transform(payloadJSON)
	if err != nil {
		t.Fatalf("canonicalize rekor payload: %v", err)
	}
	setDigest := sha256.Sum256(canonical)
	set, err := ecdsa.SignASN1(rand.Reader, f.rekorKey, setDigest[:])
	if err != nil {
		t.Fatalf("sign set: %v", err)
	}
	out, err := json.Marshal(rekorBundle{SignedEntryTimestamp: set, Payload: rekorPayload})
	if err != nil {
		t.Fatalf("marshal bundle: %v", err)
	}
	return string(out)
}

// trustedRootFile writes a Sigstore TrustedRoot (protojson field names,
// bytes as base64) listing f.root as the one certificate authority,
// with the intermediate before it in certChain when there is one, valid
// for five hours either side of the integrated time, and the Rekor key
// as the one transparency log, and returns its path.
func (f *sigstoreFixture) trustedRootFile(t *testing.T) string {
	t.Helper()
	return writeTrustedRoot(t, f.trustedRootDoc(t))
}

// trustedRootDoc is the TrustedRoot trustedRootFile writes, before
// marshalling, so a test can add a log or move a validity window.
func (f *sigstoreFixture) trustedRootDoc(t *testing.T) map[string]any {
	t.Helper()
	spki, err := x509.MarshalPKIXPublicKey(&f.rekorKey.PublicKey)
	if err != nil {
		t.Fatalf("marshal rekor key: %v", err)
	}
	keyID, err := hex.DecodeString(f.logID)
	if err != nil {
		t.Fatalf("decode log id: %v", err)
	}
	// Leaf-to-root order, as the public root lists Fulcio's chain.
	var chain []map[string]any
	if f.intermediate != nil {
		chain = append(chain, map[string]any{"rawBytes": base64.StdEncoding.EncodeToString(f.intermediate.Raw)})
	}
	chain = append(chain, map[string]any{"rawBytes": base64.StdEncoding.EncodeToString(f.root.Raw)})
	doc := map[string]any{
		"mediaType": "application/vnd.dev.sigstore.trustedroot+json;version=0.1",
		"certificateAuthorities": []map[string]any{{
			"subject": map[string]any{"organization": "sigstore.dev", "commonName": "sigstore"},
			"uri":     "https://fulcio.test",
			"certChain": map[string]any{
				"certificates": chain,
			},
			"validFor": map[string]any{
				"start": f.integrated.Add(-5 * time.Hour).UTC().Format(time.RFC3339),
				"end":   f.integrated.Add(5 * time.Hour).UTC().Format(time.RFC3339),
			},
		}},
		"tlogs": []map[string]any{{
			"baseUrl":       "https://rekor.test",
			"hashAlgorithm": "SHA2_256",
			"publicKey": map[string]any{
				"rawBytes":   base64.StdEncoding.EncodeToString(spki),
				"keyDetails": "PKIX_ECDSA_P256_SHA_256",
				"validFor":   map[string]any{"start": f.integrated.Add(-5 * time.Hour).UTC().Format(time.RFC3339)},
			},
			"logId": map[string]any{"keyId": base64.StdEncoding.EncodeToString(keyID)},
		}},
	}
	return doc
}

// writeTrustedRoot marshals doc to a file and returns its path.
func writeTrustedRoot(t *testing.T, doc map[string]any) string {
	t.Helper()
	raw, err := json.Marshal(doc)
	if err != nil {
		t.Fatalf("marshal trusted root: %v", err)
	}
	path := filepath.Join(t.TempDir(), "trusted_root.json")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatalf("write trusted root: %v", err)
	}
	return path
}

// keylessAnnotations is the annotation map cosign writes for a keyless
// signature layer: the signature, the certificate and the Rekor bundle,
// plus the chain annotation holding the intermediate when the fixture
// has one. The chain is left out when the leaf is issued by the root.
func (f *sigstoreFixture) keylessAnnotations(b64sig string, cert *x509.Certificate, bundle string) map[string]string {
	out := map[string]string{
		cosignSignatureAnnotation:   b64sig,
		cosignCertificateAnnotation: string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert.Raw})),
		cosignBundleAnnotation:      bundle,
	}
	if f.intermediate != nil {
		out[cosignChainAnnotation] = string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: f.intermediate.Raw}))
	}
	return out
}

const (
	keylessIssuer  = "https://token.actions.githubusercontent.com"
	keylessSubject = "signer@example.com"
	keylessRef     = "docker.io/org/model:v1"
)

// keylessLayer is one well-formed keyless signature layer over
// keylessRef/targetA from the fixture, with its leaf valid for ten
// minutes around the integrated time.
func keylessLayer(t *testing.T, f *sigstoreFixture) ([]byte, map[string]string) {
	t.Helper()
	cert, signer := f.leaf(t, keylessIssuer, keylessSubject, f.integrated.Add(-time.Minute), f.integrated.Add(10*time.Minute))
	payload, b64 := signedPayload(t, signer, keylessRef, targetA)
	return payload, f.keylessAnnotations(b64, cert, f.bundle(t, b64, cert, payload))
}

// A leaf issued by an intermediate, which is what public Fulcio does:
// the chain annotation carries the intermediate, and the trusted root
// may list it too (as the public root does) or leave it to the layer.
func TestVerifySignatureLayerKeylessAcceptsIntermediateIssuedLeaf(t *testing.T) {
	f := newSigstoreFixture(t).withIntermediate(t)
	payload, annotations := keylessLayer(t, f)
	if annotations[cosignChainAnnotation] == "" {
		t.Fatal("fixture layer carries no chain annotation")
	}
	rootOnly := f.trustedRootDoc(t)
	rootOnly["certificateAuthorities"].([]map[string]any)[0]["certChain"] = map[string]any{
		"certificates": []map[string]any{{"rawBytes": base64.StdEncoding.EncodeToString(f.root.Raw)}},
	}
	for _, tc := range []struct {
		name string
		root string
	}{
		{"intermediate in trusted root and chain annotation", f.trustedRootFile(t)},
		{"intermediate in the chain annotation only", writeTrustedRoot(t, rootOnly)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			identity := &identityPolicy{Issuer: keylessIssuer, Subject: keylessSubject, TrustedRoot: tc.root}
			match, err := verifySignatureLayer(payload, annotations, keylessRef, digest.Digest(targetA), verifyPolicy{identity: identity}, "")
			if err != nil {
				t.Fatalf("verifySignatureLayer: %v", err)
			}
			if want := keylessIssuer + " " + keylessSubject; match.Signer != want {
				t.Errorf("match.Signer = %q, want %q", match.Signer, want)
			}
		})
	}

	// Without the chain annotation the intermediate must come from the
	// trusted root; a root-only file then has nothing to bridge the gap.
	bare := map[string]string{}
	for k, v := range annotations {
		if k != cosignChainAnnotation {
			bare[k] = v
		}
	}
	identity := &identityPolicy{Issuer: keylessIssuer, Subject: keylessSubject, TrustedRoot: writeTrustedRoot(t, rootOnly)}
	if _, err := verifySignatureLayer(payload, bare, keylessRef, digest.Digest(targetA), verifyPolicy{identity: identity}, ""); err == nil || !strings.Contains(err.Error(), "does not chain to a trusted root") {
		t.Fatalf("no intermediate anywhere: err = %v, want does not chain", err)
	}
}

// A workflow identity is a URI SAN, not an email: the subject is
// matched against every SAN kind cryptoutils reports.
func TestVerifySignatureLayerKeylessAcceptsURISubject(t *testing.T) {
	f := newSigstoreFixture(t)
	const workflow = "https://github.com/org/models/.github/workflows/sign.yml@refs/heads/main"
	u, err := url.Parse(workflow)
	if err != nil {
		t.Fatalf("parse workflow URI: %v", err)
	}
	template := f.leafTemplate(t, keylessIssuer, "", f.integrated.Add(-time.Minute), f.integrated.Add(10*time.Minute))
	template.EmailAddresses = nil
	template.URIs = []*url.URL{u}
	cert, signer := f.issue(t, template)
	payload, b64 := signedPayload(t, signer, keylessRef, targetA)
	annotations := f.keylessAnnotations(b64, cert, f.bundle(t, b64, cert, payload))

	identity := &identityPolicy{Issuer: keylessIssuer, Subject: workflow, TrustedRoot: f.trustedRootFile(t)}
	match, err := verifySignatureLayer(payload, annotations, keylessRef, digest.Digest(targetA), verifyPolicy{identity: identity}, "")
	if err != nil {
		t.Fatalf("URI SAN subject: %v", err)
	}
	if want := keylessIssuer + " " + workflow; match.Signer != want {
		t.Errorf("match.Signer = %q, want %q", match.Signer, want)
	}
	identity = &identityPolicy{Issuer: keylessIssuer, Subject: "https://github.com/other/models/.github/workflows/sign.yml@refs/heads/main", TrustedRoot: f.trustedRootFile(t)}
	if _, err := verifySignatureLayer(payload, annotations, keylessRef, digest.Digest(targetA), verifyPolicy{identity: identity}, ""); err == nil || !strings.Contains(err.Error(), "do not include") {
		t.Fatalf("other workflow URI: err = %v, want do not include", err)
	}
}

// certificateIssuer reads the Fulcio issuer from 1.3.6.1.4.1.57264.1.8
// (DER UTF8String) first and falls back to the raw-bytes 1.3.6.1.4.1.57264.1.1
// that preceded it; a malformed .1.8 is refused rather than skipped.
func TestCertificateIssuer(t *testing.T) {
	f := newSigstoreFixture(t)
	v2, err := asn1.MarshalWithParams("https://v2.issuer", "utf8")
	if err != nil {
		t.Fatalf("marshal issuer: %v", err)
	}
	v2ext := pkix.Extension{Id: certificate.OIDIssuerV2, Value: v2}
	v1ext := pkix.Extension{Id: certificate.OIDIssuer, Value: []byte("https://v1.issuer")} //nolint:staticcheck // the deprecated OID is the one under test
	malformed := pkix.Extension{Id: certificate.OIDIssuerV2, Value: []byte("https://not.der")}
	for _, tc := range []struct {
		name    string
		exts    []pkix.Extension
		want    string
		wantErr string
	}{
		{".1.8 alone", []pkix.Extension{v2ext}, "https://v2.issuer", ""},
		{".1.1 alone", []pkix.Extension{v1ext}, "https://v1.issuer", ""},
		{".1.8 takes precedence over .1.1", []pkix.Extension{v1ext, v2ext}, "https://v2.issuer", ""},
		{"malformed .1.8 is an error even beside .1.1", []pkix.Extension{v1ext, malformed}, "", "certificate issuer extension"},
		{"neither", nil, "", "no Fulcio issuer extension"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			template := f.leafTemplate(t, "unused", keylessSubject, f.integrated.Add(-time.Minute), f.integrated.Add(10*time.Minute))
			template.ExtraExtensions = tc.exts
			cert, _ := f.issue(t, template)
			got, err := certificateIssuer(cert)
			switch {
			case tc.wantErr != "":
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("err = %v, want %q", err, tc.wantErr)
				}
			case err != nil:
				t.Fatalf("certificateIssuer: %v", err)
			case got != tc.want:
				t.Errorf("certificateIssuer = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestVerifySignatureLayerKeylessAcceptsMatchingIdentity(t *testing.T) {
	f := newSigstoreFixture(t)
	payload, annotations := keylessLayer(t, f)
	for _, tc := range []struct {
		name            string
		issuer, subject string
		wantErr         string
	}{
		{"issuer and subject", keylessIssuer, keylessSubject, ""},
		// One half alone is refused: the shim does not widen the
		// check even if a caller hands it a partial identity.
		{"issuer only", keylessIssuer, "", "both an issuer and a subject"},
		{"subject only", "", keylessSubject, "both an issuer and a subject"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			identity := &identityPolicy{Issuer: tc.issuer, Subject: tc.subject, TrustedRoot: f.trustedRootFile(t)}
			match, err := verifySignatureLayer(payload, annotations, keylessRef, digest.Digest(targetA), verifyPolicy{identity: identity}, "")
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("err = %v, want %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("verifySignatureLayer: %v", err)
			}
			// The signer label names the certificate.
			if want := keylessIssuer + " " + keylessSubject; match.Signer != want {
				t.Errorf("match.Signer = %q, want %q", match.Signer, want)
			}
			if match.KeyPath != "" {
				t.Errorf("match.KeyPath = %q, want empty for an identity match", match.KeyPath)
			}
			if match.Identity != "docker.io/org/model" {
				t.Errorf("match.Identity = %q, want the repository", match.Identity)
			}
		})
	}
}

func TestVerifySignatureLayerKeylessRefusals(t *testing.T) {
	f := newSigstoreFixture(t)
	goodPayload, good := keylessLayer(t, f)
	rule := func(issuer, subject string) *identityPolicy {
		return &identityPolicy{Issuer: issuer, Subject: subject, TrustedRoot: f.trustedRootFile(t)}
	}
	// with returns a copy of good with one annotation replaced.
	with := func(name, value string) map[string]string {
		out := map[string]string{}
		for k, v := range good {
			out[k] = v
		}
		out[name] = value
		return out
	}

	// A leaf whose validity starts after the integrated time; the
	// signature itself is sound, only the moment of signing is outside
	// the certificate's life.
	lateCert, lateSigner := f.leaf(t, keylessIssuer, keylessSubject, f.integrated.Add(time.Hour), f.integrated.Add(2*time.Hour))
	latePayload, lateB64 := signedPayload(t, lateSigner, keylessRef, targetA)
	late := f.keylessAnnotations(lateB64, lateCert, f.bundle(t, lateB64, lateCert, latePayload))

	// A leaf that expired an hour before the integrated time.
	expiredCert, expiredSigner := f.leaf(t, keylessIssuer, keylessSubject, f.integrated.Add(-2*time.Hour), f.integrated.Add(-time.Hour))
	expiredPayload, expiredB64 := signedPayload(t, expiredSigner, keylessRef, targetA)
	expired := f.keylessAnnotations(expiredB64, expiredCert, f.bundle(t, expiredB64, expiredCert, expiredPayload))

	// A SET with one byte flipped.
	var tampered rekorBundle
	if err := json.Unmarshal([]byte(good[cosignBundleAnnotation]), &tampered); err != nil {
		t.Fatalf("parse bundle: %v", err)
	}
	tampered.SignedEntryTimestamp[len(tampered.SignedEntryTimestamp)-1] ^= 0x01
	tamperedJSON, err := json.Marshal(tampered)
	if err != nil {
		t.Fatalf("marshal bundle: %v", err)
	}

	// A log entry about some other signature by the same certificate,
	// and one about this signature over some other payload.
	goodCert, err := cryptoutils.UnmarshalCertificatesFromPEM([]byte(good[cosignCertificateAnnotation]))
	if err != nil || len(goodCert) != 1 {
		t.Fatalf("parse good certificate: %v (%d certs)", err, len(goodCert))
	}
	otherSig := base64.StdEncoding.EncodeToString([]byte("not the signature"))
	otherPayload := append([]byte(nil), goodPayload...)
	otherPayload[len(otherPayload)-1] ^= 0x01

	// A second root-issued certificate with the same identity: logged in
	// place of the layer certificate, and used to sign a payload that is
	// then presented under the good certificate with an honest bundle,
	// so every check before the final signature check passes.
	otherCert, otherSigner := f.leaf(t, keylessIssuer, keylessSubject, f.integrated.Add(-time.Minute), f.integrated.Add(10*time.Minute))
	foreignPayload, foreignB64 := signedPayload(t, otherSigner, keylessRef, targetA)
	foreign := f.keylessAnnotations(foreignB64, goodCert[0], f.bundle(t, foreignB64, goodCert[0], foreignPayload))

	// A root-issued leaf with only the server-auth EKU; sound in every
	// other respect.
	serverTemplate := f.leafTemplate(t, keylessIssuer, keylessSubject, f.integrated.Add(-time.Minute), f.integrated.Add(10*time.Minute))
	serverTemplate.ExtKeyUsage = []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth}
	serverCert, serverSigner := f.issue(t, serverTemplate)
	serverPayload, serverB64 := signedPayload(t, serverSigner, keylessRef, targetA)
	server := f.keylessAnnotations(serverB64, serverCert, f.bundle(t, serverB64, serverCert, serverPayload))

	for _, tc := range []struct {
		name        string
		payload     []byte
		annotations map[string]string
		identity    *identityPolicy
		want        string
	}{
		{"wrong issuer", goodPayload, good, rule("https://other.issuer", keylessSubject), "certificate issuer"},
		{"wrong subject", goodPayload, good, rule(keylessIssuer, "other@example.com"), "do not include"},
		{"leaf not yet valid at the integrated time", latePayload, late, rule(keylessIssuer, keylessSubject), "does not chain to a trusted root"},
		{"leaf expired before the integrated time", expiredPayload, expired, rule(keylessIssuer, keylessSubject), "does not chain to a trusted root"},
		{"tampered signed entry timestamp", goodPayload, with(cosignBundleAnnotation, string(tamperedJSON)), rule(keylessIssuer, keylessSubject), "signed entry timestamp does not verify"},
		{"body signature is another signature", goodPayload, with(cosignBundleAnnotation, f.bundle(t, otherSig, goodCert[0], goodPayload)), rule(keylessIssuer, keylessSubject), "body signature is not the layer signature"},
		{"body hash is another payload", goodPayload, with(cosignBundleAnnotation, f.bundle(t, good[cosignSignatureAnnotation], goodCert[0], otherPayload)), rule(keylessIssuer, keylessSubject), "body hash is not the payload"},
		// Signature and payload untouched; only the logged certificate differs.
		{"body certificate is another certificate", goodPayload, with(cosignBundleAnnotation, f.bundle(t, good[cosignSignatureAnnotation], otherCert, goodPayload)), rule(keylessIssuer, keylessSubject), "body certificate is not the layer certificate"},
		{"leaf without the code-signing EKU", serverPayload, server, rule(keylessIssuer, keylessSubject), "does not chain to a trusted root"},
		{"signature by another key", foreignPayload, foreign, rule(keylessIssuer, keylessSubject), "does not verify under the certificate key"},
		{"missing bundle", goodPayload, with(cosignBundleAnnotation, ""), rule(keylessIssuer, keylessSubject), "has no " + cosignBundleAnnotation},
		{"missing certificate", goodPayload, with(cosignCertificateAnnotation, ""), rule(keylessIssuer, keylessSubject), "has no " + cosignCertificateAnnotation},
		{"missing trusted root", goodPayload, good, &identityPolicy{Issuer: keylessIssuer, Subject: keylessSubject, TrustedRoot: filepath.Join(t.TempDir(), "missing.json")}, "read trusted root"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := verifySignatureLayer(tc.payload, tc.annotations, keylessRef, digest.Digest(targetA), verifyPolicy{identity: tc.identity}, "")
			if err == nil {
				t.Fatal("verified, want refusal")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("err = %v, want %q", err, tc.want)
			}
			// Every refusal here is a hard error about the signature or
			// the configuration, never the "no key accepted" wording a
			// caller reads as unsigned.
			if strings.Contains(err.Error(), "no trusted key accepted") {
				t.Errorf("err = %v reads as unsigned", err)
			}
		})
	}
}

// A self-signed certificate in the chain annotation is not a root: only
// the trusted root file anchors a chain, so a registry serving its own
// CA alongside a leaf it issued is refused.
func TestVerifySignatureLayerKeylessIgnoresSelfSignedChainEntry(t *testing.T) {
	f := newSigstoreFixture(t)
	rogue := newSigstoreFixture(t)
	rogue.integrated = f.integrated
	rogue.rekorKey, rogue.logID = f.rekorKey, f.logID
	cert, signer := rogue.leaf(t, keylessIssuer, keylessSubject, f.integrated.Add(-time.Minute), f.integrated.Add(10*time.Minute))
	payload, b64 := signedPayload(t, signer, keylessRef, targetA)
	annotations := rogue.keylessAnnotations(b64, cert, rogue.bundle(t, b64, cert, payload))
	annotations[cosignChainAnnotation] = string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: rogue.root.Raw}))

	identity := &identityPolicy{Issuer: keylessIssuer, Subject: keylessSubject, TrustedRoot: f.trustedRootFile(t)}
	_, err := verifySignatureLayer(payload, annotations, keylessRef, digest.Digest(targetA), verifyPolicy{identity: identity}, "")
	if err == nil || !strings.Contains(err.Error(), "does not chain to a trusted root") {
		t.Fatalf("rogue root in chain: err = %v, want does not chain", err)
	}
}

// The public Sigstore TrustedRoot lists rekor.sigstore.dev with an ECDSA
// P-256 key and log2025-1.rekor.sigstore.dev with an Ed25519 key. A
// root with both must parse, and a bundle naming the ECDSA log must
// still verify; the Ed25519 key only matters to a bundle that names it.
func TestParseTrustedRootKeepsNonECDSALogKeys(t *testing.T) {
	f := newSigstoreFixture(t)
	edPub, _, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("generate ed25519 key: %v", err)
	}
	edSPKI, err := x509.MarshalPKIXPublicKey(edPub)
	if err != nil {
		t.Fatalf("marshal ed25519 key: %v", err)
	}
	edID := sha256.Sum256(edSPKI)
	doc := f.trustedRootDoc(t)
	doc["tlogs"] = append(doc["tlogs"].([]map[string]any), map[string]any{
		"baseUrl":       "https://log2025-1.rekor.test",
		"hashAlgorithm": "SHA2_256",
		"publicKey": map[string]any{
			"rawBytes":   base64.StdEncoding.EncodeToString(edSPKI),
			"keyDetails": "PKIX_ED25519",
			"validFor":   map[string]any{"start": f.integrated.Add(-time.Hour).UTC().Format(time.RFC3339)},
		},
		"logId": map[string]any{"keyId": base64.StdEncoding.EncodeToString(edID[:])},
	})
	path := writeTrustedRoot(t, doc)

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read trusted root: %v", err)
	}
	root, err := parseTrustedRoot(raw)
	if err != nil {
		t.Fatalf("parseTrustedRoot: %v", err)
	}
	if len(root.logKeys) != 2 {
		t.Fatalf("parsed %d log keys, want 2", len(root.logKeys))
	}
	if _, ok := root.logKeys[hex.EncodeToString(edID[:])].key.(ed25519.PublicKey); !ok {
		t.Errorf("ed25519 log key not kept as ed25519.PublicKey")
	}

	payload, annotations := keylessLayer(t, f)
	identity := &identityPolicy{Issuer: keylessIssuer, Subject: keylessSubject, TrustedRoot: path}
	if _, err := verifySignatureLayer(payload, annotations, keylessRef, digest.Digest(targetA), verifyPolicy{identity: identity}, ""); err != nil {
		t.Fatalf("bundle naming the ECDSA log beside an Ed25519 log: %v", err)
	}
}

// A SET is only the log's word while its key was in use: tlogs[]
// .publicKey.validFor bounds the integrated times the key vouches for,
// the same way certificateAuthorities[].validFor bounds the issuers.
func TestVerifyKeylessHonoursLogKeyValidFor(t *testing.T) {
	f := newSigstoreFixture(t)
	payload, annotations := keylessLayer(t, f)
	stamp := func(d time.Duration) string { return f.integrated.Add(d).UTC().Format(time.RFC3339) }
	for _, tc := range []struct {
		name     string
		validFor map[string]any
		wantErr  string
	}{
		{"inside the window", map[string]any{"start": stamp(-time.Hour), "end": stamp(time.Hour)}, ""},
		{"no end", map[string]any{"start": stamp(-time.Hour)}, ""},
		{"before start", map[string]any{"start": stamp(time.Hour)}, "outside the validity window"},
		{"after end", map[string]any{"start": stamp(-2 * time.Hour), "end": stamp(-time.Hour)}, "outside the validity window"},
		{"no start", map[string]any{"end": stamp(time.Hour)}, "no publicKey.validFor.start"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			doc := f.trustedRootDoc(t)
			doc["tlogs"].([]map[string]any)[0]["publicKey"].(map[string]any)["validFor"] = tc.validFor
			identity := &identityPolicy{Issuer: keylessIssuer, Subject: keylessSubject, TrustedRoot: writeTrustedRoot(t, doc)}
			_, err := verifySignatureLayer(payload, annotations, keylessRef, digest.Digest(targetA), verifyPolicy{identity: identity}, "")
			switch {
			case tc.wantErr == "" && err != nil:
				t.Fatalf("err = %v, want verified", err)
			case tc.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErr)):
				t.Fatalf("err = %v, want %q", err, tc.wantErr)
			}
		})
	}
}

// A trusted root that cannot be read is a configuration failure, so
// verifySignatures returns it as an error (which the Rust side maps to
// an indeterminate verdict) rather than a report that reads as
// "unsigned" and would have a cached copy rejected over it. It is
// checked before any registry access, hence no registry in this test.
func TestVerifySignaturesRefusesUnreadableTrustedRoot(t *testing.T) {
	for _, tc := range []struct {
		name, path, want string
	}{
		{"missing", filepath.Join(t.TempDir(), "missing.json"), "read trusted root"},
		{"malformed", writeFile(t, "trusted_root.json", "{not json"), "parse trusted root"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			identity := &identityPolicy{Issuer: keylessIssuer, Subject: keylessSubject, TrustedRoot: tc.path}
			report, err := verifySignatures(context.Background(), keylessRef, digest.Digest(targetA), nil, identity, "")
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("report = %+v, err = %v, want error containing %q", report, err, tc.want)
			}
		})
	}
}

// writeFile writes content to name under a fresh temp dir and returns the path.
func writeFile(t *testing.T, name, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
	return path
}
