package main

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"
	"time"

	digest "github.com/opencontainers/go-digest"
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

func TestVerifySignatureLayerAcceptsAGoodSignature(t *testing.T) {
	signer, key := ecdsaSignerVerifier(t)
	const ref = "docker.io/org/model:v1"
	payload, b64 := signedPayload(t, signer, ref, targetA)

	match, err := verifySignatureLayer(payload, b64, ref, digest.Digest(targetA), []trustedKey{key}, "")
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
	if _, err := verifySignatureLayer(payload, b64, ref, digest.Digest(targetA), []trustedKey{{path: "ed25519.pub", verifier: verifier}}, ""); err != nil {
		t.Fatalf("verifySignatureLayer: %v", err)
	}
}

func TestVerifySignatureLayerRejectsUntrustedKey(t *testing.T) {
	signer, _ := ecdsaSignerVerifier(t)
	_, stranger := ecdsaSignerVerifier(t)
	const ref = "docker.io/org/model:v1"
	payload, b64 := signedPayload(t, signer, ref, targetA)

	_, err := verifySignatureLayer(payload, b64, ref, digest.Digest(targetA), []trustedKey{stranger}, "")
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

	_, err := verifySignatureLayer(payload, b64, ref, digest.Digest(targetB), []trustedKey{key}, "")
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

	match, err := verifySignatureLayer(payload, b64, "mirror.corp/myorg/name:v1", digest.Digest(targetA), []trustedKey{key}, "docker.io/myorg/**")
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
		_, err := verifySignatureLayer(payload, b64, "mirror.corp/myorg/name:v1", digest.Digest(targetA), []trustedKey{key}, claims)
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
	_, err = verifySignatureLayer(payload, base64.StdEncoding.EncodeToString([]byte("not a signature")), "mirror.corp/myorg/name:v1", digest.Digest(targetA), []trustedKey{key}, claims)
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

	_, err := verifySignatureLayer(payload, b64, "docker.io/attacker/model:v1", digest.Digest(targetA), []trustedKey{key}, "")
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
	_, err := verifySignatureLayer(tampered, b64, ref, digest.Digest(targetB), []trustedKey{key}, "")
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

	if _, err := verifySignatureLayer(payload, b64, ref, digest.Digest(targetA), []trustedKey{key}, ""); err == nil {
		t.Fatal("a payload of an unrelated signature type was accepted")
	}
}

func TestVerifySignatureLayerRejectsMalformedInput(t *testing.T) {
	_, key := ecdsaSignerVerifier(t)
	keys := []trustedKey{key}

	if _, err := verifySignatureLayer([]byte("{}"), "", "docker.io/org/model", digest.Digest(targetA), keys, ""); err == nil {
		t.Error("a layer with no signature annotation was accepted")
	}
	if _, err := verifySignatureLayer([]byte("{}"), "!!!not base64!!!", "docker.io/org/model", digest.Digest(targetA), keys, ""); err == nil {
		t.Error("a non-base64 signature was accepted")
	}
	if _, err := verifySignatureLayer([]byte("not json"), base64.StdEncoding.EncodeToString([]byte("sig")), "docker.io/org/model", digest.Digest(targetA), keys, ""); err == nil {
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
