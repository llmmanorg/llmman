//go:build !podman

package main

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestCosignCanVerifyWhatWeSign is the interop claim, checked against
// the real cosign binary rather than against our own reading of its
// format. Skipped when cosign isn't installed.
func TestCosignCanVerifyWhatWeSign(t *testing.T) {
	cosign, err := exec.LookPath("cosign")
	if err != nil {
		t.Skip("cosign not installed")
	}
	ctx := context.Background()
	host := startRegistry(t)
	ref := host + "/org/model:v1"
	target := publishModel(t, ref)

	keys := t.TempDir()
	privPath, pubPath := writeKeyPair(t, keys, "signer")
	if err := signManifest(ctx, ref, target, privPath, nil); err != nil {
		t.Fatalf("sign: %v", err)
	}

	cmd := exec.Command(cosign, "verify", "--insecure-ignore-tlog=true",
		"--allow-http-registry", "--key", pubPath, ref)
	cmd.Env = append(os.Environ(), "COSIGN_EXPERIMENTAL=0")
	out, err := cmd.CombinedOutput()
	t.Logf("cosign verify output:\n%s", out)
	if err != nil {
		t.Fatalf("cosign could not verify a signature llmman wrote: %v", err)
	}
}

// TestABundleSignedManifestIsReportedAsSuchNotAsUnsigned pins the
// honesty of the one interop direction that does *not* work.
//
// Current cosign publishes a sigstore bundle at the suffix-less fallback
// tag rather than the simple-signing artifact llmman reads, so a
// cosign-signed model does not verify here. What must not happen is
// llmman calling it "unsigned": that would send a user who did
// everything right off to debug a signature that is present and fine.
func TestABundleSignedManifestIsReportedAsSuchNotAsUnsigned(t *testing.T) {
	cosign, err := exec.LookPath("cosign")
	if err != nil {
		t.Skip("cosign not installed")
	}
	ctx := context.Background()
	host := startRegistry(t)
	ref := host + "/org/model:v1"
	target := publishModel(t, ref)

	keys := t.TempDir()
	gen := exec.Command(cosign, "generate-key-pair")
	gen.Dir = keys
	gen.Env = append(os.Environ(), "COSIGN_PASSWORD=")
	if out, err := gen.CombinedOutput(); err != nil {
		t.Fatalf("cosign generate-key-pair: %v\n%s", err, out)
	}

	cmd := exec.Command(cosign, "sign", "--yes", "--tlog-upload=false",
		"--use-signing-config=false", "--allow-http-registry",
		"--key", filepath.Join(keys, "cosign.key"), ref+"@"+target.String())
	cmd.Env = append(os.Environ(), "COSIGN_PASSWORD=")
	out, err := cmd.CombinedOutput()
	t.Logf("cosign sign output:\n%s", out)
	if err != nil {
		t.Skipf("this cosign build could not sign against the test registry: %v", err)
	}

	report, err := verifySignatures(ctx, ref, target, []string{filepath.Join(keys, "cosign.pub")})
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	// Not verified — that part is correct and must stay fail-closed.
	if report.Verified {
		t.Fatal("a bundle-format signature verified, which this build cannot actually check")
	}
	if !strings.Contains(report.Reason, "sigstore bundle") {
		t.Errorf("a bundle-signed manifest was reported as %q, want it named as a bundle", report.Reason)
	}
}
