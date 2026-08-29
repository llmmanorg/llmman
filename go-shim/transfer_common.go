// transfer_common.go — backend-agnostic helpers shared by both
// transfer_docker.go (!podman) and transfer_podman.go (podman).
package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	digest "github.com/opencontainers/go-digest"
)

// transferStatusChanged/transferStatusUnchanged are the `data` values
// llmman_transfer's response envelope carries back to the Rust CLI layer
// (see ffi::transfer / cmd::transfer::run) to report whether a transfer
// actually pushed anything new, or found the destination already up to
// date with the source and did nothing.
const (
	transferStatusChanged   = "changed"
	transferStatusUnchanged = "unchanged"
)

// transferViaStaging implements source→destination transfers that can't be
// done as a direct registry-to-registry blob transfer (ms://, ngc://, s3://,
// gs://, a local path) by pulling into a temporary local OCI layout and
// then pushing that layout to the destination registry. Identical for both
// the docker and podman backends since it only calls the shared
// pullToLayout/pushToRegistry entry points. Returns whether anything was
// actually pushed — see pushToRegistry.
func transferViaStaging(ctx context.Context, source, destination string) (changed bool, err error) {
	tmp, err := os.MkdirTemp("", "llmman-transfer-")
	if err != nil {
		return false, fmt.Errorf("create staging directory: %w", err)
	}
	defer os.RemoveAll(tmp)

	if err := pullToLayout(ctx, source, tmp); err != nil {
		return false, err
	}
	// findManifestForTransfer, not findManifestForPush: tmp is a fresh
	// directory holding exactly the one model just pulled, under
	// whatever ref pullToLayout normalized source to — which destination
	// may not match verbatim (that's the whole point of a transfer) — so
	// its single-entry fallback is safe here specifically.
	return pushToRegistry(ctx, tmp, destination, findManifestForTransfer)
}

type transferSourceKind int

const (
	sourceOCI transferSourceKind = iota
	sourceHF
	sourceOther
)

// classifySource mirrors pullToLayout's own OCI-vs-HuggingFace routing
// (known-host shortcuts, then a live /v2/ probe for anything else), so a
// source that `llmman pull` would treat as HuggingFace is also treated as
// HuggingFace here, and likewise for OCI registries. Returns the ref
// normalized the same way (tag defaulted to :latest, hf:// scheme
// resolved to a host-qualified hf.co/... form).
func classifySource(ctx context.Context, ref string) (transferSourceKind, string) {
	for _, scheme := range []string{"ms://", "modelscope://", "ngc://", "s3://", "gs://"} {
		if strings.HasPrefix(ref, scheme) {
			return sourceOther, ref
		}
	}
	if strings.HasPrefix(ref, "/") {
		return sourceOther, ref
	}
	if r, ok := cutAnyPrefix(ref, "hf://", "huggingface://"); ok {
		if strings.Count(r, "/") < 2 {
			r = "hf.co/" + r
		}
		return sourceHF, normalizeTag(r)
	}

	normalized := normalizeTag(ref)
	host := strings.SplitN(normalized, "/", 2)[0]
	if isOCIHost(ctx, host) {
		return sourceOCI, normalized
	}
	return sourceHF, normalized
}

func cutAnyPrefix(s string, prefixes ...string) (string, bool) {
	for _, p := range prefixes {
		if r, ok := strings.CutPrefix(s, p); ok {
			return r, true
		}
	}
	return s, false
}

func normalizeTag(ref string) string {
	if strings.LastIndex(ref, ":") <= strings.LastIndex(ref, "/") {
		return ref + ":latest"
	}
	return ref
}

func shortDigest(d digest.Digest) string {
	h := d.Hex()
	if len(h) > 12 {
		return h[:12]
	}
	return h
}
