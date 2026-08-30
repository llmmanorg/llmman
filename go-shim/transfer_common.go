// transfer_common.go — backend-agnostic helpers shared by both
// transfer_docker.go (!podman) and transfer_podman.go (podman).
package main

import (
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

// classifySource normalizes an `llmman transfer` source for the registry
// client and rejects anything Rust should already have routed elsewhere
// — see classify.go for why this is a tripwire rather than a decision,
// and in particular why it no longer probes /v2/. A HuggingFace source
// (src/hf/transfer.rs) and the ms:///ngc:///s3:///gs:///local-path
// sources (src/sources) both reach their destination without this shim
// doing the pull at all; only the registry→registry case gets here.
func classifySource(ref string) (normalized string, err error) {
	if r, ok := cutAnyPrefix(ref, "hf://", "huggingface://"); ok {
		if strings.Count(r, "/") < 2 {
			r = "hf.co/" + r
		}
		return r, errHFNotHandledHere(normalizeTag(r))
	}
	normalized = normalizeTag(ref)
	return normalized, notHandledHere(normalized)
}

func cutAnyPrefix(s string, prefixes ...string) (string, bool) {
	for _, p := range prefixes {
		if r, ok := strings.CutPrefix(s, p); ok {
			return r, true
		}
	}
	return s, false
}

func shortDigest(d digest.Digest) string {
	h := d.Hex()
	if len(h) > 12 {
		return h[:12]
	}
	return h
}
