// transfer_common.go — backend-agnostic helpers shared by both
// transfer_docker.go (!podman) and transfer_podman.go (podman).
package main

import (
	"encoding/json"
	"strings"

	digest "github.com/opencontainers/go-digest"
)

// transferOutcome is the `data` payload of llmman_transfer's response
// envelope (see ffi::TransferOutcome / cmd::transfer::run).
//
// Changed reports whether a transfer actually pushed anything new, or
// found the destination already up to date with the source and did
// nothing. Digest is the manifest digest now at the destination, which
// `--sign-key` signs — carried out of the transfer rather than
// re-resolved from the destination afterwards, so there is no window in
// which the tag could move between what was pushed and what gets signed.
//
// This used to be a bare "changed"/"unchanged" string; the JSON object
// is what made room for Digest.
type transferOutcome struct {
	Changed bool   `json:"changed"`
	Digest  string `json:"digest,omitempty"`
}

// transferResultJSON renders a completed transfer as that payload.
// Returns a plain string rather than the usual *C.char envelope so this
// file stays free of cgo and directly unit-testable; each backend's
// llmman_transfer wraps the result in okResp itself.
//
// A zero digest (an empty digest.Digest, which only arises if a backend
// ever returns success without one) marshals away entirely via omitempty
// rather than as the string "", so a caller sees "absent", not "empty".
func transferResultJSON(changed bool, pushed digest.Digest) string {
	out := transferOutcome{Changed: changed}
	if pushed != "" {
		out.Digest = pushed.String()
	}
	// Cannot fail: two scalar fields, no cycles, no unsupported types.
	data, _ := json.Marshal(out)
	return string(data)
}

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
