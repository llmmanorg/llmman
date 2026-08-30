// classify.go — what reaches this shim, and what must never.
//
// Every reference is classified once, in Rust (src/hf/mod.rs's
// `classify`), before any FFI call. Only OCI registry references are
// routed here; HuggingFace goes to src/hf, and ms://, ngc://, s3://,
// gs:// and local paths to src/sources.
//
// So this is a tripwire, not a decision: normalize the tag, reject
// loudly what should never have arrived. No network probe. This used to
// end in a live GET of https://<host>/v2/, re-deriving "is this an OCI
// registry?" for a host Rust had already answered that about — an extra
// round trip per pull, and a bug when the two disagreed (Rust's probe
// also tries plain HTTP and accepts a bare 401, so a reference it
// correctly routed here could be rejected as HuggingFace). Ollama has no
// such probe either: it assumes the protocol and learns the auth
// challenge from a 401 on the real manifest GET (server/images.go's
// makeRequestWithRetry).
package main

import (
	"fmt"
	"strings"
)

// errHFNotHandledHere reports a HuggingFace reference reaching this shim.
// Unreachable unless hf::classify's routing regresses or something calls
// the C API directly — either of which should fail loudly rather than
// silently take a path that no longer exists.
func errHFNotHandledHere(ref string) error {
	return fmt.Errorf(
		"%s: HuggingFace references are handled natively in Rust (src/hf), not by the Go shim", ref)
}

// errSourceNotHandledHere is the same tripwire for the URI-scheme and
// local-path sources, which moved to Rust alongside the HuggingFace path.
func errSourceNotHandledHere(ref string) error {
	return fmt.Errorf(
		"%s: ms://, ngc://, s3://, gs:// and local-path sources are handled natively in Rust (src/sources), not by the Go shim", ref)
}

// isKnownHFHost returns true for the HuggingFace-compatible hosts Rust
// routes to src/hf. An allowlist, not a probe: its only job here is
// spotting a reference that should never have arrived.
func isKnownHFHost(host string) bool {
	switch host {
	case "hf.co", "huggingface.co", "modelscope.cn":
		return true
	}
	return false
}

// sourceSchemes are the URI schemes src/sources owns. A reference
// carrying one of these must never reach the registry client below.
var sourceSchemes = []string{"ms://", "modelscope://", "ngc://", "s3://", "gs://"}

// notHandledHere reports a reference that should have been routed away
// from this shim, or nil if ref really is a registry reference.
func notHandledHere(ref string) error {
	for _, scheme := range sourceSchemes {
		if strings.HasPrefix(ref, scheme) {
			return errSourceNotHandledHere(ref)
		}
	}
	if strings.HasPrefix(ref, "/") {
		return errSourceNotHandledHere(ref)
	}
	if isKnownHFHost(strings.SplitN(ref, "/", 2)[0]) {
		return errHFNotHandledHere(ref)
	}
	return nil
}

// classifyPullRef normalizes ref for both backends' pullToLayout and
// rejects anything that isn't an OCI registry reference.
func classifyPullRef(ref string) (normalizedRef string, err error) {
	ref = normalizeTag(ref)
	return ref, notHandledHere(ref)
}

// normalizeTag appends ":latest" unless ref already carries a tag or a
// digest. A ":" only starts a tag if it comes after the last "/", so a
// registry port ("host:5000/owner/repo") isn't mistaken for one.
func normalizeTag(ref string) string {
	if strings.LastIndex(ref, ":") <= strings.LastIndex(ref, "/") {
		return ref + ":latest"
	}
	return ref
}
