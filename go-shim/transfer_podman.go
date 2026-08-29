//go:build podman

// transfer_podman.go — `llmman transfer`'s go.podman.io/image-backed
// implementation.
//
// For an OCI registry source, this is a one-line reuse of go.podman.io/image's
// own `copy.Image`: hand it a docker://source and a docker://destination
// reference directly, with no local OCI layout in between at all.
// copy.Image already streams each blob straight from source to destination
// (it reads the source manifest first, so every blob's digest/size is known
// up front, then opens a GetBlob reader and a PutBlob writer for each one —
// see copy/copy.go upstream); there's nothing llmman needs to add on top
// for that case.
//
// go.podman.io/image has no ms:///ngc:///s3:///gs:///local-path source
// transport, though, so those fall back to staging through a throwaway
// local OCI layout — pull, then push from it. A HuggingFace source never
// reaches here: it is handled in Rust (src/hf/transfer.rs).
package main

import (
	"context"
	"fmt"
	"os"

	"go.podman.io/image/v5/copy"
	"go.podman.io/image/v5/transports/alltransports"
)

// podmanTransfer returns whether anything was actually pushed to
// destination — see dockerTransfer's doc comment (transfer_docker.go) for
// why this matters for a re-run against an unchanged source.
func podmanTransfer(ctx context.Context, source, destination string) (changed bool, err error) {
	// See transfer_docker.go's dockerTransfer for why a tagless
	// destination must default to :latest explicitly here.
	destination = normalizeTag(destination)
	kind, normalized := classifySource(ctx, source)
	if kind == sourceOCI {
		return podmanTransferOCI(ctx, normalized, destination)
	}
	if kind == sourceHF {
		return false, errHFNotHandledHere(normalized)
	}
	return transferViaStaging(ctx, source, destination)
}

// podmanTransferOCI streams directly between two registries via
// go.podman.io/image's copy.Image — no local OCI layout involved. Routed
// through copyImageWithProgress (backend_podman.go), rather than a bare
// copy.Image call, purely so its ProgressEventDone/Skipped events can
// report whether anything was actually transferred — see that function's
// doc comment.
func podmanTransferOCI(ctx context.Context, source, destination string) (changed bool, err error) {
	srcStr := "docker://" + source
	srcRef, err := alltransports.ParseImageName(srcStr)
	if err != nil {
		return false, fmt.Errorf("parse src ref %q: %w", srcStr, err)
	}
	dstStr := "docker://" + destination
	dstRef, err := alltransports.ParseImageName(dstStr)
	if err != nil {
		return false, fmt.Errorf("parse dst ref %q: %w", dstStr, err)
	}

	pctx, err := insecurePolicy()
	if err != nil {
		return false, fmt.Errorf("policy context: %w", err)
	}
	defer pctx.Destroy()

	// "" — llmman transfer doesn't go through the daemon's per-model
	// progress poll (see progress_state.go), so there's no key to credit
	// these bytes to.
	if _, err := copyImageWithProgress(ctx, pctx, dstRef, srcRef, "Transferring", "Transferred", &copy.Options{
		ReportWriter: os.Stderr,
	}, &changed, ""); err != nil {
		return false, fmt.Errorf("transfer image: %w", err)
	}
	return changed, nil
}
