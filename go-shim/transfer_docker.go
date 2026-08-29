//go:build !podman

// transfer_docker.go — `llmman transfer`'s docker/containerd-backed
// implementation.
//
// A direct transfer never fully materializes an image locally before
// pushing it: it already knows every blob's digest and size up front from
// the source's own OCI manifest, so it can open a reader on the source
// blob and a writer on the destination blob at the same time and stream
// one directly into the other. This file implements that property for
// OCI registry → OCI registry (dockerTransferOCI): trivial — the source
// manifest already gives every blob's digest/size, so it's a straight
// Fetcher → Pusher stream per blob.
//
// A HuggingFace source never reaches this file: it is handled in Rust
// (src/hf/transfer.rs).
//
// Anything else `llmman pull` understands (ms://, ngc://, s3://, gs://, a
// local path) falls back to transferViaStaging: pull into a throwaway
// local OCI layout, then push from it, exactly like `llmman transfer` did
// before this file existed. Reimplementing zero-disk streaming for every
// one of those source kinds isn't worth it yet — none of them are the
// large-model-file case this exists for.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/containerd/containerd/v2/core/remotes"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/vbauerster/mpb/v8"
)

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

// dockerTransfer returns whether anything was actually pushed to
// destination — false if the source turned out to already be identical
// (by digest) to what's already there, e.g. re-running a transfer for a
// model that hasn't been updated at its source since the last transfer.
func dockerTransfer(ctx context.Context, source, destination string) (changed bool, err error) {
	// A tagless destination (e.g. "docker.io/owner/repo") must default to
	// :latest here explicitly: unlike a local OCI layout's index.json
	// (which always has some ref-name annotation to look up),
	// resolver.Pusher parses the ref as given, and a repository object
	// left empty pushes the manifest addressable only by digest — no tag
	// is ever created, silently, so a plain `docker pull owner/repo`
	// afterwards would find nothing.
	destination = normalizeTag(destination)
	kind, normalized := classifySource(ctx, source)
	switch kind {
	case sourceOCI:
		return dockerTransferOCI(ctx, normalized, destination)
	case sourceHF:
		return false, errHFNotHandledHere(normalized)
	default:
		return transferViaStaging(ctx, source, destination)
	}
}

// ---------------------------------------------------------------------------
// OCI registry → OCI registry
// ---------------------------------------------------------------------------

func dockerTransferOCI(ctx context.Context, source, destination string) (changed bool, err error) {
	resolver := newResolver(ctx)
	name, manifestDesc, err := resolver.Resolve(ctx, source)
	if err != nil {
		return false, fmt.Errorf("resolve %s: %w", source, err)
	}
	fetcher, err := resolver.Fetcher(ctx, name)
	if err != nil {
		return false, fmt.Errorf("create fetcher: %w", err)
	}
	pusher, err := resolver.Pusher(ctx, destination)
	if err != nil {
		return false, fmt.Errorf("create pusher: %w", err)
	}

	rc, err := fetcher.Fetch(ctx, manifestDesc)
	if err != nil {
		return false, fmt.Errorf("fetch manifest: %w", err)
	}
	manifestData, err := io.ReadAll(rc)
	rc.Close()
	if err != nil {
		return false, fmt.Errorf("read manifest: %w", err)
	}

	var manifest ocispec.Manifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		// An image index (manifest list): push it as-is. Per-instance
		// (multi-arch) selection isn't implemented here.
		alreadyExists, err := pushBytes(ctx, pusher, manifestDesc, manifestData)
		return !alreadyExists, err
	}

	// "Transferring blob/config <digest>" progress bars — "Transferring",
	// not "Copying", because this is genuinely simultaneous: the fetch
	// from source and the push to destination happen at the same time,
	// streamed straight through with nothing landing on local disk in
	// between (see pushStreamLazy). "Copying" is left for llmman_push's
	// own progress bars (pushToRegistry in this file), where the source
	// is already sitting on local disk beforehand — an actual
	// one-directional copy, not a simultaneous transfer.
	//
	// Each blob gets its own progress pool (rather than one pool shared
	// across the whole manifest) because retryStream may need to restart
	// a blob's transfer from scratch after a transient failure — see
	// streamBlobFromFetcher — and a bar that's already partway through
	// can't be rewound back to zero for a retry; a fresh pool per attempt
	// sidesteps that instead of fighting it.
	changed = false
	streamOne := func(desc ocispec.Descriptor, kind string) error {
		short := shortDigest(desc.Digest)
		var alreadyExists bool
		err := retryStream(ctx, kind+" "+short, isHTTP4xx, func() error {
			prog := newProgressPool(40)
			newBar := func() *mpb.Bar {
				return addLayerBar(prog, "Transferring "+kind+" "+short, "Transferred  "+kind+" "+short, desc.Size, "")
			}
			exists, err := streamBlobFromFetcher(ctx, fetcher, pusher, desc, newBar)
			prog.Wait()
			if err != nil {
				return err
			}
			alreadyExists = exists
			return nil
		})
		if err != nil {
			return fmt.Errorf("push %s: %w", desc.Digest, err)
		}
		if alreadyExists {
			fmt.Fprintf(os.Stderr, "Transferred %s %s (already present)\n", kind, short)
		} else {
			changed = true
		}
		return nil
	}

	for _, layer := range manifest.Layers {
		if err := streamOne(layer, "blob"); err != nil {
			return false, err
		}
	}
	if err := streamOne(manifest.Config, "config"); err != nil {
		return false, err
	}

	// Manifest push: no progress bar (a few hundred bytes of JSON) — just
	// a plain "Writing manifest to image destination" message instead of
	// a bar for this step.
	manifestAlreadyExists, err := pushBytes(ctx, pusher, manifestDesc, manifestData)
	if err != nil {
		return false, err
	}
	if !manifestAlreadyExists {
		changed = true
	}
	if changed {
		fmt.Fprintln(os.Stderr, "Writing manifest to image destination")
	}
	return changed, nil
}

// streamBlobFromFetcher streams one blob from an OCI registry fetcher
// straight into a registry pusher, without ever touching local disk.
// Wrapped in a per-attempt context so a
// stalled source read (no bytes for dlStallTimeout) cancels this attempt
// instead of hanging forever — the caller (streamOne) is what actually
// retries a failed attempt from scratch via retryStream.
func streamBlobFromFetcher(ctx context.Context, fetcher remotes.Fetcher, pusher remotes.Pusher, desc ocispec.Descriptor, newBar func() *mpb.Bar) (alreadyExists bool, err error) {
	attemptCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	return pushStreamLazy(attemptCtx, pusher, desc, newBar, func() (io.ReadCloser, error) {
		rc, err := fetcher.Fetch(attemptCtx, desc)
		if err != nil {
			return nil, fmt.Errorf("fetch %s: %w", desc.Digest, err)
		}
		return newStallReadCloser(rc, dlStallTimeout, cancel), nil
	})
}
