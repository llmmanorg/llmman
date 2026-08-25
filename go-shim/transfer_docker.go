//go:build !podman

// transfer_docker.go — `llmman transfer`'s docker/containerd-backed
// implementation.
//
// A direct transfer never fully materializes an image locally before
// pushing it: it already knows every blob's digest and size up front from
// the source's own OCI manifest, so it can open a reader on the source
// blob and a writer on the destination blob at the same time and stream
// one directly into the other. This file implements that property for
// two cases:
//
//   - OCI registry → OCI registry (dockerTransferOCI): trivial — the
//     source manifest already gives every blob's digest/size, so it's a
//     straight Fetcher → Pusher stream per blob.
//
//   - HuggingFace → OCI registry (dockerTransferHF): harder, because there
//     is no pre-existing manifest to read a digest from. But a HEAD
//     request against an LFS-tracked file's resolve URL exposes the real
//     content sha256 via the X-Linked-Etag header *before* any bytes are
//     downloaded (see hf.go's hfHeadMetadata) — which gives exactly the
//     "digest known ahead of time" property a registry push needs, the
//     same way an OCI manifest would. That's what makes streaming a
//     multi-gigabyte GGUF file straight from huggingface.co into a
//     registry possible without ever writing it to local disk.
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
	"net/http"
	"os"
	"path/filepath"

	"github.com/containerd/containerd/v2/core/remotes"
	modelspec "github.com/modelpack/model-spec/specs-go/v1"
	digest "github.com/opencontainers/go-digest"
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
		return dockerTransferHF(ctx, normalized, destination)
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
// straight into a registry pusher, same disk-free property as
// streamHFFileToRegistry below. Wrapped in a per-attempt context so a
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

// ---------------------------------------------------------------------------
// HuggingFace → OCI registry
// ---------------------------------------------------------------------------

// dockerTransferHF returns whether anything was actually pushed — false
// if the chosen HuggingFace file(s) and the CNCF manifest built from them
// all turned out to already be present at destination by digest (i.e.
// the repo's commit for this file hasn't changed since the last transfer
// to this destination).
func dockerTransferHF(ctx context.Context, ref, destination string) (changed bool, err error) {
	host, owner, repo, tag, err := parseHFRef(ref)
	if err != nil {
		return false, err
	}
	endpoint := hfEndpoint(host)
	token := hfToken()

	apiClient := hfAPIClient()
	dlClient := hfDownloadClient()

	info, err := hfFetchModelInfo(ctx, apiClient, endpoint, owner, repo, token)
	if err != nil {
		return false, err
	}
	commit := info.commit()
	meta := modelMeta{}
	if license, ok := info.license(); ok {
		meta.Licenses = []string{license}
	}
	files, err := hfFetchFiles(ctx, apiClient, endpoint, owner, repo, commit, token)
	if err != nil {
		return false, err
	}

	resolver := newResolver(ctx)
	pusher, err := resolver.Pusher(ctx, destination)
	if err != nil {
		return false, fmt.Errorf("create pusher: %w", err)
	}

	// Try GGUF first; fall back to safetensors if the repo has none — same
	// selection logic pullHF uses. shards is every part of a multi-part
	// split together (see selectGGUF/ggufShards in hf.go) — just one
	// element for an ordinarily single-file GGUF.
	if shards, err := selectGGUF(files, tag); err == nil {
		meta.Format = "gguf"
		var ggufLayers []ocispec.Descriptor
		for _, f := range shards {
			desc, blobChanged, err := streamHFFileToRegistry(
				ctx, dlClient, pusher, endpoint, owner, repo, commit, token,
				f, modelspec.MediaTypeModelWeightRaw,
			)
			if err != nil {
				return false, err
			}
			if blobChanged {
				changed = true
			}
			ggufLayers = append(ggufLayers, desc)
		}
		// filepathAnnotation only makes sense at the manifest level for
		// the single-file case — see storeGGUFAsOCI's own doc comment on
		// the same tradeoff for the local-store path.
		filepathAnnotation := ""
		if len(shards) == 1 {
			filepathAnnotation = filepath.Base(shards[0].Path)
		}
		// mmproj: an optional extra weight layer alongside the chosen
		// GGUF shard(s) — see selectMMProj's own doc comment (hf.go) for
		// why this has no dedicated media type of its own.
		if mmproj, ok := selectMMProj(files); ok {
			desc, blobChanged, err := streamHFFileToRegistry(
				ctx, dlClient, pusher, endpoint, owner, repo, commit, token,
				mmproj, modelspec.MediaTypeModelWeightRaw,
			)
			if err != nil {
				return false, err
			}
			if blobChanged {
				changed = true
			}
			ggufLayers = append(ggufLayers, desc)
			meta.Vision = true
		}
		// LICENSE: a doc-type layer per spec.md's own example of what
		// application/vnd.cncf.model.doc.v1.raw is for.
		if lic, ok := selectLicenseFile(files); ok {
			desc, blobChanged, err := streamHFFileToRegistry(
				ctx, dlClient, pusher, endpoint, owner, repo, commit, token,
				lic, modelspec.MediaTypeModelDocRaw,
			)
			if err != nil {
				return false, err
			}
			if blobChanged {
				changed = true
			}
			ggufLayers = append(ggufLayers, desc)
		}
		if err := pushCNCFGGUFManifest(ctx, pusher, meta, owner+"/"+repo, filepathAnnotation, ggufLayers, &changed); err != nil {
			return false, err
		}
		if changed {
			fmt.Fprintln(os.Stderr, "Writing manifest to image destination")
		}
		return changed, nil
	}

	meta.Format = "safetensors"
	toSend := selectDownloadableHFFiles(files)
	if len(toSend) == 0 {
		return false, fmt.Errorf("no model files found in repository %s/%s", owner, repo)
	}
	var layers []ocispec.Descriptor
	for _, f := range toSend {
		desc, blobChanged, err := streamHFFileToRegistry(
			ctx, dlClient, pusher, endpoint, owner, repo, commit, token,
			f, safetensorsMediaType(f.Path),
		)
		if err != nil {
			return false, fmt.Errorf("transfer %s: %w", f.Path, err)
		}
		if blobChanged {
			changed = true
		}
		desc.Annotations = map[string]string{modelspec.AnnotationFilepath: f.Path}
		layers = append(layers, desc)
	}
	if err := pushCNCFMultiManifest(ctx, pusher, meta, owner+"/"+repo, layers, &changed); err != nil {
		return false, err
	}
	if changed {
		fmt.Fprintln(os.Stderr, "Writing manifest to image destination")
	}
	return changed, nil
}

// streamHFFileToRegistry transfers one HuggingFace file directly into the
// registry pusher. When the file's real content digest can be learned
// ahead of time via a HEAD request (true for essentially every real
// LFS-tracked weight file — see hfHeadMetadata), the GET response body is
// piped straight into the push with no buffering at all: the file never
// touches local disk or is ever fully held in memory. Otherwise (small,
// non-LFS files such as config.json or a tokenizer file, where the ETag is
// a git blob sha1, not a sha256 of the content) it's buffered in memory —
// still zero disk I/O, and harmless given how small these files are.
func streamHFFileToRegistry(
	ctx context.Context,
	client *http.Client,
	pusher remotes.Pusher,
	endpoint, owner, repo, commit, token string,
	file hfFile,
	mediaType string,
) (ocispec.Descriptor, bool, error) {
	url := endpoint + owner + "/" + repo + "/resolve/" + commit + "/" + file.Path
	// org.cncf.model.filepath on the *layer* descriptor itself (not just
	// the manifest) is what cmd::serve's layer_filepath/is_gguf_layer
	// actually look at to recognize a servable GGUF/safetensors layer —
	// see downloadAttempt in hf.go, which sets the same annotation for
	// `llmman pull`'s local-layout path. Omitting it here doesn't fail
	// the transfer itself (the push succeeds either way), but leaves the
	// pushed image unservable by `llmman run`/`llmman serve` afterwards.
	annotations := map[string]string{modelspec.AnnotationFilepath: filepath.Base(file.Path)}
	label := filepath.Base(file.Path)

	dgst, size, digestOK, headErr := hfHeadMetadata(ctx, client, url, token)
	if headErr == nil && digestOK {
		desc := ocispec.Descriptor{MediaType: mediaType, Digest: dgst, Size: size, Annotations: annotations}
		short := shortDigest(dgst)
		var alreadyExists bool
		// retryStream: a network blip partway through a multi-gigabyte
		// weight file — very plausible for the kind of file this path
		// exists for — otherwise kills this file's transfer outright and
		// (before this) the whole `llmman transfer` invocation with it.
		// This restarts the failed attempt from byte zero rather than
		// resuming it: containerd's docker Pusher has no chunked/resumable
		// upload support (see backend_docker.go's package doc and
		// pushStreamLazy), so there's no partial destination state to
		// resume into. Still meaningfully better than the prior
		// single-shot behavior — see the comment on retryStream itself.
		err := retryStream(ctx, label, isHTTP4xx, func() error {
			// One progress pool per attempt: a bar can't be rewound back
			// to zero for a retry (mpb has no supported way to do that
			// once a bar has a definite total — see pushLazy's own
			// comment on this), so each retry gets its own pool/bar
			// instead of trying to reuse one across attempts.
			prog := newProgressPool(40)
			// "Transferring": the GET from HuggingFace and the push to
			// the destination happen simultaneously here — streamed
			// straight through with nothing landing on local disk in
			// between (see pushStreamLazy) — unlike the buffered
			// small-file fallback below, which really is
			// download-then-push, but still reports "Transferring" for
			// its (download-phase-only) bar to keep `llmman transfer`'s
			// progress output consistent end to end.
			newBar := func() *mpb.Bar {
				return addLayerBar(prog, "Transferring blob "+short, "Transferred  blob "+short, size, "")
			}
			exists, err := streamHFGet(ctx, client, url, token, pusher, desc, newBar)
			prog.Wait()
			if err != nil {
				return err
			}
			alreadyExists = exists
			return nil
		})
		if err != nil {
			return ocispec.Descriptor{}, false, fmt.Errorf("stream %s: %w", file.Path, err)
		}
		if alreadyExists {
			fmt.Fprintf(os.Stderr, "Transferred  blob %s (already present)\n", short)
		}
		return desc, !alreadyExists, nil
	}

	// Fallback: buffer a small, non-LFS file in memory. Its digest isn't
	// known until after the download, so the bar is labeled by filename
	// instead of digest (still shows real byte progress whenever the HEAD
	// above did manage to learn a size, even though its digest wasn't
	// usable — a HEAD failure (headErr != nil) leaves size at its zero
	// value, so the bar just falls back to a spinner in that case). These
	// files are small enough (config.json, tokenizer files, ...) that a
	// full-file retry is cheap regardless.
	var data []byte
	err := retryStream(ctx, label, isHTTP4xx, func() error {
		prog := newProgressPool(40)
		bar := addLayerBar(prog, "Transferring blob "+label, "Transferred  blob "+label, size, "")
		d, err := hfGetBytes(ctx, client, url, token, bar)
		if err != nil {
			bar.Abort(false)
			prog.Wait()
			return err
		}
		prog.Wait()
		data = d
		return nil
	})
	if err != nil {
		return ocispec.Descriptor{}, false, fmt.Errorf("download %s: %w", file.Path, err)
	}
	desc := ocispec.Descriptor{
		MediaType:   mediaType,
		Digest:      digest.FromBytes(data),
		Size:        int64(len(data)),
		Annotations: annotations,
	}
	alreadyExists, err := pushBytes(ctx, pusher, desc, data)
	if err != nil {
		return ocispec.Descriptor{}, false, fmt.Errorf("push %s: %w", file.Path, err)
	}
	return desc, !alreadyExists, nil
}

// streamHFGet pushes desc, only opening the GET against HuggingFace at all
// if the destination doesn't already have this exact blob (see
// pushStreamLazy) — the common case for re-transferring the same model, or
// transferring a file whose content happens to match one already pushed
// under a different name or tag. Runs under its own cancelable context so
// a stalled read (no bytes for dlStallTimeout — a connection that's gone
// dead without an actual TCP-level error, which a plain http.Client won't
// notice on its own) aborts this attempt instead of hanging indefinitely;
// the caller (streamHFFileToRegistry) is what retries a failed attempt
// from scratch via retryStream.
func streamHFGet(ctx context.Context, client *http.Client, url, token string, pusher remotes.Pusher, desc ocispec.Descriptor, newBar func() *mpb.Bar) (alreadyExists bool, err error) {
	attemptCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	return pushStreamLazy(attemptCtx, pusher, desc, newBar, func() (io.ReadCloser, error) {
		req, err := http.NewRequestWithContext(attemptCtx, "GET", url, nil)
		if err != nil {
			return nil, err
		}
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		// See downloadAttempt's identical header in hf.go: some HF CDNs
		// 400 a full-object GET with no Range at all past a few tens of
		// GB, which isHTTP4xx treats as permanent — every retry would
		// just repeat the same failure otherwise.
		req.Header.Set("Range", "bytes=0-")
		resp, err := client.Do(req)
		if err != nil {
			return nil, err
		}
		if resp.StatusCode != 200 && resp.StatusCode != 206 {
			resp.Body.Close()
			return nil, newHTTPStatusError("GET "+url, resp)
		}
		return newStallReadCloser(resp.Body, dlStallTimeout, cancel), nil
	})
}

func hfGetBytes(ctx context.Context, client *http.Client, url, token string, bar *mpb.Bar) ([]byte, error) {
	attemptCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	req, err := http.NewRequestWithContext(attemptCtx, "GET", url, nil)
	if err != nil {
		return nil, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	// See streamHFGet's identical header — simpler to always send this
	// than to branch by file size.
	req.Header.Set("Range", "bytes=0-")
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 && resp.StatusCode != 206 {
		return nil, newHTTPStatusError("GET "+url, resp)
	}
	sr := newStallReadCloser(resp.Body, dlStallTimeout, cancel)
	defer sr.Close()
	r := proxyOrNop(bar, sr, "")
	defer r.Close()
	return io.ReadAll(r)
}

// ---------------------------------------------------------------------------
// CNCF ModelPack manifest construction — pushed directly, never written
// to local disk (mirrors storeGGUFAsOCI/storeSafetensorsAsOCI in hf.go,
// which do the local-layout equivalent for `llmman pull`).
// ---------------------------------------------------------------------------

// pushCNCFGGUFManifest builds and pushes the CNCF manifest/config for a
// GGUF model — one layer for an ordinary single-file GGUF, or one layer
// per shard for a multi-part split (see selectGGUF/ggufShards in hf.go).
// filepathAnnotation sets the manifest-level org.cncf.model.filepath
// annotation; pass "" once there's more than one layer, matching
// storeGGUFAsOCI's own convention for the same tradeoff on the
// local-store path. If changed is non-nil, it's set to true whenever the
// config or manifest blob wasn't already present at the destination —
// see pusherBlobSink.
func pushCNCFGGUFManifest(ctx context.Context, pusher remotes.Pusher, meta modelMeta, modelRepo, filepathAnnotation string, layers []ocispec.Descriptor, changed *bool) error {
	_, err := buildCNCFManifest(pusherBlobSink(ctx, pusher, changed), meta, modelRepo, filepathAnnotation, layers)
	return err
}

// pushCNCFMultiManifest is pushCNCFGGUFManifest's safetensors equivalent.
func pushCNCFMultiManifest(ctx context.Context, pusher remotes.Pusher, meta modelMeta, modelRepo string, layers []ocispec.Descriptor, changed *bool) error {
	_, err := buildCNCFManifest(pusherBlobSink(ctx, pusher, changed), meta, modelRepo, "", layers)
	return err
}

// pusherBlobSink is the cncfBlobSink for pushing blobs directly to a
// registry pusher (see buildCNCFManifest in hf.go), mirroring
// layoutBlobSink's local-OCI-layout equivalent. If changed is non-nil,
// it's set to true whenever a blob this sink stores (the CNCF config or
// manifest JSON) wasn't already present at the destination by digest —
// letting callers tell whether a transfer actually pushed anything new,
// even when every weight-file layer turned out to already be present too.
func pusherBlobSink(ctx context.Context, pusher remotes.Pusher, changed *bool) cncfBlobSink {
	return func(mediaType string, data []byte) (ocispec.Descriptor, error) {
		desc := ocispec.Descriptor{
			MediaType: mediaType,
			Digest:    digest.FromBytes(data),
			Size:      int64(len(data)),
		}
		alreadyExists, err := pushBytes(ctx, pusher, desc, data)
		if err != nil {
			return ocispec.Descriptor{}, err
		}
		if !alreadyExists && changed != nil {
			*changed = true
		}
		return desc, nil
	}
}
