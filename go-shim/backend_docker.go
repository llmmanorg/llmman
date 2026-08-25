//go:build !podman

package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/containerd/containerd/v2/core/content"
	"github.com/containerd/containerd/v2/core/remotes"
	"github.com/containerd/containerd/v2/core/remotes/docker"
	dockerconfig "github.com/containerd/containerd/v2/core/remotes/docker/config"
	remoteerrors "github.com/containerd/containerd/v2/core/remotes/errors"
	"github.com/containerd/errdefs"
	dockercliconfig "github.com/docker/cli/cli/config"
	"github.com/docker/cli/cli/config/credentials"
	clitypes "github.com/docker/cli/cli/config/types"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/vbauerster/mpb/v8"
	"golang.org/x/sync/errgroup"
)

// ---------------------------------------------------------------------------
// Credential helpers
// ---------------------------------------------------------------------------

// dockerCredentials looks up stored credentials for host. containerd's
// dockerAuthorizer calls this with the *actual connection host* it's
// talking to (see AddResponses in containerd/v2/core/remotes/docker/
// authorizer.go: `host := last.Request.URL.Host`) — which for Docker Hub
// is "registry-1.docker.io" (containerd/dockerconfig.ConfigureHosts
// rewrites "docker.io" to that for the connection itself, but the
// Credentials callback still sees the post-rewrite host). `llmman login`/
// `docker login` store credentials under "docker.io" (or the legacy
// "index.docker.io"/"https://index.docker.io/v1/" keys real `docker
// login` also writes), never under "registry-1.docker.io" — so without
// this normalization, every push/pull that reaches an authenticated Hub
// endpoint (bearer or basic) silently runs anonymously instead (a
// credential-store miss isn't an error here, by design, so nothing ever
// surfaced this beyond a confusing downstream "insufficient_scope" or
// 401 on push).
func dockerCredentials(host string) (string, string, error) {
	cfg := dockercliconfig.LoadDefaultConfigFile(io.Discard)
	for _, lookup := range dockerHubCredentialKeys(host) {
		store := cfg.GetCredentialsStore(lookup)
		creds, err := getCredentialsWithTimeout(store, lookup)
		if err != nil {
			continue // not found under this key — try the next one
		}
		if creds.Username == "" && creds.Password == "" && creds.IdentityToken == "" {
			continue
		}
		if creds.IdentityToken != "" {
			return "", creds.IdentityToken, nil
		}
		return creds.Username, creds.Password, nil
	}
	return "", "", nil // not an error — just not found under any key
}

// getCredentialsWithTimeout runs store.Get(lookup) with a hard deadline,
// timing out (rather than hanging this call — and so the pull/push it's
// blocking — forever) if it doesn't return in time.
//
// credentials.Store's own interface (see its doc comment) has no
// context/deadline parameter at all: when configFile.CredentialsStore
// (Docker's own config.json "credsStore" field) names a native OS
// credential helper, this call execs an external "docker-credential-
// <name>" *subprocess* (see docker-cli's NewNativeStore/
// client.NewShellProgramFunc), piping the lookup over stdin/stdout —
// with no timeout of its own either. If such a helper is configured but
// can't actually complete (e.g. a GUI-backed one — Docker Desktop's own
// credsStore, say — invoked on a headless machine with no session for it
// to reach), that subprocess call blocks forever, and so — since this is
// on containerd's own credential-lookup path, called synchronously
// before the actual registry request it's authenticating even goes out —
// does every pull/push through it, with nothing above this in the call
// stack able to time it out on its own. This was investigated as a
// candidate for an elusive Windows-only pull hang (see this repo's own
// git history) but ultimately ruled out — trace logging showed that hang
// occurring before this function, or even llmman_pull itself, ever ran.
// (That hang's actual cause turned out to be one layer further down
// than any Go code: see ffi.rs's ensure_runtime_init on the Rust side —
// Go's own c-archive auto-init constructor for Windows never runs under
// an MSVC-ABI target at all.) This timeout remains here purely as a
// real, independent hardening fix against a genuinely hanging
// credential helper on any platform, not a fix for that specific issue.
func getCredentialsWithTimeout(store credentials.Store, lookup string) (clitypes.AuthConfig, error) {
	type result struct {
		creds clitypes.AuthConfig
		err   error
	}
	ch := make(chan result, 1)
	go func() {
		creds, err := store.Get(lookup)
		ch <- result{creds, err}
	}()
	select {
	case r := <-ch:
		return r.creds, r.err
	case <-time.After(5 * time.Second):
		fmt.Fprintf(os.Stderr, "[llmman] credential store lookup for %q timed out after 5s — a native credential helper may be hung; continuing without stored credentials\n", lookup)
		return clitypes.AuthConfig{}, fmt.Errorf("credential store lookup for %q timed out", lookup)
	}
}

// dockerHubCredentialKeys returns every credential-store key that could
// plausibly hold Docker Hub credentials for a given connection host,
// broadest/most-canonical first. For any non-Hub host this is just the
// host itself, unchanged.
func dockerHubCredentialKeys(host string) []string {
	switch host {
	case "registry-1.docker.io", "index.docker.io", "docker.io", "https://index.docker.io/v1/":
		return []string{"docker.io", "index.docker.io", "https://index.docker.io/v1/", "registry-1.docker.io"}
	default:
		return []string{host}
	}
}

func newResolver(ctx context.Context) remotes.Resolver {
	return docker.NewResolver(docker.ResolverOptions{
		Hosts: dockerconfig.ConfigureHosts(ctx, dockerconfig.HostOptions{
			Credentials: dockerCredentials,
		}),
		Client: &http.Client{Timeout: 120 * time.Second},
	})
}

// describeErr enriches a containerd registry error with the response body,
// when there is one — containerd's own ErrUnexpectedStatus.Error() deliberately
// omits it (only logged at debug level), which is exactly the detail needed to
// tell "repository doesn't exist", "insufficient scope", and similar registry-
// side rejections apart from each other instead of a bare, unexplained status
// code.
func describeErr(err error) error {
	var ue remoteerrors.ErrUnexpectedStatus
	if errors.As(err, &ue) && len(ue.Body) > 0 {
		return fmt.Errorf("%w: %s", err, strings.TrimSpace(string(ue.Body)))
	}
	return err
}

// ---------------------------------------------------------------------------
// ociProvider implements content.Provider backed by an OCI layout directory.
// ---------------------------------------------------------------------------

type ociProvider struct{ dir string }

func (p *ociProvider) ReaderAt(ctx context.Context, desc ocispec.Descriptor) (content.ReaderAt, error) {
	path := blobPath(p.dir, desc.Digest)
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("blob %s: %w", desc.Digest, err)
	}
	fi, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	return &fileReaderAt{f: f, size: fi.Size()}, nil
}

type fileReaderAt struct {
	f    *os.File
	size int64
}

func (r *fileReaderAt) ReadAt(p []byte, off int64) (int, error) { return r.f.ReadAt(p, off) }
func (r *fileReaderAt) Close() error                            { return r.f.Close() }
func (r *fileReaderAt) Size() int64                             { return r.size }

// pushLazy is the one place that actually talks to pusher.Push. It checks
// whether the destination already has desc *before* calling open — open
// returns the content to upload plus a cleanup func (call it even on a nil
// reader/error, may be nil itself) — and open is only ever invoked once
// that check has confirmed an upload is actually needed. Returns whether
// the blob already existed, so callers can print their own "already
// present" line instead of ever creating a progress bar for it.
//
// Checking existence first, unconditionally, matters for two different
// reasons depending on the caller:
//   - It avoids ever opening a (potentially multi-gigabyte) source reader
//     for content that's just going to be thrown away unread — mattering
//     beyond bandwidth, since leaving a large HTTP response body unread
//     and then closing it can itself take as long as reading it would
//     have (the transport may drain it to keep the connection reusable),
//     which otherwise looks exactly like an unexplained hang.
//   - It means a progress bar is only ever created for a blob that's
//     actually going to be incremented. mpb's (*Bar).SetTotal is
//     documented as a no-op for any bar constructed with a definite
//     (>0) total — which every bar here is, since every desc.Size is
//     already known up front — so there is no supported way to
//     retroactively mark such a bar "already done" after creating it.
//     Doing so anyway (an earlier version of this code did) silently
//     leaves that bar incomplete forever, and mpb's pool.Wait() blocks
//     forever waiting for every bar it knows about to finish — hanging
//     the whole transfer with no error and no output to explain why.
func pushLazy(
	ctx context.Context,
	pusher remotes.Pusher,
	desc ocispec.Descriptor,
	open func() (r io.Reader, bar *mpb.Bar, cleanup func(), err error),
) (alreadyExists bool, err error) {
	// Owns its own cancelable derivative of ctx so a stalled write (see
	// stallWriter) can abort this attempt without needing every caller
	// to plumb a cancel func down to here itself — canceling it only
	// ever affects this one push attempt, and it composes fine with a
	// caller's own cancelable ctx (e.g. streamHFGet's attemptCtx, for its
	// read side): cancellation flows one way, parent to child, so either
	// one firing aborts this attempt, and firing this one can't reach
	// back up to affect the caller's.
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	cw, err := pusher.Push(ctx, desc)
	if err != nil {
		if errdefs.IsAlreadyExists(err) {
			return true, nil
		}
		return false, describeErr(err)
	}
	defer cw.Close()

	r, bar, cleanup, err := open()
	if cleanup != nil {
		defer cleanup()
	}
	if err != nil {
		return false, err
	}

	// See stallWriter's own comment (shared_oci.go) for why this exists:
	// content.Copy below has no way on its own to notice or escape a
	// destination write that's stopped making progress — observed in
	// practice as a registry PUT that goes quiet mid-upload and simply
	// never completes or fails, hanging the whole transfer (and, per
	// transfer.yml's job timeout, the whole CI job) for hours rather
	// than the few seconds it'd otherwise take retryStream to notice and
	// retry this file from scratch.
	sw := newStallWriter(cw, dlStallTimeout, cancel)
	defer sw.stop()

	// containerd's pushWriter (core/remotes/docker/pusher.go) only ever
	// initializes its underlying pipe inside its own Write method, the
	// first time that's called with actual data — but content.Copy below
	// never calls Write at all for a zero-byte blob (its io.ReadAtLeast
	// loop immediately gets 0, io.EOF, so the "if nr > 0" guard around
	// the Write call is never entered), leaving that pipe nil. Commit
	// then unconditionally does pw.pipe.Write(...) to check for a prior
	// read error, which panics on that nil pipe rather than returning
	// one — see docker/llmman-publisher's nemotron-3-nano:4b-safetensors
	// transfer, which crashed exactly this way pushing a real, ordinary
	// zero-byte file from that HuggingFace repository. A zero-length
	// write of our own first is a genuine no-op otherwise, and is enough
	// to make Write actually run once, initializing that pipe before
	// Commit ever gets to it.
	if desc.Size == 0 {
		if _, err := sw.Write(nil); err != nil {
			return false, describeErr(err)
		}
	}

	if copyErr := describeErr(content.Copy(ctx, sw, r, desc.Size, desc.Digest)); copyErr != nil {
		// A real failure partway through: the bar (if any) was already
		// incremented some amount short of its total, and never will be
		// any further — abort it explicitly so it doesn't likewise leave
		// pool.Wait() hanging on a bar that's now never going anywhere.
		if bar != nil {
			bar.Abort(false)
		}
		return false, copyErr
	}
	return false, nil
}

// withBar wraps r in newBar's progress bar (if newBar is non-nil, via the
// same proxyOrNop every other progress-reporting download/upload path in
// this package uses — see shared_oci.go), and returns that bar (so
// pushLazy can abort it on a copy failure) plus a cleanup func that closes
// both the proxy reader and r itself (if r is an io.Closer) — for use as
// pushLazy's open callback.
func withBar(r io.Reader, newBar func() *mpb.Bar, progressKey string) (io.Reader, *mpb.Bar, func()) {
	var bar *mpb.Bar
	closers := []io.Closer{}
	if rc, ok := r.(io.Closer); ok {
		closers = append(closers, rc)
	}
	if newBar != nil {
		bar = newBar()
		proxyRC := proxyOrNop(bar, r, progressKey)
		r = proxyRC
		closers = append(closers, proxyRC)
	}
	return r, bar, func() {
		for _, c := range closers {
			c.Close()
		}
	}
}

// pushBlob pushes a single blob from the OCI layout to the registry
// pusher, reporting progress via newBar — called, and its resulting bar
// wrapped around the read, only if the blob isn't already at the
// destination (see pushLazy). Pass nil for no progress reporting.
func pushBlob(ctx context.Context, pusher remotes.Pusher, provider *ociProvider, desc ocispec.Descriptor, newBar func() *mpb.Bar, progressKey string) (alreadyExists bool, err error) {
	return pushLazy(ctx, pusher, desc, func() (io.Reader, *mpb.Bar, func(), error) {
		ra, err := provider.ReaderAt(ctx, desc)
		if err != nil {
			return nil, nil, nil, err
		}
		r, bar, cleanup := withBar(io.NewSectionReader(ra, 0, ra.Size()), newBar, progressKey)
		return r, bar, func() { cleanup(); ra.Close() }, nil
	})
}

// pushBytes pushes an in-memory blob (a manifest or a small config/metadata
// file) directly to the registry pusher — no local file involved. Returns
// whether the destination already had this exact blob (by digest), so
// callers can tell whether anything actually changed.
func pushBytes(ctx context.Context, pusher remotes.Pusher, desc ocispec.Descriptor, data []byte) (alreadyExists bool, err error) {
	return pushLazy(ctx, pusher, desc, func() (io.Reader, *mpb.Bar, func(), error) {
		return bytes.NewReader(data), nil, nil, nil
	})
}

// pushStreamLazy pushes a blob whose digest and size are already known
// (see hfHeadMetadata) from a source opened lazily by openSource — called,
// and its resulting reader wrapped in a progress bar via newBar, only if
// the blob isn't already at the destination (see pushLazy). This is what
// lets `llmman transfer` stream large HuggingFace files straight through:
// bytes flow source → destination without ever landing on disk (or
// getting downloaded at all, if the destination turns out to already have
// them) in between. Pass a nil newBar for no progress reporting.
func pushStreamLazy(ctx context.Context, pusher remotes.Pusher, desc ocispec.Descriptor, newBar func() *mpb.Bar, openSource func() (io.ReadCloser, error)) (alreadyExists bool, err error) {
	return pushLazy(ctx, pusher, desc, func() (io.Reader, *mpb.Bar, func(), error) {
		rc, err := openSource()
		if err != nil {
			return nil, nil, nil, err
		}
		// pushStreamLazy is only used by llmman transfer (streamed
		// HuggingFace/OCI to a registry, never through the daemon's
		// per-model-keyed progress poll — see progress_state.go), so
		// there's no meaningful key to credit these bytes to.
		r, bar, cleanup := withBar(rc, newBar, "")
		return r, bar, cleanup, nil
	})
}

// ---------------------------------------------------------------------------
// Exported CGO functions
// ---------------------------------------------------------------------------

// llmman_login stores credentials for a registry in the Docker credential store.
//
//export llmman_login
func llmman_login(cServer, cUsername, cPassword *C.char) *C.char {
	server := C.GoString(cServer)
	username := C.GoString(cUsername)
	password := C.GoString(cPassword)

	cfg := dockercliconfig.LoadDefaultConfigFile(io.Discard)
	store := cfg.GetCredentialsStore(server)

	if err := store.Store(clitypes.AuthConfig{
		ServerAddress: server,
		Username:      username,
		Password:      password,
	}); err != nil {
		return errResp(fmt.Errorf("store credentials: %w", err))
	}
	if err := cfg.Save(); err != nil {
		return errResp(fmt.Errorf("save config: %w", err))
	}
	return okResp("")
}

// llmman_logout removes credentials for a registry from the Docker credential store.
//
//export llmman_logout
func llmman_logout(cServer *C.char) *C.char {
	server := C.GoString(cServer)

	cfg := dockercliconfig.LoadDefaultConfigFile(io.Discard)
	store := cfg.GetCredentialsStore(server)
	if err := store.Erase(server); err != nil {
		return errResp(fmt.Errorf("erase credentials: %w", err))
	}
	if err := cfg.Save(); err != nil {
		return errResp(fmt.Errorf("save config: %w", err))
	}
	return okResp("")
}

// llmman_push pushes an image from a local OCI layout directory to a registry.
// layoutDir is the path to the OCI layout root; ref is the full registry reference.
//
//export llmman_push
func llmman_push(cLayoutDir, cRef *C.char) *C.char {
	ref := C.GoString(cRef)
	progressReset(ref, "retrieving manifest")
	defer progressDone(ref)
	if _, err := pushToRegistry(context.Background(), C.GoString(cLayoutDir), ref, findManifestForPush); err != nil {
		return errResp(err)
	}
	return okResp("")
}

// pushToRegistry is llmman_push's implementation, factored out so
// llmman_transfer's staging-directory fallback (transferViaStaging, in
// transfer_common.go) can reuse it without going through CGO. find
// resolves ref to its local manifest — findManifestForPush (exact match
// only) for a direct user push, or findManifestForTransfer (also allows
// a single-entry fallback, safe only for a staging directory known to
// hold exactly the one just-pulled model) for transferViaStaging.
// Returns whether anything was actually pushed — false if every layer,
// the config, and the manifest were all already present at the
// destination by digest.
func pushToRegistry(ctx context.Context, layoutDir, ref string, find func(string, string) (ocispec.Descriptor, error)) (changed bool, err error) {
	manifestDesc, err := find(layoutDir, ref)
	if err != nil {
		return false, err
	}

	// Read manifest
	manifestData, err := readBlob(layoutDir, manifestDesc.Digest)
	if err != nil {
		return false, fmt.Errorf("read manifest blob: %w", err)
	}
	var manifest ocispec.Manifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		return false, fmt.Errorf("parse manifest: %w", err)
	}

	resolver := newResolver(ctx)
	// normalizeTag: a tagless ref pushes the manifest addressable only by
	// digest, with no tag ever created — silently, since containerd has
	// no opinion on what a missing tag should default to. See
	// transfer_docker.go's dockerTransfer for the same fix applied there.
	pusher, err := resolver.Pusher(ctx, normalizeTag(ref))
	if err != nil {
		return false, fmt.Errorf("create pusher: %w", err)
	}
	provider := &ociProvider{dir: layoutDir}

	// "Copying blob/config <digest>" progress bars, matching the familiar
	// registry-copy progress-bar wording (see copy/progress_bars.go
	// upstream).
	prog := newProgressPool(40)
	changed = false
	pushWithBar := func(desc ocispec.Descriptor, kind string) error {
		short := shortDigest(desc.Digest)
		newBar := func() *mpb.Bar {
			return addLayerBar(prog, "Copying "+kind+" "+short, "Copied  "+kind+" "+short, desc.Size, ref)
		}
		alreadyExists, err := pushBlob(ctx, pusher, provider, desc, newBar, ref)
		if err != nil {
			return err
		}
		if alreadyExists {
			fmt.Fprintf(os.Stderr, "Copied  %s %s (already present)\n", kind, short)
		} else {
			changed = true
		}
		return nil
	}

	// Push layers
	progressSetStatus(ref, "pushing")
	for _, layer := range manifest.Layers {
		if err := pushWithBar(layer, "blob"); err != nil {
			prog.Wait()
			return false, fmt.Errorf("push layer %s: %w", layer.Digest, err)
		}
	}
	// Push config
	if err := pushWithBar(manifest.Config, "config"); err != nil {
		prog.Wait()
		return false, fmt.Errorf("push config: %w", err)
	}
	prog.Wait()

	// Push manifest — no progress bar (a few hundred bytes of JSON) —
	// just a plain "Writing manifest to image destination" message
	// instead of a bar for this step.
	manifestAlreadyExists, err := pushBlob(ctx, pusher, provider, manifestDesc, nil, "")
	if err != nil {
		return false, fmt.Errorf("push manifest: %w", err)
	}
	if !manifestAlreadyExists {
		changed = true
	}
	if changed {
		fmt.Fprintln(os.Stderr, "Writing manifest to image destination")
	}
	return changed, nil
}

// llmman_pull pulls an image from a registry into a local OCI layout directory.
//
//export llmman_pull
func llmman_pull(cRef, cLayoutDir *C.char) *C.char {
	ref := C.GoString(cRef)
	progressReset(ref, "pulling manifest")
	defer progressDone(ref)
	if err := pullToLayout(context.Background(), ref, C.GoString(cLayoutDir)); err != nil {
		return errResp(err)
	}
	return okResp("")
}

// pullToLayout is llmman_pull's implementation, factored out so
// llmman_transfer's staging-directory fallback can reuse it.
func pullToLayout(ctx context.Context, ref, layoutDir string) error {
	// progressKey is the exact ref llmman_pull was originally called with
	// (and the exact string the Rust daemon polls llmman_progress with —
	// see progress_state.go) — captured before classifyPullRef below
	// potentially normalizes ref itself (e.g. defaulting in ":latest"),
	// so progress tracking always uses the same key regardless of that
	// normalization.
	progressKey := ref
	ref, isOCI, handled, err := classifyPullRef(ctx, ref, layoutDir)
	if handled {
		return err
	}
	if !isOCI {
		return pullHF(ctx, ref, layoutDir, progressKey)
	}

	if err := ensureLayout(layoutDir); err != nil {
		return fmt.Errorf("init OCI layout: %w", err)
	}

	resolver := newResolver(ctx)
	// Deliberately not "resolve %s: %w" — containerd's own resolve errors
	// (e.g. errdefs.ErrNotFound) already embed ref themselves, and every
	// caller of llmman_pull (the Rust daemon's /api/pull handler) already
	// prefixes whatever error comes back with the reference it asked for.
	// Including ref here too just repeats it two or three times over.
	name, manifestDesc, err := resolver.Resolve(ctx, ref)
	if err != nil {
		return fmt.Errorf("resolve: %w", err)
	}
	fetcher, err := resolver.Fetcher(ctx, name)
	if err != nil {
		return fmt.Errorf("create fetcher: %w", err)
	}

	// Fetch and store manifest
	rc, err := fetcher.Fetch(ctx, manifestDesc)
	if err != nil {
		return fmt.Errorf("fetch manifest: %w", err)
	}
	manifestData, err := io.ReadAll(rc)
	rc.Close()
	if err != nil {
		return fmt.Errorf("read manifest: %w", err)
	}
	if _, err := writeBlob(layoutDir, manifestDesc.MediaType, manifestData); err != nil {
		return fmt.Errorf("write manifest blob: %w", err)
	}

	// Decode manifest to learn about layers and config.
	var manifest ocispec.Manifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		return fmt.Errorf("parse manifest: %w", err)
	}
	if manifest.Config.Digest == "" {
		// A valid image index unmarshals above without error (unknown
		// fields are ignored), just with a zero-value Config — store the
		// descriptor as-is rather than fetching a zero-digest "config".
		return writeManifestRef(layoutDir, ref, manifestDesc)
	}

	// Fetch config
	configRC, err := fetcher.Fetch(ctx, manifest.Config)
	if err != nil {
		return fmt.Errorf("fetch config: %w", err)
	}
	configData, readErr := io.ReadAll(configRC)
	configRC.Close()
	if readErr != nil {
		return fmt.Errorf("read config: %w", readErr)
	}
	if _, err := writeBlob(layoutDir, manifest.Config.MediaType, configData); err != nil {
		return fmt.Errorf("write config blob: %w", err)
	}

	// Fetch layers in parallel — up to 6 concurrent downloads, matching podman's
	// default maxParallelDownloads.  All bars share one mpb.Progress; OnComplete
	// decorators flip each bar to "Pulled   <digest>" when done so the final static
	// line is always correct regardless of render-tick timing.
	const maxParallel = 6
	progressSetStatus(progressKey, "pulling")
	prog := mpb.New(
		mpb.WithWidth(80),
		mpb.WithOutput(os.Stderr),
		mpb.WithRefreshRate(180*time.Millisecond),
	)
	sem := make(chan struct{}, maxParallel)
	g, gctx := errgroup.WithContext(ctx)
	var barMu sync.Mutex // serialise bar creation so order matches layer order
	for _, layer := range manifest.Layers {
		layer := layer // capture
		shortDigest := layer.Digest.Hex()
		if len(shortDigest) > 12 {
			shortDigest = shortDigest[:12]
		}
		if blobExists(layoutDir, layer) {
			fmt.Fprintf(prog, "Cached   %s\n", shortDigest)
			continue
		}
		// Create the bar before launching the goroutine so bars appear in
		// manifest order even when downloads finish out of order.
		barMu.Lock()
		bar := addLayerBar(prog, "Pulling  "+shortDigest, "Pulled   "+shortDigest, layer.Size, progressKey)
		barMu.Unlock()
		sem <- struct{}{}
		g.Go(func() error {
			defer func() { <-sem }()
			// Deduplicate against any other pull in this process (a
			// different model, running concurrently — see
			// blobFetchGroup's own doc comment) that's fetching this
			// exact same blob digest right now, rather than racing it
			// to append to the same deterministic .part file.
			_, err := dedupBlobFetch(layer.Digest.String(), progressKey, layer.Size, func() (ocispec.Descriptor, error) {
				layerRC, err := fetcher.Fetch(gctx, layer)
				if err != nil {
					return ocispec.Descriptor{}, fmt.Errorf("fetch layer %s: %w", layer.Digest, err)
				}
				// Resume from an existing partial download: seek the HTTP reader to
				// the already-downloaded offset (containerd's httpReadSeeker issues a
				// Range: bytes=N- request, or discards N bytes if the server doesn't
				// support range requests) and pre-fill the progress bar.
				partOffset := int64(0)
				partPath := blobPath(layoutDir, layer.Digest) + ".part"
				if fi, statErr := os.Stat(partPath); statErr == nil && fi.Size() > 0 {
					if seeker, ok := layerRC.(io.ReadSeeker); ok {
						if _, seekErr := seeker.Seek(fi.Size(), io.SeekStart); seekErr == nil {
							partOffset = fi.Size()
							bar.IncrInt64(partOffset)
							progressAddCompleted(progressKey, partOffset)
						}
					}
				}
				proxyRC := proxyOrNop(bar, layerRC, progressKey)
				desc, writeErr := writeBlobStream(layoutDir, layer.MediaType, proxyRC, layer.Size, layer.Digest, partOffset)
				proxyRC.Close()
				if writeErr != nil {
					return ocispec.Descriptor{}, fmt.Errorf("write layer %s: %w", layer.Digest, writeErr)
				}
				return desc, nil
			})
			if err != nil {
				bar.Abort(false)
				return err
			}
			// Whether this goroutine did the actual fetch or a concurrent
			// pull of a different model got there first (see
			// dedupBlobFetch), the blob is now on disk — force this bar
			// to 100% so mpb's pool.Wait() below doesn't hang waiting on
			// a bar this goroutine never itself incremented.
			bar.SetTotal(layer.Size, true)
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		prog.Wait()
		return err
	}
	prog.Wait()

	return writeManifestRef(layoutDir, ref, manifestDesc)
}

// llmman_inspect fetches and returns the raw manifest JSON for a remote reference.
//
//export llmman_inspect
func llmman_inspect(cRef *C.char) *C.char {
	ref := C.GoString(cRef)
	ctx := context.Background()

	resolver := newResolver(ctx)
	name, manifestDesc, err := resolver.Resolve(ctx, ref)
	if err != nil {
		return errResp(fmt.Errorf("resolve: %w", err))
	}
	fetcher, err := resolver.Fetcher(ctx, name)
	if err != nil {
		return errResp(fmt.Errorf("create fetcher: %w", err))
	}
	rc, err := fetcher.Fetch(ctx, manifestDesc)
	if err != nil {
		return errResp(fmt.Errorf("fetch manifest: %w", err))
	}
	data, err := io.ReadAll(rc)
	rc.Close()
	if err != nil {
		return errResp(fmt.Errorf("read manifest: %w", err))
	}

	// Pretty-print
	var buf bytes.Buffer
	if err := json.Indent(&buf, data, "", "  "); err != nil {
		return okResp(string(data))
	}
	return okResp(buf.String())
}

// llmman_transfer transfers an image directly from source to destination,
// without ever writing it to the persistent local store. See
// transfer_docker.go for the three strategies this picks between
// (streamed OCI→OCI, streamed HuggingFace→OCI, and a staging-directory
// fallback for everything else) and why each exists.
//
//export llmman_transfer
func llmman_transfer(cSource, cDestination *C.char) *C.char {
	changed, err := dockerTransfer(context.Background(), C.GoString(cSource), C.GoString(cDestination))
	if err != nil {
		return errResp(err)
	}
	// data carries whether anything was actually pushed, so the Rust CLI
	// layer (cmd::transfer) can report "already up to date" instead of
	// "Transferred" when re-running a transfer for content that hasn't
	// changed since the last one — see transferStatusChanged/Unchanged.
	if changed {
		return okResp(transferStatusChanged)
	}
	return okResp(transferStatusUnchanged)
}
