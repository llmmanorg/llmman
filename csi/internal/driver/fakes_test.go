package driver

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/container-storage-interface/spec/lib/go/csi"
)

// fakeMounter records mounts in memory. A bind mount of a directory at a
// target marks the target mounted; nothing is mounted for real.
type fakeMounter struct {
	mu        sync.Mutex
	mounted   map[string]string // target -> source (the topmost mount)
	stack     map[string]int    // target -> how many mounts sit on it
	binds     [][2]string
	unmounts  []string
	bindErr   error
	unmountEr error
	checkErr  error
}

func newFakeMounter() *fakeMounter {
	return &fakeMounter{mounted: map[string]string{}, stack: map[string]int{}}
}

func (m *fakeMounter) IsMountPoint(path string) (bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.checkErr != nil {
		return false, m.checkErr
	}
	return m.stack[path] > 0, nil
}

func (m *fakeMounter) MountReadOnly(source, target string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.bindErr != nil {
		return m.bindErr
	}
	m.binds = append(m.binds, [2]string{source, target})
	m.mounted[target] = source
	m.stack[target]++
	return nil
}

func (m *fakeMounter) Unmount(target string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.unmountEr != nil {
		return m.unmountEr
	}
	m.unmounts = append(m.unmounts, target)
	m.stack[target]--
	return nil
}

// fakeResolver answers every reference with a directory it made, counting
// calls. If gate is set, calls wait on it.
type fakeResolver struct {
	mu    sync.Mutex
	calls []string
	dir   string
	err   error
	gate  chan struct{}
}

func (r *fakeResolver) Resolve(ctx context.Context, reference string) (Tree, error) {
	r.mu.Lock()
	r.calls = append(r.calls, reference)
	r.mu.Unlock()
	if r.gate != nil {
		select {
		case <-r.gate:
		case <-ctx.Done():
			return Tree{}, ctx.Err()
		}
	}
	if r.err != nil {
		return Tree{}, r.err
	}
	return Tree{Dir: r.dir, Digest: "sha256:abc", Format: "safetensors"}, nil
}

func (r *fakeResolver) callCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.calls)
}

type fixture struct {
	d        *Driver
	mounter  *fakeMounter
	resolver *fakeResolver
	kubelet  string // a temp directory standing in for /var/lib/kubelet
	tree     string
}

// newFixture is a Driver wired to fakes, with real temp directories for the
// model tree and the kubelet's target paths.
func newFixture(t *testing.T, mutate ...func(*Config)) *fixture {
	t.Helper()
	root := t.TempDir()
	f := &fixture{
		mounter:  newFakeMounter(),
		kubelet:  filepath.Join(root, "kubelet"),
		tree:     filepath.Join(root, "cache", "abc"),
		resolver: &fakeResolver{},
	}
	for _, dir := range []string{f.kubelet, f.tree} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	f.resolver.dir = f.tree
	cfg := Config{
		NodeID:     "node-1",
		KubeletDir: f.kubelet,
		Logger:     slog.New(slog.NewTextHandler(io.Discard, nil)),
		Resolver:   f.resolver,
		Mounter:    f.mounter,
		LlmmanBin:  os.Args[0], // something that exists
	}
	for _, m := range mutate {
		m(&cfg)
	}
	d, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	f.d = d
	return f
}

func (f *fixture) target(name string) string {
	return filepath.Join(f.kubelet, "pods", name, "volumes", "kubernetes.io~csi", "model", "mount")
}

func mountCap() *csi.VolumeCapability {
	return &csi.VolumeCapability{
		AccessType: &csi.VolumeCapability_Mount{Mount: &csi.VolumeCapability_MountVolume{}},
		AccessMode: &csi.VolumeCapability_AccessMode{Mode: csi.VolumeCapability_AccessMode_MULTI_NODE_READER_ONLY},
	}
}

func publishReq(target string, ctxAttrs map[string]string) *csi.NodePublishVolumeRequest {
	return &csi.NodePublishVolumeRequest{
		VolumeId:         "csi-1234",
		TargetPath:       target,
		VolumeCapability: mountCap(),
		VolumeContext:    ctxAttrs,
		Readonly:         true,
	}
}

var errBoom = errors.New("boom")

// callers is how many calls have joined the running work for reference.
func (s *SingleFlight) callers(reference string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if f := s.inflight[reference]; f != nil {
		return f.callers
	}
	return 0
}
