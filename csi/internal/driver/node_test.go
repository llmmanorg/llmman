package driver

import (
	"context"
	"fmt"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const ref = "registry.example.com/models/qwen3-0.6b:latest"

var modelpackAttrs = map[string]string{"model.csi.modelpack.org/reference": ref}

func wantCode(t *testing.T, err error, want codes.Code) {
	t.Helper()
	if got := status.Code(err); got != want {
		t.Fatalf("code = %v (%v), want %v", got, err, want)
	}
}

func TestPublishInlineEphemeralVolumeMountsTheModelDirReadOnly(t *testing.T) {
	f := newFixture(t)
	target := f.target("pod-a")
	// What kubelet sends for an inline volume with podInfoOnMount.
	attrs := map[string]string{
		"model.csi.modelpack.org/reference": ref,
		"csi.storage.k8s.io/ephemeral":      "true",
		"csi.storage.k8s.io/pod.name":       "vllm",
		"csi.storage.k8s.io/pod.namespace":  "default",
	}
	if _, err := f.d.NodePublishVolume(context.Background(), publishReq(target, attrs)); err != nil {
		t.Fatal(err)
	}
	if got := f.resolver.calls; len(got) != 1 || got[0] != ref {
		t.Fatalf("resolver calls = %v", got)
	}
	if got := f.mounter.binds; len(got) != 1 || got[0] != [2]string{f.tree, target} {
		t.Fatalf("binds = %v", got)
	}
	if st, err := os.Stat(target); err != nil || !st.IsDir() {
		t.Fatalf("target not created: %v", err)
	}
}

func TestPublishPreProvisionedVolumeUsesTheSameAttributes(t *testing.T) {
	// A PersistentVolume's volumeAttributes arrive as the volume context
	// too, without the ephemeral marker.
	f := newFixture(t)
	if _, err := f.d.NodePublishVolume(context.Background(), publishReq(f.target("pod-b"), modelpackAttrs)); err != nil {
		t.Fatal(err)
	}
	if len(f.mounter.binds) != 1 {
		t.Fatalf("binds = %v", f.mounter.binds)
	}
}

func TestPublishIsIdempotent(t *testing.T) {
	f := newFixture(t)
	target := f.target("pod-a")
	for i := 0; i < 3; i++ {
		if _, err := f.d.NodePublishVolume(context.Background(), publishReq(target, modelpackAttrs)); err != nil {
			t.Fatal(err)
		}
	}
	if len(f.mounter.binds) != 1 || f.resolver.callCount() != 1 {
		t.Fatalf("binds = %v, resolves = %d; want one of each", f.mounter.binds, f.resolver.callCount())
	}
}

func TestPublishAcceptsTheDriversOwnAttributePrefix(t *testing.T) {
	f := newFixture(t, func(c *Config) { c.Name = "llmman.csi.llmman.org" })
	attrs := map[string]string{"llmman.csi.llmman.org/reference": ref}
	if _, err := f.d.NodePublishVolume(context.Background(), publishReq(f.target("a"), attrs)); err != nil {
		t.Fatal(err)
	}
	// ...and the modelpack one still, so manifests written for that
	// driver work under any name.
	if _, err := f.d.NodePublishVolume(context.Background(), publishReq(f.target("b"), modelpackAttrs)); err != nil {
		t.Fatal(err)
	}
	if f.resolver.callCount() != 2 {
		t.Fatalf("resolves = %d", f.resolver.callCount())
	}
}

func TestPublishRejectsBadRequests(t *testing.T) {
	f := newFixture(t)
	good := f.target("pod-a")
	cases := map[string]struct {
		mutate func(*csi.NodePublishVolumeRequest)
	}{
		"no volume id":  {func(r *csi.NodePublishVolumeRequest) { r.VolumeId = "" }},
		"no capability": {func(r *csi.NodePublishVolumeRequest) { r.VolumeCapability = nil }},
		"block volume": {func(r *csi.NodePublishVolumeRequest) {
			r.VolumeCapability = &csi.VolumeCapability{AccessType: &csi.VolumeCapability_Block{Block: &csi.VolumeCapability_BlockVolume{}}}
		}},
		"no target":       {func(r *csi.NodePublishVolumeRequest) { r.TargetPath = "" }},
		"relative target": {func(r *csi.NodePublishVolumeRequest) { r.TargetPath = "pods/a/mount" }},
		"unclean target":  {func(r *csi.NodePublishVolumeRequest) { r.TargetPath = f.kubelet + "/pods/../../etc/x" }},
		"outside kubelet": {func(r *csi.NodePublishVolumeRequest) { r.TargetPath = "/etc/shadow" }},
		"the kubelet dir": {func(r *csi.NodePublishVolumeRequest) { r.TargetPath = f.kubelet }},
		"no reference":    {func(r *csi.NodePublishVolumeRequest) { r.VolumeContext = map[string]string{} }},
		"blank reference": {func(r *csi.NodePublishVolumeRequest) {
			r.VolumeContext = map[string]string{"model.csi.modelpack.org/reference": "  "}
		}},
		"option-like": {func(r *csi.NodePublishVolumeRequest) {
			r.VolumeContext = map[string]string{"model.csi.modelpack.org/reference": "--dir"}
		}},
		"local path": {func(r *csi.NodePublishVolumeRequest) {
			r.VolumeContext = map[string]string{"model.csi.modelpack.org/reference": "/etc/llmman-csi/docker"}
		}},
		"inner space": {func(r *csi.NodePublishVolumeRequest) {
			r.VolumeContext = map[string]string{"model.csi.modelpack.org/reference": "a b"}
		}},
		"newline": {func(r *csi.NodePublishVolumeRequest) {
			r.VolumeContext = map[string]string{"model.csi.modelpack.org/reference": "a\nb"}
		}},
		"exclude weights": {func(r *csi.NodePublishVolumeRequest) {
			r.VolumeContext["model.csi.modelpack.org/exclude-model-weights"] = "true"
		}},
		"exclude patterns": {func(r *csi.NodePublishVolumeRequest) {
			r.VolumeContext["model.csi.modelpack.org/exclude-file-patterns"] = `["*.bin"]`
		}},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			req := publishReq(good, map[string]string{"model.csi.modelpack.org/reference": ref})
			c.mutate(req)
			_, err := f.d.NodePublishVolume(context.Background(), req)
			wantCode(t, err, codes.InvalidArgument)
		})
	}
	if f.resolver.callCount() != 0 || len(f.mounter.binds) != 0 {
		t.Fatal("a rejected request pulled or mounted something")
	}
}

func TestPublishAcceptsUnsupportedAttributesAtTheirNoOpValues(t *testing.T) {
	f := newFixture(t)
	attrs := map[string]string{
		"model.csi.modelpack.org/reference":             ref,
		"model.csi.modelpack.org/exclude-model-weights": "false",
		"model.csi.modelpack.org/exclude-file-patterns": "[]",
	}
	if _, err := f.d.NodePublishVolume(context.Background(), publishReq(f.target("a"), attrs)); err != nil {
		t.Fatal(err)
	}
}

func TestPublishFailsCleanlyWhenThePullFails(t *testing.T) {
	f := newFixture(t)
	f.resolver.err = fmt.Errorf("manifest unknown: %w", errBoom)
	target := f.target("pod-a")
	_, err := f.d.NodePublishVolume(context.Background(), publishReq(target, modelpackAttrs))
	wantCode(t, err, codes.Internal)
	if want := "manifest unknown"; !contains(err.Error(), want) {
		t.Fatalf("error %q lacks the cause %q", err, want)
	}
	if len(f.mounter.binds) != 0 {
		t.Fatal("mounted after a failed pull")
	}
	if _, statErr := os.Stat(target); !os.IsNotExist(statErr) {
		t.Fatal("created the target for a pull that failed")
	}
}

func TestPublishReportsAMountFailure(t *testing.T) {
	f := newFixture(t)
	f.mounter.bindErr = errBoom
	_, err := f.d.NodePublishVolume(context.Background(), publishReq(f.target("a"), modelpackAttrs))
	wantCode(t, err, codes.Internal)
}

func TestPublishReportsAMountCheckFailure(t *testing.T) {
	f := newFixture(t)
	f.mounter.checkErr = errBoom
	_, err := f.d.NodePublishVolume(context.Background(), publishReq(f.target("a"), modelpackAttrs))
	wantCode(t, err, codes.Internal)
	if f.resolver.callCount() != 0 {
		t.Fatal("pulled without knowing whether it was mounted")
	}
}

func TestPublishOfTheSameTargetInFlightIsAborted(t *testing.T) {
	f := newFixture(t)
	f.resolver.gate = make(chan struct{})
	target := f.target("pod-a")
	first := make(chan error, 1)
	go func() {
		_, err := f.d.NodePublishVolume(context.Background(), publishReq(target, modelpackAttrs))
		first <- err
	}()
	waitFor(t, func() bool { return f.resolver.callCount() == 1 })

	_, err := f.d.NodePublishVolume(context.Background(), publishReq(target, modelpackAttrs))
	wantCode(t, err, codes.Aborted)
	_, err = f.d.NodeUnpublishVolume(context.Background(), &csi.NodeUnpublishVolumeRequest{VolumeId: "csi-1234", TargetPath: target})
	wantCode(t, err, codes.Aborted)

	close(f.resolver.gate)
	if err := <-first; err != nil {
		t.Fatal(err)
	}
}

func TestManyPodsPublishingTheSameReferencePullOnce(t *testing.T) {
	inner := &fakeResolver{gate: make(chan struct{})}
	f := newFixture(t)
	inner.dir = f.tree
	sf := NewSingleFlight(context.Background(), inner)
	f.d.cfg.Resolver = sf

	const pods = 25
	var wg sync.WaitGroup
	errs := make(chan error, pods)
	for i := 0; i < pods; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, err := f.d.NodePublishVolume(context.Background(), publishReq(f.target(fmt.Sprintf("pod-%d", i)), modelpackAttrs))
			errs <- err
		}(i)
	}
	waitFor(t, func() bool { return sf.callers(ref) == pods })
	close(inner.gate)
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if n := inner.callCount(); n != 1 {
		t.Fatalf("llmman ran %d times for one reference, want 1", n)
	}
	if n := len(f.mounter.binds); n != pods {
		t.Fatalf("mounted %d of %d pods", n, pods)
	}
}

func TestAKubeletThatGivesUpDoesNotRestartThePull(t *testing.T) {
	inner := &fakeResolver{gate: make(chan struct{})}
	f := newFixture(t)
	inner.dir = f.tree
	sf := NewSingleFlight(context.Background(), inner)
	f.d.cfg.Resolver = sf
	target := f.target("pod-a")

	// The kubelet's RPC deadline passes mid-pull.
	short, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_, err := f.d.NodePublishVolume(short, publishReq(target, modelpackAttrs))
	if code := status.Code(err); code != codes.DeadlineExceeded {
		t.Fatalf("code = %v (%v), want DeadlineExceeded", code, err)
	}
	if len(f.mounter.binds) != 0 {
		t.Fatal("mounted before the pull finished")
	}

	// It calls again; the pull that was running finishes and serves it.
	done := make(chan error, 1)
	go func() {
		_, err := f.d.NodePublishVolume(context.Background(), publishReq(target, modelpackAttrs))
		done <- err
	}()
	waitFor(t, func() bool { return sf.callers(ref) == 2 })
	close(inner.gate)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if n := inner.callCount(); n != 1 {
		t.Fatalf("pull started %d times, want 1", n)
	}
}

func TestUnpublishUnmountsAndRemovesTheTarget(t *testing.T) {
	f := newFixture(t)
	target := f.target("pod-a")
	if _, err := f.d.NodePublishVolume(context.Background(), publishReq(target, modelpackAttrs)); err != nil {
		t.Fatal(err)
	}
	req := &csi.NodeUnpublishVolumeRequest{VolumeId: "csi-1234", TargetPath: target}
	if _, err := f.d.NodeUnpublishVolume(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if len(f.mounter.unmounts) != 1 {
		t.Fatalf("unmounts = %v", f.mounter.unmounts)
	}
	if _, err := os.Stat(target); !os.IsNotExist(err) {
		t.Fatalf("target still there: %v", err)
	}
	// again: not mounted, not there
	if _, err := f.d.NodeUnpublishVolume(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if len(f.mounter.unmounts) != 1 {
		t.Fatalf("unmounted again: %v", f.mounter.unmounts)
	}
}

func TestUnpublishUnmountsAStackOfMounts(t *testing.T) {
	f := newFixture(t)
	target := f.target("pod-a")
	if err := os.MkdirAll(target, 0o755); err != nil {
		t.Fatal(err)
	}
	f.mounter.stack[target] = 3
	if _, err := f.d.NodeUnpublishVolume(context.Background(), &csi.NodeUnpublishVolumeRequest{VolumeId: "v", TargetPath: target}); err != nil {
		t.Fatal(err)
	}
	if len(f.mounter.unmounts) != 3 {
		t.Fatalf("unmounts = %v", f.mounter.unmounts)
	}
}

func TestUnpublishReportsAnUnmountFailureAndKeepsTheTarget(t *testing.T) {
	f := newFixture(t)
	target := f.target("pod-a")
	if _, err := f.d.NodePublishVolume(context.Background(), publishReq(target, modelpackAttrs)); err != nil {
		t.Fatal(err)
	}
	f.mounter.unmountEr = errBoom
	_, err := f.d.NodeUnpublishVolume(context.Background(), &csi.NodeUnpublishVolumeRequest{VolumeId: "v", TargetPath: target})
	wantCode(t, err, codes.Internal)
	if _, err := os.Stat(target); err != nil {
		t.Fatal("removed a mounted target")
	}
}

func TestUnpublishRejectsBadRequests(t *testing.T) {
	f := newFixture(t)
	for name, req := range map[string]*csi.NodeUnpublishVolumeRequest{
		"no volume id": {TargetPath: f.target("a")},
		"no target":    {VolumeId: "v"},
		"outside":      {VolumeId: "v", TargetPath: "/"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := f.d.NodeUnpublishVolume(context.Background(), req)
			wantCode(t, err, codes.InvalidArgument)
		})
	}
}

func TestIdentityAndNodeInfo(t *testing.T) {
	f := newFixture(t, func(c *Config) { c.Version = "1.2.3" })
	ctx := context.Background()

	info, err := f.d.GetPluginInfo(ctx, &csi.GetPluginInfoRequest{})
	if err != nil || info.GetName() != "model.csi.modelpack.org" || info.GetVendorVersion() != "1.2.3" {
		t.Fatalf("plugin info = %v, %v", info, err)
	}
	caps, err := f.d.GetPluginCapabilities(ctx, &csi.GetPluginCapabilitiesRequest{})
	if err != nil || len(caps.GetCapabilities()) != 0 {
		t.Fatalf("plugin capabilities = %v, %v (no controller service, no topology)", caps, err)
	}
	probe, err := f.d.Probe(ctx, &csi.ProbeRequest{})
	if err != nil || !probe.GetReady().GetValue() {
		t.Fatalf("probe = %v, %v", probe, err)
	}
	node, err := f.d.NodeGetInfo(ctx, &csi.NodeGetInfoRequest{})
	if err != nil || node.GetNodeId() != "node-1" {
		t.Fatalf("node info = %v, %v", node, err)
	}
	nodeCaps, err := f.d.NodeGetCapabilities(ctx, &csi.NodeGetCapabilitiesRequest{})
	if err != nil || len(nodeCaps.GetCapabilities()) != 0 {
		t.Fatalf("node capabilities = %v, %v (no staging, no stats)", nodeCaps, err)
	}
}

func TestProbeFailsWithoutTheLlmmanBinary(t *testing.T) {
	f := newFixture(t, func(c *Config) { c.LlmmanBin = "/nonexistent/llmman" })
	_, err := f.d.Probe(context.Background(), &csi.ProbeRequest{})
	wantCode(t, err, codes.FailedPrecondition)
}

func TestUnimplementedNodeCallsSayWhy(t *testing.T) {
	f := newFixture(t)
	_, err := f.d.NodeStageVolume(context.Background(), &csi.NodeStageVolumeRequest{})
	wantCode(t, err, codes.Unimplemented)
	_, err = f.d.NodeGetVolumeStats(context.Background(), &csi.NodeGetVolumeStatsRequest{})
	wantCode(t, err, codes.Unimplemented)
}

func TestNewRequiresWhatItNeeds(t *testing.T) {
	ok := Config{NodeID: "n", Resolver: &fakeResolver{}, Mounter: newFakeMounter()}
	if _, err := New(ok); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*Config){
		"node id":  func(c *Config) { c.NodeID = "" },
		"resolver": func(c *Config) { c.Resolver = nil },
		"mounter":  func(c *Config) { c.Mounter = nil },
	} {
		c := ok
		mutate(&c)
		if _, err := New(c); err == nil {
			t.Errorf("New accepted a config without a %s", name)
		}
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatal("timed out")
		}
		time.Sleep(5 * time.Millisecond)
	}
}
