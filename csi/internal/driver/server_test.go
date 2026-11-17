package driver

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

func TestParseEndpoint(t *testing.T) {
	for in, want := range map[string]string{
		"unix:///csi/csi.sock": "/csi/csi.sock",
		"/csi/csi.sock":        "/csi/csi.sock",
	} {
		network, addr, err := parseEndpoint(in)
		if err != nil || network != "unix" || addr != want {
			t.Errorf("parseEndpoint(%q) = %q %q %v", in, network, addr, err)
		}
	}
	for _, in := range []string{"", "tcp://0.0.0.0:50051", "unix://", "unix://relative.sock", "csi.sock"} {
		if _, _, err := parseEndpoint(in); err == nil {
			t.Errorf("parseEndpoint(%q) accepted it", in)
		}
	}
}

// The plugin over its real transport: a gRPC server on a unix socket and the
// generated CSI clients, as the kubelet and the registrar sidecar use them.
func TestServesIdentityAndNodeOverAUnixSocket(t *testing.T) {
	f := newFixture(t)
	sock := filepath.Join(t.TempDir(), "csi.sock")

	ctx, stop := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- Serve(ctx, "unix://"+sock, f.d, f.d.log) }()
	t.Cleanup(func() {
		stop()
		select {
		case err := <-done:
			if err != nil {
				t.Errorf("Serve: %v", err)
			}
		case <-time.After(15 * time.Second):
			t.Error("Serve did not stop")
		}
	})

	conn, err := grpc.NewClient("unix://"+sock, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	rpc, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	identity := csi.NewIdentityClient(conn)
	waitFor(t, func() bool { _, err := identity.Probe(rpc, &csi.ProbeRequest{}); return err == nil })
	info, err := identity.GetPluginInfo(rpc, &csi.GetPluginInfoRequest{})
	if err != nil || info.GetName() != DefaultName {
		t.Fatalf("GetPluginInfo = %v, %v", info, err)
	}

	node := csi.NewNodeClient(conn)
	target := f.target("pod-a")
	if _, err := node.NodePublishVolume(rpc, publishReq(target, modelpackAttrs)); err != nil {
		t.Fatal(err)
	}
	if len(f.mounter.binds) != 1 {
		t.Fatalf("binds = %v", f.mounter.binds)
	}
	if _, err := node.NodeUnpublishVolume(rpc, &csi.NodeUnpublishVolumeRequest{VolumeId: "csi-1234", TargetPath: target}); err != nil {
		t.Fatal(err)
	}
	if _, err := node.NodePublishVolume(rpc, publishReq(target, map[string]string{})); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("err = %v", err)
	}
	// the controller service is not served at all
	_, err = csi.NewControllerClient(conn).ControllerGetCapabilities(rpc, &csi.ControllerGetCapabilitiesRequest{})
	if status.Code(err) != codes.Unimplemented {
		t.Fatalf("controller call: %v", err)
	}
}

func TestAHandlerPanicIsAnErrorNotACrash(t *testing.T) {
	f := newFixture(t, func(c *Config) { c.Resolver = panicResolver{} })
	sock := filepath.Join(t.TempDir(), "csi.sock")
	ctx, stop := context.WithCancel(context.Background())
	defer stop()
	go func() { _ = Serve(ctx, "unix://"+sock, f.d, f.d.log) }()

	conn, err := grpc.NewClient("unix://"+sock, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	rpc, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	node := csi.NewNodeClient(conn)
	waitFor(t, func() bool { _, err := csi.NewIdentityClient(conn).Probe(rpc, &csi.ProbeRequest{}); return err == nil })
	_, err = node.NodePublishVolume(rpc, publishReq(f.target("a"), modelpackAttrs))
	if status.Code(err) != codes.Internal {
		t.Fatalf("err = %v", err)
	}
	// still serving
	if _, err := csi.NewIdentityClient(conn).Probe(rpc, &csi.ProbeRequest{}); err != nil {
		t.Fatal(err)
	}
}

type panicResolver struct{}

func (panicResolver) Resolve(context.Context, string) (Tree, error) { panic("boom") }
