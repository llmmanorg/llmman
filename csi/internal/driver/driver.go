// Package driver is the CSI node plugin that mounts ModelPack model
// artifacts, pulled and unpacked by the llmman binary, into pods.
//
// It serves the CSI Identity and Node services only. There is no controller
// service and no staging: NodePublishVolume asks `llmman resolve --dir` for
// the artifact's complete model directory (pulling it first if the node does
// not have it) and bind-mounts that directory read-only at the pod's target
// path. Many pods that name the same artifact share one pull and one copy on
// disk.
package driver

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os/exec"
	"slices"
	"strings"

	"github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// DefaultName is the driver name of modelpack/model-csi-driver. It is the
// default so that manifests written for that driver (`csi.driver` and the
// `model.csi.modelpack.org/reference` volume attribute) work unchanged.
const DefaultName = "model.csi.modelpack.org"

// compatPrefix is the volume attribute prefix of that same driver, honoured
// whatever the driver is named.
const compatPrefix = DefaultName

// Config is everything a Driver needs.
type Config struct {
	// Name is the CSI driver name the plugin reports; it must equal the name
	// of the CSIDriver object and of the `csi.driver` of the volumes.
	Name string
	// Version is reported as the plugin's vendor version.
	Version string
	// NodeID identifies this node to the container orchestrator.
	NodeID string
	// LlmmanBin is the llmman binary to run (a path, or a name looked up in
	// PATH).
	LlmmanBin string
	// KubeletDir, when set, is the directory every target path must be under.
	KubeletDir string

	Logger   *slog.Logger
	Resolver Resolver
	Mounter  Mounter
}

// Driver implements csi.IdentityServer and csi.NodeServer.
type Driver struct {
	csi.UnimplementedIdentityServer
	csi.UnimplementedNodeServer

	cfg   Config
	log   *slog.Logger
	locks keyedLocks
}

var (
	_ csi.IdentityServer = (*Driver)(nil)
	_ csi.NodeServer     = (*Driver)(nil)
)

// New returns a Driver, or an error for a Config it cannot work with.
func New(cfg Config) (*Driver, error) {
	if cfg.Name == "" {
		cfg.Name = DefaultName
	}
	if cfg.NodeID == "" {
		return nil, errors.New("node ID is required")
	}
	if cfg.Resolver == nil {
		return nil, errors.New("resolver is required")
	}
	if cfg.Mounter == nil {
		return nil, errors.New("mounter is required")
	}
	if cfg.LlmmanBin == "" {
		cfg.LlmmanBin = "llmman"
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	return &Driver{cfg: cfg, log: cfg.Logger}, nil
}

func (d *Driver) GetPluginInfo(context.Context, *csi.GetPluginInfoRequest) (*csi.GetPluginInfoResponse, error) {
	return &csi.GetPluginInfoResponse{Name: d.cfg.Name, VendorVersion: d.cfg.Version}, nil
}

// GetPluginCapabilities reports none: no controller service, and volumes are
// not topology constrained.
func (d *Driver) GetPluginCapabilities(context.Context, *csi.GetPluginCapabilitiesRequest) (*csi.GetPluginCapabilitiesResponse, error) {
	return &csi.GetPluginCapabilitiesResponse{}, nil
}

// Probe reports ready when the llmman binary can be found: without it no
// volume can be published, which is what the orchestrator wants to know.
func (d *Driver) Probe(context.Context, *csi.ProbeRequest) (*csi.ProbeResponse, error) {
	if _, err := exec.LookPath(d.cfg.LlmmanBin); err != nil {
		return nil, status.Errorf(codes.FailedPrecondition, "llmman binary: %v", err)
	}
	return &csi.ProbeResponse{Ready: wrapBool(true)}, nil
}

// NodeGetCapabilities reports none: no staging, no volume statistics.
func (d *Driver) NodeGetCapabilities(context.Context, *csi.NodeGetCapabilitiesRequest) (*csi.NodeGetCapabilitiesResponse, error) {
	return &csi.NodeGetCapabilitiesResponse{}, nil
}

func (d *Driver) NodeGetInfo(context.Context, *csi.NodeGetInfoRequest) (*csi.NodeGetInfoResponse, error) {
	return &csi.NodeGetInfoResponse{NodeId: d.cfg.NodeID}, nil
}

// referenceKeys are the volume attribute keys that carry the artifact
// reference: the plugin's own name as the prefix, and the one
// modelpack/model-csi-driver uses.
func (d *Driver) referenceKeys() []string {
	keys := []string{d.cfg.Name + "/reference"}
	if d.cfg.Name != compatPrefix {
		keys = append(keys, compatPrefix+"/reference")
	}
	return keys
}

// unsupportedKeys are the attributes modelpack/model-csi-driver takes that
// have no equivalent here, with the value that means "not asked for". Any
// other value is refused rather than silently ignored.
func (d *Driver) unsupportedKeys() map[string][]string {
	out := map[string][]string{}
	for _, prefix := range []string{d.cfg.Name, compatPrefix} {
		out[prefix+"/exclude-model-weights"] = []string{"", "false"}
		out[prefix+"/exclude-file-patterns"] = []string{"", "[]"}
	}
	return out
}

// reference is the artifact a volume context names.
func (d *Driver) reference(volumeContext map[string]string) (string, error) {
	for key, allowed := range d.unsupportedKeys() {
		if !slices.Contains(allowed, strings.TrimSpace(volumeContext[key])) {
			return "", status.Errorf(codes.InvalidArgument, "volume attribute %s is not supported by this driver", key)
		}
	}
	var ref string
	for _, key := range d.referenceKeys() {
		if v := strings.TrimSpace(volumeContext[key]); v != "" {
			ref = v
			break
		}
	}
	if ref == "" {
		return "", status.Errorf(codes.InvalidArgument, "volume attribute %s is required", d.referenceKeys()[0])
	}
	if err := validateReference(ref); err != nil {
		return "", status.Error(codes.InvalidArgument, err.Error())
	}
	return ref, nil
}

// validateReference rejects what no model reference contains, before it
// reaches a command line. A leading '/' is llmman's local-directory import:
// a pod could have it copy the driver's own files (its registry credentials).
func validateReference(ref string) error {
	if len(ref) > 1024 {
		return errors.New("reference is too long")
	}
	if strings.HasPrefix(ref, "-") {
		return errors.New("reference must not start with '-'")
	}
	if strings.HasPrefix(ref, "/") {
		return errors.New("reference must not be a local path")
	}
	for _, r := range ref {
		if r <= ' ' || r == 0x7f {
			return fmt.Errorf("reference contains a space or control character: %q", ref)
		}
	}
	return nil
}
