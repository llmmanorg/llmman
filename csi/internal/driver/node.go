package driver

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/wrapperspb"
)

func wrapBool(v bool) *wrapperspb.BoolValue { return wrapperspb.Bool(v) }

// keyedLocks is a set of try-locks, one per key.
type keyedLocks struct {
	mu   sync.Mutex
	held map[string]struct{}
}

// tryLock takes the lock for key, or reports false if it is held.
func (l *keyedLocks) tryLock(key string) (unlock func(), ok bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if _, busy := l.held[key]; busy {
		return nil, false
	}
	if l.held == nil {
		l.held = map[string]struct{}{}
	}
	l.held[key] = struct{}{}
	return func() {
		l.mu.Lock()
		delete(l.held, key)
		l.mu.Unlock()
	}, true
}

// lockTarget takes the lock for target, or fails with Aborted if an
// operation on it is running.
func (d *Driver) lockTarget(target string) (func(), error) {
	unlock, ok := d.locks.tryLock(target)
	if !ok {
		return nil, status.Errorf(codes.Aborted, "an operation on %s is already in progress", target)
	}
	return unlock, nil
}

// targetPath checks a request's target path: absolute, already clean (so
// `..` cannot lead out of the kubelet directory), and under it if one is
// configured.
func (d *Driver) targetPath(target string) (string, error) {
	if target == "" {
		return "", status.Error(codes.InvalidArgument, "target path is required")
	}
	if !filepath.IsAbs(target) || filepath.Clean(target) != target {
		return "", status.Errorf(codes.InvalidArgument, "target path %q is not an absolute, clean path", target)
	}
	if dir := d.cfg.KubeletDir; dir != "" && !strings.HasPrefix(target, strings.TrimSuffix(dir, "/")+"/") {
		return "", status.Errorf(codes.InvalidArgument, "target path %q is not under %s", target, dir)
	}
	return target, nil
}

// NodePublishVolume mounts the artifact a volume names, read-only, at the
// target path: an inline ephemeral volume's `volumeAttributes`, or a
// pre-provisioned PersistentVolume's (both arrive as the volume context).
// The model directory itself lives once per node, in llmman's cache, and is
// shared by every pod that names the artifact.
func (d *Driver) NodePublishVolume(ctx context.Context, req *csi.NodePublishVolumeRequest) (*csi.NodePublishVolumeResponse, error) {
	if req.GetVolumeId() == "" {
		return nil, status.Error(codes.InvalidArgument, "volume ID is required")
	}
	capability := req.GetVolumeCapability()
	if capability == nil {
		return nil, status.Error(codes.InvalidArgument, "volume capability is required")
	}
	if capability.GetMount() == nil {
		return nil, status.Error(codes.InvalidArgument, "only mount volumes are supported")
	}
	target, err := d.targetPath(req.GetTargetPath())
	if err != nil {
		return nil, err
	}
	reference, err := d.reference(req.GetVolumeContext())
	if err != nil {
		return nil, err
	}

	unlock, err := d.lockTarget(target)
	if err != nil {
		return nil, err
	}
	defer unlock()

	log := d.log.With("volume", req.GetVolumeId(), "target", target, "reference", reference)
	// Sent because the CSIDriver sets podInfoOnMount; only for the log.
	if name := req.GetVolumeContext()["csi.storage.k8s.io/pod.name"]; name != "" {
		log = log.With("pod", req.GetVolumeContext()["csi.storage.k8s.io/pod.namespace"]+"/"+name)
	}
	mounted, err := d.cfg.Mounter.IsMountPoint(target)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "check %s: %v", target, err)
	}
	if mounted {
		log.Info("already published")
		return &csi.NodePublishVolumeResponse{}, nil
	}

	tree, err := d.cfg.Resolver.Resolve(ctx, reference)
	if err != nil {
		if ctx.Err() != nil {
			// The pull carries on in the background; the retry joins it.
			return nil, status.FromContextError(ctx.Err()).Err()
		}
		log.Error("resolve failed", "error", err)
		return nil, status.Errorf(codes.Internal, "pull %s: %v", reference, err)
	}
	if err := os.MkdirAll(target, 0o750); err != nil {
		return nil, status.Errorf(codes.Internal, "create %s: %v", target, err)
	}
	if err := d.cfg.Mounter.MountReadOnly(tree.Dir, target); err != nil {
		log.Error("mount failed", "dir", tree.Dir, "error", err)
		return nil, status.Errorf(codes.Internal, "mount %s at %s: %v", tree.Dir, target, err)
	}
	log.Info("published", "dir", tree.Dir, "digest", tree.Digest, "format", tree.Format)
	return &csi.NodePublishVolumeResponse{}, nil
}

// maxStackedMounts bounds the unmount loop in NodeUnpublishVolume.
const maxStackedMounts = 32

// NodeUnpublishVolume unmounts the target path and removes it. It is
// idempotent: a target that is not mounted, or does not exist, succeeds.
func (d *Driver) NodeUnpublishVolume(_ context.Context, req *csi.NodeUnpublishVolumeRequest) (*csi.NodeUnpublishVolumeResponse, error) {
	if req.GetVolumeId() == "" {
		return nil, status.Error(codes.InvalidArgument, "volume ID is required")
	}
	target, err := d.targetPath(req.GetTargetPath())
	if err != nil {
		return nil, err
	}
	unlock, err := d.lockTarget(target)
	if err != nil {
		return nil, err
	}
	defer unlock()

	log := d.log.With("volume", req.GetVolumeId(), "target", target)
	// A retried publish can leave one mount on top of another.
	for i := 0; ; i++ {
		mounted, err := d.cfg.Mounter.IsMountPoint(target)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "check %s: %v", target, err)
		}
		if !mounted {
			break
		}
		if i == maxStackedMounts {
			return nil, status.Errorf(codes.Internal, "%s is still mounted after %d unmounts", target, i)
		}
		if err := d.cfg.Mounter.Unmount(target); err != nil {
			return nil, status.Errorf(codes.Internal, "unmount %s: %v", target, err)
		}
	}
	if err := os.Remove(target); err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, status.Errorf(codes.Internal, "remove %s: %v", target, err)
	}
	log.Info("unpublished")
	return &csi.NodeUnpublishVolumeResponse{}, nil
}
