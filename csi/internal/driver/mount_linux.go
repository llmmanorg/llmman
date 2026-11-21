//go:build linux

package driver

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/sys/unix"
)

// SystemMounter mounts with mount(2). The plugin runs privileged, with
// /var/lib/kubelet/pods propagated to the host, so what it mounts under a
// pod's target path is what the pod sees.
type SystemMounter struct {
	// MountInfo is the file listing mounts; /proc/self/mountinfo if empty.
	MountInfo string
	// EmptyDir is an empty directory for the mounts' second layer; a
	// directory under os.TempDir() if empty.
	EmptyDir string
}

func (m SystemMounter) IsMountPoint(path string) (bool, error) {
	name := m.MountInfo
	if name == "" {
		name = "/proc/self/mountinfo"
	}
	f, err := os.Open(name)
	if err != nil {
		return false, err
	}
	defer f.Close()
	points, err := mountPoints(f)
	if err != nil {
		return false, err
	}
	_, ok := points[path]
	return ok, nil
}

// MountReadOnly mounts source at target as a read-only overlay filesystem.
//
// A read-only bind mount would not hold: the runtime binds the volume into
// the container read-write unless the pod says readOnly, and that would let
// a pod write to model files shared with every other pod. An overlay with no
// upper layer is read-only in the filesystem itself.
//
// Without an upper layer, some kernels need two lower layers; the second is
// an empty directory.
func (m SystemMounter) MountReadOnly(source, target string) error {
	empty := m.EmptyDir
	if empty == "" {
		empty = filepath.Join(os.TempDir(), "llmman-csi-empty")
	}
	if err := os.MkdirAll(empty, 0o755); err != nil {
		return err
	}
	// ':' and ',' separate layers and options; not every kernel can escape them.
	for _, dir := range []string{source, empty} {
		if strings.ContainsAny(dir, ":,\\") {
			return fmt.Errorf("%s has a character (one of : , \\) that cannot be in an overlay layer path", dir)
		}
	}
	const flags = unix.MS_RDONLY | unix.MS_NOSUID | unix.MS_NODEV
	if err := unix.Mount("overlay", target, "overlay", flags, "lowerdir="+source+":"+empty); err != nil {
		return fmt.Errorf("mount overlay of %s: %w (the model store must be on a filesystem overlayfs can use)", source, err)
	}
	return nil
}

func (SystemMounter) Unmount(target string) error {
	err := unix.Unmount(target, 0)
	if errors.Is(err, unix.EBUSY) {
		// still open in a container that is going away: detach now
		err = unix.Unmount(target, unix.MNT_DETACH)
	}
	if errors.Is(err, unix.EINVAL) {
		return nil // not a mount point
	}
	return err
}

// NewSystemMounter returns the Mounter for this host.
func NewSystemMounter() Mounter { return SystemMounter{} }
