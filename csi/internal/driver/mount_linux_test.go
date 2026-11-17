//go:build linux

package driver

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/sys/unix"
)

func TestSystemMounterFindsMountPointsInTheFileItIsGiven(t *testing.T) {
	name := filepath.Join(t.TempDir(), "mountinfo")
	if err := os.WriteFile(name, []byte(sampleMountinfo), 0o644); err != nil {
		t.Fatal(err)
	}
	m := SystemMounter{MountInfo: name}
	if ok, err := m.IsMountPoint("/mnt2"); err != nil || !ok {
		t.Fatalf("/mnt2: %v, %v", ok, err)
	}
	if ok, err := m.IsMountPoint("/mnt3"); err != nil || ok {
		t.Fatalf("/mnt3: %v, %v", ok, err)
	}
	if _, err := (SystemMounter{MountInfo: name + ".missing"}).IsMountPoint("/"); err == nil {
		t.Fatal("no error for a missing mountinfo")
	}
}

func TestMountReadOnlyRefusesPathsThatCannotBeOverlayLayers(t *testing.T) {
	m := SystemMounter{EmptyDir: t.TempDir()}
	for _, src := range []string{"/a:b", "/a,b", `/a\b`} {
		if err := m.MountReadOnly(src, t.TempDir()); err == nil {
			t.Errorf("accepted %q", src)
		}
	}
}

// mounter returns a SystemMounter, skipping the test where the process
// cannot mount (anything but root, or a user and mount namespace:
// `unshare -Urm go test ./...`).
func mounter(t *testing.T) SystemMounter {
	t.Helper()
	probe := t.TempDir()
	if err := unix.Mount(probe, probe, "", unix.MS_BIND, ""); err != nil {
		t.Skipf("cannot mount here: %v", err)
	}
	_ = unix.Unmount(probe, 0)
	return SystemMounter{EmptyDir: t.TempDir()}
}

func modelDir(t *testing.T, root string) string {
	t.Helper()
	src := filepath.Join(root, "model")
	if err := os.MkdirAll(src, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(src, "config.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	return src
}

func TestSystemMounterMountsReadOnlyAndUnmounts(t *testing.T) {
	m := mounter(t)
	root := t.TempDir()
	src := modelDir(t, root)
	target := filepath.Join(root, "pod", "mount")
	if err := os.MkdirAll(target, 0o755); err != nil {
		t.Fatal(err)
	}

	if err := m.MountReadOnly(src, target); err != nil {
		t.Skipf("no overlayfs here: %v", err)
	}
	t.Cleanup(func() { _ = unix.Unmount(target, unix.MNT_DETACH) })

	if ok, err := m.IsMountPoint(target); err != nil || !ok {
		t.Fatalf("IsMountPoint = %v, %v", ok, err)
	}
	opts := mountOptions(t, target)
	for _, want := range []string{"ro", "nosuid", "nodev"} {
		if !opts[want] {
			t.Errorf("mount options %v lack %s", opts, want)
		}
	}
	if b, err := os.ReadFile(filepath.Join(target, "config.json")); err != nil || string(b) != "{}" {
		t.Fatalf("read through the mount: %q, %v", b, err)
	}
	for _, write := range []func() error{
		func() error { return os.WriteFile(filepath.Join(target, "new"), nil, 0o644) },
		func() error { return os.WriteFile(filepath.Join(target, "config.json"), []byte("x"), 0o644) },
		func() error { return os.Remove(filepath.Join(target, "config.json")) },
	} {
		if write() == nil {
			t.Fatal("changed the model through a read-only mount")
		}
	}
	if b, _ := os.ReadFile(filepath.Join(src, "config.json")); string(b) != "{}" {
		t.Fatalf("source changed: %q", b)
	}
	if err := os.WriteFile(filepath.Join(src, "new"), nil, 0o644); err != nil {
		t.Fatalf("the source went read-only: %v", err)
	}

	if err := m.Unmount(target); err != nil {
		t.Fatal(err)
	}
	if ok, _ := m.IsMountPoint(target); ok {
		t.Fatal("still mounted")
	}
	// not mounted any more: unmounting again is not an error
	if err := m.Unmount(target); err != nil {
		t.Fatalf("second unmount: %v", err)
	}
}

// Found on a real cluster: with a read-only bind mount, a pod whose
// volumeMount is not `readOnly` (the default) could write to the shared model
// files, because the container runtime's bind of the volume asks for a
// read-write mount and gets one. A bind of the volume remounted read-write
// stands in for that, in a shared directory and a peer of it standing in for
// the plugin's view of /var/lib/kubelet/pods and the host's.
func TestAPodThatRemountsItsVolumeReadWriteStillCannotWrite(t *testing.T) {
	m := mounter(t)
	root := t.TempDir()
	src := modelDir(t, root)
	pods := filepath.Join(root, "pods")
	peer := filepath.Join(root, "host-pods")
	podView := filepath.Join(root, "container-model")
	for _, d := range []string{pods, peer, podView} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := unix.Mount(pods, pods, "", unix.MS_BIND, ""); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = unix.Unmount(pods, unix.MNT_DETACH) })
	if err := unix.Mount("", pods, "", unix.MS_SHARED, ""); err != nil {
		t.Skipf("cannot make a shared mount here: %v", err)
	}
	if err := unix.Mount(pods, peer, "", unix.MS_BIND, ""); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = unix.Unmount(peer, unix.MNT_DETACH) })

	target := filepath.Join(pods, "uid", "mount")
	if err := os.MkdirAll(target, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := m.MountReadOnly(src, target); err != nil {
		t.Skipf("no overlayfs here: %v", err)
	}
	t.Cleanup(func() { _ = unix.Unmount(target, unix.MNT_DETACH) })

	// the host's copy
	propagated := filepath.Join(peer, "uid", "mount")
	if opts := mountOptions(t, propagated); !opts["ro"] || opts["rw"] {
		t.Fatalf("the host's copy of the mount has options %v, want ro", opts)
	}

	// the container runtime's bind of it, made read-write
	if err := unix.Mount(propagated, podView, "", unix.MS_BIND, ""); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = unix.Unmount(podView, unix.MNT_DETACH) })
	if err := unix.Mount("", podView, "", unix.MS_REMOUNT|unix.MS_BIND|unix.MS_NOSUID|unix.MS_NODEV, ""); err != nil {
		t.Fatalf("remount read-write: %v", err)
	}
	if opts := mountOptions(t, podView); !opts["rw"] {
		t.Fatalf("test setup: the pod's mount has options %v, want rw", opts)
	}
	for _, name := range []string{"tamper", "config.json"} {
		if err := os.WriteFile(filepath.Join(podView, name), []byte("tampered"), 0o644); err == nil {
			t.Fatalf("wrote %s through a read-write bind of the volume", name)
		}
	}
	if b, _ := os.ReadFile(filepath.Join(src, "config.json")); string(b) != "{}" {
		t.Fatalf("the shared model file was changed: %q", b)
	}
}

// mountOptions are the per-mount options of the mount at path.
func mountOptions(t *testing.T, path string) map[string]bool {
	t.Helper()
	f, err := os.Open("/proc/self/mountinfo")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	var opts map[string]bool
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) > 5 && unescapeMountinfo(fields[4]) == path {
			opts = map[string]bool{} // the last one is the topmost
			for _, o := range strings.Split(fields[5], ",") {
				opts[o] = true
			}
		}
	}
	if opts == nil {
		t.Fatalf("%s is not in mountinfo", path)
	}
	return opts
}
