package driver

import (
	"strings"
	"testing"
)

const sampleMountinfo = `22 28 0:21 / /sys rw,nosuid,nodev,noexec,relatime shared:7 - sysfs sysfs rw
36 35 98:0 /mnt1 /mnt2 rw,noatime master:1 - ext3 /dev/root rw,errors=continue
512 28 259:3 /var/lib/llmman-csi/cache/abc /var/lib/kubelet/pods/uid-1/volumes/kubernetes.io~csi/model/mount ro,nosuid,nodev,relatime shared:200 - ext4 /dev/nvme0n1p3 rw
513 28 259:3 /x /with\040space/and\011tab/back\134slash rw - ext4 /dev/sda rw
short line
`

func TestMountPointsReadsMountinfo(t *testing.T) {
	points, err := mountPoints(strings.NewReader(sampleMountinfo))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"/sys",
		"/mnt2",
		"/var/lib/kubelet/pods/uid-1/volumes/kubernetes.io~csi/model/mount",
		"/with space/and\ttab/back\\slash",
	} {
		if _, ok := points[want]; !ok {
			t.Errorf("%q is not a mount point in %v", want, points)
		}
	}
	// a root path (field 3) is not a mount point
	if _, ok := points["/mnt1"]; ok {
		t.Error("took a mount's root for its mount point")
	}
	if len(points) != 4 {
		t.Errorf("points = %v", points)
	}
}

func TestUnescapeMountinfo(t *testing.T) {
	for in, want := range map[string]string{
		"/plain":         "/plain",
		`/a\040b`:        "/a b",
		`/trailing\04`:   `/trailing\04`, // not a full escape
		`/not\9octal`:    `/not\9octal`,
		`\134\134`:       `\\`,
		`/utf8-\303\251`: "/utf8-\u00e9",
	} {
		if got := unescapeMountinfo(in); got != want {
			t.Errorf("unescapeMountinfo(%q) = %q, want %q", in, got, want)
		}
	}
}
