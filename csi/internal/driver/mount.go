package driver

import (
	"bufio"
	"io"
	"strings"
)

// Mounter does the mounting NodePublishVolume and NodeUnpublishVolume need.
type Mounter interface {
	// IsMountPoint reports whether something is mounted at path.
	IsMountPoint(path string) (bool, error)
	// MountReadOnly mounts the directory source at target so that nothing
	// can write to it through target, or through any mount made of target,
	// whatever flags a container runtime gives that one. It leaves nothing
	// mounted if it fails.
	MountReadOnly(source, target string) error
	// Unmount removes the topmost mount at target.
	Unmount(target string) error
}

// mountPoints returns the mount points listed in a /proc/self/mountinfo.
func mountPoints(r io.Reader) (map[string]struct{}, error) {
	points := map[string]struct{}{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		// id parent major:minor root mountpoint options ...
		fields := strings.Fields(sc.Text())
		if len(fields) < 5 {
			continue
		}
		points[unescapeMountinfo(fields[4])] = struct{}{}
	}
	return points, sc.Err()
}

// unescapeMountinfo undoes the kernel's octal escapes (\040 for a space,
// \011, \012, \134) in a mountinfo path.
func unescapeMountinfo(s string) string {
	if !strings.Contains(s, `\`) {
		return s
	}
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		if s[i] == '\\' && i+3 < len(s) && isOctal(s[i+1]) && isOctal(s[i+2]) && isOctal(s[i+3]) {
			b.WriteByte((s[i+1]-'0')<<6 | (s[i+2]-'0')<<3 | (s[i+3] - '0'))
			i += 3
			continue
		}
		b.WriteByte(s[i])
	}
	return b.String()
}

func isOctal(c byte) bool { return c >= '0' && c <= '7' }
