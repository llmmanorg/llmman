//go:build !windows

package main

import "syscall"

// diskFree returns the bytes this user may still write to the filesystem
// holding dir; false where that can't be read, or for a filesystem that
// reports no size, as some FUSE ones do.
func diskFree(dir string) (uint64, bool) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(dir, &st); err != nil || st.Blocks == 0 {
		return 0, false
	}
	return uint64(st.Bavail) * uint64(st.Bsize), true
}
