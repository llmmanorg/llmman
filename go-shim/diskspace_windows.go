//go:build windows

package main

import "golang.org/x/sys/windows"

// diskFree returns the bytes this user may still write to the volume
// holding dir; false where that can't be read.
func diskFree(dir string) (uint64, bool) {
	path, err := windows.UTF16PtrFromString(dir)
	if err != nil {
		return 0, false
	}
	var free, total uint64
	if err := windows.GetDiskFreeSpaceEx(path, &free, &total, nil); err != nil || total == 0 {
		return 0, false
	}
	return free, true
}
