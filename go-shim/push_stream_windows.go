//go:build !podman && windows

package main

import (
	"os"
	"syscall"
)

// newRawPipe is push_stream_unix.go's Windows equivalent — see that
// file's doc comment for why the write end is deliberately returned as
// a bare HANDLE (widened to uint64) rather than wrapped in an *os.File.
func newRawPipe() (readFile *os.File, writeHandle uint64, err error) {
	var r, w syscall.Handle
	if err := syscall.CreatePipe(&r, &w, nil, 0); err != nil {
		return nil, 0, err
	}
	return os.NewFile(uintptr(r), "push-stream-read"), uint64(w), nil
}
