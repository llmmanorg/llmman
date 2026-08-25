//go:build !podman && unix

package main

import (
	"os"
	"syscall"
)

// newRawPipe creates a pipe and returns its read end wrapped as an
// *os.File (for this package's own goroutine to read from, exactly like
// any other os.Pipe()-created file) and its write end as a bare fd —
// deliberately never wrapped in an *os.File on this side, so nothing
// here ever finalizes/closes it out from under the caller (Rust) that's
// the fd's actual, sole owner from this point on. See push_stream.go's
// own doc comment for why Go creates the pipe at all rather than
// accepting one from the FFI caller.
func newRawPipe() (readFile *os.File, writeFD uint64, err error) {
	// syscall.Pipe leaves both ends inheritable; a fork/exec racing with
	// this, in the window before CloseOnExec runs, could otherwise keep
	// the write end alive in a child, so llmman_push_stream_wait's read
	// goroutine never sees EOF. Holding ForkLock.RLock across both closes
	// that window — the same thing os.Pipe itself does on unix.
	syscall.ForkLock.RLock()
	defer syscall.ForkLock.RUnlock()
	var fds [2]int
	if err := syscall.Pipe(fds[:]); err != nil {
		return nil, 0, err
	}
	syscall.CloseOnExec(fds[0])
	syscall.CloseOnExec(fds[1])
	return os.NewFile(uintptr(fds[0]), "push-stream-read"), uint64(fds[1]), nil
}
