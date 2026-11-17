//go:build !linux

package driver

import "errors"

var errUnsupported = errors.New("mounting is only supported on Linux")

type unsupportedMounter struct{}

func (unsupportedMounter) IsMountPoint(string) (bool, error) { return false, errUnsupported }
func (unsupportedMounter) MountReadOnly(_, _ string) error   { return errUnsupported }
func (unsupportedMounter) Unmount(string) error              { return errUnsupported }

// NewSystemMounter returns a Mounter that refuses everything: the plugin
// only runs on Linux nodes; this keeps the package building elsewhere.
func NewSystemMounter() Mounter { return unsupportedMounter{} }
