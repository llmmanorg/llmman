package main

import (
	"errors"
	"fmt"
	"testing"
)

// A pull failure carries not_found only when the backend marked it, and
// marking changes neither the message nor what the error unwraps to.
func TestPullFailureReportsOnlyAMarkedNotFound(t *testing.T) {
	cause := errors.New("resolve: docker.io/ai/nope:1b: not found")

	plain := pullFailure(cause)
	if plain.NotFound || plain.Error != cause.Error() || plain.OK {
		t.Errorf("unmarked failure = %+v, want only the message", plain)
	}

	marked := markImageNotFound(cause)
	got := pullFailure(fmt.Errorf("wrapped: %w", marked))
	if !got.NotFound {
		t.Errorf("a marked failure, wrapped, lost not_found: %+v", got)
	}
	if want := "wrapped: " + cause.Error(); got.Error != want {
		t.Errorf("message = %q, want %q", got.Error, want)
	}
	if !errors.Is(marked, cause) {
		t.Error("a marked error no longer unwraps to its cause")
	}
}
