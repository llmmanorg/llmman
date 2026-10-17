package main

import (
	"errors"
	"fmt"
	"testing"
)

// not_found is set only for an error the backend marked, and marking
// changes neither the message nor what the error unwraps to.
func TestPullFailureReportsOnlyAMarkedNotFound(t *testing.T) {
	cause := errors.New("resolve: docker.io/ai/nope:1b: not found")

	if got := pullFailure(cause); got.NotFound || got.Error != cause.Error() {
		t.Errorf("unmarked failure = %+v, want only the message", got)
	}

	marked := markImageNotFound(cause)
	got := pullFailure(fmt.Errorf("wrapped: %w", marked))
	if !got.NotFound || got.Error != "wrapped: "+cause.Error() {
		t.Errorf("marked failure, wrapped = %+v", got)
	}
	if !errors.Is(marked, cause) {
		t.Error("a marked error no longer unwraps to its cause")
	}
}
