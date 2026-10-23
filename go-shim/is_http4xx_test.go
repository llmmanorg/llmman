package main

import (
	"errors"
	"fmt"
	"testing"

	remoteerrors "github.com/containerd/containerd/v2/core/remotes/errors"
)

func TestIsHTTP4xx(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"nil is not permanent", nil, false},
		{"HTTP 400 substring is permanent", errors.New("GET https://example.com/f: HTTP 400"), true},
		{"HTTP 401 substring is permanent", errors.New("GET https://example.com/f: HTTP 401"), true},
		{"HTTP 403 substring is permanent", errors.New("GET https://example.com/f: HTTP 403"), true},
		{"HTTP 404 substring is permanent", errors.New("GET https://example.com/f: HTTP 404"), true},
		{"HTTP 415 substring is permanent", errors.New("PUT https://example.com/f: HTTP 415"), true},
		{"containerd typed 415 is permanent", remoteerrors.ErrUnexpectedStatus{Status: "415 Unsupported Media Type", StatusCode: 415}, true},
		{"wrapped containerd typed 415 is permanent", fmt.Errorf("push failed: %w", remoteerrors.ErrUnexpectedStatus{Status: "415 Unsupported Media Type", StatusCode: 415}), true},
		{"podman status spelling is permanent", errors.New("uploading manifest: StatusCode: 415"), true},
		{"HTTP 500 stays retryable", errors.New("GET https://example.com/f: HTTP 500"), false},
		{"plain error stays retryable", errors.New("connection refused"), false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := isHTTP4xx(c.err); got != c.want {
				t.Errorf("isHTTP4xx(%v) = %v, want %v", c.err, got, c.want)
			}
		})
	}
}
