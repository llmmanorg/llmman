package main

import (
	"errors"
	"testing"
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
