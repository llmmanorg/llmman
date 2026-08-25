package main

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"
)

func TestParseRetryAfterDelaySeconds(t *testing.T) {
	cases := []struct {
		name    string
		header  string
		wantDur time.Duration
		wantOK  bool
	}{
		{"absent", "", 0, false},
		{"typical", "30", 30 * time.Second, true},
		{"zero is a valid, distinct result (retry immediately)", "0", 0, true},
		{"negative", "-5", 0, false},
		{"whitespace", "  12  ", 12 * time.Second, true},
		{"huge value is capped, not left to overflow", "999999999999", retryAfterCap, true},
		{"http-date form is not supported (matches huggingface_hub)", "Wed, 21 Oct 2015 07:28:00 GMT", 0, false},
		{"garbage", "soon", 0, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := http.Header{}
			if c.header != "" {
				h.Set("Retry-After", c.header)
			}
			gotDur, gotOK := parseRetryAfter(h)
			if gotDur != c.wantDur || gotOK != c.wantOK {
				t.Errorf("parseRetryAfter(%q) = (%v, %v), want (%v, %v)", c.header, gotDur, gotOK, c.wantDur, c.wantOK)
			}
		})
	}
}

func TestRetryAfterFromError(t *testing.T) {
	if _, ok := retryAfter(nil); ok {
		t.Error("retryAfter(nil) ok = true, want false")
	}
	if _, ok := retryAfter(errors.New("plain error, not an httpStatusError")); ok {
		t.Error("retryAfter(plain error) ok = true, want false")
	}

	resp := &http.Response{StatusCode: 429, Header: http.Header{"Retry-After": []string{"7"}}}
	err := newHTTPStatusError("GET https://example.com/f", resp)
	if got, ok := retryAfter(err); !ok || got != 7*time.Second {
		t.Errorf("retryAfter(429 w/ Retry-After: 7) = (%v, %v), want (7s, true)", got, ok)
	}
	// err.Error() must still contain the "HTTP <code>" substring isHTTP4xx greps for.
	if want := "GET https://example.com/f: HTTP 429"; err.Error() != want {
		t.Errorf("err.Error() = %q, want %q", err.Error(), want)
	}

	// Wrapped (via %w) errors must still be found by errors.As.
	wrapped := errWrap{err}
	if got, ok := retryAfter(wrapped); !ok || got != 7*time.Second {
		t.Errorf("retryAfter(wrapped) = (%v, %v), want (7s, true)", got, ok)
	}

	// A zero Retry-After ("retry immediately") must not be mistaken for "absent".
	zeroResp := &http.Response{StatusCode: 429, Header: http.Header{"Retry-After": []string{"0"}}}
	zeroErr := newHTTPStatusError("GET https://example.com/f", zeroResp)
	if got, ok := retryAfter(zeroErr); !ok || got != 0 {
		t.Errorf("retryAfter(Retry-After: 0) = (%v, %v), want (0, true)", got, ok)
	}
}

// errWrap is a minimal error wrapper (like fmt.Errorf("...: %w", err))
// confirming retryAfter's errors.As unwraps through one level.
type errWrap struct{ err error }

func (e errWrap) Error() string { return "wrapped: " + e.err.Error() }
func (e errWrap) Unwrap() error { return e.err }

// TestRetryStreamHonorsRetryAfterOverExponentialBackoff confirms
// retryStream waits the server-specified duration on a 429 instead of
// the usual exponential backoff (~0.75-1.25s for the first retry), by
// using a Retry-After short enough (100ms) to tell the two apart.
func TestRetryStreamHonorsRetryAfterOverExponentialBackoff(t *testing.T) {
	attempts := 0
	start := time.Now()
	err := retryStream(context.Background(), "test", isHTTP4xx, func() error {
		attempts++
		if attempts == 1 {
			return &httpStatusError{
				prefix: "GET https://example.com/f", statusCode: 429,
				retryAfter: 100 * time.Millisecond, hasRetryAfter: true,
			}
		}
		return nil
	})
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("retryStream: %v", err)
	}
	if attempts != 2 {
		t.Fatalf("attempts = %d, want 2", attempts)
	}
	if elapsed >= 500*time.Millisecond {
		t.Errorf("elapsed = %v, want well under ~0.75-1.25s — retryStream doesn't appear to be honoring Retry-After", elapsed)
	}
}
