package main

import (
	"strings"
	"testing"
)

// TestGlobMatchAgreesWithRustGlob is the Go half of the glob parity
// convention (see TestCanonicalRepositoryFoldsDockerHubSpellings). Its
// cases are the assertions of the Rust tests in src/verify.rs:
// glob_matches_literals_exactly, single_star_stays_within_one_path_segment,
// double_star_crosses_path_segments, glob_is_anchored_at_both_ends and
// glob_does_not_blow_up_on_a_pathological_pattern. Change both or neither.
func TestGlobMatchAgreesWithRustGlob(t *testing.T) {
	// The trailing "b" makes the pathological case the bad one: every
	// one of the 24 `*a` branches has to be explored before the match
	// can be ruled out. Unmemoized that is exponential and this test
	// does not finish.
	pathological := strings.Repeat("*a", 24)
	as := strings.Repeat("a", 64)
	cases := []struct {
		pattern, s string
		want       bool
	}{
		// glob_matches_literals_exactly
		{"docker.io/org/model", "docker.io/org/model", true},
		{"docker.io/org/model", "docker.io/org/model2", false},
		// single_star_stays_within_one_path_segment
		{"docker.io/org/*", "docker.io/org/model", true},
		{"docker.io/*/model", "docker.io/org/model", true},
		{"docker.io/org/*", "docker.io/org/team/model", false},
		// double_star_crosses_path_segments
		{"docker.io/**", "docker.io/org/model", true},
		{"docker.io/**", "docker.io/org/team/model", true},
		{"docker.io/**/model", "docker.io/org/team/model", true},
		{"quay.io/**", "docker.io/org/model", false},
		// glob_is_anchored_at_both_ends
		{"docker.io/org/model*", "docker.io/org/model", true},
		{"docker.io", "docker.io/org/model", false},
		// glob_does_not_blow_up_on_a_pathological_pattern
		{pathological + "b", as, false},
		{pathological, as, true},
	}
	for _, c := range cases {
		if got := globMatch(c.pattern, c.s); got != c.want {
			t.Errorf("globMatch(%q, %q) = %v, want %v", c.pattern, c.s, got, c.want)
		}
	}
}
