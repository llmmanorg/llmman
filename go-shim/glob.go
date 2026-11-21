package main

// globMatch reports whether pattern matches s in full. `*` matches
// within one path segment (never a `/`), `**` crosses segments, and the
// match is anchored at both ends. It is a port of glob_match in
// src/verify.rs; the two sides must agree because the Rust side matches
// `pattern` against the pulled reference and the Go side matches `claims`
// against the signature's claimed repository.
//
// Byte-wise and case-sensitive, like the Rust original.
func globMatch(pattern, s string) bool {
	// Memoized on (remaining pattern, remaining text). Without it, a
	// pattern of repeated `*a` branches is exponential in its length.
	seen := make(map[[2]int]struct{})
	return globBytes(pattern, s, seen)
}

func globBytes(p, t string, seen map[[2]int]struct{}) bool {
	key := [2]int{len(p), len(t)}
	if _, failed := seen[key]; failed {
		return false // this state already failed
	}
	seen[key] = struct{}{}
	if len(p) == 0 {
		return len(t) == 0
	}
	if p[0] != '*' {
		return len(t) > 0 && p[0] == t[0] && globBytes(p[1:], t[1:], seen)
	}
	if len(p) > 1 && p[1] == '*' {
		rest := p[2:]
		for i := 0; i <= len(t); i++ {
			if globBytes(rest, t[i:], seen) {
				return true
			}
		}
		return false
	}
	rest := p[1:]
	for i := 0; i <= len(t); i++ {
		if globBytes(rest, t[i:], seen) {
			return true
		}
		// Having tried leaving the separator to the rest of the
		// pattern, a single `*` may not swallow it.
		if i < len(t) && t[i] == '/' {
			break
		}
	}
	return false
}
