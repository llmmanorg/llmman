//! Small string helpers shared by more than one module — kept in one
//! place rather than duplicated per-file so a fix or behavior change
//! only has to happen once.

/// Length of the longest suffix of `s` that is also a prefix of `delim`.
///
/// Used by streaming parsers ([`crate::harmony`], [`crate::thinking`]) to
/// decide how many trailing bytes of already-seen text might be the start
/// of a delimiter split across two chunks (e.g. one chunk ending in
/// `<thi` and the next starting with `nk>`), so those bytes can be held
/// back instead of emitted early.
pub fn overlap(s: &str, delim: &str) -> usize {
    let max = delim.len().min(s.len());
    for i in (1..=max).rev() {
        if s.ends_with(&delim[..i]) {
            return i;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_finds_longest_suffix_prefix_match() {
        assert_eq!(overlap("abc</thi", "</think>"), 5);
        assert_eq!(overlap("abcdef", "</think>"), 0);
        assert_eq!(overlap("", "</think>"), 0);
    }
}
