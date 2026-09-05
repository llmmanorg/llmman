//! General string and other utilities.
pub fn overlap(s: &str, delim: &str) -> usize {
    let max = delim.len().min(s.len());
    for i in (1..=max).rev() {
        if s.ends_with(&delim[..i]) {
            return i;
        }
    }
    0
}
