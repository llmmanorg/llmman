#![no_main]

use libfuzzer_sys::fuzz_target;

// Fuzzes shortnames::parse_registry_ref (private to the parent crate) via
// the fuzzing-feature-gated wrapper fuzz_check_parse_registry_ref, which
// panics if a successful parse ever yields a ".." component or an
// over-length part. See src/shortnames.rs's
// parse_registry_ref_holds_the_part_invariants unit test for the same
// invariant pinned by hand.
fuzz_target!(|data: &[u8]| {
    let Ok(s) = std::str::from_utf8(data) else {
        return;
    };
    llmman::shortnames::fuzz_check_parse_registry_ref(s);
});
