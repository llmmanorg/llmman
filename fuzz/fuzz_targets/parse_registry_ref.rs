#![no_main]

use libfuzzer_sys::fuzz_target;

// Fuzzes shortnames::parse_registry_ref (private to the parent crate) via
// the fuzzing-feature-gated wrapper fuzz_check_parse_registry_ref, which
// panics whenever a field of a successful parse violates its own grammar:
// a ".." or over-length part, a byte outside a name part's allowlist, a
// malformed host port, a non-algo:hex digest, or an unknown scheme. See
// src/shortnames.rs's parse_registry_ref_holds_the_part_invariants and
// parse_registry_ref_oracle_holds_on_the_seed_corpus unit tests for the
// same oracle pinned against fixed inputs and this seed corpus.
fuzz_target!(|data: &[u8]| {
    let Ok(s) = std::str::from_utf8(data) else {
        return;
    };
    llmman::shortnames::fuzz_check_parse_registry_ref(s);
});
