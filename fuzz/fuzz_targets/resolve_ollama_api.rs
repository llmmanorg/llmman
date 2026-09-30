#![no_main]

use libfuzzer_sys::fuzz_target;

// Fuzzes shortnames::resolve_ollama_api (private to the parent crate) via
// the fuzzing-feature-gated wrapper fuzz_check_resolve_ollama_api. Two live
// checks: resolve_ollama_api() must never panic on arbitrary input, and its
// Ok/Err must match validate_reference()'s. Today both call the same
// validate_reference_parsed internally, so the agreement check cannot fail;
// it is a drift guard against a future refactor dropping that shared
// validation from either function. See src/shortnames.rs's
// fuzz_check_resolve_ollama_api doc comment, and
// resolve_ollama_api_oracle_holds_on_the_seed_corpus for the same oracle
// pinned against this seed corpus.
fuzz_target!(|data: &[u8]| {
    let Ok(s) = std::str::from_utf8(data) else {
        return;
    };
    llmman::shortnames::fuzz_check_resolve_ollama_api(s);
});
