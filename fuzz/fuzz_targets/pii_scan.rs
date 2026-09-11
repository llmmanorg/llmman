#![no_main]

use libfuzzer_sys::fuzz_target;

// Fuzzes pii::scan, which reads the body of every request a hybrid pair
// is asked to route — arbitrary text from an arbitrary client, on the
// path that decides whether that text may leave the machine. The oracle
// is the contract every caller relies on: spans are real, in-bounds,
// non-overlapping character ranges of the input, in order. A span that
// is none of those would make a redacting caller cut the wrong bytes,
// or panic.
fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    let mut previous_end = 0;
    for span in llmman::pii::scan(text) {
        assert!(span.start < span.end, "empty or reversed span: {span:?}");
        assert!(span.end <= text.len(), "span past the end: {span:?}");
        assert!(
            span.start >= previous_end,
            "span overlaps the one before it: {span:?}"
        );
        // Panics unless both ends fall on character boundaries, which is
        // the property a caller slicing by these offsets depends on.
        let _ = &text[span.start..span.end];
        previous_end = span.end;
    }
});
