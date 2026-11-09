//! Text clean-up shared by every filter, and the generic fallback.

/// More lines than this are cut to a head and a tail.
const MAX_LINES: usize = 600;
const HEAD_LINES: usize = 350;
const TAIL_LINES: usize = 200;
/// A single line longer than this (in chars) is cut.
const MAX_LINE_CHARS: usize = 2000;
/// Identical consecutive lines are folded from this many on.
const MIN_REPEATS: usize = 3;

/// `text` without terminal escape sequences (colors, cursor movement,
/// window titles).
pub(crate) fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\x1b' {
            out.push(c);
            continue;
        }
        match chars.next() {
            // CSI: parameters, then one final byte in `@`..=`~`.
            Some('[') => {
                for c in chars.by_ref() {
                    if ('\x40'..='\x7e').contains(&c) {
                        break;
                    }
                }
            }
            // OSC: up to BEL, or ESC `\`.
            Some(']') => {
                while let Some(c) = chars.next() {
                    if c == '\x07' {
                        break;
                    }
                    if c == '\x1b' {
                        chars.next();
                        break;
                    }
                }
            }
            // Any other escape is the ESC and one more char.
            _ => {}
        }
    }
    out
}

/// Progress bars redraw one line with `\r`; only what the last redraw left
/// on screen is kept. `\r\n` is a plain line end.
pub(crate) fn collapse_redraws(text: &str) -> String {
    if !text.contains('\r') {
        return text.to_string();
    }
    let text = text.replace("\r\n", "\n");
    let mut out = String::with_capacity(text.len());
    for (i, line) in text.split('\n').enumerate() {
        if i > 0 {
            out.push('\n');
        }
        let shown = line.rsplit('\r').find(|s| !s.is_empty()).unwrap_or("");
        out.push_str(shown);
    }
    out
}

/// Escape sequences and redraws removed; nothing else changes.
pub(crate) fn clean(text: &str) -> String {
    collapse_redraws(&strip_ansi(text))
}

/// A regex compiled on first use. `pattern` is a literal in the caller, so
/// it is known to compile.
pub(crate) fn cached(
    cell: &'static std::sync::OnceLock<regex::Regex>,
    pattern: &str,
) -> &'static regex::Regex {
    cell.get_or_init(|| regex::Regex::new(pattern).expect("literal pattern"))
}

/// The marker for lines a filter left out.
pub(crate) fn omitted(n: usize, what: &str) -> String {
    format!("... {n} {what} omitted ...")
}

/// Lines a filter drops, counted, with the marker standing where the first
/// one was.
#[derive(Default)]
pub(crate) struct Dropped {
    slot: Option<usize>,
    n: usize,
}

impl Dropped {
    /// Records one dropped line, and reserves the marker's place in `out`.
    pub(crate) fn one(&mut self, out: &mut Vec<String>) {
        if self.slot.is_none() {
            self.slot = Some(out.len());
            out.push(String::new());
        }
        self.n += 1;
    }

    /// Writes the marker (`... N <what> omitted ...`) into its place.
    pub(crate) fn finish(self, out: &mut [String], what: &str) {
        if let Some(slot) = self.slot {
            out[slot] = omitted(self.n, what);
        }
    }
}

impl Dropped {
    /// Closes the run counted so far: its marker is written and counting
    /// starts again, so the next run gets a marker of its own.
    pub(crate) fn finish_run(&mut self, out: &mut [String], what: &str) {
        std::mem::take(self).finish(out, what);
    }
}

/// `text` without the lines `is_noise` picks, counted in one marker where
/// the first stood.
pub(crate) fn drop_lines(text: &str, what: &str, is_noise: impl Fn(&str) -> bool) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut dropped = Dropped::default();
    for line in text.lines() {
        if is_noise(line) {
            dropped.one(&mut out);
        } else {
            out.push(line.to_string());
        }
    }
    dropped.finish(&mut out, what);
    join(&out)
}

/// What every command gets when no filter claims it: over-long lines cut,
/// runs of blank lines folded to one, runs of identical lines folded to one
/// with a count, and a long output reduced to its head and tail with an
/// explicit marker. A final newline is added when there is text.
pub(crate) fn squeeze(text: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    let mut prev_blank = false;
    // The line being repeated, and how many times it has been seen.
    let mut run: Option<(String, usize)> = None;
    let flush = |run: &mut Option<(String, usize)>, lines: &mut Vec<String>| {
        if let Some((line, n)) = run.take() {
            if n >= MIN_REPEATS {
                lines.push(format!("{line} (x{n})"));
            } else {
                lines.extend(std::iter::repeat_n(line, n));
            }
        }
    };
    for raw in text.lines() {
        let line = cut_line(raw, MAX_LINE_CHARS);
        let blank = line.trim().is_empty();
        if blank {
            flush(&mut run, &mut lines);
            if !prev_blank {
                lines.push(String::new());
            }
            prev_blank = true;
            continue;
        }
        prev_blank = false;
        match &mut run {
            Some((last, n)) if *last == line => *n += 1,
            _ => {
                flush(&mut run, &mut lines);
                run = Some((line, 1));
            }
        }
    }
    flush(&mut run, &mut lines);
    limit(lines)
}

/// What a filter's output gets: only the bounds. Lines are kept exactly as
/// the filter left them, since a diff or a test report has whitespace and
/// repeats that mean something.
pub(crate) fn cap(text: &str) -> String {
    limit(text.lines().map(|l| cut_line(l, MAX_LINE_CHARS)).collect())
}

/// `lines` without trailing blank lines, reduced to a head and a tail with
/// a marker when there are too many, as text.
fn limit(mut lines: Vec<String>) -> String {
    while lines.last().is_some_and(|l| l.trim().is_empty()) {
        lines.pop();
    }
    if lines.len() > MAX_LINES {
        let omitted_n = lines.len() - HEAD_LINES - TAIL_LINES;
        let tail = lines.split_off(lines.len() - TAIL_LINES);
        lines.truncate(HEAD_LINES);
        lines.push(omitted(omitted_n, "lines"));
        lines.extend(tail);
    }
    join(&lines)
}

/// `line` cut to `max` chars, with a note of how many went.
pub(crate) fn cut_line(line: &str, max: usize) -> String {
    let total = line.chars().count();
    if total <= max {
        return line.to_string();
    }
    let kept: String = line.chars().take(max).collect();
    format!("{kept} ... [{} more chars]", total - max)
}

/// `lines` as text, newline-terminated unless empty.
pub(crate) fn join<S: AsRef<str>>(lines: &[S]) -> String {
    let mut out = String::new();
    for line in lines {
        out.push_str(line.as_ref());
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ansi_colors_titles_and_cursor_moves_are_stripped() {
        assert_eq!(strip_ansi("\x1b[31mred\x1b[0m plain"), "red plain");
        assert_eq!(strip_ansi("a\x1b[2K\x1b[1Gb"), "ab");
        assert_eq!(strip_ansi("\x1b]0;title\x07text"), "text");
        assert_eq!(
            strip_ansi("\x1b]8;;http://x\x1b\\link\x1b]8;;\x1b\\"),
            "link"
        );
        assert_eq!(strip_ansi("no escapes é"), "no escapes é");
        // An escape cut off by the end of the text is dropped, not kept.
        assert_eq!(strip_ansi("tail\x1b[3"), "tail");
    }

    #[test]
    fn progress_redraws_keep_only_the_last_state() {
        assert_eq!(collapse_redraws("10%\r50%\r100%\ndone\n"), "100%\ndone\n");
        assert_eq!(collapse_redraws("a\r\nb\r\n"), "a\nb\n");
        assert_eq!(collapse_redraws("x\r"), "x");
        assert_eq!(collapse_redraws("plain\n"), "plain\n");
    }

    #[test]
    fn repeated_lines_fold_with_a_count_from_three_on() {
        assert_eq!(squeeze("a\na\nb\n"), "a\na\nb\n");
        assert_eq!(squeeze("a\na\na\nb\n"), "a (x3)\nb\n");
        assert_eq!(squeeze("x\n".repeat(50).as_str()), "x (x50)\n");
        // Not consecutive: kept as they are.
        assert_eq!(squeeze("a\nb\na\nb\na\nb\n"), "a\nb\na\nb\na\nb\n");
    }

    #[test]
    fn blank_runs_fold_to_one_and_trailing_blanks_go() {
        assert_eq!(squeeze("a\n\n\n\n\nb\n\n\n"), "a\n\nb\n");
        assert_eq!(squeeze("a\n  \n\t\nb"), "a\n\nb\n");
        assert_eq!(squeeze(""), "");
        assert_eq!(squeeze("\n\n"), "");
    }

    #[test]
    fn a_long_output_keeps_its_head_and_tail_with_a_marker() {
        let text: String = (0..2000).map(|i| format!("line {i}\n")).collect();
        let out = squeeze(&text);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), HEAD_LINES + 1 + TAIL_LINES);
        assert_eq!(lines[0], "line 0");
        assert_eq!(lines[HEAD_LINES - 1], format!("line {}", HEAD_LINES - 1));
        assert_eq!(
            lines[HEAD_LINES],
            format!("... {} lines omitted ...", 2000 - HEAD_LINES - TAIL_LINES)
        );
        assert_eq!(*lines.last().unwrap(), "line 1999");
    }

    #[test]
    fn cap_keeps_every_line_as_it_is_and_only_bounds_the_output() {
        let text = "a\na\na\n \n \nb   \n\n\n";
        assert_eq!(cap(text), "a\na\na\n \n \nb   \n");
        let long: String = (0..1000).map(|i| format!("l{i}\n")).collect();
        let out = cap(&long);
        assert_eq!(out.lines().count(), HEAD_LINES + 1 + TAIL_LINES);
        assert!(out.contains("... 450 lines omitted ..."));
        assert!(cap(&format!("{}\n", "z".repeat(MAX_LINE_CHARS + 5))).contains("[5 more chars]"));
    }

    #[test]
    fn an_output_at_the_limit_is_left_whole() {
        let text: String = (0..MAX_LINES).map(|i| format!("l{i}\n")).collect();
        assert_eq!(squeeze(&text), text);
    }

    #[test]
    fn an_over_long_line_is_cut_with_a_count() {
        let long = "x".repeat(MAX_LINE_CHARS + 10);
        let out = squeeze(&format!("{long}\n"));
        assert!(out.starts_with(&"x".repeat(MAX_LINE_CHARS)));
        assert!(out.contains("[10 more chars]"), "{out}");
        // Cuts on chars, never inside one.
        let wide = "é".repeat(MAX_LINE_CHARS + 1);
        assert!(squeeze(&wide).contains("[1 more chars]"));
    }
}
