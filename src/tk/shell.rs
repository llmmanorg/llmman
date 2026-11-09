//! Reading a shell command line just far enough to know whether it is one
//! plain command, and what its words are.

/// `word` as one shell word, safe to paste into a command line.
pub(crate) fn quote(word: &str) -> String {
    format!("'{}'", word.replace('\'', r"'\''"))
}

/// `command` without the blanks a shell ignores at its ends. Not
/// `str::trim`: a Unicode space (U+00A0) is part of a word to a shell.
pub(crate) fn trim(command: &str) -> &str {
    command.trim_matches([' ', '\t', '\n'])
}

/// The words of `command` when it is exactly one simple command: words,
/// quotes and backslash escapes, and nothing a shell would act on beyond
/// that. `None` for pipes, lists, redirections, subshells, substitutions
/// and expansions (including unquoted `*`, `?` and `[`), heredocs,
/// comments, newlines between commands, leading variable assignments, and
/// anything unbalanced.
///
/// The words are read as typed (quotes removed, no expansion), so they are
/// for deciding what the command is; the command is still run by the shell
/// from its original text.
pub(crate) fn simple_words(command: &str) -> Option<Vec<String>> {
    let mut words: Vec<String> = Vec::new();
    let mut word = String::new();
    let mut in_word = false;
    let mut chars = trim(command).chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            ' ' | '\t' => {
                if in_word {
                    words.push(std::mem::take(&mut word));
                    in_word = false;
                }
            }
            '\'' => {
                in_word = true;
                loop {
                    match chars.next()? {
                        '\'' => break,
                        c => word.push(c),
                    }
                }
            }
            '"' => {
                in_word = true;
                loop {
                    match chars.next()? {
                        '"' => break,
                        '$' | '`' => return None,
                        '\\' => match chars.next()? {
                            '\n' => return None,
                            c @ ('$' | '`' | '"' | '\\') => word.push(c),
                            c => {
                                word.push('\\');
                                word.push(c);
                            }
                        },
                        c => word.push(c),
                    }
                }
            }
            '\\' => match chars.next()? {
                '\n' => return None,
                c => {
                    in_word = true;
                    word.push(c);
                }
            },
            '\n' | '\r' | '|' | '&' | ';' | '<' | '>' | '(' | ')' | '$' | '`' | '{' | '}' => {
                return None;
            }
            '#' if !in_word => return None,
            // A glob: the shell would put the files it matches in the words.
            '*' | '?' | '[' => return None,
            c => {
                in_word = true;
                word.push(c);
            }
        }
    }
    if in_word {
        words.push(word);
    }
    // `NAME=value cmd` sets a variable for the command.
    let assignment = words.first().is_some_and(|w| {
        w.split_once('=').is_some_and(|(name, _)| {
            !name.is_empty()
                && !name.starts_with(|c: char| c.is_ascii_digit())
                && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        })
    });
    (!words.is_empty() && !assignment).then_some(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(c: &str) -> Option<Vec<String>> {
        simple_words(c)
    }

    #[test]
    fn plain_and_quoted_words_are_split_as_a_shell_would() {
        assert_eq!(
            words("git status"),
            Some(vec!["git".into(), "status".into()])
        );
        assert_eq!(
            words("  git   commit -m 'fix: a b' \"c d\"  "),
            Some(vec![
                "git".into(),
                "commit".into(),
                "-m".into(),
                "fix: a b".into(),
                "c d".into()
            ])
        );
        assert_eq!(
            words(r#"grep -rn "a\|b" src"#),
            Some(vec![
                "grep".into(),
                "-rn".into(),
                r"a\|b".into(),
                "src".into()
            ])
        );
        assert_eq!(
            words(r"git add my\ file"),
            Some(vec!["git".into(), "add".into(), "my file".into()])
        );
        assert_eq!(words("echo ''"), Some(vec!["echo".into(), "".into()]));
        // Operators and `$` are plain inside single quotes.
        assert_eq!(
            words("rg 'a|b;c$' ."),
            Some(vec!["rg".into(), "a|b;c$".into(), ".".into()])
        );
        // A newline inside quotes is part of the word.
        assert_eq!(
            words("git commit -m 'one\ntwo'"),
            Some(vec![
                "git".into(),
                "commit".into(),
                "-m".into(),
                "one\ntwo".into()
            ])
        );
    }

    #[test]
    fn anything_a_shell_would_act_on_is_refused() {
        for command in [
            "git status | head",
            "git status > out",
            "git status 2>&1",
            "git status < in",
            "git status && git diff",
            "git status || true",
            "git status; git diff",
            "git status &",
            "(git status)",
            "{ git status; }",
            "git log $(git rev-parse HEAD)",
            "git log `git rev-parse HEAD`",
            "git log $HEAD",
            "git log \"$HEAD\"",
            "git log \"`x`\"",
            "git status\ngit diff",
            "git status # why",
            "git diff *",
            "git diff --[o]utput=x",
            "git diff -- file?.rs",
            "git commit -m 'unterminated",
            "git commit -m \"unterminated",
            "git status \\",
            "git commit -F - <<EOF\nmsg\nEOF",
            "FOO=1 git status",
            "_a1=x cargo test",
            "",
            "   ",
        ] {
            assert_eq!(words(command), None, "{command:?}");
        }
        // Quoted or escaped, a glob character is a plain one.
        assert_eq!(
            words(r"git diff -- '*.rs' a\* "),
            Some(vec![
                "git".into(),
                "diff".into(),
                "--".into(),
                "*.rs".into(),
                "a*".into()
            ])
        );
        // Not an assignment: the word is not a variable name.
        assert!(words("a-b=c x").is_some());
        assert!(words("git log --format=%H").is_some());
    }

    #[test]
    fn only_the_blanks_a_shell_ignores_are_trimmed() {
        assert_eq!(trim(" \t\ngit status \n"), "git status");
        // U+00A0 is not a blank to a shell: it is part of the word.
        assert_eq!(trim("git diff -- notes\u{a0}"), "git diff -- notes\u{a0}");
        assert_eq!(
            words("git diff -- notes\u{a0}"),
            Some(vec![
                "git".into(),
                "diff".into(),
                "--".into(),
                "notes\u{a0}".into()
            ])
        );
    }

    #[cfg(unix)]
    #[test]
    fn quote_round_trips_through_a_real_shell() {
        for word in [
            "plain",
            "with space",
            "it's",
            "a'b'c",
            "$HOME `x` \"y\"",
            "",
            "\\",
        ] {
            let quoted = quote(word);
            let out = std::process::Command::new("sh")
                .arg("-c")
                .arg(format!("printf %s {quoted}"))
                .output()
                .unwrap();
            assert_eq!(String::from_utf8(out.stdout).unwrap(), word, "{quoted}");
        }
    }
}
