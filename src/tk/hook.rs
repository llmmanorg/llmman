//! The Claude Code `PreToolUse` hook behind `launch claude --optimizer tk`.
//!
//! For a Bash call that is one plain command with a filter in this tree, the
//! hook answers with `updatedInput`: the same command line behind
//! `<llmman> tk `. Only text is added in front, so quoting survives. For
//! anything else, and on any error, it says nothing and exits 0.
//!
//! Claude Code checks permission rules against the rewritten command, so:
//!
//! - A program is left alone when a deny or ask rule in a settings file
//!   could apply to it, or when a settings file cannot be read.
//! - The hook never approves a command itself, except that an allow rule the
//!   user wrote carries over, and so do plain `git status|diff|log`, which
//!   Claude Code runs unprompted.
//!
//! Rules from managed policy, MDM or `--disallowedTools` are not visible.

use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use super::shell::{quote, simple_words, trim};

/// The most of the hook's input that is read.
const MAX_INPUT: u64 = 1 << 20;
/// A command longer than this (in bytes), before or after the rewrite, is
/// left alone: Claude Code always asks about one (2.1.214 on), and an
/// `allow` from here would skip that.
const MAX_COMMAND: usize = 10_000;

/// What `--settings` is given to register the hook: one `PreToolUse` entry
/// for the Bash tool that runs `<exe> tk hook`.
pub(crate) fn settings_json(exe: &str) -> String {
    json!({
        "hooks": {
            "PreToolUse": [{
                "matcher": "Bash",
                "hooks": [{
                    "type": "command",
                    "command": format!("{} tk hook", quote(exe)),
                    "timeout": 10
                }]
            }]
        }
    })
    .to_string()
}

/// `llmman tk hook`: reads the hook input on stdin, and prints the answer
/// when there is one. Never fails and never panics out.
pub fn run() {
    std::panic::set_hook(Box::new(|_| {}));
    let _ = std::panic::catch_unwind(|| {
        // Off means off: no rewrite, so nothing about the call changes.
        if super::is_off(std::env::var("LLMMAN_TK").ok().as_deref()) {
            return;
        }
        let mut input = String::new();
        if std::io::stdin()
            .take(MAX_INPUT)
            .read_to_string(&mut input)
            .is_err()
        {
            return;
        }
        let Some(exe) = std::env::current_exe()
            .ok()
            .filter(|p| p.is_file())
            .and_then(|p| p.to_str().map(String::from))
        else {
            return;
        };
        if let Some(answer) = answer(&input, &exe, &|cwd| load_rules(&settings_files(cwd)?)) {
            println!("{answer}");
        }
    });
}

/// The permission rules of the settings files, by decision.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Rules {
    deny: Vec<String>,
    ask: Vec<String>,
    allow: Vec<String>,
}

/// The settings files Claude Code reads rules from: the user's, the
/// project's (shared and local), and the managed policy files. `None` when
/// a managed policy directory exists but cannot be listed.
fn settings_files(cwd: Option<&Path>) -> Option<Vec<PathBuf>> {
    let mut files = Vec::new();
    let user = std::env::var_os("CLAUDE_CONFIG_DIR")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".claude")));
    files.extend(user.map(|d| d.join("settings.json")));
    let mut projects: Vec<PathBuf> = Vec::new();
    projects.extend(std::env::var_os("CLAUDE_PROJECT_DIR").map(PathBuf::from));
    projects.extend(cwd.map(Path::to_path_buf));
    projects.dedup();
    for project in projects {
        files.push(project.join(".claude/settings.json"));
        files.push(project.join(".claude/settings.local.json"));
    }
    for managed in [
        "/etc/claude-code",
        "/Library/Application Support/ClaudeCode",
    ] {
        let managed = Path::new(managed);
        files.push(managed.join("managed-settings.json"));
        files.extend(dropins(&managed.join("managed-settings.d"))?);
    }
    Some(files)
}

/// The `.json` files in `dir`, sorted; none when it does not exist. `None`
/// when it cannot be listed, so the rules are unknown.
fn dropins(dir: &Path) -> Option<Vec<PathBuf>> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Some(Vec::new()),
        Err(_) => return None,
    };
    let mut files = Vec::new();
    for entry in entries {
        let path = entry.ok()?.path();
        if path.extension().is_some_and(|x| x == "json") {
            files.push(path);
        }
    }
    files.sort();
    Some(files)
}

/// The rules in `files`, which need not exist. `None` when one that does
/// exist cannot be read or understood: then nothing is known about what
/// Claude Code will enforce.
fn load_rules(files: &[PathBuf]) -> Option<Rules> {
    let mut rules = Rules::default();
    for file in files {
        let text = match std::fs::read_to_string(file) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(_) => return None,
        };
        let doc: Value = serde_json::from_str(&text).ok()?;
        let Some(permissions) = doc.as_object()?.get("permissions") else {
            continue;
        };
        for (key, into) in [
            ("deny", &mut rules.deny),
            ("ask", &mut rules.ask),
            ("allow", &mut rules.allow),
        ] {
            match permissions.get(key) {
                None => {}
                Some(list) => into.extend(
                    list.as_array()?
                        .iter()
                        .filter_map(Value::as_str)
                        .map(String::from),
                ),
            }
        }
    }
    Some(rules)
}

/// A permission rule as `(tool, content)`: `Bash(git push:*)` is
/// `("Bash", Some("git push:*"))`, `Bash` is `("Bash", None)`.
fn split_rule(rule: &str) -> (&str, Option<&str>) {
    match rule.trim().split_once('(') {
        Some((tool, rest)) => (tool.trim(), rest.trim_end().strip_suffix(')')),
        None => (rule.trim(), None),
    }
}

/// The part of a Bash rule before its first space, without a `:*` suffix:
/// the program it is about. A `*` left in it is a wildcard.
fn rule_program(content: &str) -> &str {
    let first = content.split_whitespace().next().unwrap_or("");
    first.trim_end_matches(":*")
}

impl Rules {
    /// Whether a deny or ask rule could apply to `program`, or a rule is
    /// about files in a way a wrapped command could dodge. Deliberately
    /// wider than Claude Code's own matching: a doubt is a yes.
    fn restricts(&self, program: &str) -> bool {
        self.deny.iter().chain(&self.ask).any(|rule| {
            let (tool, content) = split_rule(rule);
            match tool {
                "Bash" => {
                    let Some(content) = content else { return true };
                    let named = rule_program(content);
                    named.is_empty()
                        || named.contains(['*', '=', '?', '['])
                        || super::program_name(named) == program
                }
                "Read" | "Grep" | "Glob" | "Edit" | "Write" | "NotebookEdit" => true,
                _ => false,
            }
        })
    }

    /// Whether an allow rule written for `command` covers it: `Bash`, a
    /// `Bash(<command>)`, or a `Bash(<prefix>:*)` / `Bash(<prefix> *)`
    /// whose prefix ends where a word does.
    fn allows(&self, command: &str) -> bool {
        self.allow.iter().any(|rule| match split_rule(rule) {
            ("Bash", None) => true,
            ("Bash", Some(content)) => {
                let content = content.trim();
                if content == command {
                    return true;
                }
                let prefix = content
                    .strip_suffix(":*")
                    .or_else(|| content.strip_suffix(" *"));
                prefix.is_some_and(|p| {
                    !p.is_empty()
                        && !p.contains('*')
                        && command
                            .strip_prefix(p)
                            .is_some_and(|rest| rest.is_empty() || rest.starts_with([' ', '\t']))
                })
            }
            _ => false,
        })
    }
}

/// `git status`, `diff` and `log` run without a prompt in Claude Code's
/// default mode. Only a command that is that subcommand and arguments that
/// stay inside the working tree counts: no option before the subcommand
/// (`-c` and `-C` can run programs or read elsewhere), and no path that
/// leaves the directory.
fn is_plain_read_only(words: &[String]) -> bool {
    let program = words.first().map(|w| super::program_name(w));
    let sub = words.get(1).map(String::as_str);
    program == Some("git")
        && words[0] == "git"
        && matches!(sub, Some("status" | "diff" | "log"))
        && words[2..].iter().all(|a| {
            !a.starts_with('/')
                && !a.starts_with('~')
                && !a.split('/').any(|part| part == "..")
                && !a.starts_with("--output")
                && !a.starts_with("--ext-diff")
                && !a.starts_with("--open-files-in-pager")
        })
}

/// The hook's stdout for `input`, or `None` to say nothing. `rules` reads
/// the permission rules given the call's working directory.
pub(crate) fn answer(
    input: &str,
    exe: &str,
    rules: &dyn Fn(Option<&Path>) -> Option<Rules>,
) -> Option<String> {
    let call: Value = serde_json::from_str(input).ok()?;
    if call
        .get("hook_event_name")
        .is_some_and(|e| e != "PreToolUse")
        || call.get("tool_name")? != "Bash"
    {
        return None;
    }
    let tool_input = call.get("tool_input")?.as_object()?;
    let raw = tool_input.get("command")?.as_str()?;
    if raw.len() > MAX_COMMAND {
        return None;
    }
    let command = trim(raw);
    let words = simple_words(command)?;
    if !super::is_filtered(&words) {
        return None;
    }
    let program = super::program_name(&words[0]);
    let cwd = call.get("cwd").and_then(Value::as_str).map(Path::new);
    let rules = rules(cwd)?;
    if rules.restricts(program) {
        return None;
    }
    let rewritten = format!("{} tk {command}", quote(exe));
    if rewritten.len() > MAX_COMMAND {
        return None;
    }
    let mut updated = tool_input.clone();
    updated.insert("command".into(), Value::String(rewritten));
    let mut output = json!({
        "hookEventName": "PreToolUse",
        "updatedInput": updated,
    });
    if rules.allows(command) || is_plain_read_only(&words) {
        output["permissionDecision"] = json!("allow");
        output["permissionDecisionReason"] =
            json!("llmman tk: the rewritten command keeps the decision for the original");
    }
    Some(json!({ "hookSpecificOutput": output }).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXE: &str = "/opt/llm man/llmman";

    fn call(tool: &str, command: &str) -> String {
        json!({
            "session_id": "s",
            "cwd": "/work/app",
            "hook_event_name": "PreToolUse",
            "tool_name": tool,
            "tool_input": { "command": command, "description": "d", "timeout": 5000 },
        })
        .to_string()
    }

    fn no_rules(_: Option<&Path>) -> Option<Rules> {
        Some(Rules::default())
    }

    fn with_rules(deny: &[&str], ask: &[&str], allow: &[&str]) -> Rules {
        let own = |l: &[&str]| l.iter().map(|s| s.to_string()).collect();
        Rules {
            deny: own(deny),
            ask: own(ask),
            allow: own(allow),
        }
    }

    fn parsed(answer: Option<String>) -> Value {
        serde_json::from_str(&answer.expect("an answer")).unwrap()
    }

    #[test]
    fn a_supported_simple_command_is_wrapped_with_the_original_text_intact() {
        let out = parsed(answer(&call("Bash", "cargo test --lib"), EXE, &no_rules));
        let hook = &out["hookSpecificOutput"];
        assert_eq!(hook["hookEventName"], "PreToolUse");
        assert_eq!(
            hook["updatedInput"]["command"],
            "'/opt/llm man/llmman' tk cargo test --lib"
        );
        // The rest of the input is carried over: updatedInput replaces it all.
        assert_eq!(hook["updatedInput"]["description"], "d");
        assert_eq!(hook["updatedInput"]["timeout"], 5000);
        // Nothing is approved on the user's behalf.
        assert!(hook.get("permissionDecision").is_none());
    }

    #[test]
    fn quoting_in_the_original_survives_byte_for_byte() {
        for command in [
            r#"git commit -m "fix: a 'quoted' thing""#,
            r#"git commit -m 'it'\''s'"#,
            r#"grep -rn "a\|b" src"#,
            "git commit -m 'one\ntwo'",
            "git log --author='A B' -n 3",
        ] {
            let out = parsed(answer(&call("Bash", command), EXE, &no_rules));
            let rewritten = out["hookSpecificOutput"]["updatedInput"]["command"]
                .as_str()
                .unwrap()
                .to_string();
            assert_eq!(rewritten, format!("'/opt/llm man/llmman' tk {command}"));
        }
    }

    #[test]
    fn what_cannot_be_wrapped_safely_is_left_alone() {
        for command in [
            "git status | head",
            "git status && git diff",
            "git diff > out.patch",
            "git log $(git rev-parse HEAD)",
            "git commit -F - <<EOF\nmsg\nEOF",
            "FOO=1 cargo test",
            "(cd sub && cargo test)",
            "cd sub; cargo test",
            "echo hi",
            "ls -la",
            "cat Cargo.toml",
            "git rev-parse HEAD",
            "git status --porcelain",
            "git log --oneline",
            "cargo build --message-format=json",
            "rg --json foo",
            "grep -c x f",
            "grep -o x f",
            "git commit",
            "",
        ] {
            assert_eq!(
                answer(&call("Bash", command), EXE, &no_rules),
                None,
                "{command:?}"
            );
        }
    }

    #[test]
    fn an_already_wrapped_command_is_not_wrapped_again() {
        for command in [
            "llmman tk cargo test",
            "'/opt/llm man/llmman' tk cargo test",
            "/usr/bin/llmman tk git status",
        ] {
            assert_eq!(
                answer(&call("Bash", command), EXE, &no_rules),
                None,
                "{command}"
            );
        }
    }

    #[test]
    fn other_tools_and_malformed_input_get_no_answer() {
        assert_eq!(answer(&call("Read", "cargo test"), EXE, &no_rules), None);
        assert_eq!(answer(&call("Edit", "git status"), EXE, &no_rules), None);
        for input in [
            "",
            "not json",
            "[]",
            "{}",
            r#"{"tool_name":"Bash"}"#,
            r#"{"tool_name":"Bash","tool_input":{}}"#,
            r#"{"tool_name":"Bash","tool_input":{"command":7}}"#,
            r#"{"tool_name":"Bash","tool_input":"git status"}"#,
            r#"{"tool_name":"Bash","tool_input":{"command":"git status"},"hook_event_name":"PostToolUse"}"#,
        ] {
            assert_eq!(answer(input, EXE, &no_rules), None, "{input:?}");
        }
        // A command Claude Code always asks about is not touched, nor is one
        // the rewrite would make so.
        let long = format!("git commit -m '{}'", "x".repeat(MAX_COMMAND));
        assert_eq!(answer(&call("Bash", &long), EXE, &no_rules), None);
        let padded = format!("git status{}", " ".repeat(MAX_COMMAND));
        assert_eq!(answer(&call("Bash", &padded), EXE, &no_rules), None);
        let near = format!("git commit -m '{}'", "x".repeat(MAX_COMMAND - 20));
        assert!(near.len() <= MAX_COMMAND);
        assert_eq!(answer(&call("Bash", &near), EXE, &no_rules), None);
    }

    #[test]
    fn a_unicode_space_at_the_end_is_part_of_the_command() {
        let command = "git diff -- notes\u{a0}";
        let out = parsed(answer(&call("Bash", command), EXE, &no_rules));
        assert_eq!(
            out["hookSpecificOutput"]["updatedInput"]["command"],
            format!("'/opt/llm man/llmman' tk {command}")
        );
        // Blanks a shell ignores are not.
        let out = parsed(answer(&call("Bash", "\n git status \n"), EXE, &no_rules));
        assert_eq!(
            out["hookSpecificOutput"]["updatedInput"]["command"],
            "'/opt/llm man/llmman' tk git status"
        );
    }

    #[test]
    fn a_glob_is_left_for_claude_code_to_judge() {
        for command in ["git diff *", "git diff --[o]utput=x", "git log -- src/?.rs"] {
            assert_eq!(
                answer(&call("Bash", command), EXE, &no_rules),
                None,
                "{command}"
            );
        }
        assert!(answer(&call("Bash", "git diff -- '*.rs'"), EXE, &no_rules).is_some());
    }

    #[test]
    fn unreadable_rules_mean_no_rewrite() {
        let none = |_: Option<&Path>| None;
        assert_eq!(answer(&call("Bash", "cargo test"), EXE, &none), None);
    }

    #[test]
    fn a_deny_or_ask_rule_that_could_apply_keeps_the_program_unwrapped() {
        let run = |rules: Rules, command: &str| {
            answer(&call("Bash", command), EXE, &move |_| {
                Some(Rules {
                    deny: rules.deny.clone(),
                    ask: rules.ask.clone(),
                    allow: rules.allow.clone(),
                })
            })
        };
        for (deny, ask) in [
            (&["Bash(git push:*)"][..], &[][..]),
            (&["Bash(git push *)"], &[]),
            (&["Bash(git:*)"], &[]),
            (&["Bash(/usr/bin/git push:*)"], &[]),
            (&[], &["Bash(git commit:*)"]),
            (&["Bash"], &[]),
            (&["Bash(*)"], &[]),
            (&["Bash(*push*)"], &[]),
            (&["Bash(g*)"], &[]),
            (&["Bash(gi* status)"], &[]),
            (&["Bash(git*)"], &[]),
            (&["Bash(FOO=1 git:*)"], &[]),
            (&["Read(./.env)"], &[]),
            (&["Edit(src/**)"], &[]),
        ] {
            assert_eq!(
                run(with_rules(deny, ask, &[]), "git status"),
                None,
                "{deny:?} {ask:?}"
            );
        }
        // A rule about another program, or another tool, does not matter.
        for (deny, ask) in [
            (&["Bash(rm:*)"][..], &[][..]),
            (&["Bash(curl:*)"], &["Bash(npm publish:*)"]),
            (&["WebFetch(domain:example.com)"], &[]),
            (&["mcp__x__y"], &[]),
        ] {
            assert!(
                run(with_rules(deny, ask, &[]), "git status").is_some(),
                "{deny:?} {ask:?}"
            );
        }
    }

    #[test]
    fn an_allow_rule_for_the_original_carries_over_as_an_allow() {
        let decision = |allow: &[&str], command: &str| {
            let allow: Vec<String> = allow.iter().map(|s| s.to_string()).collect();
            let out = parsed(answer(&call("Bash", command), EXE, &move |_| {
                Some(Rules {
                    deny: Vec::new(),
                    ask: Vec::new(),
                    allow: allow.clone(),
                })
            }));
            out["hookSpecificOutput"]
                .get("permissionDecision")
                .map(|d| d.as_str().unwrap().to_string())
        };
        assert_eq!(
            decision(&["Bash(cargo test:*)"], "cargo test --lib"),
            Some("allow".into())
        );
        assert_eq!(
            decision(&["Bash(cargo test *)"], "cargo test"),
            Some("allow".into())
        );
        assert_eq!(
            decision(&["Bash(cargo test)"], "cargo test"),
            Some("allow".into())
        );
        assert_eq!(decision(&["Bash"], "cargo test"), Some("allow".into()));
        // The prefix has to end at a word: `cargo t` does not allow `cargo test`.
        assert_eq!(decision(&["Bash(cargo t:*)"], "cargo test"), None);
        assert_eq!(decision(&["Bash(cargo test)"], "cargo test --lib"), None);
        assert_eq!(decision(&["Bash(go test:*)"], "cargo test"), None);
        assert_eq!(decision(&["Bash(cargo*:*)"], "cargo test"), None);
        assert_eq!(decision(&["Read(x)"], "cargo test"), None);
        assert_eq!(decision(&[], "cargo test"), None);
    }

    #[test]
    fn plain_read_only_git_is_allowed_and_nothing_else_is() {
        let decision = |command: &str| {
            let out = parsed(answer(&call("Bash", command), EXE, &no_rules));
            out["hookSpecificOutput"]
                .get("permissionDecision")
                .is_some()
        };
        for command in [
            "git status",
            "git diff",
            "git diff HEAD~1 -- src/a.rs",
            "git log -n 5",
        ] {
            assert!(decision(command), "{command}");
        }
        for command in [
            "git -C ../other status",
            "git -c core.pager=x diff",
            "git diff ../other/file",
            "git diff -- /etc/passwd",
            "git diff ~/x",
            "git diff --ext-diff",
            "git commit -m x",
            "git push",
            "git pull",
            "cargo test",
            "grep -rn x src",
            "pytest",
            "npm test",
            "docker ps",
        ] {
            assert!(!decision(command), "{command}");
        }
    }

    #[test]
    fn registered_settings_run_this_binary_for_bash_only() {
        let settings: Value = serde_json::from_str(&settings_json(EXE)).unwrap();
        let entry = &settings["hooks"]["PreToolUse"][0];
        assert_eq!(entry["matcher"], "Bash");
        assert_eq!(entry["hooks"][0]["type"], "command");
        assert_eq!(
            entry["hooks"][0]["command"],
            "'/opt/llm man/llmman' tk hook"
        );
        assert_eq!(
            settings.as_object().unwrap().len(),
            1,
            "hooks and nothing else"
        );
    }

    #[test]
    fn rules_are_read_from_every_settings_file_that_exists() {
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, body: &str| {
            let p = dir.path().join(name);
            std::fs::write(&p, body).unwrap();
            p
        };
        let user = write(
            "user.json",
            r#"{"permissions":{"deny":["Bash(git push:*)"],"allow":["Bash(cargo test:*)"],"defaultMode":"default"},"model":"x"}"#,
        );
        let project = write(
            "project.json",
            r#"{"permissions":{"ask":["Bash(npm publish:*)"]}}"#,
        );
        let none = write("none.json", r#"{"hooks":{}}"#);
        let missing = dir.path().join("absent.json");
        let rules = load_rules(&[user, project, none, missing.clone()]).unwrap();
        assert_eq!(
            rules,
            with_rules(
                &["Bash(git push:*)"],
                &["Bash(npm publish:*)"],
                &["Bash(cargo test:*)"]
            )
        );
        assert_eq!(load_rules(&[missing]).unwrap(), Rules::default());
    }

    #[test]
    fn a_settings_file_that_cannot_be_understood_means_unknown_rules() {
        let dir = tempfile::tempdir().unwrap();
        for (i, body) in ["{", "[]", r#"{"permissions":{"deny":"Bash"}}"#, ""]
            .iter()
            .enumerate()
        {
            let p = dir.path().join(format!("{i}.json"));
            std::fs::write(&p, body).unwrap();
            assert_eq!(load_rules(&[p]), None, "{body:?}");
        }
        // A directory where a file should be cannot be read either.
        let sub = dir.path().join("sub.json");
        std::fs::create_dir(&sub).unwrap();
        assert_eq!(load_rules(&[sub]), None);
    }

    #[test]
    fn a_managed_directory_that_cannot_be_listed_means_unknown_rules() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["b.json", "a.json", "c.txt"] {
            std::fs::write(dir.path().join(name), "{}").unwrap();
        }
        let listed = dropins(dir.path()).unwrap();
        let names: Vec<_> = listed.iter().map(|p| p.file_name().unwrap()).collect();
        assert_eq!(names, ["a.json", "b.json"]);
        // No directory is no rules; a file where it should be is not that.
        assert_eq!(dropins(&dir.path().join("absent")), Some(Vec::new()));
        assert_eq!(dropins(&dir.path().join("a.json")), None);
    }

    #[test]
    fn the_project_and_user_files_are_among_those_read() {
        let files = settings_files(Some(Path::new("/work/app"))).unwrap();
        assert!(files.contains(&PathBuf::from("/work/app/.claude/settings.json")));
        assert!(files.contains(&PathBuf::from("/work/app/.claude/settings.local.json")));
        assert!(files.contains(&PathBuf::from("/etc/claude-code/managed-settings.json")));
        assert!(files.iter().any(|f| f.ends_with(".claude/settings.json")));
    }
}
