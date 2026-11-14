//! `llmman launch --optimizer`: what an integration is launched with, besides
//! the model, to make a session cheaper.
//!
//! Each optimizer says which integrations it works with ([`Optimizer::check`],
//! run before the daemon starts) and what it adds to a launch
//! ([`Optimizer::addition`]); a launcher just applies the [`Addition`].
//! `none` adds nothing.

use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::ValueEnum;

use super::common::write_if_changed;
use super::{claude, codex, opencode};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum Optimizer {
    /// Launch as without the flag; nothing is added
    #[default]
    None,
    /// Compact the output of the shell commands the agent runs (claude and
    /// opencode)
    Tk,
    /// Ask the model for terse replies (claude, codex and opencode)
    Cvmn,
}

/// What an optimizer adds to a launch.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Addition {
    /// Arguments placed before the ones the launcher builds.
    pub args: Vec<String>,
    /// Environment variables set for the integration.
    pub env: Vec<(String, String)>,
    /// Top-level keys added to the config llmman hands opencode.
    pub config: Option<serde_json::Value>,
}

/// The integrations `tk` works with. codex is left out: its hooks need a
/// manual review, and a rewrite needs the hook to approve the command.
const TK_INTEGRATIONS: &[&str] = &["claude", "opencode"];

/// The integrations `cvmn` works with.
const CVMN_INTEGRATIONS: &[&str] = &["claude", "codex", "opencode"];

const CVMN_PROMPT: &str = include_str!("cvmn.md");

/// `__LLMMAN__` is the path of llmman.
const TK_OPENCODE_PLUGIN: &str = include_str!("opencode_tk.js");

impl Optimizer {
    /// The name `--optimizer` spells it with.
    fn name(self) -> &'static str {
        match self {
            Optimizer::None => "none",
            Optimizer::Tk => "tk",
            Optimizer::Cvmn => "cvmn",
        }
    }

    /// The integrations this optimizer works with.
    fn integrations(self) -> &'static [&'static str] {
        match self {
            Optimizer::None => &[],
            Optimizer::Tk => TK_INTEGRATIONS,
            Optimizer::Cvmn => CVMN_INTEGRATIONS,
        }
    }

    /// Refuses a launch of `integration` (a name from `llmman launch`'s list)
    /// that this optimizer cannot serve. Runs before the daemon starts and
    /// before a model is pulled; `none` accepts everything.
    pub(super) fn check(
        self,
        integration: &str,
        sandboxed: bool,
        extra_args: &[String],
    ) -> anyhow::Result<()> {
        if self == Optimizer::None {
            return Ok(());
        }
        self.ensure_integration(integration)?;
        anyhow::ensure!(
            !sandboxed,
            "--optimizer {} does not work with --sandbox.",
            self.name()
        );
        let conflict = match (self, integration) {
            (Optimizer::Tk, "claude") => claude::optimizer_conflict(extra_args),
            (Optimizer::Cvmn, "claude") => claude::cvmn_conflict(extra_args),
            (Optimizer::Cvmn, "codex") => codex::cvmn_conflict(extra_args),
            _ => None,
        };
        if let Some(why) = conflict {
            anyhow::bail!("--optimizer {} does not work with {why}", self.name());
        }
        if self == Optimizer::Tk {
            anyhow::ensure!(
                !cfg!(windows),
                "--optimizer {} is not supported on Windows.",
                self.name()
            );
            if integration == "opencode" {
                opencode::ensure_plugins()?;
            }
        }
        Ok(())
    }

    fn ensure_integration(self, integration: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.integrations().contains(&integration),
            "--optimizer {} does not work with {integration}: only {} can use it.\n\
             Launch {integration} without --optimizer, or use one of those.",
            self.name(),
            self.integrations().join(", ")
        );
        Ok(())
    }

    /// What this optimizer adds to a launch of `integration`. `none` adds
    /// nothing and looks at nothing, so a launch without the flag does no
    /// work it did not do before.
    pub(super) fn addition(self, integration: &str) -> anyhow::Result<Addition> {
        if self == Optimizer::None {
            return Ok(Addition::default());
        }
        let exe = match self {
            Optimizer::Tk => {
                std::env::current_exe().context("cannot find the path of the running llmman")?
            }
            _ => PathBuf::new(),
        };
        let dir = match integration {
            "opencode" => opencode::opencode_state_dir()?.join("llmman"),
            _ => PathBuf::new(),
        };
        self.addition_with(integration, &exe, &dir)
    }

    /// [`Optimizer::addition`] with the path of llmman and the directory for
    /// the files an integration reads.
    pub(super) fn addition_with(
        self,
        integration: &str,
        exe: &Path,
        dir: &Path,
    ) -> anyhow::Result<Addition> {
        match (self, integration) {
            (Optimizer::None, _) => Ok(Addition::default()),
            (Optimizer::Tk, "claude") => Ok(Addition {
                args: vec![
                    "--settings".into(),
                    crate::tk::hook::settings_json(utf8(exe)?),
                ],
                ..Addition::default()
            }),
            (Optimizer::Tk, "opencode") => {
                let plugin = dir.join("tk");
                let source = TK_OPENCODE_PLUGIN.replace("__LLMMAN__", &json_string(utf8(exe)?));
                write_if_changed(&plugin.join("index.js"), &source)?;
                Ok(Addition {
                    config: Some(serde_json::json!({ "plugins": [utf8(&plugin)?] })),
                    ..Addition::default()
                })
            }
            (Optimizer::Cvmn, "claude") => Ok(Addition {
                args: vec!["--append-system-prompt".into(), CVMN_PROMPT.trim().into()],
                ..Addition::default()
            }),
            (Optimizer::Cvmn, "codex") => Ok(Addition {
                args: vec![
                    "-c".into(),
                    codex::developer_instructions(CVMN_PROMPT.trim()),
                ],
                ..Addition::default()
            }),
            (Optimizer::Cvmn, "opencode") => {
                let file = dir.join("cvmn.md");
                write_if_changed(&file, CVMN_PROMPT)?;
                Ok(Addition {
                    config: Some(serde_json::json!({ "instructions": [utf8(&file)?] })),
                    ..Addition::default()
                })
            }
            (optimizer, other) => anyhow::bail!(
                "--optimizer {} does not work with {other}",
                optimizer.name()
            ),
        }
    }

    /// The line `launch` prints for `integration`, if the optimizer is on.
    pub(super) fn notice(self, integration: &str) -> Option<String> {
        match self {
            Optimizer::None => None,
            Optimizer::Tk => Some(format!(
                "[llmman] {integration}: optimizer tk (command output compacted; LLMMAN_TK=off disables)"
            )),
            Optimizer::Cvmn => Some(format!(
                "[llmman] {integration}: optimizer cvmn (terse replies requested)"
            )),
        }
    }
}

fn utf8(path: &Path) -> anyhow::Result<&str> {
    path.to_str()
        .ok_or_else(|| anyhow::anyhow!("not a valid UTF-8 path: {}", path.display()))
}

/// `text` as a JavaScript string literal.
fn json_string(text: &str) -> String {
    serde_json::to_string(text).expect("a string serializes")
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::ValueEnum;

    #[test]
    fn the_choices_are_spelled_none_tk_and_cvmn() {
        let names: Vec<String> = Optimizer::value_variants()
            .iter()
            .map(|o| o.to_possible_value().unwrap().get_name().to_string())
            .collect();
        assert_eq!(names, ["none", "tk", "cvmn"]);
        assert_eq!(Optimizer::default(), Optimizer::None);
        for o in Optimizer::value_variants() {
            assert_eq!(o.to_possible_value().unwrap().get_name(), o.name());
        }
    }

    #[test]
    fn none_works_with_everything_and_adds_nothing() {
        for name in [
            "claude",
            "opencode",
            "codex",
            "goose",
            "docker-agent",
            "nonesuch",
        ] {
            Optimizer::None.check(name, false, &[]).unwrap();
            Optimizer::None
                .check(name, true, &["--settings".into()])
                .unwrap();
            assert_eq!(Optimizer::None.addition(name).unwrap(), Addition::default());
        }
        assert_eq!(Optimizer::None.notice("claude"), None);
    }

    #[cfg(not(windows))]
    #[test]
    fn tk_works_with_claude() {
        Optimizer::Tk.check("claude", false, &[]).unwrap();
        Optimizer::Tk
            .check("claude", false, &["-p".into(), "hello".into()])
            .unwrap();
    }

    #[test]
    fn tk_names_the_integration_and_the_ones_that_work() {
        for name in ["codex", "aider", "goose", "docker-agent"] {
            let err = Optimizer::Tk
                .check(name, false, &[])
                .unwrap_err()
                .to_string();
            assert!(err.contains(&format!("does not work with {name}")), "{err}");
            assert!(err.contains("only claude, opencode can use it"), "{err}");
        }
    }

    #[test]
    fn every_integration_with_tk_is_one_the_capability_list_names() {
        // The list is the only place that says which integrations work.
        assert_eq!(TK_INTEGRATIONS, ["claude", "opencode"]);
    }

    #[test]
    fn tk_is_refused_in_a_sandbox() {
        let err = Optimizer::Tk
            .check("claude", true, &[])
            .unwrap_err()
            .to_string();
        assert!(err.contains("--sandbox"), "{err}");
    }

    #[test]
    fn tk_is_refused_beside_claude_arguments_it_cannot_honor() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        for (extra, flag) in [
            (&["--settings", "mine.json"][..], "--settings"),
            (&["--settings=mine.json"], "--settings"),
            (&["--bare"], "--bare"),
            (&["--disallowedTools", "Bash(rm:*)"], "--disallowedTools"),
            (&["--disallowed-tools=Bash(rm:*)"], "--disallowedTools"),
        ] {
            let err = Optimizer::Tk
                .check("claude", false, &args(extra))
                .unwrap_err()
                .to_string();
            assert!(err.contains(flag), "{extra:?}: {err}");
        }
        // A prompt that reads like a flag, after a second `--`, is a prompt.
        #[cfg(not(windows))]
        Optimizer::Tk
            .check("claude", false, &args(&["-p", "--", "--settings", "x"]))
            .unwrap();
    }

    #[test]
    fn tk_adds_a_settings_argument_for_claude_and_nothing_else() {
        let added = Optimizer::Tk
            .addition_with("claude", Path::new("/opt/llmman"), Path::new(""))
            .unwrap();
        assert_eq!(added.args.len(), 2);
        assert_eq!(added.args[0], "--settings");
        let settings: serde_json::Value = serde_json::from_str(&added.args[1]).unwrap();
        assert_eq!(
            settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"],
            "'/opt/llmman' tk hook"
        );
        assert!(added.env.is_empty());
        assert!(added.config.is_none());
        assert!(Optimizer::Tk
            .addition_with("codex", Path::new("/opt/llmman"), Path::new(""))
            .is_err());
    }

    #[test]
    fn the_notice_says_how_to_turn_tk_off() {
        assert_eq!(
            Optimizer::Tk.notice("claude").as_deref(),
            Some(
                "[llmman] claude: optimizer tk (command output compacted; LLMMAN_TK=off disables)"
            )
        );
    }

    #[test]
    fn cvmn_works_with_claude_and_opencode() {
        Optimizer::Cvmn.check("claude", false, &[]).unwrap();
        Optimizer::Cvmn.check("opencode", false, &[]).unwrap();
        Optimizer::Cvmn
            .check("claude", false, &["-p".into(), "hello".into()])
            .unwrap();
        assert_eq!(CVMN_INTEGRATIONS, ["claude", "codex", "opencode"]);
        for name in ["aider", "goose", "docker-agent"] {
            let err = Optimizer::Cvmn
                .check(name, false, &[])
                .unwrap_err()
                .to_string();
            assert!(err.contains(&format!("does not work with {name}")), "{err}");
            assert!(
                err.contains("only claude, codex, opencode can use it"),
                "{err}"
            );
        }
    }

    #[test]
    fn cvmn_is_refused_in_a_sandbox() {
        let err = Optimizer::Cvmn
            .check("claude", true, &[])
            .unwrap_err()
            .to_string();
        assert!(err.contains("--sandbox"), "{err}");
    }

    #[test]
    fn cvmn_is_refused_beside_a_system_prompt_of_the_callers() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        for extra in [
            &["--append-system-prompt", "mine"][..],
            &["--append-system-prompt=mine"],
            &["--append-system-prompt-file", "mine.md"],
        ] {
            let err = Optimizer::Cvmn
                .check("claude", false, &args(extra))
                .unwrap_err()
                .to_string();
            assert!(err.contains("--append-system-prompt"), "{extra:?}: {err}");
        }
        // After a second `--`, it is a prompt.
        Optimizer::Cvmn
            .check(
                "claude",
                false,
                &args(&["-p", "--", "--append-system-prompt", "x"]),
            )
            .unwrap();
    }

    #[test]
    fn cvmn_appends_its_prompt_for_claude_and_nothing_else() {
        let added = Optimizer::Cvmn.addition("claude").unwrap();
        assert_eq!(added.args.len(), 2);
        assert_eq!(added.args[0], "--append-system-prompt");
        assert_eq!(added.args[1], CVMN_PROMPT.trim());
        assert!(added.args[1].starts_with("Answer tersely."));
        assert!(added.env.is_empty());
        assert!(added.config.is_none());
        assert!(Optimizer::Cvmn.addition("aider").is_err());
    }

    #[test]
    fn cvmn_sets_codexs_developer_instructions_with_a_config_override() {
        let added = Optimizer::Cvmn.addition("codex").unwrap();
        assert_eq!(added.args.len(), 2);
        assert_eq!(added.args[0], "-c");
        let (key, value) = added.args[1].split_once('=').unwrap();
        assert_eq!(key, "developer_instructions");
        // The override parses as TOML, and gives the prompt back.
        let doc: toml_edit::DocumentMut = added.args[1].parse().unwrap();
        assert_eq!(doc[key].as_str(), Some(CVMN_PROMPT.trim()));
        assert!(!value.is_empty());
    }

    #[test]
    fn opencode_gets_its_files_from_the_directory_it_is_given() {
        let dir = tempfile::tempdir().unwrap();
        let exe = Path::new("/opt/llm man/llmman");

        let added = Optimizer::Cvmn
            .addition_with("opencode", exe, dir.path())
            .unwrap();
        let file = dir.path().join("cvmn.md");
        assert_eq!(std::fs::read_to_string(&file).unwrap(), CVMN_PROMPT);
        assert_eq!(
            added.config,
            Some(serde_json::json!({ "instructions": [file.to_str().unwrap()] }))
        );
        assert!(added.args.is_empty() && added.env.is_empty());

        let added = Optimizer::Tk
            .addition_with("opencode", exe, dir.path())
            .unwrap();
        let plugin = dir.path().join("tk");
        assert_eq!(
            added.config,
            Some(serde_json::json!({ "plugins": [plugin.to_str().unwrap()] }))
        );
        let source = std::fs::read_to_string(plugin.join("index.js")).unwrap();
        assert!(
            source.contains(r#"const LLMMAN = "/opt/llm man/llmman""#),
            "{source}"
        );
        assert!(!source.contains("__LLMMAN__"));
    }

    #[test]
    fn the_notice_says_cvmn_is_on() {
        assert_eq!(
            Optimizer::Cvmn.notice("claude").as_deref(),
            Some("[llmman] claude: optimizer cvmn (terse replies requested)")
        );
    }
}
