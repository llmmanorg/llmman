//! `llmman launch --optimizer`: what an integration is launched with, besides
//! the model, to make a session cheaper.
//!
//! Each optimizer says which integrations it works with ([`Optimizer::check`],
//! run before the daemon starts) and what it adds to a launch
//! ([`Optimizer::addition`]); a launcher just applies the [`Addition`].
//! `none` adds nothing.

use std::path::Path;

use anyhow::Context;
use clap::ValueEnum;

use super::claude;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum Optimizer {
    /// Launch as without the flag; nothing is added
    #[default]
    None,
    /// Compact the output of the shell commands the agent runs (claude only)
    Tk,
}

/// What an optimizer adds to a launch.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Addition {
    /// Arguments placed before the ones the launcher builds.
    pub args: Vec<String>,
    /// Environment variables set for the integration.
    pub env: Vec<(String, String)>,
}

/// The integrations `tk` works with: those with a hook that can rewrite a
/// shell command before it runs.
const TK_INTEGRATIONS: &[&str] = &["claude"];

impl Optimizer {
    /// The name `--optimizer` spells it with.
    fn name(self) -> &'static str {
        match self {
            Optimizer::None => "none",
            Optimizer::Tk => "tk",
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
        match self {
            Optimizer::None => Ok(()),
            Optimizer::Tk => {
                anyhow::ensure!(
                    TK_INTEGRATIONS.contains(&integration),
                    "--optimizer {} does not work with {integration}: only {} can use it.\n\
                     Launch {integration} without --optimizer, or use one of those.",
                    self.name(),
                    TK_INTEGRATIONS.join(", ")
                );
                anyhow::ensure!(
                    !sandboxed,
                    "--optimizer {} does not work with --sandbox: the sandbox does not have \
                     this machine's llmman, which the {integration} hook runs.",
                    self.name()
                );
                if let Some(why) = claude::optimizer_conflict(extra_args) {
                    anyhow::bail!("--optimizer {} does not work with {why}", self.name());
                }
                anyhow::ensure!(
                    !cfg!(windows),
                    "--optimizer {} is not supported on Windows.",
                    self.name()
                );
                Ok(())
            }
        }
    }

    /// What this optimizer adds to a launch of `integration`. `none` adds
    /// nothing and looks at nothing, so a launch without the flag does no
    /// work it did not do before.
    pub(super) fn addition(self, integration: &str) -> anyhow::Result<Addition> {
        match self {
            Optimizer::None => Ok(Addition::default()),
            Optimizer::Tk => {
                let exe = std::env::current_exe()
                    .context("cannot find the path of the running llmman")?;
                self.addition_with(integration, &exe)
            }
        }
    }

    /// [`Optimizer::addition`] for a given path of the running `llmman`.
    pub(super) fn addition_with(self, integration: &str, exe: &Path) -> anyhow::Result<Addition> {
        match (self, integration) {
            (Optimizer::None, _) => Ok(Addition::default()),
            (Optimizer::Tk, "claude") => {
                let exe = exe.to_str().ok_or_else(|| {
                    anyhow::anyhow!("the path of llmman is not valid UTF-8: {}", exe.display())
                })?;
                Ok(Addition {
                    args: vec!["--settings".into(), crate::tk::hook::settings_json(exe)],
                    env: Vec::new(),
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::ValueEnum;

    #[test]
    fn the_choices_are_spelled_none_and_tk() {
        let names: Vec<String> = Optimizer::value_variants()
            .iter()
            .map(|o| o.to_possible_value().unwrap().get_name().to_string())
            .collect();
        assert_eq!(names, ["none", "tk"]);
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
        for name in ["opencode", "codex", "aider", "goose", "docker-agent"] {
            let err = Optimizer::Tk
                .check(name, false, &[])
                .unwrap_err()
                .to_string();
            assert!(err.contains(&format!("does not work with {name}")), "{err}");
            assert!(err.contains("only claude can use it"), "{err}");
        }
    }

    #[test]
    fn every_integration_with_tk_is_one_the_capability_list_names() {
        // The list is the only place that says which integrations work.
        assert_eq!(TK_INTEGRATIONS, ["claude"]);
    }

    #[test]
    fn tk_is_refused_in_a_sandbox() {
        let err = Optimizer::Tk
            .check("claude", true, &[])
            .unwrap_err()
            .to_string();
        assert!(err.contains("--sandbox"), "{err}");
        assert!(err.contains("llmman"), "{err}");
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
            .addition_with("claude", Path::new("/opt/llmman"))
            .unwrap();
        assert_eq!(added.args.len(), 2);
        assert_eq!(added.args[0], "--settings");
        let settings: serde_json::Value = serde_json::from_str(&added.args[1]).unwrap();
        assert_eq!(
            settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"],
            "'/opt/llmman' tk hook"
        );
        assert!(added.env.is_empty());
        assert!(Optimizer::Tk
            .addition_with("opencode", Path::new("/opt/llmman"))
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
}
