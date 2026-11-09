//! `llmman tk`: runs a command and prints its output compacted. The shell
//! tool of a coding agent reaches it through `llmman launch --optimizer tk`;
//! it also works on its own. Hidden from `llmman --help`, like the other
//! internal commands. See [`crate::tk`].

use std::ffi::OsString;

use clap::{Args, Subcommand};

#[derive(Args, Debug)]
#[command(
    arg_required_else_help = true,
    subcommand_required = true,
    disable_help_subcommand = true
)]
pub struct TkArgs {
    #[command(subcommand)]
    command: TkCommand,
}

#[derive(Subcommand, Debug)]
enum TkCommand {
    /// Claude Code PreToolUse hook: reads a tool call as JSON on stdin and,
    /// for a Bash command `llmman tk` can compact, answers with it rewritten
    /// to run through `llmman tk`. Says nothing otherwise, and never fails.
    Hook,
    #[command(external_subcommand)]
    Run(Vec<OsString>),
}

pub fn run(args: &TkArgs) -> anyhow::Result<()> {
    match &args.command {
        TkCommand::Hook => {
            crate::tk::hook::run();
            Ok(())
        }
        // The command's own exit status is the result.
        TkCommand::Run(argv) => std::process::exit(crate::tk::run(argv)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        args: TkArgs,
    }

    fn parse(argv: &[&str]) -> Result<TkCommand, clap::Error> {
        Cli::try_parse_from(["tk"].into_iter().chain(argv.iter().copied())).map(|c| c.args.command)
    }

    #[test]
    fn the_command_and_its_flags_reach_tk_untouched() {
        let TkCommand::Run(argv) = parse(&["git", "diff", "--stat", "-U0", "--", "x"]).unwrap()
        else {
            panic!("not a command");
        };
        assert_eq!(argv, ["git", "diff", "--stat", "-U0", "--", "x"]);
        assert!(matches!(parse(&["hook"]).unwrap(), TkCommand::Hook));
    }

    #[test]
    fn a_bare_tk_asks_for_a_command() {
        assert!(parse(&[]).is_err());
    }
}
