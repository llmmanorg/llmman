//! Cheap CLI checks that spawn the built binary; no model or network.

use std::process::Command;

fn llmman() -> Command {
    Command::new(env!("CARGO_BIN_EXE_llmman"))
}

/// A bare `llmman` prints help and exits 0 (clap's default for a missing
/// subcommand is 2, which winget's package validator flags as an error).
#[test]
fn bare_invocation_prints_help_and_exits_0() {
    let out = llmman().output().expect("spawn llmman");
    assert!(out.status.success(), "exit status: {}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Usage: llmman"), "stdout: {stdout}");
    assert!(stdout.contains("launch"), "stdout: {stdout}");
}

/// An unknown subcommand is still a usage error.
#[test]
fn unknown_subcommand_exits_2() {
    let out = llmman().arg("bogus").output().expect("spawn llmman");
    assert_eq!(out.status.code(), Some(2));
}
