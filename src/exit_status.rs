//! Readable child exit statuses. A child that dies in the OS loader writes
//! nothing to stderr, so its exit status is the only clue.

use std::process::ExitStatus;

pub(crate) const CODE_INTEGRITY_HINT: &str = "Smart App Control or an App Control policy refused \
     to load a file; see Event Viewer > Applications and Services Logs > Microsoft > Windows > \
     CodeIntegrity > Operational";

/// Win32 `ERROR_CODE_INTEGRITY_POLICY_VIOLATION`, from `CreateProcess`.
pub(crate) const ERROR_CODE_INTEGRITY: i32 = 4551;

/// NTSTATUS exit codes users meet from a child that never reached `main`.
pub(crate) const NTSTATUS: &[(u32, &str, &str)] = &[
    (
        0xC000_0135,
        "STATUS_DLL_NOT_FOUND",
        "a required DLL is missing, usually the Visual C++ runtime or a CUDA runtime",
    ),
    (
        0xC000_007B,
        "STATUS_INVALID_IMAGE_FORMAT",
        "an executable or DLL has the wrong architecture",
    ),
    (
        0xC000_0142,
        "STATUS_DLL_INIT_FAILED",
        "a DLL failed to initialise, usually a missing runtime or a blocking policy",
    ),
    (
        0xC0E9_0002,
        "blocked by Windows Code Integrity",
        CODE_INTEGRITY_HINT,
    ),
    (
        0xC000_0005,
        "STATUS_ACCESS_VIOLATION",
        "the process crashed",
    ),
    (
        0xC000_001D,
        "STATUS_ILLEGAL_INSTRUCTION",
        "this build needs a CPU instruction this machine lacks",
    ),
];

/// `exit code 0xC0E90002 (blocked by ...)`, or `killed by signal 4 (SIGILL: ...)`.
pub(crate) fn describe(status: &ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return describe_signal(signal);
        }
    }
    match status.code() {
        Some(code) => describe_code(code),
        None => status.to_string(),
    }
}

pub(crate) fn describe_code(code: i32) -> String {
    // Windows reports an NTSTATUS as a negative `i32`.
    if code < 0 {
        let code = code as u32;
        return match NTSTATUS.iter().find(|(c, ..)| *c == code) {
            Some((_, name, hint)) => format!("exit code {code:#010X} ({name}: {hint})"),
            None => format!("exit code {code:#010X}"),
        };
    }
    match code {
        126 => "exit code 126 (command found but not executable)".into(),
        127 => "exit code 127 (command not found, or a library it needs is missing)".into(),
        _ => format!("exit code {code}"),
    }
}

#[cfg(unix)]
fn describe_signal(signal: i32) -> String {
    let (name, hint) = match signal {
        libc::SIGSEGV => ("SIGSEGV", "the process crashed"),
        libc::SIGILL => (
            "SIGILL",
            "this build needs a CPU instruction this machine lacks",
        ),
        // "OOM", not "out of memory": `looks_like_oom` greps these messages.
        libc::SIGKILL => ("SIGKILL", "often the kernel's OOM killer"),
        libc::SIGBUS => ("SIGBUS", "the process crashed"),
        libc::SIGABRT => ("SIGABRT", "the process aborted"),
        libc::SIGFPE => ("SIGFPE", "the process crashed"),
        libc::SIGTERM => ("SIGTERM", "it was asked to stop"),
        _ => return format!("killed by signal {signal}"),
    };
    format!("killed by signal {signal} ({name}: {hint})")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_loader_failures_are_named_in_unsigned_hex() {
        let blocked = describe_code(0xC0E9_0002_u32 as i32);
        assert!(blocked.starts_with("exit code 0xC0E90002 (blocked by Windows Code Integrity"));
        assert!(blocked.contains("Smart App Control"), "{blocked}");
        assert!(blocked.contains("CodeIntegrity"), "{blocked}");
        assert!(describe_code(0xC000_0135_u32 as i32).contains("STATUS_DLL_NOT_FOUND"));
        assert_eq!(
            describe_code(0xC123_4567_u32 as i32),
            "exit code 0xC1234567"
        );
    }

    #[test]
    fn plain_codes_stay_decimal_and_shell_codes_are_explained() {
        assert_eq!(describe_code(1), "exit code 1");
        assert!(describe_code(127).contains("not found"));
        assert!(describe_code(126).contains("not executable"));
    }

    #[cfg(unix)]
    #[test]
    fn unix_signals_are_named() {
        use std::os::unix::process::ExitStatusExt;
        let ill = describe(&ExitStatus::from_raw(libc::SIGILL));
        assert!(ill.contains("SIGILL") && ill.contains("CPU"), "{ill}");
        let kill = describe(&ExitStatus::from_raw(libc::SIGKILL));
        assert!(kill.contains("SIGKILL") && kill.contains("OOM"), "{kill}");
        assert_eq!(describe(&ExitStatus::from_raw(3 << 8)), "exit code 3");
    }
}
