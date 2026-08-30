//! Discovery of the local host's CPU core topology from /sys, used by
//! [`crate::cmd::serve`] to derive the `--threads` value passed to
//! local `llama-server` spawns. The only public surface is
//! [`math_core_count`]; the parsing and counting helpers below it are
//! pure functions so they are testable without a real /sys.

/// Math-core count from /sys/devices/system/cpu/cpu*/topology,
/// approximating the core set llama.cpp's `common_cpu_get_num_math()`
/// targets: on hybrid x86 (kernel 5.16+) the probed IDs are the
/// performance-core threads from /sys/devices/cpu_core/cpus, so
/// efficiency cores don't inflate the cap; everywhere else they are
/// every ID in /sys/devices/system/cpu/present (POWER's up-to-SMT-2
/// rule is not mirrored). The caller still bounds the result by
/// `available_parallelism`. The probed IDs never come from a CPU
/// *count*: counts capped by quota or affinity are not IDs, and on
/// hosts with adjacent SMT sibling numbering probing cpu0..count would
/// see one core twice instead of two cores. Skipping per-CPU entries
/// whose topology is unreadable is deliberate, not a fallback trigger:
/// `present` keeps offline CPUs listed while the kernel unregisters
/// their topology directory, and an offline core contributes no
/// execution capacity, so counting only the readable ones beats
/// returning `None` and having the caller fall back to
/// `available_parallelism` (the SMT thread count). Only when *no*
/// entry is readable does this return `None`. The pure parsing and
/// counting live in [`math_cpu_ids`], [`cpu_id_list`], and
/// [`count_physical_cores`].
pub fn math_core_count() -> Option<u32> {
    let cpu_core = std::fs::read_to_string("/sys/devices/cpu_core/cpus").ok();
    let present = std::fs::read_to_string("/sys/devices/system/cpu/present").ok();
    let sibling_lists: Vec<String> = math_cpu_ids(cpu_core.as_deref(), present.as_deref())?
        .into_iter()
        .filter_map(|i| {
            std::fs::read_to_string(format!(
                "/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list"
            ))
            .ok()
        })
        .collect();
    count_physical_cores(sibling_lists.iter().map(String::as_str))
}

/// The CPU IDs whose physical cores bound the math-thread count: the
/// hybrid-x86 performance-core list (/sys/devices/cpu_core/cpus
/// content) when parseable, otherwise every present CPU. Unparseable
/// `cpu_core` content falls back rather than failing: absence and
/// garbage both mean "no trustworthy P-core list", and the
/// all-physical-core count is the right answer on non-hybrid hardware
/// either way.
fn math_cpu_ids(cpu_core: Option<&str>, present: Option<&str>) -> Option<Vec<u32>> {
    cpu_core
        .and_then(cpu_id_list)
        .or_else(|| cpu_id_list(present?))
}

/// CPU IDs from a kernel CPU list such as /sys/devices/system/cpu/present:
/// comma-separated single IDs or inclusive ranges (`0-15`, `0,4-7`).
/// `None` on empty or malformed content.
fn cpu_id_list(list: &str) -> Option<Vec<u32>> {
    let mut ids = Vec::new();
    for part in list.trim().split(',') {
        match part.split_once('-') {
            Some((lo, hi)) => {
                let (lo, hi): (u32, u32) = (lo.trim().parse().ok()?, hi.trim().parse().ok()?);
                if lo > hi {
                    return None;
                }
                ids.extend(lo..=hi);
            }
            None => ids.push(part.trim().parse().ok()?),
        }
    }
    (!ids.is_empty()).then_some(ids)
}

/// Distinct physical cores among per-CPU `thread_siblings_list`
/// contents: SMT siblings of one core all report the same list, so the
/// number of distinct values is the core count (the same counting
/// llama.cpp's `cpu_get_num_physical_cores` does). `None` when nothing
/// was readable.
fn count_physical_cores<'a>(sibling_lists: impl IntoIterator<Item = &'a str>) -> Option<u32> {
    let cores: std::collections::HashSet<&str> = sibling_lists
        .into_iter()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    u32::try_from(cores.len()).ok().filter(|&n| n > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_physical_cores_dedups_smt_siblings() {
        // Two SMT threads per core report the same siblings list.
        assert_eq!(
            count_physical_cores([
                "0,4\n", "1,5\n", "2,6\n", "3,7\n", "0,4\n", "1,5\n", "2,6\n", "3,7\n"
            ]),
            Some(4)
        );
        // No SMT: one entry per core.
        assert_eq!(count_physical_cores(["0\n", "1\n"]), Some(2));
        // Unreadable/empty topology: None, so callers fall back.
        assert_eq!(count_physical_cores([]), None);
        assert_eq!(count_physical_cores(["", "\n"]), None);
    }

    #[test]
    fn math_cpu_ids_prefers_the_hybrid_p_core_list() {
        // (cpu_core content, present content, expected)
        let cases = [
            // Hybrid x86: /sys/devices/cpu_core/cpus lists the P-core
            // threads; E-core IDs in `present` must not be probed.
            (Some("0-7\n"), Some("0-15\n"), Some((0..=7).collect())),
            // Non-hybrid (or pre-5.16 kernel): the file is absent.
            (None, Some("0-15\n"), Some((0..=15).collect())),
            // Unparseable P-core list falls back to `present`.
            (Some("garbage\n"), Some("0-3\n"), Some(vec![0, 1, 2, 3])),
            (Some("\n"), Some("0-3\n"), Some(vec![0, 1, 2, 3])),
            (None, Some("garbage\n"), None),
            (None, None, None),
            (Some("0-7\n"), None, Some((0..=7).collect())),
        ];
        for (cpu_core, present, expected) in &cases {
            assert_eq!(
                &math_cpu_ids(*cpu_core, *present),
                expected,
                "cpu_core={cpu_core:?} present={present:?}"
            );
        }
    }

    #[test]
    fn cpu_id_list_parses_ids_and_inclusive_ranges() {
        assert_eq!(cpu_id_list("0-3\n"), Some(vec![0, 1, 2, 3]));
        assert_eq!(cpu_id_list("0"), Some(vec![0]));
        assert_eq!(cpu_id_list("0,4-6\n"), Some(vec![0, 4, 5, 6]));
        for malformed in ["", "\n", "3-1", "0-x", "a", "0,,2"] {
            assert_eq!(cpu_id_list(malformed), None, "list={malformed:?}");
        }
    }
}
