//! Filter for `docker ps` (and `docker container ls`): the COMMAND and
//! CREATED columns go, and the IPv6 port entries that repeat an IPv4 one.

use super::text::join;
use super::{Compact, Family, Plan, Run};

/// Docker's own options that take a value, as `--context remote`.
const VALUE_OPTIONS: &[&str] = &[
    "--config",
    "-c",
    "--context",
    "-H",
    "--host",
    "-l",
    "--log-level",
    "--tlscacert",
    "--tlscert",
    "--tlskey",
];

pub(super) fn plan(rest: &[String]) -> Plan {
    let words: Vec<&str> = super::skip_options(rest, VALUE_OPTIONS)
        .iter()
        .map(String::as_str)
        .filter(|a| !a.starts_with('-'))
        .collect();
    let lists = matches!(
        words.as_slice(),
        ["ps", ..] | ["container", "ls" | "ps" | "list", ..]
    );
    // `-q` prints bare IDs; those are for a program.
    let ids_only = super::has_flag(rest, &["--quiet"]) || super::has_short(rest, "q");
    if lists && !ids_only {
        Plan::Filter(Family::DockerPs)
    } else {
        Plan::Passthrough
    }
}

/// Where each column of the header starts, in chars: docker separates its
/// columns by at least two spaces, and a name may hold one (`CONTAINER ID`).
fn column_starts(header: &[char]) -> Vec<usize> {
    let mut starts = Vec::new();
    for (i, &c) in header.iter().enumerate() {
        let gap = i == 0 || (header[i - 1] == ' ' && i >= 2 && header[i - 2] == ' ');
        if c != ' ' && gap {
            starts.push(i);
        }
    }
    starts
}

/// A PORTS cell without the `:::` and `[::]:` twins of its IPv4 entries. An
/// IPv6 entry with no twin is the only way the port is published: it stays.
fn ports(cell: &str) -> String {
    let entries: Vec<&str> = cell.split(", ").collect();
    let kept: Vec<&str> = entries
        .iter()
        .copied()
        .filter(|p| {
            let v6 = p.strip_prefix(":::").or_else(|| p.strip_prefix("[::]:"));
            v6.is_none_or(|rest| !entries.contains(&format!("0.0.0.0:{rest}").as_str()))
        })
        .collect();
    kept.join(", ")
}

pub(super) fn ps(run: &Run<'_>) -> Option<Compact> {
    let mut lines = run.stdout.lines();
    let header: Vec<char> = lines.next()?.chars().collect();
    let head: String = header.iter().collect();
    if run.code != 0 || !head.starts_with("CONTAINER ID") {
        return None;
    }
    let starts = column_starts(&header);
    let name = |k: usize| -> String {
        let end = starts.get(k + 1).copied().unwrap_or(header.len());
        header[starts[k]..end]
            .iter()
            .collect::<String>()
            .trim()
            .to_string()
    };
    let names: Vec<String> = (0..starts.len()).map(name).collect();
    let keep: Vec<usize> = (0..starts.len())
        .filter(|&k| names[k] != "COMMAND" && names[k] != "CREATED")
        .collect();
    // Dropping nothing would be a rewrite for no gain, and without COMMAND the
    // columns are not the ones this expects.
    if keep.len() == starts.len() || !names.iter().any(|n| n == "IMAGE") {
        return None;
    }
    let row = |cells: Vec<String>| -> String {
        cells
            .iter()
            .map(|c| c.trim())
            .filter(|c| !c.is_empty())
            .collect::<Vec<_>>()
            .join("  ")
    };
    let mut out = vec![row(keep.iter().map(|&k| names[k].clone()).collect())];
    for line in lines {
        let chars: Vec<char> = line.chars().collect();
        let cells = keep
            .iter()
            .map(|&k| {
                let from = starts[k].min(chars.len());
                let to = starts
                    .get(k + 1)
                    .copied()
                    .unwrap_or(chars.len())
                    .min(chars.len());
                let cell: String = chars[from..to].iter().collect();
                if names[k] == "PORTS" {
                    ports(cell.trim())
                } else {
                    cell
                }
            })
            .collect();
        out.push(row(cells));
    }
    Some(Compact {
        stdout: join(&out),
        stderr: run.stderr.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tk::tests::{run, words as args};

    /// A table as docker lays it out: each column as wide as its longest
    /// cell, and three spaces between columns.
    fn table(rows: &[[&str; 7]]) -> String {
        let widths: Vec<usize> = (0..7)
            .map(|c| rows.iter().map(|r| r[c].chars().count()).max().unwrap() + 3)
            .collect();
        let mut out = String::new();
        for row in rows {
            let mut line = String::new();
            for (c, cell) in row.iter().enumerate() {
                if c == 6 {
                    line.push_str(cell);
                } else {
                    line.push_str(&format!("{cell:<width$}", width = widths[c]));
                }
            }
            out.push_str(line.trim_end());
            out.push('\n');
        }
        out
    }

    fn ps_fixture() -> String {
        table(&[
            [
                "CONTAINER ID",
                "IMAGE",
                "COMMAND",
                "CREATED",
                "STATUS",
                "PORTS",
                "NAMES",
            ],
            [
                "3f2a1b9c8d7e",
                "nginx:latest",
                "\"/docker-entrypoint.…\"",
                "2 hours ago",
                "Up 2 hours",
                "0.0.0.0:8080->80/tcp, :::8080->80/tcp",
                "web",
            ],
            [
                "a1b2c3d4e5f6",
                "postgres:16",
                "\"docker-entrypoint.s…\"",
                "3 days ago",
                "Up 3 days",
                "5432/tcp",
                "db",
            ],
            [
                "0a9b8c7d6e5f",
                "redis:7",
                "\"redis-server\"",
                "10 minutes ago",
                "Exited (0) 1 minute ago",
                "",
                "cache",
            ],
        ])
    }

    #[test]
    fn ps_keeps_id_image_status_ports_and_names() {
        let fixture = ps_fixture();
        let r = run(0, &fixture, "");
        let out = ps(&r).unwrap().stdout;
        assert_eq!(
            out,
            "CONTAINER ID  IMAGE  STATUS  PORTS  NAMES
3f2a1b9c8d7e  nginx:latest  Up 2 hours  0.0.0.0:8080->80/tcp  web
a1b2c3d4e5f6  postgres:16  Up 3 days  5432/tcp  db
0a9b8c7d6e5f  redis:7  Exited (0) 1 minute ago  cache
"
        );
        assert!(out.len() < fixture.len());
    }

    #[test]
    fn a_table_it_does_not_recognize_is_left_alone() {
        for stdout in [
            "NAME   STATUS\nweb    up\n",
            "CONTAINER ID   IMAGE   STATUS   NAMES\nabc   x   Up   web\n",
            "",
        ] {
            let r = Run {
                code: 0,
                stdout,
                stderr: "",
            };
            assert!(ps(&r).is_none(), "{stdout:?}");
        }
        let fixture = ps_fixture();
        let failed = run(1, &fixture, "Cannot connect to the Docker daemon\n");
        assert!(ps(&failed).is_none());
    }

    #[test]
    fn an_ipv6_port_stays_unless_an_ipv4_one_repeats_it() {
        assert_eq!(
            ports("0.0.0.0:8080->80/tcp, :::8080->80/tcp"),
            "0.0.0.0:8080->80/tcp"
        );
        assert_eq!(
            ports("0.0.0.0:8080->80/tcp, [::]:8080->80/tcp, [::]:9090->90/tcp"),
            "0.0.0.0:8080->80/tcp, [::]:9090->90/tcp"
        );
        assert_eq!(ports("[::]:8080->80/tcp"), "[::]:8080->80/tcp");
        assert_eq!(ports(":::8080->80/tcp"), ":::8080->80/tcp");
        assert_eq!(ports("5432/tcp"), "5432/tcp");
    }

    #[test]
    fn plans() {
        let p = |line: &str| crate::tk::plan(&args(line));
        for line in [
            "docker ps",
            "docker ps -a",
            "docker container ls",
            "docker ps --filter status=exited",
            "docker --context remote ps",
            "docker -H tcp://h:2375 container ls",
            "docker --context=remote ps -a",
        ] {
            assert_eq!(p(line), Plan::Filter(Family::DockerPs), "{line}");
        }
        for line in [
            "docker ps -q",
            "docker ps -aq",
            "docker ps --quiet",
            "docker ps --format {{.Names}}",
            "docker run -it ubuntu",
            "docker logs -f web",
            "docker exec web ls",
            "docker images",
            "docker",
        ] {
            assert_eq!(p(line), Plan::Passthrough, "{line}");
        }
    }
}
