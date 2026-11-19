//! Who a container's process runs as on the host, shared by the launch
//! sandbox (`cmd::launch::sandbox`) and the hardened engine containers
//! (`container::HardenedUser`): the same daemon-versus-user question
//! decides both what a bind mount is written as and what one is read as.

use std::process::{Command, Stdio};
use std::sync::OnceLock;

use super::ContainerManager;

/// Who the guest writes files as. A Linux bind mount keeps the guest's
/// uid; Docker Desktop and the VM backends map it to the host user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Identity {
    Image,
    /// `docker run --user uid:gid`: the user, or root (`0:0`) for a
    /// rootless daemon, whose container root is the host user.
    User {
        uid: u32,
        gid: u32,
    },
    /// Rootless podman: `--userns=keep-id`.
    KeepId,
}

impl Identity {
    /// The `--user` value for [`Identity::User`].
    pub(crate) fn user_arg(uid: u32, gid: u32) -> String {
        format!("{uid}:{gid}")
    }

    /// The pure rule behind [`identity`], for callers that resolved
    /// `rootless_docker` and `ids` themselves: `None` ids (not unix) is
    /// the image's own user; a Linux docker daemon gets the host user,
    /// or root when rootless (its root is the host user already); a
    /// rootless podman (`euid != 0`) keeps the host id; everything else
    /// (Docker Desktop, rootful podman) runs as the image's user.
    pub(crate) fn from_parts(
        engine: Option<ContainerManager>,
        linux: bool,
        rootless_docker: bool,
        ids: Option<(u32, u32, u32)>,
    ) -> Identity {
        let Some((uid, gid, euid)) = ids else {
            return Identity::Image;
        };
        match engine {
            Some(ContainerManager::Docker) if linux && rootless_docker => {
                Identity::User { uid: 0, gid: 0 }
            }
            Some(ContainerManager::Docker) if linux => Identity::User { uid, gid },
            Some(ContainerManager::Podman) if euid != 0 => Identity::KeepId,
            _ => Identity::Image,
        }
    }
}

/// [`Identity::from_parts`] for this process and host; `docker info` is
/// consulted only for a Linux docker daemon.
pub(crate) fn identity(engine: Option<ContainerManager>, linux: bool) -> Identity {
    let rootless = engine == Some(ContainerManager::Docker) && linux && docker_is_rootless();
    Identity::from_parts(engine, linux, rootless, user_ids())
}

/// This process's `(uid, gid, euid)`; `None` where there is none.
#[cfg(unix)]
pub(crate) fn user_ids() -> Option<(u32, u32, u32)> {
    // SAFETY: these cannot fail and touch no memory.
    Some(unsafe { (libc::getuid(), libc::getgid(), libc::geteuid()) })
}

#[cfg(not(unix))]
pub(crate) fn user_ids() -> Option<(u32, u32, u32)> {
    None
}

/// The answer of the first [`docker_is_rootless`] probe that reached the
/// daemon: it cannot change under a running process, and the hardened
/// spawns ask on every model load from the async request path, so only
/// the first pays for `docker info`.
static DOCKER_ROOTLESS: OnceLock<bool> = OnceLock::new();

/// Whether the reachable docker daemon is rootless, per the
/// `SecurityOptions` of `docker info`, waiting at most
/// [`PROBE_TIMEOUT`](super::PROBE_TIMEOUT) as [`ContainerManager::probe`]
/// does. A daemon that does not answer counts as rootful for this call
/// only, with a note on stderr, so a later call can still get the real
/// answer.
pub(crate) fn docker_is_rootless() -> bool {
    if let Some(known) = DOCKER_ROOTLESS.get() {
        return *known;
    }
    match probe_docker_rootless() {
        Ok(rootless) => *DOCKER_ROOTLESS.get_or_init(|| rootless),
        Err(err) => {
            eprintln!("[llmman] docker info did not say whether the daemon is rootless ({err}); assuming not");
            false
        }
    }
}

fn probe_docker_rootless() -> Result<bool, String> {
    let mut cmd = Command::new("docker");
    cmd.args(["info", "--format", "{{json .SecurityOptions}}"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = cmd.spawn().map_err(|e| e.to_string())?;
    match super::wait_with_timeout(&mut child, super::PROBE_TIMEOUT) {
        Ok(Some(status)) if status.success() => {}
        Ok(Some(status)) => return Err(format!("exited with {status}")),
        Ok(None) => return Err(format!("no answer within {:?}", super::PROBE_TIMEOUT)),
        Err(e) => return Err(e.to_string()),
    }
    // The child has exited, so this only drains the pipe.
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    Ok(String::from_utf8_lossy(&out.stdout).contains("rootless"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_follows_the_engine_and_who_runs_it() {
        let ids = Some((1000, 1000, 1000));
        let docker = Some(ContainerManager::Docker);
        let podman = Some(ContainerManager::Podman);
        assert_eq!(
            Identity::from_parts(docker, true, false, ids),
            Identity::User {
                uid: 1000,
                gid: 1000
            }
        );
        assert_eq!(
            Identity::from_parts(docker, true, true, ids),
            Identity::User { uid: 0, gid: 0 }
        );
        // Docker Desktop maps the uid itself.
        assert_eq!(
            Identity::from_parts(docker, false, false, ids),
            Identity::Image
        );
        assert_eq!(
            Identity::from_parts(podman, true, false, ids),
            Identity::KeepId
        );
        assert_eq!(
            Identity::from_parts(podman, true, false, Some((0, 0, 0))),
            Identity::Image
        );
        assert_eq!(
            Identity::from_parts(None, true, false, ids),
            Identity::Image
        );
        assert_eq!(
            Identity::from_parts(docker, true, false, None),
            Identity::Image
        );
        assert_eq!(Identity::user_arg(1000, 100), "1000:100");
    }
}
