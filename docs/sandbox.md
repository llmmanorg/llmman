# Sandboxed launches

`llmman launch --sandbox <sandbox>` runs the integration inside a sandbox
instead of directly on your machine. The model setup is the same with or
without it: `--model`, `--provider` and `--overflow-*` work as usual, and
the integration still talks to `llmman serve`.

```sh
llmman launch claude --model qwen3.8 --sandbox docker
llmman launch opencode --provider openrouter --model qwen/qwen3-coder --sandbox podman
llmman launch codex --model qwen3.8 --sandbox microsandbox -- exec "Explain this repository"
```

| Sandbox | Runs | Host |
|---|---|---|
| `sbx` | `sbx run <agent>` (Docker Sandboxes), which also serves the model | Wherever `sbx` runs |
| `seatbelt` | The installed integration under `sandbox-exec` | macOS |
| `docker` | The integration from an image, in a Docker container | Linux, macOS |
| `podman` | The same, in a Podman container | Linux, macOS |
| `apple-container` | The same, in an Apple `container` VM | macOS on Apple silicon |
| `microsandbox` | The same, in a microsandbox (`msb`) microVM | Linux with KVM, macOS on Apple silicon |
| `openshell` | The same, in an NVIDIA OpenShell sandbox, with the workspace uploaded | Wherever the OpenShell gateway runs |

Windows is not supported except through `sbx`.

## What the integration can reach

The **workspace** is the Git work tree containing the current directory,
or the current directory itself when there is no work tree. The integration
can change the workspace and nothing else in your project tree. llmman
refuses to start when the workspace would be your home directory or a
directory above it. If your home directory is itself a Git work tree (for
dotfiles), the current directory is used instead.

The workspace's `.git` is read-only, because Git runs hooks and config
settings such as `core.fsmonitor` from there when you next use it on your
machine. So the integration can't commit inside the sandbox. Seatbelt also
stops it from creating a `.git` anywhere in the workspace. The image-based
sandboxes can't prevent that, so check a repository the integration created
before you run `git` in it.

The integration also keeps its own **state**, shared with runs outside the
sandbox: its settings and session directories (`~/.claude`,
`~/.config/opencode`, `~/.codex`, ...), plus the directories llmman writes
its configuration into (for example `~/.config/llmman/launch/dsh`). Only
those subdirectories are shared, never `llmman.conf` or the rest of
`~/.config/llmman`. A state directory set by an environment variable (such
as `CLAUDE_CONFIG_DIR`) that contains your home directory is refused.

- **seatbelt** leaves everything readable and the network open. Writes are
  allowed only to the workspace, the state directories, the temporary
  directories, `~/Library/Caches`, `~/.cache` and `~/.npm`.
- **docker, podman, apple-container and microsandbox** mount the workspace
  and each state directory at its real path. `HOME` is set to your home
  directory's path, but backed by a directory llmman keeps for each
  integration (`~/.local/share/llmman/sandbox/<integration>`), so nothing
  else in your home directory is visible. Files the integration creates
  there, such as caches, persist between runs. A state *file* such as
  `~/.claude.json` is not shared, because a bind-mounted file cannot be
  replaced by renaming another over it. The integration keeps its own copy
  in the sandbox's home.
- **openshell** has no bind mounts by default. The workspace is uploaded
  (skipping `.gitignore`d files), and llmman prints the `openshell sandbox
  download` command that copies the changes back when the integration
  exits. The sandbox is kept until you delete it. OpenShell cannot see
  files in your home directory, so the integrations llmman configures
  through files there (codex, pi, omp, cline, qwen, hermes, openclaw, agy,
  dsh, grok, docker-agent, and opencode with `--variant`) are refused.
  `host.openshell.internal` is the gateway's machine, so a loopback
  daemon needs the gateway running on this machine. Otherwise, point
  `LLMMAN_HOST` at a daemon the gateway can reach.

## Images

docker, podman, apple-container, microsandbox and openshell run the
integration from an image, so the integration does not need to be
installed on your machine. By default this is Docker Sandboxes' image for
the agent, `docker.io/docker/sandbox-templates:<tag>`:

| Integration | Tag |
|---|---|
| `claude` | `claude-code` |
| `codex` | `codex` |
| `opencode` | `opencode` |
| `gemini` | `gemini` |
| `docker-agent` | `docker-agent` |

Any other integration needs `LLMMAN_SANDBOX_IMAGE`, an image with it
installed on `PATH`. The variable also overrides the default image for the
integrations above. The integration's command runs as the image's command,
after its entrypoint.

`sbx` ignores `LLMMAN_SANDBOX_IMAGE`. sbx picks its own image for each
agent, which has to be an sbx template, so an image meant for the
sandboxes above would break it. `seatbelt` runs the integration installed
on your machine, so it uses no image.

## Reaching `llmman serve`

The daemon runs on your machine as usual, and llmman tells the integration
the address the sandbox reaches it at:

| Sandbox | Address of this machine's loopback |
|---|---|
| `seatbelt` | unchanged |
| `docker`, `podman` on Linux | unchanged: the container shares the host's network (`--network host`) |
| `docker` elsewhere | `host.docker.internal` |
| `podman` elsewhere | `host.containers.internal` |
| `apple-container` | `host.container.internal` |
| `microsandbox` | `host.microsandbox.internal` (with `--net public,host`) |
| `openshell` | `host.openshell.internal`, which llmman opens in the sandbox's network policy |

A daemon that `LLMMAN_HOST` points at another machine is reached at that
address. A loopback daemon with TLS (`https://`) is refused whenever the
address has to change, because its certificate does not carry the new
name.

`apple-container` needs a DNS name for the host created once, with sudo:

```sh
sudo container system dns create host.container.internal --localhost 203.0.113.113
```

Apple's `container` removes the packet filter rule this creates when the
Mac restarts. Run the command again after a restart.

## Per-sandbox notes

- **sbx** takes over the whole launch: `llmman launch <integration>
  --sandbox sbx [--model ...] [--provider ...] -- <args>` runs `sbx run
  <agent>` with the same flags and arguments, in the current directory.
  sbx serves the model with its own llmman, so this machine's daemon is
  not started. Supported integrations: claude, codex, copilot,
  docker-agent, gemini, opencode.
- **docker** on Linux runs the container as your user (`--user`). Rootless
  Docker's host network doesn't include your machine's loopback, so it is
  refused unless `LLMMAN_HOST` points elsewhere. In that case the container
  runs as root, which for rootless Docker is your user. **podman** uses
  `--userns=keep-id`. On Linux both disable SELinux labelling for the
  container (`--security-opt label=disable`). Relabelling instead would
  change the labels of your own files.
- **microsandbox** mounts with `host-perms=mirror`, so files the
  integration creates get their real permissions on the host.
- **openshell** can only pass the integration's environment on its
  command line (`--env`), and it stores that environment with the sandbox.
  So it refuses any launch that would hand the integration a real key:
  `--provider`, `--overflow-provider`, or a daemon key (`LLMMAN_API_KEY`).
- A path containing `:` or `,` cannot be mounted, and seatbelt refuses
  paths containing `"` or `\`.
