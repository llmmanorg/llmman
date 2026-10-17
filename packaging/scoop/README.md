# llmmanorg/scoop-bucket

Scoop bucket for [llmman](https://github.com/llmmanorg/llmman) — run any
agent on any model, models stored as OCI images.

## Install

```powershell
scoop bucket add llmman https://github.com/llmmanorg/scoop-bucket
scoop install llmman
```

Supported platforms (the platforms llmman publishes Windows builds for):

| Platform | Architecture |
|---|---|
| Windows | `x64` |
| Windows | `arm64` |

The binaries link against the Visual C++ runtime (`VCRUNTIME140.dll`). It
is present on almost every Windows machine; if `llmman` fails to start
because it is missing, `scoop install extras/vcredist2022` provides it
(after `scoop bucket add extras`).

## Run in the background

The CLI starts the daemon on demand. To keep it running across logons, register
a Task Scheduler task that runs `llmman serve` when you log on (PowerShell, no
elevation needed):

```powershell
$action   = New-ScheduledTaskAction -Execute (Get-Command llmman).Source -Argument 'serve'
$trigger  = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName llmman -Action $action -Trigger $trigger -Settings $settings
Start-ScheduledTask -TaskName llmman    # now, without logging off
```

`-ExecutionTimeLimit ([TimeSpan]::Zero)` matters: a task is otherwise ended after
72 hours. The task runs in a console window of its own, which is its log;
minimise it. It listens on `http://127.0.0.1:17434`; to change that, set
`LLMMAN_HOST` (plus `LLMMAN_API_KEYS` for a non-loopback bind) for your user,
log off and on, and set the same variables in the shell you run the CLI from so
it talks to the daemon instead of starting its own:

```powershell
[Environment]::SetEnvironmentVariable('LLMMAN_HOST', '0.0.0.0:17434', 'User')
```

After `scoop update llmman`, run `Stop-ScheduledTask llmman` and
`Start-ScheduledTask llmman` before any other `llmman` command: the CLI replaces
a daemon of another version with a detached one, which would leave the task's
`llmman serve` unable to bind. Remove it with
`Unregister-ScheduledTask -TaskName llmman -Confirm:$false`.

## Versions

Every commit that passes CI on llmman's `main` is a release, versioned
`MAJOR.MINOR.<commit count>` (e.g. `0.1.324`), and this manifest is updated
to it within minutes. `scoop update llmman` therefore tracks `main`; there
is no separate stable channel.

## This manifest is generated

`bucket/llmman.json` is rendered by
[`packaging/render.sh`](https://github.com/llmmanorg/llmman/blob/main/packaging/render.sh)
in the main repo and pushed here by its CI on every release. **Edits made
directly in this repo are overwritten by the next release** — change
`packaging/scoop/bucket/llmman.json.in` upstream instead.
