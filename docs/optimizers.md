# Optimizers

`llmman launch <integration> --optimizer <name>` starts an integration with
something extra that makes the session cheaper to run.

| Optimizer | What it does |
|-----------|--------------|
| `none` | The default. The integration is launched exactly as without the flag: same arguments, same environment, same files. |
| `tk` | Compacts the output of the shell commands the agent runs, before the agent reads it. `claude` only. |

```sh
llmman launch claude --model qwen3.8 --optimizer tk
```

llmman prints one line to stderr when an optimizer is on, and nothing for
`none`:

```
[llmman] claude: optimizer tk (command output compacted; LLMMAN_TK=off disables)
```

## `tk`

A coding agent's shell tool returns the whole output of a command, and the
model pays for every line of it in context and tokens: a `git status` with
its hints, a test run that lists every passing test, a build that prints one
line per crate. `tk` runs the same command and prints a shorter version of
what it wrote. The exit status is the command's own.

### How it is wired in

Claude Code can run a `PreToolUse` hook before a tool call and replace the
tool's input. `llmman launch claude --optimizer tk` passes claude a
`--settings` argument, as inline JSON, that registers one such hook for the
Bash tool: `llmman tk hook`. Nothing is written to disk, and your
`~/.claude`, project settings and `CLAUDE_CONFIG_DIR` are not touched. The
settings apply to that session and are merged with yours.

For a Bash call that is one plain command with a filter, the hook answers
with the same command line behind `llmman tk`, using the absolute path of the
running `llmman`. The original text is appended unchanged, so its quoting is
exactly what the agent wrote:

```
git commit -m "fix: a 'quoted' thing"
  becomes
'/usr/bin/llmman' tk git commit -m "fix: a 'quoted' thing"
```

Anything else is left alone: pipes, `&&`, `||`, `;`, redirections,
subshells, command substitution, unquoted wildcards (`*`, `?`, `[`),
heredocs, variable assignments, commands over 10,000 bytes (UTF-8 bytes, not
characters, counted for the command and again once the `llmman tk` wrapper is
added), commands that are not on the list below, and anything the hook cannot
parse. On any error the hook prints nothing and exits 0, so it cannot fail a
tool call.

The wrapped command runs the program found on `PATH`, not a shell function
or alias of the same name.

`llmman tk <command> [args...]` also works on its own, for trying a filter.

### What is filtered

| Command | What changes |
|---------|--------------|
| `git status` | Hint lines and blank lines go; long file lists are cut with a count. |
| `git diff` | `index` lines go, long lines are cut, and a file's diff is cut after a fixed number of lines with a marker naming the file. |
| `git log` | One line per commit (short hash, decorations, subject, author, date) and the first lines of its message. |
| `git commit -m`, `push`, `pull` | Progress, diffstat and file-mode lines go; the summary, ref updates and remote messages stay. |
| `cargo build`, `check`, `clippy`, `test` | `Compiling` and download lines are counted; the lines of tests that passed go. Warnings, errors, panics and failure reports stay whole. |
| `go build`, `vet`, `install`, `test` | Module download lines and, for `test`, passing-test narration and packages without tests are counted. A `=== RUN` line stays when the test printed something after it. |
| `pytest` | The session banner, progress lines and (under `-v`) `PASSED` lines go. From the first `=====` rule on (failures, captured output, summary) nothing changes. |
| `npm`, `pnpm`, `yarn test` | The package manager's banner and passing-test lines go; everything else stays, on both streams. |
| `grep`, `rg` | Matches are grouped under their file, long lines cut. Only when every line is a `path:` of a file that exists; a search of one file is left as it is. |
| `docker ps` | The COMMAND and CREATED columns and duplicate IPv6 port entries go. |

A command that has no filter of its own is not rewritten by the hook. Run by
hand, `llmman tk` gives it a generic clean-up: escape sequences and progress
bar redraws removed, runs of identical lines folded into one with a count
(`(x5)`), runs of blank lines folded, very long lines cut, and a long output
reduced to its head and tail with a `... N lines omitted ...` marker.

### Left untouched

These run exactly as without `tk`, on the terminal's own streams:

- Commands asked for output a program reads: `--json`, `--porcelain`,
  `--format`, `--message-format`, `--template`, `-z`, `--null`, and the
  summary-only forms such as `git diff --stat`, `git status -s`, `git log
  --oneline`, `grep -c`, `grep -l`, `grep -o`, `docker ps -q`.
- Git plumbing and everything else in `git` that is not in the table:
  `git rev-parse`, `git show`, `git ls-files`, `git branch`, and so on.
- `cat`, `head`, `tail`, `sed`, `jq`, `curl` and other commands whose output
  is the point.
- Interactive commands, and commands that never end: `git commit` without a
  message, `git pull --edit` or `--rebase=interactive`, `git rebase -i`,
  `--watch`, `docker logs -f`. `llmman tk` also runs everything untouched
  when stdin is a terminal.
- Everything, when `LLMMAN_TK=off`.

### What it guarantees

- The exit status is the command's. A signal is `128 + n`, as in a shell; a
  missing program is 127 and one that cannot run is 126.
- A command that failed keeps its error text, within the bounds every output
  has. A stream of more than 600 lines is cut to its first 350 and last 200
  with a marker, and a line of more than 2,000 characters is cut with a
  count, so an error in the middle of a long failed run can be among what is
  cut (see Bypassing it for the whole text). Within those bounds, filters
  shorten; they do not hide. Whatever a filter returns, a stream that had
  text is never left empty. Past the bounded memory below, text in the
  middle is gone too, and a marker says how many bytes.
- Output of a shape a filter does not recognize comes back as it was, apart
  from the generic clean-up.
- Output is read from both streams while the command runs, in bounded memory,
  so a command that prints gigabytes does not stall or exhaust the machine.
  Only the start and the end (4 MiB each) are kept.
- A process the command leaves running does not hold its output back: one
  second after the command exits, what was read is shown, with a note on
  stderr that what the process writes later is not.

### Permissions

Claude Code checks its permission rules against the command a hook returns,
not the one the agent wrote. So the hook does the following:

- It reads the `permissions` of the user, project, project-local and managed
  settings files it can see. When a deny or ask rule could apply to a program
  (any `Bash(<program> ...)` rule for it, a rule with a wildcard where the
  program would be, or a `Read`, `Edit`, `Write`, `Grep` or `Glob` rule), it
  leaves that program alone. When a settings file exists but cannot be read
  or parsed, or a managed policy directory cannot be listed, it rewrites
  nothing.
- It never approves a command on its own account, with two exceptions. A
  command an allow rule of yours already covers keeps that decision after the
  rewrite, so it does not start to ask. And plain `git status`, `git diff`
  and `git log` (no option before the subcommand, no path that leaves the
  working directory), which Claude Code runs without asking, still do.
- Any other rewritten command goes through Claude Code's normal permission
  flow and, if it would have asked, asks, showing the rewritten command.

Rules Claude Code gets from somewhere the hook cannot read, such as an
organization's managed policy delivered by MDM or the admin console, are not
seen. If you rely on those, do not use `tk`.

### Bypassing it

- `LLMMAN_TK=off` in the environment (of the session, or of one command)
  makes `llmman tk` a pure passthrough, and the hook stops rewriting.
- Start a command with a variable assignment (`X=1 git diff`), or pipe it
  (`git diff | cat`): the hook does not rewrite either.
- Launch without `--optimizer`.

### Limitations

- Only `claude`. Any other integration is refused with `--optimizer tk`
  before the daemon starts.
- Only the Bash tool. Claude's built-in Read, Grep and Glob tools do not run
  shell commands and are not affected, nor are commands started by the Monitor
  tool.
- Not with `--sandbox`: the sandbox does not have this machine's `llmman`.
- Not on Windows.
- Not with `--settings`, `--bare`, `--disallowedTools` or `--setting-sources`
  after `--`. claude keeps only the last `--settings`, so the hook and yours
  could not both apply; `--bare` skips hooks; deny rules from
  `--disallowedTools` cannot be seen by the hook; and the hook reads project
  settings files that `--setting-sources` can tell claude to skip. Put
  settings and rules in a settings file instead.
- It is not a `[launch.<integration>]` default in `llmman.conf`; type the
  flag.
- One plain command per call. A chain such as `cd sub && cargo test` is left
  as it is.
- When a filter cuts a diff or a list, the agent sees a marker, and the full
  output is one pipe away (`git diff | cat`).

### How much it saves

It depends on the command: a passing test run with hundreds of tests shrinks
a lot, a short `git status` hardly at all. No figure is claimed here. The
`noisy_fixtures_shrink` test in `src/tk/tests.rs` runs fixtures of noisy
output (a passing `cargo test`, `pytest`, `go test`, a `git status` with its
hints, a repeated line) through the filters and asserts that each comes out
smaller; `cargo test --lib noisy_fixtures_shrink` runs it.
