#!/usr/bin/env python3
"""Record the README's opening command as an asciicast.

    python3 docs/record-launch.py docs/launch.cast
    agg --font-size 20 --theme dracula --speed 1.95 docs/launch.cast launch.gif
    gh release upload docs-assets launch.gif --clobber

Drives a real pty: only the keystrokes are scripted, the timing is the
machine's. The GIF is a release asset, so only the cast is committed.

Knobs are REC_-prefixed (REC_AGENT, REC_MODEL, REC_PROMPT, REC_BOOT_WAIT,
REC_ANSWER_WAIT, REC_HOLD, REC_WARM); bare names like AGENT are already taken.
"""

import fcntl
import json
import os
import select
import shlex
import signal
import struct
import sys
import termios
import time
import urllib.request

COLS, ROWS = 100, 28
DEFAULT_PORT = 17434
AGENT = os.environ.get("REC_AGENT", "claude")
MODEL = os.environ.get("REC_MODEL", "qwen3.8")
PROMPT = os.environ.get("REC_PROMPT", "explain what git stash does in one sentence")


def daemon_url(path):
    """Resolve LLMMAN_HOST the way the daemon does: [scheme://]host[:port][/path]."""
    raw = os.environ.get("LLMMAN_HOST", "").strip().strip("\"'")
    scheme, sep, rest = raw.partition("://")
    if not sep:
        scheme, rest = "http", raw
    hostport = rest.split("/", 1)[0] or "127.0.0.1"
    if ":" not in hostport.rsplit("]", 1)[-1]:  # colons inside [::1] are not a port
        port = {"http": 80, "https": 443}.get(scheme, DEFAULT_PORT) if sep else DEFAULT_PORT
        hostport = f"{hostport}:{port}"
    return f"{scheme}://{hostport}{path}"


def warm_up():
    """Load the model first, so the cast shows the warm start the README claims.

    Fatal on failure: a silent cold recording would contradict that.
    """
    body = json.dumps(
        {"model": MODEL, "prompt": "hi", "stream": False, "options": {"num_predict": 4}}
    ).encode()
    req = urllib.request.Request(
        daemon_url("/api/generate"),
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600):
            pass
    except OSError as err:
        raise SystemExit(f"warm-up failed ({err}); start the daemon, or set REC_WARM=0")


class Recorder:
    def __init__(self, fd):
        self.fd = fd
        self.events = []
        self.recording = False
        self.start = None
        self.eof = False

    def begin(self):
        self.events.clear()
        self.recording = True
        self.start = time.time()

    def _read(self):
        """Append one chunk of pty output. False once the child is gone."""
        try:
            data = os.read(self.fd, 65536)
        except OSError:
            data = b""
        if not data:
            self.eof = True
            return False
        if self.recording:
            self.events.append(
                [round(time.time() - self.start, 4), "o", data.decode("utf-8", "replace")]
            )
        return True

    def pump(self, duration):
        """Forward pty output for `duration` seconds."""
        deadline = time.time() + duration
        while not self.eof:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            if select.select([self.fd], [], [], min(0.05, remaining))[0]:
                self._read()

    def pump_until_quiet(self, max_wait, quiet=1.5):
        """Forward output until the agent goes quiet, i.e. has finished.

        It spins while working, and prefill varies by ten seconds run to run,
        which a fixed sleep would either cut off or pad with dead air.
        """
        deadline = time.time() + max_wait
        last = time.time()
        while not self.eof and time.time() < deadline:
            if select.select([self.fd], [], [], 0.1)[0]:
                if self._read():
                    last = time.time()
            elif time.time() - last >= quiet:
                return True
        return False

    def type(self, text, delay=0.045):
        for ch in text:
            os.write(self.fd, ch.encode())
            self.pump(delay)

    def enter(self):
        os.write(self.fd, b"\r")

    def hold(self, seconds):
        """Keep the finished answer on screen: a cast's length is its last event."""
        if self.events:
            self.events.append([round(self.events[-1][0] + seconds, 4), "o", "\x1b[s"])

    def write(self, path):
        header = {
            "version": 2,
            "width": COLS,
            "height": ROWS,
            "timestamp": int(self.start),
            "env": {"TERM": "xterm-256color", "SHELL": "/bin/bash"},
        }
        with open(path, "w") as fh:
            fh.write(json.dumps(header) + "\n")
            for event in self.events:
                fh.write(json.dumps(event) + "\n")


def record(rec):
    """Type the command, wait for the agent, ask it one thing. True if answered."""
    # Settle the shell and clear the banner without recording any of it.
    rec.pump(1.5)
    os.write(rec.fd, b"clear\r")
    rec.pump(1.0)

    if os.environ.get("REC_WARM", "1") == "1":
        warm_up()

    rec.begin()
    rec.type(f"llmman launch {shlex.quote(AGENT)} --model {shlex.quote(MODEL)}")
    rec.pump(0.7)
    rec.enter()

    # Daemon boots, llama-server loads the model, the agent execs against it.
    rec.pump(float(os.environ.get("REC_BOOT_WAIT", 4.5)))

    rec.type(PROMPT)
    rec.pump(0.5)
    rec.enter()

    finished = rec.pump_until_quiet(float(os.environ.get("REC_ANSWER_WAIT", 45)))
    rec.pump(0.9)
    rec.hold(float(os.environ.get("REC_HOLD", 2.2)))
    return finished


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/launch.cast"
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    env = dict(os.environ)
    env["PATH"] = os.path.join(repo, "target", "release") + os.pathsep + env["PATH"]
    env["PS1"] = "\\[\\033[1;36m\\]$\\[\\033[0m\\] "
    env["TERM"] = "xterm-256color"
    env["COLUMNS"], env["LINES"] = str(COLS), str(ROWS)
    # The agent's own update toast is unrelated chrome that lands mid-answer.
    env["DISABLE_AUTOUPDATER"] = "1"

    pid, fd = os.forkpty()
    if pid == 0:
        os.chdir(repo)
        os.execvpe("bash", ["bash", "--norc", "--noprofile", "-i"], env)

    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    rec = Recorder(fd)
    try:
        finished = record(rec)
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        except OSError:
            pass

    # Writing a truncated cast would let `&& agg ...` publish a half-answer.
    if not finished:
        raise SystemExit(f"answer did not finish within REC_ANSWER_WAIT; {out} not written")

    rec.write(out)
    print(f"\n[recorded {len(rec.events)} events, {rec.events[-1][0]:.1f}s -> {out}]")


if __name__ == "__main__":
    main()
