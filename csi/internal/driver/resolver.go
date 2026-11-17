package driver

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Tree is a model's complete directory, as `llmman resolve --dir` reports it.
type Tree struct {
	// Dir is the directory: every layer of the artifact at its path.
	Dir string `json:"dir"`
	// Digest is the manifest the directory was built from.
	Digest string `json:"digest"`
	// Format is what llmman would serve the artifact as ("gguf",
	// "safetensors", ...), or "unknown".
	Format string `json:"format"`
}

// Resolver turns an artifact reference into its model directory on this
// node, pulling the artifact first if the node does not have it.
type Resolver interface {
	Resolve(ctx context.Context, reference string) (Tree, error)
}

// Runner runs a command to completion and returns what it wrote to stdout.
// What it wrote to stderr (progress, notices) goes to stderr as it comes.
type Runner interface {
	Run(ctx context.Context, name string, args []string, stderr io.Writer) (stdout []byte, err error)
}

// LlmmanResolver resolves references by running `llmman resolve --dir`.
// llmman does the registry authentication (the Docker config of the
// process's environment), the signature policy, the pull, and the unpacking.
type LlmmanResolver struct {
	Bin    string
	Runner Runner
	Log    *slog.Logger
	// Timeout bounds one run; zero means no limit.
	Timeout time.Duration
}

// stderrTail is how much of llmman's stderr an error carries.
const stderrTail = 2048

func (r *LlmmanResolver) Resolve(ctx context.Context, reference string) (Tree, error) {
	if r.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, r.Timeout)
		defer cancel()
	}
	if r.Log != nil {
		r.Log.Info("resolving", "reference", reference)
	}
	tail := &tailWriter{max: stderrTail}
	stderr := io.MultiWriter(tail, &logWriter{log: r.Log, reference: reference})
	// "--" so that nothing in a reference can be read as an option.
	out, err := r.Runner.Run(ctx, r.Bin, []string{"resolve", "--dir", "--", reference}, stderr)
	if err != nil {
		if msg := failureMessage(tail.String()); msg != "" {
			return Tree{}, fmt.Errorf("%w: %s", err, msg)
		}
		return Tree{}, err
	}
	var tree Tree
	if err := json.Unmarshal(bytes.TrimSpace(out), &tree); err != nil {
		return Tree{}, fmt.Errorf("unreadable llmman output %q: %w", truncate(string(out), 200), err)
	}
	if !filepath.IsAbs(tree.Dir) {
		return Tree{}, fmt.Errorf("llmman reported %q, which is not an absolute directory", tree.Dir)
	}
	if st, err := os.Stat(tree.Dir); err != nil || !st.IsDir() {
		return Tree{}, fmt.Errorf("llmman reported %s, which is not a directory here (is the model store mounted in this container?)", tree.Dir)
	}
	return tree, nil
}

// failureMessage is what llmman says went wrong: its last "Error: " line,
// else the end of what it wrote.
func failureMessage(stderr string) string {
	lines := strings.Split(strings.TrimSpace(stderr), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if msg, ok := strings.CutPrefix(strings.TrimSpace(lines[i]), "Error: "); ok {
			return msg
		}
	}
	return strings.TrimSpace(stderr)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// ExecRunner runs commands with os/exec.
type ExecRunner struct{}

func (ExecRunner) Run(ctx context.Context, name string, args []string, stderr io.Writer) ([]byte, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = stderr
	// Give llmman a moment to stop on cancellation, then kill it and the
	// pipes with it.
	cmd.Cancel = func() error { return cmd.Process.Signal(os.Interrupt) }
	cmd.WaitDelay = 10 * time.Second
	if err := cmd.Run(); err != nil {
		var exit *exec.ExitError
		if errors.As(err, &exit) {
			return nil, fmt.Errorf("%s %s: %w", name, args[0], err)
		}
		return nil, err
	}
	return stdout.Bytes(), nil
}

// tailWriter keeps the last max bytes written to it.
type tailWriter struct {
	mu  sync.Mutex
	max int
	buf []byte
}

func (t *tailWriter) Write(p []byte) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.buf = append(t.buf, p...)
	if len(t.buf) > t.max {
		t.buf = t.buf[len(t.buf)-t.max:]
	}
	return len(p), nil
}

func (t *tailWriter) String() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return string(t.buf)
}

// logWriter logs each line llmman writes to stderr.
type logWriter struct {
	log       *slog.Logger
	reference string
	mu        sync.Mutex
	partial   []byte
}

func (w *logWriter) Write(p []byte) (int, error) {
	if w.log == nil {
		return len(p), nil
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.partial = append(w.partial, p...)
	for {
		i := bytes.IndexByte(w.partial, '\n')
		if i < 0 {
			break
		}
		if line := strings.TrimSpace(string(w.partial[:i])); line != "" {
			w.log.Info("llmman", "reference", w.reference, "output", line)
		}
		w.partial = w.partial[i+1:]
	}
	return len(p), nil
}

// SingleFlight shares one Resolve per reference among concurrent callers.
//
// The work runs under the context given to NewSingleFlight, not under any
// caller's: a kubelet gives up on a NodePublishVolume after a couple of
// minutes and calls again, and a pull of a large model must not be started
// over each time. A caller that stops waiting leaves the work running for the
// next one.
type SingleFlight struct {
	inner Resolver
	base  context.Context

	mu       sync.Mutex
	inflight map[string]*flight
}

type flight struct {
	done    chan struct{}
	tree    Tree
	err     error
	callers int
}

// NewSingleFlight wraps inner. base bounds the lifetime of every call made to
// it: cancelling base (at shutdown) stops them.
func NewSingleFlight(base context.Context, inner Resolver) *SingleFlight {
	return &SingleFlight{inner: inner, base: base, inflight: map[string]*flight{}}
}

func (s *SingleFlight) Resolve(ctx context.Context, reference string) (Tree, error) {
	s.mu.Lock()
	f, running := s.inflight[reference]
	if !running {
		f = &flight{done: make(chan struct{})}
		s.inflight[reference] = f
		go func() {
			f.tree, f.err = s.inner.Resolve(s.base, reference)
			s.mu.Lock()
			delete(s.inflight, reference)
			s.mu.Unlock()
			close(f.done)
		}()
	}
	f.callers++
	s.mu.Unlock()

	select {
	case <-f.done:
		return f.tree, f.err
	case <-ctx.Done():
		return Tree{}, ctx.Err()
	}
}
