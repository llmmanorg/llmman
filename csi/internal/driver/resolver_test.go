package driver

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fakeRunner returns canned output and records what it was asked to run.
type fakeRunner struct {
	name   string
	args   []string
	stdout string
	stderr string
	err    error
}

func (r *fakeRunner) Run(_ context.Context, name string, args []string, stderr io.Writer) ([]byte, error) {
	r.name, r.args = name, args
	if r.stderr != "" {
		_, _ = io.WriteString(stderr, r.stderr)
	}
	return []byte(r.stdout), r.err
}

func TestLlmmanResolverRunsResolveDirAndReadsItsJSON(t *testing.T) {
	dir := t.TempDir()
	runner := &fakeRunner{stdout: `{"reference":"x","dir":"` + dir + `","digest":"sha256:abc","format":"gguf"}` + "\n"}
	r := &LlmmanResolver{Bin: "/usr/bin/llmman", Runner: runner}
	tree, err := r.Resolve(context.Background(), "reg.example.com/m:v1")
	if err != nil {
		t.Fatal(err)
	}
	if tree != (Tree{Dir: dir, Digest: "sha256:abc", Format: "gguf"}) {
		t.Fatalf("tree = %+v", tree)
	}
	// "--" before the reference: nothing in it can be an option.
	want := []string{"resolve", "--dir", "--", "reg.example.com/m:v1"}
	if runner.name != "/usr/bin/llmman" || !reflect.DeepEqual(runner.args, want) {
		t.Fatalf("ran %s %v, want %v", runner.name, runner.args, want)
	}
}

func TestLlmmanResolverErrors(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "f")
	if err := os.WriteFile(file, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	cases := map[string]struct {
		runner *fakeRunner
		want   string
	}{
		"failure carries llmman's own words": {
			&fakeRunner{err: errors.New("exit status 1"), stderr: "Error: model not found: reg/m\n"},
			"model not found: reg/m",
		},
		"not json":     {&fakeRunner{stdout: "hello"}, "unreadable llmman output"},
		"relative dir": {&fakeRunner{stdout: `{"dir":"cache/abc"}`}, "not an absolute directory"},
		"missing dir":  {&fakeRunner{stdout: `{"dir":"` + dir + `/nope"}`}, "not a directory here"},
		"a file":       {&fakeRunner{stdout: `{"dir":"` + file + `"}`}, "not a directory here"},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := (&LlmmanResolver{Bin: "llmman", Runner: c.runner}).Resolve(context.Background(), "m")
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("error = %v, want one containing %q", err, c.want)
			}
		})
	}
}

func TestLlmmanResolverReportsLlmmansOwnErrorLine(t *testing.T) {
	runner := &fakeRunner{
		err:    errors.New("exit status 1"),
		stderr: "time=now level=info msg=\"host will try HTTPS first\"\nError: pulling reg/m:v1: unauthorized\n",
	}
	_, err := (&LlmmanResolver{Bin: "llmman", Runner: runner}).Resolve(context.Background(), "m")
	if err == nil || err.Error() != "exit status 1: pulling reg/m:v1: unauthorized" {
		t.Fatalf("error = %v", err)
	}
}

func TestLlmmanResolverKeepsOnlyTheEndOfLongStderr(t *testing.T) {
	runner := &fakeRunner{err: errors.New("exit status 1"), stderr: strings.Repeat("progress\n", 1000) + "no error line, just the end\n"}
	_, err := (&LlmmanResolver{Bin: "llmman", Runner: runner}).Resolve(context.Background(), "m")
	if err == nil || !strings.Contains(err.Error(), "just the end") || len(err.Error()) > stderrTail+200 {
		t.Fatalf("error is %d bytes: %.80s...", len(err.Error()), err)
	}
}

func TestExecRunnerCapturesStdoutAndStreamsStderr(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("needs sh")
	}
	var stderr bytes.Buffer
	out, err := ExecRunner{}.Run(context.Background(), "sh", []string{"-c", `echo '{"a":1}'; echo progress >&2`}, &stderr)
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(string(out)) != `{"a":1}` || strings.TrimSpace(stderr.String()) != "progress" {
		t.Fatalf("stdout %q stderr %q", out, stderr.String())
	}
	_, err = ExecRunner{}.Run(context.Background(), "sh", []string{"-c", "exit 3"}, io.Discard)
	if err == nil || !strings.Contains(err.Error(), "exit status 3") {
		t.Fatalf("error = %v", err)
	}
	_, err = ExecRunner{}.Run(context.Background(), "/nonexistent/llmman", []string{"resolve"}, io.Discard)
	if err == nil {
		t.Fatal("ran a binary that does not exist")
	}
}

func TestExecRunnerStopsTheCommandWhenItsContextIsCancelled(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("needs sh")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err := ExecRunner{}.Run(ctx, "sh", []string{"-c", "exec sleep 30"}, io.Discard)
	if err == nil || time.Since(start) > 5*time.Second {
		t.Fatalf("returned after %v with %v", time.Since(start), err)
	}
}

type countingResolver struct {
	calls atomic.Int32
	gate  chan struct{}
	tree  Tree
	err   error
}

func (r *countingResolver) Resolve(ctx context.Context, _ string) (Tree, error) {
	r.calls.Add(1)
	if r.gate != nil {
		select {
		case <-r.gate:
		case <-ctx.Done():
			return Tree{}, ctx.Err()
		}
	}
	return r.tree, r.err
}

func TestSingleFlightSharesOneCallPerReference(t *testing.T) {
	inner := &countingResolver{gate: make(chan struct{}), tree: Tree{Dir: "/d"}}
	sf := NewSingleFlight(context.Background(), inner)
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if tree, err := sf.Resolve(context.Background(), "a"); err != nil || tree.Dir != "/d" {
				t.Errorf("tree %+v err %v", tree, err)
			}
		}()
	}
	// a different reference is its own call
	other := make(chan struct{})
	go func() { _, _ = sf.Resolve(context.Background(), "b"); close(other) }()
	waitFor(t, func() bool { return inner.calls.Load() == 2 && sf.callers("a") == 10 })
	close(inner.gate)
	wg.Wait()
	<-other
	if n := inner.calls.Load(); n != 2 {
		t.Fatalf("calls = %d, want 2", n)
	}
	// finished: the next caller runs it again (llmman finds it present)
	if _, err := sf.Resolve(context.Background(), "a"); err != nil {
		t.Fatal(err)
	}
	if n := inner.calls.Load(); n != 3 {
		t.Fatalf("calls = %d, want 3", n)
	}
}

func TestSingleFlightSharesFailuresAndForgetsThem(t *testing.T) {
	inner := &countingResolver{err: errBoom}
	sf := NewSingleFlight(context.Background(), inner)
	if _, err := sf.Resolve(context.Background(), "a"); !errors.Is(err, errBoom) {
		t.Fatalf("err = %v", err)
	}
	inner.err = nil
	if _, err := sf.Resolve(context.Background(), "a"); err != nil {
		t.Fatalf("a failure was remembered: %v", err)
	}
}

func TestSingleFlightWorkOutlivesACallerButNotShutdown(t *testing.T) {
	inner := &countingResolver{gate: make(chan struct{})}
	base, shutdown := context.WithCancel(context.Background())
	sf := NewSingleFlight(base, inner)

	caller, cancel := context.WithCancel(context.Background())
	got := make(chan error, 1)
	go func() { _, err := sf.Resolve(caller, "a"); got <- err }()
	waitFor(t, func() bool { return inner.calls.Load() == 1 })
	cancel()
	if err := <-got; !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	// still running: a second caller joins rather than starting another
	second := make(chan error, 1)
	go func() { _, err := sf.Resolve(context.Background(), "a"); second <- err }()
	waitFor(t, func() bool { return sf.callers("a") == 2 })
	if n := inner.calls.Load(); n != 1 {
		t.Fatalf("calls = %d", n)
	}
	shutdown()
	if err := <-second; !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want the shutdown", err)
	}
}
