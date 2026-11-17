// Command llmman-csi is the Kubernetes CSI node plugin that mounts ModelPack
// model artifacts into pods. See docs/kubernetes.md.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/llmmanorg/llmman/csi/internal/driver"
)

// version is set at build time with -ldflags "-X main.version=...".
var version = "dev"

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	var (
		endpoint    = flag.String("endpoint", env("CSI_ENDPOINT", "unix:///csi/csi.sock"), "CSI endpoint to serve: unix:///path/to.sock")
		nodeID      = flag.String("node-id", env("CSI_NODE_ID", os.Getenv("NODE_NAME")), "node ID reported to the orchestrator (the node's name)")
		name        = flag.String("driver-name", env("CSI_DRIVER_NAME", driver.DefaultName), "CSI driver name; must match the CSIDriver object")
		llmmanBin   = flag.String("llmman-bin", env("LLMMAN_BIN", "llmman"), "llmman binary (path, or name in PATH)")
		kubeletDir  = flag.String("kubelet-dir", env("KUBELET_DIR", "/var/lib/kubelet"), "kubelet root directory; target paths must be under it (empty: any)")
		pullTimeout = flag.Duration("pull-timeout", time.Hour, "longest a single pull and unpack may take (0: no limit)")
		logLevel    = flag.String("log-level", env("LOG_LEVEL", "info"), "debug, info, warn or error")
		showVersion = flag.Bool("version", false, "print the version and exit")
	)
	flag.Parse()
	if *showVersion {
		fmt.Println("llmman-csi", version)
		return
	}

	var level slog.Level
	if err := level.UnmarshalText([]byte(*logLevel)); err != nil {
		fmt.Fprintf(os.Stderr, "llmman-csi: bad --log-level: %v\n", err)
		os.Exit(2)
	}
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	resolver := driver.NewSingleFlight(ctx, &driver.LlmmanResolver{
		Bin:     *llmmanBin,
		Runner:  driver.ExecRunner{},
		Log:     log,
		Timeout: *pullTimeout,
	})
	d, err := driver.New(driver.Config{
		Name:       *name,
		Version:    version,
		NodeID:     *nodeID,
		LlmmanBin:  *llmmanBin,
		KubeletDir: *kubeletDir,
		Logger:     log,
		Resolver:   resolver,
		Mounter:    driver.NewSystemMounter(),
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "llmman-csi: %v\n", err)
		os.Exit(2)
	}
	if err := driver.Serve(ctx, *endpoint, d, log); err != nil {
		log.Error("serve", "error", err)
		os.Exit(1)
	}
}
