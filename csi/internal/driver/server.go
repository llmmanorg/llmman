package driver

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"os"
	"runtime/debug"
	"strings"
	"time"

	"github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// parseEndpoint splits a CSI endpoint (`unix:///path/to.sock`, or a bare
// path) into the network and address to listen on. Only unix sockets are
// accepted: the socket's file permissions are the plugin's access control.
func parseEndpoint(endpoint string) (network, addr string, err error) {
	switch {
	case strings.HasPrefix(endpoint, "unix://"):
		addr = strings.TrimPrefix(endpoint, "unix://")
	case strings.HasPrefix(endpoint, "/"):
		addr = endpoint
	default:
		return "", "", fmt.Errorf("endpoint %q is not a unix socket (want unix:///path/to.sock)", endpoint)
	}
	if addr == "" || !strings.HasPrefix(addr, "/") {
		return "", "", fmt.Errorf("endpoint %q needs an absolute socket path", endpoint)
	}
	return "unix", addr, nil
}

// NewServer returns the gRPC server for d, logging every call.
func NewServer(d *Driver, log *slog.Logger) *grpc.Server {
	s := grpc.NewServer(grpc.ChainUnaryInterceptor(recoverCalls(log), logCalls(log)))
	csi.RegisterIdentityServer(s, d)
	csi.RegisterNodeServer(s, d)
	return s
}

// Serve serves d on endpoint until ctx is done, then stops gracefully. A
// stale socket file from an earlier run is replaced.
func Serve(ctx context.Context, endpoint string, d *Driver, log *slog.Logger) error {
	network, addr, err := parseEndpoint(endpoint)
	if err != nil {
		return err
	}
	if err := os.Remove(addr); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("remove stale socket: %w", err)
	}
	lis, err := net.Listen(network, addr)
	if err != nil {
		return err
	}
	defer os.Remove(addr)

	srv := NewServer(d, log)
	errc := make(chan error, 1)
	go func() { errc <- srv.Serve(lis) }()
	log.Info("serving", "endpoint", endpoint, "driver", d.cfg.Name, "node", d.cfg.NodeID)

	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
	}
	stopped := make(chan struct{})
	go func() { srv.GracefulStop(); close(stopped) }()
	select {
	case <-stopped:
	case <-time.After(10 * time.Second):
		srv.Stop()
	}
	return nil
}

func logCalls(log *slog.Logger) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		start := time.Now()
		resp, err := handler(ctx, req)
		level := slog.LevelDebug
		if strings.HasSuffix(info.FullMethod, "NodePublishVolume") || strings.HasSuffix(info.FullMethod, "NodeUnpublishVolume") {
			level = slog.LevelInfo
		}
		attrs := []any{"method", info.FullMethod, "duration", time.Since(start).Round(time.Millisecond)}
		if err != nil {
			log.Warn("call failed", append(attrs, "error", err)...)
		} else {
			log.Log(ctx, level, "call", attrs...)
		}
		return resp, err
	}
}

// recoverCalls turns a panic in a handler into an Internal error rather than
// a dead plugin: its mounts outlive it, but new pods would stop starting.
func recoverCalls(log *slog.Logger) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp any, err error) {
		defer func() {
			if r := recover(); r != nil {
				log.Error("panic", "method", info.FullMethod, "panic", r, "stack", string(debug.Stack()))
				err = status.Errorf(codes.Internal, "internal error in %s", info.FullMethod)
			}
		}()
		return handler(ctx, req)
	}
}
