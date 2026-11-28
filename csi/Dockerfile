# The Kubernetes CSI node plugin (csi/) and the llmman binary it runs, in
# one image: docker.io/ai/llmman-csi:<same tag as the release>. The plugin
# serves the CSI Identity and Node services; each volume it publishes runs
# `llmman resolve --dir` (pull, verify, unpack) and bind-mounts the result.
#
# Build with this directory as the context. Next to the Go sources it holds
# the executable llmman-amd64 and llmman-arm64 (TARGETARCH names) from the
# build job, as packaging/Dockerfile's context does; nothing compiles
# llmman here. The binaries need glibc 2.28+.
#
#   cp ../target/release/llmman llmman-amd64
#   docker build -t llmman-csi .
ARG BASE=ubuntu:26.04
ARG GO_IMAGE=golang:1.27

# The plugin is static (no cgo) and cross-compiled on the build host.
FROM --platform=$BUILDPLATFORM ${GO_IMAGE} AS build
ARG TARGETOS
ARG TARGETARCH
ARG VERSION=dev
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY cmd ./cmd
COPY internal ./internal
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
    go build -trimpath -ldflags "-s -w -X main.version=${VERSION}" \
      -o /out/llmman-csi ./cmd/llmman-csi

FROM ${BASE}
ARG TARGETARCH
ARG VERSION
ARG REVISION
LABEL org.opencontainers.image.title="llmman-csi" \
      org.opencontainers.image.description="Kubernetes CSI driver that mounts ModelPack model artifacts, pulled with llmman" \
      org.opencontainers.image.url="https://github.com/llmmanorg/llmman" \
      org.opencontainers.image.source="https://github.com/llmmanorg/llmman" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"
# TLS roots for the registries llmman pulls from, taken from the build image
# rather than installed with apt: nothing in this stage runs, so a
# multi-platform build needs no emulation (as packaging/Dockerfile's).
COPY --link --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --link llmman-${TARGETARCH} /usr/local/bin/llmman
COPY --link --from=build /out/llmman-csi /usr/local/bin/llmman-csi
ENV LLMMAN_BIN=/usr/local/bin/llmman
ENTRYPOINT ["/usr/local/bin/llmman-csi"]
