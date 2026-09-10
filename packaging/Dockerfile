# A ghcr.io/ggml-org/llama.cpp:server-<backend> image (the ones `llmman
# serve --ociman` picks, src/container.rs) plus the llmman release binary,
# published as docker.io/ai/llmman:<same tag>. Same entrypoint as the base,
# so it drops in for both --ociman paths; the daemon itself runs with
# `--entrypoint llmman ... serve`. CI passes the pinned base; the default
# is for local builds.
#
# The context holds executable llmman-amd64 and llmman-arm64 (TARGETARCH
# names) from the build job. Nothing compiles, so no emulation; and with
# --link the base is never unpacked, only forwarded to the registry
# (server-rocm is 7 GiB compressed). --chmod would force the unpack, hence
# the mode coming from the context.
ARG BASE=ghcr.io/ggml-org/llama.cpp:server
FROM ${BASE}
ARG TARGETARCH
ARG VERSION
ARG REVISION
LABEL org.opencontainers.image.title="llmman" \
      org.opencontainers.image.description="Run any agent on any model, models stored as OCI images" \
      org.opencontainers.image.url="https://github.com/llmmanorg/llmman" \
      org.opencontainers.image.source="https://github.com/llmmanorg/llmman" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"
COPY --link llmman-${TARGETARCH} /usr/local/bin/llmman
# llama-server and its libraries are in /app. A loopback bind inside a
# container is unreachable even with -p, so bind everything; the daemon
# then requires LLMMAN_API_KEYS unless LLMMAN_AUTH=off.
ENV PATH=/app:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LLMMAN_HOST=0.0.0.0:17434
EXPOSE 17434
