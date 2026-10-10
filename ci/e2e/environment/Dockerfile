FROM ghcr.io/jdx/mise:2026.9.14 AS mise

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/root/.local/share/mise/shims:/root/.local/bin:${PATH}

WORKDIR /work
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    ca-certificates \
    clang \
    cmake \
    curl \
    g++ \
    gcc \
    git \
    libbz2-dev \
    libcurl4 \
    libffi-dev \
    libgomp1 \
    liblzma-dev \
    libnuma-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    make \
    ninja-build \
    pkg-config \
    procps \
    sudo \
    tk-dev \
    unzip \
    uuid-dev \
    xz-utils \
    zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

COPY --from=mise /usr/local/bin/mise /usr/local/bin/mise
COPY .mise.toml .mise.toml
RUN mise trust \
  && mise install \
  && mise reshim \
  && npm cache clean --force \
  && mise cache clear \
  && rm -rf /root/.local/share/mise/downloads /root/.npm

COPY ci/e2e/environment/smoke.sh /usr/local/bin/e2e-env-smoke
RUN chmod +x /usr/local/bin/e2e-env-smoke \
  && e2e-env-smoke
