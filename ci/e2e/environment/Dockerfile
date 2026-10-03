FROM ghcr.io/jdx/mise:2026.9.14 AS mise

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MISE_DATA_DIR=/root/.local/share/mise
ENV MISE_CONFIG_DIR=/root/.config/mise
ENV MISE_CACHE_DIR=/root/.cache/mise
ENV MISE_TRUSTED_CONFIG_PATHS=/work/.mise.toml
ENV PATH=/root/.local/share/mise/shims:/root/.local/bin:/opt/e2e/bin:${PATH}

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
    tk-dev \
    unzip \
    uuid-dev \
    xz-utils \
    zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY .mise.toml .mise.toml
COPY ci/e2e/environment/manifest.json ci/e2e/environment/manifest.json
COPY ci/e2e/environment/install-npm-tools.js ci/e2e/environment/install-npm-tools.js
COPY --from=mise /usr/local/bin/mise /usr/local/bin/mise

RUN mise install \
  && mise reshim \
  && go version \
  && cargo --version \
  && rustc --version \
  && node --version \
  && npm --version \
  && bun --version \
  && python3 --version \
  && node ci/e2e/environment/install-npm-tools.js > /tmp/e2e-npm-install.sh

RUN sh /tmp/e2e-npm-install.sh \
  && omp --version \
  && claude --version \
  && opencode --version \
  && codex --version \
  && cline --version \
  && pi --version \
  && qwen --version \
  && openclaw --version \
  && dsh --version

RUN mkdir -p /opt/e2e/bin

COPY ci/e2e/environment/smoke.sh /usr/local/bin/e2e-env-smoke
RUN chmod +x /usr/local/bin/e2e-env-smoke \
  && e2e-env-smoke
