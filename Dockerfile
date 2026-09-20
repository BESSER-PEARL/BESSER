# slim: 200MB smaller. 3.12 matches CI; 3.11+ is required for typing.Self
# (NN metamodel, PEP 673).
FROM python:3.12-slim

# Build behind a TLS-inspecting proxy: drop its root + signing certs into
# ca-certs-extra/ (gitignored) and pass --build-arg TRUST_EXTRA_CAS=1.
# Opt-in because deploy.sh builds from the working tree, so an unconditional
# COPY would ship whatever .crt happens to sit there to a shared host.
ARG TRUST_EXTRA_CAS=0
COPY ca-certs-extra/ /tmp/ca-certs-extra/
RUN if [ "$TRUST_EXTRA_CAS" = "1" ]; then \
        cp /tmp/ca-certs-extra/*.crt /usr/local/share/ca-certificates/ \
        && update-ca-certificates; \
    fi; \
    rm -rf /tmp/ca-certs-extra
# pip, npm and rustup each ship their own trust store and must be pointed at
# the system bundle explicitly. No-op when no extra CA was injected.
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Phase 3 compile validation soft-skips when these are missing from PATH
# (shutil.which -> None), so nextjs/rust/spring-boot output ships unchecked.
# JDK 21 because Debian Trixie no longer packages 17. Kept before the
# requirements copy so this slow layer caches.
#
# bubblewrap: run_command executes model-authored shell, and without it that
# shell could read a sibling run's workspace and /proc/1/environ. The worker
# fails closed when bwrap is missing, so this package is load-bearing, not
# optional -- and the worker needs seccomp=unconfined to use it (see
# docker-compose.prod.yml).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        unzip \
        build-essential \
        bubblewrap \
        openjdk-21-jdk-headless \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g typescript \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal --no-modify-path \
    && curl -fsSL -o /tmp/kotlinc.zip \
        https://github.com/JetBrains/kotlin/releases/download/v1.9.24/kotlin-compiler-1.9.24.zip \
    && unzip -q /tmp/kotlinc.zip -d /opt \
    && rm /tmp/kotlinc.zip \
    && apt-get purge -y --auto-remove unzip \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.cargo/bin:/opt/kotlinc/bin:${PATH}"

WORKDIR /app

# Dependencies first for layer caching.
COPY requirements.txt ./requirements.txt
COPY besser/utilities/web_modeling_editor/backend/requirements.txt ./backend-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r backend-requirements.txt

# The fix loop treats ruff's F821/F822/F823 as blockers, but ruff was only in
# CI — so _collect_ruff_issues() returned [] here and two pilot runs shipped a
# backend that NameError'd on import as "0 blockers". Pinned for reproducibility.
RUN pip install --no-cache-dir "ruff==0.16.6"

COPY pyproject.toml README.md ./
COPY besser/ ./besser/
RUN pip install --no-cache-dir -e .

# A build-time CA must not become runtime trust. Unconditional, and the grep
# is an assertion: the build fails rather than ship an image trusting the proxy.
RUN rm -f /usr/local/share/ca-certificates/*.crt \
    && update-ca-certificates --fresh >/dev/null 2>&1 \
    && ! grep -qi goskope /etc/ssl/certs/ca-certificates.crt

ENV PYTHONPATH=/app

EXPOSE 9000

CMD ["python", "-m", "besser.utilities.web_modeling_editor.backend.backend"]