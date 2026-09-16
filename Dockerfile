# Use slim variant to reduce image size (200MB smaller).
# Python 3.12 matches the CI matrix; 3.10 was dropped in v7.5.1 because
# ``typing.Self`` (used in the NN metamodel, PEP 673) requires 3.11+.
FROM python:3.12-slim

# --- Optional corporate CA injection (build behind a TLS-inspecting proxy) ---
# Some sites (e.g. LIST laptops) build behind Netskope, which re-signs TLS with
# its own CA. The clean slim image does not trust it, so pip fails with
# "self-signed certificate in certificate chain". Drop the proxy's root +
# signing certs as .crt files in ca-certs-extra/ (gitignored, site-internal)
# and build with --build-arg TRUST_EXTRA_CAS=1.
#
# OPT-IN on purpose. deploy.sh builds from the local working tree, so an
# unconditional COPY would bake whatever .crt happens to be sitting in that
# directory into the image we ship to a shared host — making the deployed
# backend trust a corporate CA for every outbound TLS call it makes. Defaulting
# to 0 means production stays clean by construction, not by remembering to
# empty the directory first.
ARG TRUST_EXTRA_CAS=0
COPY ca-certs-extra/ /tmp/ca-certs-extra/
RUN if [ "$TRUST_EXTRA_CAS" = "1" ]; then \
        cp /tmp/ca-certs-extra/*.crt /usr/local/share/ca-certificates/ \
        && update-ca-certificates; \
    fi; \
    rm -rf /tmp/ca-certs-extra
# Point every toolchain at the system bundle: curl/apt use it already, but pip
# (PIP_CERT/REQUESTS_CA_BUNDLE), node/npm (NODE_EXTRA_CA_CERTS) and rustup
# (SSL_CERT_FILE) each ship their own trust store and must be told explicitly.
# Harmless when no extra CA was injected — this is the default bundle anyway.
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Phase 3 toolchains for per-project TS/Rust/Kotlin compile validation.
# Without these binaries on PATH, the Phase 3 validation loop soft-skips
# (shutil.which returns None), so generated nextjs/rust/spring-boot
# artifacts never get type-checked / cargo-checked / kotlinc-compiled
# and per-project compile-pass stays at 0/5. Pinned versions:
#   - Node.js 20.x (provides npm -> tsc)
#   - TypeScript 5.x (npm install -g typescript)
#   - Rust stable, minimal profile (rustup)
#   - OpenJDK 21 + Kotlin compiler 1.9.24 (matches stack_metadata.py;
#     python:3.10-slim is Debian Trixie which no longer ships JDK 17)
# Placed before requirements.txt copy so this slow layer caches well.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        unzip \
        build-essential \
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

# No additional system dependencies needed - Python slim has everything for a basic Flask/FastAPI app
# If you need specific system libraries (e.g., for image processing), add them here

# Copy and install dependencies first for better layer caching
COPY requirements.txt ./requirements.txt
COPY besser/utilities/web_modeling_editor/backend/requirements.txt ./backend-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r backend-requirements.txt

# Phase 3 Python verification. The Spec-Driven fix loop promotes ruff's
# undefined-name findings (F821/F822/F823) to BLOCKERS ("ships green, boots
# dead") — but ruff was only ever installed in CI, never in this image, so on
# the hosted backend _collect_ruff_issues() silently returned [] and two pilot
# runs shipped a backend that NameError'd on import as "success / 0 blockers".
# Pinned so the check is reproducible across deploys; kept current with the
# unpinned `pip install ruff` CI runs so both see the same rule semantics.
RUN pip install --no-cache-dir "ruff==0.16.6"

# Copy only necessary files
COPY pyproject.toml README.md ./
COPY besser/ ./besser/

# Install BESSER package
RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/app

EXPOSE 9000

CMD ["python", "-m", "besser.utilities.web_modeling_editor.backend.backend"]