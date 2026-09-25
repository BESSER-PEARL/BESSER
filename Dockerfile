# slim: 200MB smaller. 3.12 matches CI; 3.11+ is required for typing.Self
# (NN metamodel, PEP 673).
FROM python:3.12-slim

# Build behind a TLS-inspecting proxy: drop its root + signing certs into
# ca-certs-extra/ (gitignored) and pass --build-arg TRUST_EXTRA_CAS=1.
# Opt-in so that a stray .crt in a local checkout never reaches an image
# built for deployment.
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
# bubblewrap confines run_command's model-authored shell to its own run
# directory. The worker fails closed without it, and needs
# seccomp=unconfined to use it (see docker-compose.prod.yml).
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

# The fix loop treats ruff's F821/F822/F823 as blockers; without ruff in the
# image those checks silently report nothing. Pinned for reproducibility.
RUN pip install --no-cache-dir "ruff==0.16.6"

COPY pyproject.toml README.md ./
COPY besser/ ./besser/
RUN pip install --no-cache-dir -e .

# A build-time CA must not become runtime trust. Unconditional; the grep is an
# assertion that fails the build if a known TLS-inspection CA is still trusted.
RUN rm -f /usr/local/share/ca-certificates/*.crt \
    && update-ca-certificates --fresh >/dev/null 2>&1 \
    && ! grep -qi goskope /etc/ssl/certs/ca-certificates.crt

ENV PYTHONPATH=/app

# Commit stamp, compared by the deploy workflow against the commit it built
# (`.git` is not in the image, and a pull succeeds on a stale tag too).
ARG GIT_SHA=unknown
ENV BESSER_BUILD_SHA=${GIT_SHA}

EXPOSE 9000

CMD ["python", "-m", "besser.utilities.web_modeling_editor.backend.backend"]