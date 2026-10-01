# Two targets:
#   backend          the web editor backend: Python, its requirements and besser.
#   smartgen-worker  the Spec-Driven Agent worker: backend plus the toolchains
#                    Phase 3 compiles with and bubblewrap for run_command.
# docker build --target backend|smartgen-worker. Without --target the last
# stage (the worker) is built.

# slim: 200MB smaller. 3.12 matches CI; 3.11+ is required for typing.Self
# (NN metamodel, PEP 673).
FROM python:3.12-slim AS python-deps

# Build behind a TLS-inspecting proxy: drop its root + signing certs into
# ca-certs-extra/ (gitignored) and pass --build-arg TRUST_EXTRA_CAS=1.
# Opt-in so that a stray .crt in a local checkout never reaches an image
# built for deployment. Each final stage strips it again before it ships.
ARG TRUST_EXTRA_CAS=0
# A BuildKit bind mount, not COPY, so the cert files get no layer of their own.
RUN --mount=type=bind,source=ca-certs-extra,target=/tmp/ca-certs-extra \
    if [ "$TRUST_EXTRA_CAS" = "1" ]; then \
        cp /tmp/ca-certs-extra/*.crt /usr/local/share/ca-certificates/ \
        && update-ca-certificates; \
    fi
# pip, npm and rustup each ship their own trust store and must be pointed at
# the system bundle explicitly. No-op when no extra CA was injected.
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

WORKDIR /app

# Shared by both targets, so the worker reuses these layers. ruff is one of
# the backend requirements (Phase 3's undefined-name checks run it).
COPY requirements.txt ./requirements.txt
COPY besser/utilities/web_modeling_editor/backend/requirements.txt ./backend-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r backend-requirements.txt


FROM python-deps AS backend

COPY pyproject.toml setup.cfg README.md ./
COPY besser/ ./besser/
RUN pip install --no-cache-dir -e .

# A build-time CA must not become runtime trust. Unconditional; the last lines
# fail the build if a known TLS-inspection CA is still trusted. They read the
# decoded subjects: the PEM bundle is base64, so grepping it never matches.
RUN rm -f /usr/local/share/ca-certificates/*.crt \
    && update-ca-certificates --fresh >/dev/null 2>&1 \
    && subjects="$(openssl crl2pkcs7 -nocrl -certfile /etc/ssl/certs/ca-certificates.crt \
        | openssl pkcs7 -print_certs -noout)" && [ -n "$subjects" ] \
    && ! printf '%s\n' "$subjects" | grep -qiE 'netskope|goskope'

ENV PYTHONPATH=/app

# Commit stamp, compared by the deploy workflow against the commit it built
# (`.git` is not in the image, and a pull succeeds on a stale tag too).
ARG GIT_SHA=unknown
ENV BESSER_BUILD_SHA=${GIT_SHA}

EXPOSE 9000

CMD ["python", "-m", "besser.utilities.web_modeling_editor.backend.backend"]


FROM python-deps AS smartgen-worker

# Phase 3 compile validation soft-skips when these are missing from PATH
# (shutil.which -> None), so nextjs/rust/spring-boot output ships unchecked.
# JDK 21 because Debian Trixie no longer packages 17. build-essential is the
# C linker cargo and native Python packages need.
#
# bubblewrap confines run_command's model-authored shell to its own run
# directory. The worker fails closed without it, and needs
# seccomp=unconfined to use it (see docker-compose.prod.yml). passt (pasta)
# gives each shell session a private network; it needs /dev/net/tun.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        unzip \
        build-essential \
        bubblewrap \
        passt \
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

# Same code layers and CA strip as the backend target.
COPY pyproject.toml setup.cfg README.md ./
COPY besser/ ./besser/
RUN pip install --no-cache-dir -e .

# The JDK keystore was built while the proxy CA was trusted, and --fresh never
# removes a cert from it: delete it so the jks-keystore hook rebuilds it from
# the clean store, then check it the same way.
RUN rm -f /usr/local/share/ca-certificates/*.crt /etc/ssl/certs/java/cacerts \
    && update-ca-certificates --fresh >/dev/null 2>&1 \
    && subjects="$(openssl crl2pkcs7 -nocrl -certfile /etc/ssl/certs/ca-certificates.crt \
        | openssl pkcs7 -print_certs -noout)" && [ -n "$subjects" ] \
    && ! printf '%s\n' "$subjects" | grep -qiE 'netskope|goskope' \
    && jks="$(keytool -list -v -cacerts -storepass changeit)" && [ -n "$jks" ] \
    && ! printf '%s\n' "$jks" | grep -qiE 'netskope|goskope'

ENV PYTHONPATH=/app

ARG GIT_SHA=unknown
ENV BESSER_BUILD_SHA=${GIT_SHA}

EXPOSE 9000

CMD ["python", "-m", "besser.utilities.web_modeling_editor.backend.backend"]
