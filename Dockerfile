# Use slim variant to reduce image size (200MB smaller).
# Python 3.12 matches the CI matrix; 3.10 was dropped in v7.5.1 because
# ``typing.Self`` (used in the NN metamodel, PEP 673) requires 3.11+.
FROM python:3.12-slim

ENV RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PGSSLROOTCERT=/opt/aws/rds/global-bundle.pem

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
    && mkdir -p /opt/aws/rds \
    && curl --proto '=https' --tlsv1.2 -fsSL \
        https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem \
        -o /opt/aws/rds/global-bundle.pem \
    && grep -q 'BEGIN CERTIFICATE' /opt/aws/rds/global-bundle.pem \
    && chmod 0444 /opt/aws/rds/global-bundle.pem \
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
ENV PATH="/opt/cargo/bin:/opt/kotlinc/bin:${PATH}"

WORKDIR /app

# No additional system dependencies needed - Python slim has everything for a basic Flask/FastAPI app
# If you need specific system libraries (e.g., for image processing), add them here

# Copy and install dependencies first for better layer caching
COPY requirements.txt ./requirements.txt
COPY besser/utilities/web_modeling_editor/backend/requirements.txt ./backend-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r backend-requirements.txt

# Copy only necessary files
COPY pyproject.toml README.md ./
COPY besser/ ./besser/

# Install BESSER package
RUN pip install --no-cache-dir -e .

RUN groupadd --gid 10001 besser \
    && useradd --uid 10001 --gid besser --create-home --home-dir /home/besser --shell /usr/sbin/nologin besser \
    && mkdir -p /app/besser/feedback_data /app/besser/session_data /app/besser/smartgen_data \
    && chown -R besser:besser /app /opt/cargo /opt/rustup /home/besser

ENV PYTHONPATH=/app \
    HOME=/home/besser

USER 10001:10001

EXPOSE 9000

CMD ["python", "-m", "besser.utilities.web_modeling_editor.backend.backend"]
