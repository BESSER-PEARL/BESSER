# syntax=docker/dockerfile:1
# BESSER agent simulator image.
#
# Build from the REPOSITORY ROOT so the simulator runs this repository's
# `besser` package (not whatever is on PyPI):
#
#   docker build -f besser/utilities/web_modeling_editor/agent_simulator/Dockerfile \
#     -t artefacts.list.lu/besser/web_modeling_editor/agent_simulator:latest .
#
# Python 3.12 matches the root Dockerfile and the CI matrix.

# ── Builder: resolve and install every dependency into /opt/venv ─────────────
FROM python:3.12-slim AS builder

# Compilers only live in this stage, for any dependency without a wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# CPU-only PyTorch first to avoid the CUDA bundle (~250 MB vs ~2 GB). pip's
# resolver then re-uses it for every package that depends on torch.
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Only the simulator's own requirements: the root requirements.txt cannot be
# co-installed with besser-agentic-framework[extras] (pydantic pins, see the
# comment in agent_simulator/requirements.txt).
COPY besser/utilities/web_modeling_editor/agent_simulator/requirements.txt ./simulator-requirements.txt
RUN pip install --no-cache-dir -r simulator-requirements.txt

# Preload the NLTK tokenizer data BAF uses, so sessions never download corpora
# at runtime (which can leave partial/corrupted zip files).
RUN python - <<'PY'
import nltk

for resource in ("punkt", "punkt_tab"):
    nltk.download(resource, download_dir="/opt/nltk_data", quiet=True, raise_on_error=True)

nltk.data.path.insert(0, "/opt/nltk_data")
nltk.data.find("tokenizers/punkt")
nltk.data.find("tokenizers/punkt_tab")
print("NLTK tokenizers installed in /opt/nltk_data")
PY

# ── Runtime ──────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# bubblewrap confines every agent session (see agent_simulator/sandbox.py).
RUN apt-get update && apt-get install -y --no-install-recommends \
        bubblewrap \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/nltk_data /opt/nltk_data
ENV PATH=/opt/venv/bin:$PATH \
    NLTK_DATA=/opt/nltk_data \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml setup.cfg README.md ./
COPY besser/ ./besser/
# This checkout's `besser`; its dependencies the simulator needs are in
# agent_simulator/requirements.txt.
RUN pip install --no-cache-dir --no-deps -e .

# The container runs as root on purpose: giving each session its own UID needs
# setuid/setgid/chown (see session_manager.py). Agent code never runs as root:
# every session process switches to its pool UID before exec'ing bubblewrap,
# which needs no privilege at all (unprivileged user namespace). Compose drops
# every capability except the few the UID switch and cleanup need.
EXPOSE 8001

CMD ["python", "-m", "besser.utilities.web_modeling_editor.agent_simulator.agent_simulator_api"]
