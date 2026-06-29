FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# Install CPU-only PyTorch first to avoid downloading the large CUDA bundle
# (~250 MB vs ~2 GB) on a CPU-only container.  This wheel is installed before
# the rest of the requirements so pip's resolver re-uses it for any package
# that also depends on torch (e.g. besser-agentic-framework[extras]).
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Preload NLTK tokenizer data used by BAF so simulation subprocesses do not
# download corpora at runtime (which can leave partial/corrupted zip files).
ENV NLTK_DATA=/opt/nltk_data
RUN mkdir -p ${NLTK_DATA} && python - <<'PY'
import nltk

for resource in ("punkt", "punkt_tab"):
    nltk.download(resource, download_dir="/opt/nltk_data", quiet=True, raise_on_error=True)

nltk.data.path.insert(0, "/opt/nltk_data")
nltk.data.find("tokenizers/punkt")
nltk.data.find("tokenizers/punkt_tab")
print("NLTK tokenizers installed in /opt/nltk_data")
PY

COPY . .

# Sessions directory will be populated at runtime from the tmpfs mount
RUN mkdir -p /tmp/sessions

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8001

ENTRYPOINT ["/entrypoint.sh"]
