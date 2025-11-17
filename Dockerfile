# Use slim variant to reduce image size (200MB smaller)
FROM python:3.10-slim

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

ENV PYTHONPATH=/app

EXPOSE 9000

CMD ["python", "-m", "besser.utilities.web_modeling_editor.backend.backend"]