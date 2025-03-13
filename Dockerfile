FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy main requirements.txt first for better caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy specific backend requirements if they contain additional dependencies
COPY besser/utilities/web_modeling_editor/backend/requirements.txt /app/backend-requirements.txt
RUN pip install --no-cache-dir -r backend-requirements.txt

# Copy Python project files needed for installation
COPY pyproject.toml README.md /app/
# Then copy the besser directory
COPY besser/ /app/besser/

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Install BESSER package in development mode
RUN pip install -e .

# Expose the port the app runs on
EXPOSE 9000

# Start the application
CMD ["python", "besser/utilities/web_modeling_editor/backend/backend.py"]