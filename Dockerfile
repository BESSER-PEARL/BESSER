FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the entire project first
COPY . .

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Install BESSER package
RUN pip install -e .

# Install additional requirements
RUN pip install --no-cache-dir -r besser_backend/requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Start the application
CMD ["python", "besser_backend/main.py"]