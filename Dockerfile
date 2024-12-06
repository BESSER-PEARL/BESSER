FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy requirements.txt first for better caching
COPY besser/utilities/besser_backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Install BESSER package
RUN pip install -e .

# Expose the port the app runs on
EXPOSE 8000

# Start the application
CMD ["python", "besser/utilities/besser_backend/main.py"]