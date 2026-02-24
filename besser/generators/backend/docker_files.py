def generate_docker_files(path: str = "output_backend"):
    generate_dockerfile(path)
    generate_docker_image(path)
    pass

def generate_dockerfile(path: str):
    with open(path + '/Dockerfile', 'w') as dockerfile:
        dockerfile.write('''FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install dependencies in a single layer
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main_api.py pydantic_classes.py sql_alchemy.py ./

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=2)" || exit 1

# Run with uvicorn (production-ready ASGI server)
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        )
    
    # Generate requirements.txt
    with open(path + '/requirements.txt', 'w') as requirements:
        requirements.write('''fastapi>=0.110.0
uvicorn>=0.28.0
pydantic>=2.6.3
sqlalchemy>=2.0.29
httpx>=0.27.0
requests>=2.31.0
'''
        )
    
    # Generate .dockerignore for smaller build context
    with open(path + '/.dockerignore', 'w') as dockerignore:
        dockerignore.write('''__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info
dist
build
.git
.gitignore
.env
.venv
venv/
*.log
.pytest_cache
.coverage
htmlcov/
.DS_Store
'''
        )

def generate_docker_image(path: str):
    with open(path + '/create_docker_image.py', 'w') as create_image:
        create_image.write('''import docker
import os
import argparse
import sys

def build_args_parser():
    parser = argparse.ArgumentParser(description='Generating and pushing the image to DockerHub')
    parser.add_argument('-u', '--username',
                        required=True,
                        help='docker username')
    parser.add_argument('-p', '--password',
                        required=True,
                        help='docker password')
    parser.add_argument('-i', '--image',
                        required=True,
                        help='docker image name')
    parser.add_argument('-t', '--tag',
                        required=True,
                        help='tag name')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser

def main():
    args = build_args_parser().parse_args()
    # Get dockerfile path
    dockerfile_path = os.path.dirname(os.path.abspath(__file__))
    # Create docker client
    client = docker.from_env()
    # Create docker image
    tag = args.username + '/' + args.image + ':' + args.tag
    image, build_logs = client.images.build(
        path = dockerfile_path,
        rm = True,
        tag = tag
    )

    for log in build_logs:
        print(log)

    client.login(username=args.username, password=args.password)
    resp = client.api.push(
        tag,
        stream=True,
        decode=True,
    )
    
    for line in resp:
        print(line)

if __name__ == '__main__':
    main()
'''
        )