def generate_docker_files(path: str = "output_backend"):
    generate_dockerfile(path)
    generate_docker_image(path)
    pass

def generate_dockerfile(path: str):
    with open(path + '/Dockerfile', 'w') as dockerfile:
        dockerfile.write('''FROM python:3.9-slim
WORKDIR /app

COPY main_api.py /app
COPY pydantic_classes.py /app
COPY sql_alchemy.py /app

RUN pip install requests==2.31.0
RUN pip install fastapi==0.110.0
RUN pip install pydantic==2.6.3
RUN pip install uvicorn==0.28.0
RUN pip install SQLAlchemy==2.0.29
RUN pip install httpx==0.27.0

EXPOSE 8000
CMD ["python", "main_api.py"]
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