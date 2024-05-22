import docker
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
