import os
import subprocess
import yaml

def remove_volumes():
    # Load the docker-compose.yml file
    with open("docker-compose.yml", "r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    # Remove the volumes section from each service
    for service in compose_data.get("services", {}):
        if "volumes" in compose_data["services"][service]:
            del compose_data["services"][service]["volumes"]

    # Save the changes back to the docker-compose.yml file
    with open("docker-compose.yml", "w", encoding="utf-8") as f:
        yaml.dump(compose_data, f, default_flow_style=False)

    print("Volumes section removed from docker-compose.yml!")

def is_deployment_running(directory: str) -> bool:
    try:
        os.chdir(directory)
        result = subprocess.run(
            ["docker-compose", "ps", "-q"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        running_containers = result.stdout.decode().strip()
        return bool(running_containers)
    except subprocess.CalledProcessError:
        return False

def get_container_on_port(port: int):
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}", "--filter", f"publish={port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        container_name = result.stdout.decode().strip()
        return container_name if container_name else None
    except subprocess.CalledProcessError:
        return None

def is_port_used(port: int, expected_container: str) -> bool:
    container_name = get_container_on_port(port)
    print(f"Container ID on port {port}: {container_name}")
    return container_name is not None and container_name != expected_container

def run_docker_compose(directory: str, project_name: str):
    try:
        # Ensure the directory exists
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory {directory} does not exist.")

        # Change working directory to the directory containing the docker-compose.yml
        os.chdir(directory)
        remove_volumes()

        # Check if the port is already in use by another container
        expected_container = project_name + "_web_1"
        if is_port_used(8000, expected_container):
            raise ValueError("A different container is already running on port 8000.")

        # Check if the deployment is already running
        #if is_deployment_running(directory):
        #    print("Deployment detected. Rebuilding before starting.")
        subprocess.run(["docker-compose", "down"], check=True)
        subprocess.run(["docker-compose", "build"], check=True)

        # Run docker-compose up command
        result = subprocess.run(
            ["docker-compose", "up", "-d"],
            check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Docker Compose executed successfully")
        print(result.stdout.decode())

    except subprocess.CalledProcessError as e:
        print(f"Error running docker-compose: {e}")
        print(e.stderr.decode())
    except FileNotFoundError as e:
        print(e)
