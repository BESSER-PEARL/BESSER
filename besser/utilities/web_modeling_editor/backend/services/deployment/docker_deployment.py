import logging
import os
import subprocess
import yaml

logger = logging.getLogger(__name__)

def remove_volumes(directory: str):
    compose_path = os.path.join(directory, "docker-compose.yml")
    # Load the docker-compose.yml file
    with open(compose_path, "r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    # Remove the volumes section from each service
    for service in compose_data.get("services", {}):
        if "volumes" in compose_data["services"][service]:
            del compose_data["services"][service]["volumes"]

    # Save the changes back to the docker-compose.yml file
    with open(compose_path, "w", encoding="utf-8") as f:
        yaml.dump(compose_data, f, default_flow_style=False)

    logger.info("Volumes section removed from docker-compose.yml")

def is_deployment_running(directory: str) -> bool:
    try:
        result = subprocess.run(
            ["docker-compose", "ps", "-q"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            cwd=directory
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
    logger.debug("Container ID on port %d: %s", port, container_name)
    return container_name is not None and container_name != expected_container

def run_docker_compose(directory: str, project_name: str):
    try:
        # Ensure the directory exists
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory {directory} does not exist.")

        remove_volumes(directory)

        # Check if the port is already in use by another container
        expected_container = project_name + "_web_1"
        if is_port_used(8000, expected_container):
            raise ValueError("A different container is already running on port 8000.")

        # Check if the deployment is already running
        #if is_deployment_running(directory):
        #    logger.info("Deployment detected. Rebuilding before starting.")
        subprocess.run(["docker-compose", "down"], check=True, cwd=directory)
        subprocess.run(["docker-compose", "build"], check=True, cwd=directory)

        # Run docker-compose up command
        result = subprocess.run(
            ["docker-compose", "up", "-d"],
            check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=directory
        )
        logger.info("Docker Compose executed successfully")
        logger.debug(result.stdout.decode())

    except subprocess.CalledProcessError as e:
        logger.error("Error running docker-compose: %s", e)
        logger.error(e.stderr.decode())
    except FileNotFoundError as e:
        logger.error(e)
