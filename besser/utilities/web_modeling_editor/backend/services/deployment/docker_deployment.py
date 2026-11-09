import logging
import os
import subprocess
import yaml

from besser.utilities.web_modeling_editor.backend.services.exceptions import UpstreamError

logger = logging.getLogger(__name__)

# A cold `docker-compose build` of a generated Django image (pip install
# included) takes a few minutes; past this the request is not worth holding.
DOCKER_COMPOSE_TIMEOUT_SECONDS = 600
# `docker ps` / `docker-compose ps` answer at once from a reachable daemon.
DOCKER_QUERY_TIMEOUT_SECONDS = 30

_DEPLOY_FAILED = "Docker Compose deployment failed; see the server log."

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

def _docker(args: list, cwd: str = None, timeout: int = DOCKER_QUERY_TIMEOUT_SECONDS) -> str:
    """Run a docker command and return its stdout; raise ``UpstreamError`` on any failure.

    The client only gets a fixed message; the command's output stays in the server log.
    """
    try:
        result = subprocess.run(
            args, cwd=cwd, check=True, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        logger.error("Docker deployment: %s is not installed on this host", args[0])
        raise UpstreamError(_DEPLOY_FAILED, status_code=500) from exc
    except subprocess.TimeoutExpired as exc:
        logger.error("Docker deployment: %s timed out after %s s", " ".join(args), timeout)
        raise UpstreamError("Docker Compose deployment timed out.", status_code=504) from exc
    except subprocess.CalledProcessError as exc:
        logger.error(
            "Docker deployment: %s exited with %s: %s", " ".join(args), exc.returncode,
            (exc.stderr or b"").decode(errors="replace"),
        )
        raise UpstreamError(_DEPLOY_FAILED, status_code=502) from exc
    return result.stdout.decode(errors="replace")


def get_container_on_port(port: int):
    container_name = _docker(["docker", "ps", "--format", "{{.Names}}", "--filter", f"publish={port}"]).strip()
    return container_name if container_name else None

def is_port_used(port: int, expected_container: str) -> bool:
    container_name = get_container_on_port(port)
    logger.debug("Container ID on port %d: %s", port, container_name)
    return container_name is not None and container_name != expected_container

def run_docker_compose(directory: str, project_name: str):
    """Build and start the Compose project in ``directory``.

    Raises:
        ValueError: port 8000 is held by another container.
        UpstreamError: docker is missing, a command failed (502) or timed out (504).
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")

    remove_volumes(directory)

    # Check if the port is already in use by another container
    expected_container = project_name + "_web_1"
    if is_port_used(8000, expected_container):
        raise ValueError("A different container is already running on port 8000.")

    _docker(["docker-compose", "down"], cwd=directory, timeout=DOCKER_COMPOSE_TIMEOUT_SECONDS)
    _docker(["docker-compose", "build"], cwd=directory, timeout=DOCKER_COMPOSE_TIMEOUT_SECONDS)
    output = _docker(["docker-compose", "up", "-d"], cwd=directory, timeout=DOCKER_COMPOSE_TIMEOUT_SECONDS)
    logger.info("Docker Compose executed successfully")
    logger.debug(output)
