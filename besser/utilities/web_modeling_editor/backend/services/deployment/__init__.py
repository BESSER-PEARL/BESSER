"""
Deployment module for handling application deployment.
"""

from .docker_deployment import run_docker_compose

__all__ = [
    "run_docker_compose",
]
