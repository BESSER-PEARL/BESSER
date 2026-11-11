"""
Deployment module for handling application deployment.

This module provides:
- Docker Compose deployment for local hosting
- The GitHub API client, repository model files and Render blueprints behind
  the GitHub endpoints (``routers/github_oauth.py``, ``routers/github_deploy_api.py``)
"""

from .docker_deployment import run_docker_compose

__all__ = [
    "run_docker_compose",
]
