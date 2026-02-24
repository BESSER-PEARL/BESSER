"""
Deployment module for handling application deployment.

This module provides:
- Docker Compose deployment for local hosting
- GitHub OAuth and repository creation for cloud deployment
- Integration with Render via GitHub repos
"""

from .docker_deployment import run_docker_compose
from .github_oauth import router as github_oauth_router
from .github_deploy_api import router as github_deploy_router

__all__ = [
    "run_docker_compose",
    "github_oauth_router",
    "github_deploy_router",
]
