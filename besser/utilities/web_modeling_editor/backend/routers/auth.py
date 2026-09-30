"""
Shared authentication helpers for router endpoints.

Holds the GitHub-session gate used by every router that requires a signed-in
user, so the check (and its error messages) lives in one place.
"""

from typing import Callable, Optional

from fastapi import HTTPException

from besser.utilities.web_modeling_editor.backend.services.deployment.github_oauth import (
    get_user_token,
)


def require_github_session(
    github_session: Optional[str],
    token_lookup: Optional[Callable[[str], Optional[str]]] = None,
) -> None:
    """Verify a GitHub OAuth session is present and active.

    Mirrors the auth gate used by the deploy endpoints in github_deploy_api.py.

    Args:
        github_session: The value of the ``X-GitHub-Session`` header (or the
            equivalent value sent by a WebSocket client).
        token_lookup: Resolves a session id to its access token. Defaults to
            :func:`github_oauth.get_user_token`; callers may pass their own
            module-level reference so it can be patched in tests.

    Raises:
        HTTPException: 401 when the session is missing or expired.
    """
    lookup = token_lookup or get_user_token
    if not github_session:
        raise HTTPException(
            status_code=401,
            detail="GitHub authentication required. Please sign in with GitHub first.",
        )
    if not lookup(github_session):
        raise HTTPException(
            status_code=401,
            detail="GitHub session expired. Please sign in again.",
        )
