"""
Shared authentication helpers for router endpoints.

Holds the GitHub-session gate used by every router that requires a signed-in
user, so the check (and its error messages) lives in one place, and the
BESSER Lab gate built on it (``LabUserDep``).
"""

from typing import Annotated, Callable, Optional

from fastapi import Depends, Header, HTTPException

from besser.utilities.web_modeling_editor.backend.dependencies import GitHubSessionsDep, SettingsDep
from besser.utilities.web_modeling_editor.backend.routers.github_oauth import (
    get_user_token,
)
from besser.utilities.web_modeling_editor.backend.services.deployment.session_store import (
    GitHubSessions,
)
from besser.utilities.web_modeling_editor.backend.services.lab.access import LabUser, is_session_allowed


def require_github_session(
    github_sessions: GitHubSessions,
    github_session: Optional[str],
    token_lookup: Optional[Callable[[GitHubSessions, str], Optional[str]]] = None,
) -> None:
    """Verify a GitHub OAuth session is present and active.

    Mirrors the auth gate used by the deploy endpoints in github_deploy_api.py.

    Args:
        github_sessions: The app's sign-in stores (``GitHubSessionsDep``).
        github_session: The value of the ``X-GitHub-Session`` header (or the
            equivalent value sent by a WebSocket client).
        token_lookup: Resolves ``(github_sessions, session id)`` to its access token. Defaults to
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
    if not lookup(github_sessions, github_session):
        raise HTTPException(
            status_code=401,
            detail="GitHub session expired. Please sign in again.",
        )


def require_lab_user(
    settings: SettingsDep,
    github_sessions: GitHubSessionsDep,
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
) -> LabUser:
    """FastAPI dependency of every BESSER Lab route: the signed-in user, if the lab allows them.

    The session is read from the ``X-GitHub-Session`` header only; the lab never
    accepts the deprecated ``session_id`` query parameter.

    Raises:
        HTTPException: 404 when the lab is off (its routes are mounted on every instance), 401 when the
            session is missing or expired, 403 when the login is neither on ``BESSER_LAB_ALLOWED_LOGINS`` nor
            signed in as a member of ``BESSER_LAB_ALLOWED_ORG``.
    """
    if not settings.lab_enabled:
        raise HTTPException(status_code=404, detail="Not Found")
    require_github_session(github_sessions, github_session)
    session = github_sessions.user_tokens.get(github_session) or {}
    login = session.get("username")
    if not is_session_allowed(settings, session):
        raise HTTPException(status_code=403, detail="This GitHub account is not allowed to use BESSER Bootstrapped.")
    return LabUser(login=login, session_id=github_session, access_token=session["access_token"])


LabUserDep = Annotated[LabUser, Depends(require_lab_user)]
