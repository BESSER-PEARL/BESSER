"""
GitHub OAuth Integration for BESSER Web Editor.

Handles OAuth flow to authenticate users with their GitHub accounts.
Sessions are stored in the app's encrypted file-based stores
(``GitHubSessions``, built in the backend's lifespan), with an automatic
fallback to in-memory storage when the cryptography package is absent.
"""

import logging
import secrets
from typing import Optional
import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request

logger = logging.getLogger(__name__)
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from besser.utilities.web_modeling_editor.backend.services.deployment.session_store import (
    GitHubSessions,
)
from besser.utilities.web_modeling_editor.backend.dependencies import GitHubSessionsDep, SettingsDep
from besser.utilities.web_modeling_editor.backend.models.responses import ResponseModel
from besser.utilities.web_modeling_editor.backend.settings import BackendSettings


class GitHubOAuthResponse(ResponseModel):
    """Response model for OAuth flow."""
    success: bool
    authenticated: Optional[bool] = None
    session_id: Optional[str] = None
    user: Optional[str] = None
    avatar: Optional[str] = None
    username: Optional[str] = None
    error: Optional[str] = None


router = APIRouter(prefix="/github", tags=["GitHub OAuth"])


def github_session_id(
    header: Optional[str] = Header(None, alias="X-GitHub-Session"),
    session_id: Optional[str] = Query(None, deprecated=True),
) -> Optional[str]:
    """The request's GitHub session: the ``X-GitHub-Session`` header.

    For one release after the move to the header, a client that still sends
    the session as the ``session_id`` query parameter (which puts it in proxy
    and server logs) is served too, with a deprecation warning.
    """
    if header:
        return header
    if session_id:
        logger.warning("GitHub session sent as the deprecated session_id query parameter; "
                       "send the X-GitHub-Session header")
    return session_id


@router.get("/auth/login")
async def github_login(request: Request, settings: SettingsDep, github_sessions: GitHubSessionsDep):
    """
    Initiate GitHub OAuth flow.

    Returns redirect URL to GitHub authorization page.
    """
    if not settings.github_client_id:
        raise HTTPException(
            status_code=500,
            detail="GitHub OAuth not configured. Set GITHUB_CLIENT_ID environment variable."
        )

    # Generate state parameter for CSRF protection
    state = secrets.token_urlsafe(32)

    # Store state in session store
    github_sessions.oauth_states.set(state, {
        "client_ip": request.client.host if request.client else "unknown"
    })

    # Build GitHub authorization URL
    # Request permissions to manage repos and create gists
    scopes = "repo,gist,user"
    auth_url = (
        f"https://github.com/login/oauth/authorize"
        f"?client_id={settings.github_client_id}"
        f"&redirect_uri={settings.github_redirect_uri}"
        f"&scope={scopes}"
        f"&state={state}"
    )

    return RedirectResponse(url=auth_url)


@router.get("/auth/callback")
async def github_callback(code: str, state: str, settings: SettingsDep, github_sessions: GitHubSessionsDep):
    """
    Handle GitHub OAuth callback.

    Exchange authorization code for access token.
    """
    # Verify state parameter
    if github_sessions.oauth_states.get(state) is None:
        return RedirectResponse(
            url=f"{settings.deployment_url}?error=invalid_state",
            status_code=302
        )

    # Remove used state
    github_sessions.oauth_states.delete(state)

    try:
        async with httpx.AsyncClient() as client:
            # Exchange code for access token
            token_response = await client.post(
                "https://github.com/login/oauth/access_token",
                headers={"Accept": "application/json"},
                data={
                    "client_id": settings.github_client_id,
                    "client_secret": settings.github_client_secret.get_secret_value(),
                    "code": code,
                    "redirect_uri": settings.github_redirect_uri
                },
                timeout=10
            )
            token_response.raise_for_status()
            token_data = token_response.json()

            if "error" in token_data:
                return RedirectResponse(
                    url=f"{settings.deployment_url}?error={token_data['error']}",
                    status_code=302
                )

            access_token = token_data.get("access_token")
            if not access_token:
                return RedirectResponse(
                    url=f"{settings.deployment_url}?error=no_access_token",
                    status_code=302
                )

            # Get user info
            user_response = await client.get(
                "https://api.github.com/user",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/vnd.github+json"
                },
                timeout=10
            )
            user_response.raise_for_status()
            user_data = user_response.json()

        username = user_data.get("login")

        # Store token in encrypted session store
        session_id = secrets.token_urlsafe(32)
        github_sessions.user_tokens.set(session_id, {
            "access_token": access_token,
            "username": username,
            "avatar_url": user_data.get("avatar_url"),
        })

        # The session id rides in the fragment, which browsers never send to a server or proxy log.
        return RedirectResponse(
            url=f"{settings.deployment_url}#github_session={session_id}&username={username}",
            status_code=302
        )

    except httpx.HTTPError:
        return RedirectResponse(
            url=f"{settings.deployment_url}?error=github_api_error",
            status_code=302
        )


@router.get("/auth/status")
async def get_auth_status(
    github_sessions: GitHubSessionsDep,
    session_id: Optional[str] = Depends(github_session_id),
) -> GitHubOAuthResponse:
    """
    Check GitHub authentication status.

    Args:
        session_id: Session ID from OAuth flow

    Returns:
        Authentication status
    """
    session_data = github_sessions.user_tokens.get(session_id) if session_id else None

    if not session_data:
        return GitHubOAuthResponse(
            success=False,
            error="Session not found or expired"
        )

    return GitHubOAuthResponse(
        success=True,
        authenticated=True,
        session_id=session_id,
        user=session_data["username"],
        avatar=session_data.get("avatar_url"),
        username=session_data["username"],
    )


class LogoutRequest(BaseModel):
    """Body of ``POST /auth/logout``, as the editor sends it."""
    session_id: str


@router.post("/auth/logout")
async def github_logout(
    settings: SettingsDep,
    github_sessions: GitHubSessionsDep,
    body: Optional[LogoutRequest] = None,
    session_id: Optional[str] = Query(None, deprecated=True),
):
    """End the BESSER session and revoke its GitHub token.

    The session comes in the JSON body; for one release the deprecated
    ``session_id`` query parameter is read too, with a warning.
    """
    if body is None:
        if not session_id:
            raise HTTPException(status_code=422, detail="session_id is required")
        logger.warning("GitHub logout sent the deprecated session_id query parameter; send it in the body")
    session_id = body.session_id if body is not None else session_id
    session_data = github_sessions.user_tokens.get(session_id)
    github_sessions.user_tokens.delete(session_id)
    if session_data:
        await _revoke_github_token(settings, session_data["access_token"])

    return {"success": True, "message": "Logged out successfully"}


async def _revoke_github_token(settings: BackendSettings, access_token: str) -> None:
    """Revoke the OAuth token at GitHub, so a copy of it stops working too."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.request(
                "DELETE",
                f"https://api.github.com/applications/{settings.github_client_id}/token",
                auth=(settings.github_client_id, settings.github_client_secret.get_secret_value()),
                headers={"Accept": "application/vnd.github+json"},
                json={"access_token": access_token},
                timeout=10,
            )
        if response.status_code not in (204, 404):
            logger.warning("GitHub token revocation returned HTTP %s", response.status_code)
    except httpx.HTTPError:
        logger.warning("Could not revoke the GitHub token at logout", exc_info=True)


def get_user_token(github_sessions: GitHubSessions, session_id: str) -> Optional[str]:
    """
    Get user's GitHub access token from session.

    Args:
        github_sessions: The app's sign-in stores (``GitHubSessionsDep``)
        session_id: Session ID

    Returns:
        Access token or None if not found/expired
    """
    session_data = github_sessions.user_tokens.get(session_id)

    if not session_data:
        return None

    return session_data["access_token"]


BESSER_REPO_OWNER = "BESSER-PEARL"
BESSER_REPO_NAME = "BESSER"


@router.get("/star/status")
async def get_star_status(
    github_sessions: GitHubSessionsDep,
    session_id: Optional[str] = Depends(github_session_id),
):
    """Check if the authenticated user has starred the BESSER repository."""
    token = get_user_token(github_sessions, session_id)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://api.github.com/user/starred/{BESSER_REPO_OWNER}/{BESSER_REPO_NAME}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                },
                timeout=10,
            )
        return {"starred": resp.status_code == 204}
    except httpx.HTTPError:
        return {"starred": False}


@router.put("/star")
async def star_besser_repo(
    github_sessions: GitHubSessionsDep,
    session_id: Optional[str] = Depends(github_session_id),
):
    """Star the BESSER repository on behalf of the authenticated user."""
    token = get_user_token(github_sessions, session_id)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.put(
                f"https://api.github.com/user/starred/{BESSER_REPO_OWNER}/{BESSER_REPO_NAME}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                },
                timeout=10,
            )
            resp.raise_for_status()
        return {"success": True}
    except httpx.HTTPError:
        logger.exception("GitHub API request failed")
        raise HTTPException(status_code=502, detail="GitHub API request failed.")


@router.delete("/star")
async def unstar_besser_repo(
    github_sessions: GitHubSessionsDep,
    session_id: Optional[str] = Depends(github_session_id),
):
    """Unstar the BESSER repository on behalf of the authenticated user."""
    token = get_user_token(github_sessions, session_id)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.delete(
                f"https://api.github.com/user/starred/{BESSER_REPO_OWNER}/{BESSER_REPO_NAME}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                },
                timeout=10,
            )
            resp.raise_for_status()
        return {"success": True}
    except httpx.HTTPError:
        logger.exception("GitHub API request failed")
        raise HTTPException(status_code=502, detail="GitHub API request failed.")
