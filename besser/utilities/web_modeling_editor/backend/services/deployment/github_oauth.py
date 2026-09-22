"""
GitHub OAuth Integration for BESSER Web Editor.

Handles OAuth flow to authenticate users with their GitHub accounts.
Sessions are stored in an encrypted file-based store (with automatic
fallback to in-memory storage when the cryptography package is absent).
"""

import logging
import os
import secrets
from typing import Optional
import httpx
from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from besser.utilities.web_modeling_editor.backend.services.deployment.session_store import (
    SessionStore,
)


# GitHub OAuth configuration
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")
GITHUB_REDIRECT_URI = os.getenv("GITHUB_REDIRECT_URI", "http://localhost:9000/besser_api/github/auth/callback")
DEPLOYMENT_URL = os.getenv("DEPLOYMENT_URL", "http://localhost:8080")


# Encrypted session stores
# - _oauth_sessions: short-lived CSRF state tokens (10 min TTL)
# - _user_tokens: authenticated user sessions (24 hour TTL)
_oauth_sessions = SessionStore(
    store_path=os.environ.get("OAUTH_SESSIONS_PATH"),
    ttl=600,
)
_user_tokens = SessionStore(
    store_path=os.environ.get("USER_TOKENS_PATH"),
    ttl=86400,
)


class GitHubOAuthResponse(BaseModel):
    """Response model for OAuth flow."""
    success: bool
    authenticated: Optional[bool] = None
    session_id: Optional[str] = None
    user: Optional[str] = None
    avatar: Optional[str] = None
    username: Optional[str] = None
    error: Optional[str] = None


router = APIRouter(prefix="/github", tags=["GitHub OAuth"])


@router.get("/auth/login")
async def github_login(request: Request):
    """
    Initiate GitHub OAuth flow.

    Returns redirect URL to GitHub authorization page.
    """
    if not GITHUB_CLIENT_ID:
        raise HTTPException(
            status_code=500,
            detail="GitHub OAuth not configured. Set GITHUB_CLIENT_ID environment variable."
        )

    # Generate state parameter for CSRF protection
    state = secrets.token_urlsafe(32)

    # Store state in session store
    _oauth_sessions.set(state, {
        "client_ip": request.client.host if request.client else "unknown"
    })

    # Build GitHub authorization URL
    # Request permissions to manage repos and create gists
    scopes = "repo,gist,user"
    auth_url = (
        f"https://github.com/login/oauth/authorize"
        f"?client_id={GITHUB_CLIENT_ID}"
        f"&redirect_uri={GITHUB_REDIRECT_URI}"
        f"&scope={scopes}"
        f"&state={state}"
    )

    return RedirectResponse(url=auth_url)


@router.get("/auth/callback")
async def github_callback(code: str, state: str):
    """
    Handle GitHub OAuth callback.

    Exchange authorization code for access token.
    """
    # Verify state parameter
    if _oauth_sessions.get(state) is None:
        return RedirectResponse(
            url=f"{DEPLOYMENT_URL}?error=invalid_state",
            status_code=302
        )

    # Remove used state
    _oauth_sessions.delete(state)

    try:
        async with httpx.AsyncClient() as client:
            # Exchange code for access token
            token_response = await client.post(
                "https://github.com/login/oauth/access_token",
                headers={"Accept": "application/json"},
                data={
                    "client_id": GITHUB_CLIENT_ID,
                    "client_secret": GITHUB_CLIENT_SECRET,
                    "code": code,
                    "redirect_uri": GITHUB_REDIRECT_URI
                },
                timeout=10
            )
            token_response.raise_for_status()
            token_data = token_response.json()

            if "error" in token_data:
                return RedirectResponse(
                    url=f"{DEPLOYMENT_URL}?error={token_data['error']}",
                    status_code=302
                )

            access_token = token_data.get("access_token")
            if not access_token:
                return RedirectResponse(
                    url=f"{DEPLOYMENT_URL}?error=no_access_token",
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
        _user_tokens.set(session_id, {
            "access_token": access_token,
            "username": username,
            "avatar_url": user_data.get("avatar_url"),
        })

        # Redirect to frontend with session ID
        return RedirectResponse(
            url=f"{DEPLOYMENT_URL}?github_session={session_id}&username={username}",
            status_code=302
        )

    except httpx.HTTPError:
        return RedirectResponse(
            url=f"{DEPLOYMENT_URL}?error=github_api_error",
            status_code=302
        )


@router.get("/auth/status")
async def get_auth_status(session_id: str) -> GitHubOAuthResponse:
    """
    Check GitHub authentication status.

    Args:
        session_id: Session ID from OAuth flow

    Returns:
        Authentication status
    """
    session_data = _user_tokens.get(session_id)

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


@router.post("/auth/logout")
async def github_logout(session_id: str):
    """
    Logout and revoke GitHub session.

    Args:
        session_id: Session ID to revoke
    """
    _user_tokens.delete(session_id)

    return {"success": True, "message": "Logged out successfully"}


def get_user_token(session_id: str) -> Optional[str]:
    """
    Get user's GitHub access token from session.

    Args:
        session_id: Session ID

    Returns:
        Access token or None if not found/expired
    """
    session_data = _user_tokens.get(session_id)

    if not session_data:
        return None

    return session_data["access_token"]


BESSER_REPO_OWNER = "BESSER-PEARL"
BESSER_REPO_NAME = "BESSER"


@router.get("/star/status")
async def get_star_status(session_id: str):
    """Check if the authenticated user has starred the BESSER repository."""
    token = get_user_token(session_id)
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
async def star_besser_repo(session_id: str):
    """Star the BESSER repository on behalf of the authenticated user."""
    token = get_user_token(session_id)
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
async def unstar_besser_repo(session_id: str):
    """Unstar the BESSER repository on behalf of the authenticated user."""
    token = get_user_token(session_id)
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
