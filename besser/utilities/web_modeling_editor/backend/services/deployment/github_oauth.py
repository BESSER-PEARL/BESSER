"""
GitHub OAuth Integration for BESSER Web Editor.

Handles OAuth flow to authenticate users with their GitHub accounts.
Sessions are stored in an encrypted file-based store (with automatic
fallback to in-memory storage when the cryptography package is absent).
"""

import logging
import os
import re
import secrets
from typing import Iterator, Optional
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel

from besser.utilities.web_modeling_editor.backend.services.deployment.session_store import (
    SessionStore,
)

logger = logging.getLogger(__name__)

# GitHub OAuth configuration
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")
GITHUB_REDIRECT_URI = os.getenv("GITHUB_REDIRECT_URI", "http://localhost:9000/besser_api/github/auth/callback")
DEPLOYMENT_URL = os.getenv("DEPLOYMENT_URL", "http://localhost:8080")
GITHUB_SESSION_COOKIE_NAME = "__Host-besser_github_session"
GITHUB_SESSION_TTL_SECONDS = 24 * 60 * 60
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{10,200}$")


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


class GitHubLogoutRequest(BaseModel):
    session_id: Optional[str] = None


def _normalise_session_id(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not _SESSION_ID_PATTERN.fullmatch(candidate):
        return None
    return candidate


def github_session_candidates(
    request: Request,
    *,
    header_session: Optional[str] = None,
    explicit_session: Optional[str] = None,
) -> Iterator[str]:
    """Yield unique, well-formed session IDs in secure-first order.

    The HttpOnly cookie is the browser default. The header and explicit value
    remain migration fallbacks for existing clients and command-line tools.
    """
    seen: set[str] = set()
    for raw_value in (
        request.cookies.get(GITHUB_SESSION_COOKIE_NAME),
        header_session,
        explicit_session,
    ):
        session_id = _normalise_session_id(raw_value)
        if session_id is not None and session_id not in seen:
            seen.add(session_id)
            yield session_id


def _authenticated_session(
    request: Request,
    *,
    header_session: Optional[str] = None,
    explicit_session: Optional[str] = None,
) -> tuple[Optional[str], Optional[dict]]:
    for session_id in github_session_candidates(
        request,
        header_session=header_session,
        explicit_session=explicit_session,
    ):
        session_data = _user_tokens.get(session_id)
        if session_data:
            return session_id, session_data
    return None, None


def _set_session_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(
        key=GITHUB_SESSION_COOKIE_NAME,
        value=session_id,
        max_age=GITHUB_SESSION_TTL_SECONDS,
        path="/",
        secure=True,
        httponly=True,
        samesite="lax",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=GITHUB_SESSION_COOKIE_NAME,
        path="/",
        secure=True,
        httponly=True,
        samesite="lax",
    )


def _frontend_redirect(**parameters: str) -> str:
    separator = "&" if "?" in DEPLOYMENT_URL else "?"
    return f"{DEPLOYMENT_URL}{separator}{urlencode(parameters)}"


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
    auth_url = "https://github.com/login/oauth/authorize?" + urlencode({
        "client_id": GITHUB_CLIENT_ID,
        "redirect_uri": GITHUB_REDIRECT_URI,
        "scope": scopes,
        "state": state,
    })

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
            url=_frontend_redirect(error="invalid_state"),
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
                    url=_frontend_redirect(error=str(token_data["error"])),
                    status_code=302
                )

            access_token = token_data.get("access_token")
            if not access_token:
                return RedirectResponse(
                    url=_frontend_redirect(error="no_access_token"),
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
        if not isinstance(username, str) or not username:
            return RedirectResponse(
                url=_frontend_redirect(error="invalid_github_user"),
                status_code=302,
            )

        # Store token in encrypted session store
        session_id = secrets.token_urlsafe(32)
        _user_tokens.set(session_id, {
            "access_token": access_token,
            "username": username,
            "avatar_url": user_data.get("avatar_url"),
            # GitHub's numeric database ID is stable across login renames.
            # Keep it in new sessions so authorization can bind resources
            # to an account rather than to a mutable username.
            "github_user_id": user_data.get("id"),
        })

        # The HttpOnly cookie is authoritative for browser authentication.
        # Keep the query value temporarily for the existing header-based
        # frontend; it can be removed once that client migrates to cookie-only.
        response = RedirectResponse(
            url=_frontend_redirect(
                github_session=session_id,
                username=username,
            ),
            status_code=302,
        )
        response.headers.update({
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
            "Referrer-Policy": "no-referrer",
        })
        _set_session_cookie(response, session_id)
        return response

    except httpx.HTTPError:
        return RedirectResponse(
            url=_frontend_redirect(error="github_api_error"),
            status_code=302
        )


@router.get("/auth/status")
async def get_auth_status(
    request: Request,
    session_id: Optional[str] = None,
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
) -> GitHubOAuthResponse:
    """
    Check GitHub authentication status.

    Args:
        session_id: Session ID from OAuth flow

    Returns:
        Authentication status
    """
    resolved_session_id, session_data = _authenticated_session(
        request,
        header_session=github_session,
        explicit_session=session_id,
    )

    if not session_data or not resolved_session_id:
        return GitHubOAuthResponse(
            success=False,
            error="Session not found or expired"
        )

    return GitHubOAuthResponse(
        success=True,
        authenticated=True,
        session_id=resolved_session_id,
        user=session_data["username"],
        avatar=session_data.get("avatar_url"),
        username=session_data["username"],
    )


@router.get("/auth/verify", include_in_schema=False)
async def verify_github_auth(
    request: Request,
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
) -> Response:
    """Minimal auth subrequest endpoint for nginx WebSocket protection."""
    _session_id, session_data = _authenticated_session(
        request,
        header_session=github_session,
    )
    headers = {"Cache-Control": "no-store"}
    if not session_data:
        return Response(status_code=401, headers=headers)
    return Response(status_code=204, headers=headers)


@router.post("/auth/logout")
async def github_logout(
    request: Request,
    payload: Optional[GitHubLogoutRequest] = None,
    session_id: Optional[str] = None,
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
):
    """
    Logout and revoke GitHub session.

    Args:
        session_id: Session ID to revoke
    """
    explicit_session = payload.session_id if payload else session_id
    for candidate in github_session_candidates(
        request,
        header_session=github_session,
        explicit_session=explicit_session,
    ):
        _user_tokens.delete(candidate)

    response = JSONResponse(
        {"success": True, "message": "Logged out successfully"},
    )
    _clear_session_cookie(response)
    return response


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


def get_user_session(session_id: str) -> Optional[dict]:
    """Return a copy of the authenticated GitHub session metadata.

    Smart-generation authentication uses this provider adapter to build
    a provider-neutral principal. Returning a copy prevents callers from
    mutating the encrypted store's in-memory value accidentally.
    """
    session_data = _user_tokens.get(session_id)
    if not session_data:
        return None
    return dict(session_data)


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
