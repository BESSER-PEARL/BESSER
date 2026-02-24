"""
GitHub OAuth Integration for BESSER Web Editor.

Handles OAuth flow to authenticate users with their GitHub accounts.
"""

import os
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import requests
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel


# GitHub OAuth configuration
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")
GITHUB_REDIRECT_URI = os.getenv("GITHUB_REDIRECT_URI", "http://localhost:9000/besser_api/github/auth/callback")
DEPLOYMENT_URL = os.getenv("DEPLOYMENT_URL", "http://localhost:8080")


# In-memory session store (replace with Redis in production)
_oauth_sessions: Dict[str, Dict[str, Any]] = {}
_user_tokens: Dict[str, Dict[str, Any]] = {}


class GitHubOAuthResponse(BaseModel):
    """Response model for OAuth flow."""
    success: bool
    access_token: Optional[str] = None
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
    
    # Store state in session
    _oauth_sessions[state] = {
        "created_at": datetime.utcnow(),
        "client_ip": request.client.host if request.client else "unknown"
    }
    
    # Clean up old sessions (older than 10 minutes)
    _cleanup_old_sessions()
    
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
    if state not in _oauth_sessions:
        return RedirectResponse(
            url=f"{DEPLOYMENT_URL}?error=invalid_state",
            status_code=302
        )
    
    # Remove used state
    del _oauth_sessions[state]
    
    try:
        # Exchange code for access token
        token_response = requests.post(
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
        user_response = requests.get(
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
        
        # Store token (in production, use encrypted session/cookie)
        session_id = secrets.token_urlsafe(32)
        _user_tokens[session_id] = {
            "access_token": access_token,
            "username": username,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(hours=8)
        }
        
        # Redirect to frontend with session ID
        return RedirectResponse(
            url=f"{DEPLOYMENT_URL}?github_session={session_id}&username={username}",
            status_code=302
        )
        
    except requests.RequestException as e:
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
    
    # Check if token expired
    if datetime.utcnow() > session_data["expires_at"]:
        del _user_tokens[session_id]
        return GitHubOAuthResponse(
            success=False,
            error="Session expired"
        )
    
    return GitHubOAuthResponse(
        success=True,
        access_token=session_data["access_token"],
        username=session_data["username"]
    )


@router.post("/auth/logout")
async def github_logout(session_id: str):
    """
    Logout and revoke GitHub session.
    
    Args:
        session_id: Session ID to revoke
    """
    if session_id in _user_tokens:
        del _user_tokens[session_id]
    
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
    
    # Check expiration
    if datetime.utcnow() > session_data["expires_at"]:
        del _user_tokens[session_id]
        return None
    
    return session_data["access_token"]


def _cleanup_old_sessions():
    """Remove OAuth sessions older than 10 minutes."""
    cutoff_time = datetime.utcnow() - timedelta(minutes=10)
    
    # Clean OAuth sessions
    to_remove = [
        state for state, data in _oauth_sessions.items()
        if data["created_at"] < cutoff_time
    ]
    for state in to_remove:
        del _oauth_sessions[state]
    
    # Clean expired user tokens
    to_remove = [
        session_id for session_id, data in _user_tokens.items()
        if data["expires_at"] < datetime.utcnow()
    ]
    for session_id in to_remove:
        del _user_tokens[session_id]
