from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport

from besser.utilities.web_modeling_editor.backend.services import (
    principal as principal_module,
)
from besser.utilities.web_modeling_editor.backend.services.deployment import (
    github_oauth,
)


class _MemoryStore:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, Any]] = {}

    def get(self, key: str):
        value = self.values.get(key)
        return dict(value) if value is not None else None

    def set(self, key: str, value: dict[str, Any]) -> None:
        self.values[key] = dict(value)

    def delete(self, key: str) -> None:
        self.values.pop(key, None)


class _GitHubResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return dict(self._payload)


class _GitHubClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args) -> None:
        return None

    async def post(self, *_args, **_kwargs) -> _GitHubResponse:
        return _GitHubResponse({"access_token": "github-access-token"})

    async def get(self, *_args, **_kwargs) -> _GitHubResponse:
        return _GitHubResponse({
            "login": "alice",
            "id": 101,
            "avatar_url": "https://avatars.example/alice",
        })


@pytest.fixture
def oauth_app(monkeypatch):
    user_tokens = _MemoryStore()
    oauth_states = _MemoryStore()
    monkeypatch.setattr(github_oauth, "_user_tokens", user_tokens)
    monkeypatch.setattr(github_oauth, "_oauth_sessions", oauth_states)
    monkeypatch.setattr(github_oauth, "DEPLOYMENT_URL", "https://editor.example/app")

    app = FastAPI()
    app.include_router(github_oauth.router, prefix="/besser_api")
    return app, user_tokens, oauth_states


def _request(app: FastAPI, method: str, path: str, **kwargs) -> httpx.Response:
    async def run() -> httpx.Response:
        cookies = kwargs.pop("cookies", None)
        if cookies:
            headers = dict(kwargs.pop("headers", {}))
            headers["Cookie"] = "; ".join(
                f"{name}={value}" for name, value in cookies.items()
            )
            kwargs["headers"] = headers
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="https://testserver",
            follow_redirects=False,
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(run())


def test_oauth_callback_sets_hardened_session_cookie(
    oauth_app,
    monkeypatch,
) -> None:
    app, user_tokens, oauth_states = oauth_app
    oauth_states.set("valid-state-token", {"client_ip": "127.0.0.1"})
    real_async_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        if "transport" in kwargs:
            return real_async_client(*args, **kwargs)
        return _GitHubClient()

    monkeypatch.setattr(github_oauth.httpx, "AsyncClient", client_factory)

    response = _request(
        app,
        "GET",
        "/besser_api/github/auth/callback",
        params={"code": "oauth-code", "state": "valid-state-token"},
    )

    assert response.status_code == 302
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"
    session_id = next(iter(user_tokens.values))
    assert user_tokens.get(session_id)["github_user_id"] == 101
    cookie = response.headers["set-cookie"]
    assert f"{github_oauth.GITHUB_SESSION_COOKIE_NAME}={session_id}" in cookie
    assert "HttpOnly" in cookie
    assert "Secure" in cookie
    assert "SameSite=lax" in cookie
    assert "Path=/" in cookie
    assert f"Max-Age={github_oauth.GITHUB_SESSION_TTL_SECONDS}" in cookie


def test_auth_verify_accepts_cookie_and_header_fallback(oauth_app) -> None:
    app, user_tokens, _oauth_states = oauth_app
    user_tokens.set("cookie-session-123", {"username": "alice"})
    user_tokens.set("header-session-456", {"username": "bob"})

    cookie_response = _request(
        app,
        "GET",
        "/besser_api/github/auth/verify",
        cookies={github_oauth.GITHUB_SESSION_COOKIE_NAME: "cookie-session-123"},
    )
    header_response = _request(
        app,
        "GET",
        "/besser_api/github/auth/verify",
        headers={"X-GitHub-Session": "header-session-456"},
    )
    denied_response = _request(
        app,
        "GET",
        "/besser_api/github/auth/verify",
    )

    assert cookie_response.status_code == 204
    assert header_response.status_code == 204
    assert denied_response.status_code == 401
    assert cookie_response.headers["cache-control"] == "no-store"


def test_invalid_cookie_does_not_block_valid_header_fallback(oauth_app) -> None:
    app, user_tokens, _oauth_states = oauth_app
    user_tokens.set("header-session-456", {"username": "bob"})

    response = _request(
        app,
        "GET",
        "/besser_api/github/auth/verify",
        cookies={github_oauth.GITHUB_SESSION_COOKIE_NAME: "expired-session-123"},
        headers={"X-GitHub-Session": "header-session-456"},
    )

    assert response.status_code == 204


def test_principal_uses_secure_cookie(monkeypatch) -> None:
    monkeypatch.setenv("BESSER_SMART_GEN_AUTH_REQUIRED", "true")
    monkeypatch.setattr(
        principal_module,
        "get_user_session",
        lambda session_id: {
            "username": "alice",
            "github_user_id": 101,
        } if session_id == "cookie-session-123" else None,
    )

    app = FastAPI()

    @app.get("/principal")
    async def principal_endpoint(
        principal: principal_module.Principal = Depends(
            principal_module.get_current_principal,
        ),
    ):
        return {"subject": principal.subject}

    response = _request(
        app,
        "GET",
        "/principal",
        cookies={github_oauth.GITHUB_SESSION_COOKIE_NAME: "cookie-session-123"},
    )

    assert response.status_code == 200
    assert response.json() == {"subject": "github:101"}


def test_logout_revokes_cookie_session_and_expires_cookie(oauth_app) -> None:
    app, user_tokens, _oauth_states = oauth_app
    user_tokens.set("cookie-session-123", {"username": "alice"})

    response = _request(
        app,
        "POST",
        "/besser_api/github/auth/logout",
        cookies={github_oauth.GITHUB_SESSION_COOKIE_NAME: "cookie-session-123"},
    )

    assert response.status_code == 200
    assert user_tokens.get("cookie-session-123") is None
    cookie = response.headers["set-cookie"]
    assert f"{github_oauth.GITHUB_SESSION_COOKIE_NAME}=" in cookie
    assert "Max-Age=0" in cookie
    assert "HttpOnly" in cookie
    assert "Secure" in cookie
    assert "SameSite=lax" in cookie


def test_logout_keeps_json_header_client_compatibility(oauth_app) -> None:
    app, user_tokens, _oauth_states = oauth_app
    user_tokens.set("legacy-session-123", {"username": "legacy"})

    response = _request(
        app,
        "POST",
        "/besser_api/github/auth/logout",
        json={"session_id": "legacy-session-123"},
    )

    assert response.status_code == 200
    assert user_tokens.get("legacy-session-123") is None
