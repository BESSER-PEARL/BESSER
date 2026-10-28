"""GitHub logout: the editor's request ends the session and revokes the token at GitHub."""

import base64
import json

import httpx
from fastapi.testclient import TestClient

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.services.deployment import github_oauth
from besser.utilities.web_modeling_editor.backend.services.deployment.session_store import SessionStore


def test_logout_with_the_editors_json_body_ends_the_session_and_revokes_the_token(monkeypatch, tmp_path):
    """The editor posts ``{"session_id": ...}``; the endpoint read a query parameter and answered 422."""
    store = SessionStore(store_path=str(tmp_path / "tokens.enc"), ttl=600)
    store.set("sess-1", {"access_token": "gho_user_token", "username": "octocat"})
    monkeypatch.setattr(github_oauth, "_user_tokens", store)
    monkeypatch.setattr(github_oauth, "GITHUB_CLIENT_ID", "client-id")
    monkeypatch.setattr(github_oauth, "GITHUB_CLIENT_SECRET", "client-secret")

    revoked = []

    def github(request: httpx.Request) -> httpx.Response:
        revoked.append((request.method, request.url.path, request.headers["authorization"],
                        json.loads(request.read())))
        return httpx.Response(204)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(github_oauth.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(github)))

    response = TestClient(app).post("/besser_api/github/auth/logout", json={"session_id": "sess-1"})

    assert response.status_code == 200
    assert store.get("sess-1") is None
    basic = "Basic " + base64.b64encode(b"client-id:client-secret").decode()
    assert revoked == [("DELETE", "/applications/client-id/token", basic, {"access_token": "gho_user_token"})]
