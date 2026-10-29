"""GitHub sign-in: the session id stays out of URLs, and logout ends the session and revokes the token."""

import base64
import json
import logging

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


def _store(monkeypatch, tmp_path):
    store = SessionStore(store_path=str(tmp_path / "tokens.enc"), ttl=600)
    store.set("sess-1", {"access_token": "gho_user_token", "username": "octocat"})
    monkeypatch.setattr(github_oauth, "_user_tokens", store)
    return store


def _deprecations(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "deprecated session_id" in r.getMessage()]


def test_the_session_status_reads_the_session_from_a_header(monkeypatch, tmp_path, caplog):
    """``?session_id=`` put the session, which unlocks the user's GitHub token, in proxy and server logs."""
    _store(monkeypatch, tmp_path)
    client = TestClient(app)

    with caplog.at_level(logging.WARNING, logger=github_oauth.logger.name):
        r = client.get("/besser_api/github/auth/status", headers={"X-GitHub-Session": "sess-1"})

    assert r.json()["username"] == "octocat"
    assert _deprecations(caplog) == []


def test_the_session_status_still_reads_the_old_query_parameter_with_a_warning(monkeypatch, tmp_path, caplog):
    """A frontend one release behind still sends ``?session_id=``; it lost its session after the backend moved."""
    _store(monkeypatch, tmp_path)
    client = TestClient(app)

    with caplog.at_level(logging.WARNING, logger=github_oauth.logger.name):
        r = client.get("/besser_api/github/auth/status?session_id=sess-1")

    assert r.json()["username"] == "octocat"
    assert len(_deprecations(caplog)) == 1


def test_the_star_endpoints_read_the_session_from_the_header_or_the_old_query(monkeypatch, tmp_path, caplog):
    _store(monkeypatch, tmp_path)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(github_oauth.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(lambda r: httpx.Response(204))))
    client = TestClient(app)

    with caplog.at_level(logging.WARNING, logger=github_oauth.logger.name):
        assert client.get("/besser_api/github/star/status", headers={"X-GitHub-Session": "sess-1"}).json() == {"starred": True}
        assert client.put("/besser_api/github/star", headers={"X-GitHub-Session": "sess-1"}).status_code == 200
        assert _deprecations(caplog) == []
        assert client.get("/besser_api/github/star/status?session_id=sess-1").json() == {"starred": True}
        assert client.put("/besser_api/github/star?session_id=sess-1").status_code == 200
        assert client.delete("/besser_api/github/star?session_id=sess-1").status_code == 200
    assert len(_deprecations(caplog)) == 3
    assert client.get("/besser_api/github/star/status").status_code == 401


def test_logout_still_reads_the_old_query_parameter_with_a_warning(monkeypatch, tmp_path, caplog):
    store = _store(monkeypatch, tmp_path)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(github_oauth.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(lambda r: httpx.Response(204))))

    with caplog.at_level(logging.WARNING, logger=github_oauth.logger.name):
        r = TestClient(app).post("/besser_api/github/auth/logout?session_id=sess-1")

    assert r.status_code == 200, r.text
    assert store.get("sess-1") is None
    assert len(_deprecations(caplog)) == 1
    assert TestClient(app).post("/besser_api/github/auth/logout").status_code == 422


def test_the_oauth_callback_hands_the_session_over_in_the_url_fragment(monkeypatch, tmp_path):
    _store(monkeypatch, tmp_path)
    states = SessionStore(store_path=str(tmp_path / "states.enc"), ttl=600)
    states.set("state-1", {"client_ip": "test"})
    monkeypatch.setattr(github_oauth, "_oauth_sessions", states)

    def github(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/login/oauth/access_token":
            return httpx.Response(200, json={"access_token": "gho_new"})
        return httpx.Response(200, json={"login": "octocat", "avatar_url": ""})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(github_oauth.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(github)))
    response = TestClient(app).get("/besser_api/github/auth/callback?code=c&state=state-1", follow_redirects=False)

    location = response.headers["location"]
    base, _, fragment = location.partition("#")
    assert "github_session" not in base and fragment.startswith("github_session=")
    assert fragment.endswith("&username=octocat")
