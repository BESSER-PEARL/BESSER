"""Who may use BESSER Lab: the login list, the route gate and the refused sign-in."""

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.routers import github_oauth
from besser.utilities.web_modeling_editor.backend.routers.auth import LabUserDep
from besser.utilities.web_modeling_editor.backend.services.lab.access import (
    LabUser,
    allowed_logins,
    is_login_allowed,
)
from besser.utilities.web_modeling_editor.backend.settings import get_settings


@pytest.fixture
def settings(route_settings):
    """A lab instance that allows two logins, with a GitHub OAuth app configured."""
    route_settings.lab_enabled = True
    route_settings.lab_allowed_logins = " Octocat ,, hubot "
    route_settings.github_client_id = "client-id"
    route_settings.github_client_secret = SecretStr("client-secret")
    return route_settings


# -- The login list -------------------------------------------------------------

def test_allowed_logins_are_trimmed_and_lower_cased_and_blanks_are_skipped(settings):
    assert allowed_logins(settings) == frozenset({"octocat", "hubot"})


@pytest.mark.parametrize("enabled, logins, login, expected", [
    (True, "octocat,hubot", "octocat", True),
    (True, "octocat,hubot", "OctoCat", True),
    (True, "octocat,hubot", "mallory", False),
    (True, "octocat", None, False),
    (True, "octocat", "", False),
    (True, "", "octocat", False),
    (True, " , ", "", False),
    (False, "octocat", "octocat", False),
])
def test_a_login_is_allowed_only_on_a_lab_that_lists_it(settings, enabled, logins, login, expected):
    settings.lab_enabled = enabled
    settings.lab_allowed_logins = logins
    assert is_login_allowed(settings, login) is expected


def test_a_lab_user_never_shows_its_token_in_its_repr():
    user = LabUser(login="octocat", session_id="sess-1", access_token="gho_secret")
    assert "gho_secret" not in repr(user)


# -- The route gate ---------------------------------------------------------------

@pytest.fixture
def gate(settings):
    """An app with one route behind ``LabUserDep``, sharing the backend's sign-in stores."""
    gate_app = FastAPI()
    gate_app.state.github_sessions = app.state.github_sessions
    gate_app.dependency_overrides[get_settings] = lambda: settings

    @gate_app.get("/lab/whoami")
    def whoami(user: LabUserDep) -> dict:
        return {"login": user.login, "session_id": user.session_id, "token": user.access_token}

    app.state.github_sessions.user_tokens.set("sess-octocat", {"access_token": "gho_octo", "username": "OctoCat"})
    app.state.github_sessions.user_tokens.set("sess-mallory", {"access_token": "gho_mal", "username": "mallory"})
    return TestClient(gate_app)


def test_a_listed_user_passes_the_gate(gate):
    response = gate.get("/lab/whoami", headers={"X-GitHub-Session": "sess-octocat"})

    assert response.status_code == 200, response.text
    assert response.json() == {"login": "OctoCat", "session_id": "sess-octocat", "token": "gho_octo"}


def test_every_lab_route_is_404_when_the_lab_is_off(gate, settings):
    settings.lab_enabled = False

    assert gate.get("/lab/whoami", headers={"X-GitHub-Session": "sess-octocat"}).status_code == 404
    assert gate.get("/lab/whoami").status_code == 404


@pytest.mark.parametrize("headers, params", [
    ({}, {}),
    ({"X-GitHub-Session": "sess-unknown"}, {}),
    # The deprecated query parameter puts the session in logs; the lab does not read it.
    ({}, {"session_id": "sess-octocat"}),
], ids=["no session", "expired session", "query parameter"])
def test_no_valid_session_header_is_401(gate, headers, params):
    assert gate.get("/lab/whoami", headers=headers, params=params).status_code == 401


def test_a_signed_in_user_not_on_the_list_is_403(gate):
    response = gate.get("/lab/whoami", headers={"X-GitHub-Session": "sess-mallory"})

    assert response.status_code == 403
    assert "gho_mal" not in response.text


# -- Sign-in on a lab instance ------------------------------------------------------

@pytest.fixture
def github(monkeypatch):
    """GitHub's OAuth and user endpoints, answering for ``github.login``; records token revocations."""
    states = app.state.github_sessions.oauth_states
    states.set("state-1", {"client_ip": "test"})

    class FakeGitHub:
        login = "octocat"
        revoked: list = []

        @classmethod
        def handle(cls, request: httpx.Request) -> httpx.Response:
            if request.url.path == "/login/oauth/access_token":
                return httpx.Response(200, json={"access_token": "gho_new"})
            if request.method == "DELETE" and request.url.path == "/applications/client-id/token":
                cls.revoked.append(json.loads(request.read())["access_token"])
                return httpx.Response(204)
            return httpx.Response(200, json={"login": cls.login, "avatar_url": ""})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(github_oauth.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(FakeGitHub.handle)))
    return FakeGitHub


def _sign_in() -> str:
    response = TestClient(app).get("/besser_api/github/auth/callback?code=c&state=state-1", follow_redirects=False)
    assert response.status_code == 302
    return response.headers["location"]


def _stored_sessions() -> dict:
    # SessionStore has no listing; its content is what the callback stored.
    return app.state.github_sessions.user_tokens._load()


def test_a_lab_refuses_the_sign_in_of_a_login_not_on_its_list(settings, github):
    github.login = "mallory"

    location = _sign_in()

    assert location == f"{settings.deployment_url}?error=not_allowed"
    assert _stored_sessions() == {}
    assert github.revoked == ["gho_new"]


def test_a_lab_signs_in_a_listed_login(settings, github):
    github.login = "HUBOT"

    location = _sign_in()

    session_id = location.partition("#github_session=")[2].partition("&")[0]
    assert app.state.github_sessions.user_tokens.get(session_id)["username"] == "HUBOT"
    assert github.revoked == []


def test_sign_in_is_unchanged_on_an_instance_that_is_not_a_lab(settings, github):
    settings.lab_enabled = False
    settings.lab_allowed_logins = ""
    github.login = "mallory"

    location = _sign_in()

    session_id = location.partition("#github_session=")[2].partition("&")[0]
    assert app.state.github_sessions.user_tokens.get(session_id)["access_token"] == "gho_new"
    assert location.endswith("&username=mallory")
    assert github.revoked == []
