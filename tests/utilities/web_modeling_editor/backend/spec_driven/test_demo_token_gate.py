"""The server-paid ``sponsored`` tier is reachable only with the demo secret.

Demo links carry it as ``?demo=<token>``. Without this gate the tier is an
open API proxy for the org's OpenAI key: the request model accepts
``provider: "sponsored"`` from anyone and strips the client key, and the
factory then builds a client on the SERVER's endpoint and bearer token. A
single curl would spend our credits, bounded only by the per-run cost cap.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.models.spec_driven import (
    SmartGenerateRequest,
)
from besser.utilities.web_modeling_editor.backend.routers.spec_driven_router import (
    _require_demo_token,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_spec_driven_router import (
    _build_project_body,
)

BASE_URL = "http://testserver"
DEMO_TOKEN = "s3cret-demo-token"


def _sponsored_body(**overrides) -> dict:
    body = _build_project_body(provider="sponsored", llm_model="gpt-5.6-terra")
    # The sponsored tier never carries a client key — the server injects both
    # the endpoint and the bearer token.
    body.pop("api_key", None)
    body.update(overrides)
    return body


def _request(**overrides) -> SmartGenerateRequest:
    return SmartGenerateRequest(**_sponsored_body(**overrides))


def _post(path: str, body: dict) -> httpx.Response:
    async def _go() -> httpx.Response:
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
            return await ac.post(path, json=body)

    return asyncio.run(_go())


@pytest.fixture
def _demo_enabled(monkeypatch):
    monkeypatch.setenv("BESSER_DEMO_TOKEN", DEMO_TOKEN)


# ======================================================================
# The gate itself
# ======================================================================

def test_the_right_token_is_accepted(_demo_enabled):
    _require_demo_token(_request(demo_token=DEMO_TOKEN))  # does not raise


def test_no_token_is_refused(_demo_enabled):
    with pytest.raises(HTTPException) as exc:
        _require_demo_token(_request())
    assert exc.value.status_code == 403


def test_a_wrong_token_is_refused(_demo_enabled):
    with pytest.raises(HTTPException) as exc:
        _require_demo_token(_request(demo_token="not-the-token"))
    assert exc.value.status_code == 403


def test_a_prefix_of_the_token_is_refused(_demo_enabled):
    """Guards against a truthy/startswith comparison creeping in."""
    with pytest.raises(HTTPException) as exc:
        _require_demo_token(_request(demo_token=DEMO_TOKEN[:5]))
    assert exc.value.status_code == 403


def test_a_non_ascii_token_is_refused_not_crashed(_demo_enabled):
    """``hmac.compare_digest`` raises TypeError on a non-ASCII str.

    Comparing the raw strings turned junk input into a 500 (and an incident)
    instead of a plain 403.
    """
    with pytest.raises(HTTPException) as exc:
        _require_demo_token(_request(demo_token="tökén-with-accents"))
    assert exc.value.status_code == 403


def test_an_oversized_token_is_rejected_by_the_model():
    """The secret is bounded before it reaches the comparison."""
    with pytest.raises(ValidationError):
        _request(demo_token="a" * 201)


def test_an_unset_server_token_fails_closed(monkeypatch):
    """A half-finished env edit must not leave the org's key wide open."""
    monkeypatch.delenv("BESSER_DEMO_TOKEN", raising=False)
    with pytest.raises(HTTPException) as exc:
        _require_demo_token(_request(demo_token=DEMO_TOKEN))
    assert exc.value.status_code == 403


def test_a_blank_server_token_fails_closed(monkeypatch):
    monkeypatch.setenv("BESSER_DEMO_TOKEN", "   ")
    with pytest.raises(HTTPException) as exc:
        _require_demo_token(_request(demo_token="   "))
    assert exc.value.status_code == 403


def test_the_refusal_does_not_say_which_half_was_wrong(_demo_enabled):
    with pytest.raises(HTTPException) as exc:
        _require_demo_token(_request(demo_token="not-the-token"))
    detail = exc.value.detail.lower()
    assert "token" not in detail and "invalid" not in detail


@pytest.mark.parametrize("provider", ["anthropic", "openai", "mistral", "free"])
def test_other_providers_need_no_demo_token(provider, monkeypatch):
    """The gate guards the org's wallet, not every run on the server."""
    monkeypatch.delenv("BESSER_DEMO_TOKEN", raising=False)
    body = _build_project_body(provider=provider)
    if provider == "free":
        body.pop("api_key", None)
        body.pop("llm_model", None)
    else:
        body["api_key"] = "sk-test-key"
        body["llm_model"] = None
    _require_demo_token(SmartGenerateRequest(**body))  # does not raise


# ======================================================================
# Both endpoints that accept a SmartGenerateRequest
# ======================================================================

def test_generate_refuses_an_unauthorised_sponsored_run(_demo_enabled):
    resp = _post("/besser_api/spec-driven/generate", _sponsored_body())
    assert resp.status_code == 403


def test_resume_refuses_an_unauthorised_sponsored_run(_demo_enabled):
    """Resume takes the same body and spends the same way.

    It is checked before the run_id is even validated, so it cannot be used
    as the way around the gate on /generate.
    """
    resp = _post(
        "/besser_api/spec-driven/resume/" + "a" * 32, _sponsored_body(),
    )
    assert resp.status_code == 403


def test_the_secret_never_appears_in_the_models_repr():
    """demo_token is a SecretStr — a logged request must not leak it."""
    assert DEMO_TOKEN not in repr(_request(demo_token=DEMO_TOKEN))
