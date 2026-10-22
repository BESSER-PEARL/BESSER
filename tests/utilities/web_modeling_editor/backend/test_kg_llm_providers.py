"""The KG LLM endpoints work with any supported provider key, not only OpenAI.

OpenAI keeps its own transport (covered in ``test_llm_clean_kg_endpoint.py``).
The other providers go through the Spec-Driven Agent's ``create_llm_client``,
which is stubbed here so no SDK or network is involved. The point of these
tests is the adapter in ``services/kg/llm.py``: which client is built, what it
is asked, and above all that a provider failure never reaches the browser.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from besser.spec_driven_agent.providers.llm_client import DEFAULT_MODELS
from tests.utilities.web_modeling_editor.backend.test_llm_clean_kg_endpoint import (
    _kg_diagram,
    _kg_with_orphan_diagram,
)

_API = "/besser_api"
_SUGGESTION = {
    "code": "LLM_DROP_CLASS",
    "description": "Car is unrelated.",
    "affected_node_ids": ["Car"],
    "affected_edge_ids": [],
    "recommended_action": {"key": "drop_class", "parameters": {"node_id": "Car"}, "label": "Drop Car"},
}
KEY = "sk-live-SECRET123"


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


class _FakeClient:
    """Stands in for an ``LLMProvider``; records the one call it receives."""

    def __init__(self, content=None, error=None):
        self._content = content
        self._error = error
        self.calls = []

    def chat(self, system, messages, tools, **kwargs):
        self.calls.append({"system": system, "messages": messages, "tools": tools})
        if self._error is not None:
            raise self._error
        return {"stop_reason": "end_turn", "content": self._content}


@pytest.fixture
def fake_factory(monkeypatch):
    """Patch ``create_llm_client``; returns the list of (kwargs) it was called with."""
    state = SimpleNamespace(created=[], client=_FakeClient([_text_block(json.dumps({"suggestions": [_SUGGESTION]}))]),
                            create_error=None)

    def create(**kwargs):
        state.created.append(kwargs)
        if state.create_error is not None:
            raise state.create_error
        return state.client

    monkeypatch.setattr("besser.spec_driven_agent.providers.llm_client.create_llm_client", create)
    return state


def _clean(client, **over):
    body = {"diagram": _kg_diagram(), "description": "people", "api_key": KEY, "provider": "anthropic", **over}
    return client.post(f"{_API}/kg/llm-clean", json=body)


@pytest.mark.parametrize("provider", ["anthropic", "mistral", "nebius"])
def test_a_non_openai_key_is_used_with_its_own_provider(client, fake_factory, provider):
    response = _clean(client, provider=provider)

    assert response.status_code == 200, response.text
    assert response.json()["issues"][0]["recommendedAction"]["key"] == "drop_class"
    (created,) = fake_factory.created
    assert created["provider"] == provider
    assert created["api_key"] == KEY
    assert created["model"] == DEFAULT_MODELS[provider]
    # A text-only call: the prompt goes in as one user message and no tools are offered.
    (call,) = fake_factory.client.calls
    assert call["tools"] == []
    assert call["messages"][0]["role"] == "user"
    assert "Car" in call["messages"][0]["content"]


def test_an_explicit_model_overrides_the_provider_default(client, fake_factory):
    _clean(client, llm_model="claude-haiku-4-5")

    assert fake_factory.created[0]["model"] == "claude-haiku-4-5"


def test_text_blocks_are_joined_and_tool_blocks_ignored(client, fake_factory):
    payload = json.dumps({"suggestions": [_SUGGESTION]})
    fake_factory.client = _FakeClient([
        _text_block(payload[:20]),
        SimpleNamespace(type="tool_use", name="x", input={}),
        _text_block(payload[20:]),
    ])

    response = _clean(client)

    assert response.status_code == 200, response.text
    assert response.json()["issueCount"] == 1


def test_classify_orphans_uses_the_provider_too(client, fake_factory):
    reply = {"suggestions": [{
        "code": "LLM_DROP_NODE", "description": "stray", "affected_node_ids": ["ghost1"],
        "affected_edge_ids": [],
        "recommended_action": {"key": "drop_node", "parameters": {"node_id": "ghost1"}, "label": "Drop"},
    }]}
    fake_factory.client = _FakeClient([_text_block(json.dumps(reply))])

    response = client.post(f"{_API}/kg/classify-orphans", json={
        "diagram": _kg_with_orphan_diagram(), "api_key": KEY, "provider": "mistral",
        "nodeIds": ["ghost1"],
    })

    assert response.status_code == 200, response.text
    assert fake_factory.created[0]["provider"] == "mistral"
    assert response.json()["diagramType"] == "OrphanClassification"


@pytest.mark.parametrize("failure", [
    RuntimeError(f"Claude API call failed: 401 invalid x-api-key {KEY} org-internal-42"),
    ValueError(f"Claude API rejected the key: {KEY}"),
])
def test_a_provider_failure_is_a_502_that_leaks_nothing(client, fake_factory, caplog, failure):
    fake_factory.client = _FakeClient(error=failure)

    with caplog.at_level("WARNING"):
        response = _clean(client)

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert KEY not in detail and "org-internal" not in detail and "x-api-key" not in detail
    assert "Anthropic" in detail
    # The cause is logged for the operator, with the key redacted.
    assert "org-internal-42" in caplog.text or "rejected the key" in caplog.text
    assert KEY not in caplog.text


def test_failing_to_build_the_client_is_a_502_without_the_cause(client, fake_factory):
    fake_factory.create_error = ValueError(f"bad key {KEY}")

    response = _clean(client)

    assert response.status_code == 502
    assert KEY not in response.text


def test_a_server_without_the_provider_sdk_says_so(client, fake_factory):
    fake_factory.create_error = ImportError("No module named 'anthropic'")

    response = _clean(client)

    assert response.status_code == 502
    assert "not available on this server" in response.json()["detail"]


def test_a_blank_key_is_a_readable_400(client, fake_factory):
    response = _clean(client, api_key="   ")

    assert response.status_code == 400
    assert response.json()["detail"] == "An API key is required."
    assert fake_factory.created == []


@pytest.mark.parametrize("provider", ["free", "sponsored", "pia", "local", "gemini"])
def test_only_byok_providers_are_accepted(client, fake_factory, provider):
    """The server-paid tiers and the base_url providers (an SSRF surface) are not offered."""
    response = _clean(client, provider=provider)

    assert response.status_code == 422
    assert fake_factory.created == []


def test_a_base_url_in_the_body_is_never_forwarded(client, fake_factory):
    _clean(client, base_url="http://169.254.169.254/")

    assert "base_url" not in fake_factory.created[0]


def test_the_key_is_not_rendered_by_the_request_model():
    from besser.utilities.web_modeling_editor.backend.models import KGLLMRequest

    request = KGLLMRequest(diagram={"title": "t", "model": {}}, api_key=KEY)

    assert KEY not in repr(request) and KEY not in str(request.model_dump())
    assert request.resolved_api_key() == KEY
