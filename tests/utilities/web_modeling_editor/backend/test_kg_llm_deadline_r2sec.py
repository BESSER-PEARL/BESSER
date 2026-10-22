"""One server deadline per KG LLM request, published through ``GET /kg-config``.

Also covers the legacy ``/get-json-model-from-kg`` endpoint, which now goes
through the shared OpenAI transport, off the event loop, under the same deadline.
"""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import pytest

from besser.utilities.web_modeling_editor.backend.constants.constants import (
    KG_LLM_REQUEST_TIMEOUT_SECONDS,
    MAX_ORPHAN_CLASSIFY_NODES,
    TEMP_DIR_PREFIX,
)
from besser.BUML.notations.kg_to_buml.llm_cleanup import MAX_ORPHAN_BATCH
from besser.utilities.kg_llm_to_buml import LLMAnswerError
from tests.utilities.web_modeling_editor.backend.test_llm_clean_kg_endpoint import _kg_diagram

_API = "/besser_api"
_ROUTER = "besser.utilities.web_modeling_editor.backend.routers.kg_router"
_LLM = "besser.utilities.web_modeling_editor.backend.services.kg.llm"
KEY = "sk-live-SECRET123"
EX = "http://example.org/"


# --------------------------------------------------------------------------
# GET /kg-config
# --------------------------------------------------------------------------


def test_kg_config_publishes_the_server_limits(client):
    response = client.get(f"{_API}/kg-config")

    assert response.status_code == 200
    assert response.json() == {
        "llmRequestTimeoutSeconds": KG_LLM_REQUEST_TIMEOUT_SECONDS,
        "maxOrphanClassifyNodes": MAX_ORPHAN_CLASSIFY_NODES,
    }


def test_one_orphan_request_is_at_most_two_model_calls():
    assert MAX_ORPHAN_CLASSIFY_NODES <= 2 * MAX_ORPHAN_BATCH


# --------------------------------------------------------------------------
# The deadline
# --------------------------------------------------------------------------


class _SlowClient:
    def __init__(self, delay: float, content: str):
        self.delay = delay
        self.content = content
        self.calls = 0

    def chat(self, system, messages, tools, **kwargs):
        self.calls += 1
        time.sleep(self.delay)
        return {"stop_reason": "end_turn", "content": [SimpleNamespace(type="text", text=self.content)]}


@pytest.fixture
def slow_provider(monkeypatch):
    """A provider client that takes ``delay`` seconds per call; records each creation."""
    state = SimpleNamespace(created=[], client=_SlowClient(0.0, json.dumps({"suggestions": []})))

    def create(**kwargs):
        state.created.append(kwargs)
        return state.client

    monkeypatch.setattr("besser.spec_driven_agent.providers.llm_client.create_llm_client", create)
    return state


def _budget(monkeypatch, seconds: float, min_call: float) -> None:
    monkeypatch.setattr(f"{_ROUTER}.KG_LLM_REQUEST_TIMEOUT_SECONDS", seconds)
    monkeypatch.setattr(f"{_LLM}._MIN_CALL_SECONDS", min_call)


def test_each_provider_call_gets_the_remaining_budget_as_its_timeout(client, slow_provider):
    response = client.post(f"{_API}/llm-clean-kg", json={
        "diagram": _kg_diagram(), "api_key": KEY, "provider": "anthropic",
    })

    assert response.status_code == 200, response.text
    (created,) = slow_provider.created
    assert 0 < created["timeout"] <= KG_LLM_REQUEST_TIMEOUT_SECONDS
    assert created["max_retries"] == 0


def test_a_call_that_outlives_the_deadline_is_a_502(client, slow_provider, monkeypatch):
    _budget(monkeypatch, 1.0, 0.1)
    slow_provider.client.delay = 2.0

    started = time.monotonic()
    response = client.post(f"{_API}/llm-clean-kg", json={
        "diagram": _kg_diagram(), "api_key": KEY, "provider": "anthropic",
    })

    assert response.status_code == 502
    assert "did not finish within the 1-second limit" in response.json()["detail"]
    # The test client waits for the abandoned worker thread; the server does not.
    assert time.monotonic() - started < 10


def _many_orphans_diagram(count: int):
    nodes = [{"id": "Person", "nodeType": "class", "label": "Person", "iri": EX + "Person"}]
    nodes += [
        {"id": f"ghost{i}", "nodeType": "individual", "label": f"ghost{i}", "iri": f"{EX}ghost{i}"}
        for i in range(count)
    ]
    return {"title": "Orphans", "model": {"type": "KnowledgeGraphDiagram", "version": "1.0.0",
                                          "nodes": nodes, "edges": []}}


def test_no_further_batch_starts_once_the_budget_is_nearly_spent(client, slow_provider, monkeypatch):
    """Two batches: the first leaves less than the minimum, so the second is refused."""
    _budget(monkeypatch, 2.0, 1.5)
    slow_provider.client.delay = 0.8
    count = MAX_ORPHAN_BATCH + 1

    response = client.post(f"{_API}/classify-orphans-with-llm", json={
        "diagram": _many_orphans_diagram(count), "api_key": KEY, "provider": "mistral",
        "node_ids": [f"ghost{i}" for i in range(count)],
    })

    assert response.status_code == 502
    assert "limit for this request" in response.json()["detail"]
    assert slow_provider.client.calls == 1


def test_the_orphan_cap_is_enforced_with_the_published_value(client):
    count = MAX_ORPHAN_CLASSIFY_NODES + 1
    response = client.post(f"{_API}/classify-orphans-with-llm", json={
        "diagram": _many_orphans_diagram(count), "api_key": KEY,
        "node_ids": [f"ghost{i}" for i in range(count)],
    })

    assert response.status_code == 400
    assert str(MAX_ORPHAN_CLASSIFY_NODES) in response.json()["detail"]


def test_the_openai_path_passes_the_remaining_budget_too(client, monkeypatch):
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        seen["timeout"] = timeout
        return SimpleNamespace(ok=True, status_code=200, text="",
                               json=lambda: {"choices": [{"message": {"content": '{"suggestions": []}'}}]})

    monkeypatch.setattr("requests.post", fake_post)
    response = client.post(f"{_API}/llm-clean-kg", json={"diagram": _kg_diagram(), "api_key": KEY})

    assert response.status_code == 200, response.text
    assert 0 < seen["timeout"] <= KG_LLM_REQUEST_TIMEOUT_SECONDS


# --------------------------------------------------------------------------
# Legacy /get-json-model-from-kg
# --------------------------------------------------------------------------

_TTL = b"@prefix ex: <http://ex.org/> . ex:Person a ex:Class ."


def _legacy(client, api_key=KEY, filename="graph.ttl"):
    return client.post(
        f"{_API}/get-json-model-from-kg",
        files={"kg_file": (filename, _TTL, "text/turtle")},
        data={"api_key": api_key},
    )


def test_legacy_upstream_failure_is_a_502_without_the_body_or_key(client, monkeypatch, caplog):
    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        assert timeout is not None and timeout <= KG_LLM_REQUEST_TIMEOUT_SECONDS
        return SimpleNamespace(ok=False, status_code=401, text=f"invalid key {KEY} org-internal-42",
                               json=lambda: {})

    monkeypatch.setattr("requests.post", fake_post)
    with caplog.at_level("WARNING"):
        response = _legacy(client)

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail == "OpenAI API call failed: HTTP 401"
    assert KEY not in response.text and "org-internal" not in response.text
    assert KEY not in caplog.text


def test_legacy_uses_a_prefixed_temp_dir_and_a_server_chosen_name(client, monkeypatch):
    seen = {}

    def fake_kg_to_buml(kg_path, openai_token, *, timeout):
        seen.update(path=kg_path, token=openai_token, timeout=timeout)
        raise LLMAnswerError("Failed to parse JSON from the model's answer: it quoted secrets")

    monkeypatch.setattr(f"{_ROUTER}.kg_to_buml", fake_kg_to_buml)
    response = _legacy(client, filename="../../evil.ttl")

    assert response.status_code == 400
    assert "could not be turned into a class diagram" in response.json()["detail"]
    assert "secrets" not in response.text
    assert os.path.basename(os.path.dirname(seen["path"])).startswith(TEMP_DIR_PREFIX)
    assert os.path.basename(seen["path"]) == "knowledge_graph.ttl"
    assert seen["token"] == KEY
    assert 0 < seen["timeout"] <= KG_LLM_REQUEST_TIMEOUT_SECONDS


def test_legacy_failure_outside_the_llm_is_not_blamed_on_it(client, monkeypatch):
    def fake_kg_to_buml(kg_path, openai_token, *, timeout):
        raise RuntimeError("Failed to read KG file: disk on fire")

    monkeypatch.setattr(f"{_ROUTER}.kg_to_buml", fake_kg_to_buml)
    response = _legacy(client)

    assert response.status_code == 500
    assert "LLM" not in response.json()["detail"]
    assert "disk on fire" not in response.text


def test_legacy_rejects_a_file_that_is_not_utf8_before_calling_the_llm(client, monkeypatch):
    called = []
    monkeypatch.setattr(f"{_ROUTER}.kg_to_buml", lambda *a, **k: called.append(a))
    response = client.post(
        f"{_API}/get-json-model-from-kg",
        files={"kg_file": ("graph.ttl", b"\xff\xfe not utf-8 \x80", "text/turtle")},
        data={"api_key": KEY},
    )

    assert response.status_code == 400
    assert "UTF-8" in response.json()["detail"]
    assert called == []


def test_legacy_endpoint_is_marked_deprecated_in_openapi(client):
    operation = client.get("/openapi.json").json()["paths"]["/besser_api/get-json-model-from-kg"]["post"]
    assert operation.get("deprecated") is True


def test_legacy_blank_key_is_a_400(client):
    response = _legacy(client, api_key="   ")

    assert response.status_code == 400
    assert response.json()["detail"] == "An API key is required."


def test_a_provider_call_is_one_attempt_so_no_retry_outlives_the_deadline(monkeypatch):
    """The spec-driven clients retry with backoff by default; the KG completer turns that off.

    A retry would start after ``run_with_deadline`` has given up, in a worker
    thread nobody waits for, so a failed call must fail at once.
    """
    from besser.spec_driven_agent.providers import llm_client
    from besser.utilities.llm_utils import LLMUpstreamError
    from besser.utilities.web_modeling_editor.backend.models import KGLLMRequest
    from besser.utilities.web_modeling_editor.backend.services.kg.llm import (
        RequestDeadline,
        build_completer,
    )

    attempts, sleeps = [], []

    class _OverloadedCompletions:
        def create(self, **kwargs):
            attempts.append(kwargs)
            raise Exception("Error code: 503 - the model is overloaded")  # retryable

    real_create = llm_client.create_llm_client

    def create(**kwargs):
        provider_client = real_create(**kwargs)
        provider_client._client = SimpleNamespace(chat=SimpleNamespace(completions=_OverloadedCompletions()))
        return provider_client

    monkeypatch.setattr(llm_client, "create_llm_client", create)
    monkeypatch.setattr(llm_client.time, "sleep", sleeps.append)
    request = KGLLMRequest(diagram=_kg_diagram(), api_key=KEY, provider="mistral")
    complete = build_completer(request, RequestDeadline.start(60))

    with pytest.raises(LLMUpstreamError):
        complete("system", "user")

    assert len(attempts) == 1
    assert sleeps == []


@pytest.mark.parametrize("max_retries, attempts", [(None, 5), (0, 1), (2, 3)])
def test_the_provider_retry_budget_is_a_constructor_knob(monkeypatch, max_retries, attempts):
    from besser.spec_driven_agent.errors import UpstreamLLMError
    from besser.spec_driven_agent.providers import llm_client

    calls = []

    class _Overloaded:
        def create(self, **kwargs):
            calls.append(kwargs)
            raise Exception("Error code: 503 - the model is overloaded")

    provider_client = llm_client.MistralProvider(api_key="k", max_retries=max_retries)
    provider_client._client = SimpleNamespace(chat=SimpleNamespace(completions=_Overloaded()))
    monkeypatch.setattr(llm_client.time, "sleep", lambda _s: None)

    with pytest.raises(UpstreamLLMError):
        provider_client.chat("system", [{"role": "user", "content": "hi"}], tools=[])

    assert len(calls) == attempts
