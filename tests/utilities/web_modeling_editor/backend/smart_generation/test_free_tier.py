"""Regression tests for the keyless "free" tier (server-hosted open-weight model).

Covers the full contract so a provider Literal that forgets ``"free"`` (which
once broke every free run at the StartEvent) fails loudly here instead of in
production:

  * StartEvent accepts provider='free'  (the exact production bug)
  * SmartGenerateRequest: 'free' needs no key; other providers still require one
  * create_llm_client('free') injects the server endpoint + bearer header,
    and fails fast when the server isn't configured
  * unknown/local models price at $0 so a free run can't trip the cost cap
"""

import pytest

from besser.generators.llm.llm_client import (
    _get_pricing,
    create_llm_client,
    free_tier_available,
    free_tier_model,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.sse_events import (
    StartEvent,
)


# ---------------------------------------------------------------------
# StartEvent — the field that broke every free run in production
# ---------------------------------------------------------------------

def test_start_event_accepts_free_provider():
    ev = StartEvent(
        runId="abc", provider="free", llmModel="qwen3-coder:30b",
        maxCost=1.0, maxRuntime=600,
    )
    assert ev.provider == "free"


@pytest.mark.parametrize("provider", ["anthropic", "openai", "mistral", "free"])
def test_start_event_all_supported_providers(provider):
    StartEvent(runId="x", provider=provider, llmModel="m", maxCost=1.0, maxRuntime=1)


def test_start_event_rejects_unknown_provider():
    with pytest.raises(Exception):
        StartEvent(runId="x", provider="ollama", llmModel="m", maxCost=1.0, maxRuntime=1)


# ---------------------------------------------------------------------
# Pricing — free / local models must be $0
# ---------------------------------------------------------------------

@pytest.mark.parametrize("model", ["qwen3-coder:30b", "llama3.1:8b", "qwen2.5-coder:32b"])
def test_local_models_price_zero(model):
    p = _get_pricing(model)
    assert p == {"input": 0.0, "output": 0.0, "cache_write": 0.0, "cache_read": 0.0}


def test_unknown_paid_model_keeps_protective_fallback():
    # A genuinely-unknown *paid* cloud id must NOT drop to $0 (keeps the cap
    # meaningful for BYOK runs on a new model we don't have a rate for yet).
    assert _get_pricing("some-new-cloud-model")["input"] > 0


# ---------------------------------------------------------------------
# create_llm_client('free') — server-injected, fails fast when unset
# ---------------------------------------------------------------------

def test_free_client_unconfigured_raises(monkeypatch):
    monkeypatch.delenv("BESSER_FREE_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("BESSER_FREE_LLM_MODEL", raising=False)
    assert free_tier_available() is False
    with pytest.raises(ValueError, match="free tier is not available"):
        create_llm_client(provider="free")


def test_free_client_configured_injects_endpoint_and_header(monkeypatch):
    monkeypatch.setenv("BESSER_FREE_LLM_BASE_URL", "https://ollama.example/v1")
    monkeypatch.setenv("BESSER_FREE_LLM_TOKEN", "secret-token")
    monkeypatch.setenv("BESSER_FREE_LLM_MODEL", "qwen3-coder:30b")

    assert free_tier_available() is True
    assert free_tier_model() == "qwen3-coder:30b"

    client = create_llm_client(provider="free", api_key=None, model=None, base_url=None)
    # endpoint + model come from env, not the (ignored) call args
    assert str(client._client.base_url).rstrip("/") == "https://ollama.example/v1"
    assert client._model == "qwen3-coder:30b"
    # the bearer token is the real gate — it must reach the SDK client
    assert client._client.default_headers.get("Authorization") == "Bearer secret-token"


def test_free_client_ignores_client_supplied_model_and_url(monkeypatch):
    # A client cannot repoint the free tier at another host/model.
    monkeypatch.setenv("BESSER_FREE_LLM_BASE_URL", "https://ollama.example/v1")
    monkeypatch.setenv("BESSER_FREE_LLM_MODEL", "qwen3-coder:30b")
    monkeypatch.delenv("BESSER_FREE_LLM_TOKEN", raising=False)

    client = create_llm_client(
        provider="free", api_key="attacker", model="gpt-4o",
        base_url="https://evil.example/v1",
    )
    assert str(client._client.base_url).rstrip("/") == "https://ollama.example/v1"
    assert client._model == "qwen3-coder:30b"
