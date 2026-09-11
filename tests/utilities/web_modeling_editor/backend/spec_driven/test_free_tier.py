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
    free_alt_choice,
    free_alt_models,
    free_fallback_model,
    free_tier_available,
    free_tier_model,
    is_free_fallback_choice,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.sse_events import (
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


# ---------------------------------------------------------------------
# Explicit free-model choice — allowlist is exactly {primary, fallback}
# ---------------------------------------------------------------------


def _configure_free_tier_with_fallback(monkeypatch):
    monkeypatch.setenv("BESSER_FREE_LLM_BASE_URL", "https://cloud.example/v1")
    monkeypatch.setenv("BESSER_FREE_LLM_TOKEN", "primary-token")
    monkeypatch.setenv("BESSER_FREE_LLM_MODEL", "meituan/LongCat-2.0:free")
    monkeypatch.setenv("BESSER_FREE_LLM_FALLBACK_BASE_URL", "https://ollama.example/v1")
    monkeypatch.setenv("BESSER_FREE_LLM_FALLBACK_TOKEN", "fallback-token")
    monkeypatch.setenv("BESSER_FREE_LLM_FALLBACK_MODEL", "qwen3.8:27b")
    # Keep the two-model baseline deterministic even if the host running the
    # tests has alt models configured.
    monkeypatch.delenv("BESSER_FREE_LLM_ALT_MODELS", raising=False)


def test_is_free_fallback_choice_matrix(monkeypatch):
    _configure_free_tier_with_fallback(monkeypatch)
    assert free_fallback_model() == "qwen3.8:27b"
    assert is_free_fallback_choice("qwen3.8:27b") is True
    # Everything else — empty, primary, arbitrary — is NOT a fallback choice.
    assert is_free_fallback_choice(None) is False
    assert is_free_fallback_choice("") is False
    assert is_free_fallback_choice("meituan/LongCat-2.0:free") is False
    assert is_free_fallback_choice("gpt-4o") is False


def test_is_free_fallback_choice_false_without_fallback(monkeypatch):
    _configure_free_tier_with_fallback(monkeypatch)
    monkeypatch.delenv("BESSER_FREE_LLM_FALLBACK_BASE_URL", raising=False)
    assert free_fallback_model() == ""
    assert is_free_fallback_choice("qwen3.8:27b") is False


def test_free_client_explicit_fallback_model_uses_fallback_endpoint(monkeypatch):
    # A user who explicitly picks the self-hosted fallback model gets a
    # client built directly against the fallback endpoint...
    _configure_free_tier_with_fallback(monkeypatch)
    client = create_llm_client(provider="free", model="qwen3.8:27b")
    assert str(client._client.base_url).rstrip("/") == "https://ollama.example/v1"
    assert client._model == "qwen3.8:27b"
    assert client._client.default_headers.get("Authorization") == "Bearer fallback-token"
    # ...with NO outage fallback of its own: if the self-hosted box is down
    # the run fails with an honest error instead of silently switching the
    # user to the cloud model they opted out of.
    assert client._fallback is None


def test_free_client_arbitrary_model_pins_primary(monkeypatch):
    # Any id outside {primary, fallback} is ignored: primary endpoint,
    # primary model, normal outage-fallback chain. The server's free-tier
    # credentials can never be steered to an arbitrary model.
    _configure_free_tier_with_fallback(monkeypatch)
    client = create_llm_client(provider="free", model="gpt-4o")
    assert str(client._client.base_url).rstrip("/") == "https://cloud.example/v1"
    assert client._model == "meituan/LongCat-2.0:free"
    assert client._client.default_headers.get("Authorization") == "Bearer primary-token"
    assert client._fallback == (
        "https://ollama.example/v1", "fallback-token", "qwen3.8:27b",
    )


def test_free_client_explicit_primary_behaves_like_default(monkeypatch):
    # Requesting the primary explicitly is identical to sending no model:
    # primary endpoint + the normal fallback chain stays armed.
    _configure_free_tier_with_fallback(monkeypatch)
    client = create_llm_client(provider="free", model="meituan/LongCat-2.0:free")
    assert str(client._client.base_url).rstrip("/") == "https://cloud.example/v1"
    assert client._model == "meituan/LongCat-2.0:free"
    assert client._fallback == (
        "https://ollama.example/v1", "fallback-token", "qwen3.8:27b",
    )


# ---------------------------------------------------------------------
# Alt free models — extra ids on the PRIMARY endpoint
#
# The provider meters each free model separately, so an alt such as
# poolside/laguna-s-2.1-free (no stated daily quota) is the only keyless option
# left once the primary's 100/day is spent. The hazard these tests pin down is
# the endpoint/token PAIRING: an alt must ride the primary's base URL + token,
# never the fallback's (different host, different credentials).
# ---------------------------------------------------------------------

_LAGUNA = "poolside/laguna-s-2.1-free"


def _configure_free_tier_with_alt(monkeypatch, alts=_LAGUNA):
    _configure_free_tier_with_fallback(monkeypatch)
    monkeypatch.setenv("BESSER_FREE_LLM_ALT_MODELS", alts)


def test_free_alt_models_empty_by_default(monkeypatch):
    # A deploy that never sets the var behaves exactly as before.
    _configure_free_tier_with_fallback(monkeypatch)
    assert free_alt_models() == []
    assert free_alt_choice(_LAGUNA) == ""


def test_free_alt_models_parses_list(monkeypatch):
    _configure_free_tier_with_alt(
        monkeypatch, f" {_LAGUNA} , inclusionai/ling-3.0-flash-sante:free ,,"
    )
    assert free_alt_models() == [_LAGUNA, "inclusionai/ling-3.0-flash-sante:free"]


def test_free_alt_models_drops_primary_and_fallback_ids(monkeypatch):
    # Both ids are already owned by a specific endpoint+token pair; re-serving
    # them as alts would send the primary's bearer to the fallback host.
    _configure_free_tier_with_alt(
        monkeypatch, f"meituan/LongCat-2.0:free,qwen3.8:27b,{_LAGUNA},{_LAGUNA}"
    )
    assert free_alt_models() == [_LAGUNA]
    # The fallback id still routes to the fallback, not to the primary.
    assert is_free_fallback_choice("qwen3.8:27b") is True
    assert free_alt_choice("qwen3.8:27b") == ""


def test_free_alt_choice_matrix(monkeypatch):
    _configure_free_tier_with_alt(monkeypatch)
    assert free_alt_choice(_LAGUNA) == _LAGUNA
    assert free_alt_choice(f"  {_LAGUNA}  ") == _LAGUNA
    assert free_alt_choice(None) == ""
    assert free_alt_choice("") == ""
    assert free_alt_choice("meituan/LongCat-2.0:free") == ""
    assert free_alt_choice("gpt-4o") == ""
    # An alt is never mistaken for the fallback choice.
    assert is_free_fallback_choice(_LAGUNA) is False


def test_free_client_alt_model_uses_primary_endpoint_and_token(monkeypatch):
    # The pairing assertion: alt model, PRIMARY base URL, PRIMARY bearer.
    _configure_free_tier_with_alt(monkeypatch)
    client = create_llm_client(provider="free", model=_LAGUNA)
    assert str(client._client.base_url).rstrip("/") == "https://cloud.example/v1"
    assert client._model == _LAGUNA
    assert client._client.default_headers.get("Authorization") == "Bearer primary-token"
    # Same shared cloud endpoint as the primary, so it keeps the same outage
    # fallback (unlike an explicit fallback choice, which has none).
    assert client._fallback == (
        "https://ollama.example/v1", "fallback-token", "qwen3.8:27b",
    )


def test_free_client_alt_model_not_offered_pins_primary(monkeypatch):
    # An id that is a valid alt on ANOTHER deploy is still just an unknown id
    # here: it must pin to the primary, never be forwarded blindly.
    _configure_free_tier_with_fallback(monkeypatch)
    client = create_llm_client(provider="free", model=_LAGUNA)
    assert client._model == "meituan/LongCat-2.0:free"


def test_free_client_alt_model_still_fails_fast_when_unconfigured(monkeypatch):
    # Naming an alt must not bypass the "free tier not available" guard.
    monkeypatch.delenv("BESSER_FREE_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("BESSER_FREE_LLM_MODEL", raising=False)
    monkeypatch.setenv("BESSER_FREE_LLM_ALT_MODELS", _LAGUNA)
    with pytest.raises(ValueError, match="free tier is not available"):
        create_llm_client(provider="free", model=_LAGUNA)
