"""A second keyless cloud endpoint sits between the primary's alt models and
the self-hosted last resort.

Motivation (2026-11-20): a Mistral deployment (Mistral Medium / Small, which
make forced tool calls) was to be offered for free while the self-hosted Qwen
stays the global last resort. The free tier had room for one endpoint besides
the primary, so adding Mistral meant dropping Qwen.
"""

import pytest

from besser.spec_driven_agent.agent import compaction
from besser.spec_driven_agent.providers.llm_client import (
    _resolve_free_fallback_chain,
    create_llm_client,
    free_pilot_model,
    free_secondary_choice,
    free_secondary_models,
)

PRIMARY_URL = "https://gateway.example/v1"
SEC_URL = "https://mistral.example/v1"
LOCAL_URL = "https://ollama.example/v1"
PRIMARY = "z-ai/glm-5.3-flash"
ALT = "minimax/minimax-m3.1-flash"
MEDIUM = "mistral-medium-latest"
SMALL = "mistral-small-latest"
LOCAL = "qwen3-coder:30b"


@pytest.fixture
def free_env(monkeypatch):
    monkeypatch.setenv("BESSER_FREE_LLM_BASE_URL", PRIMARY_URL)
    monkeypatch.setenv("BESSER_FREE_LLM_MODEL", PRIMARY)
    monkeypatch.setenv("BESSER_FREE_LLM_TOKEN", "primary-token")
    monkeypatch.setenv("BESSER_FREE_LLM_ALT_MODELS", ALT)
    monkeypatch.setenv("BESSER_FREE_LLM_SECONDARY_BASE_URL", SEC_URL)
    monkeypatch.setenv("BESSER_FREE_LLM_SECONDARY_MODELS", f"{MEDIUM},{SMALL}")
    monkeypatch.setenv("BESSER_FREE_LLM_SECONDARY_TOKEN", "mistral-token")
    monkeypatch.setenv("BESSER_FREE_LLM_FALLBACK_BASE_URL", LOCAL_URL)
    monkeypatch.setenv("BESSER_FREE_LLM_FALLBACK_MODEL", LOCAL)
    monkeypatch.setenv("BESSER_FREE_LLM_FALLBACK_TOKEN", "local-token")


def test_chain_is_alts_then_secondary_then_the_self_hosted_box(free_env):
    chain = _resolve_free_fallback_chain(PRIMARY)
    assert [step[2] for step in chain] == [ALT, MEDIUM, SMALL, LOCAL]


def test_secondary_models_keep_their_own_endpoint_and_token(free_env):
    by_model = {s[2]: (s[0], s[1]) for s in _resolve_free_fallback_chain(PRIMARY)}
    assert by_model[MEDIUM] == (SEC_URL, "mistral-token")
    assert by_model[SMALL] == (SEC_URL, "mistral-token")
    assert by_model[LOCAL] == (LOCAL_URL, "local-token")


def test_choosing_a_secondary_model_builds_it_on_its_own_endpoint(free_env):
    client = create_llm_client(provider="free", model=MEDIUM)
    assert str(client._client.base_url).rstrip("/") == SEC_URL
    assert client._model == MEDIUM
    assert client._client.default_headers.get("Authorization") == "Bearer mistral-token"
    # Keeps the chain, without itself, and Qwen stays last.
    assert [s[2] for s in client._fallback_chain] == [ALT, SMALL, LOCAL]


def test_a_rejected_secondary_key_skips_to_the_self_hosted_box(free_env):
    """A dead Mistral key must not end the run: the remaining Mistral entries
    share that key, so the next stop is Qwen."""
    client = create_llm_client(provider="free")
    assert client._activate_fallback(RuntimeError("503"))  # -> alt
    assert client._activate_fallback(RuntimeError("503"))  # -> mistral medium
    assert client._model == MEDIUM
    assert client._activate_fallback(RuntimeError("401"), past_endpoint=SEC_URL)
    assert client._model == LOCAL
    assert str(client._client.base_url).rstrip("/") == LOCAL_URL


def test_secondary_ids_already_served_elsewhere_are_dropped(free_env, monkeypatch):
    monkeypatch.setenv("BESSER_FREE_LLM_SECONDARY_MODELS", f"{PRIMARY},{ALT},{LOCAL},{MEDIUM}")
    assert free_secondary_models() == [MEDIUM]


def test_unconfigured_secondary_offers_nothing(free_env, monkeypatch):
    monkeypatch.delenv("BESSER_FREE_LLM_SECONDARY_BASE_URL")
    assert free_secondary_models() == []
    assert free_secondary_choice(MEDIUM) == ""
    # A request naming it pins to the primary, as for any unknown id.
    assert create_llm_client(provider="free", model=MEDIUM)._model == PRIMARY
    assert [s[2] for s in _resolve_free_fallback_chain(PRIMARY)] == [ALT, LOCAL]


def test_a_secondary_model_can_be_the_study_default(free_env, monkeypatch):
    monkeypatch.setenv("BESSER_PILOT_LLM_MODEL", MEDIUM)
    assert free_pilot_model() == MEDIUM


def test_config_endpoint_lists_secondary_between_alts_and_fallback(free_env):
    from tests.utilities.web_modeling_editor.backend.spec_driven.test_concurrency_and_config import (
        _get_config,
    )

    models = _get_config().json()["free_tier"]["models"]
    assert [m["id"] for m in models] == [PRIMARY, ALT, MEDIUM, SMALL, LOCAL]
    assert [m["id"] for m in models if m["default"]] == [PRIMARY]


def test_context_window_read_from_mistral_field(monkeypatch):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"data": [{"id": "mistral-medium-latest", "max_context_length": 262144}]}'

    monkeypatch.setattr(compaction.urllib.request, "urlopen", lambda *a, **k: _Resp())
    assert compaction._fetch_models(SEC_URL, "t") == {"mistral-medium-latest": 262144}
