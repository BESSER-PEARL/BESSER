"""Tests for the Nebius Token Factory provider: endpoint pinning, model
defaults, pricing, planning-model routing, and factory wiring.

Nebius Token Factory (formerly Nebius AI Studio) is OpenAI-compatible, so the
provider is a thin subclass of ``OpenAIProvider``. These tests assert the parts
that are NOT inherited — the ones a future refactor could silently break:

* the endpoint is pinned to Nebius and never leaks to/from ``OPENAI_BASE_URL``;
* ``planning_model`` does not inherit OpenAI's ``gpt-4o-mini``, which Nebius
  does not serve;
* the vendor-namespaced model id is priced from the Nebius row rather than
  being mistaken for a $0 self-hosted model or billed at the ``gpt-4o``
  fallback;
* the OpenAI request-shaping helpers make the right call for a Nebius id.
"""

from unittest.mock import MagicMock, patch

import pytest

from besser.spec_driven_agent.providers.llm_client import (
    DEFAULT_MODELS,
    LLMProvider,
    NebiusProvider,
    _get_pricing,
    _is_free_local_model,
    _needs_reasoning_none_for_tools,
    _openai_max_tokens_key,
    _resolve_nebius_api_key,
    create_llm_client,
)
from besser.spec_driven_agent.errors import InvalidApiKeyError

NEBIUS_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"


# ======================================================================
# Endpoint pinning
# ======================================================================

class TestNebiusEndpoint:

    def test_default_base_url_is_the_token_factory_endpoint(self):
        # Confirmed against Nebius's own docs (docs.tokenfactory.nebius.com);
        # docs.nebius.com/studio/* now 307-redirects there.
        assert NebiusProvider.DEFAULT_BASE_URL == NEBIUS_BASE_URL

    def test_client_is_pointed_at_nebius(self, monkeypatch):
        monkeypatch.delenv("NEBIUS_BASE_URL", raising=False)
        provider = NebiusProvider(api_key="nebius-test")
        assert provider._base_url == NEBIUS_BASE_URL

    def test_openai_base_url_env_does_not_leak_into_a_nebius_client(self, monkeypatch):
        # The inherited OpenAIProvider constructor falls back to
        # OPENAI_BASE_URL when it gets no base_url. NebiusProvider must
        # resolve its own default FIRST, or a user with a gateway configured
        # for OpenAI would silently send their Nebius key somewhere else.
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
        monkeypatch.delenv("NEBIUS_BASE_URL", raising=False)
        provider = NebiusProvider(api_key="nebius-test")
        assert provider._base_url == NEBIUS_BASE_URL

    def test_server_env_may_repoint_the_endpoint(self, monkeypatch):
        # Server env only — never request data. Mirrors MISTRAL_BASE_URL.
        monkeypatch.setenv("NEBIUS_BASE_URL", "https://nebius.internal/v1")
        provider = NebiusProvider(api_key="nebius-test")
        assert provider._base_url == "https://nebius.internal/v1"

    def test_factory_ignores_a_request_supplied_base_url(self):
        # SSRF guard: create_llm_client must not forward a caller's base_url
        # to the Nebius client. The runner's BESSER_LLM_ALLOW_CUSTOM_BASE_URL
        # gate is the first line of defence; this is the second.
        client = create_llm_client(
            provider="nebius",
            api_key="nebius-test",
            base_url="http://169.254.169.254/latest/meta-data/",
        )
        assert client._base_url == NEBIUS_BASE_URL


# ======================================================================
# Model defaults and planning-model routing
# ======================================================================

class TestNebiusModelDefaults:

    def test_default_model_is_the_requested_qwen_endpoint(self):
        assert NebiusProvider.DEFAULT_MODEL == NEBIUS_MODEL

    def test_registered_in_default_models(self):
        # The runner and GET /spec-driven/config both read this map, so a
        # missing entry means a run with no explicit llm_model sends "".
        assert DEFAULT_MODELS["nebius"] == NEBIUS_MODEL

    def test_planning_model_is_the_primary_not_gpt_4o_mini(self):
        # Inheriting OpenAIProvider.PLANNING_MODEL would send an id Nebius
        # does not serve, costing two failing round-trips per gap analysis.
        provider = NebiusProvider(api_key="nebius-test")
        assert provider.planning_model is None

    def test_planning_model_env_override_is_honoured(self, monkeypatch):
        monkeypatch.setenv("BESSER_LLM_PLANNING_MODEL", "Qwen/Qwen3-4B-Instruct")
        provider = NebiusProvider(api_key="nebius-test")
        assert provider.planning_model == "Qwen/Qwen3-4B-Instruct"

    def test_planning_model_env_primary_disables_routing(self, monkeypatch):
        monkeypatch.setenv("BESSER_LLM_PLANNING_MODEL", "primary")
        provider = NebiusProvider(api_key="nebius-test")
        assert provider.planning_model is None

    def test_explicit_model_overrides_the_default(self):
        provider = NebiusProvider(api_key="nebius-test", model="Qwen/Qwen3-32B")
        assert provider.model == "Qwen/Qwen3-32B"


# ======================================================================
# Pricing
# ======================================================================

class TestNebiusPricing:

    def test_namespaced_qwen_id_is_not_mistaken_for_a_free_local_model(self):
        # A vendor-namespaced id is served by a gateway, so somebody is being
        # billed. Pricing it at $0 would silently disable max_cost_usd.
        assert _is_free_local_model(NEBIUS_MODEL.lower()) is False

    def test_target_model_uses_the_nebius_row(self):
        p = _get_pricing(NEBIUS_MODEL)
        # Console endpoint properties.
        assert p["input"] == 0.1
        assert p["output"] == 0.3

    def test_target_model_does_not_fall_back_to_gpt_4o(self):
        # Without the Nebius row this id lands on the protective gpt-4o
        # default ($2.50/$10) and aborts a cheap run at a fraction of the
        # budget the user actually authorised.
        assert _get_pricing(NEBIUS_MODEL)["input"] != 2.5

    def test_pricing_is_case_insensitive(self):
        assert _get_pricing(NEBIUS_MODEL.lower()) == _get_pricing(NEBIUS_MODEL)

    def test_self_hosted_qwen_tag_is_still_free(self):
        # The Nebius row must not broaden into the free tier's Ollama tags.
        assert _get_pricing("qwen3-coder:30b")["input"] == 0.0


# ======================================================================
# OpenAI request shaping for a Nebius model id
# ======================================================================

class TestNebiusRequestShaping:

    def test_uses_plain_max_tokens(self):
        # Nebius ids contain none of gpt-5 / o1 / o3 / o4, so the
        # max_completion_tokens branch must not fire.
        assert _openai_max_tokens_key(NEBIUS_MODEL) == "max_tokens"

    def test_never_sends_reasoning_effort(self):
        # reasoning_effort='none' is an api.openai.com-only workaround; a
        # gateway validates it against its own enum and 400s.
        assert _needs_reasoning_none_for_tools(NEBIUS_MODEL, NEBIUS_BASE_URL) is False


# ======================================================================
# API key resolution
# ======================================================================

class TestNebiusApiKeyResolution:

    def test_explicit_key_wins(self):
        assert _resolve_nebius_api_key(api_key="explicit") == "explicit"

    def test_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("NEBIUS_API_KEY", "from-env")
        assert _resolve_nebius_api_key() == "from-env"

    def test_raises_with_an_actionable_message_when_absent(self, monkeypatch):
        monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
        with pytest.raises(InvalidApiKeyError, match="NEBIUS_API_KEY"):
            _resolve_nebius_api_key()

    def test_does_not_borrow_the_openai_key(self, monkeypatch):
        # Cross-provider key bleed would send the user's OpenAI key to Nebius.
        monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
        with pytest.raises(InvalidApiKeyError):
            _resolve_nebius_api_key()


# ======================================================================
# Factory wiring
# ======================================================================

class TestCreateNebiusClient:

    @patch("besser.spec_driven_agent.providers.llm_client.NebiusProvider")
    def test_factory_builds_a_nebius_provider(self, MockNebius):
        MockNebius.return_value = MagicMock(spec=LLMProvider)
        create_llm_client(provider="nebius", api_key="nebius-test")
        MockNebius.assert_called_once()

    def test_factory_returns_a_real_llm_provider(self):
        client = create_llm_client(provider="nebius", api_key="nebius-test")
        assert isinstance(client, NebiusProvider)
        assert isinstance(client, LLMProvider)
        assert client.model == NEBIUS_MODEL

    def test_factory_requires_a_key(self, monkeypatch):
        monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
        with pytest.raises(InvalidApiKeyError):
            create_llm_client(provider="nebius")

    def test_unknown_provider_message_lists_nebius(self):
        with pytest.raises(ValueError, match="nebius"):
            create_llm_client(provider="cohere", api_key="x")

    def test_usage_tracker_is_seeded_with_the_nebius_rate(self):
        client = create_llm_client(provider="nebius", api_key="nebius-test")
        assert client.usage.pricing["input"] == 0.1


# ======================================================================
# Request model / SSE schema accept the provider
# ======================================================================

class TestNebiusAcceptedByTheWebContract:

    def test_start_event_accepts_nebius(self):
        from besser.utilities.web_modeling_editor.backend.services.spec_driven.sse_events import (
            StartEvent,
        )
        ev = StartEvent(
            runId="abc", provider="nebius", llmModel=NEBIUS_MODEL,
            maxCost=1.0, maxRuntime=600,
        )
        assert ev.provider == "nebius"

    def test_model_id_passes_the_request_validator(self):
        # The request-model regex must allow the vendor/model slash form.
        from besser.utilities.web_modeling_editor.backend.models.spec_driven import (
            _LLM_MODEL_NAME_RE,
        )
        assert _LLM_MODEL_NAME_RE.match(NEBIUS_MODEL)


class TestNebiusContextWindow:
    """The console reports 262K context for Qwen3-30B-A3B-Instruct-2507.

    ``_SMALL_CONTEXT_WINDOWS`` carries a measured ``("qwen3", 60_000)`` row for
    a self-hosted Ollama endpoint. It matches by substring, so before the
    namespace guard the Nebius model was clamped to that endpoint's window and compacted constantly.
    """

    def test_namespaced_nebius_id_is_not_clamped_to_the_ollama_window(self):
        from besser.spec_driven_agent.agent.compaction import effective_threshold
        nebius = effective_threshold("Qwen/Qwen3-30B-A3B-Instruct-2507")
        ollama = effective_threshold("qwen3-coder:30b")
        assert nebius > ollama, (
            "a 262k cloud endpoint must not inherit the self-hosted 60k window")

    def test_self_hosted_qwen3_still_uses_the_measured_window(self):
        """The row protects the production free tier — it must keep working."""
        from besser.spec_driven_agent.agent.compaction import (
            COMPACT_RESERVE_TOKENS, COMPACT_TOKEN_THRESHOLD, effective_threshold)
        got = effective_threshold("qwen3-coder:30b")
        assert got <= min(COMPACT_TOKEN_THRESHOLD, 60_000 - COMPACT_RESERVE_TOKENS)

    def test_other_self_hosted_rows_are_unaffected(self):
        from besser.spec_driven_agent.agent.compaction import effective_threshold
        for bare in ("devstral:latest", "mistral-small", "mistral-7b"):
            assert effective_threshold(bare) <= 32_000

    def test_namespaced_mistral_small_is_not_clamped_either(self):
        """The guard is about self-hosted vs cloud, not about qwen."""
        from besser.spec_driven_agent.agent.compaction import effective_threshold
        assert effective_threshold("mistralai/Mistral-Small-Instruct") >             effective_threshold("mistral-small")
