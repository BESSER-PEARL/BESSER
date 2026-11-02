"""The model catalog: providers, models and gateway rates are deployment config.

Before it, every new model, price change or gateway meant code edits in the
key dialog, ``llm_client._ROUTE_PRICING`` and the agent, then a release; and
the internal PIA gateway host and its rates were public in the repo and the
frontend bundle. The tests use a fictional gateway: a deployment's own
catalog file is the only place a real one belongs.
"""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.spec_driven_agent.providers import model_catalog
from besser.spec_driven_agent.providers.llm_client import (
    OpenAIProvider,
    _get_pricing,
    create_llm_client,
)
from besser.spec_driven_agent.providers.model_catalog import (
    ModelCatalogError,
    load_model_catalog,
)

GATEWAY_URL = "https://gateway.example.internal/v1"
ONE_MILLION_EACH = SimpleNamespace(
    input_tokens=1_000_000, output_tokens=1_000_000,
    cache_creation_input_tokens=0, cache_read_input_tokens=0,
)

GATEWAY_CATALOG = {
    "providers": {
        "pia": {
            "label": "Example gateway",
            "hint": "Your gateway key.",
            "key_prefix": "sk-",
            "base_url": GATEWAY_URL,
            "models": [
                {"id": "claude-sonnet-5-5", "label": "Sonnet 5.5", "default": True,
                 "price": {"input": 2.5, "output": 12.0}},
                {"id": "gpt-6-luna", "label": "Luna",
                 "price": {"input": 0.3, "output": 0.7, "cache_read": 0.03}},
            ],
        },
        "openai": {
            "label": "OpenAI (GPT)",
            "models": [
                {"id": "gpt-9-nova", "label": "GPT-9 Nova",
                 "price": {"input": 7.0, "output": 21.0}},
            ],
        },
        "mistral": None,
    },
}


@pytest.fixture(autouse=True)
def _fresh_catalog():
    load_model_catalog.cache_clear()
    yield
    load_model_catalog.cache_clear()


@pytest.fixture
def use_catalog(tmp_path, monkeypatch):
    def _use(data) -> None:
        path = tmp_path / "catalog.json"
        path.write_text(json.dumps(data) if not isinstance(data, str) else data,
                        encoding="utf-8")
        monkeypatch.setenv(model_catalog.CATALOG_ENV_VAR, str(path))
        load_model_catalog.cache_clear()
    return _use


# --- loading and merging ---------------------------------------------------

def test_the_public_default_holds_no_gateway():
    catalog = load_model_catalog()
    assert {"anthropic", "openai", "mistral", "nebius", "local"} <= set(catalog.providers)
    assert "pia" not in catalog.providers
    assert all(entry["base_url"] is None for entry in catalog.providers.values())


def test_an_override_adds_replaces_and_removes_providers(use_catalog):
    use_catalog(GATEWAY_CATALOG)
    catalog = load_model_catalog()
    assert catalog.base_url("pia") == GATEWAY_URL
    assert [m["id"] for m in catalog.providers["openai"]["models"]] == ["gpt-9-nova"]
    assert "mistral" not in catalog.providers
    # Untouched providers keep the default entry.
    assert catalog.providers["anthropic"]["models"]
    assert catalog.default_model("pia") == "claude-sonnet-5-5"


def test_replace_drops_the_default_catalog(use_catalog):
    use_catalog({"replace": True, "providers": {"local": {"label": "Local", "models": []}}})
    assert list(load_model_catalog().providers) == ["local"]


@pytest.mark.parametrize("bad, message", [
    ("{not json", "not valid JSON"),
    ({"providers": {"pia": {"models": []}}}, r"providers\.pia\.label: Field required"),
    ({"providers": {"pia": {"label": "P", "base_url": "gateway", "models": []}}},
     r"base_url must be an http\(s\) URL"),
    ({"providers": {"x": {"label": "X", "models": [{"id": "m", "price": {"input": 1}}]}}},
     r"models\[0\]\.price\.output"),
    ({"providers": {"x": {"label": "X", "models": [{"id": "m"}, {"id": "M"}]}}},
     "duplicate id"),
    # Review 2026-10-30 (R8, model_catalog_validation.py): a typo key loaded
    # silently as a provider with no gateway, a URL with no host crashed with
    # AttributeError, and "true" (a string) loaded as not-the-default.
    ({"providers": {"pia": {"label": "P", "base_ur1": "https://gw.example/v1", "models": []}}},
     r"providers\.pia\.base_ur1: Extra inputs are not permitted"),
    ({"providers": {"pia": {"label": "P", "base_url": "https:///v1",
                            "models": [{"id": "m", "price": {"input": 1, "output": 2}}]}}},
     "with a host"),
    ({"providers": {"x": {"label": "X", "models": [{"id": "m", "default": "true"}]}}},
     r"models\[0\]\.default: Input should be a valid boolean"),
    ({"providers": {"x": {"label": "X", "models": [{"id": "m", "price": {"input": True, "output": 1}}]}}},
     r"price\.input"),
    ({"route_prices": {"gw.example": {"m": {"input": 1, "outptu": 2}}}},
     r"route_prices\.gw\.example\.m\.output: Field required"),
])
def test_a_malformed_override_fails_loudly(use_catalog, bad, message):
    use_catalog(bad)
    with pytest.raises(ModelCatalogError, match=message):
        load_model_catalog()


def test_a_missing_override_file_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setenv(model_catalog.CATALOG_ENV_VAR, str(tmp_path / "absent.json"))
    with pytest.raises(ModelCatalogError, match="cannot be read"):
        load_model_catalog()


def test_the_public_list_carries_neither_endpoint_nor_rates(use_catalog):
    use_catalog(GATEWAY_CATALOG)
    payload = json.dumps(load_model_catalog().public_providers())
    assert "gateway.example.internal" not in payload
    assert "price" not in payload


# --- pricing ---------------------------------------------------------------

@pytest.mark.parametrize("model, expected", [
    ("claude-sonnet-5-5", 2.5 + 12.0),
    ("gpt-6-luna", 0.3 + 0.7),
])
def test_gateway_runs_are_priced_at_the_catalog_rates(use_catalog, model, expected):
    use_catalog(GATEWAY_CATALOG)
    client = create_llm_client(provider="pia", api_key="sk-test", model=model)
    client.usage.record(ONE_MILLION_EACH)
    assert client.usage.estimated_cost == pytest.approx(expected)


def test_a_missing_cache_rate_bills_at_the_input_rate(use_catalog):
    use_catalog(GATEWAY_CATALOG)
    rate = _get_pricing("claude-sonnet-5-5", route=GATEWAY_URL)
    assert rate["cache_read"] == rate["cache_write"] == 2.5
    assert _get_pricing("gpt-6-luna", route=GATEWAY_URL)["cache_read"] == 0.03


def test_the_same_id_off_the_gateway_keeps_its_vendor_price(use_catalog):
    use_catalog(GATEWAY_CATALOG)
    assert _get_pricing("gpt-6-luna", billed=True)["input"] == 0.1


def test_a_catalog_price_on_a_vendor_model_beats_the_fallback(use_catalog):
    assert _get_pricing("gpt-9-nova", billed=True)["input"] == 2.5  # gpt-4o fallback
    use_catalog(GATEWAY_CATALOG)
    assert _get_pricing("gpt-9-nova", billed=True) == {
        "input": 7.0, "output": 21.0, "cache_read": 7.0, "cache_write": 7.0,
    }


def test_route_prices_from_the_default_still_price_the_sponsored_gateway():
    rate = _get_pricing("moonshotai/Kimi-K3", route="https://api.commandcode.ai/provider/v1")
    assert (rate["input"], rate["cache_read"]) == (3.0, 0.30)


# --- the gateway provider ---------------------------------------------------

def test_the_gateway_endpoint_comes_from_the_catalog_not_the_request(use_catalog):
    use_catalog(GATEWAY_CATALOG)
    client = create_llm_client(
        provider="pia", api_key="sk-test", base_url="http://169.254.169.254/",
    )
    assert isinstance(client, OpenAIProvider)
    assert client._base_url == GATEWAY_URL
    assert client._model == "claude-sonnet-5-5"


def test_a_server_without_the_gateway_rejects_it():
    with pytest.raises(ValueError, match="Unknown provider: 'pia'"):
        create_llm_client(provider="pia", api_key="sk-test")


# --- GET /spec-driven/config -----------------------------------------------

def _get_config() -> dict:
    from besser.utilities.web_modeling_editor.backend.backend import app

    async def _go() -> httpx.Response:
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
            return await ac.get("/besser_api/spec-driven/config")

    response = asyncio.run(_go())
    assert response.status_code == 200
    return response.json()


def test_config_lists_the_default_providers():
    providers = {p["id"]: p for p in _get_config()["providers"]}
    assert "pia" not in providers
    assert {"id": "gpt-6-sol", "label": "GPT-6 Sol — balanced (recommended)", "default": True} in (
        providers["openai"]["models"])


def test_a_model_added_only_in_the_override_reaches_the_picker(use_catalog):
    use_catalog(GATEWAY_CATALOG)
    providers = {p["id"]: p for p in _get_config()["providers"]}
    assert [m["id"] for m in providers["openai"]["models"]] == ["gpt-9-nova"]
    assert providers["pia"]["label"] == "Example gateway"
    assert providers["pia"]["key_prefix"] == "sk-"
    assert "base_url" not in providers["pia"]


def test_a_gateway_run_announces_its_provider():
    from besser.utilities.web_modeling_editor.backend.services.spec_driven.sse_events import (
        StartEvent,
    )
    event = StartEvent(runId="r", provider="pia", llmModel="m", maxCost=1.0, maxRuntime=60)
    assert event.provider == "pia"


def test_a_gateway_request_drops_a_stray_base_url():
    """Its endpoint is the catalog's; a request URL must not trip the SSRF gate."""
    from tests.utilities.web_modeling_editor.backend.spec_driven.test_runner import (
        _build_request,
    )
    request = _build_request(provider="pia", base_url="http://169.254.169.254/")
    assert request.provider == "pia"
    assert request.base_url is None


# --- the provider's default model -------------------------------------------

def test_the_default_catalog_sets_current_defaults():
    from besser.spec_driven_agent.providers.llm_client import default_model
    assert default_model("anthropic") == "claude-sonnet-5-5"
    assert default_model("openai") == "gpt-6-sol"


def test_without_a_flag_the_built_in_default_holds(use_catalog):
    from besser.spec_driven_agent.providers.llm_client import DEFAULT_MODELS, default_model
    use_catalog({"replace": True, "providers": {"openai": {"label": "OpenAI", "models": [
        {"id": "gpt-6-luna"}]}}})
    assert default_model("openai") == DEFAULT_MODELS["openai"]
    assert default_model("anthropic") == DEFAULT_MODELS["anthropic"]


def test_a_flagged_model_is_the_default_a_run_and_the_config_use(use_catalog):
    use_catalog({"providers": {"openai": {"label": "OpenAI", "models": [
        {"id": "gpt-6-luna", "label": "Luna"},
        {"id": "gpt-9-nova", "label": "Nova", "default": True},
    ]}}})
    client = create_llm_client(provider="openai", api_key="sk-test")
    assert client._model == "gpt-9-nova"
    assert create_llm_client(provider="openai", api_key="sk-test", model="gpt-6-luna")._model == "gpt-6-luna"
    assert _get_config()["default_models"]["openai"] == "gpt-9-nova"


# --- request quirks from the catalog ----------------------------------------

def test_a_new_model_gets_its_request_quirks_from_the_catalog(use_catalog):
    """A model with the Sonnet 5.5 quirks, before any code row knows it."""
    from besser.spec_driven_agent.providers.model_settings import (
        forced_tool_choice_supported,
        reasoning_effort_for_tools,
        thinking_can_be_disabled,
    )
    assert forced_tool_choice_supported("claude-sonnet-6") is True
    use_catalog({"providers": {
        "anthropic": {"label": "Anthropic", "models": [
            {"id": "claude-sonnet-6", "request": {
                "forced_tool_choice": False, "thinking_can_be_disabled": False}},
        ]},
        "openai": {"label": "OpenAI", "models": [
            {"id": "gpt-7-sol", "request": {"reasoning_effort_for_tools": "none"}},
        ]},
    }})
    assert forced_tool_choice_supported("claude-sonnet-6") is False
    assert thinking_can_be_disabled("claude-sonnet-6") is False
    assert reasoning_effort_for_tools("gpt-7-sol") == "none"
    # Models the catalog does not mention keep their built-in rows.
    assert forced_tool_choice_supported("claude-opus-5-5") is False
    assert forced_tool_choice_supported("claude-sonnet-5") is True


def test_the_catalog_can_correct_a_built_in_row(use_catalog):
    from besser.spec_driven_agent.providers.model_settings import reasoning_effort_for_tools
    assert reasoning_effort_for_tools("gpt-6-sol") == "none"
    use_catalog({"providers": {"openai": {"label": "OpenAI", "models": [
        {"id": "gpt-6-sol", "request": {"reasoning_effort_for_tools": None}},
    ]}}})
    assert reasoning_effort_for_tools("gpt-6-sol") is None


@pytest.mark.parametrize("request_block, message", [
    ({"forced_tools": False}, r"request\.forced_tools: Extra inputs are not permitted"),
    ({"forced_tool_choice": "no"}, r"request\.forced_tool_choice: Input should be a valid boolean"),
    ({"forced_tool_choice": None}, r"request\.forced_tool_choice: Input should be a valid boolean"),
])
def test_a_malformed_request_block_fails_loudly(use_catalog, request_block, message):
    use_catalog({"providers": {"x": {"label": "X", "models": [{"id": "m", "request": request_block}]}}})
    with pytest.raises(ModelCatalogError, match=message):
        load_model_catalog()


def test_notes_are_ignored_at_any_depth(use_catalog):
    use_catalog({"providers": {"x": {"_note": "ops", "label": "X", "models": [
        {"id": "m", "_why": "cheap", "price": {"input": 1, "output": 2, "_source": "vendor"}}]}}})
    assert load_model_catalog().providers["x"]["models"][0]["price"]["cache_read"] == 1.0
