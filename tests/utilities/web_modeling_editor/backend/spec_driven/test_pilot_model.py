"""A facilitated pilot session may default to a stronger keyless model.

Pilots arrive through `?pilot=<label>` and are a small, known population we
deliberately spend more on. `BESSER_PILOT_LLM_MODEL` lets the server point
them at a better model without changing what anonymous visitors get.

Server-side on purpose: which model is "the good one" changed three times in a
single afternoon, so it must be an env edit plus a container restart, never a
frontend release.
"""
import pytest

from besser.spec_driven_agent.providers.llm_client import free_pilot_model


@pytest.fixture(autouse=True)
def _free_tier(monkeypatch):
    monkeypatch.setenv("BESSER_FREE_LLM_BASE_URL", "https://cloud.example/v1")
    monkeypatch.setenv("BESSER_FREE_LLM_MODEL", "meituan/LongCat-2.0:free")
    monkeypatch.setenv("BESSER_FREE_LLM_ALT_MODELS", "poolside/laguna-s-2.1-free,gpt-5.6-luna")
    monkeypatch.setenv("BESSER_FREE_LLM_FALLBACK_BASE_URL", "https://ollama.example/v1")
    monkeypatch.setenv("BESSER_FREE_LLM_FALLBACK_MODEL", "qwen3-coder:30b")


def test_unset_means_pilots_get_the_ordinary_default(monkeypatch):
    monkeypatch.delenv("BESSER_PILOT_LLM_MODEL", raising=False)
    assert free_pilot_model() == ""


@pytest.mark.parametrize("model", [
    "gpt-5.6-luna",                 # an alt model
    "meituan/LongCat-2.0:free",     # the primary
    "qwen3-coder:30b",              # the fallback
])
def test_any_offered_model_may_be_the_pilot_default(model, monkeypatch):
    monkeypatch.setenv("BESSER_PILOT_LLM_MODEL", model)
    assert free_pilot_model() == model


def test_a_model_the_server_does_not_offer_is_refused(monkeypatch):
    """Advertising a default we would refuse to honour is worse than none.

    The client sends the pre-selected id as `llm_model`; an unknown id is
    pinned back to the default server-side, so the pilot would silently get the
    public model while the UI showed something else.
    """
    monkeypatch.setenv("BESSER_PILOT_LLM_MODEL", "claude-sonnet-4-6")
    assert free_pilot_model() == ""


def test_whitespace_is_not_a_model(monkeypatch):
    monkeypatch.setenv("BESSER_PILOT_LLM_MODEL", "   ")
    assert free_pilot_model() == ""


def test_the_config_endpoint_advertises_it(monkeypatch):
    monkeypatch.setenv("BESSER_PILOT_LLM_MODEL", "gpt-5.6-luna")
    from tests.utilities.web_modeling_editor.backend.spec_driven.test_concurrency_and_config import (
        _get_config,
    )
    free_tier = _get_config().json()["free_tier"]
    assert free_tier["pilot_model"] == "gpt-5.6-luna"
    assert "gpt-5.6-luna" in [m["id"] for m in free_tier["models"]], (
        "the pilot model must also be choosable, or the client cannot select it"
    )


def test_the_config_reports_null_when_unset(monkeypatch):
    monkeypatch.delenv("BESSER_PILOT_LLM_MODEL", raising=False)
    from tests.utilities.web_modeling_editor.backend.spec_driven.test_concurrency_and_config import (
        _get_config,
    )
    assert _get_config().json()["free_tier"]["pilot_model"] is None
