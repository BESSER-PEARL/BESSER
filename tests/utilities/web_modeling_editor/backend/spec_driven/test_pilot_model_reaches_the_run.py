"""A study session's model must reach the RUN, not just the dropdown.

When only the client resolved `BESSER_PILOT_LLM_MODEL`, it never took effect:
the free tier is the no-popup default, so a session that never opens the model
dialog sends no `llm_model` at all, and one that accepts the pre-selected model
had that choice collapsed to "use the server default" on save. Both paths land
on the public model, so the session default is resolved server-side.
"""

from __future__ import annotations

import pytest

from besser.utilities.web_modeling_editor.backend.models.spec_driven import (
    SmartGenerateRequest,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SmartGenerationRunner,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_spec_driven_router import (
    _build_project_body,
)

PILOT_MODEL = "gpt-5.6-luna"
PUBLIC_DEFAULT = "meituan/LongCat-2.0:free"


@pytest.fixture(autouse=True)
def _free_tier(monkeypatch):
    monkeypatch.setenv("BESSER_FREE_LLM_BASE_URL", "https://cloud.example/v1")
    monkeypatch.setenv("BESSER_FREE_LLM_MODEL", PUBLIC_DEFAULT)
    monkeypatch.setenv("BESSER_FREE_LLM_ALT_MODELS", f"poolside/laguna-s-2.1-free,{PILOT_MODEL}")
    monkeypatch.setenv("BESSER_FREE_LLM_FALLBACK_BASE_URL", "https://ollama.example/v1")
    monkeypatch.setenv("BESSER_FREE_LLM_FALLBACK_MODEL", "qwen3-coder:30b")
    monkeypatch.setenv("BESSER_PILOT_LLM_MODEL", PILOT_MODEL)


def _requested(**overrides) -> str | None:
    """The model a run built from this request would actually ask for."""
    body = _build_project_body(provider="free")
    body.pop("api_key", None)
    # The free tier's normal wire shape: no key, no model. The shared fixture
    # ships a BYOK model, which is not what a free-tier request looks like.
    body["llm_model"] = None
    body.update(overrides)
    runner = SmartGenerationRunner(SmartGenerateRequest(**body), run_id="a" * 32)
    return runner._requested_llm_model()


# ======================================================================
# The regression this file exists for
# ======================================================================

def test_a_pilot_who_sends_no_model_still_gets_the_pilot_model():
    """No dialog opened, so no llm_model on the wire."""
    assert _requested(telemetry_participant="P11") == PILOT_MODEL


def test_a_non_pilot_who_sends_no_model_gets_the_public_default():
    """Everyone else must be untouched — None means the factory's default."""
    assert _requested() is None


# ======================================================================
# An explicit choice always wins over the session default
# ======================================================================

@pytest.mark.parametrize("chosen", [
    PUBLIC_DEFAULT,                 # deliberately picked the public model
    "poolside/laguna-s-2.1-free",   # a different alt
    "qwen3-coder:30b",              # the self-hosted fallback
])
def test_an_explicit_choice_beats_the_pilot_default(chosen):
    assert _requested(telemetry_participant="P11", llm_model=chosen) == chosen


# ======================================================================
# Scope: only the free tier, only a labelled study session
# ======================================================================

def test_an_unconfigured_pilot_model_changes_nothing(monkeypatch):
    monkeypatch.delenv("BESSER_PILOT_LLM_MODEL", raising=False)
    assert _requested(telemetry_participant="P11") is None


def test_a_pilot_model_the_server_does_not_offer_is_ignored(monkeypatch):
    """`free_pilot_model()` refuses an id outside the allowlist, and the run
    must fall back to the default rather than pin to something unserved."""
    monkeypatch.setenv("BESSER_PILOT_LLM_MODEL", "claude-sonnet-4-6")
    assert _requested(telemetry_participant="P11") is None


def test_a_byok_pilot_is_not_pushed_onto_a_free_model():
    """A session that pasted its own key runs on THEIR provider."""
    body = _build_project_body(
        provider="anthropic", api_key="sk-ant-test", llm_model=None,
        telemetry_participant="P11",
    )
    runner = SmartGenerationRunner(SmartGenerateRequest(**body), run_id="b" * 32)
    assert runner._requested_llm_model() is None


def test_an_invalid_participant_label_is_not_a_pilot():
    """The validator nulls a label that fails the collection pattern, and a
    nulled label must not silently grant the session model."""
    assert _requested(telemetry_participant="not a valid label!!") is None
