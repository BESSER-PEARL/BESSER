"""The keyless-tier fallback is an ordered CHAIN, cloud first, local last.

Before this, the free tier had exactly one fallback: when LongCat's
100-requests/day quota ran out, every run dropped straight onto our
self-hosted Tesla V100 — a box that serves one request at a time and pays
60-75s to load a model cold. Measured 2026-09-11, that produced 8-minute
turns and apps that never finished inside the runtime cap.

A cloud alternative with its own quota (Laguna, which documents none) should
be tried BEFORE the shared GPU. Each step carries its own endpoint and token,
so a model id can never be sent with another endpoint's bearer.
"""

import pytest

from besser.generators.llm import llm_client as lc
from besser.generators.llm.llm_client import (
    OpenAIProvider,
    _resolve_free_fallback_chain,
)

PRIMARY_URL = "https://api.commandcode.ai/provider/v1"
LOCAL_URL = "https://ollama.besser-pearl.org/v1"
PRIMARY = "meituan/LongCat-2.0:free"
ALT = "poolside/laguna-s-2.1-free"
LOCAL = "qwen3-coder:30b"


@pytest.fixture
def free_env(monkeypatch):
    """A fully configured free tier: primary + one cloud alt + the local box."""
    monkeypatch.setenv("BESSER_FREE_LLM_BASE_URL", PRIMARY_URL)
    monkeypatch.setenv("BESSER_FREE_LLM_MODEL", PRIMARY)
    monkeypatch.setenv("BESSER_FREE_LLM_TOKEN", "primary-token")
    monkeypatch.setenv("BESSER_FREE_LLM_ALT_MODELS", ALT)
    monkeypatch.setenv("BESSER_FREE_LLM_FALLBACK_BASE_URL", LOCAL_URL)
    monkeypatch.setenv("BESSER_FREE_LLM_FALLBACK_MODEL", LOCAL)
    monkeypatch.setenv("BESSER_FREE_LLM_FALLBACK_TOKEN", "local-token")


# ---------------------------------------------------------------- ordering


def test_cloud_alt_comes_before_the_self_hosted_box(free_env):
    """The whole point: try another cloud model before the shared GPU."""
    chain = _resolve_free_fallback_chain(PRIMARY)
    assert [step[2] for step in chain] == [ALT, LOCAL]


def test_each_step_keeps_its_own_endpoint_and_token(free_env):
    """Pairing hazard: the local model's id must never ride the cloud bearer."""
    chain = _resolve_free_fallback_chain(PRIMARY)
    by_model = {step[2]: (step[0], step[1]) for step in chain}
    assert by_model[ALT] == (PRIMARY_URL, "primary-token")
    assert by_model[LOCAL] == (LOCAL_URL, "local-token")


def test_the_chosen_model_is_never_in_its_own_chain(free_env):
    """A run pinned to a model must not 'fall back' to that same model."""
    assert ALT not in [s[2] for s in _resolve_free_fallback_chain(ALT)]
    assert LOCAL not in [s[2] for s in _resolve_free_fallback_chain(LOCAL)]


def test_chain_is_just_the_box_when_no_alts_are_configured(free_env, monkeypatch):
    monkeypatch.delenv("BESSER_FREE_LLM_ALT_MODELS")
    assert [s[2] for s in _resolve_free_fallback_chain(PRIMARY)] == [LOCAL]


def test_chain_is_empty_when_nothing_is_configured(monkeypatch):
    for var in ("BESSER_FREE_LLM_BASE_URL", "BESSER_FREE_LLM_MODEL",
                "BESSER_FREE_LLM_TOKEN", "BESSER_FREE_LLM_ALT_MODELS",
                "BESSER_FREE_LLM_FALLBACK_BASE_URL",
                "BESSER_FREE_LLM_FALLBACK_MODEL"):
        monkeypatch.delenv(var, raising=False)
    assert _resolve_free_fallback_chain(PRIMARY) == []


def test_an_unconfigured_primary_does_not_hide_the_local_box(monkeypatch):
    """The builder must not blow up when the primary is unset — the local box
    is still a usable fallback on an on-prem deploy."""
    for var in ("BESSER_FREE_LLM_BASE_URL", "BESSER_FREE_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("BESSER_FREE_LLM_FALLBACK_BASE_URL", LOCAL_URL)
    monkeypatch.setenv("BESSER_FREE_LLM_FALLBACK_MODEL", LOCAL)
    assert [s[2] for s in _resolve_free_fallback_chain(PRIMARY)] == [LOCAL]


# ------------------------------------------------------------- advancing


def _provider(fallback):
    return OpenAIProvider(
        api_key="free", model=PRIMARY, base_url=PRIMARY_URL, fallback=fallback,
    )


def test_advances_through_every_step_then_refuses(free_env):
    p = _provider(_resolve_free_fallback_chain(PRIMARY))
    err = RuntimeError("429 quota exhausted")

    assert p._activate_fallback(err) is True
    assert p.model == ALT
    assert p._activate_fallback(err) is True
    assert p.model == LOCAL
    # Chain exhausted: the caller must now raise rather than loop.
    assert p._activate_fallback(err) is False
    assert p.model == LOCAL


def test_a_single_tuple_still_works(free_env):
    """Back-compat: other callers (e.g. the sponsored tier) pass one tuple."""
    p = _provider((LOCAL_URL, "local-token", LOCAL))
    assert p._activate_fallback(RuntimeError("boom")) is True
    assert p.model == LOCAL
    assert p._activate_fallback(RuntimeError("boom")) is False


def test_no_fallback_configured_refuses_immediately(free_env):
    p = _provider(None)
    assert p._activate_fallback(RuntimeError("boom")) is False
    assert p.model == PRIMARY


def test_quota_exhaustion_is_reported_as_the_reason(free_env):
    """The UI says "free daily quota exhausted" off this field, so it must
    survive a multi-step chain."""
    p = _provider(_resolve_free_fallback_chain(PRIMARY))
    p._activate_fallback(RuntimeError("Error code: 429 - insufficient_quota"))
    assert p.fallback_reason == "quota_exhausted"
