"""Tests for daily-quota fast-fail + served-model alias normalization.

Measured on the free tier (2026-09-10 campaign): Command Code's free LongCat
returned ``429 "You've used all 100 free LongCat 2.0 requests for today…"`` on
39/39 fallback activations. That quota cannot recover inside a retry window, yet
the provider retried 5x (0.5s -> 30s backoff) per call before falling back —
~15-40s of pure waiting per call. These tests pin:

* `_is_quota_exhausted` recognises a period-quota 429 but NOT a transient 429;
* `chat()` skips the backoff/retries on a quota 429 and goes straight to the
  fallback check, while a transient 429 still retries;
* `_activate_fallback` records WHY it fell back ("quota_exhausted" vs
  "unavailable") so the runner/UI can tell the user;
* `UsageTracker.set_served_model` ignores aggregator spelling churn
  ("meituan/LongCat-2.0:free" vs "LongCat-2.0") but still catches a real
  LongCat -> qwen switch.
"""

import pytest

from besser.generators.llm import llm_client
from besser.generators.llm.llm_client import (
    OpenAIProvider,
    UpstreamLLMError,
    UsageTracker,
    _is_quota_exhausted,
    _is_retryable,
    _normalize_model_id,
)

QUOTA_MSG = (
    "Error code: 429 - {'error': {'message': \"You've used all 100 free LongCat "
    "2.0 requests for today. Your quota resets at 2026-09-10T00:00:00.000Z. "
    "Run /model or -m to pick another model.\"}}"
)
TRANSIENT_429 = "Error code: 429 - {'error': {'message': 'Rate limit exceeded, retry shortly'}}"


class TestQuotaClassification:
    def test_daily_quota_429_is_quota_exhausted(self):
        e = Exception(QUOTA_MSG)
        assert _is_quota_exhausted(e) is True
        assert _is_retryable(e) is True  # still a 429 — fallback path stays eligible

    def test_transient_429_is_not_quota_exhausted(self):
        assert _is_quota_exhausted(Exception(TRANSIENT_429)) is False

    def test_5xx_is_not_quota_exhausted(self):
        assert _is_quota_exhausted(Exception("Error code: 503 - service unavailable")) is False

    def test_openai_insufficient_quota_is_quota_exhausted(self):
        assert _is_quota_exhausted(
            Exception("Error code: 429 - insufficient_quota: You exceeded your current quota")
        ) is True


class TestModelIdNormalization:
    def test_aliases_compare_equal(self):
        assert _normalize_model_id("meituan/LongCat-2.0:free") == _normalize_model_id("LongCat-2.0")
        assert _normalize_model_id("Qwen3.8:27B") == _normalize_model_id("qwen3.8:27b")

    def test_real_switch_compares_unequal(self):
        assert _normalize_model_id("meituan/LongCat-2.0:free") != _normalize_model_id("qwen3.8:27b")

    def test_served_model_ignores_spelling_churn_but_catches_real_switch(self):
        t = UsageTracker("meituan/LongCat-2.0:free")
        t.set_served_model("meituan/LongCat-2.0:free")
        t.set_served_model("LongCat-2.0")  # aggregator alias flip — NOT a switch
        assert t.served_model == "meituan/LongCat-2.0:free"
        t.set_served_model("qwen3.8:27b")  # real fallback — IS a switch
        assert t.served_model == "qwen3.8:27b"


def _provider_with_fallback():
    return OpenAIProvider(
        api_key="sk-test",
        model="meituan/LongCat-2.0:free",
        fallback=("http://127.0.0.1:9/v1", "tok", "qwen3.8:27b"),
    )


class _RaisingCompletions:
    def __init__(self, exc):
        self._exc = exc
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        raise self._exc


class _FakeClient:
    def __init__(self, exc):
        self.chat = type("C", (), {})()
        self.chat.completions = _RaisingCompletions(exc)


class TestChatFastFail:
    def test_quota_429_skips_retries_and_goes_to_fallback_check(self, monkeypatch):
        p = _provider_with_fallback()
        p._client = _FakeClient(Exception(QUOTA_MSG))
        sleeps = []
        monkeypatch.setattr(llm_client.time, "sleep", lambda s: sleeps.append(s))
        fallback_calls = []
        monkeypatch.setattr(
            p, "_activate_fallback", lambda err: (fallback_calls.append(str(err)), False)[1]
        )
        with pytest.raises(UpstreamLLMError):
            p.chat("sys", [{"role": "user", "content": "hi"}], tools=[])
        assert sleeps == []  # no backoff burned on a quota error
        assert p._client.chat.completions.calls == 1  # a single attempt
        assert len(fallback_calls) == 1 and "used all 100" in fallback_calls[0]

    def test_transient_429_still_retries(self, monkeypatch):
        p = _provider_with_fallback()
        p._client = _FakeClient(Exception(TRANSIENT_429))
        sleeps = []
        monkeypatch.setattr(llm_client.time, "sleep", lambda s: sleeps.append(s))
        monkeypatch.setattr(p, "_activate_fallback", lambda err: False)
        with pytest.raises(UpstreamLLMError):
            p.chat("sys", [{"role": "user", "content": "hi"}], tools=[])
        assert len(sleeps) == llm_client._MAX_RETRIES  # retried with backoff
        assert p._client.chat.completions.calls == llm_client._MAX_RETRIES + 1


class TestFallbackReason:
    @pytest.fixture(autouse=True)
    def _no_network_client(self, monkeypatch):
        import openai
        monkeypatch.setattr(openai, "OpenAI", lambda **kw: object())

    def test_quota_error_records_quota_exhausted(self):
        p = _provider_with_fallback()
        assert p.fallback_reason is None
        assert p._activate_fallback(Exception(QUOTA_MSG)) is True
        assert p._on_fallback is True
        assert p.fallback_reason == "quota_exhausted"
        assert p.model == "qwen3.8:27b"

    def test_outage_records_unavailable(self):
        p = _provider_with_fallback()
        assert p._activate_fallback(Exception("Error code: 503 - upstream unavailable")) is True
        assert p.fallback_reason == "unavailable"

    def test_second_activation_is_refused(self):
        p = _provider_with_fallback()
        assert p._activate_fallback(Exception(QUOTA_MSG)) is True
        assert p._activate_fallback(Exception(QUOTA_MSG)) is False  # sticky, once per run
