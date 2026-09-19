"""A bounded retry for a judge call that returns nothing usable.

An adversarial review of 17 live runs measured the judge call - never the
extraction call, which shares ``_call_with_tool`` and the same client but
sends a prompt two orders of magnitude smaller - returning nothing usable
in 3 of them: fcdh0s9k, n_6i2i5r, ys4gfj4v. All three ran on Nebius/Qwen3-
30B, whose ``planning_model`` is ``None`` by design ("no cheap sibling"),
and every failure completed in under a second - too fast to be genuine
inference over a prompt that can carry a 200k-char digest, and consistent
with a raised exception or an empty completion rather than a slow response
that got cut off. Before this fix, one such failure was final: nothing
retried it, and the run paid the call's full cost for zero signal. These
tests exercise the real ``_call_with_tool`` / ``_tool_input`` path (not a
mocked stand-in for it), so they prove the retry integrates with the
existing exception handling and planning/primary model routing, not just
that a loop exists.
"""

import pytest

from besser.generators.llm import requirements_ledger as ledger
from besser.generators.llm.llm_client import UsageTracker

REQUIREMENTS = [
    {"id": 1, "text": "Room numbers are unique", "kind": "uniqueness"},
    {"id": 2, "text": "Guests never exceed capacity", "kind": "rule"},
]

_GOOD_PAYLOAD = {"verdicts": [
    {"id": 1, "status": "implemented", "evidence": "room.py: unique=True"},
    {"id": 2, "status": "missing", "note": "no capacity check found"},
]}


class _ScriptedClient:
    """A real provider (``_client`` is set) whose ``chat`` plays back one
    scripted outcome per call: ``"raise"`` (the call fails), ``"empty"``
    (a successful response with no tool call - the model answered in
    prose), or a tool-payload dict."""

    model = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    def __init__(self, script, planning_model=None):
        self._client = object()
        self.usage = UsageTracker("mock-model")
        self.planning_model = planning_model
        self._script = list(script)
        self.calls: list = []  # model_override used on each raw call

    def chat(self, system, messages, tools, force_tool=None, model_override=None):
        self.calls.append(model_override)
        step = self._script.pop(0)
        if step == "raise":
            raise RuntimeError("simulated transient provider error")
        if step == "empty":
            text_block = type("T", (), {"type": "text", "text": "I cannot help with that."})()
            return {"stop_reason": "end_turn", "content": [text_block]}
        block = type("B", (), {"type": "tool_use", "name": force_tool, "input": step})()
        return {"stop_reason": "tool_use", "content": [block]}


@pytest.fixture(autouse=True)
def no_real_backoff(monkeypatch):
    """The retry backs off between attempts; tests assert behaviour, not
    wall-clock time. Recording (not zeroing) the constant also lets a test
    confirm the backoff actually fired."""
    sleeps = []
    monkeypatch.setattr(ledger.time, "sleep", lambda seconds: sleeps.append(seconds))
    return sleeps


def test_a_single_failed_call_now_gets_one_bounded_retry_and_recovers(no_real_backoff):
    """Reproduces the live shape of fcdh0s9k / n_6i2i5r / ys4gfj4v: a
    provider with no planning-model sibling (Nebius) and a single failed
    judge call. Pre-fix: ``_call_with_tool`` catches the exception, sees
    ``planning_model`` is already ``None`` and returns ``None`` straight
    away - one raw call, zero signal, exactly what the live runs show.
    Post-fix: one bounded retry recovers the verdicts."""
    client = _ScriptedClient(["raise", _GOOD_PAYLOAD], planning_model=None)

    verdicts = ledger.judge_coverage(REQUIREMENTS, "digest", client)

    assert verdicts is not None
    assert [v["status"] for v in verdicts] == ["implemented", "missing"]
    # Two genuine calls, not one call repeated as a no-op: on a provider
    # with no cheap sibling, the "model switch" is unavailable, but the
    # retry must still happen.
    assert client.calls == [None, None]
    assert no_real_backoff == [ledger._JUDGE_RETRY_BACKOFF_SECONDS]


def test_an_empty_tool_call_is_retried_too_not_just_an_exception():
    """``_call_with_tool``'s own planning/primary fallback only re-fires on
    a raised exception (see its ``except Exception`` block). A response
    that comes back 200-OK with prose but no ``tool_use`` block - the
    model answered instead of calling the tool - is a different failure
    shape that the pre-fix code accepted as final. It must be retried
    exactly like an exception."""
    client = _ScriptedClient(["empty", _GOOD_PAYLOAD])

    verdicts = ledger.judge_coverage(REQUIREMENTS, "digest", client)

    assert verdicts is not None
    assert len(client.calls) == 2


def test_the_retry_is_bounded_and_the_silent_failure_stays_loud():
    """Only two calls are ever made - a third would ``IndexError`` on this
    two-step script, proving the retry cannot run away - and the existing
    single honest finding (``None``, not one blocker per requirement) must
    still be what callers see once the retry is exhausted."""
    client = _ScriptedClient(["raise", "raise"])

    assert ledger.judge_coverage(REQUIREMENTS, "digest", client) is None
    assert len(client.calls) == ledger._JUDGE_MAX_ATTEMPTS == 2


def test_a_successful_first_attempt_is_never_retried():
    """A working call must not silently double its cost. Only one payload
    is scripted here; a second call would ``IndexError``."""
    client = _ScriptedClient([_GOOD_PAYLOAD])

    verdicts = ledger.judge_coverage(REQUIREMENTS, "digest", client)

    assert verdicts is not None
    assert len(client.calls) == 1


def test_retry_forces_the_primary_model_off_the_planning_sibling():
    """Where a distinct planning model exists (Anthropic/OpenAI/Mistral,
    unlike the Nebius runs in the live evidence), attempt 1 tries it and
    the retry goes straight to the primary model instead of repeating the
    identical cheap-tier call that just came back empty."""
    client = _ScriptedClient(["empty", _GOOD_PAYLOAD], planning_model="cheap-model")

    verdicts = ledger.judge_coverage(REQUIREMENTS, "digest", client)

    assert verdicts is not None
    assert client.calls == ["cheap-model", None]
