"""A judge that verdicts nothing is a failed check, not N failed requirements.

``judge_coverage`` fills in "no verdict returned" for any requirement the
judge skipped — correct when it judged most and missed a few. When the call
comes back with nothing usable, that same fill-in turns one failure into one
blocker per requirement.

Live run lsrnaime (Qwen, 2026-09-19): 104 requirements extracted, 104 verdicts
of ``unverified`` / "no verdict returned", and a recipe reporting **115
blockers** — burying the eleven real ones. The caller already has a single
honest finding for a judge that returns nothing; it just never fired, because
a full list of placeholders is not ``None``.
"""

import pytest

from besser.generators.llm import requirements_ledger as ledger


REQUIREMENTS = [
    {"id": 1, "text": "Room numbers are unique", "kind": "uniqueness"},
    {"id": 2, "text": "Guests never exceed capacity", "kind": "rule"},
    {"id": 3, "text": "A bill records its issue date", "kind": "validation"},
]


class _Client:
    model = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    def chat(self, **kwargs):  # pragma: no cover - patched per test
        raise AssertionError("unexpected call")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(ledger, "_is_real_provider", lambda _c: True)
    monkeypatch.setattr(ledger, "_JUDGE_RETRY_BACKOFF_SECONDS", 0)
    return _Client()


def _judge(monkeypatch, client, payload):
    monkeypatch.setattr(ledger, "_call_with_tool", lambda *a, **k: payload)
    return ledger.judge_coverage(REQUIREMENTS, "digest", client)


def test_no_usable_verdict_returns_none(monkeypatch, client):
    assert _judge(monkeypatch, client, {"verdicts": []}) is None


def test_unparseable_ids_also_count_as_no_verdict(monkeypatch, client):
    payload = {"verdicts": [{"id": "not-a-number", "status": "implemented"},
                            {"id": 99, "status": "implemented"}]}

    assert _judge(monkeypatch, client, payload) is None


def test_a_partial_judgement_is_kept_and_the_rest_filled_in(monkeypatch, client):
    """The fill-in is right when the judge actually judged something."""
    payload = {"verdicts": [{"id": 1, "status": "implemented", "evidence": "models.py: x"}]}

    verdicts = _judge(monkeypatch, client, payload)

    assert verdicts is not None
    assert len(verdicts) == 3
    by_id = {v["id"]: v for v in verdicts}
    assert by_id[1]["status"] == "implemented"
    assert by_id[2]["note"] == "no verdict returned"


def test_a_verification_only_ledger_still_returns_its_gate(monkeypatch, client):
    """An extraction-completeness item is not a judged requirement."""
    monkeypatch.setattr(ledger, "_call_with_tool", lambda *a, **k: {"verdicts": []})

    verdicts = ledger.judge_coverage(
        [{"id": 1, "text": ledger._EXTRACTION_INCOMPLETE, "kind": "verification"}],
        "digest", client,
    )

    assert verdicts is not None
    assert verdicts[0]["status"] == "unverified"
