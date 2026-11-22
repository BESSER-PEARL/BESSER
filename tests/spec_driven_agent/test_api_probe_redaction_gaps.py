"""test_api reports hide every credential and still let a secret-only record be confirmed.

Review probes on PR 788: ``session_id``, ``reset_code``, a numeric ``otp`` and the
values inside ``tokens: {access, refresh}`` reached the report in clear; and a record
whose only field besides ``id`` is credential-named could never confirm its create,
because the read-back was compared after both values had become ``[REDACTED]``.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from besser.spec_driven_agent.validation import api_probe

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

_REDACTED = "[REDACTED]"


def _run(app, requests):
    return asyncio.run(api_probe._run_requests(app, requests))


def test_session_ids_reset_codes_numbers_and_nested_tokens_are_hidden():
    shown = api_probe._scrub({
        "session_id": "s-9f8e7d6c5b4a3928", "sessionId": "s-1a2b3c4d5e6f7a8b", "reset_code": "RC-4471",
        "verification_code": "VC-9921", "otp": 482913,
        "tokens": {"access": "opaque-access-1", "refresh": "opaque-refresh-2", "token_type": "bearer",
                   "expires_in": 3600},
        "api_keys": ["k-123456789", "k-987654321"],
    }, ())
    assert shown == {
        "session_id": _REDACTED, "sessionId": _REDACTED, "reset_code": _REDACTED, "verification_code": _REDACTED,
        "otp": _REDACTED,
        "tokens": {"access": _REDACTED, "refresh": _REDACTED, "token_type": "bearer", "expires_in": 3600},
        "api_keys": [_REDACTED, _REDACTED],
    }


def test_ordinary_ids_numbers_and_flags_stay_readable():
    record = {"id": 7, "user_id": 3, "order_id": "ord-12", "session_id": 4, "sessionCount": 2,
              "max_tokens": 5, "tokens": 1500, "password_set": True, "otp": None, "code": 200,
              "country_code": "LU", "usage": {"tokens": 12}}
    assert api_probe._scrub(record, ()) == record


@pytest.mark.parametrize("name", ["réinitialisationToken", "überPassword", "clé_secret", "motDePasseÜberPassword"])
def test_a_credential_word_after_an_accented_one_is_hidden(name):
    """Review probe, round 6: ``überPassword`` read as one word ``überpassword`` and was shown."""
    assert api_probe._scrub({name: "opaque-9f8e7d6c5b4a"}, ()) == {name: _REDACTED}


@pytest.mark.parametrize("name, value", [
    ("password_attempts", 3), ("password_min_length", 8), ("otp_attempts", 2), ("password_max_length", 64),
    ("token_count", 4), ("password_changed_at", 1700000000),
])
def test_a_number_that_only_describes_a_credential_is_shown(name, value):
    """Review probe, round 6: once numbers could be redacted, counts and limits of a credential were hidden."""
    assert api_probe._scrub({name: value}, ()) == {name: value}


def test_a_reference_into_nested_tokens_is_hidden_in_the_shown_path():
    from fastapi import FastAPI

    app = FastAPI()

    @app.post("/auth/login")
    def login():
        return {"tokens": {"access": "opaque-access-0123456789"}}

    @app.get("/auth/check/{token}")
    def check(token: str):
        return {"ok": True}

    report = _run(app, [{"method": "POST", "path": "/auth/login"},
                        {"method": "GET", "path": "/auth/check/{{0.tokens.access}}"}])
    assert report["status"] == "passed", report
    assert "opaque-access-0123456789" not in json.dumps(report)


def _vault_app(read_secret):
    from fastapi import FastAPI

    app = FastAPI()

    @app.post("/vault/")
    def create():
        return {"id": 1, "secretKey": "vk-123456789"}

    @app.get("/vault/{item_id}")
    def read(item_id: int):
        return {"id": item_id, "secretKey": read_secret}

    return app


def test_a_record_holding_only_a_secret_is_confirmed_when_the_read_back_matches():
    report = _run(_vault_app("vk-123456789"), [{"method": "POST", "path": "/vault/"},
                                                {"method": "GET", "path": "/vault/{{0.id}}"}])
    assert report["responses"][1]["json"] == {"id": 1, "secretKey": _REDACTED}
    assert "vk-123456789" not in json.dumps(report)
    assert api_probe.confirmed_create_paths(report) == {"/vault"}


def test_a_read_back_with_another_secret_is_no_evidence():
    report = _run(_vault_app("vk-000000000"), [{"method": "POST", "path": "/vault/"},
                                                {"method": "GET", "path": "/vault/{{0.id}}"}])
    assert report["responses"][1]["json"] == {"id": 1, "secretKey": _REDACTED}
    assert api_probe.confirmed_create_paths(report) == set()
