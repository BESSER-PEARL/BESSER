"""A credential sent back through a wrapper-resolved reference is still scrubbed.

Review of round 10: ``{{0.reset_token}}`` against ``{"data": {"reset_token": ...}}``
now resolves through ``data``, but ``_sent_credentials`` still looked the path up
without that fallback, so the token the request sent came back unredacted in a
failure message (``got 'token opaqueResetSecretValue12345 accepted'``).
"""
from __future__ import annotations

from besser.spec_driven_agent.validation import api_probe

_SECRET = "opaqueResetSecretValue12345"


def test_a_wrapper_resolved_credential_is_a_sent_credential():
    request = {"method": "GET", "path": "/auth/reset/{{0.reset_token}}"}
    previous = [{"data": {"reset_token": _SECRET}}]

    assert api_probe._sent_credentials(request, previous) == {_SECRET}


def test_the_flat_shape_is_unchanged():
    request = {"method": "GET", "path": "/auth/reset/{{0.reset_token}}"}
    assert api_probe._sent_credentials(request, [{"reset_token": _SECRET}]) == {_SECRET}


_APP = '''from fastapi import FastAPI
app = FastAPI()
SECRET = "opaqueResetSecretValue12345"


@app.post("/auth/forgot")
def forgot():
    return {"data": {"reset_token": SECRET}}


@app.post("/auth/forgot-flat")
def forgot_flat():
    return {"reset_token": SECRET}


@app.get("/auth/reset/{token}")
def reset(token: str):
    return {"detail": f"token {token} accepted"}
'''


def test_a_sent_credential_is_scrubbed_from_failures_and_echoed_bodies(tmp_path):
    import json

    import pytest

    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "main_api.py").write_text(_APP, encoding="utf-8")
    (backend / "sql_alchemy.py").write_text("", encoding="utf-8")
    for route in ("/auth/forgot", "/auth/forgot-flat"):
        report = api_probe.probe_api_scenario(str(tmp_path), [
            {"method": "POST", "path": route},
            {"method": "GET", "path": "/auth/reset/{{0.reset_token}}", "expected_fields": {"detail": "x"}},
        ])
        assert report["status"] == "failed", report
        assert _SECRET not in json.dumps(report), (route, report)
