"""``besser.utilities.secret_redaction`` is the one redaction module.

It replaced the agent trace's own patterns, which loaded the backend's
redactor lazily and set it to ``False`` (logged at debug) when that import
failed, the ``_redact`` helpers of ``llm_utils`` and the KG LLM service, and
the generation router's key-name-only ``sanitize_config``. The expected
outputs below are what those copies produced, plus what the last one missed.
"""
import logging
import subprocess
import sys

import pytest

from besser.utilities.secret_redaction import (
    redact_literal, redact_text, redact_trace_text, sanitize_config,
)

JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhbm4ifQ.c2lnbmF0dXJlLXZhbHVl"
KEY = "sk-proj-" + "A" * 40


@pytest.mark.parametrize("text, expected", [
    (f"python .besser_probe.py req GET /note -H 'Authorization: Bearer {JWT}'",
     "python .besser_probe.py req GET /note -H 'Authorization: Bearer [REDACTED]'"),
    (f'{{"access_token": "{JWT}", "token_type": "bearer"}}',
     '{"access_token": "[REDACTED]", "token_type": "bearer"}'),
    ('{\\"refresh_token\\": \\"opaque-refresh-value\\"}', '{\\"refresh_token\\": \\"[REDACTED]\\"}'),
    ("authorization=Basic dXNlcjpwYXNz", "authorization=Basic [REDACTED]"),
    (f"OPENAI={KEY}\nTerminated", "OPENAI=[REDACTED]\nTerminated"),
    ("postgresql://admin:S3cr3tPassw0rd@db.example.com:5432/app",
     "postgresql://admin:[REDACTED]@db.example.com:5432/app"),
    # Value rules only: a NAME-based .env assignment is not a trace secret.
    ("SECRET_KEY = 'not-a-provider-token'", "SECRET_KEY = 'not-a-provider-token'"),
    ("max_tokens = 4000", "max_tokens = 4000"),
])
def test_trace_redaction_matches_the_agents_former_copy(text, expected):
    assert redact_trace_text(text) == expected


@pytest.mark.parametrize("text, secret, expected", [
    (f"Incorrect API key provided: {KEY}", KEY, "Incorrect API key provided: ***"),
    (f"{KEY} and {KEY}", KEY, "*** and ***"),
    ("no key here", KEY, "no key here"),
    ("an empty key masks nothing", "", "an empty key masks nothing"),
])
def test_redact_literal_matches_the_llm_clients_former_copies(text, secret, expected):
    assert redact_literal(text, secret) == expected


def test_sanitize_config_masks_credential_named_keys_as_before():
    config = {
        "openaiApiKey": KEY, "X-Auth-Token": "t", "nested": {"llm": {"api_key": KEY, "model": "gpt-5"}},
        "items": [{"password": "pw"}, ("plain", {"session_id": 7})], "temperature": 0.2, "name": "Greeter",
    }

    assert sanitize_config(config) == {
        "openaiApiKey": "***", "X-Auth-Token": "***", "nested": {"llm": {"api_key": "***", "model": "gpt-5"}},
        "items": [{"password": "***"}, ("plain", {"session_id": "***"})], "temperature": 0.2, "name": "Greeter",
    }


def test_sanitize_config_redacts_credentials_under_ordinary_keys():
    # The review's repro: the key-name-only copy logged both values in clear.
    config = {
        "database_url": "postgresql://admin:S3cr3tPassw0rd@db.example.com:5432/app",
        "llm": {"provider": "openai", "value": KEY},
        "notes": [f"use {KEY} here"],
    }

    assert sanitize_config(config) == {
        "database_url": "postgresql://admin:[REDACTED]@db.example.com:5432/app",
        "llm": {"provider": "openai", "value": "[REDACTED]"},
        "notes": ["use [REDACTED] here"],
    }


@pytest.mark.parametrize("url, expected", [
    ("postgresql://admin:pw@db/x", "postgresql://admin:[REDACTED]@db/x"),
    ("redis://:S3cr3t@localhost:6379", "redis://:[REDACTED]@localhost:6379"),
    ("mysql://root:hunter2@10.0.0.5/app", "mysql://root:[REDACTED]@10.0.0.5/app"),
    # Code that builds a URL holds no password.
    ("postgresql://admin:${DB_PASSWORD}@db/x", "postgresql://admin:${DB_PASSWORD}@db/x"),
])
def test_sanitize_config_masks_a_local_hosts_url_password(url, expected):
    # Generated artifacts keep a local host's password (docker-compose sets the
    # same value on the database); a log line has no such need.
    assert sanitize_config({"database": url}) == {"database": expected}
    assert redact_text(url, env_style=False)[0] == url


def test_sanitize_config_leaves_ordinary_values_and_the_input_alone():
    config = {"url": "postgresql://db:5432/app", "retries": 3, "label": "task-list-item"}

    assert sanitize_config(config) == config
    assert sanitize_config(config) is not config


def test_the_agent_transform_endpoint_logs_its_config_redacted(caplog):
    # /transform-agent-model-json logged the raw config at DEBUG, OpenAI key included.
    from fastapi.testclient import TestClient
    from besser.utilities.web_modeling_editor.backend.backend import app

    caplog.set_level(logging.DEBUG)
    with TestClient(app, raise_server_exceptions=False) as client:
        client.post("/besser_api/transform-agent-model-json", json={
            "title": "a", "model": {"type": "AgentDiagram", "elements": {}, "relationships": {}},
            "config": {"openai_api_key": KEY, "language": "en"}})

    logged = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[Agent transform] config")]
    assert logged and '"openai_api_key": "***"' in logged[0]
    assert not [r for r in caplog.records if KEY in r.getMessage()]


def _run(code: str) -> str:
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def test_the_agents_trace_redacts_without_the_editor_backend():
    # The backend package made unimportable: the old lazy import fell back to
    # no redaction here, leaving the provider key in the trace.
    code = (
        "import sys; sys.modules['besser.utilities.web_modeling_editor'] = None\n"
        "from besser.spec_driven_agent.pipeline import orchestrator\n"
        f"print(orchestrator._trace_result_excerpt('key {KEY} done')['excerpt'])\n"
    )
    excerpt = _run(code)
    assert KEY not in excerpt
    assert "[REDACTED]" in excerpt


def test_the_module_needs_neither_the_agent_nor_the_backend():
    code = ("import sys, besser.utilities.secret_redaction; "
            "print(sorted(m for m in sys.modules if m.startswith(("
            "'besser.utilities.web_modeling_editor', 'besser.spec_driven_agent', 'besser.generators'))))")
    assert _run(code) == "[]"
