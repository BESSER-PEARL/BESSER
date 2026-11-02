"""``besser.utilities.secret_redaction`` is the one redaction module.

It replaced the agent trace's own patterns, which loaded the backend's
redactor lazily and set it to ``False`` (logged at debug) when that import
failed, the ``_redact`` helpers of ``llm_utils`` and the KG LLM service, and
the generation router's key-name-only ``sanitize_config``. The expected
outputs below are what those copies produced, plus what the last one missed.
"""
import subprocess
import sys

import pytest

from besser.utilities.secret_redaction import redact_literal, redact_trace_text, sanitize_config

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


def test_sanitize_config_leaves_ordinary_values_and_the_input_alone():
    config = {"url": "postgresql://user:pw@db:5432/app", "retries": 3, "label": "task-list-item"}

    assert sanitize_config(config) == config
    assert sanitize_config(config) is not config


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
