"""``besser.utilities.secret_redaction`` is the one redaction module.

It replaced the agent trace's own patterns, which loaded the backend's
redactor lazily and set it to ``False`` (logged at debug) when that import
failed, and the ``_redact`` helpers of ``llm_utils`` and the KG LLM service.
The expected outputs below are what those copies produced.
"""
import subprocess
import sys

import pytest

from besser.utilities.secret_redaction import mask_secret, redact_trace_text

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
def test_mask_secret_matches_the_llm_clients_former_copies(text, secret, expected):
    assert mask_secret(text, secret) == expected


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
