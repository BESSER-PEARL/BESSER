"""The incident log is a host-mounted file kept across redeploys, so it must
never hold a credential quoted by an exception or a traceback."""

import json

from besser.utilities.web_modeling_editor.backend.services.spec_driven.incidents import record_incident

TOKEN = "ghp_0123456789abcdefghijklmnopqrstuvwxyz"


def test_incident_text_fields_are_redacted(tmp_path, monkeypatch):
    monkeypatch.setenv("BESSER_INCIDENT_LOG_DIR", str(tmp_path))

    record_incident(
        kind="stream_crash",
        message=f"401 Unauthorized for token {TOKEN}",
        traceback_text=f'Traceback (most recent call last):\n  client = Github("{TOKEN}")\nBadCredentials',
        instructions=f"push it with {TOKEN}",
    )

    raw = (tmp_path / "incidents.jsonl").read_text(encoding="utf-8")
    assert TOKEN not in raw
    entry = json.loads(raw)
    assert entry["message"] == "401 Unauthorized for token [REDACTED]"
    assert 'Github("[REDACTED]")' in entry["traceback"]
    assert entry["instructions"] == "push it with [REDACTED]"


def test_a_linked_run_trace_is_not_copied_into_the_incident_dir(tmp_path, monkeypatch):
    # The trace sits in the run workspace, where the run's shell can replace it
    # with a link to a worker file; the operator's dir must not receive its target.
    from besser.utilities.web_modeling_editor.backend.services.spec_driven.incidents import persist_run_trace

    incidents = tmp_path / "incidents"
    monkeypatch.setenv("BESSER_INCIDENT_LOG_DIR", str(incidents))
    secret = tmp_path / "environ"
    secret.write_text(f"BESSER_FREE_LLM_TOKEN={TOKEN}\n", encoding="utf-8")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    trace = workspace / ".besser_trace.jsonl"
    trace.symlink_to(secret)

    persist_run_trace("a" * 32, str(trace))

    copied = incidents / "traces" / f"{'a' * 32}.besser_trace.jsonl"
    assert not copied.exists() or TOKEN not in copied.read_text(encoding="utf-8")
