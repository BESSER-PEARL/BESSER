"""A required gate that cannot decide refuses completion, and says so.

Review 2026-10-30: the runtime gate returned on any exception with a debug log
(B8, S3; ``probe_gate_failopen.py`` injected a failing backend walk and got no
finding), and backend discovery turned its own failure into "no backend here",
which also silences the gate. The selector failure (S1) and the sanitizer
drops that remove planned work (S2) changed the run's outcome below warning.
"""

from __future__ import annotations

import logging

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning.gap_analyzer import _sanitize_tasks
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.validation.issues import is_completion_issue


class _Client:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self) -> None:
        self.usage = UsageTracker("mock-model")


@pytest.fixture
def orch(tmp_path):
    (tmp_path / "main_api.py").write_text("raise SystemExit('cannot boot')\n", encoding="utf-8")
    (tmp_path / "sql_alchemy.py").write_text("", encoding="utf-8")
    return LLMOrchestrator(
        llm_client=_Client(), domain_model=DomainModel(name="M", types={Class(name="Item")}),
        output_dir=str(tmp_path), enable_checkpointing=False, enable_tracing=False,
    )


def _walk_fails():
    raise OSError("walk failed")


def test_runtime_gate_exception_is_not_a_pass(orch, caplog):
    orch._validation_issues = []
    orch._runtime_probe_verdict = None
    orch._probeable_backends = _walk_fails

    with caplog.at_level(logging.WARNING):
        orch._apply_runtime_gate()
        orch._apply_runtime_gate()

    assert len(orch._validation_issues) == 1
    issue = orch._validation_issues[0]
    assert is_completion_issue(issue)
    assert issue.message.startswith("validation unverified: the runtime gate")
    assert "OSError: walk failed" in issue.message
    assert any("Runtime gate evaluation failed" in r.message for r in caplog.records)


def test_backend_discovery_failure_is_not_no_backend(orch, monkeypatch):
    import besser.spec_driven_agent.validation.constructibility as constructibility
    monkeypatch.setattr(constructibility, "_fastapi_backends", lambda _d: _walk_fails())

    with pytest.raises(OSError):
        orch._probeable_backends()


def test_selector_exception_warns(orch, monkeypatch, caplog):
    import besser.spec_driven_agent.agent.tools as tools
    monkeypatch.setattr(tools, "get_available_generator_names", _walk_fails)

    with caplog.at_level(logging.WARNING):
        assert orch._select_generator_with_llm("Build a library app") is None

    assert any("keyword routing" in r.message for r in caplog.records if r.levelno == logging.WARNING)


@pytest.mark.parametrize("task", [
    "Delete the generated React frontend scaffold",
    "Install Flask and move the routes to it",
])
def test_sanitizer_drops_warn(task, caplog):
    with caplog.at_level(logging.WARNING):
        assert _sanitize_tasks([task], "generate_web_app") == []

    assert any(r.levelno == logging.WARNING and "Gap sanitizer dropped" in r.message
               for r in caplog.records)
