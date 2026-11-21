"""A server no check can start is said in the run's headline, not left silent.

Only the FastAPI backends the probe drives are ever started. A run that built
another kind of server (a Django app, an Express API written from scratch) had
no runtime class at all, so its headline read exactly like a probed run's: an app
that crashes on boot ended "completed" with nothing saying its runtime was never
checked. Such a run now carries a ``runtime`` class with a fixed reason, marked
like a check that is off (no edit can give it evidence, so it never makes the run
"not verified"), and its ``stop_message`` says the runtime could not be checked.
"""

from __future__ import annotations

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning.target import BACKEND, OTHER, Target
from besser.spec_driven_agent.validation.issues import RUNTIME_NOT_CHECKABLE
from besser.utilities.web_modeling_editor.backend.services.spec_driven import run_record
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    build_verification_report,
    run_summary_markdown,
)
from tests.spec_driven_agent.test_not_verified_state import _Client, _stop

NOT_CHECKABLE = ("runtime", RUNTIME_NOT_CHECKABLE, True)


@pytest.fixture
def orch(simple_library_book_model, tmp_path):
    orchestrator = LLMOrchestrator(
        llm_client=_Client(), domain_model=simple_library_book_model,
        output_dir=str(tmp_path), enable_checkpointing=False, enable_tracing=False,
    )
    orchestrator._validation_issues = []
    return orchestrator


def _write(root, files: dict) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_a_generated_server_no_check_starts_gets_a_runtime_note(orch, tmp_path):
    _write(tmp_path, {"django/manage.py": "import sys\n", "django/library/views.py": "def index(request):\n    pass\n"})
    orch._generator_used = "generate_django"

    gaps = orch.verification_gaps()

    assert gaps == [NOT_CHECKABLE]
    facts = _stop(gaps)
    assert facts.outcome == "completed"  # no edit can settle it: never "not verified"
    assert facts.stop_message == f"The app was generated, but its runtime could not be checked: {RUNTIME_NOT_CHECKABLE}."
    section = run_record.run_section(facts, object(), provider="free", models=[], request_chars=0)
    summary = "\n".join(run_summary_markdown({"run": section}, build_verification_report({})))
    assert f"- Not checked: the app's runtime (`{RUNTIME_NOT_CHECKABLE}`)" in summary
    assert "Checks off on this server" not in summary


def test_a_backend_target_written_from_scratch_gets_the_note(orch, tmp_path):
    _write(tmp_path, {"server/package.json": '{"name": "api"}', "server/index.js": "require('express')\n"})
    orch.target = Target(kind=BACKEND, description="REST API", has_user_interface=False)
    assert NOT_CHECKABLE in orch.verification_gaps()


def test_a_run_that_built_no_server_has_no_runtime_class(orch, tmp_path):
    _write(tmp_path, {"cli/main.py": "print('hi')\n"})
    orch.target = Target(kind=OTHER, description="Python CLI", has_user_interface=False)
    assert orch.verification_gaps() == []


def test_a_backend_the_probe_drives_is_judged_by_the_probe(orch, monkeypatch):
    monkeypatch.setattr(orch, "_probeable_backends", lambda: ["web_app/backend"])
    orch._generator_used = "generate_web_app"
    assert all(gap[1] != RUNTIME_NOT_CHECKABLE for gap in orch.verification_gaps())
