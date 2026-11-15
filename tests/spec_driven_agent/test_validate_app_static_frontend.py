"""validate_app reports the static frontend checks Phase 3 judges, once a frontend exists.

It ran only the execution checks, so a model got ``"verified": true`` in Phase 2 and
Phase 3 then reported a frontend blocker the model never saw. The added checks are
source-only (no build, no node).
"""
from unittest import mock

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning.target import Target
from tests.spec_driven_agent.shelf_app import GENERATOR, generate_backend, shelf_model

MISSING_IMPORT = "imports './pages/BookList'"


def _orch(tmp_path, frontend=True):
    generate_backend(tmp_path, shelf_model())
    if frontend:
        (tmp_path / "frontend" / "src").mkdir(parents=True)
        (tmp_path / "frontend" / "package.json").write_text(
            '{"name": "shelf", "dependencies": {"react": "^18.3.1"}}', encoding="utf-8")
        (tmp_path / "frontend" / "src" / "App.jsx").write_text(
            "import React from 'react';\nimport BookList from './pages/BookList';\n"
            "export default function App() { return <BookList />; }\n", encoding="utf-8")
    orch = LLMOrchestrator(llm_client=mock.MagicMock(), domain_model=shelf_model(), output_dir=str(tmp_path),
                           enable_tracing=False, enable_checkpointing=False, enable_import_smoke_check=False,
                           enable_requirements_ledger=False)
    orch._generator_used = GENERATOR
    orch.target = Target(kind="web_app", description="shelf web app", has_user_interface=True)
    return orch


def test_validate_app_shows_a_frontend_blocker_phase3_would_report(tmp_path):
    report = _orch(tmp_path)._validate_app()
    assert not report["verified"] and report["blocker_count"] >= 1
    assert any(MISSING_IMPORT in issue for issue in report["issues"])
    # The acceptance matrix's warnings (Library has no page here), marked as what they are.
    assert any(issue.startswith("advisory, not a blocker: acceptance: entity Library") for issue in report["issues"])


def test_no_frontend_no_frontend_findings(tmp_path):
    report = _orch(tmp_path, frontend=False)._validate_app()
    assert not any("frontend contract:" in issue or "acceptance:" in issue for issue in report["issues"])


def test_with_a_gui_model_the_matrix_is_left_to_phase3(tmp_path):
    orch = _orch(tmp_path)
    orch.gui_model = mock.sentinel.gui_model
    issues = orch._validate_app()["issues"]
    assert any(MISSING_IMPORT in issue for issue in issues)
    assert not any("acceptance:" in issue for issue in issues)


def test_phase3_lists_each_static_frontend_finding_once(tmp_path):
    messages = [issue.message for issue in _orch(tmp_path)._collect_validation_issues()]
    assert sum(MISSING_IMPORT in message for message in messages) == 1
    assert sum(message.startswith("acceptance: entity Library") for message in messages) == 1
