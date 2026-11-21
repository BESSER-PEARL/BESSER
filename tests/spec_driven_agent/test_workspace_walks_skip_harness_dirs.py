"""The Phase 3 workspace sweeps read the app, never the harness's own folders.

Review (2026-11, PR 788): four orchestrator sweeps carried inline copies of
``walk_app_tree``'s pruning. The two frontend checks skipped only the
``.besser_snapshot`` folder, so a frontend file in another harness folder (the
tree a rollback parks in ``.besser_rollback_discard``) counted as the app's
frontend and hid the "no frontend at all" blocker.
"""

import types

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning.target import WEB_APP, Target
from besser.spec_driven_agent.providers.llm_client import UsageTracker

WEB_APP_REQUEST = "Build a hotel booking web app with a frontend."


class _Client:
    model = "mock-model"

    def __init__(self):
        self.usage = UsageTracker("mock-model")


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def orch(tmp_path, simple_library_book_model):
    orchestrator = LLMOrchestrator(llm_client=_Client(), domain_model=simple_library_book_model,
                                   output_dir=str(tmp_path),
                                   enable_tracing=False, enable_checkpointing=False)
    orchestrator._instructions = WEB_APP_REQUEST
    orchestrator.target = Target(WEB_APP, "web app", True)
    _write(tmp_path, "backend/main_api.py", "from fastapi import FastAPI\napp = FastAPI()\n")
    return orchestrator


@pytest.mark.parametrize("harness_dir", [".besser_snapshot", ".besser_rollback_discard"])
def test_a_frontend_only_in_a_harness_folder_is_no_frontend(orch, tmp_path, harness_dir):
    _write(tmp_path, f"{harness_dir}/frontend/src/App.tsx", "export const App = () => null;\n")

    assert orch._has_frontend_files() is False
    assert orch._collect_missing_frontend_issue(), "the missing-frontend blocker was hidden"


def test_the_app_frontend_still_counts(orch, tmp_path):
    _write(tmp_path, "frontend/src/App.tsx", "export const App = () => null;\n")

    assert orch._has_frontend_files() is True
    assert orch._collect_missing_frontend_issue() == []


def test_a_rival_framework_in_the_snapshot_is_no_switch(tmp_path):
    _write(tmp_path, ".besser_snapshot/backend/app.py", "from flask import Flask\n")
    _write(tmp_path, "backend/node_modules/x/app.py", "from flask import Flask\n")
    shim = types.SimpleNamespace(output_dir=str(tmp_path), _scaffold_family=lambda: "fastapi",
                                 _instructions="Build a hotel booking API")
    assert LLMOrchestrator._collect_framework_switch_issues(shim) == []

    _write(tmp_path, "backend/app.py", "from flask import Flask\n")
    assert len(LLMOrchestrator._collect_framework_switch_issues(shim)) == 1
