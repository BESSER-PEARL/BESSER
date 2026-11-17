"""A relative output directory must not change what the checks see.

Live runs started with ``--out ../live/out`` reported ``E902 The system cannot
find the path specified`` for ``live/out/live/out/<run>``: ruff was given the
relative target and a cwd pinned to it, so the path resolved twice and no
Python file was linted, while the run still ended with no blocker.
"""
from __future__ import annotations

import os
import shutil
from types import SimpleNamespace

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.validation.toolchain import _collect_ruff_issues


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is not installed")
def test_ruff_lints_a_workspace_named_by_a_relative_path(tmp_path, monkeypatch):
    app = tmp_path / "runs" / "app"
    app.mkdir(parents=True)
    (app / "main.py").write_text("print(never_defined)\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    issues, _ = _collect_ruff_issues(os.path.join("runs", "app"), [], frozenset(), False)

    assert len(issues) == 1 and "F821" in issues[0] and "E902" not in issues[0], issues


def test_the_orchestrator_keeps_an_absolute_output_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = SimpleNamespace(model="mock-model", usage=UsageTracker("mock-model"))

    orchestrator = LLMOrchestrator(
        llm_client=client, domain_model=DomainModel(name="App", types={Class(name="Note")}),
        output_dir="out", enable_checkpointing=False)

    assert orchestrator.output_dir == str(tmp_path / "out")
