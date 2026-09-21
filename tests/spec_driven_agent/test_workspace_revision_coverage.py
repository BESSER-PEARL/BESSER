"""``_workspace_revision`` must see every source file, not only Python's.

The extension allowlist ``continue``d BEFORE ``fingerprint.update(rel)``, so a
non-matching file contributed neither content nor presence. Measured against
the pre-fix method, ``App.vue``, ``Page.svelte``, ``main.dart``,
``Program.cs``, ``App.swift``, ``Dockerfile.backend``, ``Dockerfile.frontend``,
``go.mod`` and ``Makefile`` were all invisible, and so was DELETING one.

``_UNSUPPORTED_STACK_RE`` names vue, svelte, flutter, swift, csharp and golang
as stacks Phase 2 builds from scratch, so those runs wrote files every turn and
registered zero source progress: the Phase 2 inspection handoff fired at turn
20 of 120 claiming the model had only inspected, a Phase 3 attempt repairing
``Dockerfile.backend`` counted as no-progress, and ``_validate_app``'s
revision-keyed cache returned a verdict measured on a different tree.

The revision also feeds the Phase 3 plateau guard and best-tree score, so the
no-progress direction is pinned here too: a real edit must change it, and a
run that only reads must not.
"""

from __future__ import annotations

import os

import pytest

from besser.BUML.metamodel.structural import (
    Class,
    DomainModel,
    PrimitiveDataType,
    Property,
)
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator

StringType = PrimitiveDataType("str")


class _Usage:
    estimated_cost = 0.0


class _Client:
    model = "mock-model"

    def __init__(self) -> None:
        self.usage = _Usage()

    def chat(self, system, messages, tools):  # pragma: no cover - never called
        return {"stop_reason": "end_turn", "content": []}


@pytest.fixture
def orch(tmp_path) -> LLMOrchestrator:
    model = DomainModel(
        name="Tiny",
        types={Class(name="Note", attributes={Property(name="body", type=StringType)})},
    )
    return LLMOrchestrator(
        llm_client=_Client(),
        domain_model=model,
        output_dir=str(tmp_path),
        enable_tracing=False,
        enable_checkpointing=False,
    )


def _write(orch: LLMOrchestrator, rel: str, text: str) -> str:
    path = os.path.join(orch.output_dir, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


# The stacks _UNSUPPORTED_STACK_RE hands to Phase 2, plus the build files a
# Phase 3 repair edits.
UNSEEN_BEFORE = [
    "frontend/src/App.vue",
    "frontend/src/Page.svelte",
    "lib/main.dart",
    "src/Program.cs",
    "Sources/App.swift",
    "src/Main.scala",
    "lib/app.ex",
    "Dockerfile.backend",
    "Dockerfile.frontend",
    "go.mod",
    "go.sum",
    "Makefile",
    "README.md",
    ".env",
]


@pytest.mark.parametrize("rel", UNSEEN_BEFORE)
def test_creating_a_source_file_changes_the_revision(orch, rel):
    before = orch._workspace_revision()
    _write(orch, rel, "first version\n")
    assert orch._workspace_revision() != before, f"writing {rel} registered no progress"


@pytest.mark.parametrize("rel", UNSEEN_BEFORE)
def test_editing_a_source_file_changes_the_revision(orch, rel):
    _write(orch, rel, "first version\n")
    before = orch._workspace_revision()
    _write(orch, rel, "second version\n")
    assert orch._workspace_revision() != before, f"editing {rel} registered no progress"


def test_deleting_a_file_changes_the_revision(orch):
    """Presence must count, or a removal looks like an unchanged tree."""
    path = _write(orch, "frontend/src/App.vue", "<template/>\n")
    before = orch._workspace_revision()
    os.remove(path)
    assert orch._workspace_revision() != before


def test_python_paths_still_tracked(orch):
    """Calibration: the extensions that already worked must keep working."""
    _write(orch, "main.py", "x = 1\n")
    before = orch._workspace_revision()
    _write(orch, "main.py", "x = 2\n")
    assert orch._workspace_revision() != before


# ------------------------------------- the plateau guard's other direction


def test_reading_the_tree_leaves_the_revision_alone(orch):
    """No-progress detection must still fire when nothing was written."""
    _write(orch, "frontend/src/App.vue", "<template/>\n")
    first = orch._workspace_revision()
    for rel in ("frontend/src/App.vue", "go.mod"):
        path = os.path.join(orch.output_dir, rel)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                handle.read()
    assert orch._workspace_revision() == first


def test_rewriting_identical_bytes_is_not_progress(orch):
    """The stall guard's whole premise: same content, same revision."""
    _write(orch, "Dockerfile.backend", "FROM python:3.11\n")
    before = orch._workspace_revision()
    _write(orch, "Dockerfile.backend", "FROM python:3.11\n")
    assert orch._workspace_revision() == before


@pytest.mark.parametrize("rel", ["app.db", "npm-debug.log", "tsconfig.tsbuildinfo"])
def test_build_and_runtime_leftovers_are_not_progress(orch, rel):
    """A check that compares the revision across its own run must not trip.

    ``collect_frontend_build_issues`` calls ``source_revision()`` before and
    after ``npm run build`` and reports "source changed during build" when
    they differ; the boot probe's sqlite file is the same shape of problem.
    Neither the appearance nor the content of these counts.
    """
    before = orch._workspace_revision()
    _write(orch, rel, "generated v1")
    assert orch._workspace_revision() == before
    _write(orch, rel, "generated v2, quite different")
    assert orch._workspace_revision() == before


def test_excluded_dirs_stay_excluded(orch):
    """node_modules / __pycache__ churn must not count as source progress."""
    before = orch._workspace_revision()
    _write(orch, "node_modules/left-pad/index.js", "module.exports = 1\n")
    _write(orch, "__pycache__/main.cpython-311.pyc", "bytes\n")
    assert orch._workspace_revision() == before
