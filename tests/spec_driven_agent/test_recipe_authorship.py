"""The recipe must say which generator files the LLM actually changed.

``source`` records who CREATED a file, and resume re-seeds the scaffold
guardrail from it, so it stays two-valued. That makes it unable to answer the
question a reviewer asks: what did the LLM touch? A recorded run landed 13
edits and reported ``from_llm: 1`` — all but one edit went into files the
generator had created. That count is also what says which work a regeneration
would overwrite.
"""

import json

import pytest

from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator


@pytest.fixture
def orchestrator(simple_library_book_model, tmp_path):
    class Client:
        model = "mock-model"
        usage = UsageTracker("mock-model")

        def chat(self, **kwargs):
            raise AssertionError("no LLM call expected")

    built = LLMOrchestrator(
        llm_client=Client(), domain_model=simple_library_book_model,
        output_dir=str(tmp_path), enable_checkpointing=False,
    )
    for name in ("scaffold_a.py", "scaffold_b.py"):
        (tmp_path / name).write_text("x = 1\n", encoding="utf-8")
        built.executor._generator_files.add(name)
    (tmp_path / "written_by_llm.py").write_text("y = 2\n", encoding="utf-8")
    return built


def _recipe(orchestrator, tmp_path):
    orchestrator._save_recipe("do the thing", 1.0)
    return json.loads((tmp_path / ".besser_recipe.json").read_text(encoding="utf-8"))


def test_an_edited_generator_file_is_counted_separately(orchestrator, tmp_path):
    orchestrator.tool_calls_log = [
        {"tool": "modify_file", "input": {"path": "scaffold_a.py"}, "success": True},
        {"tool": "write_file", "input": {"path": "written_by_llm.py"}, "success": True},
    ]

    summary = _recipe(orchestrator, tmp_path)["output_summary"]

    assert summary["from_generator"] == 2
    assert summary["from_llm"] == 1
    assert summary["generator_files_edited_by_llm"] == 1


def test_source_still_says_who_created_the_file(orchestrator, tmp_path):
    """Resume re-seeds the scaffold guardrail from ``source``."""
    orchestrator.tool_calls_log = [
        {"tool": "modify_file", "input": {"path": "scaffold_a.py"}, "success": True},
    ]

    files = {f["path"]: f for f in _recipe(orchestrator, tmp_path)["output_files"]}

    assert files["scaffold_a.py"]["source"] == "generator"
    assert files["scaffold_a.py"]["llm_modified"] is True
    assert "llm_modified" not in files["scaffold_b.py"]


def test_a_refused_edit_does_not_count_as_authorship(orchestrator, tmp_path):
    orchestrator.tool_calls_log = [
        {"tool": "modify_file", "input": {"path": "scaffold_a.py"}, "success": False},
    ]

    summary = _recipe(orchestrator, tmp_path)["output_summary"]

    assert summary["generator_files_edited_by_llm"] == 0


def test_a_range_edit_counts_like_any_other_edit(orchestrator, tmp_path):
    orchestrator.tool_calls_log = [
        {"tool": "replace_file_lines", "input": {"path": "scaffold_b.py"}, "success": True},
    ]

    summary = _recipe(orchestrator, tmp_path)["output_summary"]

    assert summary["generator_files_edited_by_llm"] == 1
