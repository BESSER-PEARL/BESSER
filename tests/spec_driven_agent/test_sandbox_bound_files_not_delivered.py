"""A file the agent wrote for its own use inside the run folder is not part of the app.

The run folder is a temporary ``besser_llm_*`` directory removed after the
run. A check script the agent wrote with that absolute path in it (``cd`` into
the run folder, then curl the test server) cannot run anywhere else, yet it
was delivered at the root of the download, exposing the server's directory
layout. A script that is meant to ship refers to its own location instead.
"""
import json

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker


def _orchestrator(model, workspace):
    class Client:
        model = "mock-model"
        usage = UsageTracker("mock-model")

        def chat(self, **kwargs):
            raise AssertionError("no LLM call expected")

    return LLMOrchestrator(llm_client=Client(), domain_model=model, output_dir=str(workspace),
                           enable_checkpointing=False)


def _written(orchestrator, *paths):
    orchestrator.tool_calls_log = [{"tool": "write_file", "input": {"path": p}, "success": True} for p in paths]


def _recipe(orchestrator, workspace):
    orchestrator._save_recipe("add order export", 1.0)
    return json.loads((workspace / ".besser_recipe.json").read_text(encoding="utf-8"))


def _check_script(workspace):
    return f"#!/bin/sh\ncd {workspace}\ncurl -s http://localhost:41234/order/\n"


def test_a_script_bound_to_the_run_folder_is_not_delivered(simple_library_book_model, tmp_path):
    workspace = tmp_path / "besser_llm_0123abcd_x1"
    workspace.mkdir()
    orchestrator = _orchestrator(simple_library_book_model, workspace)
    (workspace / "check_orders.sh").write_text(_check_script(workspace), encoding="utf-8")
    (workspace / "export.sh").write_text('#!/bin/sh\ncd "$(dirname "$0")"\npython export.py\n', encoding="utf-8")
    _written(orchestrator, "check_orders.sh", "export.sh")

    recipe = _recipe(orchestrator, workspace)

    assert not (workspace / "check_orders.sh").exists()
    assert (workspace / "export.sh").exists()
    assert recipe["scratch_files_removed"] == ["check_orders.sh"]
    assert "check_orders.sh" not in [entry["path"] for entry in recipe["output_files"]]


def test_a_path_to_a_folder_the_caller_chose_stays_valid(simple_library_book_model, tmp_path):
    workspace = tmp_path / "order_app"
    workspace.mkdir()
    orchestrator = _orchestrator(simple_library_book_model, workspace)
    (workspace / "check_orders.sh").write_text(_check_script(workspace), encoding="utf-8")
    _written(orchestrator, "check_orders.sh")

    assert _recipe(orchestrator, workspace)["scratch_files_removed"] == []
    assert (workspace / "check_orders.sh").exists()


def test_only_files_this_run_wrote_are_considered(simple_library_book_model, tmp_path):
    workspace = tmp_path / "besser_llm_0123abcd_x2"
    workspace.mkdir()
    (workspace / "seeded.sh").write_text(_check_script(workspace), encoding="utf-8")
    orchestrator = _orchestrator(simple_library_book_model, workspace)
    (workspace / "generated.cfg").write_text(_check_script(workspace), encoding="utf-8")
    orchestrator.executor._generator_files.add("generated.cfg")
    _written(orchestrator, "seeded.sh")

    assert _recipe(orchestrator, workspace)["scratch_files_removed"] == []
    assert (workspace / "seeded.sh").exists() and (workspace / "generated.cfg").exists()
