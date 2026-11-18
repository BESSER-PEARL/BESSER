"""A file the agent wrote for its own use inside the run folder is not part of the app.

The run folder is a temporary ``besser_llm_*`` directory removed after the
run. A check script the agent wrote with that absolute path in it (``cd`` into
the run folder, then curl the test server) cannot run anywhere else, yet it
was delivered at the root of the download, exposing the server's directory
layout. A script that is meant to ship refers to its own location instead.
"""
import json

import pytest

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


def _run_folder_with(model, tmp_path, files, generated=()):
    workspace = tmp_path / "besser_llm_0123abcd_x3"
    (workspace / "backend").mkdir(parents=True)
    (workspace / "backend" / "main_api.py").write_text("app = None\n", encoding="utf-8")
    (workspace / "backend" / "sql_alchemy.py").write_text("Base = None\n", encoding="utf-8")
    orchestrator = _orchestrator(model, workspace)
    for rel, text in files.items():
        (workspace / rel).parent.mkdir(parents=True, exist_ok=True)
        (workspace / rel).write_text(text.format(root=workspace), encoding="utf-8")
    orchestrator.executor._generator_files.update(generated)
    _written(orchestrator, *files)
    return workspace, _recipe(orchestrator, workspace)


def test_app_files_naming_the_run_folder_are_kept_and_listed(simple_library_book_model, tmp_path):
    files = {
        "backend/settings.py": 'UPLOAD_DIR = r"{root}/backend/uploads"\n',
        "backend/storage.py": "from settings import UPLOAD_DIR\n",
        "README.md": "# Orders\n\nRun locally:\n\n    cd {root}/backend && uvicorn main_api:app\n",
        "backend/tests/test_orders.py": 'import sys\nsys.path.insert(0, r"{root}/backend")\n',
        "check_orders.sh": "#!/bin/sh\ncd {root}\ncurl -s http://localhost:41234/order/\n",
    }
    workspace, recipe = _run_folder_with(simple_library_book_model, tmp_path, files)

    assert recipe["scratch_files_removed"] == ["check_orders.sh"]
    kept = ["README.md", "backend/settings.py", "backend/tests/test_orders.py"]
    assert all((workspace / rel).exists() for rel in kept)
    assert recipe["temporary_path_files"] == kept


def test_a_helper_script_another_file_refers_to_is_kept(simple_library_book_model, tmp_path):
    files = {
        "seed.sh": "#!/bin/sh\ncd {root}/backend && python seed.py\n",
        "README.md": "Load sample data with ./seed.sh\n",
        "check.py": "import os\nos.chdir(r'{root}')\n",
        "run_checks.py": "import check\n",
    }
    workspace, recipe = _run_folder_with(simple_library_book_model, tmp_path, files)

    assert recipe["scratch_files_removed"] == []
    assert (workspace / "seed.sh").exists() and (workspace / "check.py").exists()
    assert recipe["temporary_path_files"] == ["check.py", "seed.sh"]


@pytest.mark.parametrize("rel,text", [
    pytest.param("app.py", 'from flask import Flask\napp = Flask(__name__)\nDB = "sqlite:///{root}/data.db"\n',
                 id="root-entry-point-of-another-stack"),
    pytest.param("export_orders.py", 'import sqlite3\ncon = sqlite3.connect(r"{root}/backend/app.db")\n',
                 id="root-tool-the-user-asked-for"),
    pytest.param("conftest.py", 'import sys\nsys.path.insert(0, r"{root}/backend")\n', id="root-conftest"),
    pytest.param("scripts/start.sh", "#!/bin/sh\ncd {root}/backend && uvicorn main_api:app\n",
                 id="shell-script-in-a-folder"),
])
def test_python_files_and_scripts_in_folders_are_never_deleted(simple_library_book_model, tmp_path, rel, text):
    workspace, recipe = _run_folder_with(simple_library_book_model, tmp_path, {rel: text})

    assert recipe["scratch_files_removed"] == []
    assert (workspace / rel).exists()
    assert recipe["temporary_path_files"] == [rel]


def test_an_unreferenced_root_check_script_is_removed(simple_library_book_model, tmp_path):
    files = {"verify_workflow.sh": "#!/bin/sh\ncd {root}\ncurl -s http://localhost:41234/\n"}
    workspace, recipe = _run_folder_with(simple_library_book_model, tmp_path, files)

    assert recipe["scratch_files_removed"] == ["verify_workflow.sh"]
    assert not (workspace / "verify_workflow.sh").exists()


def test_python_files_in_folders_are_kept_even_without_a_detected_backend(simple_library_book_model, tmp_path):
    workspace = tmp_path / "besser_llm_0123abcd_x4"
    workspace.mkdir()
    orchestrator = _orchestrator(simple_library_book_model, workspace)
    files = {"server/tests/test_orders.py": f"import sys\nsys.path.insert(0, r'{workspace}/server')\n",
             "scripts/check.py": f"import os\nos.chdir(r'{workspace}')\n"}
    for rel, text in files.items():
        (workspace / rel).parent.mkdir(parents=True, exist_ok=True)
        (workspace / rel).write_text(text, encoding="utf-8")
    _written(orchestrator, *files)

    recipe = _recipe(orchestrator, workspace)

    assert recipe["scratch_files_removed"] == []
    assert recipe["temporary_path_files"] == ["scripts/check.py", "server/tests/test_orders.py"]
