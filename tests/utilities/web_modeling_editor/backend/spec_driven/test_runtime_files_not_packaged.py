"""Packaging leaves out what the app wrote while the agent test-ran it.

The orchestrator deletes those files before it writes the recipe. A run that
ends without a recipe (a provider or internal error after files exist) is
still packaged, as incomplete, and a push copies the same folder: packaging
runs the removal itself, and the run summary says what was left out.
"""
import json
import os
import zipfile

from besser.spec_driven_agent.agent.runbook import RUNTIME_FILES_FILENAME
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SmartGenerationRunner,
    run_summary_markdown,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.sse_events import VerificationReport
from tests.utilities.web_modeling_editor.backend._runtime import make_runtime
from tests.utilities.web_modeling_editor.backend.spec_driven.test_runner import _build_request

KEY = "backend/data/signing.key"


class _Client:
    model = "mock-model"
    usage = UsageTracker("mock-model")

    def chat(self, **kwargs):
        raise AssertionError("no LLM call expected")


def _run_with_a_key_the_app_wrote(tmp_path, simple_library_book_model):
    result = tmp_path / "besser_llm_example" / "result"
    (result / "backend" / "data").mkdir(parents=True)
    for name in ("main_api.py", "sql_alchemy.py", "requirements.txt"):
        (result / "backend" / name).write_text("app = None\n", encoding="utf-8")
    orchestrator = LLMOrchestrator(llm_client=_Client(), domain_model=simple_library_book_model,
                                   output_dir=str(result), enable_checkpointing=False)
    orchestrator.executor._generator_files.update(
        {"backend/main_api.py", "backend/sql_alchemy.py", "backend/requirements.txt"})
    (result / KEY).write_text("0f1e2d3c" * 8, encoding="utf-8")
    info = os.stat(result / KEY)
    (result / RUNTIME_FILES_FILENAME).write_text(
        json.dumps({"files": {KEY: [info.st_size, info.st_mtime_ns]}, "overflow": 0}), encoding="utf-8")
    runner = SmartGenerationRunner(_build_request(), runtime=make_runtime())
    runner.temp_dir = str(result.parent)
    runner._t_orchestrator = orchestrator
    return runner, result


def test_a_run_that_ended_without_a_recipe_is_packaged_without_the_apps_key(tmp_path, simple_library_book_model):
    runner, result = _run_with_a_key_the_app_wrote(tmp_path, simple_library_book_model)

    _done, entry = runner._package_result(str(result))

    with zipfile.ZipFile(entry.file_path) as archive:
        names = archive.namelist()
        summary = archive.read("BESSER_GENERATION.md").decode("utf-8")
    assert not [name for name in names if name.endswith("signing.key")], names
    assert "backend/main_api.py" in names
    assert not (result / KEY).exists()  # a push copies this folder
    assert "- Left out: 1 file(s) the app wrote while it was test-run" in summary


def test_the_run_summary_says_what_could_not_be_removed():
    lines = run_summary_markdown({
        "runtime_files_removed_count": 3, "runtime_files_not_removed_count": 1,
        "runtime_files_note": "the record of files the app wrote could not be read",
    }, VerificationReport())

    assert ("- Left out: 3 file(s) the app wrote while it was test-run; 1 more could not be removed "
            "and are still in the output (`the record of files the app wrote could not be read`)") in lines


def test_the_run_summary_is_silent_when_the_app_wrote_nothing():
    lines = run_summary_markdown({"runtime_files_removed_count": 0, "runtime_files_note": ""}, VerificationReport())
    assert not [line for line in lines if "Left out" in line]


def test_the_run_summary_names_files_left_out_for_naming_the_run_folder():
    lines = run_summary_markdown({"scratch_files_removed": ["check_orders.sh"]}, VerificationReport())
    assert ("- Left out: helper script(s) the agent wrote that named the run's temporary folder: "
            "`check_orders.sh`") in lines


def test_the_run_summary_names_kept_files_that_name_a_temporary_path():
    lines = run_summary_markdown({"temporary_path_files": ["backend/settings.py"]}, VerificationReport())
    assert ("- Names a temporary path: the run's folder, which does not exist after the run; "
            "replace it with a relative path: `backend/settings.py`") in lines
