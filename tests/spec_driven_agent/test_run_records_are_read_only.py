"""The run's own records are not the model's to edit.

``.besser_*`` files and directories at the workspace root (trace, checkpoint,
attribution, recipe, spilled command logs, ...) and ``.agent-trace/`` are
written by the harness. list_files and search_in_files already hide most of
them, but write_file / modify_file / replace_file_lines / delete_file did
not refuse them: a confused or prompt-injected model could erase the trace,
forge the attribution record or corrupt the checkpoint a resume trusts.
Reads stay allowed: the model is told to read spilled command logs and the
probe's server log, and nothing in these records is hidden from it.
"""
import os

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.execution.process import COMMAND_OUTPUT_DIR
from besser.spec_driven_agent.state.agent_trace import TRACE_DIRNAME
from besser.spec_driven_agent.state.checkpoint import CHECKPOINT_FILENAME
from besser.spec_driven_agent.state.tracing import TRACE_FILENAME
from besser.utilities.provenance import ATTRIBUTION_FILENAME

ORIGINAL = "{}\n"
RECORDS = [TRACE_FILENAME, CHECKPOINT_FILENAME, ATTRIBUTION_FILENAME,
           f"{COMMAND_OUTPUT_DIR}/001-npm-run-build.log", f"{TRACE_DIRNAME}/run.json"]


@pytest.fixture
def executor(tmp_path):
    for rel in RECORDS:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ORIGINAL, encoding="utf-8")
    return ToolExecutor(workspace=str(tmp_path), per_write_diagnostics=False)


def call(executor, name, **args):
    return executor.execute_typed(name, args).payload


@pytest.mark.parametrize("rel", RECORDS)
def test_run_records_cannot_be_written_edited_or_deleted(executor, rel):
    path = os.path.join(executor.workspace, rel)
    read = call(executor, "read_file", path=rel)

    results = [
        call(executor, "write_file", path=rel, content='{"forged": 1}\n'),
        call(executor, "modify_file", path=rel, old_text="{}", new_text='{"forged": 1}'),
        call(executor, "replace_file_lines", path=rel, read_id=read.get("read_id", ""),
             start_line=1, end_line=1, new_text='{"forged": 1}\n'),
        call(executor, "delete_file", path=rel),
    ]

    for result in results:
        assert "run record" in result.get("error", ""), result
    with open(path, encoding="utf-8") as handle:
        assert handle.read() == ORIGINAL


def test_a_new_run_record_cannot_be_created(executor):
    result = call(executor, "write_file", path=".besser_recipe.json", content="{}\n")

    assert "run record" in result.get("error", ""), result
    assert not os.path.exists(os.path.join(executor.workspace, ".besser_recipe.json"))


def test_run_records_stay_readable(executor):
    for rel in RECORDS:
        assert "error" not in call(executor, "read_file", path=rel), rel


def test_a_lookalike_project_file_is_still_writable(executor):
    for rel in ("src/.besser_notes.txt", "besser_trace.jsonl"):
        assert call(executor, "write_file", path=rel, content="ok\n").get("status") == "written", rel
