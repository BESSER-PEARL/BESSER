"""A truncated command log must not lose the errors Phase 3 has to act on.

``_run_command`` caps each stream at ``MAX_OUTPUT_SIZE // 2`` and, for stderr,
``_truncate(keep_tail=True)`` keeps head 20% + tail 60%. A failing ``tsc`` or
``npm run build`` prints a banner, then the diagnostics, then a summary line —
so the 20% discarded from the middle is exactly the list of errors, while the
head (the banner) and the tail (``Found N errors.``) both survive. The agent
is then told the build failed and not which lines to fix.

The full output is now spilled to ``.besser_command_output/`` inside the run
workspace and the tool result carries its path, so ``search_in_files`` and
``read_file`` can reach the dropped middle. The spill is run-internal: it must
not reach the download zip, a GitHub push, the scaffold inventory or the
recipe manifest.
"""

from __future__ import annotations

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from besser.spec_driven_agent.execution.process import COMMAND_OUTPUT_DIR
from besser.spec_driven_agent.tool_executor import (
    MAX_OUTPUT_SIZE,
    MAX_SPILL_SIZE,
    ToolExecutor,
)

MARKER = "src/pages/Booking.tsx(214,9): error TS2322: TYPE-ERROR-MARKER"


def _executor(tmp_path) -> ToolExecutor:
    return ToolExecutor(workspace=str(tmp_path), allow_shell=True)


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> MagicMock:
    mock = MagicMock(spec=subprocess.CompletedProcess)
    mock.returncode = returncode
    mock.stdout = stdout
    mock.stderr = stderr
    return mock


def _failing_build_log() -> str:
    """Banner, then diagnostics, then a summary — the tsc/npm shape."""
    limit = MAX_OUTPUT_SIZE // 2
    banner = "> vite build\nvite v5.4.2 building for production...\n" + ("-" * 4000) + "\n"
    middle = (
        "\n".join(
            f"src/pages/Page{i}.tsx({i},9): error TS2322: Type 'string' is not "
            "assignable to type 'number'." for i in range(400)
        )
        + "\n" + MARKER + "\n"
    )
    tail = "\n".join(f"note: see tsconfig option {i}" for i in range(200))
    log = f"{banner}{middle}{tail}\nFound 401 errors.\n"
    assert len(log) > limit, "fixture must actually overrun the cap"
    return log


def _run(ex: ToolExecutor, command: str, stdout: str = "", stderr: str = "") -> dict:
    completed = _completed(1 if stderr else 0, stdout=stdout, stderr=stderr)
    with patch("besser.spec_driven_agent.tool_executor.subprocess.run", return_value=completed):
        return ex._run_command({"command": command})


# ----------------------------------------------------------------------
# The measured loss
# ----------------------------------------------------------------------


def test_head_tail_truncation_really_does_drop_the_errors(tmp_path):
    """Pins the motivating defect: the marker is in neither stream returned."""
    ex = _executor(tmp_path)
    result = _run(ex, "npm run build", stderr=_failing_build_log())

    assert result["success"] is False
    assert MARKER not in result["stderr"], (
        "fixture no longer reproduces the loss; the middle must be discarded"
    )


def test_the_full_log_is_spilled_and_its_path_returned(tmp_path):
    ex = _executor(tmp_path)
    result = _run(ex, "npm run build", stderr=_failing_build_log())

    rel = result["full_output_path"]
    assert rel.startswith(COMMAND_OUTPUT_DIR + "/")
    spilled = (tmp_path / rel).read_text(encoding="utf-8")
    assert MARKER in spilled, "the dropped middle must survive in the spill"
    assert "npm run build" in spilled, "the log must name the command it came from"


def test_the_model_is_told_the_middle_is_missing(tmp_path):
    """A path alone is not actionable — the note has to say why to open it."""
    ex = _executor(tmp_path)
    result = _run(ex, "npm run build", stderr=_failing_build_log())

    note = result["full_output_note"]
    assert result["full_output_path"] in note
    assert "search_in_files" in note
    assert "truncated" in note.lower()


def test_search_in_files_can_reach_the_spill(tmp_path):
    """``.besser_*`` paths are hidden from search; this one must not be."""
    ex = _executor(tmp_path)
    rel = _run(ex, "npm run build", stderr=_failing_build_log())["full_output_path"]

    found = ex._search_in_files({"pattern": "TYPE-ERROR-MARKER"})
    assert [m["file"] for m in found["matches"]] == [rel]


def test_the_spill_never_crowds_source_out_of_a_search(tmp_path):
    """search_in_files stops at 50 matches; the log must be searched last."""
    ex = _executor(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    (src / "Booking.tsx").write_text("error TS2322\n" * 60, encoding="utf-8")
    _run(ex, "npm run build", stderr=_failing_build_log())

    found = ex._search_in_files({"pattern": "error TS2322"})
    assert {m["file"] for m in found["matches"]} == {"src/Booking.tsx"}


def test_read_file_can_reach_the_spill(tmp_path):
    ex = _executor(tmp_path)
    rel = _run(ex, "npm run build", stderr=_failing_build_log())["full_output_path"]

    assert "error TS2322" in ex._read_file({"path": rel})["content"]


def test_a_long_stdout_spills_too(tmp_path):
    """pytest writes its failures to stdout, not stderr."""
    ex = _executor(tmp_path)
    result = _run(ex, "python -m pytest", stdout=_failing_build_log())

    assert MARKER not in result["stdout"]
    assert MARKER in (tmp_path / result["full_output_path"]).read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# No junk for the ordinary case
# ----------------------------------------------------------------------


def test_output_within_the_cap_does_not_spill(tmp_path):
    ex = _executor(tmp_path)
    result = _run(ex, "python -m pytest -q", stdout="3 passed in 0.4s\n")

    assert "full_output_path" not in result
    assert not os.path.isdir(tmp_path / COMMAND_OUTPUT_DIR)


def test_each_spill_gets_its_own_file(tmp_path):
    ex = _executor(tmp_path)
    first = _run(ex, "npm run build", stderr=_failing_build_log())["full_output_path"]
    second = _run(ex, "npm run build", stderr=_failing_build_log())["full_output_path"]

    assert first != second, "a second run must not overwrite the first log"
    assert len(os.listdir(tmp_path / COMMAND_OUTPUT_DIR)) == 2


def test_a_runaway_command_cannot_fill_the_disk(tmp_path, monkeypatch):
    """The spill is deliberately huge next to the context cap, but bounded."""
    monkeypatch.setattr(
        "besser.spec_driven_agent.tool_executor.MAX_SPILL_SIZE", MAX_OUTPUT_SIZE * 2,
    )
    ex = _executor(tmp_path)
    runaway = "x" * (MAX_OUTPUT_SIZE * 10)

    rel = _run(ex, "npm run build", stdout=runaway)["full_output_path"]
    spilled = (tmp_path / rel).read_text(encoding="utf-8")

    assert len(spilled) < len(runaway)
    assert "chars dropped from the spill" in spilled


def test_the_spill_ceiling_is_far_above_the_context_cap():
    """A cap that clipped a real build log would defeat the point."""
    assert MAX_SPILL_SIZE > MAX_OUTPUT_SIZE * 100


def test_a_hostile_command_string_cannot_escape_the_spill_directory(tmp_path):
    ex = _executor(tmp_path)
    rel = _run(ex, "../../etc/passwd && echo x", stderr=_failing_build_log())[
        "full_output_path"
    ]

    assert rel.startswith(COMMAND_OUTPUT_DIR + "/")
    assert ".." not in rel
    written = os.path.realpath(str(tmp_path / rel))
    assert written.startswith(os.path.realpath(str(tmp_path)) + os.sep)


def test_the_spill_is_not_advertised_as_a_project_file(tmp_path):
    ex = _executor(tmp_path)
    _run(ex, "npm run build", stderr=_failing_build_log())

    listed = [f["path"] for f in ex._list_files({})["files"]]
    assert listed == [], "list_files describes the generated project, not run logs"


# ----------------------------------------------------------------------
# Run-internal: never packaged, pushed, inventoried or listed in the recipe
# ----------------------------------------------------------------------


def test_the_scaffold_inventory_ignores_the_spill(tmp_path):
    from besser.spec_driven_agent.prompt_builder import build_inventory

    ex = _executor(tmp_path)
    (tmp_path / "main_api.py").write_text("app = 1\n", encoding="utf-8")
    _run(ex, "npm run build", stderr=_failing_build_log())

    inventory = build_inventory(str(tmp_path), None, "generate_fastapi_backend")
    assert "main_api.py" in inventory
    assert COMMAND_OUTPUT_DIR not in inventory
    assert "produced 1 files" in inventory


def test_the_recipe_manifest_excludes_the_spill():
    from besser.spec_driven_agent.orchestrator import _RECIPE_EXCLUDED_DIRS

    assert COMMAND_OUTPUT_DIR in _RECIPE_EXCLUDED_DIRS


@pytest.mark.parametrize("name", [COMMAND_OUTPUT_DIR])
def test_packaging_and_push_exclude_the_spill(name):
    from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
        _EXCLUDED_OUTPUT_DIRS,
    )

    assert name in _EXCLUDED_OUTPUT_DIRS
