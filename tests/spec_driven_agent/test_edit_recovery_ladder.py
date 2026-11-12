"""The edit-recovery ladder must only name a step the executor will run.

Every refused edit costs a paid turn, and so does every instruction that
leads into another refusal. Reproduced before these tests: two refused range
edits on a 230-line generated router told the model to read the whole file
and ``write_file`` it, and ``write_file`` then refused it - its unlock counted
``modify_file`` calls only.
"""

import os

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor


def call(executor, name, **args):
    return executor.execute_typed(name, args).payload


def _seed(tmp_path, rel, lines):
    target = tmp_path / rel
    os.makedirs(target.parent, exist_ok=True)
    target.write_text("".join(f"v_{n} = {n}\n" for n in range(lines)), encoding="utf-8")


def _refused_range_edit(executor, rel):
    read = call(executor, "read_file", path=rel, offset=2, limit=2)
    refused = call(executor, "replace_file_lines", path=rel, read_id=read["read_id"],
                   start_line=3, end_line=3, new_text="v_2 = (\n")
    assert refused["rejection_kind"] == "syntax_error", refused
    return refused


def _miss(executor, rel):
    return call(executor, "modify_file", path=rel, old_text="v_3 = 99\n", new_text="v_3 = 4\n")


def _follow_rewrite(executor, rel):
    """Do exactly what the write_file step says: read whole, write it back."""
    call(executor, "read_file", path=rel)
    source = open(os.path.join(executor.workspace, rel), encoding="utf-8").read()
    return call(executor, "write_file", path=rel, content=source.replace("v_3 = 3", "v_3 = 4"))


@pytest.mark.parametrize("first", ["range", "modify"])
def test_the_rewrite_step_on_a_large_generated_file_is_not_refused(tmp_path, first):
    rel = "backend/routers/book.py"
    _seed(tmp_path, rel, 230)
    ex = ToolExecutor(workspace=str(tmp_path))
    ex._generator_files.add(rel)

    _refused_range_edit(ex, rel) if first == "range" else _miss(ex, rel)
    refused = _refused_range_edit(ex, rel)
    assert refused["edit_recovery"]["next_tool"] == "write_file"

    written = _follow_rewrite(ex, rel)
    assert written.get("status") == "written", written


def test_the_rewrite_step_in_a_modify_run_is_not_refused(tmp_path):
    _seed(tmp_path, "app.py", 40)
    ex = ToolExecutor(workspace=str(tmp_path))
    ex.enable_modify_guard()

    _refused_range_edit(ex, "app.py")
    assert _refused_range_edit(ex, "app.py")["edit_recovery"]["next_tool"] == "write_file"

    written = _follow_rewrite(ex, "app.py")
    assert written.get("status") == "written", written


def test_the_modify_run_rule_still_refuses_a_cold_rewrite(tmp_path):
    _seed(tmp_path, "app.py", 40)
    ex = ToolExecutor(workspace=str(tmp_path))
    ex.enable_modify_guard()

    _refused_range_edit(ex, "app.py")
    refused = _follow_rewrite(ex, "app.py")
    assert "MODIFY run" in refused.get("error", ""), refused


# -- one instruction per result ---------------------------------------------

# Twelve near-identical functions: a quote with one mistyped line brackets one
# of them, so modify_file returns a located_range.
SERVICE = "\n".join(
    f"def f{i}(a, b):\n    total = a + b + {i}\n    if total > {i}:\n        return total\n    return None\n"
    for i in range(12)
)


def _bracketed_miss(executor, i):
    quote = f"def f{i}(a, b):\n    total = a+b+{i}\n    if total > {i}:\n        return total\n    return None"
    return call(executor, "modify_file", path="svc.py", old_text=quote,
                new_text=quote.replace("None", "0"))


@pytest.fixture
def service(tmp_path):
    (tmp_path / "svc.py").write_text(SERVICE, encoding="utf-8")
    ex = ToolExecutor(workspace=str(tmp_path))
    call(ex, "read_file", path="svc.py")
    return ex


def test_a_first_miss_names_the_located_range_once(service):
    first = _bracketed_miss(service, 3)

    assert first["located_range"]["read_id"]
    # The orchestrator reads only next_tool; the range carries the rest.
    assert first["edit_recovery"] == {"next_tool": "replace_file_lines"}
    assert "advice" not in first


def test_a_second_refusal_carries_only_the_rewrite_step(service):
    _bracketed_miss(service, 3)
    second = _bracketed_miss(service, 4)

    assert second["edit_recovery"]["next_tool"] == "write_file"
    for competing in ("located_range", "advice", "did_you_mean"):
        assert competing not in second, (competing, second[competing])
    assert "copy old_text" not in second["error"]


def test_the_rewrite_step_rides_on_the_refusal_not_on_every_read(service):
    """A replayed 13-miss loop carried the ~430-char block on 25 reads."""
    _bracketed_miss(service, 3)
    assert _bracketed_miss(service, 4)["edit_recovery"]["next_tool"] == "write_file"

    for args in ({}, {"offset": 3, "limit": 5}):
        read = call(service, "read_file", path="svc.py", **args)
        assert "edit_recovery" not in read, read["edit_recovery"]
