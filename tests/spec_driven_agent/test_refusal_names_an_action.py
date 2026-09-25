"""Every refusal must name a next action the model can actually take.

Measured over the 1,825 refused Qwen3-30B-A3B edit calls in a recorded
corpus (362 run workspaces): 504 refusals (28%) ended without naming any next
action, and 78 of those carried no ``did_you_mean`` / ``advice`` /
``edit_recovery`` field either - a dead end in the literal sense, all of them
in two classes:

    41  a replace_file_lines range edit that changes nothing
    37  an edit that leaves the ORM module importable-broken

A weak model answers a dead end by resending. The corpus shows exactly that:
126 of the 234 streak-guard refusals follow another streak-guard refusal, and
41 of 202 Qwen runs drove a file into the miss guard at all.

The third refusal audited here is the one that used to contradict itself -
``write_file`` refused a generator file and sent the model to
``delete_file + write_file``, while ``delete_file`` refused the same file.
That pair is already gone at HEAD; this locks it shut.
"""

import re

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor


ORM = (
    "from sqlalchemy.orm import Mapped, mapped_column\n"
    "\n"
    "class Booking:\n"
    "    id: Mapped[int] = mapped_column(primary_key=True)\n"
)


def _executor(tmp_path, **files) -> ToolExecutor:
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    return ToolExecutor(workspace=str(tmp_path))


def _actionable(message: str) -> bool:
    """A refusal is actionable when it names a tool to call next."""
    return any(t in message for t in (
        "read_file", "modify_file", "write_file", "replace_file_lines",
        "run_command", "search_in_files", "task_list",
    ))


class TestRangeEditNoOp:
    """154 Qwen refusals said only "this is not evidence of implementation"."""

    def test_no_op_range_edit_says_what_to_do_next(self, tmp_path):
        ex = _executor(tmp_path, **{"m.py": ORM})
        read = ex._read_file({"path": "m.py"})
        res = ex._replace_file_lines({
            "path": "m.py", "read_id": read["read_id"],
            "start_line": 3, "end_line": 4,
            "new_text": "class Booking:\n    id: Mapped[int] = mapped_column(primary_key=True)\n",
        })
        assert "error" in res
        assert "makes no change" in res["error"]
        assert _actionable(res["error"]), res["error"]
        assert "read_file" in res["error"]


class TestImportBreakNamesARecovery:
    """148 Qwen refusals ended at "The file was left unchanged"."""

    def test_modify_that_breaks_the_orm_import_names_a_recovery(self, tmp_path):
        ex = _executor(tmp_path, **{"sql_alchemy.py": ORM})
        ex.mark_known(["sql_alchemy.py"])
        res = ex._modify_file({
            "path": "sql_alchemy.py",
            "old_text": "    id: Mapped[int] = mapped_column(primary_key=True)\n",
            "new_text": "    id: Mapped[int] = mapped_column(primary_key=True)\n"
                        "    guest: Mapped[Guest] = mapped_column()\n",
        })
        if "error" not in res:
            pytest.skip("this tree does not refuse the edit as import-breaking")
        assert res.get("rejection_kind") == "breaks_import"
        assert _actionable(res["error"]), res["error"]
        assert "Do not resend it unchanged" in res["error"]


class TestNoRefusalPointsAtARefusedAction:
    """The contradiction: write_file sent the model to delete_file, which
    refuses the same file. Neither message may name the other as the way out."""

    def test_write_file_refusal_never_routes_through_delete_file(self, tmp_path):
        ex = _executor(tmp_path, **{"sql_alchemy.py": ORM * 80})
        ex._generator_files.add("sql_alchemy.py")
        res = ex._write_file({"path": "sql_alchemy.py", "content": "x = 1\n"})
        if "error" not in res:
            pytest.skip("write_file is not gated for generator files in this tree")
        assert "delete_file" not in res["error"], res["error"]
        assert _actionable(res["error"]), res["error"]

    def test_delete_file_refusal_names_a_route_that_is_not_delete_file(self, tmp_path):
        ex = _executor(tmp_path, **{"sql_alchemy.py": ORM})
        ex._generator_files.add("sql_alchemy.py")
        ex._protect_scaffold = True
        res = ex._delete_file({"path": "sql_alchemy.py"})
        assert "error" in res
        assert _actionable(res["error"]), res["error"]


class TestWriteFileStripsTheReadGutter:
    """Mode 9: the read gutter pasted into a whole-file rewrite.

    Two of the three write paths strip read_file's "NNN| " gutter -
    modify_file at tier 5 of the apply ladder, replace_file_lines at its own
    call site, which records that shipping without it baked " 101|  </nav>"
    into a delivered Booking.tsx. write_file was the third and did not, so a
    rewrite arriving as "   1| import re" was refused as "unexpected indent
    at line 1" and the model resent it unchanged.

    Measured over the same corpus (362 workspaces): 35 Qwen write_file
    calls carried a gutter and 1 landed (2.9%), against 98.5% for the 726
    clean ones - 34 of write_file's 56 Qwen refusals. gpt-5.6 never does it.
    The refusal was safe (nothing corrupt reached disk); the cost was the
    run's budget.
    """

    SOURCE = (
        "import re\n"
        "from enum import Enum\n"
        "\n"
        "class Status(Enum):\n"
        "    OPEN = 'open'\n"
    )
    # exactly what read_file shows, and what the model sent back
    NUMBERED = (
        "   1| import re\n"
        "   2| from enum import Enum\n"
        "   3| \n"
        "   4| class Status(Enum):\n"
        "   5|     OPEN = 'open'\n"
        "   6|     CLOSED = 'closed'\n"
    )

    def test_a_numbered_rewrite_lands_and_the_gutter_never_reaches_disk(self, tmp_path):
        ex = _executor(tmp_path, **{"models.py": self.SOURCE})
        ex.mark_known(["models.py"])
        res = ex._write_file({"path": "models.py", "content": self.NUMBERED})
        assert "error" not in res, res.get("error")
        on_disk = (tmp_path / "models.py").read_text(encoding="utf-8")
        assert "1|" not in on_disk and "   1| import re" not in on_disk
        assert on_disk.startswith("import re\nfrom enum import Enum\n")
        assert "CLOSED = 'closed'" in on_disk          # the model's actual change
        import ast
        ast.parse(on_disk)

    def test_unnumbered_content_is_written_byte_for_byte(self, tmp_path):
        ex = _executor(tmp_path, **{"models.py": self.SOURCE})
        ex.mark_known(["models.py"])
        body = self.SOURCE + "    CLOSED = 'closed'\n"
        res = ex._write_file({"path": "models.py", "content": body})
        assert "error" not in res, res.get("error")
        assert (tmp_path / "models.py").read_text(encoding="utf-8") == body

    def test_a_pipe_table_is_not_mistaken_for_a_gutter(self, tmp_path):
        """A Markdown table starts with '|', not with digits, and must survive."""
        ex = _executor(tmp_path, **{"README.md": "# doc\n"})
        ex.mark_known(["README.md"])
        body = "| n | name |\n|---|------|\n| 1 | one  |\n| 2 | two  |\n"
        res = ex._write_file({"path": "README.md", "content": body})
        assert "error" not in res, res.get("error")
        assert (tmp_path / "README.md").read_text(encoding="utf-8") == body


class TestNoWritePathLetsTheGutterReachDisk:
    """The invariant behind mode 9, asserted across every model-content path.

    Exactly three tools put model-supplied bytes on disk: modify_file (via the
    apply ladder), replace_file_lines, and write_file. Two stripped read_file's
    "NNN| " gutter and the third did not, which cost 34 of write_file's 56
    Qwen refusals. Every other write in the executor is harness-generated (the
    import-smoke probe's save/restore, import_repair, the command-log spill),
    so these three are the whole surface. A fourth must strip too.
    """

    SOURCE = "import re\nVALUE = 1\nOTHER = 2\n"

    def test_modify_file(self, tmp_path):
        ex = _executor(tmp_path, **{"m.py": self.SOURCE})
        ex.mark_known(["m.py"])
        res = ex._modify_file({
            "path": "m.py",
            "old_text": "   2| VALUE = 1\n   3| OTHER = 2\n",
            "new_text": "   2| VALUE = 99\n   3| OTHER = 2\n",
        })
        assert "error" not in res, res.get("error")
        self._assert_clean(tmp_path / "m.py", "VALUE = 99")

    def test_replace_file_lines(self, tmp_path):
        ex = _executor(tmp_path, **{"m.py": self.SOURCE})
        read = ex._read_file({"path": "m.py"})
        res = ex._replace_file_lines({
            "path": "m.py", "read_id": read["read_id"],
            "start_line": 2, "end_line": 3,
            "new_text": "   2| VALUE = 99\n   3| OTHER = 2\n",
        })
        assert "error" not in res, res.get("error")
        self._assert_clean(tmp_path / "m.py", "VALUE = 99")

    def test_write_file(self, tmp_path):
        ex = _executor(tmp_path, **{"m.py": self.SOURCE})
        ex.mark_known(["m.py"])
        res = ex._write_file({
            "path": "m.py",
            "content": "   1| import re\n   2| VALUE = 99\n   3| OTHER = 2\n",
        })
        assert "error" not in res, res.get("error")
        self._assert_clean(tmp_path / "m.py", "VALUE = 99")

    @staticmethod
    def _assert_clean(path, expected):
        import ast
        text = path.read_text(encoding="utf-8")
        assert expected in text
        for line in text.splitlines():
            assert not re.match(r"^\s*\d+\|", line), f"gutter reached disk: {line!r}"
        ast.parse(text)
