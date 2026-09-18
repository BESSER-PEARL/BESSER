"""An abbreviated quote must be named as such, refused, and bounded.

Live run 2026-09-18 (Qwen3-30B-A3B-Instruct): 38 modify_file calls, 37 failed.
The model sent `contact_id:...` and `contact = rel...` instead of the real
lines; "..." was in 37/38 old_text and 38/38 new_text, so the single edit that
applied wrote an ellipsis into sql_alchemy.py. The error said "make sure
old_text matches exactly including whitespace/indentation", so the model
re-read the file and re-elided - 37 times, 1043s, $0.63. The loop guard only
appended "Move on." to the result and was ignored 61 times.
"""
import pytest

from besser.generators.llm.edit_apply import find_elision


# -- detector: the live shapes it must catch ----------------------------
@pytest.mark.parametrize("line", [
    "    contact_id:...",
    "    contact = rel...",
    "    # ... rest of the methods unchanged",
    "        totalPrice = compute...",
    "    // remaining fields unchanged ...",
])
def test_abbreviations_are_detected(line):
    assert find_elision(line) is not None


# -- and the legitimate Python it must NOT catch ------------------------
@pytest.mark.parametrize("line", [
    "    name: str = Field(...)",
    "    q: int = Query(...)",
    "    def method(self) -> int: ...",
    "    arr = a[..., 0]",
    "    x = foo(a, ...)",
    "    # ...and the row still points at the old id",
    "    contact_id: Mapped_[int] = mapped_column(ForeignKey_('person.id'))",
    "",
    "    plain code with no dots at all",
])
def test_legitimate_code_is_not_flagged(line):
    assert find_elision(line) is None


def test_reports_the_line_number_and_text():
    block = "class Booking(Base):\n    id: int\n    contact_id:...\n"
    got = find_elision(block)
    assert got is not None
    line_no, line = got
    assert line_no == 3
    assert "contact_id" in line


def test_first_elision_wins():
    block = "a = rel...\nb = other...\n"
    assert find_elision(block)[0] == 1


def test_whole_repo_stays_clean():
    """The detector is useless if it fires on our own source."""
    import pathlib
    flagged = []
    root = pathlib.Path(__file__).resolve().parents[3] / "besser"
    for p in root.rglob("*.py"):
        try:
            if find_elision(p.read_text(encoding="utf-8")):
                flagged.append(str(p))
        except OSError:
            continue
    assert not flagged, f"false positives in our own code: {flagged[:5]}"


# -- executor behaviour -------------------------------------------------
REAL_FILE = (
    'class Booking(Base):\n'
    '    __tablename__ = "booking"\n'
    '    totalPrice: Mapped_[float] = mapped_column(Float_)\n'
    '    contact_id: Mapped_[int] = mapped_column(ForeignKey_("person.id"))\n'
)

# Verbatim from the failing run.
ELIDED_OLD = (
    'class Booking(Base):\n'
    '    __tablename__ = "booking"\n'
    '    totalPrice: Mapped_[float] = mapped_column(Float_)\n'
    '    contact_id:...\n'
)


@pytest.fixture
def executor(tmp_path):
    from besser.generators.llm.tool_executor import ToolExecutor
    (tmp_path / "sql_alchemy.py").write_text(REAL_FILE, encoding="utf-8")
    return ToolExecutor(str(tmp_path))


def _modify(ex, old, new):
    return ex._modify_file({"path": "sql_alchemy.py", "old_text": old, "new_text": new})


def test_elided_old_text_blames_the_abbreviation_not_whitespace(executor):
    err = _modify(executor, ELIDED_OLD, REAL_FILE)["error"]
    assert "abbreviates" in err
    assert "contact_id:..." in err
    assert "whitespace" not in err.lower(), "the misleading hint must not lead"


def test_elided_new_text_is_refused_before_touching_the_file(executor, tmp_path):
    before = (tmp_path / "sql_alchemy.py").read_text(encoding="utf-8")
    res = _modify(executor, '    totalPrice: Mapped_[float] = mapped_column(Float_)',
                  '    totalPrice = compute...')
    assert "error" in res and "abbreviates" in res["error"]
    assert (tmp_path / "sql_alchemy.py").read_text(encoding="utf-8") == before


def test_a_placeholder_already_in_old_text_is_allowed(executor, tmp_path):
    """Gemini's carve-out: preserving an existing placeholder is a real edit."""
    (tmp_path / "doc.py").write_text("x = rel...\n", encoding="utf-8")
    res = executor._modify_file({"path": "doc.py", "old_text": "x = rel...",
                                 "new_text": "y = rel..."})
    assert "error" not in res, res


def test_modify_is_refused_after_three_consecutive_misses(executor):
    for _ in range(3):
        assert "error" in _modify(executor, "nothing like this exists", "z")
    res = _modify(executor, "still nothing", "z")
    assert "refused" in res["error"]
    assert "read_file" in res["error"]


def test_a_good_edit_still_applies(executor, tmp_path):
    res = _modify(executor, '    totalPrice: Mapped_[float] = mapped_column(Float_)',
                  '    totalPrice: Mapped_[float] = mapped_column(Float_, default=0)')
    assert "error" not in res, res
    assert "default=0" in (tmp_path / "sql_alchemy.py").read_text(encoding="utf-8")
