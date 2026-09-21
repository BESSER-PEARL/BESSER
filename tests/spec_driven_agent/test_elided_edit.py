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

from besser.spec_driven_agent.edit_apply import find_elision


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


# Files the detector is known to fire on, and why each is acceptable. A bare
# count was the wrong shape: it scanned 2,000+ files with no headroom, so any
# new file anywhere in the product broke this test for reasons unrelated to
# edit_apply - it tripped twice in one day on unrelated work. Naming them
# makes a new offender an explicit decision rather than a mystery failure.
_ACCEPTED_ELISION_HITS = {
    # Legal Python stub bodies: a dots-only line IS the statement.
    "error_handler.py",
    "quantum_diagram_processor.py",
    # The detector's own module, describing the form it catches.
    "edit_apply.py",
}


def test_false_positives_stay_bounded():
    """The detector is useless if it fires all over our own source.

    Accepting the dots-only line costs a handful of legal stub bodies. That
    is the measured price of catching the form that corrupted
    booking_methods.py. This fails if a change makes the detector noisier,
    and names what it newly fired on.
    """
    import pathlib
    flagged = set()
    # Derived from the package, not a parent hop count: the hop count was
    # 3 while these tests lived at tests/generators/llm/ and silently became
    # wrong when they moved one level up. On Windows it did not even fail
    # loudly -- parents[3]/"besser" case-insensitively matched the repo root
    # BESSER/, so this scanned tests/ too and flagged the test files.
    import besser
    root = pathlib.Path(besser.__file__).resolve().parent
    for path in root.rglob("*.py"):
        try:
            if find_elision(path.read_text(encoding="utf-8")):
                flagged.add(path.name)
        except OSError:
            continue

    unexpected = flagged - _ACCEPTED_ELISION_HITS
    assert not unexpected, (
        f"the elision detector newly fires on {sorted(unexpected)}. Either that "
        f"source has an ellipsis reading as an elided edit - reword it - or it "
        f"is a genuine new stub body, in which case add it to "
        f"_ACCEPTED_ELISION_HITS with the reason."
    )


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
    from besser.spec_driven_agent.tool_executor import ToolExecutor
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


def test_a_placeholder_preserved_verbatim_is_allowed(executor, tmp_path):
    """Gemini's carve-out, narrowed: preserving a placeholder is a real edit."""
    (tmp_path / "doc.py").write_text("x = rel...\nkeep = 1\n", encoding="utf-8")
    res = executor._modify_file({"path": "doc.py", "old_text": "x = rel...\nkeep = 1",
                                 "new_text": "x = rel...\nkeep = 2"})
    assert "error" not in res, res


def test_a_placeholder_that_changed_text_is_refused(executor, tmp_path):
    """Run 36e9c8a6 leaked 6 elisions and refused 0: an ellipsis anywhere in
    old_text excused every ellipsis in new_text. A CHANGED elided line is a
    new abbreviation, not a preserved one."""
    (tmp_path / "doc.py").write_text("x = rel...\n", encoding="utf-8")
    res = executor._modify_file({"path": "doc.py", "old_text": "x = rel...",
                                 "new_text": "y = rel..."})
    assert "abbreviates" in res.get("error", "")


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


# -- the bare "..." line (live 2026-09-18, run 15a8ac7d) -----------------
# The model used a dots-only line on 10 of 10 edits to booking_methods.py.
# Four were rejected; the SIX THAT APPLIED spliced a second `try:` inside an
# unclosed one, so the module failed to import: "line 37: expected 'except'
# or 'finally' block". An accepted elision is worse than a rejected one.
RUN13_OLD_TEXT = (
    "        booking = database.query(Booking).first()\n"
    "...\n"
    "        return bill\n"
)


def test_a_dots_only_line_is_an_elision():
    got = find_elision(RUN13_OLD_TEXT)
    assert got is not None
    assert got[0] == 2


@pytest.mark.parametrize("line", ["...", "    ...", "\t...", "   ...   "])
def test_bare_ellipsis_at_any_indent(line):
    assert find_elision(line) is not None


@pytest.mark.parametrize("line", [
    "    def f(self) -> int: ...",     # one-line stub keeps the code on the line
    "    x: str = Field(...)",
    "    arr = a[..., 0]",
    "    result = call(...)",
])
def test_inline_ellipsis_is_still_legitimate(line):
    assert find_elision(line) is None


def test_run13_old_text_is_refused_by_the_executor(tmp_path):
    """End to end: the shape that corrupted booking_methods.py is rejected."""
    from besser.spec_driven_agent.tool_executor import ToolExecutor
    (tmp_path / "m.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
    ex = ToolExecutor(str(tmp_path))
    res = ex._modify_file({"path": "m.py", "old_text": RUN13_OLD_TEXT,
                           "new_text": "a = 1\n...\nb = 2\n"})
    assert "error" in res
    assert "abbreviates" in res["error"]
    assert (tmp_path / "m.py").read_text(encoding="utf-8") == "a = 1\nb = 2\n"
