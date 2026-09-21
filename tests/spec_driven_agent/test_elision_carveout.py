"""A NEW ellipsis is an abbreviation even when old_text also had one.

Run 36e9c8a6 (2026-09-18) let 6 elided edits through and refused 0. The guard
read ``if new_elision and not find_elision(old_text)``: any ellipsis anywhere
in the quoted region excused every ellipsis in the replacement. The real edit
below quoted a stub body (bare ``...`` on its own line) and wrote back
``success = bill.register...``.
"""
from besser.spec_driven_agent.edit_apply import elided_lines, find_elision

# Verbatim from the run: old_text carries a stub body, new_text abbreviates.
OLD_TEXT = '''    def registerPayment(self, amount: float) -> bool:
        ...
'''
NEW_TEXT = '''    def registerPayment(self, amount: float) -> bool:
        success = bill.register...
'''


def _old_guard(old_text, new_text):
    return bool(find_elision(new_text)) and not find_elision(old_text)


def _new_guard(old_text, new_text):
    found = find_elision(new_text)
    return bool(found) and found[1].strip() not in elided_lines(old_text)


def test_the_shipped_guard_missed_this_edit():
    assert _old_guard(OLD_TEXT, NEW_TEXT) is False


def test_the_fixed_guard_refuses_it():
    assert _new_guard(OLD_TEXT, NEW_TEXT) is True


def test_preserving_an_ellipsis_the_file_already_had_is_still_allowed():
    kept = '''    def registerPayment(self, amount: float) -> bool:
        ...

    def cancel(self) -> None:
        return None
'''
    assert _new_guard(OLD_TEXT, kept) is False


def test_indentation_changes_do_not_break_the_match():
    """The comparison strips, so re-indenting a preserved stub is not a new elision."""
    assert _new_guard(OLD_TEXT, "class X:\n            ...\n") is False


def test_a_clean_edit_is_untouched():
    assert _new_guard(OLD_TEXT, "    def registerPayment(self) -> bool:\n        return True\n") is False


def test_elided_lines_collects_every_shape():
    text = "a = foo...\n...\n# rest of the file unchanged ...\nb = 1\n"
    assert len(elided_lines(text)) == 3
    assert "b = 1" not in elided_lines(text)


def test_elided_lines_is_empty_for_ordinary_code():
    assert elided_lines("def f(x=...):\n    return Body(...)\n") == set()
