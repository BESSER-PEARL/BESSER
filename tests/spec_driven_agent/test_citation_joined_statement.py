"""A statement the judge quoted on one line locates in its formatted, multi-line form.

Live evidence, r13-p81: the first judge said implemented for all 8 requirements and 6
of the 8 citations failed. Two natural forms in ``backend/routers/risk.py``:
- an ``if`` header the formatter wrapped in parentheses over four lines, quoted as
  ``if a or b:``; it was a near match, since a header alone does not parse;
- an f-string split by the formatter into two implicitly joined pieces, ``"...rated "``
  then ``f"{assessment.level} ..."``, quoted as one f-string.
An invented or changed condition, or a partial one, still fails.
"""

from besser.spec_driven_agent.planning import requirements_ledger as ledger

RISK = "backend/routers/risk.py"
SOURCE = '''LEVEL_ORDER = {"low": 1, "medium": 2, "high": 3}


def warnings_for(database, contacts):
    warnings = []
    for account in contacts:
        assessment = latest_assessment(database, account.id)
        if (
            assessment is None
            or LEVEL_ORDER.get(assessment.level, 0) < LEVEL_ORDER["medium"]
        ):
            continue
        warnings.append(
            {
                "warning": f"{account.handle} ({account.platform}) has interacted with your accounts and is rated "
                f"{assessment.level} risk (score {assessment.concernScore}).",
            }
        )
    return warnings
'''
HEADER = 'if assessment is None or LEVEL_ORDER.get(assessment.level, 0) < LEVEL_ORDER["medium"]:'
FSTRING = ('"warning": f"{account.handle} ({account.platform}) has interacted with your accounts and is rated '
           '{assessment.level} risk (score {assessment.concernScore}).",')


def _status(tmp_path, line, kind="action"):
    path = tmp_path / RISK
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SOURCE, encoding="utf-8")
    verdict = {"id": 8, "text": "Warn users about accounts rated medium risk or above", "kind": kind,
               "status": "implemented", "evidence": [{"path": RISK, "line": line}], "note": ""}
    return ledger.verify_evidence([verdict], str(tmp_path))[0]


def test_a_wrapped_if_header_quoted_on_one_line_locates(tmp_path):
    assert _status(tmp_path, HEADER)["status"] == "implemented"
    assert _status(tmp_path, HEADER, kind="rule")["status"] == "implemented"


def test_a_split_f_string_quoted_as_one_locates(tmp_path):
    assert _status(tmp_path, FSTRING)["status"] == "implemented"


def test_a_changed_or_partial_statement_still_fails(tmp_path):
    for line in (HEADER.replace('"medium"', '"high"'),
                 'if assessment is None or LEVEL_ORDER.get(assessment.level, 0) < 2:',
                 "if assessment is None:",
                 FSTRING.replace("is rated", "was rated")):
        assert _status(tmp_path, line)["status"] == "unverified", line


GRADES = "backend/grades.py"
GRADES_SOURCE = '''def grade(score):
    return "F" if score < 50 else compute_grade(
        score)


def check(items):
    for item in items:
        if (
            item.price is None
            or item.price < 0
        ):
            # negative prices are not allowed
            raise ValueError("bad price")
    return items
'''


def _grades(tmp_path, line):
    path = tmp_path / GRADES
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(GRADES_SOURCE, encoding="utf-8")
    verdict = {"id": 1, "text": "Grades below 50 fail", "kind": "rule", "status": "implemented",
               "evidence": [{"path": GRADES, "line": line}], "note": ""}
    return ledger.verify_evidence([verdict], str(tmp_path))[0]


def test_a_one_letter_literal_is_not_a_string_seam(tmp_path):
    """Review 11: the seam rule deleted any "F", "U" or "rb" literal from the quote and the
    source alike, so a changed or dropped literal verified."""
    assert _grades(tmp_path, 'return "F" if score < 50 else compute_grade(score)')["status"] == "implemented"
    for line in ('return "U" if score < 50 else compute_grade(score)',
                 'return "rb" if score < 50 else compute_grade(score)',
                 "return if score < 50 else compute_grade(score)"):
        assert _grades(tmp_path, line)["status"] == "unverified", line


def test_a_wrapped_header_followed_by_a_comment_ends_at_its_colon(tmp_path):
    """Review 11: the header span ended on the comment line, read as "a comment"."""
    assert _grades(tmp_path, "if item.price is None or item.price < 0:")["status"] == "implemented"
