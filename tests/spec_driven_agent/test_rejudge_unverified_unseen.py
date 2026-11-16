"""An unverified verdict about an unseen file gets that file, and keeps the note of the judge that saw it.

Live evidence, r12-p81 (R1): the judge digest left out backend/risk_service.py and
the judge, as its prompt says for code it was not shown, answered ``unverified``
("risk_service.py, which was not provided"). The rule that adds a named omitted
file to the focused round applied to partial/missing only, so the re-judge never
saw the file; and when the re-judge's answer was not a verified implemented, it
was dropped, and the user-facing blocker kept the first judge's stale note.
"""

from besser.spec_driven_agent.planning import requirements_ledger as ledger
from tests.spec_driven_agent.test_rejudge_digest_gaps import OMITTED_NOTE, _judge, orch  # noqa: F401

UNVERIFIED = {"status": "unverified", "evidence": "", "note": OMITTED_NOTE}
STILL_UNVERIFIED = {"status": "unverified", "evidence": "",
                    "note": "detect() matches keywords but nothing flags repeated messages"}


def test_an_unverified_verdict_naming_an_omitted_file_is_judged_with_it(orch, monkeypatch):  # noqa: F811
    digests = _judge(monkeypatch, UNVERIFIED, STILL_UNVERIFIED)
    orch._collect_requirement_issues()
    assert len(digests) == 2 and "### backend/risk_service.py" in digests[1]


def test_the_blocker_carries_the_note_of_the_judge_that_saw_the_file(orch, monkeypatch):  # noqa: F811
    _judge(monkeypatch, UNVERIFIED, STILL_UNVERIFIED)
    issues = orch._collect_requirement_issues()
    (blocker,) = [issue for issue in issues if issue.startswith("requirement unverified:")]
    assert "nothing flags repeated messages" in blocker and "backend/risk_service.py shown" in blocker
    assert "omitted" not in blocker
    # The status stands: a focused digest that omits most of the app proves no absence.
    assert orch._requirement_verdicts[0]["status"] == "unverified"
    # Kept for the next validation of the same source, which reuses the cached verdict.
    assert orch._collect_requirement_issues() == issues


def test_the_digest_still_omits_the_service_in_this_fixture(orch):  # noqa: F811
    assert "backend/risk_service.py" in ledger.build_app_digest_and_gaps(orch.output_dir)[1]


def test_the_rejudge_message_carries_its_own_evidence_and_names_only_files_shown(orch, monkeypatch):  # noqa: F811
    """Review: the note named a focus path that resolved to no file, and was the re-judge's
    while the cited evidence and its reason were still the first judge's."""
    first = {"status": "implemented", "evidence": [{"path": "backend/services/detector.py",
                                                    "line": "def flag_aggressive(msg):"}],
             "note": "flag_aggressive() flags aggressive messages"}
    second = {"status": "implemented", "evidence": [{"path": "backend/risk_service.py",
                                                     "line": "return [m for m in sent if is_aggressive(m)]"}],
              "note": "detect() filters aggressive messages"}
    _judge(monkeypatch, first, second)
    (blocker,) = [issue for issue in orch._collect_requirement_issues()
                  if issue.startswith("requirement unverified:")]
    assert "detector.py" not in blocker
    assert "is_aggressive" in blocker and "detect() filters aggressive messages" in blocker
