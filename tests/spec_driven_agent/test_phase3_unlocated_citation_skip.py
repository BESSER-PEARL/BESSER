"""No fix attempt when the only blockers are citations the judge could not locate.

Live evidence (2026-11-14, claude-sonnet-5-5, run r9-p81): Phase 3 entered
with two blockers, both ``requirement unverified`` because a citation was not
found in its file, and the judge's citation-repair round had already run. The
fix agent has no tool that changes a citation, and on unchanged source the
next validation re-verifies the same cached verdict, so the attempt spent 3
turns and could not move either blocker. The verifier now marks such a
verdict, and Phase 3 ends with the blockers reported as they are.

Only when the cited code demonstrably EXISTS: a near match in the cited file,
or the line verbatim in another file. A citation of code that exists nowhere
keeps its fix attempt; the blocker says to implement behaviour that is absent.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, ValidationIssue
from besser.spec_driven_agent.planning import requirements_ledger as ledger
from besser.spec_driven_agent.providers.llm_client import UsageTracker

SERVICES = '''import re

AGGRESSIVE_RE = re.compile(r"kill|hate you", re.I)


def detect(sent):
    return [m for m in sent if AGGRESSIVE_RE.search(m.content or "")]
'''


class _Client:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self) -> None:
        self.usage = UsageTracker("mock-model")

    def chat(self, **kwargs):  # pragma: no cover - no test here calls the LLM
        raise AssertionError("no LLM call expected")


@pytest.fixture
def orch(simple_library_book_model, tmp_path):
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "services.py").write_text(SERVICES, encoding="utf-8")
    (tmp_path / "backend" / "other.py").write_text("def unrelated():\n    return 1\n", encoding="utf-8")
    return LLMOrchestrator(
        llm_client=_Client(), domain_model=simple_library_book_model,
        output_dir=str(tmp_path), enable_checkpointing=False,
        enable_tracing=False, auto_fix_issues=True,
    )


def _verdict(rid, *lines, path="backend/services.py"):
    return {"id": rid, "kind": "action", "text": f"Requirement {rid}", "status": "implemented",
            "evidence": [{"path": path, "line": line} for line in lines]}


REAL = 'return [m for m in sent if AGGRESSIVE_RE.search(m.content or "")]'
NEAR_MATCH = _verdict(1, "return [m for m in sent if AGGRESSIVE_RE.search(m.content)]")  # real line, misquoted
MISFILED = _verdict(4, REAL, path="backend/other.py")  # real line, wrong file
ABSENT = _verdict(5, "aggressive = detect_aggression(account)")  # exists nowhere
NOT_PROVING = _verdict(2, 'AGGRESSIVE_RE = re.compile(r"kill|hate you", re.I)')  # found; a declaration
VERIFIED = _verdict(3, REAL)


def _run(orch, verdicts, *, repaired=True, other_blockers=()):
    """Phase 3 over one validation whose requirement verdicts are ``verdicts``."""
    orch._requirement_verdicts = ledger.verify_evidence(verdicts, orch.output_dir)
    blockers = [ValidationIssue("blocker", message) for message in ledger.ledger_issues(orch._requirement_verdicts)]
    blockers += [ValidationIssue("blocker", message) for message in other_blockers]
    orch._requirement_evidence_retries = {"judged-source"} if repaired else set()
    with patch.object(orch, "_collect_validation_issues", return_value=blockers), \
            patch.object(orch, "_requirement_cache_key", return_value="judged-source"), \
            patch.object(orch, "_requirements_to_judge", return_value=[]), \
            patch.object(orch, "_create_snapshot"), \
            patch.object(orch, "_restore_snapshot", return_value=True), \
            patch.object(orch, "_workspace_revision", return_value="rev"), \
            patch.object(orch, "_repair_obligations_revision", return_value="obl"), \
            patch.object(orch, "_invoke_phase3_fix_loop", return_value=0) as fix:
        orch._run_phase3_repair_cycle()
    return fix.call_count, blockers


def test_the_verifier_marks_only_a_citation_it_could_not_locate(orch):
    near, misfiled, absent, not_proving, verified = ledger.verify_evidence(
        [NEAR_MATCH, MISFILED, ABSENT, NOT_PROVING, VERIFIED], orch.output_dir)
    assert near["status"] == "unverified" and near[ledger.EVIDENCE_UNLOCATED] == ledger.CITATION_MALFORMED
    assert misfiled["status"] == "unverified" and misfiled[ledger.EVIDENCE_UNLOCATED] == ledger.CITATION_MALFORMED
    assert absent["status"] == "unverified" and absent[ledger.EVIDENCE_UNLOCATED] == ledger.CITATION_ABSENT
    assert not_proving["status"] == "unverified" and ledger.EVIDENCE_UNLOCATED not in not_proving
    assert verified["status"] == "implemented" and ledger.EVIDENCE_UNLOCATED not in verified
    # A re-verified verdict carries no stale mark: the field is recomputed.
    [again] = ledger.verify_evidence([dict(VERIFIED, **{ledger.EVIDENCE_UNLOCATED: ledger.CITATION_MALFORMED})],
                                     orch.output_dir)
    assert ledger.EVIDENCE_UNLOCATED not in again


def test_one_absent_citation_makes_the_verdict_absent(orch):
    """Every unlocated citation counts, not only the first one the verifier reports."""
    mixed = _verdict(6, "return [m for m in sent if AGGRESSIVE_RE.search(m.content)]",
                     "aggressive = detect_aggression(account)")
    [verdict] = ledger.verify_evidence([mixed], orch.output_dir)
    assert "citation 1" in verdict["note"]
    assert verdict[ledger.EVIDENCE_UNLOCATED] == ledger.CITATION_ABSENT


def test_only_malformed_citations_after_their_repair_round_get_no_fix_attempt(orch):
    calls, blockers = _run(orch, [NEAR_MATCH, MISFILED, VERIFIED])
    assert calls == 0
    assert orch._phase3_exit_reason == "only unlocated requirement citations remain"
    reported = {issue.message for issue in orch._validation_issues}
    assert {issue.message for issue in blockers} <= reported


def test_a_citation_of_code_that_exists_nowhere_still_gets_an_attempt(orch):
    calls, _ = _run(orch, [NEAR_MATCH, ABSENT])
    assert calls == 1


def test_a_citation_not_yet_through_its_repair_round_still_gets_an_attempt(orch):
    calls, _ = _run(orch, [NEAR_MATCH], repaired=False)
    assert calls == 1


def test_any_other_blocker_still_gets_an_attempt(orch):
    calls, _ = _run(orch, [NEAR_MATCH], other_blockers=["action contract: POST /book/ returned 500"])
    assert calls == 1


def test_a_requirement_unverified_for_another_reason_still_gets_an_attempt(orch):
    """A located citation that proves nothing: editing the code can help there."""
    calls, _ = _run(orch, [NEAR_MATCH, NOT_PROVING])
    assert calls == 1


def test_a_malformed_citation_beside_a_fragment_the_agent_can_complete_gets_an_attempt(orch):
    """The fragment names its one line, which the fix agent may cite whole (validate_app citations)."""
    citable = _verdict(7, "return [m for m in sent if AGGRESSIVE_RE.search(m.content)]", "AGGRESSIVE_RE = re.compile(")
    [verdict] = ledger.verify_evidence([citable], orch.output_dir)
    assert verdict[ledger.EVIDENCE_UNLOCATED] == ledger.CITATION_MALFORMED
    assert verdict[ledger.CITED_FRAGMENT_LINES] == ["backend/services.py:3"]
    calls, _ = _run(orch, [citable])
    assert calls == 1
