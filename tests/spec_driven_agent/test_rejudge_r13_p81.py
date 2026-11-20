"""The citation re-judge sees every requirement's files and keeps what the judge said.

Live evidence, r13-p81: one false blocker, R8, "the cited evidence '' does not verify it
(... not only a decorator)". The first judge said implemented for all 8 requirements and
6 citations failed. The focused re-judge took the first ten focus paths before removing
repeats: all ten were ``backend/risk_service.py``, so R8's ``backend/routers/risk.py`` and
``frontend/src/pages/Home.jsx`` were never shown. Its "unverified" answer (no citation,
inspection path risk.py) was merged onto the RAW first verdict, giving "implemented" with
an empty citation, which the verifier then failed as a decorator-only, absent citation.
"""
import json

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning import requirements_ledger as ledger
from besser.spec_driven_agent.providers.llm_client import UsageTracker

SERVICE = "backend/risk_service.py"
RISK = "backend/routers/risk.py"
HOME = "frontend/src/pages/Home.jsx"
OTHER = "backend/routers/other.py"


class _Usage:
    input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens = 100, 10, 0, 0


class _Client:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self):
        self.usage = UsageTracker("mock-model")

    def chat(self, **kwargs):  # pragma: no cover - judge_coverage is replaced
        raise AssertionError("no LLM call expected")


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


REQS = [{"id": rid, "text": f"Requirement {rid}", "kind": "action"} for rid in (1, 2, 3, 4, 5, 8)]


@pytest.fixture
def orch(tmp_path):
    _write(tmp_path, "backend/pydantic_classes.py", "class A:\n    x: int\n")
    _write(tmp_path, SERVICE, "".join(f"def score_{i}(s):\n    return s * {i}\n" for i in range(40)))
    _write(tmp_path, RISK, "def warnings(db):\n    warnings = []\n    warnings.append(db.first())\n"
                           "    return warnings\n")
    _write(tmp_path, HOME, "export default function Home({ w }) {\n  return <strong>{w.warning}</strong>;\n}\n")
    _write(tmp_path, OTHER, "def other():\n    return 1\n")
    orchestrator = LLMOrchestrator(llm_client=_Client(), domain_model=DomainModel(name="R", types={Class(name="A")}),
                                   output_dir=str(tmp_path), enable_checkpointing=False)
    orchestrator._instructions = "Warn users about risky accounts."
    orchestrator._generator_used = "generate_fastapi_backend"
    orchestrator._requirements = [dict(item) for item in REQS]
    return orchestrator


def _first_round():
    """Implemented x6, each citation misquoted; R1-R5 cite the service six times each."""
    verdicts = [{**req, "status": "implemented", "note": "first",
                 "evidence": [{"path": SERVICE, "line": f"return s * {req['id']} + 1"} for _ in range(6)]}
                for req in REQS[:5]]
    return verdicts + [{**REQS[5], "status": "implemented", "note": "first",
                        "evidence": [{"path": RISK, "line": "warnings.append(db.last())"},
                                     {"path": HOME, "line": "<em>{w.warning}</em>"}]}]


def _judge(monkeypatch, orch, *rounds):
    """Scripted judge rounds: each returns verdicts for the requirements it is asked about."""
    digests, script = [], list(rounds)

    def judge(requirements, digest, client, **kwargs):
        digests.append(digest)
        client.usage.record(_Usage())
        answer = script.pop(0)
        asked = {item["id"] for item in requirements}
        return [verdict for verdict in answer if verdict["id"] in asked]

    monkeypatch.setattr(ledger, "judge_coverage", judge)
    return digests


R8_UNSEEN = {**REQS[5], "status": "unverified", "evidence": "", "note": "risk.py omitted",
             "inspection_paths": [RISK]}


def test_the_focused_round_shows_every_requirements_files(orch, monkeypatch):
    digests = _judge(monkeypatch, orch, _first_round(), [R8_UNSEEN])
    orch._collect_requirement_issues()
    assert len(digests) == 2
    assert f"### {RISK}" in digests[1] and f"### {HOME}" in digests[1] and f"### {SERVICE}" in digests[1]


def test_an_unverified_rejudge_answer_is_merged_onto_the_checked_verdict(orch, monkeypatch):
    _judge(monkeypatch, orch, _first_round(), [R8_UNSEEN])
    issues = orch._collect_requirement_issues()
    r8 = next(item for item in orch._requirement_verdicts if item["id"] == 8)
    assert r8["status"] == "unverified" and r8["inspection_paths"] == [RISK]
    assert r8.get(ledger.EVIDENCE_UNLOCATED) != ledger.CITATION_ABSENT
    (message,) = [issue for issue in issues if "R8" in issue]
    assert "decorator" not in message and "risk.py omitted" in message


def test_an_empty_citation_is_no_citation():
    [verdict] = ledger.verify_evidence([{**REQS[5], "status": "implemented", "evidence": "", "note": ""}], ".")
    assert verdict["status"] == "unverified" and "no source line is cited" in verdict["note"]
    assert ledger.EVIDENCE_UNLOCATED not in verdict


def test_a_file_the_focused_digest_left_out_gets_one_more_round(orch, monkeypatch):
    names_other = {**R8_UNSEEN, "note": f"the warning text is built in {OTHER}, which was not shown"}
    digests = _judge(monkeypatch, orch, _first_round(), [names_other], [names_other], [names_other])
    orch._collect_requirement_issues()
    assert len(digests) == 3  # one extra round per revision, not more
    assert f"### {OTHER}" not in digests[1] and f"### {OTHER}" in digests[2]
    # An implemented answer still has to pass the citation check.
    real = {**REQS[5], "status": "implemented", "note": "found", "evidence": [{"path": OTHER, "line": "return 1"}]}
    orch2_digests = _judge(monkeypatch, orch, _first_round(), [names_other], [real])
    orch._requirement_judgments.clear()
    orch._requirement_evidence_retries.clear()
    orch._collect_requirement_issues()
    assert len(orch2_digests) == 3
    assert next(v for v in orch._requirement_verdicts if v["id"] == 8)["status"] == "implemented"


def test_the_judge_trace_says_how_many_citations_verified_and_why_the_rest_failed(orch, monkeypatch):
    _judge(monkeypatch, orch, _first_round(), [R8_UNSEEN])
    orch._collect_requirement_issues()
    events = [json.loads(line) for line in (orch._trace.path and open(orch._trace.path, encoding="utf-8"))]
    judge = next(e["payload"] for e in events if e["event"] == "llm_call" and e["payload"]["purpose"] == "requirements_judge")
    assert judge["citations"] == {"claimed": 6, "verified": 0}
    assert set(judge["failures"]) == {"1", "2", "3", "4", "5", "8"} or set(judge["failures"]) == {1, 2, 3, 4, 5, 8}
    assert judge["model"] == "mock-model"
    rejudge = next(e["payload"] for e in events
                   if e["event"] == "llm_call" and e["payload"]["purpose"] == "requirements_rejudge")
    assert "citations" in rejudge and rejudge["model"] == "mock-model"


def test_a_schema_file_the_focused_digest_showed_is_not_left_out(orch, monkeypatch):
    """Review 11: a note naming pydantic_classes.py, which every focused digest shows,
    bought a third round."""
    schema_note = {**R8_UNSEEN, "note": "backend/pydantic_classes.py has no warning field"}
    digests = _judge(monkeypatch, orch, _first_round(), [schema_note], [schema_note])
    orch._collect_requirement_issues()
    assert len(digests) == 2


def test_a_kept_first_citation_keeps_the_reason_it_failed(orch, monkeypatch):
    """Review 11: with no citation from the re-judge the first one is kept, but the reason
    the check gave for it was dropped."""
    _judge(monkeypatch, orch, _first_round(), [R8_UNSEEN])
    issues = orch._collect_requirement_issues()
    (message,) = [issue for issue in issues if "R8" in issue]
    assert "warnings.append(db.last())" in message and "citation 1 (backend/routers/risk.py)" in message


# A no-op validator cited for a rule the field's declared type enforces is refuted on the
# cited line alone. The refuted verdict is judged once more with the file shown whole and
# told why: citing the enforcing declaration leaves it not confirmed, not a blocker.
SCHEMA = "backend/schemas.py"
REFERENCE_RULE = {"id": 9, "text": "An order's reference must be a valid order code", "kind": "validation"}
NO_OP = {**REFERENCE_RULE, "status": "implemented", "note": "validated",
         "evidence": [{"path": SCHEMA, "line": "def keep_reference(cls, v):"}]}


@pytest.fixture
def refuted(orch, tmp_path):
    _write(tmp_path, SCHEMA, "from pydantic import BaseModel, field_validator\nfrom .codes import OrderCode\n\n"
                             "class OrderIn(BaseModel):\n    reference: OrderCode\n\n"
                             "    @field_validator('reference')\n    def keep_reference(cls, v):\n        return v\n")
    orch._requirements = [dict(REFERENCE_RULE)]
    return orch


def test_a_refuted_verdict_is_judged_again_with_its_file_and_the_reason(refuted, monkeypatch):
    seen = []
    declared = {**NO_OP, "evidence": [{"path": SCHEMA, "line": "reference: OrderCode"}]}

    def judge(requirements, digest, client, previous_verdicts=None, **kwargs):
        seen.append((digest, previous_verdicts))
        return [NO_OP] if len(seen) == 1 else [declared]

    monkeypatch.setattr(ledger, "judge_coverage", judge)
    issues = refuted._collect_requirement_issues()

    assert len(seen) == 2 and f"### {SCHEMA}" in seen[1][0] and "return v" in seen[1][0]
    assert "validator is a no-op" in seen[1][1][0]["note"]
    (message,) = [issue for issue in issues if "R9" in issue]
    assert message.startswith("requirement unverified:")


def test_a_rejudge_that_cites_the_no_op_validator_again_still_blocks(refuted, monkeypatch):
    digests = _judge(monkeypatch, refuted, [NO_OP], [NO_OP])
    issues = refuted._collect_requirement_issues()

    assert len(digests) == 2
    (message,) = [issue for issue in issues if "R9" in issue]
    assert message.startswith("requirement unsupported:")


# The focused re-judge's own answer about a verdict the first check could not confirm.
CODES = "backend/codes.py"
CHECK_LINE = "raise ValueError('an order code starts with ORD-')"


@pytest.fixture
def with_codes(refuted, tmp_path):
    _write(tmp_path, CODES, "def check_code(value):\n    if not value.startswith('ORD-'):\n"
                            f"        {CHECK_LINE}\n    return value\n")
    return refuted


def _r9(issues):
    (message,) = [issue for issue in issues if "R9" in issue]
    return message


@pytest.mark.parametrize("answer, prefix", [
    pytest.param({**REFERENCE_RULE, "status": "missing", "evidence": [], "note": "nothing checks the code"},
                 "requirement: ", id="missing"),
    pytest.param({**REFERENCE_RULE, "status": "partial", "evidence": [], "note": "only the type is declared"},
                 "requirement partial: ", id="partial"),
])
def test_a_refuted_verdict_takes_the_rejudges_missing_or_partial(refuted, monkeypatch, answer, prefix):
    _judge(monkeypatch, refuted, [NO_OP], [answer])

    message = _r9(refuted._collect_requirement_issues())

    assert message.startswith(prefix), message
    verdict = next(v for v in refuted._requirement_verdicts if v["id"] == 9)
    assert verdict["status"] == answer["status"] and not verdict.get(ledger.EVIDENCE_REFUTED)


def test_a_refuted_verdict_whose_rejudge_cites_working_code_is_implemented(with_codes, monkeypatch):
    real = {**NO_OP, "evidence": [{"path": CODES, "line": CHECK_LINE}]}
    _judge(monkeypatch, with_codes, [NO_OP], [real])

    assert not [issue for issue in with_codes._collect_requirement_issues() if "R9" in issue]
    assert next(v for v in with_codes._requirement_verdicts if v["id"] == 9)["status"] == "implemented"


@pytest.mark.parametrize("answer", [None, []], ids=["call failed", "no verdict"])
def test_a_refuted_verdict_with_no_rejudge_answer_stays_a_blocker(refuted, monkeypatch, answer):
    rounds = iter([[NO_OP], answer])
    monkeypatch.setattr(ledger, "judge_coverage", lambda *args, **kwargs: next(rounds))

    assert _r9(refuted._collect_requirement_issues()).startswith("requirement unsupported:")


@pytest.mark.parametrize("first", [
    pytest.param({**NO_OP, "evidence": [{"path": SCHEMA, "line": "return v.upper()"}]}, id="citation not located"),
    pytest.param({**REFERENCE_RULE, "status": "unverified", "evidence": [], "note": "not shown",
                  "inspection_paths": [SCHEMA]}, id="judge answered unverified"),
])
@pytest.mark.parametrize("status", ["missing", "partial"])
def test_a_focused_missing_proves_no_absence_unless_the_citation_was_refuted(refuted, monkeypatch, first, status):
    """The focused digest omits most of the app: only a refuted citation's own file, shown whole, settles it."""
    answer = {**REFERENCE_RULE, "status": status, "evidence": [], "note": "nothing checks the code"}
    _judge(monkeypatch, refuted, [first], [answer])

    assert _r9(refuted._collect_requirement_issues()).startswith("requirement unverified:")
