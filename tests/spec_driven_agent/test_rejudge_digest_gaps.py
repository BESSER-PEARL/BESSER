"""A partial/missing verdict about a file the judge digest left out is judged once more.

Live run r11-p81 (Claude Sonnet 5.5): the judge returned ``partial`` for R2 with the
note "the actual detection/analysis logic that populates these signals is in the
omitted backend/risk_service.py". The 200k digest had listed that file under
"Omitted files": seven LLM-edited routers outranked the one hand-written service. The
focused re-judge retried only ``unverified`` verdicts, so the partial went to a Phase 3
repair round (6 turns, about $0.21) that changed nothing; a later judge said implemented.

Now a partial or missing verdict that names a file the digest omitted or truncated
joins the one focused round per judged source, with a digest holding that file. As in
the citation repair, only a verified ``implemented`` replaces the verdict.
"""
import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning import requirements_ledger as ledger
from besser.spec_driven_agent.providers.llm_client import UsageTracker

SPEC = "Detect aggressive messages sent to an account."
REQUIREMENT = {"id": 1, "text": "The system detects aggressive messages", "kind": "action"}
SERVICE = '''import re

AGGRESSIVE_RE = re.compile(r"kill|hate you", re.I)


def detect(sent):
    return [m for m in sent if AGGRESSIVE_RE.search(m.content or "")]
''' + "".join(f"\n\ndef score_{i}(signal):\n    return signal.weight * {i}\n" for i in range(40))
LOGIC = 'return [m for m in sent if AGGRESSIVE_RE.search(m.content or "")]'
OMITTED_NOTE = ("The signals are stored, but the actual detection logic is in the omitted "
                "backend/risk_service.py.")
# The two edited routers (about 6k) fit; the service module (about 2.5k) then falls out.
DIGEST_CHARS = 8_000


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


@pytest.fixture
def orch(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "_DIGEST_MAX_TOTAL_CHARS", DIGEST_CHARS)
    message = Class(name="Message")
    message.attributes = {Property(name="content", type=PrimitiveDataType("str"))}
    _write(tmp_path, "backend/pydantic_classes.py", "class MessageCreate:\n    content: str\n")
    _write(tmp_path, "backend/sql_alchemy.py", "class Message:\n    content = None\n")
    for name in ("message", "account"):
        body = "".join(f"def handler_{name}_{i}(item):\n    return item.value_{i} + {i}\n" for i in range(50))
        _write(tmp_path, f"backend/routers/{name}.py", body)
    _write(tmp_path, "backend/risk_service.py", SERVICE)
    orchestrator = LLMOrchestrator(llm_client=_Client(), domain_model=DomainModel(name="Risk", types={message}),
                                   output_dir=str(tmp_path), enable_tracing=False, enable_checkpointing=False)
    orchestrator._instructions = SPEC
    orchestrator._generator_used = "generate_fastapi_backend"
    orchestrator._requirements = [dict(REQUIREMENT)]
    return orchestrator


def _judge(monkeypatch, *answers):
    """Replace the judge with scripted verdicts; returns the digests it was sent."""
    digests, script = [], list(answers)

    def judge(requirements, digest, client, **kwargs):
        digests.append(digest)
        return [dict(REQUIREMENT, **script.pop(0))]

    monkeypatch.setattr(ledger, "judge_coverage", judge)
    return digests


PARTIAL = {"status": "partial", "evidence": "", "note": OMITTED_NOTE}
IMPLEMENTED = {"status": "implemented", "evidence": [{"path": "backend/risk_service.py", "line": LOGIC}],
               "note": "detect() filters messages with AGGRESSIVE_RE"}


def test_the_workspace_overflows_the_digest_and_omits_the_service(orch):
    digest, gaps = ledger.build_app_digest_and_gaps(orch.output_dir)
    assert "backend/risk_service.py" in gaps and "### backend/risk_service.py" not in digest
    assert "Omitted files: backend/risk_service.py" in digest
    assert ledger.named_digest_gaps({"note": OMITTED_NOTE, "inspection_paths": []}, gaps) == [
        "backend/risk_service.py"]


def test_a_partial_about_an_omitted_file_is_judged_once_more_with_that_file(orch, monkeypatch):
    digests = _judge(monkeypatch, PARTIAL, IMPLEMENTED)
    issues = orch._collect_requirement_issues()
    assert len(digests) == 2
    assert "### backend/risk_service.py" not in digests[0]
    assert "### backend/risk_service.py" in digests[1]
    assert not any(issue.startswith(("requirement", "requirement partial")) for issue in issues), issues
    assert orch._requirement_verdicts[0]["status"] == "implemented"


def test_a_second_partial_keeps_the_blocker(orch, monkeypatch):
    digests = _judge(monkeypatch, PARTIAL, PARTIAL)
    issues = orch._collect_requirement_issues()
    assert len(digests) == 2
    assert any(issue.startswith("requirement partial:") for issue in issues)


def test_a_partial_naming_no_omitted_file_gets_no_extra_call(orch, monkeypatch):
    digests = _judge(monkeypatch, {"status": "partial", "evidence": "",
                                   "note": "the detection ignores replies; backend/routers/message.py"})
    issues = orch._collect_requirement_issues()
    assert len(digests) == 1
    assert any(issue.startswith("requirement partial:") for issue in issues)


def test_the_round_is_not_repeated_on_the_same_source(orch, monkeypatch):
    digests = _judge(monkeypatch, PARTIAL, PARTIAL)
    first = orch._collect_requirement_issues()
    assert orch._collect_requirement_issues() == first
    assert len(digests) == 2


def test_no_extra_call_when_the_cost_gate_refuses(orch, monkeypatch):
    digests = _judge(monkeypatch, PARTIAL, IMPLEMENTED)
    gate = iter([True])  # the full judgment is allowed, nothing after it
    monkeypatch.setattr(orch, "_verification_call_allowed", lambda: next(gate, False))
    issues = orch._collect_requirement_issues()
    assert len(digests) == 1
    assert any(issue.startswith("requirement partial:") for issue in issues)
