"""The requirements ledger is built from the requirements the editor sends.

The modeling assistant extracts the user's requirements before it models them,
and the editor records each as a RequirementArtifact in the project trace. The
run judges those same requirements instead of extracting its own, so every
verdict names the trace artifact (``ref``) the requirement is linked from.
"""

import json

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning import requirements_ledger as ledger
from besser.spec_driven_agent.planning.specification import MAX_REQUIREMENTS
from besser.spec_driven_agent.providers.llm_client import UsageTracker

SPEC = "Every room is identified by its room number. Guests never exceed the rooms' capacity. Use FastAPI."
LATER = (
    f"\n\n{ledger.LATER_REQUEST_MARKER}\n\n"
    "This later request takes precedence where it changes earlier requirements.\n\n"
    "Bookings can also be cancelled."
)
STATED = [
    {"id": "call-1:R1", "text": "The system manages rooms", "kind": "data"},
    {"id": "call-1:R2", "text": "A room is identified by its number", "kind": "uniqueness"},
    {"id": "call-1:R3", "text": "Guests never exceed the rooms' capacity", "kind": "rule"},
    {"id": "call-1:R4", "text": "The backend uses FastAPI", "kind": "technical"},
]


class _MockClient:
    """No ``_client``: the test double the harness must never call out from."""

    model = "mock-model"
    usage = UsageTracker("mock-model")

    def chat(self, system, messages, tools):
        raise AssertionError("a mock client must never be called")


class _ToolClient:
    """Answers each forced tool call with the next canned payload."""

    model = "tool-model"

    def __init__(self, payloads):
        self.usage = UsageTracker("tool-model")
        self._client = object()  # a real provider, to the ledger
        self.payloads = list(payloads)

    def chat(self, system, messages, tools, force_tool=None, model_override=None):
        block = type("B", (), {"type": "tool_use", "name": force_tool, "input": self.payloads.pop(0)})()
        return {"stop_reason": "tool_use", "content": [block]}


@pytest.fixture
def hotel_model():
    room = Class(name="Room")
    room.attributes = {Property(name="number", type=PrimitiveDataType("str"), is_id=True)}
    return DomainModel(name="Hotel", types={room})


def test_the_ledger_judges_the_behavioural_requirements_and_keeps_their_trace_id():
    requirements = ledger.stated_requirements_ledger(STATED, SPEC)
    assert requirements == [
        {"id": 1, "text": "A room is identified by its number", "kind": "uniqueness", "ref": "call-1:R2"},
        {"id": 2, "text": "Guests never exceed the rooms' capacity", "kind": "rule", "ref": "call-1:R3"},
    ]


def test_only_structure_and_stack_is_an_empty_ledger_not_a_failed_one():
    assert ledger.stated_requirements_ledger([STATED[0], STATED[3]], SPEC) == []


def test_a_request_with_later_messages_is_extracted_instead():
    """The assistant extracted from the first message; the follow-up states more."""
    assert ledger.stated_requirements_ledger(STATED, SPEC + LATER) is None


def test_a_full_list_cannot_look_complete():
    stated = [{"id": f"r{i}", "text": f"rule {i}", "kind": "rule"} for i in range(MAX_REQUIREMENTS)]
    requirements = ledger.stated_requirements_ledger(stated, SPEC)
    assert len(requirements) == MAX_REQUIREMENTS + 1
    assert requirements[-1]["kind"] == "verification"


def test_every_verdict_names_its_trace_id():
    requirements = ledger.stated_requirements_ledger(STATED, SPEC)
    client = _ToolClient([{"verdicts": [
        {"id": 1, "status": "implemented", "evidence": "app/models.py: number = Column(String, unique=True)"},
    ]}])
    verdicts = ledger.judge_coverage(requirements, "### app/models.py\ncode", client, original_spec=SPEC)
    assert [(v["ref"], v["status"]) for v in verdicts] == [("call-1:R2", "implemented"),
                                                           ("call-1:R3", "unverified")]
    assert [v["ref"] for v in ledger.scope_to_output(verdicts, has_frontend=False)] == ["call-1:R2", "call-1:R3"]


def test_an_extracted_ledger_has_no_trace_id():
    verdicts = ledger.judge_coverage([{"id": 1, "text": "a", "kind": "rule"}], "### x.py\ncode",
                                     _ToolClient([{"verdicts": [{"id": 1, "status": "missing"}]}]))
    assert "ref" not in verdicts[0]


def test_the_run_uses_the_editor_requirements_without_extracting(hotel_model, tmp_path, monkeypatch):
    def no_extraction(*args):
        raise AssertionError("the editor's requirements must not be extracted again")

    monkeypatch.setattr(ledger, "extract_requirements", no_extraction)
    orch = LLMOrchestrator(llm_client=_MockClient(), domain_model=hotel_model, output_dir=str(tmp_path),
                           stated_requirements=STATED)
    planner_text = orch._planner_instructions(SPEC)
    assert "R1. A room is identified by its number" in planner_text
    assert "The backend uses FastAPI" not in planner_text.split(SPEC, 1)[1]
    orch._save_recipe("build it", elapsed=1.0)
    recipe = json.loads((tmp_path / ".besser_recipe.json").read_text(encoding="utf-8"))
    assert recipe["requirements_source"] == "editor"


def test_a_follow_up_request_falls_back_to_extraction(hotel_model, tmp_path, monkeypatch):
    extracted = [{"id": 1, "text": "Bookings can be cancelled", "kind": "action"}]
    monkeypatch.setattr(ledger, "extract_requirements", lambda instructions, client: list(extracted))
    orch = LLMOrchestrator(llm_client=_MockClient(), domain_model=hotel_model, output_dir=str(tmp_path),
                           stated_requirements=STATED)
    orch._planner_instructions(SPEC + LATER)
    assert orch._requirements == extracted
    assert orch._requirements_source == "extracted"


def test_without_editor_requirements_the_run_extracts_as_before(hotel_model, tmp_path, monkeypatch):
    extracted = [{"id": 1, "text": "Room numbers are unique", "kind": "uniqueness"}]
    monkeypatch.setattr(ledger, "extract_requirements", lambda instructions, client: list(extracted))
    orch = LLMOrchestrator(llm_client=_MockClient(), domain_model=hotel_model, output_dir=str(tmp_path))
    orch._planner_instructions(SPEC)
    assert orch._requirements == extracted
    assert orch._requirements_source == "extracted"
