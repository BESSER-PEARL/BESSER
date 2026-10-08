"""Tests for published GovernanceDSL parsing in the WME backend."""
from importlib import import_module
from importlib.metadata import version
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("governancedsl")

from besser.generators.docker_compose.docker_compose_generator import (  # noqa: E402
    _a2a_descriptor,
    _a2a_descriptor_from_tags,
)
from besser.utilities.web_modeling_editor.backend.services.governance import govdsl_runtime  # noqa: E402

from besser.utilities.web_modeling_editor.backend.services.exceptions import (  # noqa: E402
    GovernanceDslValidationError,
)
from besser.utilities.web_modeling_editor.backend.services.governance.govdsl_runtime import (  # noqa: E402
    summarize_governance,
)

_VOTING_DSL = """\
// WME-generated header
Scopes:
    Tasks :
        mergeTask
Participants:
    Individuals :
        (Agent) Coder {
            confidence : 0.8
        },
        (Agent) Reviewer {
            confidence : 0.6
        }
MajorityPolicy mergePolicy {
    Scope: mergeTask
    DecisionType as BooleanDecision
    Participant list : Coder, Reviewer
    Parameters:
        ratio : 0.5
}
"""


def test_published_namespaced_package_is_used():
    assert version("governancedsl") == "0.1.1"

    assert import_module("governancedsl.grammar.govdslLexer")
    assert import_module("governancedsl.grammar.govdslParser")
    assert import_module("governancedsl.grammar.PolicyCreationListener")
    assert import_module("governancedsl.grammar.govErrorListener")
    assert import_module("governancedsl.metamodel.governance")


def test_empty_input_returns_none():
    assert summarize_governance("") is None
    assert summarize_governance(" \n ") is None


def test_majority_policy_surfaces_participants_and_ratio():
    result = summarize_governance(_VOTING_DSL)

    assert result["policy_type"] == "MajorityPolicy"
    assert result["ratio"] == 0.5
    assert {participant["name"] for participant in result["participants"]} == {
        "Coder",
        "Reviewer",
    }

    confidences = {
        participant["name"]: participant["confidence"]
        for participant in result["participants"]
    }
    assert confidences == {"Coder": 0.8, "Reviewer": 0.6}


def test_malformed_dsl_raises_syntax_error():
    with pytest.raises(
        GovernanceDslValidationError,
        match=r"Governance DSL syntax error: line \d+:\d+",
    ):
        summarize_governance("MajorityPolicy broken {{{")


def test_invalid_ratio_raises_validation_error():
    invalid_dsl = _VOTING_DSL.replace("ratio : 0.5", "ratio : 1.5")

    with pytest.raises(GovernanceDslValidationError, match="ratio"):
        summarize_governance(invalid_dsl)

class _OrderedParticipants(set):
    """Supply a chosen iteration order for the same GovernanceDSL participant set."""

    def __init__(self, participants):
        self._ordered = tuple(participants)
        super().__init__(self._ordered)

    def __iter__(self):
        return iter(self._ordered)


def test_policy_summary_preserves_metadata_in_canonical_order():
    from governancedsl.metamodel.governance import Agent, Human, Role

    policy = govdsl_runtime._parse_policies(govdsl_runtime._strip_comments(_VOTING_DSL))[0]
    roles = {Role("ZRole"), Role("ARole")}
    participants = [
        Role("Shared"),
        Human("Shared", roles=roles),
        Agent("Shared", confidence=0.8, roles=roles),
        Agent("Reviewer", confidence=0.6),
    ]
    expected = {
        "policy_type": "MajorityPolicy",
        "ratio": 0.5,
        "decision_type": "BooleanDecision",
        "requires_human": True,
        "participants": [
            {"name": "Reviewer", "kind": "agent", "confidence": 0.6, "roles": []},
            {"name": "Shared", "kind": "agent", "confidence": 0.8, "roles": ["ARole", "ZRole"]},
            {"name": "Shared", "kind": "human", "confidence": None, "roles": ["ARole", "ZRole"]},
            {"name": "Shared", "kind": "role", "confidence": None, "roles": []},
        ],
    }
    for order in (participants, list(reversed(participants))):
        original = _OrderedParticipants(order)
        policy.participants = original
        assert govdsl_runtime._summarize_policy(policy) == expected
        assert policy.participants is original
        assert list(policy.participants) == order


@pytest.mark.parametrize("tagged", [False, True], ids=["legacy", "tagged"])
def test_governance_descriptor_has_canonical_order(monkeypatch, tagged):
    policy = govdsl_runtime._parse_policies(govdsl_runtime._strip_comments(_VOTING_DSL))[0]
    participants = sorted(policy.participants, key=lambda participant: participant.name)
    monkeypatch.setattr(govdsl_runtime, "_parse_policies", lambda text: [policy])
    summaries, descriptors = [], []
    for order in (participants, list(reversed(participants))):
        original = _OrderedParticipants(order)
        policy.participants = original
        summary = summarize_governance(_VOTING_DSL)
        summaries.append(summary)
        owner = SimpleNamespace(name="Reviewer", states=[], _governance=[{
            **summary, "producers": ["Coder"],
        }])
        builder = _a2a_descriptor
        if tagged:
            owner._a2a = {"outbound": [], "inbound": [{"peer": "Coder", "flow": "merge"}]}
            builder = _a2a_descriptor_from_tags
        descriptor = builder(owner, {"coder", "reviewer"}, self_service="reviewer")
        descriptors.append(descriptor)
        governance = descriptor["governance"]
        assert governance["weights"] == {"coder": 0.8, "reviewer": 0.6}
        assert governance["producer_services"] == ["coder"]
        assert governance["owner_votes"] is True
        assert governance["owner_produces"] is False
        assert governance["requires_human"] is False
        assert policy.participants is original
        assert list(policy.participants) == order
    assert summaries[0] == summaries[1]
    # Preserve mapping insertion order in this comparison, as the template does.
    assert json.dumps(descriptors[0]) == json.dumps(descriptors[1])
    assert list(descriptors[0]["governance"]["weights"]) == ["coder", "reviewer"]
