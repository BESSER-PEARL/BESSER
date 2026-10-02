"""Tests for published GovernanceDSL parsing in the WME backend."""
from importlib import import_module
from importlib.metadata import version

import pytest

from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    GovernanceDslValidationError,
)
from besser.utilities.web_modeling_editor.backend.services.governance.govdsl_runtime import (
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