"""A constraint word counts as a whole word, and an enumeration enforces the values a requirement lists.

A 'validation' or 'rule' requirement needs an enforcing line only when its text
states a constraint. Matched as substrings, the words that decide this fired
inside ordinary words ("lesson", "information"). A value of an enumeration that
is a constraint word ("one of: positive, mixed, negative") still states the
rule, and the enumeration that implements it was reported as a blocker.
"""

import pytest

from besser.spec_driven_agent.planning import requirements_ledger as ledger

SCHEMAS = "backend/pydantic_classes.py"
_SCHEMAS_SOURCE = '''from enum import Enum
from pydantic import BaseModel


class Tone(Enum):
    Positive = "Positive"
    Mixed = "Mixed"
    Negative = "Negative"


class Stage(Enum):
    DRAFT = "DRAFT"
    IN_REVIEW = "IN_REVIEW"
    PUBLISHED = "PUBLISHED"


class ArticleCreate(BaseModel):
    title: str
    tone: Tone
    stage: Stage
    price: float
'''


@pytest.mark.parametrize("text", [
    "Each lesson must have a title",
    "Each customer must have contact information",
    "A class has a morning or afternoon slot",
    "Each article has a stage: draft, in review or published",
])
def test_an_ordinary_word_states_no_constraint(text):
    assert not ledger._demands_enforcement("validation", text)
    assert not ledger._demands_enforcement("rule", text)


@pytest.mark.parametrize("text", [
    # A listed value restricts the field to the list: a declaration does not.
    "Feedback tone must be one of: positive, mixed, negative",
    "Feedback tone must be one of positive, mixed or negative",
    "The state of a coupon is either valid or expired",
    "Record an entry with its date and outcome (match, mismatch, pending)",
    "Each photo has a 'before' or 'after' label",
    # A single listed word, or one beside zero, is the rule itself.
    "Each account has an email (unique)",
    "A note has a discount (positive)",
    "The password must be 'valid'",
    "The discount must be either positive or zero",
    # Inflections the whole-word match must keep.
    "Rejecting a payment keeps the order open",
    "A total mismatch flags the invoice",
    "Codes are shaped like AB-1234",
    "The fee is the lesser of the two amounts",
    "Refusing a booking frees the room",
    "The score is positively weighted",
])
def test_a_listed_value_or_an_inflected_word_still_states_a_constraint(text):
    assert ledger._demands_enforcement("validation", text)
    assert ledger._demands_enforcement("rule", text)


@pytest.mark.parametrize("text", [
    "A booking must not overlap another booking of the same room",
    "The price must be a valid number",
    "The rating must be between 1 and 5",
    "The end date must be after the start date",
    "The stock of an item is never negative",
    "Guests never exceed the capacity of the rooms",
    "An order (at least one line) can be confirmed",
    "The priority must be one of low, medium, high and may never be empty",
    "The code matches the format AB-1234",
    "An invalid email address is rejected",
    "The quantity ordered exceeds the stock only when backorders are allowed",
    "A member's booking may not start before the member's subscription",
])
def test_a_stated_constraint_still_needs_an_enforcing_line(text):
    assert ledger._demands_enforcement("rule", text)
    assert ledger._demands_enforcement("validation", text)


def test_only_validation_rule_and_uniqueness_demand_enforcement():
    text = "The end date must be after the start date"
    assert not ledger._demands_enforcement("action", text)
    assert not ledger._demands_enforcement("ui", text)
    assert ledger._demands_enforcement("uniqueness", "Each room has a number")


def _check(tmp_path, text, lines, kind="validation"):
    target = tmp_path / SCHEMAS
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_SCHEMAS_SOURCE, encoding="utf-8")
    return ledger.verify_evidence([{
        "id": 1, "kind": kind, "text": text, "status": "implemented", "note": "",
        "evidence": [{"path": SCHEMAS, "line": line} for line in lines],
    }], str(tmp_path))[0]


def test_an_enumeration_verifies_the_values_a_requirement_lists(tmp_path):
    """Two requirements of the same kind with the same kind of citation: only the
    one whose listed values include a constraint word was refused."""
    listed = _check(tmp_path, "Feedback tone must be one of: positive, mixed, negative",
                    ["class Tone(Enum):", 'Positive = "Positive"', 'Negative = "Negative"'])
    assert listed["status"] == "implemented", listed["note"]
    sibling = _check(tmp_path, "Article stage must be one of: draft, in review, published",
                     ["class Stage(Enum):", 'DRAFT = "DRAFT"'])
    assert sibling["status"] == "implemented", sibling["note"]


@pytest.mark.parametrize("lines", [
    ["class Tone(Enum):"],
    ['Mixed = "Mixed"'],
    ["tone: Tone"],
])
def test_values_listed_without_a_marker_are_verified_by_their_enumeration(tmp_path, lines):
    """"must be positive, mixed or negative" keeps its constraint word; the cited
    enumeration, its member or the field typed with it has exactly those values."""
    text = "The tone of a feedback must be positive, mixed or negative"
    assert ledger._demands_enforcement("validation", text)
    verdict = _check(tmp_path, text, lines)
    assert verdict["status"] == "implemented", verdict["note"]


def test_member_names_are_read_as_words(tmp_path):
    verdict = _check(tmp_path, "A stage is draft, in-review or published, never anything else",
                     ["stage: Stage"], kind="rule")
    assert verdict["status"] == "implemented", verdict["note"]


@pytest.mark.parametrize("text, line", [
    ("Each account has an email (unique)", "title: str"),
    ("A note has a discount (positive)", "price: float"),
    ("The discount must be either positive or zero", "price: float"),
    ("Feedback tone must be one of: positive, mixed, negative", "title: str"),
])
def test_a_plain_field_does_not_verify_a_listed_rule(tmp_path, text, line):
    verdict = _check(tmp_path, text, [line], kind="rule")
    assert verdict["status"] == "unverified"
    assert "declares data" in verdict["note"]


@pytest.mark.parametrize("text, lines", [
    # A numeric rule is not a list of the enumeration's values.
    ("The price must be positive", ["class Tone(Enum):"]),
    ("The price must be positive", ["price: float"]),
    # One value the requirement does not allow: the type is not the rule.
    ("The tone must be positive or negative, never anything else", ["tone: Tone"]),
    # A field of another type declares data.
    ("The tone must be positive, mixed or negative", ["title: str"]),
])
def test_an_enumeration_proves_nothing_else(tmp_path, text, lines):
    verdict = _check(tmp_path, text, lines)
    assert verdict["status"] == "unverified"
    assert "declares data" in verdict["note"]
