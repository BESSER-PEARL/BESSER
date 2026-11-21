"""An enumeration enforces the values a requirement lists.

A 'validation' or 'rule' requirement the extraction says a check meets
(``met_by``) needs an enforcing line. A value of an enumeration ("one of: positive, mixed, negative")
still states the rule, and the enumeration that implements it was reported as a
blocker.
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


class Slot(Enum):
    BEFORE_NOON = "BEFORE_NOON"
    AfterNoon = "AfterNoon"


class ArticleCreate(BaseModel):
    title: str
    tone: Tone
    stage: Stage
    slot: Slot
    price: float
'''


def test_only_validation_rule_and_uniqueness_demand_enforcement():
    assert not ledger._demands_enforcement({"kind": "action"})
    assert not ledger._demands_enforcement({"kind": "ui"})
    assert ledger._demands_enforcement({"kind": "uniqueness", "met_by": "declaration"})
    assert ledger._demands_enforcement({"kind": "rule"})
    assert ledger._demands_enforcement({"kind": "rule", "met_by": "type"})
    assert not ledger._demands_enforcement({"kind": "validation", "met_by": "declaration"})


def _check(tmp_path, text, lines, kind="validation", met_by="type"):
    target = tmp_path / SCHEMAS
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_SCHEMAS_SOURCE, encoding="utf-8")
    return ledger.verify_evidence([{
        "id": 1, "kind": kind, "text": text, "status": "implemented", "note": "", "met_by": met_by,
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
    verdict = _check(tmp_path, text, lines)
    assert verdict["status"] == "implemented", verdict["note"]


def test_member_names_are_read_as_words(tmp_path):
    text = "The slot of an article is before noon or after-noon"
    verdict = _check(tmp_path, text, ["slot: Slot"], kind="rule")
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


@pytest.mark.parametrize("text, lines, met_by", [
    # A numeric rule is not a list of the enumeration's values.
    ("The price must be positive", ["class Tone(Enum):"], "check"),
    ("The price must be positive", ["price: float"], "check"),
    # One value the requirement does not allow: the type is not the rule.
    ("The tone must be positive or negative, never anything else", ["tone: Tone"], "type"),
    # A field of another type declares data.
    ("The tone must be positive, mixed or negative", ["title: str"], "type"),
    # The values are all named, and the rule about them still takes a check.
    ("An article must not go from published back to draft or in review", ["stage: Stage"], "check"),
    ("The tone is positive, mixed or negative and may never be changed", ["class Tone(Enum):"], "check"),
])
def test_an_enumeration_proves_nothing_else(tmp_path, text, lines, met_by):
    verdict = _check(tmp_path, text, lines, met_by=met_by)
    assert verdict["status"] == "unverified"
    assert "declares data" in verdict["note"]


NOTES = "backend/notes.py"
_NOTES_SOURCE = '''from enum import Enum
from typing import Optional
from pydantic import BaseModel


class Mood(Enum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"


class Rating(Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class Spare(Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class Coded(Enum):
    P = "positive"
    N = "negative"


class NoteCreate(BaseModel):
    title: str
    mood: str
    feeling: Mood
    rating: Optional[Rating] = None
    grade: Coded
'''


def _check_notes(tmp_path, text, line):
    target = tmp_path / NOTES
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_NOTES_SOURCE, encoding="utf-8")
    return ledger.verify_evidence([{
        "id": 1, "kind": "validation", "text": text, "status": "implemented", "note": "", "met_by": "type",
        "evidence": [{"path": NOTES, "line": line}],
    }], str(tmp_path))[0]


@pytest.mark.parametrize("text, line", [
    # The field the requirement is about is a plain string; the enumeration types another field.
    ("The mood of a note is positive, neutral or negative", "class Mood(Enum):"),
    ("The mood of a note is positive, neutral or negative", 'NEUTRAL = "NEUTRAL"'),
    ("The mood of a note is positive, neutral or negative", "feeling: Mood"),
    ("Mood must be one of: positive, neutral, negative", "mood: str"),
    # No field is of this enumeration.
    ("The mood of a note is positive or negative", "class Spare(Enum):"),
    # The enumeration lacks a value the requirement lists.
    ("The rating of a note is positive, neutral or negative", "rating: Optional[Rating] = None"),
    ("The rating of a note is neutral, positive or negative", "class Rating(Enum):"),
    ("The rating must be one of: positive, negative, neutral", "class Rating(Enum):"),
])
def test_an_enumeration_must_type_the_named_field_and_have_every_listed_value(tmp_path, text, line):
    verdict = _check_notes(tmp_path, text, line)
    assert verdict["status"] == "unverified"
    assert "declares data" in verdict["note"]


@pytest.mark.parametrize("text, line", [
    ("The feeling of a note is positive, neutral or negative", "class Mood(Enum):"),
    ("The feeling of a note is positive, neutral or negative", "feeling: Mood"),
    ("The rating of a note must be one of: positive, negative", "rating: Optional[Rating] = None"),
    # Member names that are codes: the values are what the requirement lists.
    ("The grade of a note is positive or negative", "grade: Coded"),
])
def test_an_enumeration_typing_the_named_field_with_the_listed_values_verifies_it(tmp_path, text, line):
    verdict = _check_notes(tmp_path, text, line)
    assert verdict["status"] == "implemented", verdict["note"]
