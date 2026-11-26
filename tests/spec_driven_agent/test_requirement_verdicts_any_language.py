"""A requirement's verdict does not depend on the language of its text.

The extracted text keeps the user's own words. Whether a column default meets a
transition, and whether a 'rule' or 'validation' needs an enforcing line, were
decided by English words in that text: "A new order starts as PENDING" passed and
its French, Spanish and Italian wordings did not; "must be positive" needed an
enforcing line and its translations did not. The extraction now states what meets
the item (``met_by``), and a labelled item is judged the same in every language.

An item without the label (an older run's recipe, a requirement the editor states)
is read as before the label existed. Reading it as a check instead turned 25
confirmed requirements of 223 stored apps into unconfirmed ones (measure/results,
pre-j-028cb57ff against post-j-7d3368770).
"""

import pytest

from besser.spec_driven_agent.planning import requirements_ledger as ledger

MODELS = "backend/models.py"
_SOURCE = '''from enum import Enum

from sqlalchemy import Column, Float, String


class Tone(Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class Order:
    status = Column(String, default="PENDING")
    price = Column(Float)
    tone: Tone
'''
_DEFAULT_LINE = 'status = Column(String, default="PENDING")'
_DECLARATION_LINE = "price = Column(Float)"

INITIAL_STATE = [
    "A new order starts as PENDING",
    "Une nouvelle commande commence à l'état PENDING",
    "Eine neue Bestellung hat zuerst den Status PENDING",
    "Un pedido nuevo empieza en estado PENDING",
    "Un nuovo ordine inizia nello stato PENDING",
    "An order is PENDING when it is created",
]
CONSTRAINT = [
    "The price must be positive",
    "Le prix doit être positif",
    "Der Preis muss positiv sein",
    "El precio debe ser positivo",
    "Il prezzo deve essere positivo",
    "A price above zero is required",
]


def _verdict(tmp_path, kind, text, line, **fields):
    target = tmp_path / MODELS
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_SOURCE, encoding="utf-8")
    return ledger.verify_evidence([{
        "id": 1, "kind": kind, "text": text, "status": "implemented", "note": "",
        "evidence": [{"path": MODELS, "line": line}], **fields,
    }], str(tmp_path))[0]


@pytest.mark.parametrize("text", INITIAL_STATE)
def test_a_column_default_meets_an_initial_state_in_any_language(tmp_path, text):
    verdict = _verdict(tmp_path, "transition", text, _DEFAULT_LINE, met_by="declaration")
    assert verdict["status"] == "implemented", verdict["note"]


@pytest.mark.parametrize("text", INITIAL_STATE)
def test_a_column_default_does_not_meet_a_guarded_transition(tmp_path, text):
    verdict = _verdict(tmp_path, "transition", text, _DEFAULT_LINE, met_by="check")
    assert verdict["status"] == "unverified"
    assert "does not perform" in verdict["note"]


@pytest.mark.parametrize("fields", [{"met_by": "check"}, {"met_by": "type"}])
@pytest.mark.parametrize("kind", ["rule", "validation"])
@pytest.mark.parametrize("text", CONSTRAINT)
def test_a_checked_rule_needs_an_enforcing_line_in_any_language(tmp_path, text, kind, fields):
    verdict = _verdict(tmp_path, kind, text, _DECLARATION_LINE, **fields)
    assert verdict["status"] == "unverified"
    assert "declares data" in verdict["note"]


@pytest.mark.parametrize("text", CONSTRAINT)
def test_a_declared_rule_is_met_by_its_declaration(tmp_path, text):
    verdict = _verdict(tmp_path, "rule", text, _DECLARATION_LINE, met_by="declaration")
    assert verdict["status"] == "implemented", verdict["note"]


def test_an_enumeration_meets_only_an_item_a_type_meets(tmp_path):
    text = "The tone of an order is positive or negative"
    assert _verdict(tmp_path, "rule", text, "tone: Tone", met_by="type")["status"] == "implemented"
    # The same values in an item that needs a check ("may never change once set").
    assert _verdict(tmp_path, "rule", text, "tone: Tone", met_by="check")["status"] == "unverified"


def test_an_item_without_the_label_is_read_as_before_the_label_existed(tmp_path):
    """The verdicts an older recipe's items had: decided by the English words of the text."""
    # A default meets a transition that names the start; a constraint word asks for a check.
    assert _verdict(tmp_path, "transition", INITIAL_STATE[0], _DEFAULT_LINE)["status"] == "implemented"
    assert _verdict(tmp_path, "transition", "An order is PENDING when it is created",
                    _DEFAULT_LINE)["status"] == "unverified"
    assert _verdict(tmp_path, "rule", CONSTRAINT[0], _DECLARATION_LINE)["status"] == "unverified"
    # No constraint word: a declaration was enough, and so was an enumeration listing the values.
    assert _verdict(tmp_path, "rule", "An order has a price", _DECLARATION_LINE)["status"] == "implemented"
    assert _verdict(tmp_path, "rule", "The tone of an order is positive or negative",
                    "tone: Tone")["status"] == "implemented"


def test_uniqueness_always_needs_an_enforcing_line(tmp_path):
    verdict = _verdict(tmp_path, "uniqueness", "Chaque prix est unique", _DECLARATION_LINE, met_by="declaration")
    assert verdict["status"] == "unverified"


def test_the_extraction_asks_for_the_field_and_the_ledger_keeps_it():
    items = ledger._SUBMIT_REQUIREMENTS_TOOL["input_schema"]["properties"]["requirements"]["items"]
    assert ledger.MET_BY_FIELD in items["required"]
    extracted = ledger._requirement_items([
        {"text": "Une commande commence à PENDING", "kind": "transition", "met_by": "declaration"},
        {"text": "Le prix doit être positif", "kind": "rule", "met_by": "check"},
        {"text": "Une commande a un client", "kind": "rule", "met_by": "yes"},
    ])
    assert [item.get("met_by") for item in extracted] == ["declaration", "check", None]
    judged = ledger._verdict_identity(1, extracted[0])
    assert judged["met_by"] == "declaration"
    recorded = ledger._recorded_verdict({**judged, "status": "implemented", "evidence": "", "note": ""})
    assert recorded["met_by"] == "declaration"


_MET_BY_RULE_CHARS = 338  # the rule's size before the counter-example: it must not grow


def test_a_conditional_change_that_names_the_initial_state_is_a_check():
    # A column default is a 'declaration'; a change allowed only under a condition needs a
    # guard even when its condition names the state a record starts in.
    prompt = ledger._EXTRACT_SYSTEM_PROMPT
    rule = prompt[prompt.index("- met_by:"):prompt.index("\n", prompt.index("- met_by:"))]
    check = rule[rule.index("'check'"):]
    assert "a change allowed only under a condition, even if it names the initial state" in check
    assert len(rule) <= _MET_BY_RULE_CHARS
