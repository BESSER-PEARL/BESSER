"""
Tests for OCL invariant / precondition / postcondition handling in the
WME class diagram converter pipeline.

The canonical wire shape is full OCL text in each ``ClassOCLConstraint``
element's ``constraint`` field — e.g.

    context Book inv pages_positive: self.pages > 0
    context Book::decrease_stock(qty: int) pre: qty > 0

Tests cover:
  * ``parse_constraint_text`` returning the right routing kind plus the
    constraint with the user-typed name attached.
  * ``_process_constraints`` routing each block to either
    ``domain_model.constraints`` (invariants) or to ``Method.pre`` /
    ``Method.post`` lists (pre/post) via Class::method name lookup.
  * Skip-with-warning behaviour on malformed OCL or unresolved methods.
  * Back-compat ingest for body-only JSON files produced by an earlier
    intermediate iteration (kind / targetMethodId / constraintName).
  * Round-trip stability of the canonical full-text shape.
"""

import pytest

from besser.BUML.metamodel.ocl.ocl import OCLConstraint
from besser.BUML.metamodel.structural import (
    Class, DomainModel, IntegerType, BooleanType, Method, Parameter, Property,
)
from besser.BUML.notations.ocl.error_handling import BOCLSyntaxError

from besser.utilities.web_modeling_editor.backend.services.converters.parsers.ocl_parser import (
    parse_constraint_text,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
    process_class_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
    class_buml_to_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def banking_model():
    """A small Account class with a deposit method, used as a parsing context."""
    balance = Property("balance", IntegerType)
    is_active = Property("is_active", BooleanType)
    account = Class("Account", attributes={balance, is_active})
    deposit = Method(
        name="deposit",
        parameters=[Parameter("amount", IntegerType)],
        type=IntegerType,
    )
    account.methods = {deposit}
    model = DomainModel("BankingModel", types={account})
    return model, account, deposit


@pytest.fixture
def account_diagram_json():
    """A class-diagram JSON whose OCL boxes carry full-text constraints."""
    return {
        "title": "BankingTest",
        "model": {
            "elements": {
                "cls-Account": {
                    "id": "cls-Account", "name": "Account", "type": "Class",
                    "attributes": ["attr-balance", "attr-active"],
                    "methods": ["m-deposit"],
                    "bounds": {"x": 0, "y": 0, "width": 160, "height": 100},
                },
                "attr-balance": {
                    "id": "attr-balance", "name": "balance",
                    "visibility": "public", "attributeType": "int",
                },
                "attr-active": {
                    "id": "attr-active", "name": "is_active",
                    "visibility": "public", "attributeType": "bool",
                },
                "m-deposit": {
                    "id": "m-deposit", "name": "+ deposit(amount: int): int",
                    "type": "ClassMethod", "owner": "cls-Account",
                },
                "ocl-inv": {
                    "id": "ocl-inv", "type": "ClassOCLConstraint",
                    "constraint": "context Account inv positive: self.balance >= 0",
                },
                "ocl-pre": {
                    "id": "ocl-pre", "type": "ClassOCLConstraint",
                    "constraint": "context Account::deposit(amount: int) pre: amount > 0",
                },
                "ocl-post": {
                    "id": "ocl-post", "type": "ClassOCLConstraint",
                    "constraint": "context Account::deposit(amount: int) post: self.balance >= 0",
                },
            },
            "relationships": {
                "r-inv":  {"id": "r-inv",  "type": "ClassOCLLink",
                            "source": {"element": "ocl-inv"},
                            "target": {"element": "cls-Account"}},
                "r-pre":  {"id": "r-pre",  "type": "ClassOCLLink",
                            "source": {"element": "ocl-pre"},
                            "target": {"element": "cls-Account"}},
                "r-post": {"id": "r-post", "type": "ClassOCLLink",
                            "source": {"element": "ocl-post"},
                            "target": {"element": "cls-Account"}},
            },
        },
    }


# ---------------------------------------------------------------------------
# parse_constraint_text — header inspection + AST attachment
# ---------------------------------------------------------------------------

def test_parse_constraint_text_invariant_extracts_name(banking_model):
    model, account, _ = banking_model
    kind, c, class_name, method_name = parse_constraint_text(
        "context Account inv positive: self.balance >= 0", model,
    )
    assert kind == "invariant"
    assert isinstance(c, OCLConstraint)
    assert c.name == "positive"        # user-typed name preserved
    assert c.expression == "self.balance >= 0"
    assert c.context is account
    assert class_name == "Account"
    assert method_name is None


def test_parse_constraint_text_precondition_returns_method_name(banking_model):
    model, account, _ = banking_model
    kind, c, class_name, method_name = parse_constraint_text(
        "context Account::deposit(amount: int) pre: amount > 0", model,
    )
    assert kind == "precondition"
    assert isinstance(c, OCLConstraint)
    assert c.context is account
    assert class_name == "Account"
    assert method_name == "deposit"


def test_parse_constraint_text_postcondition(banking_model):
    model, account, _ = banking_model
    kind, c, class_name, method_name = parse_constraint_text(
        "context Account::deposit(amount: int) post: self.balance >= 0", model,
    )
    assert kind == "postcondition"
    assert isinstance(c, OCLConstraint)
    assert method_name == "deposit"


def test_parse_constraint_text_rejects_missing_header(banking_model):
    model, _, _ = banking_model
    with pytest.raises(ValueError, match="recognisable"):
        parse_constraint_text("self.balance >= 0", model)


def test_parse_constraint_text_rejects_unknown_class(banking_model):
    model, _, _ = banking_model
    with pytest.raises(ValueError, match="unknown class"):
        parse_constraint_text("context Nope inv x: 1 = 1", model)


def test_parse_constraint_text_propagates_syntax_error(banking_model):
    model, _, _ = banking_model
    with pytest.raises(BOCLSyntaxError):
        parse_constraint_text("context Account inv broken: self.", model)


# ---------------------------------------------------------------------------
# _process_constraints — routing into domain_model / Method.pre / Method.post
# ---------------------------------------------------------------------------

def test_invariant_routed_to_domain_model_constraints(account_diagram_json):
    dm = process_class_diagram(account_diagram_json)
    invariants = sorted(c.name for c in dm.constraints)
    assert invariants == ["positive"]
    inv = next(iter(dm.constraints))
    assert isinstance(inv, OCLConstraint)
    assert inv.expression == "self.balance >= 0"


def test_precondition_routed_to_method_pre(account_diagram_json):
    dm = process_class_diagram(account_diagram_json)
    account = next(c for c in dm.types if c.name == "Account")
    deposit = next(m for m in account.methods if m.name == "deposit")
    assert [c.expression for c in deposit.pre] == ["amount > 0"]


def test_postcondition_routed_to_method_post(account_diagram_json):
    dm = process_class_diagram(account_diagram_json)
    account = next(c for c in dm.types if c.name == "Account")
    deposit = next(m for m in account.methods if m.name == "deposit")
    assert [c.expression for c in deposit.post] == ["self.balance >= 0"]


def test_unknown_method_in_pre_skipped_with_warning(account_diagram_json):
    """Pre/post pointing at a method the class doesn't have → warning + skip."""
    elems = account_diagram_json["model"]["elements"]
    elems["ocl-pre"]["constraint"] = "context Account::missing(amount: int) pre: amount > 0"
    dm = process_class_diagram(account_diagram_json)
    account = next(c for c in dm.types if c.name == "Account")
    deposit = next(m for m in account.methods if m.name == "deposit")
    assert deposit.pre == []
    assert any("targets unknown method" in w for w in dm.ocl_warnings)


def test_invalid_ocl_skipped_with_warning(account_diagram_json):
    """Malformed OCL must not abort the conversion — other constraints must still load."""
    elems = account_diagram_json["model"]["elements"]
    elems["ocl-inv"]["constraint"] = "context Account inv broken: self."
    dm = process_class_diagram(account_diagram_json)
    # Invariant got dropped; pre/post still parse.
    assert all(c.name != "positive" for c in dm.constraints)
    account = next(c for c in dm.types if c.name == "Account")
    deposit = next(m for m in account.methods if m.name == "deposit")
    assert [c.expression for c in deposit.pre] == ["amount > 0"]
    assert any("Invalid OCL syntax" in w for w in dm.ocl_warnings)


def test_multi_block_textarea_parses_each_block_independently(banking_model, account_diagram_json):
    """A single OCL box may hold multiple constraints separated by `context` boundaries."""
    elems = account_diagram_json["model"]["elements"]
    # Drop the per-kind boxes; replace with one box that holds everything.
    for k in ("ocl-pre", "ocl-post"):
        del elems[k]
    rels = account_diagram_json["model"]["relationships"]
    for k in ("r-pre", "r-post"):
        del rels[k]
    elems["ocl-inv"]["constraint"] = (
        "context Account inv positive: self.balance >= 0\n"
        "context Account inv active: self.is_active"
    )

    dm = process_class_diagram(account_diagram_json)
    invariants = sorted(c.name for c in dm.constraints)
    assert invariants == ["active", "positive"]


# ---------------------------------------------------------------------------
# Back-compat shim — body-only legacy JSON files
# ---------------------------------------------------------------------------

def test_body_only_legacy_files_still_load():
    """Body-only ``ClassOCLConstraint`` boxes (kind + constraintName + targetMethodId)
    are lifted to full text by the back-compat shim before parsing.
    """
    legacy = {
        "title": "Legacy",
        "model": {
            "elements": {
                "cls-Account": {"id": "cls-Account", "name": "Account", "type": "Class",
                                 "attributes": ["a-balance"], "methods": ["m-deposit"]},
                "a-balance": {"id": "a-balance", "name": "balance",
                              "visibility": "public", "attributeType": "int"},
                "m-deposit": {"id": "m-deposit", "name": "+ deposit(amount: int): int",
                               "type": "ClassMethod", "owner": "cls-Account"},
                "ocl-inv": {
                    "id": "ocl-inv", "type": "ClassOCLConstraint",
                    "constraint": "self.balance >= 0",
                    "kind": "invariant", "constraintName": "positive",
                },
                "ocl-pre": {
                    "id": "ocl-pre", "type": "ClassOCLConstraint",
                    "constraint": "amount > 0",
                    "kind": "precondition",
                    "targetMethodId": "m-deposit",
                    "constraintName": "amt_pos",
                },
            },
            "relationships": {
                "r-inv": {"id": "r-inv", "type": "ClassOCLLink",
                           "source": {"element": "ocl-inv"},
                           "target": {"element": "cls-Account"}},
                "r-pre": {"id": "r-pre", "type": "ClassOCLLink",
                           "source": {"element": "ocl-pre"},
                           "target": {"element": "cls-Account"}},
            },
        },
    }
    dm = process_class_diagram(legacy)
    assert sorted(c.name for c in dm.constraints) == ["positive"]
    account = next(c for c in dm.types if c.name == "Account")
    deposit = next(m for m in account.methods if m.name == "deposit")
    assert [c.expression for c in deposit.pre] == ["amount > 0"]
    assert dm.ocl_warnings == []


def test_body_only_legacy_orphan_method_skipped_with_warning():
    """Body-only pre/post with a missing targetMethodId is dropped with a warning."""
    legacy = {
        "title": "Legacy",
        "model": {
            "elements": {
                "cls-Account": {"id": "cls-Account", "name": "Account", "type": "Class",
                                 "attributes": [], "methods": ["m-deposit"]},
                "m-deposit": {"id": "m-deposit", "name": "+ deposit(amount: int): int",
                               "type": "ClassMethod", "owner": "cls-Account"},
                "ocl-pre": {
                    "id": "ocl-pre", "type": "ClassOCLConstraint",
                    "constraint": "amount > 0",
                    "kind": "precondition",
                    "targetMethodId": "missing",  # not a real method element
                    "constraintName": "amt_pos",
                },
            },
            "relationships": {
                "r-pre": {"id": "r-pre", "type": "ClassOCLLink",
                           "source": {"element": "ocl-pre"},
                           "target": {"element": "cls-Account"}},
            },
        },
    }
    dm = process_class_diagram(legacy)
    account = next(c for c in dm.types if c.name == "Account")
    deposit = next(m for m in account.methods if m.name == "deposit")
    assert deposit.pre == []
    assert any("missing method" in w for w in dm.ocl_warnings)


# ---------------------------------------------------------------------------
# Round-trip stability — full-text canonical
# ---------------------------------------------------------------------------

def _summarize_ocl(json_out):
    elements = json_out.get("elements") or json_out["model"]["elements"]
    boxes = sorted(
        (e for e in elements.values() if e.get("type") == "ClassOCLConstraint"),
        key=lambda e: e.get("constraint", ""),
    )
    return [e.get("constraint") for e in boxes]


def test_round_trip_full_mix_is_byte_stable(account_diagram_json):
    """JSON -> BUML -> JSON -> BUML -> JSON: OCL constraint set survives byte-stably."""
    dm1 = process_class_diagram(account_diagram_json)
    out1 = class_buml_to_json(dm1)

    dm2 = process_class_diagram({"title": "BankingTest", "model": out1})
    out2 = class_buml_to_json(dm2)

    assert _summarize_ocl(out1) == _summarize_ocl(out2)


def test_emitted_ocl_boxes_carry_no_legacy_metadata(account_diagram_json):
    """The new shape strips kind / targetMethodId / constraintName on emit."""
    dm = process_class_diagram(account_diagram_json)
    out = class_buml_to_json(dm)
    for e in out["elements"].values():
        if e.get("type") != "ClassOCLConstraint":
            continue
        assert "kind" not in e
        assert "targetMethodId" not in e
        assert "constraintName" not in e
        assert e["constraint"].lstrip().lower().startswith("context")


def test_legacy_body_only_normalized_to_full_text_on_emit():
    """Round-tripping a legacy body-only project produces full-text constraints."""
    legacy = {
        "title": "Legacy",
        "model": {
            "elements": {
                "cls-Account": {"id": "cls-Account", "name": "Account", "type": "Class",
                                 "attributes": ["a-balance"], "methods": []},
                "a-balance": {"id": "a-balance", "name": "balance",
                              "visibility": "public", "attributeType": "int"},
                "ocl-inv": {
                    "id": "ocl-inv", "type": "ClassOCLConstraint",
                    "constraint": "self.balance >= 0",
                    "kind": "invariant", "constraintName": "positive",
                },
            },
            "relationships": {
                "r-inv": {"id": "r-inv", "type": "ClassOCLLink",
                           "source": {"element": "ocl-inv"},
                           "target": {"element": "cls-Account"}},
            },
        },
    }
    dm = process_class_diagram(legacy)
    out = class_buml_to_json(dm)
    ocl = next(e for e in out["elements"].values() if e.get("type") == "ClassOCLConstraint")
    assert ocl["constraint"] == "context Account inv positive: self.balance >= 0"
