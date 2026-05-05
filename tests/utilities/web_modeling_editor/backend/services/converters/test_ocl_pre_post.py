"""
Tests for OCL invariant / precondition / postcondition handling in the
WME class diagram converter pipeline.

Covers:
  * ``parse_ocl_body`` header synthesis for invariants and pre/post.
  * ``_process_constraints`` routing by ``kind`` field.
  * Skip-with-warning behaviour on malformed input or missing references.
  * Element-id round-trip stability for OCL pre/post (the targetMethodId
    references rely on method UUIDs surviving JSON->BUML->JSON cycles).
"""

import pytest

from besser.BUML.metamodel.ocl.ocl import OCLConstraint
from besser.BUML.metamodel.structural import (
    Class, DomainModel, IntegerType, BooleanType, Method, Parameter, Property,
)
from besser.BUML.notations.ocl.error_handling import BOCLSyntaxError

from besser.utilities.web_modeling_editor.backend.services.converters.parsers.ocl_parser import (
    parse_ocl_body,
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
    """A class-diagram JSON with one invariant + one pre + one post on deposit."""
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
                    "kind": "invariant", "constraintName": "positive",
                    "constraint": "self.balance >= 0",
                },
                "ocl-pre": {
                    "id": "ocl-pre", "type": "ClassOCLConstraint",
                    "kind": "precondition", "targetMethodId": "m-deposit",
                    "constraintName": "amt_active",
                    "constraint": "self.is_active",
                },
                "ocl-post": {
                    "id": "ocl-post", "type": "ClassOCLConstraint",
                    "kind": "postcondition", "targetMethodId": "m-deposit",
                    "constraintName": "nonneg",
                    "constraint": "self.balance >= 0",
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
# parse_ocl_body
# ---------------------------------------------------------------------------

def test_parse_ocl_body_invariant_returns_ast_backed_constraint(banking_model):
    model, account, _ = banking_model
    c = parse_ocl_body("self.balance >= 0", "invariant", "Account_inv_1", account, model)
    assert isinstance(c, OCLConstraint)
    assert c.name == "Account_inv_1"
    assert c.expression == "self.balance >= 0"  # body-only after parsing
    assert c.context is account
    assert c.ast is not None


def test_parse_ocl_body_precondition_uses_method_signature_header(banking_model):
    model, account, deposit = banking_model
    c = parse_ocl_body("self.is_active", "precondition", "p1", account, model, method=deposit)
    assert isinstance(c, OCLConstraint)
    assert c.expression == "self.is_active"
    assert c.context is account


def test_parse_ocl_body_postcondition(banking_model):
    model, account, deposit = banking_model
    c = parse_ocl_body("self.balance >= 0", "postcondition", "p1", account, model, method=deposit)
    assert isinstance(c, OCLConstraint)
    assert c.expression == "self.balance >= 0"


def test_parse_ocl_body_pre_post_requires_method(banking_model):
    model, account, _ = banking_model
    with pytest.raises(ValueError, match="method is required"):
        parse_ocl_body("true", "precondition", "p1", account, model, method=None)


def test_parse_ocl_body_rejects_unknown_kind(banking_model):
    model, account, _ = banking_model
    with pytest.raises(ValueError, match="unknown kind"):
        parse_ocl_body("true", "guard", "g1", account, model)


def test_parse_ocl_body_propagates_syntax_error(banking_model):
    model, account, _ = banking_model
    with pytest.raises(BOCLSyntaxError):
        parse_ocl_body("self.", "invariant", "broken", account, model)


# ---------------------------------------------------------------------------
# _process_constraints routing
# ---------------------------------------------------------------------------

def test_invariant_routed_to_domain_model_constraints(account_diagram_json):
    dm = process_class_diagram(account_diagram_json)
    invariant_names = sorted(c.name for c in dm.constraints)
    assert invariant_names == ["positive"]
    inv = next(iter(dm.constraints))
    assert isinstance(inv, OCLConstraint)
    assert inv.expression == "self.balance >= 0"


def test_precondition_routed_to_method_pre(account_diagram_json):
    dm = process_class_diagram(account_diagram_json)
    account = next(c for c in dm.types if c.name == "Account")
    deposit = next(m for m in account.methods if m.name == "deposit")
    pre_names = [c.name for c in deposit.pre]
    assert pre_names == ["amt_active"]
    assert deposit.pre[0].expression == "self.is_active"


def test_postcondition_routed_to_method_post(account_diagram_json):
    dm = process_class_diagram(account_diagram_json)
    account = next(c for c in dm.types if c.name == "Account")
    deposit = next(m for m in account.methods if m.name == "deposit")
    post_names = [c.name for c in deposit.post]
    assert post_names == ["nonneg"]


def test_orphan_pre_skipped_with_warning(account_diagram_json):
    """A precondition whose targetMethodId points at a non-existent method
    is dropped with a warning rather than raising — matches the chosen
    skip-with-warning policy."""
    elems = account_diagram_json["model"]["elements"]
    elems["ocl-pre"]["targetMethodId"] = "missing-method-id"
    dm = process_class_diagram(account_diagram_json)
    account = next(c for c in dm.types if c.name == "Account")
    deposit = next(m for m in account.methods if m.name == "deposit")
    assert deposit.pre == []
    assert any("targets missing method" in w for w in dm.ocl_warnings)


def test_unlinked_kind_constraint_skipped_with_warning(account_diagram_json):
    """Removing the ClassOCLLink for the invariant should drop it (we have
    no way to determine the context class without the link)."""
    rels = account_diagram_json["model"]["relationships"]
    del rels["r-inv"]
    dm = process_class_diagram(account_diagram_json)
    assert all(c.name != "positive" for c in dm.constraints)
    assert any("not linked to a class" in w for w in dm.ocl_warnings)


def test_invalid_ocl_body_skipped_with_warning(account_diagram_json):
    """Malformed OCL must not abort the whole conversion."""
    elems = account_diagram_json["model"]["elements"]
    elems["ocl-inv"]["constraint"] = "self."
    dm = process_class_diagram(account_diagram_json)
    # The invariant got dropped; the pre/post should still parse and attach.
    assert all(c.name != "positive" for c in dm.constraints)
    account = next(c for c in dm.types if c.name == "Account")
    deposit = next(m for m in account.methods if m.name == "deposit")
    assert [c.name for c in deposit.pre] == ["amt_active"]
    assert any("Could not parse" in w for w in dm.ocl_warnings)


def test_legacy_textarea_path_still_parses_invariants():
    """Projects from before the kind field stored full OCL text in the
    textarea. They must keep loading; the legacy path now produces an
    OCLConstraint (AST-backed) instead of a bare Constraint."""
    json_data = {
        "title": "T",
        "model": {
            "elements": {
                "cls-A": {"id": "cls-A", "name": "A", "type": "Class",
                          "attributes": ["a-x"], "methods": []},
                "a-x": {"id": "a-x", "name": "x", "visibility": "public",
                        "attributeType": "int"},
                "ocl-legacy": {
                    "id": "ocl-legacy", "type": "ClassOCLConstraint",
                    "constraint": "context A inv legacy: self.x >= 0",
                },
            },
            "relationships": {
                "r": {"id": "r", "type": "ClassOCLLink",
                      "source": {"element": "ocl-legacy"},
                      "target": {"element": "cls-A"}},
            },
        },
    }
    dm = process_class_diagram(json_data)
    assert len(dm.constraints) == 1
    c = next(iter(dm.constraints))
    assert isinstance(c, OCLConstraint)
    assert c.expression == "self.x >= 0"


def test_auto_generated_constraint_names_when_unset(account_diagram_json):
    """Removing constraintName from a pre box should yield an auto-generated
    name in the form ``{methodName}_pre_{n}`` so add_pre doesn't collide
    with another auto-named constraint on the same method."""
    elems = account_diagram_json["model"]["elements"]
    del elems["ocl-pre"]["constraintName"]
    dm = process_class_diagram(account_diagram_json)
    account = next(c for c in dm.types if c.name == "Account")
    deposit = next(m for m in account.methods if m.name == "deposit")
    assert deposit.pre and deposit.pre[0].name.startswith("deposit_pre_")


# ---------------------------------------------------------------------------
# Round-trip stability
# ---------------------------------------------------------------------------

def _summarize_ocl(json_out):
    elements = json_out.get("elements") or json_out["model"]["elements"]
    boxes = sorted(
        (e for e in elements.values() if e.get("type") == "ClassOCLConstraint"),
        key=lambda e: (e.get("kind", ""), e.get("constraintName", "")),
    )
    return [
        (e.get("kind"), e.get("constraintName"),
         e.get("constraint"), e.get("targetMethodId"))
        for e in boxes
    ]


def test_round_trip_full_mix_is_stable(account_diagram_json):
    """JSON -> BUML -> JSON -> BUML -> JSON: the OCL constraint set must
    survive byte-stably across the second cycle (the first cycle may
    normalize legacy fields)."""
    dm1 = process_class_diagram(account_diagram_json)
    out1 = class_buml_to_json(dm1)

    dm2 = process_class_diagram({"title": "BankingTest", "model": out1})
    out2 = class_buml_to_json(dm2)

    assert _summarize_ocl(out1) == _summarize_ocl(out2)


def test_round_trip_method_element_id_is_stable(account_diagram_json):
    """The whole point of stashing _method_element_ids: a method's UUID
    must survive a round-trip so OCL pre/post boxes' targetMethodId
    references stay valid on the next reload."""
    dm = process_class_diagram(account_diagram_json)
    out = class_buml_to_json(dm)
    elements = out["elements"]

    method_elements = {eid: e for eid, e in elements.items() if e.get("type") == "ClassMethod"}
    target_refs = {
        e.get("targetMethodId")
        for e in elements.values()
        if e.get("type") == "ClassOCLConstraint" and e.get("targetMethodId")
    }

    # The original JSON used "m-deposit" as the method element id; it must come back unchanged.
    assert "m-deposit" in method_elements
    assert target_refs == {"m-deposit"}
