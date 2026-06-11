"""
Tests for OCL invariant / precondition / postcondition handling in the
WME class diagram converter pipeline.

On ingest the processor accepts two shapes: rows inlined in a class
node's ``data.oclConstraints`` array (the v3->v4 migrator's
collapse-onto-owner output) — e.g.

    {
      "id": "n-account",
      "type": "class",
      "data": {
        "name": "Account",
        "oclConstraints": [
          {"id": "ocl-inv", "name": "positive",
           "expression": "context Account inv positive: self.balance >= 0"},
          ...
        ]
      }
    }

— and free-standing ``ClassOCLConstraint`` nodes (the sticky-note shape
the frontend authors), tethered to their class by a ``ClassOCLLink``
edge. On emit the converter always produces the free-standing node
shape (develop parity: constraints stay visible, editable boxes).

Tests cover:
  * ``parse_constraint_text`` returning the right routing kind plus the
    constraint with the user-typed name attached.
  * ``_process_constraints`` routing each block to either
    ``domain_model.constraints`` (invariants) or to ``Method.pre`` /
    ``Method.post`` lists (pre/post) via Class::method name lookup.
  * Skip-with-warning behaviour on malformed OCL or unresolved methods.
  * Back-compat ingest for body-only constraint rows (kind /
    targetMethodId / constraintName) still produced by older clients.
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
# v4 helpers (local — keep this module self-contained)
# ---------------------------------------------------------------------------

def _class_node_by_name(model_payload, class_name):
    """Return the v4 class node matching ``class_name``."""
    for n in model_payload.get("nodes") or []:
        if n.get("type") == "class" and (n.get("data") or {}).get("name") == class_name:
            return n
    return None


def _all_ocl_rows(payload):
    """Return every OCL constraint in ``payload`` as (anchor-class, row) pairs.

    Covers both wire shapes: rows inlined on a class node's
    ``data.oclConstraints`` (anchor = the owning class) and free-standing
    ``ClassOCLConstraint`` nodes (anchor = the class reached via the
    ``ClassOCLLink`` edge, ``None`` when unlinked).
    """
    model = payload.get("model") if "model" in payload else payload
    nodes = model.get("nodes") or []
    nodes_by_id = {n.get("id"): n for n in nodes}

    def _linked_class_name(constraint_node_id):
        for e in model.get("edges") or []:
            if e.get("type") != "ClassOCLLink":
                continue
            if e.get("source") == constraint_node_id:
                other = nodes_by_id.get(e.get("target"))
            elif e.get("target") == constraint_node_id:
                other = nodes_by_id.get(e.get("source"))
            else:
                continue
            if other is not None and other.get("type") == "class":
                return (other.get("data") or {}).get("name")
        return None

    out = []
    for n in nodes:
        data = n.get("data") or {}
        if n.get("type") == "class":
            for row in data.get("oclConstraints") or []:
                out.append((data.get("name"), row))
        elif n.get("type") == "ClassOCLConstraint":
            out.append((_linked_class_name(n.get("id")), data))
    return out


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
    """A v4 class diagram whose Account node carries inlined OCL rows.

    Three constraint rows: an invariant, a precondition and a postcondition,
    all anchored on the same ``Account`` class node.
    """
    return {
        "title": "BankingTest",
        "model": {
            "version": "4.0.0",
            "type": "ClassDiagram",
            "nodes": [
                {
                    "id": "n-account",
                    "type": "class",
                    "position": {"x": 0, "y": 0},
                    "width": 160, "height": 100,
                    "measured": {"width": 160, "height": 100},
                    "data": {
                        "name": "Account",
                        "stereotype": None,
                        "attributes": [
                            {"id": "a-balance", "name": "balance",
                             "attributeType": "int", "visibility": "public"},
                            {"id": "a-active", "name": "is_active",
                             "attributeType": "bool", "visibility": "public"},
                        ],
                        "methods": [
                            {"id": "m-deposit", "name": "+ deposit(amount: int): int"},
                        ],
                        "oclConstraints": [
                            {"id": "ocl-inv", "name": "positive",
                             "expression": "context Account inv positive: self.balance >= 0"},
                            {"id": "ocl-pre", "name": "amt_pos",
                             "expression": "context Account::deposit(amount: int) pre: amount > 0"},
                            {"id": "ocl-post", "name": "balance_nn",
                             "expression": "context Account::deposit(amount: int) post: self.balance >= 0"},
                        ],
                    },
                },
            ],
            "edges": [],
        },
    }


def _ocl_row_by_id(diagram_json, ocl_id):
    """Return the inlined OCL row with the given id (for in-place mutation)."""
    for n in diagram_json["model"]["nodes"]:
        for row in (n.get("data") or {}).get("oclConstraints") or []:
            if row.get("id") == ocl_id:
                return row
    raise KeyError(ocl_id)


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
    assert c.name == "positive"
    assert c.expression == "context Account inv positive: self.balance >= 0"
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
    assert inv.expression == "context Account inv positive: self.balance >= 0"


def test_precondition_routed_to_method_pre(account_diagram_json):
    dm = process_class_diagram(account_diagram_json)
    account = next(c for c in dm.types if c.name == "Account")
    deposit = next(m for m in account.methods if m.name == "deposit")
    assert [c.expression for c in deposit.pre] == [
        "context Account::deposit(amount: int) pre: amount > 0"
    ]


def test_postcondition_routed_to_method_post(account_diagram_json):
    dm = process_class_diagram(account_diagram_json)
    account = next(c for c in dm.types if c.name == "Account")
    deposit = next(m for m in account.methods if m.name == "deposit")
    assert [c.expression for c in deposit.post] == [
        "context Account::deposit(amount: int) post: self.balance >= 0"
    ]


def test_unknown_method_in_pre_skipped_with_warning(account_diagram_json):
    """Pre/post pointing at a method the class doesn't have → warning + skip."""
    _ocl_row_by_id(account_diagram_json, "ocl-pre")["expression"] = (
        "context Account::missing(amount: int) pre: amount > 0"
    )
    dm = process_class_diagram(account_diagram_json)
    account = next(c for c in dm.types if c.name == "Account")
    deposit = next(m for m in account.methods if m.name == "deposit")
    assert deposit.pre == []
    assert any("targets unknown method" in w for w in dm.ocl_warnings)


def test_invalid_ocl_skipped_with_warning(account_diagram_json):
    """Malformed OCL must not abort the conversion — other constraints must still load."""
    _ocl_row_by_id(account_diagram_json, "ocl-inv")["expression"] = (
        "context Account inv broken: self."
    )
    dm = process_class_diagram(account_diagram_json)
    # Invariant got dropped; pre/post still parse.
    assert all(c.name != "positive" for c in dm.constraints)
    account = next(c for c in dm.types if c.name == "Account")
    deposit = next(m for m in account.methods if m.name == "deposit")
    assert [c.expression for c in deposit.pre] == [
        "context Account::deposit(amount: int) pre: amount > 0"
    ]
    assert any("Invalid OCL syntax" in w for w in dm.ocl_warnings)


def test_multi_block_textarea_parses_each_block_independently(banking_model, account_diagram_json):
    """A single OCL row may hold multiple constraints separated by `context` boundaries."""
    # Replace the three constraint rows with a single invariant box that
    # holds two ``context`` blocks back-to-back.
    account_node = _class_node_by_name(account_diagram_json["model"], "Account")
    account_node["data"]["oclConstraints"] = [
        {
            "id": "ocl-multi",
            "name": "multi",
            "expression": (
                "context Account inv positive: self.balance >= 0\n"
                "context Account inv active: self.is_active"
            ),
        },
    ]

    dm = process_class_diagram(account_diagram_json)
    invariants = sorted(c.name for c in dm.constraints)
    assert invariants == ["active", "positive"]


# ---------------------------------------------------------------------------
# Back-compat shim — body-only legacy rows
# ---------------------------------------------------------------------------

def test_body_only_legacy_files_still_load():
    """Body-only inlined OCL rows (kind + constraintName + targetMethodId)
    are lifted to full text by the back-compat shim before parsing.
    """
    legacy = {
        "title": "Legacy",
        "model": {
            "version": "4.0.0",
            "type": "ClassDiagram",
            "nodes": [
                {
                    "id": "n-account", "type": "class",
                    "position": {"x": 0, "y": 0}, "width": 160, "height": 100,
                    "measured": {"width": 160, "height": 100},
                    "data": {
                        "name": "Account", "stereotype": None,
                        "attributes": [
                            {"id": "a-balance", "name": "balance",
                             "attributeType": "int", "visibility": "public"},
                        ],
                        "methods": [
                            {"id": "m-deposit", "name": "+ deposit(amount: int): int"},
                        ],
                        "oclConstraints": [
                            {"id": "ocl-inv",
                             "expression": "self.balance >= 0",
                             "kind": "invariant",
                             "constraintName": "positive"},
                            {"id": "ocl-pre",
                             "expression": "amount > 0",
                             "kind": "precondition",
                             "targetMethodId": "m-deposit",
                             "constraintName": "amt_pos"},
                        ],
                    },
                },
            ],
            "edges": [],
        },
    }
    dm = process_class_diagram(legacy)
    assert sorted(c.name for c in dm.constraints) == ["positive"]
    account = next(c for c in dm.types if c.name == "Account")
    deposit = next(m for m in account.methods if m.name == "deposit")
    # Legacy body-only ingest goes through ``legacy_body_only_to_text``
    # which lifts to canonical full text before parsing — so the stored
    # ``expression`` matches the post-consolidation contract.
    assert [c.expression for c in deposit.pre] == [
        "context Account::deposit(amount: int) pre: amount > 0"
    ]
    assert dm.ocl_warnings == []


def test_body_only_legacy_orphan_method_skipped_with_warning():
    """Body-only pre/post with a missing targetMethodId is dropped with a warning."""
    legacy = {
        "title": "Legacy",
        "model": {
            "version": "4.0.0",
            "type": "ClassDiagram",
            "nodes": [
                {
                    "id": "n-account", "type": "class",
                    "position": {"x": 0, "y": 0}, "width": 160, "height": 100,
                    "measured": {"width": 160, "height": 100},
                    "data": {
                        "name": "Account", "stereotype": None,
                        "attributes": [],
                        "methods": [
                            {"id": "m-deposit", "name": "+ deposit(amount: int): int"},
                        ],
                        "oclConstraints": [
                            {"id": "ocl-pre",
                             "expression": "amount > 0",
                             "kind": "precondition",
                             "targetMethodId": "missing",  # not a real method
                             "constraintName": "amt_pos"},
                        ],
                    },
                },
            ],
            "edges": [],
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

def _summarize_ocl(payload):
    """Sorted (expression, description) pairs for every OCL row in the payload."""
    rows = _all_ocl_rows(payload)
    return sorted(
        (row.get("expression"), row.get("description"))
        for _cls, row in rows
    )


def _summarize_anchors(payload):
    """Sorted (expression, anchoring-class-name) pairs for every OCL row."""
    rows = _all_ocl_rows(payload)
    return sorted(
        (row.get("expression"), cls_name)
        for cls_name, row in rows
    )


def test_round_trip_full_mix_is_byte_stable(account_diagram_json):
    """JSON -> BUML -> JSON -> BUML -> JSON: OCL rows + their anchoring survive."""
    dm1 = process_class_diagram(account_diagram_json)
    out1 = class_buml_to_json(dm1)

    dm2 = process_class_diagram({"title": "BankingTest", "model": out1})
    out2 = class_buml_to_json(dm2)

    # All three constraints (inv + pre + post) survive as visible,
    # Account-anchored ``ClassOCLConstraint`` nodes.
    assert len(_all_ocl_rows(out1)) == 3
    assert _summarize_anchors(out1) == [
        ("context Account inv positive: self.balance >= 0", "Account"),
        ("context Account::deposit(amount: int) post: self.balance >= 0", "Account"),
        ("context Account::deposit(amount: int) pre: amount > 0", "Account"),
    ]
    # Constraint text + description per row is stable.
    assert _summarize_ocl(out1) == _summarize_ocl(out2)
    # Row -> class anchoring is stable.
    assert _summarize_anchors(out1) == _summarize_anchors(out2)
    # Same number of OCL rows on both cycles.
    assert len(_all_ocl_rows(out1)) == len(_all_ocl_rows(out2))


def test_emitted_ocl_boxes_carry_no_legacy_metadata(account_diagram_json):
    """The new shape strips kind / targetMethodId / constraintName on emit."""
    dm = process_class_diagram(account_diagram_json)
    out = class_buml_to_json(dm)
    rows = _all_ocl_rows(out)
    assert rows, "expected emitted OCL constraint nodes"
    for _cls, row in rows:
        assert "kind" not in row
        assert "targetMethodId" not in row
        assert "constraintName" not in row
        assert (row.get("expression") or "").lstrip().lower().startswith("context")


def test_legacy_body_only_normalized_to_full_text_on_emit():
    """Round-tripping a legacy body-only project produces full-text constraints."""
    legacy = {
        "title": "Legacy",
        "model": {
            "version": "4.0.0",
            "type": "ClassDiagram",
            "nodes": [
                {
                    "id": "n-account", "type": "class",
                    "position": {"x": 0, "y": 0}, "width": 160, "height": 100,
                    "measured": {"width": 160, "height": 100},
                    "data": {
                        "name": "Account", "stereotype": None,
                        "attributes": [
                            {"id": "a-balance", "name": "balance",
                             "attributeType": "int", "visibility": "public"},
                        ],
                        "methods": [],
                        "oclConstraints": [
                            {"id": "ocl-inv",
                             "expression": "self.balance >= 0",
                             "kind": "invariant",
                             "constraintName": "positive"},
                        ],
                    },
                },
            ],
            "edges": [],
        },
    }
    dm = process_class_diagram(legacy)
    out = class_buml_to_json(dm)
    rows = _all_ocl_rows(out)
    assert len(rows) == 1
    assert rows[0][1]["expression"] == "context Account inv positive: self.balance >= 0"


# ---------------------------------------------------------------------------
# Duplicate constraint names across rows
# ---------------------------------------------------------------------------

def test_duplicate_constraint_name_across_boxes_does_not_crash():
    """Two ``oclConstraints`` rows that parse to the same constraint name
    must not crash the conversion.

    ``DomainModel.constraints`` rejects duplicate names with a
    ``ValueError``; ``_process_constraints`` de-dups by name before
    assigning, keeping the first occurrence and emitting a warning for
    each subsequent collision rather than letting the setter raise.
    """
    diagram = {
        "title": "DupNames",
        "model": {
            "version": "4.0.0",
            "type": "ClassDiagram",
            "nodes": [
                {
                    "id": "n-account", "type": "class",
                    "position": {"x": 0, "y": 0}, "width": 160, "height": 100,
                    "measured": {"width": 160, "height": 100},
                    "data": {
                        "name": "Account", "stereotype": None,
                        "attributes": [
                            {"id": "a-balance", "name": "balance",
                             "attributeType": "int", "visibility": "public"},
                        ],
                        "methods": [],
                        "oclConstraints": [
                            {"id": "ocl-1", "name": "dup",
                             "expression": "context Account inv dup: self.balance > 0"},
                            {"id": "ocl-2", "name": "dup",
                             "expression": "context Account inv dup: self.balance >= 0"},
                        ],
                    },
                },
            ],
            "edges": [],
        },
    }
    dm = process_class_diagram(diagram)  # must not raise
    # Exactly one ``dup`` constraint survives — first wins.
    dups = [c for c in dm.constraints if c.name == "dup"]
    assert len(dups) == 1
    assert dups[0].expression == "context Account inv dup: self.balance > 0"
    # And we surfaced the collision in ocl_warnings instead of crashing.
    assert any("duplicate constraint name" in w for w in dm.ocl_warnings)
