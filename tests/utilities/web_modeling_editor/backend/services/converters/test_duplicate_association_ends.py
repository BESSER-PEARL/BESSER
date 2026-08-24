"""Regression: two associations that give one class the same association-end
name must not crash the JSON -> BUML converter.

The metamodel forbids a class from owning two association ends with the same
name -- ``Class._validate_unique_end_names`` raises a hard ``ValueError``. A
visual-editor diagram (or an LLM assembling a spec-driven model) can legitimately
hand us two ``Loan -> Member`` links both rolled ``"member"`` on the Member end
(both ends owned by ``Loan``), which previously crashed the whole generation run:

    ValueError: The class 'Loan' cannot have two association ends with the same
    name: 'member'

``process_class_diagram`` must deterministically suffix the colliding role
instead of propagating the crash.
"""

import pytest

from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
    process_class_diagram,
)


def _class(cid, name):
    return {"id": cid, "name": name, "type": "Class", "owner": None,
            "attributes": [], "methods": []}


def _rel(rid, src_id, src_role, tgt_id, tgt_role, mult="0..*"):
    return {
        "id": rid, "name": rid, "type": "ClassBidirectional",
        "source": {"element": src_id, "multiplicity": mult, "role": src_role},
        "target": {"element": tgt_id, "multiplicity": mult, "role": tgt_role},
    }


def _diagram(elements, relationships):
    return {"title": "Model", "model": {"elements": elements,
                                        "relationships": relationships}}


def _assert_end_names_unique_per_class(domain_model):
    """The metamodel invariant: no class owns two ends with the same name."""
    for cls in domain_model.get_classes():
        names = [e.name for e in cls.all_association_ends()]
        assert len(names) == len(set(names)), (
            f"class {cls.name!r} has duplicate association-end names: {names}"
        )


def test_two_links_same_role_on_one_class_does_not_crash():
    """Two Loan->Member associations both rolled 'member' on the Member end
    (both owned by Loan) must convert without raising, with the collision
    deterministically suffixed."""
    elements = {"loan": _class("loan", "Loan"), "member": _class("member", "Member")}
    relationships = {
        # target.role is the Member end, owned by the source class (Loan).
        "r1": _rel("r1", "loan", "borrower", "member", "member"),
        "r2": _rel("r2", "loan", "returner", "member", "member"),
    }
    domain_model = process_class_diagram(_diagram(elements, relationships))

    _assert_end_names_unique_per_class(domain_model)

    loan = next(c for c in domain_model.get_classes() if c.name == "Loan")
    loan_end_names = {e.name for e in loan.all_association_ends()}
    assert loan_end_names == {"member", "member_1"}, loan_end_names


def test_missing_roles_still_unique_when_colliding():
    """Two links between the same pair with no explicit roles fall back to the
    class-name role and must still be deduped (pre-existing behaviour, kept)."""
    elements = {"a": _class("a", "Order"), "b": _class("b", "Item")}
    relationships = {
        "r1": _rel("r1", "a", None, "b", None),
        "r2": _rel("r2", "a", None, "b", None),
    }
    domain_model = process_class_diagram(_diagram(elements, relationships))
    _assert_end_names_unique_per_class(domain_model)


def test_self_association_same_role_is_deduped():
    """A self-association whose two ends carry the same role name must not crash:
    both ends are owned by the same class."""
    elements = {"e": _class("e", "Employee")}
    relationships = {
        "r1": _rel("r1", "e", "colleague", "e", "colleague"),
    }
    domain_model = process_class_diagram(_diagram(elements, relationships))
    _assert_end_names_unique_per_class(domain_model)

    emp = next(c for c in domain_model.get_classes() if c.name == "Employee")
    end_names = {e.name for e in emp.all_association_ends()}
    assert end_names == {"colleague", "colleague_1"}, end_names


def test_distinct_roles_on_distinct_classes_are_untouched():
    """The dedup is per owning class: a role name reused across *different*
    classes must NOT be renamed."""
    elements = {
        "u": _class("u", "User"), "g": _class("g", "Group"),
        "p": _class("p", "Project"),
    }
    relationships = {
        # 'name' as a role on two different owner classes -- both legitimate.
        "r1": _rel("r1", "g", "group", "u", "name"),   # 'name' owned by Group
        "r2": _rel("r2", "p", "project", "u", "name"),  # 'name' owned by Project
    }
    domain_model = process_class_diagram(_diagram(elements, relationships))
    _assert_end_names_unique_per_class(domain_model)

    names_by_owner = {
        c.name: {e.name for e in c.all_association_ends()}
        for c in domain_model.get_classes()
    }
    assert "name" in names_by_owner["Group"]
    assert "name" in names_by_owner["Project"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
