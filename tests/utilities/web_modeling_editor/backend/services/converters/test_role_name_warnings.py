"""Tests for the stale-role-name detection pass in ``process_class_diagram``.

The metamodel's ``Class.name`` setter propagates a rename to matching
association role names (BESSER commit ``7a62486d``). The visual editor,
however, rebuilds the BUML model from scratch via ``Class.__init__`` --
bypassing that propagation. If the editor (or any other JSON producer)
leaves a stale role name in the relationship JSON after renaming a class,
the BUML model ends up with a mismatched role like ``members`` on an end
typed ``User``.

``class_diagram_processor.process_class_diagram`` runs a *warning-only*
pass after building the model to flag those mismatches. It does **not**
auto-rename -- doing so would clobber intentional aliases such as
``borrower`` on a ``Member`` end.
"""

import logging

from besser.BUML.metamodel.structural import BinaryAssociation, Class
from besser.utilities.web_modeling_editor.backend.models import (
    DiagramInput, ProjectInput,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
    process_class_diagram,
)


CLASS_DIAGRAM_LOGGER = (
    "besser.utilities.web_modeling_editor.backend.services."
    "converters.json_to_buml.class_diagram_processor"
)


def _stale_role_diagram_json():
    """A class diagram with a stale role name: an association ``Book -> User``
    whose User end is named ``members``. This mirrors the scenario where a
    class was renamed from ``Member`` to ``User`` in the visual editor but
    the relationship JSON's ``role`` field was not updated.
    """
    return {
        "title": "LibraryModel",
        "model": {
            "elements": {
                "cls-user": {
                    "id": "cls-user",
                    "name": "User",
                    "type": "Class",
                    "owner": None,
                    "attributes": [],
                    "methods": [],
                },
                "cls-book": {
                    "id": "cls-book",
                    "name": "Book",
                    "type": "Class",
                    "owner": None,
                    "attributes": [],
                    "methods": [],
                },
            },
            "relationships": {
                "rel-borrows": {
                    "id": "rel-borrows",
                    "name": "borrows",
                    "type": "ClassBidirectional",
                    "source": {
                        "element": "cls-book",
                        "multiplicity": "0..*",
                        "role": "books",
                    },
                    "target": {
                        "element": "cls-user",
                        "multiplicity": "0..*",
                        "role": "members",  # stale: was set when target was the Member class
                    },
                },
            },
        },
    }


def _get_user_end_named_members(domain_model):
    """Return the association end whose role name is ``members`` and target is User."""
    for assoc in domain_model.associations:
        if not isinstance(assoc, BinaryAssociation):
            continue
        for end in assoc.ends:
            if end.name == "members" and isinstance(end.type, Class) and end.type.name == "User":
                return assoc, end
    return None, None


def test_stale_role_name_emits_warning(caplog):
    """A role ``members`` on a User-typed end should fire the stale-role warning."""
    json_data = _stale_role_diagram_json()

    with caplog.at_level(logging.WARNING, logger=CLASS_DIAGRAM_LOGGER):
        domain_model = process_class_diagram(json_data)

    # 1. The warning fired.
    matching_records = [
        record for record in caplog.records
        if "members" in record.message
        and "User" in record.message
        and "stale role name" in record.message
    ]
    assert matching_records, (
        f"Expected a stale-role warning mentioning 'members' and 'User'. "
        f"Got records: {[r.message for r in caplog.records]}"
    )

    # 2. The model was NOT auto-renamed -- the role is still ``members``.
    assoc, end = _get_user_end_named_members(domain_model)
    assert end is not None, (
        "Expected the User-typed end to still be named 'members' "
        "after process_class_diagram (this pass must NOT auto-rename)."
    )
    assert end.name == "members"
    assert end.type.name == "User"


def test_stale_role_warning_via_project_input(caplog):
    """Same scenario, but wrapped in a ``ProjectInput`` to mirror the
    real visual-editor pathway. The warning must still fire."""
    diagram_json = _stale_role_diagram_json()

    # Build a ProjectInput around the class diagram. The processor still
    # consumes the underlying diagram JSON directly, but constructing the
    # outer model verifies the shape matches what the editor sends.
    project = ProjectInput(
        id="proj-1",
        type="ClassDiagram",
        name="LibraryProject",
        createdAt="2026-05-19T00:00:00",
        diagrams={
            "ClassDiagram": [
                DiagramInput(
                    id="diag-1",
                    title="LibraryModel",
                    model=diagram_json["model"],
                ),
            ],
        },
    )
    active = project.get_active_diagram("ClassDiagram")
    assert active is not None
    payload = {
        "title": active.title,
        "model": active.model,
    }

    with caplog.at_level(logging.WARNING, logger=CLASS_DIAGRAM_LOGGER):
        domain_model = process_class_diagram(payload)

    assert any(
        "members" in r.message and "User" in r.message and "stale role" in r.message
        for r in caplog.records
    ), [r.message for r in caplog.records]

    # Confirm the conservative no-auto-rename behaviour.
    _, end = _get_user_end_named_members(domain_model)
    assert end is not None and end.name == "members"


def test_intentional_alias_role_does_not_warn(caplog):
    """A role ``borrower`` on a ``Member`` end is an intentional alias and
    must NOT trigger the warning (regression test for the conservative
    decision NOT to auto-rename)."""
    json_data = {
        "title": "LibraryModel",
        "model": {
            "elements": {
                "cls-member": {
                    "id": "cls-member",
                    "name": "Member",
                    "type": "Class",
                    "owner": None,
                    "attributes": [],
                    "methods": [],
                },
                "cls-book": {
                    "id": "cls-book",
                    "name": "Book",
                    "type": "Class",
                    "owner": None,
                    "attributes": [],
                    "methods": [],
                },
            },
            "relationships": {
                "rel-borrows": {
                    "id": "rel-borrows",
                    "name": "borrows",
                    "type": "ClassBidirectional",
                    "source": {
                        "element": "cls-book",
                        "multiplicity": "0..*",
                        "role": "borrowedBooks",
                    },
                    "target": {
                        "element": "cls-member",
                        "multiplicity": "1",
                        "role": "borrower",
                    },
                },
            },
        },
    }

    with caplog.at_level(logging.WARNING, logger=CLASS_DIAGRAM_LOGGER):
        process_class_diagram(json_data)

    # ``borrower`` is in _INTENTIONAL_ROLE_ALIASES -> no warn.
    assert not any(
        "borrower" in r.message and "stale role name" in r.message
        for r in caplog.records
    ), [r.message for r in caplog.records]


def test_matching_role_name_does_not_warn(caplog):
    """A role that already matches its target (``users`` -> ``User``) must
    NOT trigger any stale-role warning."""
    json_data = {
        "title": "Model",
        "model": {
            "elements": {
                "cls-user": {
                    "id": "cls-user",
                    "name": "User",
                    "type": "Class",
                    "owner": None,
                    "attributes": [],
                    "methods": [],
                },
                "cls-group": {
                    "id": "cls-group",
                    "name": "Group",
                    "type": "Class",
                    "owner": None,
                    "attributes": [],
                    "methods": [],
                },
            },
            "relationships": {
                "rel-membership": {
                    "id": "rel-membership",
                    "name": "membership",
                    "type": "ClassBidirectional",
                    "source": {
                        "element": "cls-group",
                        "multiplicity": "0..*",
                        "role": "groups",
                    },
                    "target": {
                        "element": "cls-user",
                        "multiplicity": "0..*",
                        "role": "users",
                    },
                },
            },
        },
    }

    with caplog.at_level(logging.WARNING, logger=CLASS_DIAGRAM_LOGGER):
        process_class_diagram(json_data)

    assert not any(
        "stale role name" in r.message for r in caplog.records
    ), [r.message for r in caplog.records]


def test_semantic_collection_role_on_compound_class_does_not_warn(caplog):
    """A ``0..*`` collection role named for its meaning (``items``) on an end
    typed with a COMPOUND class name (``OrderItem``) is a legitimate, idiomatic
    role — the modeling agent generates exactly these. The stem ``item`` is
    CONTAINED in ``OrderItem``, which is the opposite of a stale rename, so the
    stale-role warning must NOT fire (regression for the false positive)."""
    json_data = {
        "title": "ShopModel",
        "model": {
            "elements": {
                "cls-order": {
                    "id": "cls-order", "name": "Order", "type": "Class",
                    "owner": None, "attributes": [], "methods": [],
                },
                "cls-orderitem": {
                    "id": "cls-orderitem", "name": "OrderItem", "type": "Class",
                    "owner": None, "attributes": [], "methods": [],
                },
            },
            "relationships": {
                "rel-contains": {
                    "id": "rel-contains",
                    "name": "contains",
                    "type": "ClassBidirectional",
                    "source": {
                        "element": "cls-order",
                        "multiplicity": "1",
                        "role": "order",
                    },
                    "target": {
                        "element": "cls-orderitem",
                        "multiplicity": "0..*",
                        "role": "items",  # semantic collection role for a compound class
                    },
                },
            },
        },
    }

    with caplog.at_level(logging.WARNING, logger=CLASS_DIAGRAM_LOGGER):
        process_class_diagram(json_data)

    assert not any(
        "stale role name" in r.message and "items" in r.message
        for r in caplog.records
    ), [r.message for r in caplog.records]
