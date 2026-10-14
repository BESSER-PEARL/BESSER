"""The stale-role-name warning is DISABLED.

It was meant to flag a role name left stale by a visual-editor class rename
(the editor rebuilds via ``Class.__init__``, bypassing the ``Class.name``
setter's role propagation). But the heuristic could NOT distinguish a stale
rename from a perfectly legitimate collection role: ``tasks`` on a ``TodoItem``
end is structurally identical to ``members`` on a ``User`` end (plural stem,
not contained in the target class name, no matching class), so it fired on
essentially every idiomatic collection role in freshly generated models.

These tests pin the decision: ``process_class_diagram`` must NOT emit a
stale-role warning for ANY of these shapes, and must still leave role names
untouched (it never auto-renamed and still must not). The correct fix for the
underlying editor rename bug is to propagate renames into the relationship
JSON's ``role`` fields at the editor / JSON layer.
"""

import logging

import pytest

from besser.BUML.metamodel.structural import BinaryAssociation, Class
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
    process_class_diagram,
)


CLASS_DIAGRAM_LOGGER = (
    "besser.utilities.web_modeling_editor.backend.services."
    "converters.json_to_buml.class_diagram_processor"
)


def _two_class_diagram(src_name, tgt_name, src_role, tgt_role,
                       src_mult="0..*", tgt_mult="0..*"):
    """A minimal bidirectional association ``src -> tgt`` with the given role
    names/multiplicities, in the v4 JSON shape the visual editor POSTs."""
    return {
        "title": "Model",
        "model": {
            "version": "4.0.0",
            "type": "ClassDiagram",
            "nodes": [
                {"id": "cls-src", "type": "class",
                 "position": {"x": 0, "y": 0}, "width": 160, "height": 100,
                 "data": {"name": src_name, "stereotype": None,
                          "attributes": [], "methods": []}},
                {"id": "cls-tgt", "type": "class",
                 "position": {"x": 300, "y": 0}, "width": 160, "height": 100,
                 "data": {"name": tgt_name, "stereotype": None,
                          "attributes": [], "methods": []}},
            ],
            "edges": [
                {"id": "rel", "type": "ClassBidirectional",
                 "source": "cls-src", "target": "cls-tgt",
                 "data": {"name": "assoc",
                          "sourceRole": src_role, "sourceMultiplicity": src_mult,
                          "targetRole": tgt_role, "targetMultiplicity": tgt_mult,
                          "points": []}},
            ],
        },
    }


def _stale_warnings(caplog):
    return [r.message for r in caplog.records if "stale role" in r.message]


@pytest.mark.parametrize(
    ("src", "tgt", "src_role", "tgt_role", "label"),
    [
        # The reported case: a freshly generated Todo model.
        ("TodoList", "TodoItem", "list", "tasks", "tasks -> TodoItem"),
        # The old "true positive" — actually indistinguishable from idiomatic.
        ("Book", "User", "books", "members", "members -> User"),
        # Semantic collection role on a compound class name.
        ("Order", "OrderItem", "order", "items", "items -> OrderItem"),
        # An intentional alias.
        ("Book", "Member", "borrowedBooks", "borrower", "borrower -> Member"),
        # Role already matches its target class.
        ("Group", "User", "groups", "users", "users -> User"),
    ],
)
def test_no_stale_role_warning_for_any_shape(caplog, src, tgt, src_role,
                                             tgt_role, label):
    """No association-role shape may surface a stale-role warning."""
    json_data = _two_class_diagram(src, tgt, src_role, tgt_role)
    with caplog.at_level(logging.WARNING, logger=CLASS_DIAGRAM_LOGGER):
        process_class_diagram(json_data)
    assert not _stale_warnings(caplog), (
        f"[{label}] unexpected stale-role warning(s): {_stale_warnings(caplog)}"
    )


def test_reported_case_tasks_on_todoitem_does_not_warn_and_keeps_role(caplog):
    """Regression for the user report: a ``tasks`` role on a ``TodoItem`` end
    must NOT warn, and the role name must be left untouched (no auto-rename)."""
    json_data = _two_class_diagram("TodoList", "TodoItem", "list", "tasks")
    with caplog.at_level(logging.WARNING, logger=CLASS_DIAGRAM_LOGGER):
        domain_model = process_class_diagram(json_data)

    assert not any("stale role" in r.message for r in caplog.records), (
        [r.message for r in caplog.records]
    )
    roles = {
        end.name
        for assoc in domain_model.associations
        if isinstance(assoc, BinaryAssociation)
        for end in assoc.ends
        if isinstance(end.type, Class)
    }
    assert "tasks" in roles, f"role 'tasks' was altered: {roles}"
