"""Structural events that can violate an OCL invariant.

The event types follow the classification in Cabot and Teniente,
*Incremental Integrity Checking in UML/OCL Conceptual Schemas*
(chapter 4): an elementary change in the population of a class or an
association. ``Class`` replaces the thesis name ``ET`` and ``Association``
replaces ``RT``.
"""

from besser.BUML.metamodel.structural_event.structural_event import (
    AssociationEvent,
    ClassEvent,
    DeleteAssociation,
    DeleteClass,
    GeneralizeClass,
    InsertAssociation,
    InsertClass,
    SpecializeClass,
    StructuralEvent,
    UpdateAttribute,
    ViolatingEvent,
    event_from_dict,
    event_payloads,
    event_to_dict,
)

__all__ = [
    "AssociationEvent",
    "ClassEvent",
    "DeleteAssociation",
    "DeleteClass",
    "GeneralizeClass",
    "InsertAssociation",
    "InsertClass",
    "SpecializeClass",
    "StructuralEvent",
    "UpdateAttribute",
    "ViolatingEvent",
    "event_from_dict",
    "event_payloads",
    "event_to_dict",
]
