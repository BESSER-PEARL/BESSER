Structural events
=================

An OCL invariant is checked after the population of the model changes. Chapter 4
of Cabot and Teniente, *Incremental Integrity Checking in UML/OCL Conceptual
Schemas*, names the elementary changes that can matter. B-UML records those
changes as structural events. The thesis calls a class ``ET`` and an association
``RT``; the metaclasses below use ``Class`` and ``Association``.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Event
     - What changes
   * - ``InsertClass``
     - A new object is added to a class. It has no links yet.
   * - ``DeleteClass``
     - An object is removed from a class.
   * - ``UpdateAttribute``
     - The value of one attribute of an object is replaced.
   * - ``SpecializeClass``
     - An existing object is reclassified into a subclass.
   * - ``GeneralizeClass``
     - An object leaves a subclass and is afterwards only an instance of the superclass.
   * - ``InsertAssociation``
     - A new link is added to an association.
   * - ``DeleteAssociation``
     - A link is removed from an association.

A :class:`~besser.BUML.metamodel.structural_event.structural_event.ViolatingEvent` stores one of these events on the
:class:`~besser.BUML.metamodel.structural.structural.Constraint` it may
violate. The link lives in ``constraint.violating_events``. ``approximated``
is true when the event was kept because part of the expression is outside the
fragment the analysis handles exactly. Dropping it could hide a real
violation, so it is kept.

The events are computed on demand, not while parsing::

    from besser.BUML.notations.ocl import parse_ocl
    from besser.BUML.notations.ocl.incremental import analyze_invariant

    constraint = parse_ocl(
        "context Category inv AtLeastThreeCustomers: self.customer->size() >= 3",
        model,
    )
    analyze_invariant(constraint)
    for link in constraint.violating_events:
        print(link.event)

``analyze_invariants(model)`` does the same for every invariant on a domain
model. Preconditions, postconditions and initializers are left untouched.
Changing ``constraint.expression`` drops a stored result, because it was
computed from the previous text. ``violating_events_source`` keeps that text
so a caller can tell the two apart.

In the web modeling editor the same computation is the menu item
**Generate → Constraints → Incremental pre-analysis**. It is available on
class diagrams. Opening an OCL constraint afterwards lists the events. If the
constraint text has changed since the analysis, the list is shown as out of date.
