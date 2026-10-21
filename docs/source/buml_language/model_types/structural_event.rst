Structural events
=================

An OCL invariant is checked after the population of the model changes. Chapter 4
of Cabot and Teniente, *Incremental Integrity Checking in UML/OCL Conceptual
Schemas*, names the elementary changes that can matter. B-UML records those
changes as structural events. The thesis calls a class ``ET`` and an association
``RT``; the metaclasses below use ``Object`` and ``Link``.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Event
     - What it is linked to
   * - ``InsertObject``
     - A new object is added to a :class:`~besser.BUML.metamodel.structural.structural.Class`. It has no links yet.
   * - ``DeleteObject``
     - An object is removed from a class.
   * - ``UpdateAttribute``
     - The value of one :class:`~besser.BUML.metamodel.structural.structural.Property` of an object is replaced.
   * - ``SpecializeObject``
     - An existing object is reclassified into a subclass.
   * - ``GeneralizeObject``
     - An object leaves a subclass and is afterwards only an instance of the superclass.
   * - ``InsertLink``
     - A new link is added to an :class:`~besser.BUML.metamodel.structural.structural.Association`.
   * - ``DeleteLink``
     - A link is removed from an association.

An object event is linked to the class whose instances change. A link event is
linked to the association whose links change. An attribute update is linked to
the property whose value changes. The class of that object is the property's
owner.

:class:`~besser.BUML.metamodel.structural_event.structural_event.ViolatingEvent`
is the association class between a
:class:`~besser.BUML.metamodel.structural.structural.Constraint` and one
structural event. Each instance links exactly one constraint and exactly one
event. ``approximated`` is an attribute of that link: it is true when the
event was kept because part of the expression is outside the fragment the
analysis handles exactly. Dropping it could hide a real violation, so it is
kept.

The :class:`Constraint` metaclass is unchanged. The text the events were
computed from is the constraint's ``expression``. The links registered for a
constraint are returned by
:func:`~besser.BUML.metamodel.structural_event.structural_event.violating_events`.

How to compute the events, the ecommerce running example, and the web editor
menu are documented in
:doc:`../../utilities/incremental_constraint_analysis`.
