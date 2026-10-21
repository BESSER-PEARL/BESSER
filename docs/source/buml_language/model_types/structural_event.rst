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

The events are computed on demand, not while parsing::

    from besser.BUML.notations.ocl import parse_ocl
    from besser.BUML.notations.ocl.incremental import analyze_invariant
    from besser.BUML.metamodel.structural_event import violating_events

    constraint = parse_ocl(
        "context Category inv AtLeastThreeCustomers: self.customer->size() >= 3",
        model,
    )
    analyze_invariant(constraint)
    for link in violating_events(constraint):
        print(link.structural_event)

``analyze_invariants(model)`` does the same for every invariant on a domain
model. Preconditions, postconditions and initializers are left untouched.

Running example
---------------

Figure 1.2 and Table 4.7 of the thesis are the reference. The five invariants
below are that example. ``Time.now()`` is written ``Date::now()`` because that
is the date literal B-OCL parses. The literal adds no structural event, so the
sets are the ones in the table. ``ET`` is written ``Object`` and ``RT`` is
written ``Link``.

.. code-block:: text

   context Sale inv ValidShipDate:
       self.shipment->forAll(sh | sh.plannedShipDate <= self.paymentDate + 30)
   context Product inv CorrectProduct:
       self.price > 0 and self.maxDiscount <= 60
   context Category inv NotTooPendingSales:
       self.customer->forAll(c |
           c.sale->select(s | s.paymentDate > Date::now())
               ->collect(sa | sa.amount)->sum() <= self.maxPendingAmount)
   context Category inv AtLeastThreeCustomers:
       self.customer->size() >= 3
   context RestrictedProduct inv NumberOfRestrictedProducts:
       RestrictedProduct.allInstances()->size() <= 20

.. list-table:: Events that may violate each invariant
   :header-rows: 1
   :widths: 34 66

   * - Constraint
     - Events
   * - ``ValidShipDate``
     - ``UpdateAttribute(paymentDate, Sale)``, ``UpdateAttribute(plannedShipDate, Shipment)``, ``InsertLink(DeliveredIn)``
   * - ``CorrectProduct``
     - ``InsertObject(Product)``, ``UpdateAttribute(price, Product)``, ``UpdateAttribute(maxDiscount, Product)``
   * - ``NotTooPendingSales``
     - ``UpdateAttribute(maxPendingAmount, Category)``, ``InsertLink(BelongsTo)``, ``InsertLink(Purchases)``, ``UpdateAttribute(amount, Sale)``, ``UpdateAttribute(paymentDate, Sale)``
   * - ``AtLeastThreeCustomers``
     - ``DeleteLink(BelongsTo)``, ``InsertObject(Category)``
   * - ``NumberOfRestrictedProducts``
     - ``InsertObject(RestrictedProduct)``, ``SpecializeObject(RestrictedProduct)``

Events that cannot violate an invariant on their own are absent.
``ValidShipDate`` does not include ``InsertObject(Sale)``, ``DeleteObject(Sale)``,
``InsertObject(Shipment)`` or ``DeleteObject(Shipment)``. A new sale or shipment
violates it only once it is linked, and that is ``InsertLink(DeliveredIn)``.
``AtLeastThreeCustomers`` does not include ``DeleteObject(Customer)``. Deleting
a customer who is not linked to the category does not change the count. The
link deletion ``DeleteLink(BelongsTo)`` does.

In the web modeling editor the same computation is the menu item
**Generate → Constraints → Incremental pre-analysis**. It is available on
class diagrams. Opening an OCL constraint afterwards lists the events. The
editor remembers the expression that was analyzed, and shows the list as out
of date if the constraint text changes afterwards.
