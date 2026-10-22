Incremental constraint analysis
===============================

An OCL invariant is checked after the population of the model changes.
Chapter 4 of Cabot and Teniente, *Incremental Integrity Checking in
UML/OCL Conceptual Schemas*, names the elementary changes that can
violate each invariant. BESSER computes that set on demand. It does not
run while parsing, and it does not change the ``Constraint`` metaclass.

The events are
:class:`~besser.BUML.metamodel.structural_event.structural_event.ViolatingEvent`
links between the constraint and one structural event. The text they were
computed from is ``constraint.expression``. Operators outside the chapter-4
fragment are over-approximated: extra events are kept and marked
``approximated``. Only invariants are analyzed. Preconditions, postconditions
and initializers are left untouched.

The thesis calls a class ``ET`` and an association ``RT``. The metaclasses
use ``Object`` and ``Link``.

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

An object event is linked to the class whose instances change. A link event
is linked to the association whose links change. An attribute update is
linked to the property whose value changes. The class of that object is the
property's owner. ``SpecializeObject`` and ``GeneralizeObject`` each name
one class. The other class is read from the generalization hierarchy when
the event is drawn.

Each :class:`~besser.BUML.metamodel.structural_event.structural_event.ViolatingEvent`
links exactly one constraint and exactly one event. ``approximated`` is an
attribute of that link.

Computing the events
--------------------

Parse the invariant with :func:`~besser.BUML.notations.ocl.api.parse_ocl`, then call
:func:`besser.BUML.notations.ocl.incremental.analyze_invariant`.
:func:`besser.BUML.metamodel.structural_event.structural_event.violating_events`
returns the links::

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

``analyze_invariants(model)`` does the same for every invariant on a
domain model. Running the analysis again replaces the previous links.

Running example
---------------

Figure 1.2 and Table 4.7 of the thesis are the reference. The five
invariants below are that example. ``Time.now()`` is written
``Date::now()`` because that is the date literal B-OCL parses. The
literal adds no structural event, so the sets are the ones in the
table. The thesis name ``ET`` is written ``Object`` and ``RT`` is
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
``ValidShipDate`` does not include ``InsertObject(Sale)``,
``DeleteObject(Sale)``, ``InsertObject(Shipment)`` or
``DeleteObject(Shipment)``. A new sale or shipment violates it only once
it is linked, and that is ``InsertLink(DeliveredIn)``.
``AtLeastThreeCustomers`` does not include ``DeleteObject(Customer)``.
Deleting a customer who is not linked to the category does not change
the count. The link deletion ``DeleteLink(BelongsTo)`` does.

Simplifying a constraint
------------------------

Chapter 3 rewrites an invariant into a smaller set of OCL operators before
later steps look at it. BESSER does that with the same parser: the text is
parsed to an AST and :func:`besser.BUML.notations.ocl.simplify_constraint`
rewrites the tree until no rule applies. The context class is left as it
is. CorrectProduct' from the thesis::

    context Product inv CorrectProduct:
        not Product.allInstances()->exists(p | p.price <= 0 or p.maxDiscount > 60)

becomes ``self.price > 0 and self.maxDiscount <= 60``. The constraint then
shows that normalized expression. The text from before the rewrite is kept
as a comment linked to the constraint, the same comment relationship a
class already uses. When the rewrite does not change the expression, no
comment is added. Which chapter-3 rules are applied, and which table rows
the OCL grammar cannot express, is recorded with the normalizer
(:doc:`/buml_language/model_types/ocl/normalization`).

Web modeling editor
-------------------

On a class diagram, **Generate → Constraints** has two actions.
**Constraint simplification**
(``POST /besser_api/ocl/constraint-simplification``) rewrites every
invariant and, when the text changes, attaches the original expression as
a comment on that constraint.
**Incremental pre-analysis** (``POST /besser_api/ocl/incremental-preanalysis``)
computes the events for the text currently in each box. Opening an OCL
constraint afterwards lists the events. The editor remembers the expression
that was analyzed, and shows the list as out of date if the constraint text
changes afterwards. Choosing the menu item again discards the stored events
and computes a new list from the current constraint text. Simplifying an
invariant changes that text, so a previous event list is shown as out of
date until pre-analysis is run again.

API
---

.. automodule:: besser.BUML.metamodel.structural_event.structural_event
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: besser.BUML.notations.ocl.incremental.analyze
   :members:
   :undoc-members:

.. automodule:: besser.BUML.notations.ocl.normalization.simplify
   :members:
   :undoc-members:
