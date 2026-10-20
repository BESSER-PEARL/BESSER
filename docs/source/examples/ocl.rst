Object Constraint Language (OCL)
================================

This page walks through a complete OCL example end-to-end: building a
small domain model, parsing OCL constraints into ASTs, attaching
preconditions and postconditions to a method, and inspecting the
parsed AST.

For language reference, see :doc:`../buml_language/model_types/ocl`.

New to OCL?
-----------

An OCL constraint is a rule that every object of a class must satisfy. It names
the class it applies to (its *context*), the kind of rule (``inv`` for an
invariant, which must always hold), an optional name, and a Boolean expression
in which ``self`` is the object being checked. For a step-by-step introduction
to the language, see `The Ultimate Object Constraint Language (OCL) tutorial
<https://modeling-languages.com/ocl-tutorial/>`_.

The patterns below cover most everyday rules. They assume a ``Library`` with a
``name``, a ``Book`` with a ``title``, ``pages`` and ``price``, and an
association whose end ``books`` goes from a library to its books.

.. list-table::
  :header-rows: 1
  :widths: 30 70

  * - Rule
    - Constraint
  * - A value must be in a range
    - ``context Book inv positive_pages: self.pages > 0``
  * - Several conditions at once
    - ``context Book inv price_range: self.price >= 0 and self.price <= 500``
  * - A text must not be empty
    - ``context Library inv has_name: self.name <> ''``
  * - A rule that applies only in some cases
    - ``context Book inv expensive_means_long: self.price > 100 implies self.pages > 200``
  * - A collection must not be empty
    - ``context Library inv at_least_one_book: self.books->size() > 0``
  * - Every element must satisfy a condition
    - ``context Library inv all_books_have_pages: self.books->forAll(b | b.pages > 0)``
  * - At least one element must satisfy a condition
    - ``context Library inv has_long_book: self.books->exists(b | b.pages > 300)``
  * - Count the elements that satisfy a condition
    - ``context Library inv few_expensive_books: self.books->select(b | b.price > 100)->size() <= 5``

Navigation uses ``.`` for attributes and association ends (``self.books``) and
``->`` for operations on collections (``size``, ``forAll``, ``exists``,
``select``). The editor's validator does not evaluate every construct against
an object diagram yet; see :ref:`ocl-object-diagram-validation`.

Domain model
------------

Take a tiny banking model — an ``Account`` class with a balance and an
active flag, and a ``deposit`` operation:

.. code-block:: python

  from besser.BUML.metamodel.structural import (
      DomainModel, Class, Property, Method, Parameter, IntegerType, BooleanType,
  )

  account = Class("Account", attributes={
      Property("balance", IntegerType),
      Property("is_active", BooleanType),
  })
  model = DomainModel("BankingModel", types={account})

  deposit = Method(
      name="deposit",
      parameters=[Parameter("amount", IntegerType)],
      type=IntegerType,
  )

Parsing constraints
-------------------

Use :func:`~besser.BUML.notations.ocl.api.parse_ocl` to turn OCL source
text into an :class:`OCLConstraint`. The returned constraint exposes both
the source-text form (``.expression``) and the parsed AST (``.ast``):

.. code-block:: python

  from besser.BUML.notations.ocl.api import parse_ocl

  invariant = parse_ocl(
      "context Account inv: self.balance >= 0",
      model,
      context_class=account,
  )

  print(invariant.expression)
  # 'self.balance >= 0'

  print(type(invariant.ast).__name__, invariant.ast.operation)
  # OperationCallExpression >=

Preconditions and postconditions on the method
----------------------------------------------

Method contracts are anchored on the :class:`Method` itself via the
``pre`` and ``post`` first-class fields:

.. code-block:: python

  pre  = parse_ocl(
      "context Account::deposit(amount: Integer) pre: self.is_active and amount > 0",
      model, context_class=account,
  )
  post = parse_ocl(
      "context Account::deposit(amount: Integer) post: self.balance >= 0",
      model, context_class=account,
  )
  pre.name, post.name = "deposit_pre_active_and_positive", "deposit_post_nonneg"

  deposit.add_pre(pre)
  deposit.add_post(post)

  [c.name for c in deposit.pre]   # ['deposit_pre_active_and_positive']
  [c.name for c in deposit.post]  # ['deposit_post_nonneg']

The full BOCL header (``context Class::method(params) pre|post:``) is required
when calling ``parse_ocl`` directly with a precondition or postcondition —
``context_class`` must also be passed explicitly because the parser's
auto-detect regex only handles the simpler invariant header shape.

``add_pre`` / ``add_post`` raise ``ValueError`` if a precondition or
postcondition with the same name already exists on the method.

Normalization and pretty-printing
---------------------------------

Sugar operators like ``implies`` can be rewritten into a smaller core
(``not`` / ``or``) using the normalizer — handy when you only want to
implement semantics for a minimal set of operators in your consumer:

.. code-block:: python

  from besser.BUML.notations.ocl.normalization.normalize import normalize
  from besser.BUML.notations.ocl.pretty_printer import pretty_print

  c = parse_ocl(
      "context Account inv: self.is_active implies self.balance >= 0",
      model, context_class=account,
  )

  print(pretty_print(c.ast))
  # 'self.is_active implies self.balance >= 0'

  c_norm = normalize(c, model)
  print(pretty_print(c_norm.ast))
  # 'not self.is_active or self.balance >= 0'

Validating with the B-OCL Interpreter
-------------------------------------

To evaluate parsed OCL constraints against an object model, use the
external `B-OCL Interpreter <https://github.com/BESSER-PEARL/B-OCL-Interpreter>`_.
The interpreter consumes the same :class:`OCLConstraint` objects produced
by ``parse_ocl``.
