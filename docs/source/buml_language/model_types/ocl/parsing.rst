Parsing OCL constraints
=======================

The entry point for parsing OCL source into an AST is
:func:`besser.BUML.notations.ocl.api.parse_ocl`. It lexes the OCL source,
parses it against the BOCL grammar, builds an AST via the wrapping visitor,
and returns an :class:`OCLConstraint` whose ``ast`` is the parsed expression
and whose ``context`` is resolved against the supplied
:class:`~besser.BUML.metamodel.structural.DomainModel`.

Basic usage
-----------

.. code-block:: python

  from besser.BUML.metamodel.structural import (
      DomainModel, Class, Property, IntegerType, BooleanType,
  )
  from besser.BUML.notations.ocl.api import parse_ocl

  account = Class("Account", attributes={
      Property("balance", IntegerType),
      Property("is_active", BooleanType),
  })
  model = DomainModel("BankingModel", types={account})

  constraint = parse_ocl(
      "context Account inv: self.balance >= 0",
      model,
      context_class=account,
  )

  constraint.expression  # 'self.balance >= 0'   — pretty-printed source text (always str)
  constraint.ast         # <OperationCallExpression operation='>='>  — parsed AST (OCLExpression)
  constraint.context     # <Class name='Account'>

If ``context_class`` is omitted, the class is parsed from the ``context X
inv|pre|post|init`` header and resolved against ``model.types``.

.. note::

  The auto-detect only matches the simple ``context Class inv|pre|post|init``
  shape (used by invariants). Preconditions, postconditions, and init
  constraints carry the longer ``context Class::method(p: Type) pre|post:``
  or ``context Class::attribute : Type init:`` headers — for those you must
  pass ``context_class`` explicitly. The supported shapes are summarized in
  :doc:`../ocl`.

The ``expression`` / ``ast`` split
----------------------------------

Every :class:`OCLConstraint` carries two synchronized views of the same
constraint:

* ``constraint.expression`` is the **source text** as a string. It is the
  pretty-printer's output for ``constraint.ast`` and refreshes automatically
  when the AST is re-assigned. Use it for human-readable diagnostics.
* ``constraint.ast`` is the **parsed AST** as an
  :class:`~besser.BUML.metamodel.ocl.ocl.OCLExpression`. Use it when writing
  a downstream consumer (encoder, evaluator, transformer).

A non-OCL :class:`~besser.BUML.metamodel.structural.Constraint` only has
``expression`` (always ``str``); only :class:`OCLConstraint` exposes ``ast``.

Error handling
--------------

A malformed OCL source raises
:exc:`~besser.BUML.notations.ocl.error_handling.BOCLSyntaxError` with the
list of lex/parse errors collected by the BOCL error listener. The visitor
does not return a partial AST — failures are surfaced upfront.

.. code-block:: python

  from besser.BUML.notations.ocl.api import parse_ocl
  from besser.BUML.notations.ocl.error_handling import BOCLSyntaxError

  try:
      parse_ocl("context Account inv self.balance", model)  # missing ':'
  except BOCLSyntaxError as e:
      for err in e.errors:
          print(err)

If the supplied ``context_class`` is missing or cannot be resolved against
``model.types``, ``parse_ocl`` raises ``ValueError``.

Anchoring constraints on operations
-----------------------------------

Once parsed, an OCL constraint can be attached to an operation via the
first-class :attr:`~besser.BUML.metamodel.structural.Method.pre` and
:attr:`~besser.BUML.metamodel.structural.Method.post` fields:

.. code-block:: python

  from besser.BUML.metamodel.structural import Method, Parameter, IntegerType

  deposit = Method(
      name="deposit",
      parameters=[Parameter("amount", IntegerType)],
      type=IntegerType,
  )

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

``add_pre`` / ``add_post`` enforce name-uniqueness within the operation's
pre- / postcondition list. Constraints attached this way are now
discoverable structurally — downstream consumers no longer need to scrape
constraint names by convention.
