Object Constraint Language (OCL)
================================

This page walks through a complete OCL example end-to-end: building a
small domain model, parsing OCL constraints into ASTs, attaching
preconditions and postconditions to a method, and inspecting the
parsed AST.

For language reference, see :doc:`../buml_language/model_types/ocl`.

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

  pre  = parse_ocl("context Account inv: self.is_active",    model, context_class=account)
  post = parse_ocl("context Account inv: self.balance >= 0", model, context_class=account)
  pre.name, post.name = "deposit_pre_active", "deposit_post_nonneg"

  deposit.add_pre(pre)
  deposit.add_post(post)

  [c.name for c in deposit.pre]   # ['deposit_pre_active']
  [c.name for c in deposit.post]  # ['deposit_post_nonneg']

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
