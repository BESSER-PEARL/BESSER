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

Source-location tracking
------------------------

Every :class:`~besser.BUML.metamodel.ocl.ocl.OCLExpression` node returned by
``parse_ocl`` carries the location of its origin in the OCL source:

* ``node.line`` — 1-indexed line number, or ``None`` if not populated.
* ``node.col`` — 0-indexed column number, or ``None`` if not populated.
* ``node.source_text`` — the OCL source text spanned by this node, as
  concatenated by ANTLR's ``ctx.getText()``. Whitespace is stripped from
  the spanned region; this is sufficient for "underline this region"
  diagnostics, but not for byte-accurate substitutions back into the
  original source.

Locations are populated automatically by ``BOCLVisitorImpl.visit`` and
preserved through :func:`~besser.BUML.metamodel.ocl.clone.clone` and
:func:`~besser.BUML.notations.ocl.normalization.normalize.normalize`.

For synthetic nodes built outside the parser, use
``OCLExpression.copy_location_from(other)`` to inherit the origin of an
existing node. The helper only fills fields that are ``None`` on the
target, so a partially-populated synthetic node will not have its
existing location overwritten.

.. code-block:: python

  root = constraint.ast
  print(f"line={root.line} col={root.col}")
  print(f"text={root.source_text!r}")

  for child in root.arguments:
      # InfixOperator entries are not OCLExpression nodes and have no location.
      if hasattr(child, "line"):
          print(f"  child: line={child.line} text={child.source_text!r}")

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

  pre  = parse_ocl("context Account inv: self.is_active",    model, context_class=account)
  post = parse_ocl("context Account inv: self.balance >= 0", model, context_class=account)
  pre.name, post.name = "deposit_pre_active", "deposit_post_nonneg"

  deposit.add_pre(pre)
  deposit.add_post(post)

``add_pre`` / ``add_post`` enforce name-uniqueness within the operation's
pre- / postcondition list. Constraints attached this way are now
discoverable structurally — downstream consumers no longer need to scrape
constraint names by convention.
