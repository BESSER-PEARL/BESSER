Normalization, pretty-printing, and cloning
===========================================

Three AST utilities accompany the parser. Each is independently useful
for building a downstream OCL consumer.

Normalization
-------------

:func:`besser.BUML.notations.ocl.normalization.normalize.normalize`
rewrites a parsed :class:`OCLConstraint` into a canonical form: syntactic
sugar (``implies``, ``xor``, etc.) is reduced to a smaller core of
operators (``not``, ``or``, ``and``). Targeting the canonical form lets a
downstream tool define semantics for fewer node shapes.

.. code-block:: python

  from besser.BUML.notations.ocl.api import parse_ocl
  from besser.BUML.notations.ocl.normalization.normalize import normalize

  c = parse_ocl(
      "context Account inv: self.is_active implies self.balance >= 0",
      model, context_class=account,
  )
  c_norm = normalize(c, model)
  # c_norm.ast is now an `or` of `(not self.is_active)` and the original RHS.

The driver runs each rule in a fixed order and walks the AST in
post-order until no rule applies. Each rule is designed to strictly
decrease a lexicographic measure ⟨#sugar-operators, AST size, #negations⟩,
so the pipeline reaches a fixpoint in a bounded number of iterations. A
``max_iterations`` guard surfaces a developer bug (a rule that violates
the measure) instead of looping forever.

The input is not mutated: ``normalize`` clones the AST internally before
rewriting.

Pretty-printing
---------------

:func:`besser.BUML.notations.ocl.pretty_printer.pretty_print` renders an
AST back into human-readable OCL source text. It is used by
:class:`OCLConstraint` itself to keep ``constraint.expression`` in sync
with ``constraint.ast``.

.. code-block:: python

  from besser.BUML.notations.ocl.pretty_printer import pretty_print

  pretty_print(c.ast)            # 'self.is_active implies self.balance >= 0'
  pretty_print(c)                # 'context Account inv: self.is_active implies self.balance >= 0'
  pretty_print(c_norm.ast)       # 'not self.is_active or self.balance >= 0'

For a parsed-then-normalized constraint, ``pretty_print(c.ast)`` is a
useful round-trip check that the AST is well-formed — the result should
itself be re-parseable by ``parse_ocl``.

Cloning
-------

:func:`besser.BUML.metamodel.ocl.clone.clone` returns a deep copy of an
AST node. Structural-model references (Properties, Classes, Types) are
**shared**, not duplicated — only the AST scaffolding is duplicated.

.. code-block:: python

  from besser.BUML.metamodel.ocl import clone

  fresh = clone(c.ast)
  # `fresh` and `c.ast` are independent trees, but
  # fresh.referredProperty is c.ast.referredProperty (same Property instance).

Use ``clone`` whenever you mutate an AST (rewriting, normalization,
encoding-time substitution) rather than mutating the parsed tree
in-place — leaving the parsed tree untouched lets you keep showing the
user's original source alongside any transformed view.
