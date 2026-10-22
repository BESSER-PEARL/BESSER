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
post-order until no rule applies. A ``max_iterations`` guard stops a
rule set that fails to reach a fixed point.

The rules follow chapter 3 of Cabot's thesis. Two chapter-3 shapes are
kept in an equivalent form so the fixed point does not undo itself:

- ``X->exists(Y)`` becomes ``not X->forAll(not Y)``, which is how
  ``not exists`` reaches the ``forAll(not …)`` step of CorrectProduct'.
  The table also writes ``exists`` as ``select(Y)->size() > 0``; applying
  both directions would cycle, and ``select->size() > 0`` is rewritten on
  to the ``forAll`` form.
- ``not (X = Y)`` becomes ``X <> Y``. The table's ``X <> Y → not (X = Y)``
  is the opposite direction and is not applied.

``xor`` and a boolean ``if`` use the chapter's conjunctive shapes,
``(X or Y) and (not X or not Y)`` and ``(not X or Y) and (X or Z)``.
``or`` is distributed over ``and``. ``allInstances()->forAll`` over the
context class is removed only when the body does not already mention
``self``; with several iterators only the first is replaced.

B-OCL has no production for ``count``, ``at``, ``includesAll``,
``excludesAll``, ``excluding``, ``including``, ``one``, ``any``,
``asSequence``, or a collection literal, so those table rows are not
rewrites. Derived elements, query operations, and ``let`` variables are
not unfolded, and a short-form iterator (no variable before ``|``) is
left implicit. Multiplicity collapses from the 2007 paper
(``size() > 0`` on a mandatory chain, ``forAll`` on a chain of maximum
1) also run; chapter 3 does not list them.

The six invariants on page 8 of that paper are checked as same-context
inputs. MaxSalary, ValidAge, AtLeastTwoProjectManagers and ValidNHours
reach the same-context row of Table 5. NotBossFreelance continues to
conjunctive form, so ``implies`` is removed. PossibleEmployee keeps the
comparison operands in the order written on page 8. The other Table 5
rows use a different context type, which this normalizer does not change.
On the context-changed MaxSalary, the multiplicity rules stop at
``self.employer.employee->forAll(...)``; dropping that round trip is
§3.1.3 rule 3 and is not applied.

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
