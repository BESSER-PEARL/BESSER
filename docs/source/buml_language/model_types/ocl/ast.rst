The OCL AST
===========

The classes in :mod:`besser.BUML.metamodel.ocl.ocl` define the abstract
syntax tree produced by :func:`~besser.BUML.notations.ocl.api.parse_ocl`.
A tool that translates OCL into another language (e.g. evaluating
constraints, encoding them into a proof assistant, emitting them as
target-specific assertions) walks this tree.

Inheritance
-----------

::

   Element
     └── NamedElement
           └── TypedElement
                 └── OCLExpression  ◄── base class for every node
                       ├── OperationCallExpression
                       ├── PropertyCallExpression
                       ├── VariableExp
                       ├── TypeExp
                       ├── IfExp
                       ├── LoopExp
                       ├── IteratorExp
                       └── LiteralExpression
                             ├── IntegerLiteralExpression
                             ├── RealLiteralExpression
                             ├── BooleanLiteralExpression
                             ├── StringLiteralExpression
                             └── DateLiteralExpression

Major node types
----------------

:class:`OperationCallExpression`
    A binary or unary operation. Notable shape: ``arguments`` is a list of
    operands **plus** the operator marker. For a binary op, the layout is
    ``[lhs, InfixOperator, rhs]``. The ``InfixOperator`` entry is *not* an
    :class:`OCLExpression`; consumers walking ``arguments`` should filter
    on ``isinstance(child, OCLExpression)`` when they only want operands.

    The operator name is also exposed on the call as ``operation`` (e.g.
    ``'>='``, ``'and'``, ``'+'``).

:class:`PropertyCallExpression`
    A property access. The ``source`` attribute is the receiver (typically
    another :class:`OCLExpression`, e.g. ``self``); the ``referredProperty``
    is the resolved :class:`~besser.BUML.metamodel.structural.Property` from
    the domain model.

    For chained access like ``self.account.balance``, parsing produces a
    nested chain of :class:`PropertyCallExpression` nodes via the
    ``source`` link.

:class:`VariableExp`
    A reference to a name introduced by a quantifier (``forAll(e | ...)``)
    or a ``let`` binding. The ``referredVariable`` points at the
    :class:`Variable` that introduces the name.

:class:`TypeExp`
    A type reference, used by ``oclIsTypeOf(T)`` /
    ``oclAsType(T)`` / ``oclIsKindOf(T)``. The ``referredType`` is the
    target :class:`~besser.BUML.metamodel.structural.Type`.

:class:`IfExp`
    A conditional. Children: ``ifCondition``, ``thenExpression``,
    ``elseExpression`` — all :class:`OCLExpression`.

:class:`LoopExp` and :class:`IteratorExp`
    Quantifiers and bulk operations over collections. ``IteratorExp``
    covers ``forAll`` / ``exists`` / ``select`` / ``reject`` / ``collect``;
    its ``body`` is the expression evaluated for each element, and
    ``iterators`` are the bound :class:`Variable` instances.

Literals
    The literal subclasses (:class:`IntegerLiteralExpression`,
    :class:`RealLiteralExpression`, :class:`BooleanLiteralExpression`,
    :class:`StringLiteralExpression`, :class:`DateLiteralExpression`) hold
    the parsed value on a ``value`` attribute. Booleans are coerced to
    Python ``bool`` by the wrapping visitor.

The wrapping visitor
--------------------

``parse_ocl`` uses :class:`besser.BUML.notations.ocl.wrapping_visitor.WrappingVisitor`,
which post-processes the raw ANTLR-derived AST to:

* Reconstruct property chains so that ``self.account.balance`` is a clean
  chain of :class:`PropertyCallExpression` nodes (rather than a flat list
  of names).
* Coerce boolean literals from ANTLR strings to Python ``bool``.
* Resolve property and type references against the supplied
  :class:`DomainModel`.

A consumer walking the AST should expect the wrapped shape, not the raw
ANTLR shape. The non-wrapping :class:`BOCLVisitorImpl` is an internal
helper used for source-location population during parsing — direct use
is rarely needed.
