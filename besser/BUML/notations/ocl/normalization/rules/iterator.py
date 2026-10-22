"""Paper §2.1 Table 3 — iterator simplifications.

- ``exists`` rewritten to ``not(forAll(not …))``.
- ``size(select(X, p)) = 0`` rewritten to ``X->forAll(v | not p)``.
- ``size(select(X, p)) > 0`` rewritten to ``X->exists(v | p)`` (which then
  reduces via :class:`ExistsToForAll`).
- ``select(X, p)->forAll(v | Y)`` fused to ``X->forAll(v | p implies Y)``.
"""

from besser.BUML.metamodel.ocl import (
    clone, is_and, is_loop, is_loop_with_n_iterators, is_op, is_or, is_size,
    is_comparison, substitute,
)
from besser.BUML.metamodel.ocl.ocl import IteratorExp, LoopExp, PropertyCallExpression, VariableExp
from besser.BUML.metamodel.structural import Property
from besser.BUML.notations.ocl.normalization.rules._helpers import (
    expressions_equal, is_zero_literal, make_and, make_comparison, make_forall,
    make_not, make_or, make_property_nav, size_compared_with_zero,
)


class ExistsToForAll:
    name = "ExistsToForAll"

    def applies(self, node, ctx):
        return is_loop(node, "exists") and is_loop_with_n_iterators(node, 1)

    def rewrite(self, node, ctx):
        # X->exists(v | Y)  →  not (X->forAll(v | not Y))
        new_loop = LoopExp("forAll", None)
        new_loop.source = clone(node.source)
        for it in node.iterator:
            new_loop.addIterator(clone(it))
        if node.body:
            new_loop.add_body(make_not(node.body[0]))
        return make_not(new_loop)


def _is_size_of_select(node):
    return (is_size(node) and is_loop(node.source, "select")
            and is_loop_with_n_iterators(node.source, 1))


class SelectSizeZero:
    """``X->select(v | p)->size() = 0``  →  ``X->forAll(v | not p)``."""
    name = "SelectSizeZero"

    def applies(self, node, ctx):
        if not is_comparison(node) or node.operation != "=":
            return False
        return ((_is_size_of_select(node.arguments[0])
                 and is_zero_literal(node.arguments[2]))
                or (_is_size_of_select(node.arguments[2])
                    and is_zero_literal(node.arguments[0])))

    def rewrite(self, node, ctx):
        size = (node.arguments[0]
                if _is_size_of_select(node.arguments[0])
                else node.arguments[2])
        select_loop = size.source
        new_loop = LoopExp("forAll", None)
        new_loop.source = clone(select_loop.source)
        for it in select_loop.iterator:
            new_loop.addIterator(clone(it))
        new_loop.add_body(make_not(select_loop.body[0]))
        return new_loop


class SelectSizeGtZero:
    """``X->select(v | p)->size() > 0``  →  ``X->exists(v | p)`` (then ExistsToForAll)."""
    name = "SelectSizeGtZero"

    def applies(self, node, ctx):
        if not is_comparison(node) or node.operation != ">":
            return False
        return (_is_size_of_select(node.arguments[0])
                and is_zero_literal(node.arguments[2]))

    def rewrite(self, node, ctx):
        size = node.arguments[0]
        select_loop = size.source
        new_loop = LoopExp("exists", None)
        new_loop.source = clone(select_loop.source)
        for it in select_loop.iterator:
            new_loop.addIterator(clone(it))
        new_loop.add_body(clone(select_loop.body[0]))
        return new_loop


class SelectForAllFusion:
    """``select(X, v1 | p)->forAll(v2 | Y)``  →  ``X->forAll(v2 | p[v2/v1] implies Y)``."""
    name = "SelectForAllFusion"

    def applies(self, node, ctx):
        return (is_loop(node, "forAll")
                and is_loop_with_n_iterators(node, 1)
                and is_loop(node.source, "select")
                and is_loop_with_n_iterators(node.source, 1)
                and node.body)

    def rewrite(self, node, ctx):
        from besser.BUML.metamodel.ocl.ocl import OperationCallExpression, VariableExp
        select_loop = node.source
        select_var = select_loop.iterator[0].name
        forall_var = node.iterator[0].name
        # Rename select's variable to forAll's variable in the predicate.
        renamed_pred = substitute(
            select_loop.body[0], select_var, VariableExp(forall_var, None),
        )
        forall_body = clone(node.body[0])

        # implies node — eliminated by the next pass.
        implies = OperationCallExpression(
            name="IMPLIES_BINARY", operation="implies",
            arguments=[renamed_pred, forall_body],
        )

        new_loop = LoopExp("forAll", None)
        new_loop.source = clone(select_loop.source)
        for it in node.iterator:
            new_loop.addIterator(clone(it))
        new_loop.add_body(implies)
        return new_loop


class SelectSizeEqualsSource:
    """``X->select(Y)->size() = X->size()`` → ``X->forAll(Y)``."""
    name = "SelectSizeEqualsSource"

    def applies(self, node, ctx):
        return _select_size_equals_source(node) is not None

    def rewrite(self, node, ctx):
        select_loop = _select_size_equals_source(node)
        return make_forall(select_loop.source, select_loop.iterator, select_loop.body[0])


def _select_size_equals_source(node):
    if not is_comparison(node) or node.operation != "=":
        return None
    for size_node, other in (
        (node.arguments[0], node.arguments[2]),
        (node.arguments[2], node.arguments[0]),
    ):
        if not _is_size_of_select(size_node) or not is_size(other):
            continue
        select_loop = size_node.source
        if expressions_equal(select_loop.source, other.source):
            return select_loop
    return None


class SelectExistsFusion:
    """``select(X, p)->exists(Y)`` → ``X->exists(p and Y)``."""
    name = "SelectExistsFusion"

    def applies(self, node, ctx):
        return (is_loop(node, "exists")
                and is_loop_with_n_iterators(node, 1)
                and is_loop(node.source, "select")
                and is_loop_with_n_iterators(node.source, 1)
                and node.body)

    def rewrite(self, node, ctx):
        select_loop = node.source
        renamed_pred = substitute(
            select_loop.body[0],
            select_loop.iterator[0].name,
            VariableExp(node.iterator[0].name, None),
        )
        new_loop = LoopExp("exists", None)
        new_loop.source = clone(select_loop.source)
        for iterator in node.iterator:
            new_loop.addIterator(clone(iterator))
        new_loop.add_body(make_and(renamed_pred, node.body[0]))
        return new_loop


class ForAllAndFusion:
    """``X->forAll(Y) and X->forAll(Z)`` → ``X->forAll(Y and Z)``."""
    name = "ForAllAndFusion"

    def applies(self, node, ctx):
        return _two_foralls(node) is not None

    def rewrite(self, node, ctx):
        left, right = _two_foralls(node)
        body = right.body[0]
        if left.iterator and right.iterator:
            body = substitute(
                right.body[0],
                right.iterator[0].name,
                VariableExp(left.iterator[0].name, None),
            )
        iterators = left.iterator or right.iterator
        return make_forall(left.source, iterators, make_and(left.body[0], body))


def _two_foralls(node):
    if not is_and(node):
        return None
    left, right = node.arguments
    if not (is_loop(left, "forAll") and is_loop(right, "forAll")):
        return None
    if not left.body or not right.body:
        return None
    if len(left.iterator) > 1 or len(right.iterator) > 1:
        return None
    if len(left.iterator) != len(right.iterator):
        return None
    if not expressions_equal(left.source, right.source):
        return None
    return left, right


class NestedForAllFusion:
    """``X->forAll(v | Y and X->forAll(v2 | Z))`` → ``X->forAll(v, v2 | Y and Z)``.

    The inner connective may be ``and`` or ``or``, as in the thesis.
    """
    name = "NestedForAllFusion"

    def applies(self, node, ctx):
        return _nested_forall(node) is not None

    def rewrite(self, node, ctx):
        connective, kept, inner = _nested_forall(node)
        outer_name = node.iterator[0].name
        inner_name = inner.iterator[0].name
        used = {outer_name}
        fresh = inner_name if inner_name not in used else _fresh_name(inner_name, used)
        inner_body = inner.body[0]
        if fresh != inner_name:
            inner_body = substitute(inner_body, inner_name, VariableExp(fresh, None))
        combined = (make_and if connective == "and" else make_or)(kept, inner_body)
        new_loop = LoopExp("forAll", None)
        new_loop.source = clone(node.source)
        new_loop.addIterator(clone(node.iterator[0]))
        new_loop.addIterator(IteratorExp(fresh, inner.iterator[0].type))
        new_loop.add_body(combined)
        return new_loop


def _nested_forall(node):
    if not is_loop(node, "forAll") or not is_loop_with_n_iterators(node, 1) or not node.body:
        return None
    body = node.body[0]
    if not (is_and(body) or is_or(body)):
        return None
    for kept, other in (
        (body.arguments[0], body.arguments[1]),
        (body.arguments[1], body.arguments[0]),
    ):
        if (is_loop(other, "forAll")
                and is_loop_with_n_iterators(other, 1)
                and other.body
                and expressions_equal(other.source, node.source)):
            return body.operation, kept, other
    return None


def _fresh_name(name, used):
    candidate = name
    index = 2
    while candidate in used:
        candidate = f"{name}{index}"
        index += 1
    return candidate


class UnionForAllSplit:
    """``X->union(Y).r->forAll(Z)`` → both sides ``forAll``, then ``and``."""
    name = "UnionForAllSplit"

    def applies(self, node, ctx):
        return _union_under_forall(node) is not None

    def rewrite(self, node, ctx):
        union, navigations = _union_under_forall(node)
        left = _navigate(union.source, navigations)
        right = _navigate(union.arguments[0], navigations)
        return make_and(
            make_forall(left, node.iterator, node.body[0]),
            make_forall(right, node.iterator, node.body[0]),
        )


def _union_under_forall(node):
    if not is_loop(node, "forAll") or not node.body:
        return None
    navigations = []
    current = node.source
    while isinstance(current, PropertyCallExpression):
        navigations.append(current.property)
        current = current.source
    navigations.reverse()
    if not is_op(current, "UNION") or not current.arguments:
        return None
    return current, navigations


def _navigate(root, properties):
    node = clone(root)
    for prop in properties:
        call = PropertyCallExpression(prop.name, prop)
        call.source = node
        node = call
    return node


class IsUniqueToForAll:
    """``X->isUnique(Y)`` → ``X->forAll(x1, x2 | x1 <> x2 implies x1.Y <> x2.Y)``."""
    name = "IsUniqueToForAll"

    def applies(self, node, ctx):
        return is_op(node, "ISUNIQUE") and _reroot(node.arguments[0], "x1") is not None

    def rewrite(self, node, ctx):
        argument = node.arguments[0]
        different = make_comparison(VariableExp("x1", None), "<>", VariableExp("x2", None))
        values_differ = make_comparison(
            _reroot(argument, "x1"), "<>", _reroot(argument, "x2"),
        )
        # implies is removed by the next pass: not (x1 <> x2) or x1.Y <> x2.Y
        body = make_or(make_not(different), values_differ)
        new_loop = LoopExp("forAll", None)
        new_loop.source = clone(node.source)
        new_loop.addIterator(IteratorExp("x1", None))
        new_loop.addIterator(IteratorExp("x2", None))
        new_loop.add_body(body)
        return new_loop


def _reroot(node, var_name):
    """Re-root a property navigation on ``var_name``, or ``None`` if it is not one."""
    if isinstance(node, Property):
        return make_property_nav(VariableExp(var_name, None), node)
    if not isinstance(node, PropertyCallExpression):
        return None
    cloned = clone(node)
    cursor = cloned
    while isinstance(cursor.source, PropertyCallExpression):
        cursor = cursor.source
    if isinstance(cursor.source, Property):
        cursor.source = make_property_nav(VariableExp(var_name, None), cursor.source)
        return cloned
    if cursor.source is None or isinstance(cursor.source, VariableExp):
        cursor.source = VariableExp(var_name, None)
        return cloned
    return None


class SizeOrForAll:
    """``X->size() <= 0 or X->forAll(Y)`` → ``X->forAll(Y)``."""
    name = "SizeOrForAll"

    def applies(self, node, ctx):
        return _size_or_forall(node) is not None

    def rewrite(self, node, ctx):
        return clone(_size_or_forall(node))


def _size_or_forall(node):
    if not is_or(node):
        return None
    for size_side, other in (
        (node.arguments[0], node.arguments[1]),
        (node.arguments[1], node.arguments[0]),
    ):
        size_call = size_compared_with_zero(size_side, {"=", "<="})
        if size_call is None or not is_loop(other, "forAll"):
            continue
        if expressions_equal(size_call.source, other.source):
            return other
    return None


ITERATOR_RULES = [
    # The size-of-select fast paths must fire before ExistsToForAll so they
    # produce a tighter rewrite when their pattern matches the comparison.
    SelectSizeZero(), SelectSizeGtZero(), SelectSizeEqualsSource(),
    SelectForAllFusion(), SelectExistsFusion(),
    ForAllAndFusion(), NestedForAllFusion(),
    UnionForAllSplit(),
    IsUniqueToForAll(),
    SizeOrForAll(),
    ExistsToForAll(),
]
