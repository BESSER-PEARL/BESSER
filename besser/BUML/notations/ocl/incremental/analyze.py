"""Potentially-violating structural events of an OCL invariant.

Implements chapter 4 of Cabot and Teniente, *Incremental Integrity Checking
in UML/OCL Conceptual Schemas*: mark each node of the expression with the
direction of change that could falsify it (``+``, ``-``, or ``und``), then
read the structural event off that mark.

The walk follows the B-OCL AST produced by :func:`parse_ocl`. Operators
covered by the chapter's tables are handled exactly. Any other operator is
over-approximated: every structural event on a model element that
subexpression mentions is kept, and the extra events are marked
``approximated``. A missed event would let a later check skip a constraint
that might fail; an extra event only costs a check.

Only invariants are analyzed. Preconditions, postconditions and initializers
are left untouched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from besser.BUML.metamodel.ocl.ocl import (
    IfExp,
    InfixOperator,
    LiteralExpression,
    LoopExp,
    OCLConstraint,
    OperationCallExpression,
    PropertyCallExpression,
    TypeExp,
    VariableExp,
)
from besser.BUML.metamodel.ocl.walk import walk
from besser.BUML.metamodel.structural.structural import (
    Association,
    AssociationClass,
    Class,
    Constraint,
    DomainModel,
    Property,
)
from besser.BUML.metamodel.structural_event.structural_event import (
    DeleteAssociation,
    DeleteClass,
    GeneralizeClass,
    InsertAssociation,
    InsertClass,
    SpecializeClass,
    StructuralEvent,
    UpdateAttribute,
    ViolatingEvent,
)

PLUS = "+"
MINUS = "-"
UND = "und"

_KIND_RE = re.compile(r"\bcontext\b.*?\b(inv|pre|post|init)\b", re.IGNORECASE)
_COMP_UP = {"<", "<="}       # falsified by increasing the left operand
_COMP_DOWN = {">", ">="}     # falsified by decreasing the left operand
_COMP_BOTH = {"=", "<>"}
_ARITH = {"+", "-", "*", "/"}
_EXACT_LOOPS = {"forAll", "select", "collect"}
_SIZE_OPS = {"Size", "size"}
_SUM_OPS = {"Sum", "sum"}
_ALLINST = {"ALLInstances", "allInstances"}
_TYPE_OF = {"OCLISTYPEOF"}
_KIND_OF = {"OCLISKINDOF"}
_AS_TYPE = {"OCLASTYPE"}
_ABSORBING_PARENTS = {"forAll", "select", "collect", "Size", "size", "Sum", "sum"}


@dataclass(frozen=True)
class EventSpec:
    """One event type found on a node, before it is stored on the constraint."""

    kind: str
    target: object
    attribute: Property = None
    approximated: bool = False

    def identity(self) -> tuple:
        return (self.kind, self.target, self.attribute)

    def build(self) -> StructuralEvent:
        if self.kind == "InsertClass":
            return InsertClass(self.target)
        if self.kind == "DeleteClass":
            return DeleteClass(self.target)
        if self.kind == "SpecializeClass":
            return SpecializeClass(self.target)
        if self.kind == "GeneralizeClass":
            return GeneralizeClass(self.target)
        if self.kind == "UpdateAttribute":
            return UpdateAttribute(self.attribute, self.target)
        if self.kind == "InsertAssociation":
            return InsertAssociation(self.target)
        if self.kind == "DeleteAssociation":
            return DeleteAssociation(self.target)
        raise ValueError(f"unknown event kind {self.kind}")


@dataclass
class AnalysisResult:
    """Outcome of analyzing one invariant.

    Attributes:
        warnings: Human-readable notes. A non-empty list means the event set
            may contain events the exact tables would not have produced.
    """

    warnings: list[str]


@dataclass
class _Walk:
    """Context carried down the expression tree."""

    root: object
    context: Class
    in_body: bool
    approximating: bool
    select_decrease: bool
    ancestors: tuple


def analyze_invariants(model: DomainModel) -> dict[str, AnalysisResult]:
    """Analyze every invariant owned by ``model``.

    Preconditions and postconditions live on methods and are not visited.
    Constraints whose text is not an invariant are skipped.

    Returns:
        A map from constraint name to the analysis result. The events
        themselves are stored on each constraint.
    """
    results = {}
    for constraint in sorted(model.constraints, key=lambda c: c.name):
        if not _is_invariant(constraint):
            continue
        results[constraint.name] = analyze_invariant(constraint)
    return results


def analyze_invariant(constraint: Constraint) -> AnalysisResult:
    """Compute and store the events that may violate ``constraint``.

    Replaces any previously stored events. The constraint text is recorded in
    ``violating_events_source`` so a later edit can be told apart from this
    result.

    Args:
        constraint: An OCL invariant. A constraint of another kind is left
            unchanged and reported in the warnings.

    Returns:
        The warnings produced for this constraint.
    """
    if not _is_invariant(constraint):
        return AnalysisResult([
            f"'{constraint.name}' is not an invariant; incremental pre-analysis "
            f"only covers invariants."
        ])
    expression = _expression(constraint)
    if expression is None:
        return AnalysisResult([
            f"'{constraint.name}' has no parsed expression to analyze."
        ])

    walk_ctx = _Walk(
        root=expression,
        context=constraint.context,
        in_body=False,
        approximating=False,
        select_decrease=False,
        ancestors=(),
    )
    _local, found = _visit(expression, frozenset({UND}), walk_ctx)
    specs = _dedup(found)
    links = {
        ViolatingEvent(None, spec.build(), approximated=spec.approximated)
        for spec in specs
    }
    constraint.violating_events = links
    constraint.violating_events_source = constraint.expression
    warnings = []
    if any(spec.approximated for spec in specs):
        warnings.append(
            f"Part of '{constraint.name}' is outside the analyzed fragment; "
            f"the event set may contain extra events."
        )
    return AnalysisResult(warnings)


def _is_invariant(constraint: Constraint) -> bool:
    text = constraint.expression or ""
    match = _KIND_RE.search(text)
    if match is None:
        # A body with no header is treated as an invariant: that is how a
        # constraint built in Python, without a ``context`` line, is used.
        return True
    return match.group(1).lower() == "inv"


def _expression(constraint: Constraint):
    if isinstance(constraint, OCLConstraint):
        return constraint.ast
    return None


def _visit(node, received: frozenset, ctx: _Walk):
    """Return ``(local events, events in this subtree)``."""
    if node is None or isinstance(node, InfixOperator):
        return [], []
    if not received:
        received = frozenset({UND})

    if isinstance(node, LoopExp):
        return _visit_loop(node, received, ctx)
    if isinstance(node, OperationCallExpression):
        return _visit_operation(node, received, ctx)
    if isinstance(node, PropertyCallExpression):
        return _visit_feature(node.property, node.source, received, ctx, node)
    if isinstance(node, Property):
        return _visit_feature(node, None, received, ctx, node)
    if isinstance(node, VariableExp):
        local = _variable_events(node, received, ctx)
        return local, list(local)
    if isinstance(node, IfExp):
        return _visit_unknown(
            [node.ifCondition, node.thenExpression, node.elseCondition], received, ctx, node,
        )
    # Literals, type expressions and anything else with no model element.
    return [], []


def _visit_loop(node: LoopExp, received: frozenset, ctx: _Walk):
    if len(node.iterator) > 1 or node.name not in _EXACT_LOOPS:
        children = [node.source, *node.body]
        return _visit_unknown(children, received, ctx, node)

    source_marks, body_marks = _loop_marks(node.name, received)
    source_local, source_all = _descend(node.source, source_marks, ctx, node, in_body=ctx.in_body)
    body_ctx = replace(ctx, in_body=True, select_decrease=ctx.select_decrease or (
        node.name == "select" and MINUS in received
    ))
    body_events = []
    body_local_keys = []
    for body in node.body:
        local, whole = _descend(body, body_marks, body_ctx, node, in_body=True)
        body_local_keys.extend(local)
        body_events.extend(whole)
    if node.name == "select" and PLUS in received:
        body_events = _opposite_all(body_events)
    return [], source_all + body_events


def _visit_operation(node: OperationCallExpression, received: frozenset, ctx: _Walk):
    op = node.operation
    operands = _operands(node)

    if op in {"and", "or"}:
        found = []
        for child in operands:
            _local, whole = _descend(child, frozenset({UND}), ctx, node)
            found.extend(whole)
        return [], found

    if op == "not":
        if not operands:
            return [], []
        child = operands[0]
        local, whole = _descend(child, frozenset({UND}), ctx, node)
        if _is_undefined_test(child):
            return [], _opposite_all(whole)
        # Only the events generated on the child node itself are flipped.
        # Events deeper in its subtree stay, which is why ``not (a = b)``
        # keeps the same attribute updates as ``a = b``.
        kept = [event for event in whole if event.identity() not in {item.identity() for item in local}]
        return [], kept + _opposite_all(local)

    if op == "implies":
        if len(operands) < 2:
            return _visit_unknown(operands, received, ctx, node)
        left_local, left_all = _descend(operands[0], frozenset({UND}), ctx, node)
        _right_local, right_all = _descend(operands[1], frozenset({UND}), ctx, node)
        kept = [event for event in left_all if event.identity() not in {item.identity() for item in left_local}]
        return [], kept + _opposite_all(left_local) + right_all

    if op in _COMP_UP or op in _COMP_DOWN or op in _COMP_BOTH:
        if len(operands) < 2:
            return _visit_unknown(operands, received, ctx, node)
        left_marks, right_marks, exact = _comparison_marks(op, received)
        child_ctx = ctx if exact else replace(ctx, approximating=True)
        _ll, left_all = _descend(operands[0], left_marks, child_ctx, node)
        _rl, right_all = _descend(operands[1], right_marks, child_ctx, node)
        return [], left_all + right_all

    if op in _ARITH or op in _SIZE_OPS or op in _SUM_OPS:
        child_marks = _same_direction(received) if op in _SIZE_OPS or op in _SUM_OPS else _arith_marks(received)
        found = []
        # Size and sum apply to one collection. Arithmetic has the operands
        # of the operator, each of which moves in the direction of the result
        # (naturals and the signed case agree on ``+``; a ``-`` on a sum or a
        # size follows the same direction, which is what the running example
        # requires). Both directions are used when the parent asked for both.
        children = operands if op in _ARITH else operands[:1]
        for child in children:
            _local, whole = _descend(child, child_marks, ctx, node)
            found.extend(whole)
        return [], found

    if op in _ALLINST:
        local = _all_instances_events(node, received, ctx)
        return local, list(local)

    if op in _TYPE_OF or op in _KIND_OF:
        local = _type_test_events(node, ctx)
        found = list(local)
        for child in operands:
            if isinstance(child, TypeExp):
                continue
            _cl, whole = _descend(child, frozenset({UND}), ctx, node)
            found.extend(whole)
        return local, found

    if op in _AS_TYPE:
        found = []
        for child in operands:
            if isinstance(child, TypeExp):
                continue
            _cl, whole = _descend(child, received, ctx, node)
            found.extend(whole)
        return [], found

    return _visit_unknown(operands, received, ctx, node)


def _visit_feature(prop: Property, source, received: frozenset, ctx: _Walk, node):
    """An attribute access or a navigation, optionally with its source expression."""
    local = []
    if _is_navigation(prop):
        local.extend(_navigation_events(prop, received, ctx, node))
        child_marks = _same_direction(received)
    elif _is_attribute(prop):
        local.append(EventSpec(
            "UpdateAttribute", prop.owner, attribute=prop, approximated=False,
        ))
        child_marks = frozenset({PLUS})
    else:
        return _visit_unknown([source] if source is not None else [], received, ctx, node)

    found = list(local)
    if source is not None:
        _sl, whole = _descend(source, child_marks, ctx, node)
        found.extend(whole)
    elif not ctx.in_body:
        # ``self.price`` parsed without a source variable. The feature is
        # reachable from the context class, so the self-variable rules apply.
        synthetic = VariableExp("self", None)
        found.extend(_variable_events(synthetic, child_marks, replace(
            ctx, ancestors=ctx.ancestors + (node,),
        )))
    return local, found


def _visit_unknown(children, received: frozenset, ctx: _Walk, node):
    broad = frozenset({PLUS, MINUS, UND})
    child_ctx = replace(ctx, approximating=True)
    found = []
    for child in children:
        _local, whole = _descend(child, broad, child_ctx, node)
        found.extend(whole)
    return [], found


def _descend(child, marks: frozenset, ctx: _Walk, parent, in_body: bool = None):
    if child is None or isinstance(child, InfixOperator):
        return [], []
    child_ctx = _Walk(
        root=ctx.root,
        context=ctx.context,
        in_body=ctx.in_body if in_body is None else in_body,
        approximating=ctx.approximating,
        select_decrease=ctx.select_decrease,
        ancestors=ctx.ancestors + (parent,),
    )
    return _visit(child, marks, child_ctx)


def _loop_marks(name: str, received: frozenset):
    """Marks for the source collection and the body of a handled iterator."""
    if name in {"forAll", "select"}:
        source = set()
        only_decrease = MINUS in received and PLUS not in received and UND not in received
        source.add(MINUS if only_decrease else PLUS)
        return frozenset(source), frozenset({UND})
    # collect: the bag grows or shrinks with the source and with the body.
    source, body = set(), set()
    for mark in received:
        if mark == PLUS:
            source.add(PLUS)
            body.add(PLUS)
        elif mark == MINUS:
            source.add(MINUS)
            body.add(MINUS)
        else:
            source.add(UND)
            body.add(UND)
    if not source:
        source.add(UND)
        body.add(UND)
    return frozenset(source), frozenset(body)


def _comparison_marks(op: str, received: frozenset):
    """Return ``(left marks, right marks, exact)``.

    ``exact`` is false when the comparison was not used as a condition, so
    the children are over-approximated.
    """
    if received <= frozenset({UND}):
        if op in _COMP_UP:
            return frozenset({PLUS}), frozenset({MINUS}), True
        if op in _COMP_DOWN:
            return frozenset({MINUS}), frozenset({PLUS}), True
        return frozenset({PLUS, MINUS}), frozenset({PLUS, MINUS}), True
    return frozenset({PLUS, MINUS, UND}), frozenset({PLUS, MINUS, UND}), False


def _same_direction(received: frozenset) -> frozenset:
    marks = set()
    for mark in received:
        marks.add(mark if mark in {PLUS, MINUS} else UND)
    return frozenset(marks or {UND})


def _arith_marks(received: frozenset) -> frozenset:
    """Direction passed to both operands of ``+``, ``-``, ``*`` and ``/``.

    A sum or a difference of the values in the running example moves with
    its operands. When the parent asks for both directions, both are passed
    on. Signed multiplication can also invert a direction; that case is
    covered by passing both directions whenever the parent mark is not a
    single ``und``.
    """
    concrete = {mark for mark in received if mark in {PLUS, MINUS}}
    if not concrete:
        return frozenset({UND})
    if concrete == {PLUS}:
        return frozenset({PLUS})
    if concrete == {MINUS}:
        return frozenset({MINUS})
    return frozenset({PLUS, MINUS})


def _operands(node: OperationCallExpression) -> list:
    args = [arg for arg in node.arguments if not isinstance(arg, InfixOperator)]
    if node.source is not None:
        return [node.source, *args]
    return args


def _navigation_events(prop: Property, received: frozenset, ctx: _Walk, node) -> list[EventSpec]:
    association = prop.owner if isinstance(prop.owner, Association) else None
    if association is None and isinstance(prop.type, AssociationClass):
        association = prop.type.association
    if association is None:
        return []
    events = []
    flag = ctx.approximating
    if PLUS in received:
        events.append(EventSpec("InsertAssociation", association, approximated=flag))
        parent = ctx.ancestors[-1] if ctx.ancestors else None
        parent_name = getattr(parent, "name", None) or getattr(parent, "operation", None)
        if ctx.select_decrease and parent_name not in _ABSORBING_PARENTS:
            events.append(EventSpec("DeleteAssociation", association, approximated=flag))
    if MINUS in received:
        events.append(EventSpec("DeleteAssociation", association, approximated=flag))
    return events


def _variable_events(node: VariableExp, received: frozenset, ctx: _Walk) -> list[EventSpec]:
    if node.name != "self" or ctx.in_body or not ctx.ancestors:
        return []
    parent = ctx.ancestors[-1]
    prop = _feature_property(parent)
    if prop is None:
        return []
    flag = ctx.approximating
    events = []
    if _is_navigation(prop) and MINUS in received:
        events.append(EventSpec("InsertClass", ctx.context, approximated=flag))
        events.extend(_specialize_for_new_instance(ctx, flag))
    elif _is_attribute(prop) and _attribute_compared_with_self(parent, ctx):
        events.append(EventSpec("InsertClass", ctx.context, approximated=flag))
        events.extend(_specialize_for_new_instance(ctx, flag))
    return events


def _specialize_for_new_instance(ctx: _Walk, approximated: bool) -> list[EventSpec]:
    """``SpecializeClass`` for a new instance of the context class.

    Emitted only when the class has a superclass and the invariant reads an
    attribute or association defined on that superclass. A fresh subclass
    instance has no values for its own attributes until a later update, so
    specializing it cannot by itself violate a condition on those attributes.
    """
    if not ctx.context.parents():
        return []
    if not _references_supertype_feature(ctx.root, ctx.context):
        return []
    return [EventSpec("SpecializeClass", ctx.context, approximated=approximated)]


def _references_supertype_feature(root, context: Class) -> bool:
    parents = context.all_parents()
    if not parents:
        return False
    for node in walk(root):
        prop = _feature_property(node)
        if prop is None:
            continue
        owner = prop.owner
        if isinstance(owner, Class) and owner in parents:
            return True
        if isinstance(owner, Association):
            for end in owner.ends:
                if end is not prop and isinstance(end.type, Class) and end.type in parents:
                    return True
    return False


def _attribute_compared_with_self(feature_node, ctx: _Walk) -> bool:
    comparison = next((anc for anc in reversed(ctx.ancestors) if _is_comparison(anc)), None)
    if comparison is None:
        return False
    left, right = _comparison_operands(comparison)
    via = _child_towards(comparison, ctx.ancestors, feature_node)
    if via is None:
        return False
    other = right if _same_side(left, via) else left if _same_side(right, via) else None
    if other is None:
        return False
    return _only_self_attributes_or_constants(other, ctx.context)


def _all_instances_events(node: OperationCallExpression, received: frozenset, ctx: _Walk) -> list[EventSpec]:
    type_node = node.source if isinstance(node.source, TypeExp) else None
    if type_node is None:
        for arg in node.arguments:
            if isinstance(arg, TypeExp):
                type_node = arg
                break
    if type_node is None:
        return []
    target = _class_named(ctx, type_node.name)
    if target is None:
        return []
    flag = ctx.approximating
    events = []
    if PLUS in received:
        events.append(EventSpec("InsertClass", target, approximated=flag))
        if target.parents():
            events.append(EventSpec("SpecializeClass", target, approximated=flag))
    if MINUS in received:
        events.append(EventSpec("DeleteClass", target, approximated=flag))
        for parent in target.parents():
            events.append(EventSpec("GeneralizeClass", parent, approximated=flag))
    return events


def _type_test_events(node: OperationCallExpression, ctx: _Walk) -> list[EventSpec]:
    type_name = None
    for arg in node.arguments:
        if isinstance(arg, TypeExp):
            type_name = arg.name
            break
    target = _class_named(ctx, type_name) if type_name else None
    if target is None:
        return []
    flag = ctx.approximating
    events = []
    for parent in target.parents():
        events.append(EventSpec("GeneralizeClass", parent, approximated=flag))
    if node.operation in _TYPE_OF:
        for child in target.specializations():
            events.append(EventSpec("SpecializeClass", child, approximated=flag))
    return events


def _opposite_all(events: list[EventSpec]) -> list[EventSpec]:
    flipped = []
    for event in events:
        flipped.extend(_opposite(event))
    return flipped


def _opposite(event: EventSpec) -> list[EventSpec]:
    swap = {
        "InsertClass": "DeleteClass",
        "DeleteClass": "InsertClass",
        "InsertAssociation": "DeleteAssociation",
        "DeleteAssociation": "InsertAssociation",
        "UpdateAttribute": "UpdateAttribute",
    }
    if event.kind in swap:
        return [replace(event, kind=swap[event.kind])]
    if event.kind == "SpecializeClass" and isinstance(event.target, Class):
        return [
            replace(event, kind="GeneralizeClass", target=parent)
            for parent in event.target.parents()
        ]
    if event.kind == "GeneralizeClass" and isinstance(event.target, Class):
        return [
            replace(event, kind="SpecializeClass", target=child)
            for child in event.target.specializations()
        ]
    return [event]


def _dedup(events: list[EventSpec]) -> list[EventSpec]:
    """Keep one spec per event type. An exact event wins over an approximated one."""
    chosen: dict[tuple, EventSpec] = {}
    for event in events:
        key = event.identity()
        current = chosen.get(key)
        if current is None or (current.approximated and not event.approximated):
            chosen[key] = event
    return sorted(chosen.values(), key=lambda event: (
        event.kind,
        getattr(event.target, "name", ""),
        event.attribute.name if event.attribute is not None else "",
    ))


def _is_navigation(prop: Property) -> bool:
    return isinstance(prop, Property) and isinstance(prop.owner, Association)


def _is_attribute(prop: Property) -> bool:
    return isinstance(prop, Property) and isinstance(prop.owner, Class)


def _feature_property(node):
    if isinstance(node, PropertyCallExpression):
        return node.property
    if isinstance(node, Property):
        return node
    return None


def _is_comparison(node) -> bool:
    return isinstance(node, OperationCallExpression) and node.operation in (_COMP_UP | _COMP_DOWN | _COMP_BOTH)


def _is_undefined_test(node) -> bool:
    return isinstance(node, OperationCallExpression) and node.operation in {
        "oclIsUndefined", "OCLISUNDEFINED",
    }


def _comparison_operands(node: OperationCallExpression):
    operands = _operands(node)
    if len(operands) >= 2:
        return operands[0], operands[1]
    return None, None


def _child_towards(comparison, ancestors: tuple, feature_node):
    if comparison in ancestors:
        index = ancestors.index(comparison)
        if index + 1 < len(ancestors):
            return ancestors[index + 1]
    return feature_node


def _same_side(side, via) -> bool:
    return side is via


def _only_self_attributes_or_constants(node, context: Class) -> bool:
    if node is None or isinstance(node, (LiteralExpression, TypeExp)):
        return True
    if isinstance(node, VariableExp):
        return node.name == "self"
    if isinstance(node, PropertyCallExpression):
        if _is_navigation(node.property):
            return False
        if not isinstance(node.source, VariableExp) or node.source.name != "self":
            return False
        owner = node.property.owner
        return owner is context or (isinstance(owner, Class) and owner in context.all_parents())
    if isinstance(node, Property):
        if _is_navigation(node):
            return False
        owner = node.owner
        return owner is context or (isinstance(owner, Class) and owner in context.all_parents())
    if isinstance(node, OperationCallExpression) and node.operation in _ARITH:
        return all(
            _only_self_attributes_or_constants(child, context)
            for child in _operands(node)
        )
    return False


def _class_named(ctx: _Walk, name: str):
    if name is None:
        return None
    if ctx.context.name == name:
        return ctx.context
    # The context class does not point at its model. Classes referenced by
    # name are the context, one of its parents or children, or the type of a
    # property already present in the expression. Search those before giving up.
    pool = {ctx.context, *ctx.context.all_parents(), *ctx.context.all_specializations()}
    for node in walk(ctx.root):
        prop = _feature_property(node)
        if prop is not None and isinstance(prop.type, Class):
            pool.add(prop.type)
        if isinstance(node, TypeExp):
            pass
        if isinstance(node, OperationCallExpression) and isinstance(node.source, TypeExp):
            pass
    for cls in pool:
        if cls.name == name:
            return cls
    # Last resort: the domain model stashed by the converter, or any class
    # reachable as an association-end type from the pool.
    seen = set()
    frontier = list(pool)
    while frontier:
        current = frontier.pop()
        if current in seen or not isinstance(current, Class):
            continue
        seen.add(current)
        if current.name == name:
            return current
        for end in current.all_association_ends():
            if isinstance(end.type, Class):
                frontier.append(end.type)
    return None
