"""
Translation of OCL invariants into Alloy formulas.

:class:`OclToAlloy` translates the B-OCL AST of each invariant (built by
:func:`~besser.BUML.notations.ocl.api.parse_ocl`) and resolves every name through an
:class:`AlloySpecification`, so the formulas use the same names as the signatures and fields of
the specification.  Enumeration literals ``Enumeration::Literal`` are not in the B-OCL grammar
yet, so invariants using them cannot be translated.

Constructs without a faithful translation raise :class:`UnsupportedOclError` rather than being
approximated: e.g. real division, arithmetic on collections, ``size()`` or ``=`` on Bags
(Alloy relations are sets, so a Bag would lose its duplicates), string operations other
than ``=`` and ``<>`` (strings are uninterpreted), and string literals with
characters that cannot be part of an Alloy identifier (literals name their constants).  Invalid OCL raises
:class:`OclTranslationError`.

``closure`` becomes the reflexive transitive closure ``*`` of Alloy.

A date literal is a string literal compared with a date; it must be an ISO date (``'1950-01-01'``).
Dates support ``=`` and ``<>`` only: ordering comparisons, ``Date::today()``, ``addDays`` and date
arithmetic are unsupported.
The invariants are assumed to be type checked before their translation.
"""

import re
from datetime import date

from besser.BUML.metamodel.ocl.ocl import (
    BooleanLiteralExpression,
    DateLiteralExpression,
    IfExp,
    InfixOperator,
    IntegerLiteralExpression,
    LoopExp,
    OCLConstraint,
    OperationCallExpression,
    PropertyCallExpression,
    RealLiteralExpression,
    StringLiteralExpression,
    TypeExp,
    VariableExp,
)
from besser.BUML.metamodel.structural import Class, Constraint, DomainModel, Property
from besser.BUML.notations.ocl.api import parse_ocl
from besser.BUML.notations.ocl.error_handling import BOCLSyntaxError
from besser.BUML.notations.ocl.pretty_printer import pretty_print
from besser.generators.alloy.alloy_specification import AlloySpecification
from besser.generators.alloy.exceptions import AlloyTranslationError


class OclTranslationError(AlloyTranslationError):
    """Raised for OCL that cannot be translated: syntax errors, unknown names, type errors."""


class UnsupportedOclError(OclTranslationError):
    """Raised for valid OCL constructs that have no Alloy translation yet."""


# Alloy predicates of the date library comparing two dates (as for strings, (in)equality only).
_DATE_COMPARISONS = {"=": "dateEq", "<>": "dateNe"}

# Alloy predicates of the strings library comparing two strings.
_STRING_COMPARISONS = {"=": "strEq", "<>": "strNe"}

# OCL comparison operators.
_COMPARISONS = {"=", "<>", "<", "<=", ">", ">="}

# Alloy functions of the OCL integer operators (OCL ``/`` is real division: Alloy has no translation).
_ARITHMETIC = {"+": "plus", "-": "minus", "*": "mul"}

# OCL ordering operators.
_ORDERINGS = {"<", "<=", ">", ">="}

# Alloy connectives of the OCL boolean operators (``xor`` has no Alloy connective).
_CONNECTIVES = {"and": "and", "or": "or", "implies": "=>"}

# Escape sequence of a string literal: a backslash and the escaped character.
_STRING_ESCAPE = re.compile(r"\\(.)")

# Characters that cannot be part of an Alloy identifier, hence of the name of a string literal's constant.
_INVALID_IDENTIFIER_CHARACTERS = re.compile(r"[^A-Za-z0-9_]")

# Type of the OCL null value.
_NULL = "null"


class OclToAlloy:
    """Adds OCL invariants to *specification* as Alloy facts.

    The specification must declare the signatures and fields of *model*, the domain model the
    invariants constrain.  Translating ``select`` and ``reject`` adds auxiliary functions to it.
    A single translator is meant to translate all the invariants of a specification, collecting
    their literals in :attr:`literals`.
    """

    # Methods translating the operation calls with a source (``source.op(...)``, ``source->op(...)``).
    _SOURCE_OPERATIONS = {
        "Size": "_size",
        "IsEmpty": "_is_empty",
        "NotEmpty": "_not_empty",
        "INCLUDES": "_includes",
        "EXCLUDES": "_excludes",
        "UNION": "_union",
        "INTERSECTION": "_intersection",
        "ASSET": "_as_set",
        "ALLInstances": "_all_instances_of",
        "OCLISKINDOF": "_is_kind_of",
        "OCLISTYPEOF": "_is_type_of",
        "OCLASTYPE": "_as_type",
    }

    def __init__(self, specification: AlloySpecification, model: DomainModel):
        self._spec = specification
        self._model = model
        self.literals = Literals()
        self._variables = Variables()
        self._types: set[str] = set()
        self._function_count = 0
        self._closure_count = 0
        # The variable a bare property navigates from: self, or the implicit iterator of a short-form closure
        self._implicit_source = "self"

    def add_invariant(self, constraint: Constraint) -> None:
        """Adds the fact stating the OCL invariant *constraint*.

        The AST of an :class:`OCLConstraint` is translated as is; the text of any other
        constraint (``context C inv name: expression``) is parsed with :func:`parse_ocl`.
        """
        expression = self._parse(constraint)
        context = constraint.context.name
        self._variables = Variables(self_type=context)
        self._types = set()
        body = self._translate(expression).formula()
        formula = f"all self:this/{self._spec.signature_name(context)}|{body}"
        self._spec.add_fact(formula, sorted(self._types))

    def _parse(self, constraint: Constraint):
        """The B-OCL AST of the boolean expression of *constraint*."""
        if isinstance(constraint, OCLConstraint):
            return constraint.ast
        try:
            return parse_ocl(constraint.expression, self._model, constraint.context).ast
        except (BOCLSyntaxError, ValueError, TypeError) as error:
            raise OclTranslationError(f"Invalid OCL invariant '{constraint.name}': {error}") from None

    def _translate(self, node) -> "Term":
        """Translates the AST *node*."""
        if node is None:  # the parser builds null as None
            return Term("none", _NULL)
        if isinstance(node, VariableExp):
            return self._variable(node.name)
        if isinstance(node, Property):  # a property of implicit self, or of the implicit iterator of a closure
            return self._navigate(self._variable(self._implicit_source), node.name)
        if isinstance(node, PropertyCallExpression):
            return self._navigate(self._translate(node.source), node.property.name)
        if isinstance(node, IntegerLiteralExpression):
            return self._integer(node)
        if isinstance(node, RealLiteralExpression):
            raise UnsupportedOclError(
                f"Reals ('{node.value}') are not supported in the current version of the Alloy translation"
            )
        if isinstance(node, BooleanLiteralExpression):
            return self._boolean(node)
        if isinstance(node, StringLiteralExpression):
            return self._string(node)
        if isinstance(node, DateLiteralExpression):
            raise UnsupportedOclError(
                f"Date expression '{node.value}' is not supported in the current version of the Alloy translation"
            )
        if isinstance(node, IfExp):
            return self._if_then_else(node)
        if isinstance(node, LoopExp):
            return self._iteration(node)
        if isinstance(node, OperationCallExpression):
            return self._operation(node)
        raise UnsupportedOclError(
            f"OCL expression '{pretty_print(node)}' is not supported in the current version of the Alloy translation"
        )

    # ------------------------------------------------------------------
    # Literals
    # ------------------------------------------------------------------

    def _integer(self, node: IntegerLiteralExpression) -> "Term":
        self.literals.add_integer(node.value)
        return Term(str(node.value), "Int")

    def _boolean(self, node: BooleanLiteralExpression) -> "Term":
        self._types.add("Bool")
        return Term("True" if node.value else "False", "Bool")

    def _string(self, node: StringLiteralExpression) -> "Term":
        value = _STRING_ESCAPE.sub(r"\1", node.value)
        self._types.add("Str")
        return Term(self.literals.add_string(value), "Str")

    def _date(self, node: StringLiteralExpression) -> "Term":
        """Translates the string literal *node*, compared with a date, as a date literal."""
        value = _STRING_ESCAPE.sub(r"\1", node.value)
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            raise UnsupportedOclError(
                f"Date '{value}' is not supported: date literals must be ISO dates in the current version of "
                "the Alloy translation"
            ) from None
        self._types.add("Date")
        return Term(self.literals.add_date(parsed), "Date")

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def _operation(self, node: OperationCallExpression) -> "Term":
        operation = node.operation
        operands = [argument for argument in node.arguments if not isinstance(argument, InfixOperator)]
        if node.name.startswith("unary_"):
            return self._unary(node, operation, self._translate(operands[0]))
        if operation in _CONNECTIVES:
            return self._connective(_CONNECTIVES[operation], *operands)
        if operation == "xor":
            return self._xor(*operands)
        if operation == "/":
            raise UnsupportedOclError(
                f"Real division '{pretty_print(node)}' is not supported in the current version of the Alloy translation"
            )
        if operation in _ARITHMETIC:
            return self._arithmetic(node, _ARITHMETIC[operation], *operands)
        if operation in _COMPARISONS:
            return self._comparison(node, operation, *operands)
        if operation in self._SOURCE_OPERATIONS:
            return getattr(self, self._SOURCE_OPERATIONS[operation])(node)
        raise UnsupportedOclError(
            f"OCL operation '{pretty_print(node)}' is not supported in the current version of the Alloy translation"
        )

    def _unary(self, node: OperationCallExpression, operation: str, operand: "Term") -> "Term":
        if operation == "not":
            return Term(f"(not {operand.operand()})")
        if operation == "-":
            self._require_integers(node, operand)
            return Term(f"minus[0, {operand.text}]", "Int")
        raise UnsupportedOclError(
            f"OCL operation '{pretty_print(node)}' is not supported in the current version of the Alloy translation"
        )

    def _connective(self, connective: str, left, right) -> "Term":
        return Term(f"({self._translate(left).operand()} {connective} {self._translate(right).operand()})")

    def _xor(self, left, right) -> "Term":
        return Term(f"(not ({self._translate(left).operand()} <=> {self._translate(right).operand()}))")

    def _arithmetic(self, node: OperationCallExpression, function: str, left, right) -> "Term":
        left, right = self._translate(left), self._translate(right)
        self._require_integers(node, left, right)
        return Term(f"{function}[{left.text}, {right.text}]", "Int")

    def _comparison(self, node: OperationCallExpression, operator: str, left, right) -> "Term":
        left, right = self._comparison_operands(left, right)
        if left.is_null() or right.is_null():
            return self._null_comparison(left, right, operator)
        if left.is_bag or right.is_bag:
            raise UnsupportedOclError(
                f"'{pretty_print(node)}' is not supported: comparing a collection that may contain "
                "duplicates (a Bag) needs asSet() first in the current version of the Alloy translation"
            )
        enumerations = {enumeration.name for enumeration in self._model.get_enumerations()}
        if {left.type, right.type} & enumerations and "Str" in (left.type, right.type):
            # Enumeration literals are not in the B-OCL grammar: a string is not one, and would never be equal
            raise UnsupportedOclError(
                f"'{pretty_print(node)}' is not supported: enumerations cannot be compared with strings in the "
                "current version of the Alloy translation"
            )
        if left.type == right.type == "Date":
            if operator not in _DATE_COMPARISONS:
                raise UnsupportedOclError(
                    f"Date comparison '{operator}' is not supported in the current version of the Alloy translation"
                )
            return Term(f"({_DATE_COMPARISONS[operator]}[{left.text},{right.text}])")
        if left.type == right.type == "Str":
            if operator not in _STRING_COMPARISONS:
                raise UnsupportedOclError(
                    f"String comparison '{operator}' is not supported in the current version of the Alloy translation"
                )
            return Term(f"({_STRING_COMPARISONS[operator]}[{left.text},{right.text}])")
        if operator in _ORDERINGS:
            self._require_integers(node, left, right)
        alloy_operator = "!=" if operator == "<>" else operator
        return Term(f"({left.text} {alloy_operator} {right.text})")

    def _comparison_operands(self, left, right) -> tuple["Term", "Term"]:
        """Translates the operands of a comparison: a string literal compared with a date is a date literal."""
        if isinstance(left, StringLiteralExpression) and not isinstance(right, StringLiteralExpression):
            right = self._translate(right)
            return (self._date(left) if right.type == "Date" else self._translate(left)), right
        if isinstance(right, StringLiteralExpression) and not isinstance(left, StringLiteralExpression):
            left = self._translate(left)
            return left, (self._date(right) if left.type == "Date" else self._translate(right))
        return self._translate(left), self._translate(right)

    def _if_then_else(self, node: IfExp) -> "Term":
        condition = self._translate(node.ifCondition).operand()
        then, otherwise = self._translate(node.thenExpression), self._translate(node.elseCondition)
        if then.is_formula() or otherwise.is_formula():
            return Term(f"({condition} => {then.operand()} else {otherwise.operand()})")
        is_bag = then.is_bag or otherwise.is_bag
        return Term(f"({condition} => {then.text} else {otherwise.text})", then.type, then.is_set,
                    is_bag=is_bag)

    # ------------------------------------------------------------------
    # Operations with a source: collections and types
    # ------------------------------------------------------------------

    def _size(self, node: OperationCallExpression) -> "Term":
        """``size()`` is the cardinality of its source; strings are uninterpreted, so they have no length."""
        source = self._translate(node.source)
        if source.type == "Str" and not source.is_set:
            raise UnsupportedOclError(
                f"'{pretty_print(node)}' is not supported: strings are uninterpreted in the current version of the "
                "Alloy translation"
            )
        if source.is_bag:
            raise UnsupportedOclError(
                f"'{pretty_print(node)}' is not supported: the size of a collection that may contain "
                "duplicates (a Bag) needs asSet() first in the current version of the Alloy translation"
            )
        return Term(f"#({source.text})", "Int")

    def _is_empty(self, node: OperationCallExpression) -> "Term":
        return Term(f"(no {self._translate(node.source).text})")

    def _not_empty(self, node: OperationCallExpression) -> "Term":
        return Term(f"(some {self._translate(node.source).text})")

    def _includes(self, node: OperationCallExpression) -> "Term":
        collection, element = self._translate(node.source), self._translate(node.arguments[0])
        return Term(f"({element.text} in {collection.text})")

    def _excludes(self, node: OperationCallExpression) -> "Term":
        collection, element = self._translate(node.source), self._translate(node.arguments[0])
        return Term(f"({element.text} not in {collection.text})")

    def _union(self, node: OperationCallExpression) -> "Term":
        left, right = self._translate(node.source), self._translate(node.arguments[0])
        return Term(f"({left.text} + {right.text})", left.type, is_set=True, is_bag=left.is_bag or right.is_bag)

    def _intersection(self, node: OperationCallExpression) -> "Term":
        left, right = self._translate(node.source), self._translate(node.arguments[0])
        # An intersection with a Set is a Set
        return Term(f"({left.text} & {right.text})", left.type, is_set=True, is_bag=left.is_bag and right.is_bag)

    def _as_set(self, node: OperationCallExpression) -> "Term":
        source = self._translate(node.source)
        return Term(source.text, source.type, source.is_set)

    def _all_instances_of(self, node: OperationCallExpression) -> "Term":
        if not isinstance(node.source, TypeExp):
            raise UnsupportedOclError(
                f"OCL expression '{pretty_print(node)}' is not supported in the current version of the Alloy "
                "translation"
            )
        return self._all_instances(node.source.name)

    def _is_kind_of(self, node: OperationCallExpression) -> "Term":
        source, kind = self._single_source(node), self._type_argument(node)
        return Term(f"({source.text} in {self._spec.signature_name(kind)})")

    def _is_type_of(self, node: OperationCallExpression) -> "Term":
        source, kind = self._single_source(node), self._type_argument(node)
        subtypes = " + ".join(self._spec.signature_name(sub) for sub in self._spec.subsignatures(kind))
        exact = f"{self._spec.signature_name(kind)} - ({subtypes})" if subtypes else self._spec.signature_name(kind)
        return Term(f"({source.text} in {exact})")

    def _as_type(self, node: OperationCallExpression) -> "Term":
        return Term(self._single_source(node).text, self._type_argument(node))

    def _type_argument(self, node: OperationCallExpression) -> str:
        """The type of the type operation *node*, which must be a class of the model (``Real`` has no signature)."""
        kind = node.arguments[0].name
        if self._model.get_class_by_name(kind) is None:
            raise UnsupportedOclError(
                f"Type operations with '{kind}' ('{pretty_print(node)}') are not supported in the current version of "
                "the Alloy translation: the type must be a class"
            )
        return kind

    def _single_source(self, node: OperationCallExpression) -> "Term":
        """The source of the type operation *node*, which must not be a collection."""
        source = self._translate(node.source)
        if source.is_set:
            raise UnsupportedOclError(
                f"Type operations on collections ('{pretty_print(node)}') are not supported in the current version "
                "of the Alloy translation"
            )
        return source

    # ------------------------------------------------------------------
    # Iterations
    # ------------------------------------------------------------------

    def _iteration(self, node: LoopExp) -> "Term":
        collection = self._translate(node.source)
        operation = node.name
        variables = {
            iterator.name: iterator.type.name if isinstance(iterator.type, Class) else collection.type
            for iterator in node.get_iterator
        }
        (body,) = node.get_body
        if operation == "closure":
            return self._closure(collection, variables, body)
        if not variables:
            raise UnsupportedOclError(
                f"'{operation}' without iterator variables is not supported in the current version of the Alloy "
                "translation"
            )
        if operation in ("forAll", "exists"):
            return self._quantification("all" if operation == "forAll" else "some", collection, variables, body)
        if len(variables) != 1:
            raise UnsupportedOclError(
                f"'{operation}' with several iterator variables is not supported in the current version of the "
                "Alloy translation"
            )
        if operation == "collect":
            return self._collect(collection, variables, body)
        if operation in ("select", "reject"):
            return self._select(collection, variables, body, reject=operation == "reject")
        raise UnsupportedOclError(f"'{operation}' is not supported in the current version of the Alloy translation")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _variable(self, name: str) -> "Term":
        if name not in self._variables:
            raise OclTranslationError(f"Unknown variable '{name}'")
        return Term(name, self._variables.reference(name))

    def _navigate(self, source: "Term", label: str) -> "Term":
        """Navigates from *source* to its property *label*."""
        if source.is_formula() or not self._spec.has_field(source.type, label):
            raise OclTranslationError(f"'{source.type}' has no navigable property '{label}'")
        field = self._spec.field_name(source.type, label)
        is_set = source.is_set or self._spec.field_multiplicity(source.type, label) == "set"
        # Navigating from a collection collects the values of the property: a Bag
        return Term(f"{source.text}.{field}", self._spec.field_type(source.type, label), is_set, is_bag=source.is_set)

    def _enumeration_literal(self, enumeration: str, literal: str) -> "Term":
        # Not called yet: the B-OCL grammar has no enumeration literals.
        try:
            return Term(self._spec.enumeration_value(enumeration, literal), enumeration)
        except KeyError as error:
            raise OclTranslationError(f"Unknown enumeration literal '{enumeration}::{literal}': {error}") from None

    def _all_instances(self, label: str) -> "Term":
        return Term(self._spec.signature_name(label), label, is_set=True)

    def _null_comparison(self, left: "Term", right: "Term", operator: str) -> "Term":
        """``x = null`` holds when *x* is empty, ``x <> null`` when it is not."""
        value = right if left.is_null() else left
        if operator == "=":
            return Term(f"( no({value.text}))")
        if operator == "<>":
            return Term(f"(some {value.text})")
        raise OclTranslationError(f"null cannot be compared with '{operator}'")

    def _quantification(self, quantifier: str, collection: "Term", variables: dict[str, str], body) -> "Term":
        """``forAll``/``exists`` become a quantified formula over the collection."""
        formula = self._in_scope(variables, body).formula()
        return Term(f"{quantifier} {', '.join(variables)} : {collection.text} | {formula}", open_ended=True)

    def _select(self, collection: "Term", variables: dict[str, str], body, reject: bool) -> "Term":
        """``select``/``reject`` become an auxiliary function filtering the collection.

        The function takes the collection and the variables of the enclosing expression that
        the condition uses (e.g. ``self``) as parameters.
        """
        (variable,) = variables
        free = self._variables.start_capture()
        condition = self._in_scope(variables, body).formula()
        self._variables.stop_capture()

        name = f"{'reject' if reject else 'select'}{self._function_count}"
        self._function_count += 1
        element = self._spec.signature_name(collection.type)
        parameters = {"collection": f"set {element}"}
        parameters.update({var: self._spec.signature_name(type_) for var, type_ in free.items()})
        if reject:
            condition = f"(not {condition})"
        self._spec.add_function(name, parameters, f"set {element}", f"{{{variable}: collection | {condition}}}")
        arguments = ", ".join([collection.text, *free])
        return Term(f"{{{name}[{arguments}]}}", collection.type, is_set=True, is_bag=collection.is_bag)

    def _collect(self, collection: "Term", variables: dict[str, str], body) -> "Term":
        """``collect`` becomes the set of the values of the body for the collection elements: a join when
        the body is made of navigations from the iterator, unions and intersections (see
        :meth:`_navigation_relation`), as a join distributes over its left set."""
        (variable,) = variables
        value = self._in_scope(variables, body)
        if value.is_formula():
            raise UnsupportedOclError(
                "'collect' of boolean expressions is not supported in the current version of the Alloy translation"
            )
        relation = self._navigation_relation(variable, value)
        if relation is not None:
            return Term(f"(({collection.text}).{relation})", value.type, is_set=True, is_bag=True)
        result = self._spec.signature_name(value.type)
        return Term(
            f"{{collected: {result} | some {variable} : {collection.text} | collected in {value.text}}}",
            value.type,
            is_set=True,
            is_bag=True,
        )

    def _closure(self, collection: "Term", variables: dict[str, str], body) -> "Term":
        """``closure`` becomes the reflexive transitive closure of the relation from each element to the
        values of the body, applied to the collection: the collection and everything reachable from it
        (OCL 2.4, 11.9.1).

        A body made of navigations from the iterator, unions and intersections closes the relation of its
        fields (``*(A + B)``, see :meth:`_navigation_relation`); any other body closes the relation
        ``{v, reached | reached in body}``. In the short form (no iterator variables), a bare property in
        the body navigates from the implicit iterator.
        """
        number = self._closure_count
        self._closure_count += 1
        if variables:
            (variable,) = variables
            value = self._in_scope(variables, body)
        else:
            variable = f"closure{number}"
            implicit_source, self._implicit_source = self._implicit_source, variable
            try:
                value = self._in_scope({variable: collection.type}, body)
            finally:
                self._implicit_source = implicit_source
        relation = self._navigation_relation(variable, value)
        if relation is None:
            element, reached = self._spec.signature_name(collection.type), f"reached{number}"
            relation = f"{{{variable}: {element}, {reached}: {element} | {reached} in {value.text}}}"
        # *, like the other unary operators, binds tighter than the join: the relation needs parentheses
        return Term(f"(({collection.text}).*({relation}))", collection.type, is_set=True)

    @staticmethod
    def _navigation_relation(variable: str, value: "Term") -> str | None:
        """The relation R such that ``variable.R`` is *value*, when *value* is made of navigations from
        the iterator *variable*, unions and intersections; ``None`` otherwise.

        For the single atom v, v.(A + B) = v.A + v.B and v.(A & B) = v.A & v.B, so R is *value* without
        the leading ``v.``s.
        """
        iterator = rf"(?<![.\w]){re.escape(variable)}\."
        if re.fullmatch(rf"(?:{iterator}\w+(?:\.\w+)*|[\s()+&])+", value.text):
            return re.sub(iterator, "", value.text)
        return None

    @staticmethod
    def _require_integers(node: OperationCallExpression, *operands: "Term") -> None:
        """Raises unless the *operands* of *node* are single integers."""
        if any(operand.type != "Int" or operand.is_set for operand in operands):
            raise UnsupportedOclError(
                f"'{pretty_print(node)}' is not supported: its operands must be single integers in the current "
                "version of the Alloy translation"
            )

    def _in_scope(self, variables: dict[str, str], expression) -> "Term":
        """Translates *expression* with *variables* in scope."""
        self._variables.declare(variables)
        try:
            return self._translate(expression)
        finally:
            self._variables.forget()


class Term:
    """A translated OCL expression: Alloy text with the type of its values.

    A term without type is a formula.  An open-ended term (a quantified formula) extends as far
    right as possible, so it is parenthesized when it is an operand.  A bag term is a collection
    that may contain duplicates in OCL (a Bag), which its Alloy set does not keep.
    """

    def __init__(self, text: str, type: str | None = None, is_set: bool = False, open_ended: bool = False,
                 is_bag: bool = False):
        self.text = text
        self.type = type
        self.is_set = is_set
        self.open_ended = open_ended
        self.is_bag = is_bag

    def is_formula(self) -> bool:
        return self.type is None

    def is_null(self) -> bool:
        return self.type == _NULL

    def formula(self) -> str:
        """The Alloy formula stating this term, which must be a formula or a boolean value."""
        if self.is_formula():
            return self.text
        if self.type == "Bool":
            return f"({self.text} = True)"
        raise OclTranslationError(f"'{self.text}' is not a boolean expression")

    def operand(self) -> str:
        """The formula stating this term, ready to be an operand of a connective."""
        return f"({self.formula()})" if self.open_ended else self.formula()


def date_label(value: date) -> str:
    """The label of the Alloy date of *value*, ``Date_YYYY_MM_DD``."""
    return f"Date_{value:%Y_%m_%d}"


class Literals:
    """The literals of the translated invariants, each once, in the order they were found.

    Each string literal is named by the label of its Alloy constant, ``Str_<value>``, and each
    date literal by the label of its Alloy date, ``Date_YYYY_MM_DD`` (see :func:`date_label`).
    """

    def __init__(self):
        self._strings: dict[str, str] = {}  # value -> label of its constant
        self._integers: list[int] = []
        self._dates: dict[date, str] = {}  # value -> label of its date

    def add_string(self, value: str) -> str:
        """Adds the string literal *value*; returns the label of its constant.

        The label is ``Str_<value>`` (``Str_`` for the empty string): the value has only
        identifier characters, so distinct values have distinct labels.

        Raises:
            UnsupportedOclError: when *value* has characters that cannot be part of an Alloy identifier.
        """
        invalid = dict.fromkeys(_INVALID_IDENTIFIER_CHARACTERS.findall(value))
        if invalid:
            raise UnsupportedOclError(
                f"String '{value}' is not supported: the characters "
                f"{', '.join(repr(character) for character in invalid)} cannot be translated in the current "
                "version of the Alloy translation"
            )
        self._strings.setdefault(value, f"Str_{value}")
        return self._strings[value]

    def add_integer(self, value: int) -> None:
        if value not in self._integers:
            self._integers.append(value)

    def add_date(self, value: date) -> str:
        """Adds the date literal *value*; returns the label of its date."""
        self._dates.setdefault(value, date_label(value))
        return self._dates[value]

    def strings(self) -> list[str]:
        return list(self._strings)

    def string_labels(self) -> dict[str, str]:
        """The labels of the constants of the string literals, by value."""
        return dict(self._strings)

    def integers(self) -> list[int]:
        return list(self._integers)

    def dates(self) -> set[date]:
        return set(self._dates)

    def date_labels(self) -> dict[date, str]:
        """The labels of the dates of the date literals, by value."""
        return dict(self._dates)

    def largest_integer(self) -> int:
        """The largest absolute value of the integer literals, 0 when there is none."""
        return max((abs(value) for value in self._integers), default=0)


class Variables:
    """The variables in scope during a translation (``self`` and iterator variables), with their types.

    A capture records the variables declared outside it that are referenced inside it: the
    free variables of the condition of a ``select``.
    """

    def __init__(self, self_type: str | None = None):
        self._scopes: list[dict[str, str]] = [{"self": self_type}]  # innermost last
        self._captures: list[tuple[int, dict[str, str]]] = []  # (first inner scope, free variables)

    def __contains__(self, name: str) -> bool:
        return any(name in scope for scope in self._scopes)

    def declare(self, variables: dict[str, str]) -> None:
        """Opens a scope declaring *variables*."""
        self._scopes.append(variables)

    def forget(self) -> None:
        """Closes the innermost scope."""
        self._scopes.pop()

    def reference(self, name: str) -> str:
        """The type of the innermost variable *name*, recording it as free in the open captures."""
        depth = max(index for index, scope in enumerate(self._scopes) if name in scope)
        type_ = self._scopes[depth][name]
        for first_inner_scope, free in self._captures:
            if depth < first_inner_scope:
                free[name] = type_
        return type_

    def start_capture(self) -> dict[str, str]:
        """Starts recording free variables; returns the dictionary they will be recorded in."""
        free: dict[str, str] = {}
        self._captures.append((len(self._scopes), free))
        return free

    def stop_capture(self) -> None:
        self._captures.pop()
