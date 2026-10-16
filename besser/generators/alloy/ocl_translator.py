"""
Translation of OCL invariants into Alloy formulas.

:class:`OclToAlloy` visits the parse tree of the B-OCL grammar and resolves every name through
an :class:`AlloySpecification`, so the formulas use the same names as the signatures and
fields of the specification.  The grammar is used unchanged; before parsing, enumeration
literals ``Enumeration::Literal`` (not in the grammar) are folded into a single identifier.

Constructs without a translation raise :class:`UnsupportedOclError`; invalid OCL raises
:class:`OclTranslationError`.
"""

from antlr4 import CommonTokenStream, InputStream, Token
from antlr4.error.ErrorListener import ErrorListener
from antlr4.ListTokenSource import ListTokenSource

from besser.BUML.notations.ocl.BOCLLexer import BOCLLexer
from besser.BUML.notations.ocl.BOCLParser import BOCLParser
from besser.BUML.notations.ocl.BOCLVisitor import BOCLVisitor
from besser.generators.alloy.alloy_specification import AlloySpecification


class OclTranslationError(ValueError):
    """Raised for OCL that cannot be translated: syntax errors, unknown names, type errors."""


class UnsupportedOclError(OclTranslationError):
    """Raised for valid OCL constructs that have no Alloy translation yet."""


# Collection operations removed from the grammar, rejected with a clear message.
_REJECTED_OPERATIONS = {"closure", "including", "excluding"}

# Alloy predicates of the date library comparing two dates.
_DATE_COMPARISONS = {"<": "dateLt", "<=": "dateLte", ">": "dateGt", ">=": "dateGte"}

# Type of the OCL null value.
_NULL = "null"


class OclToAlloy(BOCLVisitor):
    """Adds OCL invariants to *specification* as Alloy facts.

    The specification must declare the signatures and fields of the model the invariants
    constrain.  Translating ``select`` and ``reject`` adds auxiliary functions to it.  A single
    translator is meant to translate all the invariants of a specification.
    """

    def __init__(self, specification: AlloySpecification):
        self._spec = specification
        self._parser = OclParser()
        self._variables = Variables()
        self._types: set[str] = set()
        self._function_count = 0

    def add_invariant(self, context: str, ocl: str) -> None:
        """Adds the fact stating the OCL invariant *ocl* of the class *context* (a label).

        *ocl* is either a bare boolean expression or a full ``context C inv name: expression``.
        """
        expression = self._parser.parse_invariant(ocl, context)
        self._variables = Variables(self_type=context)
        self._types = set()
        body = self.visit(expression).formula()
        formula = f"all self:this/{self._spec.signature_name(context)}|{body}"
        self._spec.add_fact(formula, sorted(self._types))

    def visitChildren(self, node):
        """Every grammar alternative without a ``visit`` method below is not supported."""
        raise UnsupportedOclError(f"OCL expression '{node.getText()}' is not supported")

    # ------------------------------------------------------------------
    # Primary expressions
    # ------------------------------------------------------------------

    def visitPrimaryExpr(self, ctx):
        return self.visit(ctx.primaryExpression())

    def visitParenExpr(self, ctx):
        return self.visit(ctx.expression())

    def visitSelfExpr(self, ctx):
        return self._variable("self")

    def visitIdExpr(self, ctx):
        name = ctx.ID().getText()
        if "::" in name:
            return self._enumeration_literal(*name.split("::"))
        if name in self._variables:
            return self._variable(name)
        return self._navigate(self._variable("self"), name)  # implicit self

    def visitNumberLiteral(self, ctx):
        number = ctx.NUMBER().getText()
        if not number.isdigit():
            raise UnsupportedOclError(f"Real number '{number}' is not supported")
        return Term(number, "Int")

    def visitBooleanLiteral(self, ctx):
        self._types.add("Bool")
        return Term("True" if ctx.getText() == "true" else "False", "Bool")

    def visitNullLiteral(self, ctx):
        return Term("none", _NULL)

    def visitAllInstancesExpr(self, ctx):  # Class::allInstances()
        return self._all_instances(ctx.ID().getText())

    def visitAllInstancesExp(self, ctx):  # Class.allInstances()
        return self._all_instances(ctx.expression().getText())

    # ------------------------------------------------------------------
    # Navigation and types
    # ------------------------------------------------------------------

    def visitDotNavigation(self, ctx):
        return self._navigate(self.visit(ctx.expression()), ctx.ID().getText())

    def visitDotOclIsKindOf(self, ctx):
        source, kind = self.visit(ctx.expression()), ctx.typeRef().getText()
        return Term(f"({source.text} in {self._spec.signature_name(kind)})")

    def visitDotOclIsTypeOf(self, ctx):
        source, kind = self.visit(ctx.expression()), ctx.typeRef().getText()
        subtypes = " + ".join(self._spec.signature_name(sub) for sub in self._spec.subsignatures(kind))
        exact = f"{self._spec.signature_name(kind)} - ({subtypes})" if subtypes else self._spec.signature_name(kind)
        return Term(f"({source.text} in {exact})")

    def visitDotOclAsType(self, ctx):
        source = self.visit(ctx.expression())
        return Term(source.text, ctx.typeRef().getText(), source.is_set)

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------

    def visitArrowSize(self, ctx):
        return Term(f"#({self.visit(ctx.expression()).text})", "Int")

    def visitArrowIsEmpty(self, ctx):
        return Term(f"(no {self.visit(ctx.expression()).text})")

    def visitArrowIncludes(self, ctx):
        collection, element = self.visit(ctx.expression(0)), self.visit(ctx.expression(1))
        return Term(f"({element.text} in {collection.text})")

    def visitArrowExcludes(self, ctx):
        collection, element = self.visit(ctx.expression(0)), self.visit(ctx.expression(1))
        return Term(f"({element.text} not in {collection.text})")

    def visitArrowUnion(self, ctx):
        left, right = self.visit(ctx.expression(0)), self.visit(ctx.expression(1))
        return Term(f"({left.text} + {right.text})", left.type, is_set=True)

    def visitIntersection(self, ctx):
        left, right = self.visit(ctx.expression(0)), self.visit(ctx.expression(1))
        return Term(f"({left.text} & {right.text})", left.type, is_set=True)

    def visitArrowasSet(self, ctx):
        return self.visit(ctx.expression())

    def visitArrowIterator(self, ctx):
        collection = self.visit(ctx.expression(0))
        operation = ctx.iteratorOp().getText()
        variables = self._iterator_variables(ctx.iteratorVarDecl(), collection.type)
        if operation in ("forAll", "exists"):
            return self._quantification("all" if operation == "forAll" else "some", collection, variables, ctx)
        if len(variables) != 1:
            raise UnsupportedOclError(f"'{operation}' with several iterator variables is not supported")
        if operation == "collect":
            return self._collect(collection, variables, ctx.expression(1))
        return self._select(collection, variables, ctx.expression(1), reject=operation == "reject")

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------

    def visitUnaryExpr(self, ctx):
        operand = self.visit(ctx.expression())
        if ctx.NOT():
            return Term(f"(not {operand.operand()})")
        return Term(f"minus[0, {operand.text}]", "Int")

    def visitMulDivExpr(self, ctx):
        return self._arithmetic("mul" if ctx.STAR() else "div", ctx)

    def visitAddSubExpr(self, ctx):
        return self._arithmetic("plus" if ctx.PLUS() else "minus", ctx)

    def visitComparisonExpr(self, ctx):
        left, right = self.visit(ctx.expression(0)), self.visit(ctx.expression(1))
        operator = ctx.compOp().getText()
        if left.is_null() or right.is_null():
            return self._null_comparison(left, right, operator)
        if left.type == right.type == "Date" and operator in _DATE_COMPARISONS:
            return Term(f"({_DATE_COMPARISONS[operator]}[{left.text},{right.text}])")
        alloy_operator = "!=" if operator == "<>" else operator
        return Term(f"({left.text} {alloy_operator} {right.text})")

    def visitAndExpr(self, ctx):
        return self._connective("and", ctx)

    def visitOrExpr(self, ctx):
        return self._connective("or", ctx)

    def visitImpliesExpr(self, ctx):
        return self._connective("=>", ctx)

    def visitXorExpr(self, ctx):
        left, right = (self.visit(e).operand() for e in ctx.expression())
        return Term(f"(not ({left} <=> {right}))")

    def visitIfThenElseExpr(self, ctx):
        condition = self.visit(ctx.expression(0)).operand()
        then, otherwise = self.visit(ctx.expression(1)), self.visit(ctx.expression(2))
        if then.is_formula() or otherwise.is_formula():
            return Term(f"({condition} => {then.operand()} else {otherwise.operand()})")
        return Term(f"({condition} => {then.text} else {otherwise.text})", then.type, then.is_set)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _variable(self, name: str) -> "Term":
        return Term(name, self._variables.reference(name))

    def _navigate(self, source: "Term", label: str) -> "Term":
        """Navigates from *source* to its property *label*."""
        if source.is_formula() or not self._spec.has_field(source.type, label):
            raise OclTranslationError(f"'{source.type}' has no navigable property '{label}'")
        field = self._spec.field_name(source.type, label)
        is_set = source.is_set or self._spec.field_multiplicity(source.type, label) == "set"
        return Term(f"{source.text}.{field}", self._spec.field_type(source.type, label), is_set)

    def _enumeration_literal(self, enumeration: str, literal: str) -> "Term":
        try:
            return Term(self._spec.enumeration_value(enumeration, literal), enumeration)
        except KeyError as error:
            raise OclTranslationError(f"Unknown enumeration literal '{enumeration}::{literal}': {error}") from None

    def _all_instances(self, label: str) -> "Term":
        return Term(self._spec.signature_name(label), label, is_set=True)

    def _arithmetic(self, function: str, ctx) -> "Term":
        left, right = self.visit(ctx.expression(0)), self.visit(ctx.expression(1))
        return Term(f"{function}[{left.text}, {right.text}]", "Int")

    def _connective(self, connective: str, ctx) -> "Term":
        left, right = (self.visit(e).operand() for e in ctx.expression())
        return Term(f"({left} {connective} {right})")

    def _null_comparison(self, left: "Term", right: "Term", operator: str) -> "Term":
        """``x = null`` holds when *x* is empty, ``x <> null`` when it is not."""
        value = right if left.is_null() else left
        if operator == "=":
            return Term(f"( no({value.text}))")
        if operator == "<>":
            return Term(f"(some {value.text})")
        raise OclTranslationError(f"null cannot be compared with '{operator}'")

    def _quantification(self, quantifier: str, collection: "Term", variables: dict[str, str], ctx) -> "Term":
        """``forAll``/``exists`` become a quantified formula over the collection."""
        body = self._in_scope(variables, ctx.expression(1)).formula()
        return Term(f"{quantifier} {', '.join(variables)} : {collection.text} | {body}", open_ended=True)

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
        return Term(f"{{{name}[{arguments}]}}", collection.type, is_set=True)

    def _collect(self, collection: "Term", variables: dict[str, str], body) -> "Term":
        """``collect`` becomes the set of the values of the body for the collection elements."""
        (variable,) = variables
        value = self._in_scope(variables, body)
        if value.is_formula():
            raise UnsupportedOclError("'collect' of boolean expressions is not supported")
        result = self._spec.signature_name(value.type)
        return Term(
            f"{{collected: {result} | some {variable} : {collection.text} | collected in {value.text}}}",
            value.type,
            is_set=True,
        )

    def _iterator_variables(self, declaration, element_type: str | None) -> dict[str, str]:
        """The iterator variables declared as ``v1 (: T1)?, v2 (: T2)?`` with their type labels."""
        variables: dict[str, str] = {}
        tokens = [child.getText() for child in declaration.getChildren()]
        index = 0
        while index < len(tokens):
            name, index = tokens[index], index + 1
            type_ = element_type
            if index < len(tokens) and tokens[index] == ":":
                type_, index = tokens[index + 1], index + 2
            variables[name] = type_
            index += 1  # comma
        return variables

    def _in_scope(self, variables: dict[str, str], expression) -> "Term":
        """Translates *expression* with *variables* in scope."""
        self._variables.declare(variables)
        try:
            return self.visit(expression)
        finally:
            self._variables.forget()


class Term:
    """A translated OCL expression: Alloy text with the type of its values.

    A term without type is a formula.  An open-ended term (a quantified formula) extends as far
    right as possible, so it is parenthesized when it is an operand.
    """

    def __init__(self, text: str, type: str | None = None, is_set: bool = False, open_ended: bool = False):
        self.text = text
        self.type = type
        self.is_set = is_set
        self.open_ended = open_ended

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


class OclParser:
    """Parses OCL invariants with the B-OCL grammar, raising :class:`OclTranslationError` on errors.

    Before parsing, the tokens are adapted to the grammar:

    - ``Enumeration::Literal`` (``ID :: ID``) becomes the single identifier ``"Enumeration::Literal"``;
      ``Class::allInstances()`` is not affected, ``allInstances`` being a keyword.
    - ``->closure``, ``->including`` and ``->excluding`` are rejected.
    """

    def parse_invariant(self, ocl: str, context: str):
        """The parse tree of the boolean expression of the OCL invariant *ocl* of *context*."""
        lexer = BOCLLexer(InputStream(ocl))
        self._raise_on_errors(lexer)
        tokens = self._adapt(lexer.getAllTokens())

        parser = BOCLParser(CommonTokenStream(ListTokenSource(tokens)))
        self._raise_on_errors(parser)
        if not tokens or tokens[0].type != BOCLLexer.CONTEXT:
            expression = parser.expression()
            if parser.getCurrentToken().type != Token.EOF:
                raise OclTranslationError(f"Unexpected '{parser.getCurrentToken().text}' in '{ocl}'")
            return expression

        declaration = parser.oclFile().contextDeclaration()
        if declaration is None:
            raise UnsupportedOclError(f"Only invariants are supported: '{ocl}'")
        if declaration.ID().getText() != context:
            raise OclTranslationError(f"Invariant of '{declaration.ID().getText()}' given for '{context}'")
        if len(declaration.constraint()) != 1:
            raise UnsupportedOclError(f"Several invariants in one constraint are not supported: '{ocl}'")
        return declaration.constraint(0).expression()

    def _raise_on_errors(self, recognizer) -> None:
        recognizer.removeErrorListeners()
        recognizer.addErrorListener(RaisingErrorListener())

    def _adapt(self, tokens: list[Token]) -> list[Token]:
        adapted: list[Token] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            following = [t.type for t in tokens[index + 1:index + 3]]
            if token.type == BOCLLexer.ID and following == [BOCLLexer.DOUBLECOLON, BOCLLexer.ID]:
                adapted.append(self._enumeration_literal(token, tokens[index + 2]))
                index += 3
                continue
            if self._is_rejected_operation(tokens, index):
                raise UnsupportedOclError(f"'{tokens[index + 1].text}' is not supported")
            adapted.append(token)
            index += 1
        return adapted

    def _is_rejected_operation(self, tokens: list[Token], index: int) -> bool:
        """Whether the tokens at *index* are ``->`` followed by a rejected operation."""
        return (
            tokens[index].type == BOCLLexer.ARROW
            and index + 1 < len(tokens)
            and tokens[index + 1].text in _REJECTED_OPERATIONS
        )

    def _enumeration_literal(self, enumeration: Token, literal: Token) -> Token:
        """A single identifier token ``"Enumeration::Literal"`` spanning both tokens."""
        token = enumeration.clone()
        token.text = f"{enumeration.text}::{literal.text}"
        token.stop = literal.stop
        return token


class RaisingErrorListener(ErrorListener):
    """Turns ANTLR syntax errors into :class:`OclTranslationError`."""

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        raise OclTranslationError(f"OCL syntax error at {line}:{column}: {msg}")
