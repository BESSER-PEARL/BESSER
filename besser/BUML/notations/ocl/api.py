"""Public parsing API for B-OCL.

:func:`parse_ocl` is the canonical entry point for any consumer that needs an
OCL AST. It mirrors the wiring already used by ``OCLWrapper.evaluate`` in the
B-OCL-Interpreter, with three differences:

- It uses :class:`WrappingVisitor` so the returned AST has reconstructed
  property chains and Python ``bool`` values for boolean literals.
- It raises :class:`BOCLSyntaxError` on parse errors instead of silently
  returning a partial AST.
- It returns an :class:`OCLConstraint` so the context class is preserved for
  downstream pretty-printing and rule lookups.
"""

import re

from antlr4 import InputStream, CommonTokenStream

from besser.BUML.metamodel.ocl.ocl import OCLConstraint
from besser.BUML.metamodel.structural import Class, DomainModel
from besser.BUML.notations.ocl.BOCLLexer import BOCLLexer
from besser.BUML.notations.ocl.BOCLParser import BOCLParser
from besser.BUML.notations.ocl.error_handling import (
    BOCLErrorListener, BOCLSyntaxError,
)
from besser.BUML.notations.ocl.wrapping_visitor import WrappingVisitor


_CONTEXT_RE = re.compile(r"\bcontext\s+(\w+)\s+(?:inv|pre|post|init)\b")


def parse_ocl(text: str, model: DomainModel,
              context_class: Class = None) -> OCLConstraint:
    """Parse an OCL constraint string into a fresh AST.

    Args:
        text: The OCL source, e.g. ``"context Employee inv: self.age > 16"``.
        model: The :class:`DomainModel` used to resolve the context class and
            property bindings.
        context_class: Optional explicit context :class:`Class`. If omitted,
            the class is parsed from the ``context X inv|pre|post|init`` header
            and resolved against ``model.types``.

    Returns:
        An :class:`OCLConstraint` whose ``expression`` is the parsed AST and
        whose ``context`` is the resolved context class.

    Raises:
        BOCLSyntaxError: if the text fails to lex or parse.
        ValueError: if the context class cannot be located in the model.
    """
    if context_class is None:
        match = _CONTEXT_RE.search(text)
        if match is None:
            raise ValueError(
                "parse_ocl: input does not contain a "
                "'context <ClassName> inv|pre|post|init' header"
            )
        ctx_name = match.group(1)
        context_class = _resolve_class(model, ctx_name)
        if context_class is None:
            raise ValueError(
                f"parse_ocl: context class {ctx_name!r} not found in domain model"
            )

    input_stream = InputStream(text)
    lexer = BOCLLexer(input_stream)
    lexer.removeErrorListeners()
    error_listener = BOCLErrorListener()
    lexer.addErrorListener(error_listener)
    stream = CommonTokenStream(lexer)
    parser = BOCLParser(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(error_listener)
    tree = parser.oclFile()
    if error_listener.has_errors():
        raise BOCLSyntaxError(error_listener.errors)

    visitor = WrappingVisitor(model, None, context_class)
    expression = visitor.visit(tree)

    return OCLConstraint(
        name="parsed",
        context=context_class,
        expression=expression,
        language="OCL",
    )


def _resolve_class(model: DomainModel, name: str):
    for t in model.types:
        if isinstance(t, Class) and t.name == name:
            return t
    return None
