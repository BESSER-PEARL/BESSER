"""B-OCL notation: parser front-end and AST printing.

Public surface:

- :func:`parse_ocl` — parse a B-OCL constraint string into an AST.
- :func:`pretty_print` — render an AST back to OCL source text.
- :class:`WrappingVisitor` — bug-fixed visitor (used internally by
  :func:`parse_ocl`); expose for callers that drive the ANTLR pipeline directly.
- :class:`BOCLSyntaxError` — raised on parse errors.

AST helpers (``walk``, ``clone``, ``predicates``, ``chain``) live next to the
metamodel in :mod:`besser.BUML.metamodel.ocl` and are re-exported here for
discoverability.
"""

from besser.BUML.notations.ocl.api import parse_ocl
from besser.BUML.notations.ocl.pretty_printer import pretty_print
from besser.BUML.notations.ocl.wrapping_visitor import WrappingVisitor
from besser.BUML.notations.ocl.error_handling import BOCLSyntaxError

# Re-exports from the metamodel for one-stop import.
from besser.BUML.metamodel.ocl import (
    walk, clone, substitute, ScopeStack,
    is_op, is_and, is_or, is_xor, is_implies, is_not,
    is_size, is_isempty, is_allinstances,
    is_comparison, is_atomic_type_test,
    is_loop, is_loop_with_n_iterators,
    is_self, is_bool_const,
    is_chain_from_self, walk_chain_from_self,
    chain_min_multiplicity, chain_max_multiplicity,
)

__all__ = [
    "parse_ocl", "pretty_print", "WrappingVisitor", "BOCLSyntaxError",
    "walk", "clone", "substitute", "ScopeStack",
    "is_op", "is_and", "is_or", "is_xor", "is_implies", "is_not",
    "is_size", "is_isempty", "is_allinstances",
    "is_comparison", "is_atomic_type_test",
    "is_loop", "is_loop_with_n_iterators",
    "is_self", "is_bool_const",
    "is_chain_from_self", "walk_chain_from_self",
    "chain_min_multiplicity", "chain_max_multiplicity",
]
