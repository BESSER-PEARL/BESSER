"""
OCL constraint parsing utilities for converting JSON to BUML format.

Two ingest paths coexist here:

* ``process_ocl_constraints`` — legacy: a single textarea blob containing
  one or more full ``context X inv name: expr`` blocks. Used when a JSON
  ``ClassOCLConstraint`` element has no ``kind`` field.
* ``parse_ocl_body`` — new shape: the JSON element carries ``kind`` (one
  of ``invariant`` / ``precondition`` / ``postcondition``) and a body-only
  expression, plus a ``targetMethodId`` for pre/post. The header is
  synthesized internally to match the BOCL grammar. Pre/post require the
  full method-signature header (``context C::m(p: T) pre|post: expr``);
  invariants accept ``context C inv name: expr``.

Both paths return parsed :class:`OCLConstraint` objects with ``.ast``
populated, so downstream consumers (validator, generators, B-OCL
Interpreter) can walk the AST directly. Malformed OCL is skipped with a
warning rather than aborting the whole conversion.
"""

import re
from typing import Optional

from besser.BUML.metamodel.structural import Class, DomainModel, Method
from besser.BUML.metamodel.ocl.ocl import OCLConstraint
from besser.BUML.notations.ocl.api import parse_ocl
from besser.BUML.notations.ocl.error_handling import BOCLSyntaxError


_VALID_KINDS = ("invariant", "precondition", "postcondition")
_KIND_KEYWORD = {"invariant": "inv", "precondition": "pre", "postcondition": "post"}


def _typeref_name(t) -> str:
    """Best-effort BOCL type name for a Property/Parameter type."""
    if t is None:
        return "any"
    if hasattr(t, "name") and t.name:
        return t.name
    return str(t)


def parse_ocl_body(
    body: str,
    kind: str,
    name: str,
    context_class: Class,
    domain_model: DomainModel,
    method: Optional[Method] = None,
) -> OCLConstraint:
    """Synthesize a BOCL header for ``body`` and parse it into an :class:`OCLConstraint`.

    The OCL grammar requires every constraint to start with a ``context``
    header. The WME stores body-only OCL on each constraint element along
    with its kind and (for pre/post) the target method, so this helper
    reconstructs the header from the surrounding metadata before handing
    it to :func:`parse_ocl`.

    Args:
        body: The OCL body, e.g. ``"self.balance >= 0"``.
        kind: One of ``"invariant"``, ``"precondition"``, ``"postcondition"``.
        name: Name to assign to the resulting constraint.
        context_class: Class providing the OCL ``self`` context.
        domain_model: Owning domain model (used by :func:`parse_ocl` for
            property/method resolution).
        method: Required iff ``kind`` is pre/post. Provides the method
            signature segment of the synthesized header.

    Returns:
        An :class:`OCLConstraint` whose ``.ast`` is the parsed expression
        and whose ``.name`` is ``name``.

    Raises:
        ValueError: invalid ``kind``, or pre/post without ``method``.
        BOCLSyntaxError: if ``body`` fails to parse.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(f"parse_ocl_body: unknown kind {kind!r}; expected one of {_VALID_KINDS}")
    if kind == "invariant":
        header = f"context {context_class.name} inv {name}: {body}"
    else:
        if method is None:
            raise ValueError(
                f"parse_ocl_body: method is required when kind={kind!r}"
            )
        params = ", ".join(
            f"{p.name}: {_typeref_name(p.type)}" for p in method.parameters
        )
        header = f"context {context_class.name}::{method.name}({params}) {_KIND_KEYWORD[kind]}: {body}"

    constraint = parse_ocl(header, domain_model, context_class=context_class)
    constraint.name = name
    return constraint


def process_ocl_constraints(ocl_text: str, domain_model: DomainModel, counter: int) -> tuple[list, list]:
    """Parse a legacy free-text OCL blob into a list of :class:`OCLConstraint` objects.

    Used for backward compatibility with WME projects whose
    ``ClassOCLConstraint`` element predates the ``kind`` field — the
    textarea then holds one or more complete ``context X inv name: expr``
    blocks. Each block is parsed independently; malformed blocks are
    skipped with a warning so a single bad constraint does not lose the
    whole document.
    """
    if not ocl_text:
        return [], []

    constraints: list[OCLConstraint] = []
    warnings: list[str] = []
    constraint_count = 0

    domain_classes = {cls.name.lower(): cls for cls in domain_model.types}

    # Split on word-boundary "context" while preserving the keyword in each chunk.
    blocks = re.split(r'(?i)(?=\bcontext\b)', ocl_text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        line = block.replace('\n', ' ').strip()
        if not line.lower().startswith('context'):
            continue

        parts = line.split()
        if len(parts) < 2:
            warnings.append(f"Warning: Malformed OCL constraint (too few tokens): {line}")
            continue

        context_class_name = parts[1].split('::')[0]  # 'Class' or 'Class::method'
        context_class = domain_classes.get(context_class_name.lower())
        if not context_class:
            warnings.append(f"Warning: Context class {context_class_name} not found")
            continue

        constraint_count += 1
        constraint_name = f"constraint_{context_class_name}_{counter}_{constraint_count}"

        try:
            constraint = parse_ocl(line, domain_model, context_class=context_class)
            constraint.name = constraint_name
            constraints.append(constraint)
        except BOCLSyntaxError as e:
            warnings.append(f"Warning: Invalid OCL syntax in constraint '{constraint_name}': {e}")
        except ValueError as e:
            warnings.append(f"Warning: Could not parse OCL constraint '{constraint_name}': {e}")

    return constraints, warnings
