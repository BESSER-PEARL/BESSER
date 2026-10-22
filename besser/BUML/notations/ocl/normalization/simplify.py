"""Apply chapter-3 simplification to one invariant and keep the original text.

:func:`normalize` rewrites the AST. This module is the step that decides
whether that rewrite changed the expression. When it did, the constraint
displays the normalized text and the previous expression is stored as the
comment linked to the constraint (``metadata.description``, the same place
class-diagram comments are kept). When it did not, nothing is attached.
"""

import re

from besser.BUML.metamodel.ocl.ocl import OCLConstraint
from besser.BUML.metamodel.structural import DomainModel, Metadata
from besser.BUML.notations.ocl.normalization.normalize import normalize
from besser.BUML.notations.ocl.pretty_printer import pretty_print


# The routing prefix of one constraint. The body starts after the colon.
_HEADER_RE = re.compile(
    r"\bcontext\s+\w+"
    r"(?:::\w+\s*\([^)]*\))?"
    r"\s+(?:inv|pre|post)\b"
    r"(?:\s+\w+)?"
    r"\s*:",
    re.IGNORECASE,
)


def simplify_constraint(constraint: OCLConstraint, model: DomainModel,
                        source_text: str | None = None) -> str | None:
    """Normalize ``constraint`` in place.

    Args:
        constraint: A parsed invariant. Not rewritten when its body is
            already in normal form.
        model: The domain model the normalizer consults for types and
            multiplicities.
        source_text: The expression currently shown for this constraint,
            header included. Defaults to ``constraint.expression``. The
            header is kept; only the body is replaced.

    Returns:
        The original expression when the normalized body differs, otherwise
        ``None``. A returned string is also stored as the constraint's
        comment. An unchanged constraint gains no comment.
    """
    if not isinstance(constraint, OCLConstraint) or constraint.ast is None:
        return None
    shown = (source_text if source_text is not None else constraint.expression) or ""
    shown = shown.strip()
    before = pretty_print(constraint.ast)
    normalized = normalize(constraint, model)
    after = pretty_print(normalized.ast)
    if before == after:
        return None
    # ``ast`` refreshes the source text to the body alone; put the header back.
    constraint.ast = normalized.ast
    constraint.expression = _with_body(shown, after)
    _remember_original(constraint, shown)
    return shown


def _with_body(source_text: str, body: str) -> str:
    """Keep the ``context … inv …:`` header and replace the body."""
    match = _HEADER_RE.search(source_text)
    if match is None:
        return body
    return f"{source_text[:match.end()].rstrip()} {body}"


def _remember_original(constraint, original: str) -> None:
    """Store ``original`` as the comment on ``constraint``.

    Comments on a model element live in ``metadata.description``. A second
    simplification that produces the same original does not add it again.
    """
    metadata = constraint.metadata
    if metadata is None:
        constraint.metadata = Metadata(description=original)
        return
    current = (metadata.description or "").strip()
    if not current:
        metadata.description = original
        return
    if original in (part.strip() for part in current.split("\n")):
        return
    metadata.description = f"{current}\n{original}"
