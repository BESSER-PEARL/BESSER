"""
Multiplicity parsing utilities for converting JSON to BUML format.
"""

import logging

from besser.BUML.metamodel.structural import Multiplicity, UNLIMITED_MAX_MULTIPLICITY
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError

logger = logging.getLogger(__name__)

_MULTIPLICITY_FORMAT_HINT = "expected 'N', 'N..M', 'N..*', or '*' (e.g. '1', '0..1', '1..*', '*')"


def parse_multiplicity(multiplicity_str):
    """Parse a multiplicity string and return a Multiplicity object with defaults."""
    if not multiplicity_str:
        return Multiplicity(min_multiplicity=1, max_multiplicity=1)

    # Handle single "*" case
    if multiplicity_str == "*":
        return Multiplicity(min_multiplicity=0, max_multiplicity=UNLIMITED_MAX_MULTIPLICITY)

    parts = multiplicity_str.split("..")
    if len(parts) > 2:
        raise ConversionError(
            f"Invalid multiplicity '{multiplicity_str}': {_MULTIPLICITY_FORMAT_HINT}"
        )
    try:
        min_multiplicity = int(parts[0]) if parts[0] and parts[0] != "*" else 0
        if len(parts) > 1:
            # Range notation: "min..max"
            max_multiplicity = (
                UNLIMITED_MAX_MULTIPLICITY if not parts[1] or parts[1] == "*"
                else int(parts[1])
            )
        else:
            # Single value: "N" means N..N
            max_multiplicity = min_multiplicity
    except ValueError as exc:
        raise ConversionError(
            f"Invalid multiplicity '{multiplicity_str}': {_MULTIPLICITY_FORMAT_HINT}"
        ) from exc

    if min_multiplicity > max_multiplicity:
        raise ConversionError(
            f"Invalid multiplicity '{multiplicity_str}': "
            f"min ({min_multiplicity}) cannot be greater than max ({max_multiplicity})"
        )

    return Multiplicity(min_multiplicity=min_multiplicity, max_multiplicity=max_multiplicity)


def format_multiplicity_label(multiplicity: Multiplicity) -> str:
    """Render a ``Multiplicity`` as its UML label.

    Collapses an exact multiplicity (``min == max``) to a single value, matching
    UML convention (``1..1`` -> ``1``, ``5..5`` -> ``5``). An unbounded upper bound
    renders as ``*`` (``0..*``, ``1..*``). The result round-trips through
    ``parse_multiplicity`` (a bare ``N`` is read back as ``N..N``).
    """
    min_val = multiplicity.min
    if multiplicity.max == UNLIMITED_MAX_MULTIPLICITY:
        return f"{min_val}..*"
    if min_val == multiplicity.max:
        return f"{min_val}"
    return f"{min_val}..{multiplicity.max}"
