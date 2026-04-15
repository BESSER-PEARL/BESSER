"""
Multiplicity parsing utilities for converting JSON to BUML format.
"""

import logging

from besser.BUML.metamodel.structural import Multiplicity, UNLIMITED_MAX_MULTIPLICITY

logger = logging.getLogger(__name__)


def parse_multiplicity(multiplicity_str):
    """Parse a multiplicity string and return a Multiplicity object with defaults."""
    if not multiplicity_str:
        return Multiplicity(min_multiplicity=1, max_multiplicity=1)

    # Handle single "*" case
    if multiplicity_str == "*":
        return Multiplicity(min_multiplicity=0, max_multiplicity=UNLIMITED_MAX_MULTIPLICITY)

    parts = multiplicity_str.split("..")
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
    except ValueError:
        # If parsing fails, return default multiplicity of 1..1
        logger.warning("Could not parse multiplicity '%s', defaulting to 1..1.", multiplicity_str)
        return Multiplicity(min_multiplicity=1, max_multiplicity=1)

    if min_multiplicity > max_multiplicity:
        raise ValueError(
            f"Invalid multiplicity: min ({min_multiplicity}) cannot be greater than max ({max_multiplicity})"
        )

    return Multiplicity(min_multiplicity=min_multiplicity, max_multiplicity=max_multiplicity)
