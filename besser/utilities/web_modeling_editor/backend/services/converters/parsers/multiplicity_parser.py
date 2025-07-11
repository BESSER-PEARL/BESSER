"""
Multiplicity parsing utilities for converting JSON to BUML format.
"""

from besser.BUML.metamodel.structural import Multiplicity, UNLIMITED_MAX_MULTIPLICITY


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
        max_multiplicity = (
            UNLIMITED_MAX_MULTIPLICITY if len(parts) > 1 and (not parts[1] or parts[1] == "*")
            else int(parts[1]) if len(parts) > 1
            else min_multiplicity
        )
    except ValueError:
        # If parsing fails, return default multiplicity of 1..1
        return Multiplicity(min_multiplicity=1, max_multiplicity=1)

    return Multiplicity(min_multiplicity=min_multiplicity, max_multiplicity=max_multiplicity)
