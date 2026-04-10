"""
Shared utilities for json_to_buml diagram processors.
"""


def sanitize_name(name: str) -> str:
    """
    Sanitize a name to be valid for BUML NamedElement.

    Replaces spaces and hyphens with underscores.

    Args:
        name (str): The original name

    Returns:
        str: Sanitized name safe for NamedElement
    """
    if not name:
        return "unnamed"

    sanitized = name.replace(' ', '_').replace('-', '_')

    if not sanitized:
        return "unnamed"

    return sanitized