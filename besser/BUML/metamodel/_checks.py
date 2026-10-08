"""Collection type checks shared by the metamodel setters.

Used by the UML Component and UML Deployment metamodels so their setters
reject wrongly-typed members with the same ``TypeError`` wording.
"""

from typing import List


def checked_set(values, expected_type, label: str) -> set:
    """Coerce ``values`` to a set, raising TypeError if any element is not ``expected_type``."""
    result = set(values)
    for value in result:
        if not isinstance(value, expected_type):
            raise TypeError(
                f"{label} must contain {expected_type.__name__} instances, "
                f"got {type(value).__name__}"
            )
    return result


def checked_str_list(values, label: str) -> List[str]:
    """Coerce ``values`` to a list of str, raising TypeError on a non-str entry."""
    result = list(values)
    for value in result:
        if not isinstance(value, str):
            raise TypeError(
                f"{label} must contain str entries, got {type(value).__name__}"
            )
    return result
