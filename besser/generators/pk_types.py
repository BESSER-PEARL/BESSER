"""Shared primary-key python-type map for code generators.

A ForeignKey field, path parameter, or relationship slot must use the
SAME python type as the primary key it references — a ``guest: int``
Pydantic field pointing at a String PK 422-rejects every real id even
though everything imports. Generators that emit cross-entity references
build this map once and thread it into their templates.
"""

from __future__ import annotations

from typing import Dict

from besser.BUML.metamodel.structural import DomainModel
from besser.generators.structural_utils import get_pk_py_types


def pk_python_types(model: DomainModel) -> Dict[str, str]:
    """Class name -> python type of its primary key (default ``int``).

    Delegates to ``structural_utils.get_pk_py_types`` so there is ONE
    implementation. Errors propagate: an empty map would silently type every
    reference as ``int``.
    """
    return get_pk_py_types(model)
