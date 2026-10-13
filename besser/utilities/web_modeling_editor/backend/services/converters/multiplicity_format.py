"""Artifact-name multiplicity suffix for the Deployment converters.

The editor's ``DeploymentArtifact`` has no multiplicity field; the metamodel
keeps the instance count on ``DeploymentRelation`` and the editor encodes it as
a suffix on the artifact's ``name`` (e.g. ``"Code Tester [3]"``).

Only a strictly shaped trailing ``[N]`` / ``[N..M]`` / ``[N..*]`` / ``[*]``
counts as a multiplicity; any other bracket text stays part of the name. The
bracket content is parsed and rendered with the same helpers as class-diagram
association ends (``parse_multiplicity`` / ``format_multiplicity_label``).
"""

import re
from typing import Optional, Tuple

from besser.BUML.metamodel.structural import Multiplicity
from besser.utilities.web_modeling_editor.backend.services.converters.parsers.multiplicity_parser import (
    format_multiplicity_label,
    parse_multiplicity,
)


# Trailing-bracket pattern: "name [N]" / "name [N..M]" / "name [N..*]" / "name [*]".
# Anchored at end-of-string; spaces inside the brackets are tolerated.
_MULTIPLICITY_SUFFIX = re.compile(
    r"^(?P<base>.*?)"
    r"\s*\[\s*(?P<bounds>\d+\s*(?:\.\.\s*(?:\d+|\*)\s*)?|\*\s*)\]\s*$"
)


def parse_from_name(name: Optional[str]) -> Tuple[str, Optional[Multiplicity]]:
    """Split ``name`` into ``(clean_name, multiplicity)`` if it carries a
    multiplicity suffix.

    Non-matching names round-trip identically: ``clean_name == name`` and the
    multiplicity is ``None`` (the caller keeps the metamodel default ``1..1``).

    Examples:
        "Code Tester"          -> ("Code Tester",      None)
        "Code Tester [3]"      -> ("Code Tester",      Multiplicity(3, 3))
        "Code Tester [1..*]"   -> ("Code Tester",      Multiplicity(1, *))
        "Code Tester [2..5]"   -> ("Code Tester",      Multiplicity(2, 5))
        ""                     -> ("",                  None)
        "Code [Tester"         -> ("Code [Tester",      None)  # non-matching

    Raises:
        ConversionError: if the suffix has multiplicity shape but invalid
            bounds (e.g. ``[5..2]``).
    """
    if not name:
        return name or "", None
    match = _MULTIPLICITY_SUFFIX.match(name)
    if not match:
        return name, None
    bounds = re.sub(r"\s+", "", match.group("bounds"))
    return match.group("base").strip(), parse_multiplicity(bounds)


def format_to_name(name: str, multiplicity: Optional[Multiplicity]) -> str:
    """Append a multiplicity suffix to ``name`` when the multiplicity differs
    from the default ``1..1``.

    Returns ``name`` unchanged when ``multiplicity`` is ``None`` or has the
    default bounds.
    """
    base_name = name or ""
    if multiplicity is None or (multiplicity.min == 1 and multiplicity.max == 1):
        return base_name
    suffix = f"[{format_multiplicity_label(multiplicity)}]"
    if not base_name:
        return suffix
    return f"{base_name} {suffix}"
