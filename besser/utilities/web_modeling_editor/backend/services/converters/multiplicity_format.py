"""Multiplicity formatter / parser for the Deployment converters.

Apollon's ``DeploymentArtifact`` carries no multiplicity field; decision D6
places multiplicity on ``DeploymentRelation`` and the WME wire shape encodes
it as a suffix on the artifact's ``name`` (e.g. ``"Code Tester [3]"``).

This module owns the strict-regex parser (Q4 resolution: regex only, no
fallback heuristics) and the round-trip-stable formatter (Q3 resolution:
first-relation-deterministic when multiple relations diverge). See
02-... §3.6.2.
"""

import re
from typing import Optional, Tuple

from besser.BUML.metamodel.structural import Multiplicity, UNLIMITED_MAX_MULTIPLICITY


# Trailing-bracket pattern: "name [N]" / "name [N..M]" / "name [N..*]".
# Anchored at end-of-string; leading whitespace optional.
_MULTIPLICITY_SUFFIX = re.compile(
    r"^(?P<base>.*?)"
    r"\s*\[\s*(?P<lo>\d+)\s*(?:\.\.\s*(?P<hi>\d+|\*)\s*)?\]\s*$"
)


def parse_from_name(name: Optional[str]) -> Tuple[str, Optional[Multiplicity]]:
    """Split ``name`` into ``(clean_name, multiplicity)`` if it carries a
    trailing ``[N]`` / ``[N..M]`` / ``[N..*]`` suffix.

    Non-matching names round-trip identically — clean_name == name and
    multiplicity is ``None`` (the caller treats this as "no multiplicity
    parsed", leaving the metamodel default of ``(1, 1)`` in place).

    Examples:
        "Code Tester"          -> ("Code Tester",      None)
        "Code Tester [3]"      -> ("Code Tester",      Multiplicity(3, 3))
        "Code Tester [1..*]"   -> ("Code Tester",      Multiplicity(1, *))
        "Code Tester [2..5]"   -> ("Code Tester",      Multiplicity(2, 5))
        "Code Tester [0..1]"   -> ("Code Tester",      Multiplicity(0, 1))
        ""                     -> ("",                  None)
        "Code [Tester"         -> ("Code [Tester",      None)  # non-matching
    """
    if not name:
        return name or "", None
    match = _MULTIPLICITY_SUFFIX.match(name)
    if not match:
        return name, None
    base = match.group("base").strip()
    lo = int(match.group("lo"))
    hi_raw = match.group("hi")
    if hi_raw is None:
        hi = lo
    elif hi_raw == "*":
        hi = UNLIMITED_MAX_MULTIPLICITY
    else:
        hi = int(hi_raw)
    try:
        return base, Multiplicity(lo, hi)
    except (ValueError, TypeError):
        # Defensive — if Multiplicity rejects the bounds, fall through.
        return name, None


def format_to_name(name: str, multiplicity: Optional[Multiplicity]) -> str:
    """Append a multiplicity suffix to ``name`` when the multiplicity differs
    from the default ``(1, 1)``.

    Returns ``name`` unchanged when ``multiplicity`` is ``None`` or has
    default bounds (preserves the compact-encoding round-trip).
    """
    if multiplicity is None:
        return name or ""
    base_name = name or ""
    lo = multiplicity.min
    hi = multiplicity.max
    if lo == 1 and hi == 1:
        return base_name
    if lo == hi:
        suffix = f"[{lo}]"
    elif hi == UNLIMITED_MAX_MULTIPLICITY:
        suffix = f"[{lo}..*]"
    else:
        suffix = f"[{lo}..{hi}]"
    if not base_name:
        return suffix
    return f"{base_name} {suffix}"
