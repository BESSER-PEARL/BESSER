"""Known-incompatible dependency pairs, pinned deterministically.

Some packages break at runtime against a newer release of a dependency they do
not pin themselves. pip resolves these without complaint, so neither a dry-run
nor an install catches them; the app only fails when the code path runs.
``KNOWN_INCOMPATIBLE_DEPENDENCIES`` records each pair once, and every stage
that writes or installs a ``requirements.txt`` applies it: scaffold repair,
``install_dependencies`` and the Phase 3 dependency check.
"""

from __future__ import annotations

from dataclasses import dataclass

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from besser.spec_driven_agent.execution.workspace_fs import open_plain_write


@dataclass(frozen=True)
class CompatibilityPin:
    """``package`` must stay within ``compatible`` whenever the trigger is declared."""

    package: str
    compatible: str
    pin: str  # written when the declared range is missing or admits a bad version
    reason: str


# trigger requirement (canonical name) -> dependencies it needs constrained.
KNOWN_INCOMPATIBLE_DEPENDENCIES: dict[str, tuple[CompatibilityPin, ...]] = {
    "passlib": (
        CompatibilityPin(
            package="bcrypt",
            compatible="<4.1",
            pin="bcrypt==4.0.1",
            reason=(
                "passlib 1.7.4 reads bcrypt.__about__ (removed in 4.1) and its "
                "startup self-test hashes a >72-byte secret, which bcrypt 5 "
                "rejects, so every password hash raises ValueError"
            ),
        ),
    ),
}

# Versions probed to decide whether a declared range admits anything outside
# the compatible one; the declared range's own boundary versions are added.
_PROBE_VERSIONS = tuple(
    Version(f"{major}.{minor}.{patch}")
    for major in range(0, 31) for minor in range(0, 16) for patch in (0, 1, 5)
)


def _parse(line: str) -> Requirement | None:
    text = line.split(" #", 1)[0].strip()
    if not text or text.startswith(("#", "-")):
        return None
    try:
        return Requirement(text)
    except InvalidRequirement:
        return None


def _admits_incompatible(declared: SpecifierSet, compatible: SpecifierSet) -> bool:
    if not str(declared):
        return True
    probes = set(_PROBE_VERSIONS)
    for spec in declared:
        try:
            probes.add(Version(spec.version.rstrip(".*")))
        except Exception:
            continue
    return any(v in declared and v not in compatible for v in probes)


def pin_known_incompatible(content: str) -> tuple[str, list[str]]:
    """Return ``content`` with the table's pins applied, and a note per change.

    A declared range that already sits inside the compatible one is kept;
    requirements the table does not mention are left exactly as written.
    """
    lines = content.splitlines()
    parsed = [_parse(line) for line in lines]
    declared = {canonicalize_name(r.name) for r in parsed if r is not None}
    notes: list[str] = []
    for trigger in sorted(declared & KNOWN_INCOMPATIBLE_DEPENDENCIES.keys()):
        for rule in KNOWN_INCOMPATIBLE_DEPENDENCIES[trigger]:
            target = canonicalize_name(rule.package)
            compatible = SpecifierSet(rule.compatible)
            found = False
            for i, req in enumerate(parsed):
                if req is None or canonicalize_name(req.name) != target:
                    continue
                found = True
                if not _admits_incompatible(req.specifier, compatible):
                    continue
                marker = f"; {req.marker}" if req.marker else ""
                notes.append(f"{lines[i].strip()} -> {rule.pin} ({trigger}: {rule.reason})")
                lines[i] = rule.pin + marker
                parsed[i] = _parse(lines[i])
            if not found:
                notes.append(f"added {rule.pin} ({trigger}: {rule.reason})")
                lines.append(rule.pin)
                parsed.append(_parse(rule.pin))
    if not notes:
        return content, []
    return "\n".join(lines) + "\n", notes


def pin_requirements_file(path: str, root: str) -> list[str]:
    """Apply :func:`pin_known_incompatible` to a requirements file in place."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    new_content, notes = pin_known_incompatible(content)
    if notes:
        with open_plain_write(path, "w", root=root, encoding="utf-8") as f:
            f.write(new_content)
    return notes

