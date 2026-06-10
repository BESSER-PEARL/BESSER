"""
Deterministic model-conformance checking for generated workspaces.

The customise loop's prompt tells the LLM "the model is the spec"
(Rule 11), but until now nothing *enforced* it — the only audit was the
LLM promising it checked. This module makes the check mechanical: walk
the workspace, look for every Class / attribute / method / Enumeration
the domain model declares, and report what is missing.

The orchestrator uses the report as a **done gate**: when the LLM
declares completion (``end_turn``) while declared model elements are
absent from the output, the missing list is injected back into the
conversation and the loop continues (bounded — see
``_MAX_CONFORMANCE_NUDGES`` in the orchestrator).

Matching is text-based with name-variant expansion (PascalCase /
camelCase / snake_case) rather than per-language AST parsing. That keeps
it cheap, dependency-free, and language-agnostic — the same checker
works for a Django project and a Rust one. The trade-off is precision:
a name appearing in a comment counts as "found". That bias is
deliberate: a false "found" keeps the gate quiet (no wasted LLM turns),
while a false "missing" would burn turns — so matching is permissive.

An attribute or method is only counted as found in files that also
mention its owning class, so ``name`` on ``Author`` isn't satisfied by
an unrelated ``name`` somewhere else in the project.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Files larger than this are skipped — generated source files are small;
# anything bigger is almost certainly a lockfile / bundle / data blob.
_MAX_FILE_BYTES = 512 * 1024

# Extensions worth scanning. Keep broad: conformance must work for any
# stack the customise loop can produce.
_SOURCE_EXTENSIONS = frozenset({
    ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".java", ".kt", ".kts", ".rs", ".go", ".cs", ".rb", ".php",
    ".sql", ".prisma", ".graphql", ".proto", ".dart", ".swift",
    ".html", ".vue", ".svelte",
})

# Directories that never contain user-facing generated source.
_SKIP_DIRS = frozenset({
    ".git", ".besser_snapshot", "node_modules", "__pycache__",
    ".venv", "venv", "target", "build", "dist", ".next",
})


@dataclass(frozen=True)
class MissingElement:
    """One declared model element absent from the generated output."""

    kind: str           # "class" | "enumeration" | "attribute" | "method" | "literal"
    name: str
    owner: str | None = None  # owning class/enum for members, None for top-level

    def describe(self) -> str:
        if self.owner:
            return f"{self.kind} '{self.owner}.{self.name}'"
        return f"{self.kind} '{self.name}'"


@dataclass
class ConformanceReport:
    """Result of checking a workspace against a domain model."""

    expected: int = 0
    found: int = 0
    missing: list[MissingElement] = field(default_factory=list)
    checked_files: int = 0

    @property
    def ratio(self) -> float:
        if self.expected == 0:
            return 1.0
        return self.found / self.expected

    @property
    def ok(self) -> bool:
        return not self.missing

    def summary(self) -> str:
        return (
            f"model conformance: {self.found}/{self.expected} declared "
            f"elements found across {self.checked_files} files"
        )

    def to_dict(self) -> dict:
        """Recipe / SSE friendly shape."""
        return {
            "expected": self.expected,
            "found": self.found,
            "ratio": round(self.ratio, 3),
            "missing": [m.describe() for m in self.missing],
        }


def _name_variants(name: str) -> list[str]:
    """Expand a model element name into the casings code generators use.

    ``BookLoan`` → BookLoan, bookLoan, book_loan, bookloan;
    ``release_date`` → release_date, releaseDate, ReleaseDate.
    """
    words = _split_words(name)
    if not words:
        return [name]
    lower = [w.lower() for w in words]
    variants = {
        name,
        "".join(lower),                                       # flat
        "_".join(lower),                                      # snake
        lower[0] + "".join(w.title() for w in lower[1:]),     # camel
        "".join(w.title() for w in lower),                    # pascal
    }
    return sorted(variants)


def _split_words(name: str) -> list[str]:
    """Split an identifier into words on case changes and separators."""
    parts = re.split(r"[_\-\s]+", name.strip())
    words: list[str] = []
    for part in parts:
        if not part:
            continue
        words.extend(re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z0-9]+", part))
    return words


def _variant_pattern(name: str) -> re.Pattern:
    alternatives = "|".join(re.escape(v) for v in _name_variants(name))
    return re.compile(r"(?<![A-Za-z0-9_])(?:" + alternatives + r")(?![A-Za-z0-9_])")


def _read_workspace(output_dir: str) -> dict[str, str]:
    """Read every scannable source file into memory, keyed by relpath."""
    contents: dict[str, str] = {}
    for root, dirs, files in os.walk(output_dir):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for file_name in files:
            _, ext = os.path.splitext(file_name)
            if ext.lower() not in _SOURCE_EXTENSIONS:
                continue
            full_path = os.path.join(root, file_name)
            try:
                if os.path.getsize(full_path) > _MAX_FILE_BYTES:
                    continue
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    contents[os.path.relpath(full_path, output_dir)] = f.read()
            except OSError:
                continue
    return contents


def check_conformance(domain_model, output_dir: str) -> ConformanceReport:
    """Check a generated workspace against a BUML ``DomainModel``.

    Every declared Class, Enumeration, attribute, method, and enum
    literal is searched for in the workspace (with casing variants).
    Members are only counted as found in files that also mention their
    owning type.
    """
    report = ConformanceReport()
    if domain_model is None:
        return report

    files = _read_workspace(output_dir)
    report.checked_files = len(files)

    def _check(kind: str, name: str, owner: str | None, haystacks) -> None:
        report.expected += 1
        pattern = _variant_pattern(name)
        for text in haystacks:
            if pattern.search(text):
                report.found += 1
                return
        report.missing.append(MissingElement(kind=kind, name=name, owner=owner))

    all_texts = list(files.values())

    try:
        classes = sorted(domain_model.get_classes(), key=lambda c: c.name)
    except Exception:
        classes = []
    for cls in classes:
        cls_pattern = _variant_pattern(cls.name)
        owning_texts = [t for t in all_texts if cls_pattern.search(t)]
        _check("class", cls.name, None, all_texts)
        for attr in sorted(cls.attributes, key=lambda a: a.name):
            _check("attribute", attr.name, cls.name, owning_texts)
        for method in sorted(getattr(cls, "methods", None) or [], key=lambda m: m.name):
            _check("method", method.name, cls.name, owning_texts)

    try:
        enums = sorted(domain_model.get_enumerations(), key=lambda e: e.name)
    except Exception:
        enums = []
    for enum in enums:
        enum_pattern = _variant_pattern(enum.name)
        owning_texts = [t for t in all_texts if enum_pattern.search(t)]
        _check("enumeration", enum.name, None, all_texts)
        for literal in sorted(enum.literals, key=lambda l: l.name):
            _check("literal", literal.name, enum.name, owning_texts)

    return report


def format_missing_for_llm(report: ConformanceReport, max_items: int = 30) -> str:
    """Render the missing-element list as a corrective message for the LLM."""
    lines = [
        "MODEL CONFORMANCE CHECK FAILED — the domain model declares "
        "elements that are absent from your generated output:",
    ]
    for element in report.missing[:max_items]:
        lines.append(f"  - {element.describe()}")
    if len(report.missing) > max_items:
        lines.append(f"  ... and {len(report.missing) - max_items} more")
    lines.append(
        "The model is the spec: implement every declared class, "
        "attribute, method, and enumeration before finishing. Add the "
        "missing elements now, then finish."
    )
    return "\n".join(lines)
