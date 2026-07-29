"""Intermediate UML + OCL model.

Pure data classes with no rdflib / PlantUML dependencies: they are the contract
between the *mapping* phase (which reads OWL-2) and the *emission* phase (which
writes PlantUML).  A future XMI emitter could consume the exact same model.

Naming: element ``name`` values are already sanitised to valid UML/PlantUML
identifiers; ``uri`` keeps the original IRI for provenance.  Multiplicity upper
bounds use the string ``"*"`` for "unbounded".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Upper = int | Literal["*"]


@dataclass
class Comment:
    text: str
    target: str | None = None  # element name; None => package-level


@dataclass
class OCLConstraint:
    """An OCL invariant/definition rendered as a PlantUML note."""

    context: str
    body: str
    kind: Literal["inv", "def", "derive"] = "inv"
    name: str | None = None          # e.g. "complete", "disjoint"
    origin_rule: str = ""            # "O07", "O18", ... (provenance / debugging)


@dataclass
class DataType:
    name: str
    uri: str | None = None
    is_primitive: bool = False
    base: str | None = None                       # constrained/derived base type
    facets: list[str] = field(default_factory=list)
    attributes: list[Attribute] = field(default_factory=list)     # O01-O03 composite value slot
    invariants: list[OCLConstraint] = field(default_factory=list)  # O01-O03 membership check
    comment: str | None = None


@dataclass
class Enumeration:
    name: str
    uri: str | None = None
    literals: list[str] = field(default_factory=list)


@dataclass
class Attribute:
    name: str
    type: str = "String"
    lower: int = 0
    upper: Upper = "*"
    is_derived: bool = False        # O19 / O27
    is_id: bool = False             # D38 hasKey participant
    default: str | None = None
    uri: str | None = None

    def multiplicity(self) -> str:
        return _mult(self.lower, self.upper)


@dataclass
class Class:
    name: str
    uri: str | None = None
    is_abstract: bool = False       # union aux (D19), complement aux (O06)
    is_auxiliary: bool = False      # materialised, not present in the source
    is_stub: bool = False           # referenced but never declared owl:Class
    is_foreign: bool = False        # declared, but outside the base namespace
    is_deprecated: bool = False     # owl:deprecated
    attributes: list[Attribute] = field(default_factory=list)
    invariants: list[OCLConstraint] = field(default_factory=list)
    keys: list[list[str]] = field(default_factory=list)   # D38
    comment: str | None = None

    def stereotypes(self) -> list[str]:
        s: list[str] = []
        if self.is_auxiliary:
            s.append("aux")
        if self.is_stub or self.is_foreign:
            s.append("external")
        if self.is_deprecated:
            s.append("deprecated")
        return s


@dataclass
class AssociationEnd:
    type: str                       # target class name
    role: str | None = None
    lower: int = 0
    upper: Upper = "*"
    navigable: bool = True
    uri: str | None = None

    def multiplicity(self) -> str:
        return _mult(self.lower, self.upper)


@dataclass
class Association:
    name: str
    source: AssociationEnd
    target: AssociationEnd
    is_derived: bool = False        # O19
    uri: str | None = None


@dataclass
class Generalization:
    subclass: str
    superclass: str
    set_name: str | None = None
    is_disjoint: bool = False
    is_complete: bool = False


@dataclass
class Slot:
    attribute: str
    values: list[str] = field(default_factory=list)


@dataclass
class Link:
    source: str                     # instance name
    target: str
    role: str


@dataclass
class InstanceSpecification:
    name: str
    classifier: str | None = None
    slots: list[Slot] = field(default_factory=list)
    uri: str | None = None


@dataclass
class UMLModel:
    name: str = "model"
    uri: str | None = None
    version: str | None = None
    imports: list[str] = field(default_factory=list)
    comments: list[Comment] = field(default_factory=list)
    classes: dict[str, Class] = field(default_factory=dict)
    datatypes: dict[str, DataType] = field(default_factory=dict)
    enumerations: dict[str, Enumeration] = field(default_factory=dict)
    associations: list[Association] = field(default_factory=list)
    generalizations: list[Generalization] = field(default_factory=list)
    instances: dict[str, InstanceSpecification] = field(default_factory=dict)
    links: list[Link] = field(default_factory=list)

    # ---- convenience accessors -------------------------------------------
    def all_type_names(self) -> set[str]:
        """Every declared type name a reference may legitimately point at."""
        return (
            set(self.classes)
            | set(self.datatypes)
            | set(self.enumerations)
        )

    def ocl_constraints(self) -> list[OCLConstraint]:
        out: list[OCLConstraint] = []
        for cls in self.classes.values():
            out.extend(cls.invariants)
        for dt in self.datatypes.values():
            out.extend(dt.invariants)
        return out


def _mult(lower: int, upper: Upper) -> str:
    up = "*" if upper == "*" else str(upper)
    if str(lower) == up:
        return up
    return f"{lower}..{up}"
