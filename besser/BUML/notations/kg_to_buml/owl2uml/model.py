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
class OCLConstraint:
    """An OCL invariant/definition rendered as a PlantUML note."""

    context: str
    body: str
    name: str | None = None          # e.g. "complete", "disjoint"
    origin_rule: str = ""            # "O07", "O18", ... (provenance / debugging)


@dataclass
class DataType:
    name: str
    uri: str | None = None
    is_primitive: bool = False
    facets: list[str] = field(default_factory=list)
    attributes: list[Attribute] = field(default_factory=list)     # O01-O03 composite value slot
    invariants: list[OCLConstraint] = field(default_factory=list)  # O01-O03 membership check


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
    uri: str | None = None


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
class Association:
    """A binary association, ``source --role--> target``.

    The same record serves twice: the mapper fills it with the names the rules
    state, and the lowering then resolves them (renames, sanitising) into copies
    it may adjust freely, without disturbing the mapper's own.
    """

    name: str
    source: str                     # source class name
    target: str                     # target class name
    role: str | None = None         # the target end's name; falls back to ``name``
    source_role: str | None = None  # the reverse end's name, if the rules gave one
    src_mult: tuple[int, Upper] = (0, "*")
    tgt_mult: tuple[int, Upper] = (0, "*")
    navigable_source: bool = False  # whether the reverse end is navigable
    uri: str | None = None


@dataclass
class Generalization:
    subclass: str
    superclass: str


@dataclass
class UMLModel:
    classes: dict[str, Class] = field(default_factory=dict)
    datatypes: dict[str, DataType] = field(default_factory=dict)
    enumerations: dict[str, Enumeration] = field(default_factory=dict)
    associations: list[Association] = field(default_factory=list)
    generalizations: list[Generalization] = field(default_factory=list)

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
