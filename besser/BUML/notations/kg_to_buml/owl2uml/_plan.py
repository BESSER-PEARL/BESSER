"""The plan the OWL/SHACL rules build before anything is lowered onto BUML.

``Mapper`` (the rule engine) fills a :class:`ModelPlan`; ``to_buml`` then turns
it into a BUML ``DomainModel``. It is a separate structure rather than BUML
objects built as the rules fire because BUML validates eagerly and the rules
cannot satisfy it one step at a time:

* a class's name is only final once *every* type name is known (primitive
  shadowing, a class and a datatype sharing a name, ``_2`` suffixes);
* a ``Generalization`` registers itself on both classes at construction and
  ``Class.all_parents()`` has no visited set, so one cycle hangs the process
  before any model exists to report it;
* association end names must be unique across a class's whole hierarchy, and
  merging associations that would collide needs all of them at once;
* the OCL can only be checked against a finished model.

So this is deliberately only data: no rdflib, no BUML. Every ``name`` is the
canonicaliser's choice; ``uri`` keeps the original IRI for provenance. Upper
bounds use the string ``"*"`` for "unbounded".

Only what the lowering reads lives here. The reference implementation this
package ports also modelled individuals, links, ontology metadata and
``owl:hasKey`` identifiers; nothing consumed them, so they are not carried.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Upper = int | Literal["*"]


@dataclass
class PlannedInvariant:
    """An OCL invariant/definition rendered as a PlantUML note."""

    context: str
    body: str
    name: str | None = None          # e.g. "complete", "disjoint"
    origin_rule: str = ""            # "O07", "O18", ... (provenance / debugging)


@dataclass
class PlannedDataType:
    name: str
    uri: str | None = None
    is_primitive: bool = False
    facets: list[str] = field(default_factory=list)
    attributes: list[PlannedAttribute] = field(default_factory=list)     # O01-O03 composite value slot
    invariants: list[PlannedInvariant] = field(default_factory=list)  # O01-O03 membership check


@dataclass
class PlannedEnumeration:
    name: str
    uri: str | None = None
    literals: list[str] = field(default_factory=list)


@dataclass
class PlannedAttribute:
    name: str
    type: str = "String"
    lower: int = 0
    upper: Upper = "*"
    uri: str | None = None


@dataclass
class PlannedClass:
    name: str
    uri: str | None = None
    is_abstract: bool = False       # union aux (D19), complement aux (O06)
    is_auxiliary: bool = False      # materialised, not present in the source
    is_stub: bool = False           # referenced but never declared owl:Class
    is_foreign: bool = False        # declared, but outside the base namespace
    is_deprecated: bool = False     # owl:deprecated
    attributes: list[PlannedAttribute] = field(default_factory=list)
    invariants: list[PlannedInvariant] = field(default_factory=list)

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
class PlannedAssociation:
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
class ModelPlan:
    classes: dict[str, PlannedClass] = field(default_factory=dict)
    datatypes: dict[str, PlannedDataType] = field(default_factory=dict)
    enumerations: dict[str, PlannedEnumeration] = field(default_factory=dict)
    associations: list[PlannedAssociation] = field(default_factory=list)
    #: ``(subclass, superclass)`` pairs, in the order the rules stated them.
    generalizations: list[tuple[str, str]] = field(default_factory=list)

    # ---- convenience accessors -------------------------------------------
    def all_type_names(self) -> set[str]:
        """Every declared type name a reference may legitimately point at."""
        return (
            set(self.classes)
            | set(self.datatypes)
            | set(self.enumerations)
        )

    def ocl_constraints(self) -> list[PlannedInvariant]:
        out: list[PlannedInvariant] = []
        for cls in self.classes.values():
            out.extend(cls.invariants)
        for dt in self.datatypes.values():
            out.extend(dt.invariants)
        return out
