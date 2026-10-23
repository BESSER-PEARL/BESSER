"""SDMX data structure definition -> B-UML domain model.

Mapping rules:

* The dataflow becomes an ``Observation`` class holding the time period, the observed value and
  the SDMX attributes (e.g. ``OBS_FLAG``), all optional except the time period.
* A coded dimension with a small, flat codelist becomes an ``Enumeration``, typed attribute of
  ``Observation``.
* A coded dimension with a large or hierarchical codelist becomes a class (``code`` as identifier,
  ``label``) linked to ``Observation`` by a 1 to 0..* association. Dimensions that share a codelist
  (reporter and partner -> one country list) share the class.
* A hierarchical codelist (SDMX parent references, a hierarchical codelist, or code prefixes) adds
  a ``parent``/``children`` self-association and a ``level`` attribute.
* A dimension whose codelist is not in the message, or that is not coded, becomes a string.

Each class carries ``Metadata``: the dimension's role, total and default code (see :mod:`.roles`)
and an ``sdmx:`` URI of its codelist or dataflow, so the model documents how to read the data.
"""

import keyword
import re
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, Enumeration, EnumerationLiteral, FloatType,
    IntegerType, Metadata, Multiplicity, Property, StringType,
)
from besser.utilities.sdmx_to_buml.roles import DimensionRole, detect_roles
from besser.utilities.sdmx_to_buml.structure import Component, DataStructure

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Attribute names the generators reserve for themselves (the primary key of every generated table).
_RESERVED_ATTRS = {"id"}


def _words(text: str) -> list[str]:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)  # split camelCase
    return [w for w in re.split(r"[^A-Za-z0-9]+", text) if w]


def pascal(text: str) -> str:
    """``REF_AREA`` -> ``RefArea``: a class or enumeration name."""
    name = "".join(w[:1].upper() + w[1:].lower() if w.isupper() or w.islower() else w[:1].upper() + w[1:]
                   for w in _words(text)) or "X"
    return name if name[0].isalpha() else f"X{name}"


def snake(text: str) -> str:
    """``TIME_PERIOD`` -> ``time_period``: an attribute name (never a Python keyword)."""
    name = "_".join(w.lower() for w in _words(text)) or "x"
    if not name[0].isalpha() or keyword.iskeyword(name) or name in _RESERVED_ATTRS:
        name = f"x_{name}"
    return name


def literal_name(code: str, label: str, taken: set[str]) -> str:
    """A valid, unique enumeration literal for an SDMX code: the code itself, else its label."""
    candidates = []
    if _IDENT.match(code) and not keyword.iskeyword(code) and not code.startswith("_"):
        candidates.append(code)
    if label:
        candidates.append("_".join(w.upper() for w in _words(label))[:60])
    candidates.append("C_" + re.sub(r"[^A-Za-z0-9_]", "_", code).strip("_"))
    for cand in candidates:
        if cand and _IDENT.match(cand) and not keyword.iskeyword(cand) and cand not in taken:
            return cand
    base, i = candidates[-1], 2
    while f"{base}_{i}" in taken:
        i += 1
    return f"{base}_{i}"


@dataclass
class DimensionMapping:
    """How one SDMX dimension maps onto the model."""
    id: str                          # SDMX dimension id, e.g. "REF_AREA"
    kind: str                        # "enum" | "class" | "string"
    attr: str                        # Observation attribute or association end name
    type_name: Optional[str] = None  # Enumeration or Class name
    literals: dict[str, str] = field(default_factory=dict)  # code -> enum literal (enum only)
    hierarchy: Optional[str] = None  # "sdmx" | "prefix" | "hcl" (class only)
    role: str = "breakdown"          # see roles.py
    label: str = ""                  # human-readable name of the dimension (SDMX concept)
    total_code: Optional[str] = None
    default_code: Optional[str] = None


@dataclass
class ImportMapping:
    """How the dataflow maps onto the model: what a data loader needs to fill the generated database."""
    dataflow: str
    agency: str
    observation_class: str
    time_attr: Optional[str]
    value_attr: str
    dimensions: list[DimensionMapping]
    attributes: dict[str, str]       # SDMX attribute id -> Observation attribute
    title: str = ""                  # dataflow name

    def to_dict(self) -> dict:
        """The mapping as plain JSON-serialisable data."""
        return asdict(self)

    def dimension(self, dim_id: str) -> DimensionMapping:
        """The mapping of one dimension."""
        return next(d for d in self.dimensions if d.id == dim_id)


def _describe(dim: Component, role: DimensionRole, codelist: Optional[str] = None) -> str:
    parts = [f"{dim.id} ({role.label}): role={role.role}"]
    if role.hierarchy:
        parts.append(f"hierarchy={role.hierarchy}")
    if role.total_code:
        parts.append(f"total={role.total_code}")
    if role.default_code:
        parts.append(f"default={role.default_code}")
    if codelist:
        parts.append(f"codelist={codelist}")
    return ", ".join(parts)


def structure_to_buml(
    dsd: DataStructure,
    model_name: Optional[str] = None,
    enum_max: int = 50,
    prefix_hierarchy: Optional[Iterable[str]] = None,
    observation_class: str = "Observation",
    class_names: Optional[dict[str, str]] = None,
) -> tuple[DomainModel, ImportMapping]:
    """Build a validated B-UML domain model from a parsed SDMX data structure.

    Args:
        dsd: parsed structure (see :func:`besser.utilities.sdmx_to_buml.parse_structure`).
        model_name: name of the domain model; defaults to the dataflow id.
        enum_max: flat codelists with at most this many codes become enumerations.
        prefix_hierarchy: dimension ids whose codes nest by prefix; ``None`` auto-detects
            numeric nomenclatures.
        observation_class: name of the fact class.
        class_names: optional names for the generated classes, keyed by dimension id or
            codelist id (e.g. ``{"CL_AREA": "Country"}``).

    Returns:
        The domain model and the import mapping (how codes and columns map onto it).
    """
    class_names = class_names or {}
    roles = detect_roles(dsd, prefix_hierarchy)
    codelist_users: dict[str, int] = {}
    for d in dsd.dimensions:
        if d.codelist_ref:
            codelist_users[d.codelist_ref] = codelist_users.get(d.codelist_ref, 0) + 1
    shared_classes: dict[str, Class] = {}
    used_names: set[str] = {observation_class}
    obs_attrs: set[Property] = set()
    types: set = set()
    associations: set[BinaryAssociation] = set()
    dim_maps: list[DimensionMapping] = []
    obs_attr_names: set[str] = set()

    def unique(name: str) -> str:
        base, i = name, 2
        while name in used_names:
            name, i = f"{base}{i}", i + 1
        used_names.add(name)
        return name

    def attr_name(text: str) -> str:
        base = snake(text)
        name, i = base, 2
        while name in obs_attr_names:
            name, i = f"{base}_{i}", i + 1
        obs_attr_names.add(name)
        return name

    time_attr = None
    if dsd.time_dimension is not None:
        time_attr = attr_name("time_period")
        obs_attrs.add(Property(name=time_attr, type=StringType))
    value_attr = attr_name("obs_value")
    obs_attrs.add(Property(name=value_attr, type=FloatType, is_optional=True))

    obs = Class(name=observation_class)
    for dim in dsd.dimensions:
        codelist = dsd.codelist_for(dim)
        attr = attr_name(dim.id)
        role = roles[dim.id]
        role_fields = dict(role=role.role, label=role.label, total_code=role.total_code,
                           default_code=role.default_code)
        if codelist is None or not codelist.codes:
            obs_attrs.add(Property(name=attr, type=StringType))
            dim_maps.append(DimensionMapping(id=dim.id, kind="string", attr=attr, **role_fields))
            continue

        if role.hierarchy is None and len(codelist.codes) <= enum_max:
            taken: set[str] = set()
            literals: dict[str, str] = {}
            for code in codelist.codes.values():
                lit = literal_name(code.id, code.name, taken)
                taken.add(lit)
                literals[code.id] = lit
            enum = Enumeration(name=unique(pascal(dim.id)),
                               literals={EnumerationLiteral(name=lit) for lit in literals.values()})
            types.add(enum)
            obs_attrs.add(Property(name=attr, type=enum))
            dim_maps.append(DimensionMapping(id=dim.id, kind="enum", attr=attr, type_name=enum.name,
                                             literals=literals, **role_fields))
            continue

        shared = codelist_users.get(codelist.id, 0) > 1
        dim_cls = shared_classes.get(codelist.id)
        new_class = dim_cls is None
        if new_class:
            cls_attrs = {
                Property(name="code", type=StringType, is_id=True),
                Property(name="label", type=StringType),
            }
            if role.hierarchy:
                cls_attrs.add(Property(name="level", type=IntegerType))
            default = re.sub(r"^(CL|CXT)_", "", codelist.id) if shared else dim.id
            name = class_names.get(dim.id) or class_names.get(codelist.id) or pascal(default)
            dim_cls = Class(name=unique(name), attributes=cls_attrs, metadata=Metadata(
                description=_describe(dim, role, codelist.id),
                uri=f"sdmx:codelist:{codelist.agency}:{codelist.id}({codelist.version})"))
            shared_classes[codelist.id] = dim_cls
            types.add(dim_cls)
        associations.add(BinaryAssociation(
            name=f"{snake(observation_class)}_{attr}",
            ends={
                Property(name=attr, type=dim_cls, multiplicity=Multiplicity(1, 1)),
                Property(name=f"observations_as_{attr}" if shared else "observations", type=obs,
                         multiplicity=Multiplicity(0, "*")),
            },
        ))
        if role.hierarchy and new_class:
            associations.add(BinaryAssociation(
                name=f"{snake(dim_cls.name)}_hierarchy",
                ends={
                    Property(name="parent", type=dim_cls, multiplicity=Multiplicity(0, 1)),
                    Property(name="children", type=dim_cls, multiplicity=Multiplicity(0, "*")),
                },
            ))
        dim_maps.append(DimensionMapping(id=dim.id, kind="class", attr=attr, type_name=dim_cls.name,
                                         hierarchy=role.hierarchy, **role_fields))

    # A class shared by several dimensions (reporter and partner -> one area class) documents all of them.
    for cl_id, cls in shared_classes.items():
        users = [d for d in dsd.dimensions if d.codelist_ref == cl_id]
        if len(users) > 1 and cls.metadata is not None:
            cls.metadata.description = "; ".join(_describe(d, roles[d.id]) for d in users) + f"; codelist={cl_id}"

    attr_map: dict[str, str] = {}
    for a in dsd.attributes:
        name = attr_name(a.id)
        attr_map[a.id] = name
        obs_attrs.add(Property(name=name, type=StringType, is_optional=True))

    obs.attributes = obs_attrs
    obs.metadata = Metadata(
        description=(f"One observation of SDMX dataflow {dsd.agency}:{dsd.dataflow_id} ({dsd.dataflow_name}). "
                     "Dimension roles: " + "; ".join(_describe(d, roles[d.id]) for d in dsd.dimensions)),
        uri=f"sdmx:dataflow:{dsd.agency}:{dsd.dataflow_id}",
    )
    types.add(obs)

    model = DomainModel(name=model_name or pascal(dsd.dataflow_id), types=types, associations=associations)
    model.validate(raise_exception=True)

    mapping = ImportMapping(
        dataflow=dsd.dataflow_id,
        agency=dsd.agency,
        observation_class=observation_class,
        time_attr=time_attr,
        value_attr=value_attr,
        dimensions=dim_maps,
        attributes=attr_map,
        title=dsd.dataflow_name,
    )
    return model, mapping
