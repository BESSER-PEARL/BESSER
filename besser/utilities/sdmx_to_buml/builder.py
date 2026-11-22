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
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

from besser.BUML.metamodel.structural import (
    UNLIMITED_MAX_MULTIPLICITY,
    BinaryAssociation,
    Class,
    DomainModel,
    Enumeration,
    EnumerationLiteral,
    FloatType,
    IntegerType,
    Metadata,
    Multiplicity,
    Property,
    StringType,
)
from besser.utilities.sdmx_to_buml.roles import DimensionRole, detect_roles
from besser.utilities.sdmx_to_buml.structure import Codelist, Component, DataStructure

logger = logging.getLogger(__name__)

# Flat codelists with at most this many codes become enumerations; larger ones become classes.
DEFAULT_ENUM_MAX = 50
# Longest enumeration literal built from a code's label (labels can be whole sentences).
_MAX_LITERAL_LENGTH = 60
# Attribute names the generators reserve for themselves (the primary key of every generated table).
_RESERVED_ATTRS = {"id"}
# Letters NFKD does not decompose into an ASCII base letter.
_TRANSLITERATE = str.maketrans({"ß": "ss", "æ": "ae", "Æ": "AE", "ø": "o", "Ø": "O", "œ": "oe", "Œ": "OE",
                                "đ": "d", "Đ": "D", "ł": "l", "Ł": "L", "þ": "th", "Þ": "Th"})


def _ascii(text: str) -> str:
    """``Österreich`` -> ``Osterreich``, ``Größe`` -> ``Grosse``: names stay readable without accents."""
    return unicodedata.normalize("NFKD", text.translate(_TRANSLITERATE)).encode("ascii", "ignore").decode("ascii")


def _words(text: str) -> list[str]:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", _ascii(text))  # split camelCase
    return [w for w in re.split(r"[^A-Za-z0-9]+", text) if w]


def _is_name(text: str) -> bool:
    """A Python identifier that is not a keyword: usable as a B-UML name in every generator."""
    return text.isidentifier() and text.isascii() and not keyword.iskeyword(text)


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
    if _is_name(code) and not code.startswith("_"):
        candidates.append(code)
    if label:
        candidates.append("_".join(w.upper() for w in _words(label))[:_MAX_LITERAL_LENGTH])
    candidates.append("C_" + re.sub(r"[^A-Za-z0-9_]", "_", _ascii(code)).strip("_"))
    for cand in candidates:
        if cand and _is_name(cand) and cand not in taken:
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
        """The mapping of one dimension.

        Raises:
            KeyError: the dataflow has no dimension ``dim_id``.
        """
        found = next((d for d in self.dimensions if d.id == dim_id), None)
        if found is None:
            raise KeyError(dim_id)
        return found


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


class _ModelBuilder:
    """The state of one :func:`structure_to_buml` call: names taken, types, associations, mappings."""

    def __init__(self, dsd: DataStructure, roles: dict[str, DimensionRole], observation_class: str,
                 class_names: dict[str, str]):
        self.dsd = dsd
        self.roles = roles
        self.class_names = class_names
        self.obs = Class(name=observation_class)
        self.used_names: set[str] = {observation_class}
        self.obs_attr_names: set[str] = set()
        self.obs_attrs: set[Property] = set()
        self.types: set = {self.obs}
        self.associations: set[BinaryAssociation] = set()
        self.dim_maps: list[DimensionMapping] = []
        self.shared_classes: dict[str, Class] = {}
        self.codelist_users: dict[str, int] = {}
        for d in dsd.dimensions:
            if d.codelist_ref:
                self.codelist_users[d.codelist_ref] = self.codelist_users.get(d.codelist_ref, 0) + 1

    def unique(self, name: str) -> str:
        """A type name not used yet (``Area``, ``Area2``...)."""
        base, i = name, 2
        while name in self.used_names:
            name, i = f"{base}{i}", i + 1
        self.used_names.add(name)
        return name

    def attr_name(self, text: str) -> str:
        """An Observation attribute name not used yet (``flow``, ``flow_2``...)."""
        base = snake(text)
        name, i = base, 2
        while name in self.obs_attr_names:
            name, i = f"{base}_{i}", i + 1
        self.obs_attr_names.add(name)
        return name

    def add_dimension(self, dim: Component, enum_max: int) -> None:
        """Map one dimension: a string, an enumeration, or a class linked to the observation."""
        codelist = self.dsd.codelist_for(dim)
        attr = self.attr_name(dim.id)
        role = self.roles[dim.id]
        role_fields = {"role": role.role, "label": role.label, "total_code": role.total_code,
                       "default_code": role.default_code}
        if codelist is None or not codelist.codes:
            self.obs_attrs.add(Property(name=attr, type=StringType))
            self.dim_maps.append(DimensionMapping(id=dim.id, kind="string", attr=attr, **role_fields))
        elif role.hierarchy is None and len(codelist.codes) <= enum_max:
            self._add_enumeration(dim, codelist, attr, role_fields)
        else:
            self._add_class(dim, codelist, attr, role, role_fields)

    def _add_enumeration(self, dim: Component, codelist: Codelist, attr: str, role_fields: dict) -> None:
        taken: set[str] = set()
        literals: dict[str, str] = {}
        for code in codelist.codes.values():
            lit = literal_name(code.id, code.name, taken)
            taken.add(lit)
            literals[code.id] = lit
        enum = Enumeration(name=self.unique(pascal(dim.id)),
                           literals={EnumerationLiteral(name=lit) for lit in literals.values()})
        self.types.add(enum)
        self.obs_attrs.add(Property(name=attr, type=enum))
        self.dim_maps.append(DimensionMapping(id=dim.id, kind="enum", attr=attr, type_name=enum.name,
                                              literals=literals, **role_fields))

    def _add_class(self, dim: Component, codelist: Codelist, attr: str, role: DimensionRole,
                   role_fields: dict) -> None:
        shared = self.codelist_users.get(codelist.id, 0) > 1
        dim_cls = self.shared_classes.get(codelist.id)
        new_class = dim_cls is None
        if new_class:
            cls_attrs = {
                Property(name="code", type=StringType, is_id=True),
                Property(name="label", type=StringType),
            }
            if role.hierarchy:
                cls_attrs.add(Property(name="level", type=IntegerType))
            default = re.sub(r"^(CL|CXT)_", "", codelist.id) if shared else dim.id
            name = self.class_names.get(dim.id) or self.class_names.get(codelist.id) or pascal(default)
            dim_cls = Class(name=self.unique(name), attributes=cls_attrs, metadata=Metadata(
                description=_describe(dim, role, codelist.id),
                uri=f"sdmx:codelist:{codelist.agency}:{codelist.id}({codelist.version})"))
            self.shared_classes[codelist.id] = dim_cls
            self.types.add(dim_cls)
        self.associations.add(BinaryAssociation(
            name=f"{snake(self.obs.name)}_{attr}",
            ends={
                Property(name=attr, type=dim_cls, multiplicity=Multiplicity(1, 1)),
                Property(name=f"observations_as_{attr}" if shared else "observations", type=self.obs,
                         multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
            },
        ))
        if role.hierarchy and new_class:
            self.associations.add(BinaryAssociation(
                name=f"{snake(dim_cls.name)}_hierarchy",
                ends={
                    Property(name="parent", type=dim_cls, multiplicity=Multiplicity(0, 1)),
                    Property(name="children", type=dim_cls, multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
                },
            ))
        self.dim_maps.append(DimensionMapping(id=dim.id, kind="class", attr=attr, type_name=dim_cls.name,
                                              hierarchy=role.hierarchy, **role_fields))

    def describe_shared_classes(self) -> None:
        """A class shared by several dimensions (reporter and partner -> one area class) documents all of them."""
        for cl_id, cls in self.shared_classes.items():
            users = [d for d in self.dsd.dimensions if d.codelist_ref == cl_id]
            if len(users) > 1 and cls.metadata is not None:
                cls.metadata.description = ("; ".join(_describe(d, self.roles[d.id]) for d in users)
                                            + f"; codelist={cl_id}")

    def finish_observation(self) -> dict[str, str]:
        """Add the SDMX attributes and the metadata to the observation; returns attribute id -> name."""
        dsd = self.dsd
        attr_map: dict[str, str] = {}
        for a in dsd.attributes:
            name = self.attr_name(a.id)
            attr_map[a.id] = name
            self.obs_attrs.add(Property(name=name, type=StringType, is_optional=True))
        self.obs.attributes = self.obs_attrs
        self.obs.metadata = Metadata(
            description=(f"One observation of SDMX dataflow {dsd.agency}:{dsd.dataflow_id} ({dsd.dataflow_name}). "
                         "Dimension roles: " + "; ".join(_describe(d, self.roles[d.id]) for d in dsd.dimensions)),
            uri=f"sdmx:dataflow:{dsd.agency}:{dsd.dataflow_id}",
        )
        return attr_map


def _validated(model: DomainModel) -> DomainModel:
    """Validate the model: warnings are logged, errors raised."""
    result = model.validate(raise_exception=False)
    for warning in result.get("warnings", []):
        logger.warning("%s: %s", model.name, warning)
    if result.get("errors"):
        raise ValueError(f"The model built for {model.name} is not valid: " + "; ".join(result["errors"]))
    return model


def structure_to_buml(
    dsd: DataStructure,
    model_name: Optional[str] = None,
    enum_max: int = DEFAULT_ENUM_MAX,
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

    Raises:
        ValueError: the model fails validation (its warnings are logged).
    """
    builder = _ModelBuilder(dsd, detect_roles(dsd, prefix_hierarchy), observation_class, class_names or {})
    time_attr = None
    if dsd.time_dimension is not None:
        time_attr = builder.attr_name("time_period")
        builder.obs_attrs.add(Property(name=time_attr, type=StringType))
    value_attr = builder.attr_name("obs_value")
    builder.obs_attrs.add(Property(name=value_attr, type=FloatType, is_optional=True))
    for dim in dsd.dimensions:
        builder.add_dimension(dim, enum_max)
    builder.describe_shared_classes()
    attr_map = builder.finish_observation()

    model = _validated(DomainModel(name=model_name or pascal(dsd.dataflow_id), types=builder.types,
                                   associations=builder.associations))
    mapping = ImportMapping(
        dataflow=dsd.dataflow_id,
        agency=dsd.agency,
        observation_class=observation_class,
        time_attr=time_attr,
        value_attr=value_attr,
        dimensions=builder.dim_maps,
        attributes=attr_map,
        title=dsd.dataflow_name,
    )
    return model, mapping
