"""Parse an SDMX-ML 2.1 structure message into plain Python objects.

Only the parts needed to build a B-UML model are read: the dataflow, its data
structure definition (dimensions, time dimension, attributes, primary measure),
concepts (for human-readable names), codelists (codes, names, parents), content
constraints (the codes actually used) and hierarchical codelists.
Tags are matched by local name so the parser does not depend on the namespace
prefixes a given SDMX provider uses.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

# SDMX messages never declare a DOCTYPE; refusing one rules out entity-expansion payloads.
_DOCTYPE = re.compile(rb"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)


@dataclass
class Code:
    """A code of a codelist, with its name and (optional) parent code."""
    id: str
    name: str
    parent: Optional[str] = None


@dataclass
class Codelist:
    """An SDMX codelist."""
    id: str
    agency: str
    version: str
    name: str
    codes: dict[str, Code] = field(default_factory=dict)

    @property
    def has_parents(self) -> bool:
        """True when at least one code references a parent code."""
        return any(c.parent for c in self.codes.values())


@dataclass
class StructureRef:
    """A reference to a maintainable artefact (agency, id, version)."""
    agency: str
    id: str
    version: str = "latest"


@dataclass
class Component:
    """A dimension, the time dimension, an attribute or the primary measure."""
    id: str
    kind: str  # "dimension" | "time" | "attribute" | "measure"
    position: int = 0
    concept: Optional[str] = None
    codelist_ref: Optional[str] = None  # codelist id
    name: Optional[str] = None
    codelist_source: Optional[StructureRef] = None  # the full codelist reference, to fetch it when missing


@dataclass
class DataStructure:
    """A dataflow and its data structure definition (DSD)."""
    id: str
    agency: str
    version: str
    name: str
    dataflow_id: str
    dataflow_name: str
    dimensions: list[Component]
    time_dimension: Optional[Component]
    attributes: list[Component]
    measure: Optional[Component]
    codelists: dict[str, Codelist]
    # Codes actually present in the data, per dimension (SDMX "Actual" content constraint);
    # falls back to "Allowed" constraints. Empty when the provider sends none.
    constraints: dict[str, list[str]] = field(default_factory=dict)
    # Annotations of the dataflow and the DSD: list of (type, title, text).
    annotations: list[tuple[str, str, str]] = field(default_factory=list)
    # Parent of each code per dimension, from a hierarchical codelist (e.g. .Stat Suite HIER_CONTEXT).
    hierarchies: dict[str, dict[str, Optional[str]]] = field(default_factory=dict)

    def codelist_for(self, comp: Component) -> Optional[Codelist]:
        """The codelist of a component, or None when it is not coded (or not in the message)."""
        return self.codelists.get(comp.codelist_ref) if comp.codelist_ref else None

    def used_codes(self, comp: Component) -> list[str]:
        """Codes of a dimension that occur in the data (constraint), else every code of its codelist."""
        if comp.id in self.constraints:
            return list(self.constraints[comp.id])
        codelist = self.codelist_for(comp)
        return list(codelist.codes) if codelist else []

    @property
    def missing_codelists(self) -> list[str]:
        """Codelist ids referenced by a dimension but not defined in the message."""
        return [d.codelist_ref for d in self.dimensions if d.codelist_ref and d.codelist_ref not in self.codelists]

    def hier_context(self) -> dict[str, tuple[str, str, str, str]]:
        """``HIER_CONTEXT`` annotations (.Stat Suite): dim -> (agency, hcl id, version, hierarchy id).

        Format of the annotation title: ``C_PART:LU1:HCL_E4103(1.0).H`` (several separated by ``,``).
        """
        out: dict[str, tuple[str, str, str, str]] = {}
        for typ, title, text in self.annotations:
            if typ != "HIER_CONTEXT":
                continue
            for part in (title or text or "").split(","):
                match = re.match(r"\s*([^:]+):([^:]+):([^(]+)\(([^)]+)\)\.(\w+)", part)
                if match:
                    out[match.group(1)] = (match.group(2), match.group(3), match.group(4), match.group(5))
        return out


def _fromstring(xml: bytes | str) -> ET.Element:
    """Parse XML, refusing a DOCTYPE or ENTITY declaration (never used by SDMX)."""
    head = xml[:65536] if isinstance(xml, bytes) else xml[:65536].encode("utf-8", "ignore")
    if _DOCTYPE.search(head):
        raise ValueError("SDMX structure messages must not declare a DOCTYPE or entities")
    try:
        return ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ValueError(f"Not a valid SDMX-ML message: {exc}") from exc


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _children(el: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in el if _local(c.tag) == name]


def _child(el: Optional[ET.Element], name: str) -> Optional[ET.Element]:
    if el is None:
        return None
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _descendants(el: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in el.iter() if _local(c.tag) == name]


def _name(el: ET.Element, lang: str = "en") -> str:
    """Return the element's Name in ``lang``, else the first Name, else its id."""
    names = _children(el, "Name")
    for n in names:
        if n.get(XML_LANG, "").lower() == lang:
            return (n.text or "").strip()
    if names:
        return (names[0].text or "").strip()
    return el.get("id", "")


def _ref(el: Optional[ET.Element]) -> Optional[StructureRef]:
    """The first <Ref> (or URN) below ``el``, as a reference."""
    if el is None:
        return None
    for r in el.iter():
        if _local(r.tag) == "Ref" and r.get("id"):
            return StructureRef(r.get("agencyID", ""), r.get("id"), r.get("version", "latest") or "latest")
        if _local(r.tag) == "URN" and r.text:
            # urn:sdmx:...Codelist=ESTAT:CL_FREQ(1.0) -> ESTAT, CL_FREQ, 1.0
            match = re.search(r"=([^:]+):([^(]+)\(([^)]*)\)", r.text.strip())
            if match:
                return StructureRef(match.group(1), match.group(2), match.group(3) or "latest")
    return None


def _ref_id(el: Optional[ET.Element]) -> Optional[str]:
    ref = _ref(el)
    return ref.id if ref else None


def parse_codelists(xml: bytes | str, lang: str = "en") -> dict[str, Codelist]:
    """The codelists defined in a structure message, by id."""
    return _parse_codelists(_fromstring(xml), lang)


def _parse_codelists(root: ET.Element, lang: str) -> dict[str, Codelist]:
    out: dict[str, Codelist] = {}
    for cl in _descendants(root, "Codelist"):
        if not _children(cl, "Code"):
            continue  # a stub or a reference, not a definition
        codelist = Codelist(
            id=cl.get("id", ""),
            agency=cl.get("agencyID", ""),
            version=cl.get("version", ""),
            name=_name(cl, lang),
        )
        for c in _children(cl, "Code"):
            code = Code(id=c.get("id", ""), name=_name(c, lang), parent=_ref_id(_child(c, "Parent")))
            codelist.codes[code.id] = code
        out[codelist.id] = codelist
    return out


def _text(el: Optional[ET.Element]) -> str:
    return (el.text or "").strip() if el is not None else ""


def _annotations(el: Optional[ET.Element]) -> list[tuple[str, str, str]]:
    anns = _child(el, "Annotations")
    if anns is None:
        return []
    out = []
    for a in _children(anns, "Annotation"):
        texts = _children(a, "AnnotationText")
        out.append((_text(_child(a, "AnnotationType")), _text(_child(a, "AnnotationTitle")),
                    _text(texts[0]) if texts else ""))
    return out


def _parse_constraints(root: ET.Element, flow_id: Optional[str]) -> dict[str, list[str]]:
    """Codes per dimension from content constraints: "Actual" wins over "Allowed"."""
    by_type: dict[str, dict[str, list[str]]] = {"Actual": {}, "Allowed": {}}
    for cc in _descendants(root, "ContentConstraint"):
        ctype = cc.get("type", "Actual")
        if ctype not in by_type:
            continue
        attach = _child(cc, "ConstraintAttachment")
        if flow_id and attach is not None:
            refs = [r.get("id") for r in attach.iter() if _local(r.tag) == "Ref" and r.get("id")]
            if refs and not any(r.upper() == flow_id.upper() for r in refs) and not any(
                    r.startswith("DSD") for r in refs):
                continue
        for region in _children(cc, "CubeRegion"):
            if region.get("include", "true").lower() != "true":
                continue
            for kv in _children(region, "KeyValue"):
                values = [_text(v) for v in _children(kv, "Value") if _text(v)]
                if values:
                    bucket = by_type[ctype].setdefault(kv.get("id", ""), [])
                    bucket.extend(v for v in values if v not in bucket)
    merged = dict(by_type["Allowed"])
    merged.update(by_type["Actual"])
    return merged


def parse_constraint(xml: bytes | str, dataflow_id: Optional[str] = None) -> dict[str, list[str]]:
    """Codes per dimension from a standalone content-constraint message."""
    return _parse_constraints(_fromstring(xml), dataflow_id)


def parse_hierarchy(xml: bytes | str, hierarchy_id: Optional[str] = None) -> dict[str, Optional[str]]:
    """Parent of each code in an SDMX 2.1 HierarchicalCodelist (one hierarchy)."""
    root = _fromstring(xml)
    hierarchies = _descendants(root, "Hierarchy")
    if hierarchy_id:
        hierarchies = [h for h in hierarchies if h.get("id") == hierarchy_id] or hierarchies
    parents: dict[str, Optional[str]] = {}

    def code_of(hc: ET.Element) -> str:
        cid = _ref_id(_child(hc, "CodeID")) or _ref_id(_child(hc, "Code"))
        return cid or hc.get("id", "")

    def walk(hc: ET.Element, parent: Optional[str]) -> None:
        code = code_of(hc)
        parents.setdefault(code, parent)
        for child in _children(hc, "HierarchicalCode"):
            walk(child, code)

    if hierarchies:
        for top in _children(hierarchies[0], "HierarchicalCode"):
            walk(top, None)
    return parents


@dataclass
class _Concept:
    name: str
    core: Optional[StructureRef]  # the codelist of the concept's core representation


def _parse_concepts(root: ET.Element, lang: str) -> dict[str, _Concept]:
    """Concepts by ``scheme:id`` and by ``id`` (the first scheme wins when ids repeat)."""
    out: dict[str, _Concept] = {}
    for scheme in _descendants(root, "ConceptScheme"):
        for c in _children(scheme, "Concept"):
            concept = _Concept(_name(c, lang), _ref(_child(_child(c, "CoreRepresentation"), "Enumeration")))
            out[f"{scheme.get('id', '')}:{c.get('id', '')}"] = concept
            out.setdefault(c.get("id", ""), concept)
    return out


def _component(el: ET.Element, kind: str, concepts: dict[str, _Concept]) -> Component:
    identity = _child(el, "ConceptIdentity")
    concept_id = _ref_id(identity)
    scheme = next((r.get("maintainableParentID") for r in identity.iter() if r.get("maintainableParentID")),
                  None) if identity is not None else None
    cid = el.get("id") or concept_id or ""
    concept = concepts.get(f"{scheme}:{concept_id}") or concepts.get(concept_id or cid) or concepts.get(cid)
    # The local representation wins; without one, SDMX takes the concept's core representation.
    enum = _ref(_child(_child(el, "LocalRepresentation"), "Enumeration")) or (concept.core if concept else None)
    try:
        position = int(el.get("position", "0") or 0)
    except ValueError:
        position = 0
    return Component(
        id=cid,
        kind=kind,
        position=position,
        concept=concept_id,
        codelist_ref=enum.id if enum else None,
        name=(concept.name if concept else None) or cid,
        codelist_source=enum,
    )


def dataflow_structure_ref(xml: bytes | str, dataflow_id: Optional[str] = None,
                           lang: str = "en") -> Optional[tuple[StructureRef, str, str]]:
    """For a message that holds the dataflow only: (its data structure reference, flow id, flow name)."""
    root = _fromstring(xml)
    flows = [f for f in _descendants(root, "Dataflow") if f.get("id")]
    flow = next((f for f in flows if dataflow_id and f.get("id", "").upper() == dataflow_id.upper()),
                flows[0] if flows else None)
    if flow is None:
        return None
    ref = _ref(_child(flow, "Structure"))
    return (ref, flow.get("id", ""), _name(flow, lang)) if ref else None


def parse_structure(xml: bytes | str, dataflow_id: Optional[str] = None, lang: str = "en") -> DataStructure:
    """Parse an SDMX-ML 2.1 structure message.

    ``xml`` should contain the data structure definition and, ideally, its codelists
    and concepts (request it with ``references=children`` or ``references=descendants``).

    Raises:
        ValueError: when the message is not XML or holds no data structure definition.
    """
    root = _fromstring(xml)
    concepts = _parse_concepts(root, lang)
    codelists = _parse_codelists(root, lang)

    dsds = [d for d in _descendants(root, "DataStructure") if _child(d, "DataStructureComponents") is not None]
    if not dsds:
        raise ValueError("No DataStructure definition found in the structure message")

    flows = [f for f in _descendants(root, "Dataflow") if f.get("id")]
    flow = None
    if dataflow_id:
        flow = next((f for f in flows if f.get("id", "").upper() == dataflow_id.upper()), None)
    flow = flow if flow is not None else (flows[0] if flows else None)

    dsd = dsds[0]
    if flow is not None:
        target = _ref_id(_child(flow, "Structure"))
        dsd = next((d for d in dsds if d.get("id") == target), dsd)

    comps = _child(dsd, "DataStructureComponents")
    dims: list[Component] = []
    time_dim = None
    attributes: list[Component] = []
    measure = None
    # Only direct children of each list: attributes also nest <Dimension> references.
    dim_list = _child(comps, "DimensionList")
    dim_list = comps if dim_list is None else dim_list
    for el in _children(dim_list, "Dimension"):
        dims.append(_component(el, "dimension", concepts))
    for el in _children(dim_list, "TimeDimension"):
        time_dim = _component(el, "time", concepts)
    attr_list = _child(comps, "AttributeList")
    if attr_list is not None:
        for el in _children(attr_list, "Attribute"):
            attributes.append(_component(el, "attribute", concepts))
    measure_list = _child(comps, "MeasureList")
    if measure_list is not None:
        for el in _children(measure_list, "PrimaryMeasure"):
            measure = _component(el, "measure", concepts)
    dims.sort(key=lambda d: d.position)

    return DataStructure(
        id=dsd.get("id", ""),
        agency=dsd.get("agencyID", ""),
        version=dsd.get("version", ""),
        name=_name(dsd, lang),
        dataflow_id=flow.get("id") if flow is not None else (dataflow_id or dsd.get("id", "")),
        dataflow_name=_name(flow, lang) if flow is not None else _name(dsd, lang),
        dimensions=dims,
        time_dimension=time_dim,
        attributes=attributes,
        measure=measure,
        codelists=codelists,
        constraints=_parse_constraints(root, flow.get("id") if flow is not None else dataflow_id),
        annotations=_annotations(flow) + _annotations(dsd),
    )
