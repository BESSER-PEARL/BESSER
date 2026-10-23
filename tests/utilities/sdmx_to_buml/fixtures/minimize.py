"""Shrink an SDMX-ML structure message to what one dataflow needs, to record it as a test fixture.

    python tests/utilities/sdmx_to_buml/fixtures/minimize.py answer.xml EXR fixtures/ecb_EXR.xml.gz [60]

Keeps the dataflow, its data structure definition, the concepts its dimensions use, the codelists
they reference (cut to the first ``max_codes`` codes, plus the parents those codes need), the
``HIER_CONTEXT`` annotations and the dataflow's content constraints. Drops non-English names, all
descriptions, other annotations, categorisations, agency schemes and hierarchical codelists.
"""

import gzip
import re
import sys
import xml.etree.ElementTree as ET

XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
DEFAULT_MAX_CODES = 60  # more than the 50-code enumeration limit, so a large codelist stays a class
_DROPPED_SECTIONS = {"Categorisations", "CategorySchemes", "AgencySchemes", "HierarchicalCodelists"}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _ref_ids(el: ET.Element) -> set[str]:
    return {r.get("id") for r in el.iter() if _local(r.tag) == "Ref" and r.get("id")}


def _children(el: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in el if _local(c.tag) == name]


def _used_codelists_and_concepts(structures: ET.Element, dimensions: ET.Element) -> tuple[set[str], set[str]]:
    codelists, concepts = set(), set()
    for dim in dimensions:
        for part in dim:
            if _local(part.tag) == "LocalRepresentation":
                codelists |= _ref_ids(part)
            if _local(part.tag) == "ConceptIdentity":
                concepts |= _ref_ids(part)
    for concept in structures.iter():
        if _local(concept.tag) == "Concept" and concept.get("id") in concepts:
            for rep in _children(concept, "CoreRepresentation"):
                codelists |= _ref_ids(rep)
    return codelists, concepts


def _trim_codelist(codelist: ET.Element, max_codes: int) -> None:
    """Keep the first ``max_codes`` codes and every parent they need."""
    codes = _children(codelist, "Code")
    if len(codes) <= max_codes:
        return
    by_id = {c.get("id"): c for c in codes}
    keep = codes[:max_codes]
    keep_ids = {c.get("id") for c in keep}
    for code in keep:  # ``keep`` grows while it is walked: parents of parents are kept too
        for parent in _children(code, "Parent"):
            pid = next(iter(_ref_ids(parent)), None)
            if pid and pid not in keep_ids and pid in by_id:
                keep.append(by_id[pid])
                keep_ids.add(pid)
    for code in codes:
        if code.get("id") not in keep_ids:
            codelist.remove(code)


def _strip_text(root: ET.Element) -> None:
    """Drop non-English names, descriptions and annotations other than HIER_CONTEXT."""
    for el in list(root.iter()):
        for child in list(el):
            name = _local(child.tag)
            if name == "Description" or (name == "Name" and child.get(XML_LANG, "en") != "en"):
                el.remove(child)
            elif name == "Annotations":
                for ann in list(child):
                    types = _children(ann, "AnnotationType")
                    if not types or (types[0].text or "") != "HIER_CONTEXT":
                        child.remove(ann)
                if len(child) == 0:
                    el.remove(child)


def minimize(src: str, flow_id: str, dst: str, max_codes: int = DEFAULT_MAX_CODES) -> None:
    """Write the gzipped, trimmed message of ``flow_id`` from the structure message ``src`` to ``dst``."""
    with open(src, "rb") as fh:
        data = fh.read()
    for match in re.finditer(rb'xmlns:(\w+)="([^"]+)"', data[:5000]):  # keep the namespace prefixes
        ET.register_namespace(match.group(1).decode(), match.group(2).decode())
    root = ET.fromstring(data)
    structures = _children(root, "Structures")[0]
    flow = next(f for f in structures.iter()
                if _local(f.tag) == "Dataflow" and f.get("id", "").upper() == flow_id.upper())
    dsd_id = next(iter(_ref_ids(_children(flow, "Structure")[0])))
    dsd = next(d for d in structures.iter() if _local(d.tag) == "DataStructure" and d.get("id") == dsd_id)
    dimensions = _children(_children(dsd, "DataStructureComponents")[0], "DimensionList")[0]
    used_codelists, used_concepts = _used_codelists_and_concepts(structures, dimensions)

    for section in list(structures):
        name = _local(section.tag)
        for item in list(section):
            keep = name not in _DROPPED_SECTIONS
            if name == "Dataflows":
                keep = item is flow
            elif name == "DataStructures":
                keep = item is dsd
            elif name == "Codelists":
                keep = item.get("id") in used_codelists
            elif name == "Concepts":
                for concept in _children(item, "Concept"):
                    if concept.get("id") not in used_concepts:
                        item.remove(concept)
            elif name == "Constraints":
                refs = _ref_ids(item)
                keep = flow_id.upper() in {r.upper() for r in refs} or any(r.startswith("DSD") for r in refs)
            if not keep:
                section.remove(item)
        if len(section) == 0:
            structures.remove(section)

    for codelist in structures.iter():
        if _local(codelist.tag) == "Codelist":
            _trim_codelist(codelist, max_codes)
    _strip_text(root)

    out = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    with gzip.open(dst, "wb") as fh:
        fh.write(out)
    print(f"{dst}: {len(out) // 1024} KB before compression")


if __name__ == "__main__":
    minimize(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_MAX_CODES)
