"""Knowledge Graph diagram conversion — BUML → JSON.

Inverse of ``process_kg_diagram``: serializes a ``KnowledgeGraph`` metamodel
instance to the JSON shape consumed by the frontend.
"""

from __future__ import annotations

from typing import Any, Dict

from besser.BUML.metamodel.kg import (
    KGBlank,
    KGClass,
    KGEdge,
    KGIndividual,
    KGLiteral,
    KGNode,
    KGProperty,
    KnowledgeGraph,
)


__all__ = ["kg_to_json", "kg_buml_to_json"]


_CLASS_TO_NODE_TYPE = {
    KGClass: "class",
    KGIndividual: "individual",
    KGProperty: "property",
    KGLiteral: "literal",
    KGBlank: "blank",
}


def kg_to_json(kg: KnowledgeGraph) -> Dict[str, Any]:
    """Serialize a ``KnowledgeGraph`` to the KG diagram JSON shape."""
    nodes = [_node_to_dict(n) for n in _sorted_nodes(kg.nodes)]
    edges = [_edge_to_dict(e) for e in _sorted_edges(kg.edges)]
    return {
        "title": kg.name,
        "model": {
            "type": "KnowledgeGraphDiagram",
            "version": "1.0.0",
            "nodes": nodes,
            "edges": edges,
        },
    }


def kg_buml_to_json(content: str) -> Dict[str, Any]:
    """Execute a BUML Python snippet and serialize the resulting ``KnowledgeGraph``.

    Mirrors ``quantum_buml_to_json`` / ``class_buml_to_json`` — used by the
    project round-trip pipeline.
    """
    import besser.BUML.metamodel.kg as kg_module

    safe_globals: Dict[str, Any] = {
        "__builtins__": {
            "set": set,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "len": len,
            "range": range,
            "True": True,
            "False": False,
            "None": None,
            "print": lambda *a, **kw: None,
        },
    }
    for name in dir(kg_module):
        if not name.startswith("_"):
            safe_globals[name] = getattr(kg_module, name)

    cleaned_lines = []
    in_import_block = False
    for line in content.splitlines():
        stripped = line.lstrip()
        if in_import_block:
            if ")" in line:
                in_import_block = False
            continue
        if stripped.startswith(("import ", "from ")):
            if "(" in line and ")" not in line:
                in_import_block = True
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)

    local_vars: Dict[str, Any] = {}
    exec(cleaned, safe_globals, local_vars)  # noqa: S102 — trusted BUML code

    kg: KnowledgeGraph | None = None
    for value in local_vars.values():
        if isinstance(value, KnowledgeGraph):
            kg = value
            break
    if kg is None:
        raise ValueError("BUML KG code did not define a KnowledgeGraph instance.")
    return kg_to_json(kg)


def _node_to_dict(n: KGNode) -> Dict[str, Any]:
    node_type = _CLASS_TO_NODE_TYPE.get(type(n), "individual")
    d: Dict[str, Any] = {
        "id": n.id,
        "nodeType": node_type,
        "label": n.label,
    }
    if node_type == "literal":
        assert isinstance(n, KGLiteral)
        d["value"] = n.value
        if n.datatype:
            d["datatype"] = n.datatype
    else:
        if n.iri:
            d["iri"] = n.iri
    return d


def _edge_to_dict(e: KGEdge) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "id": e.id,
        "source": e.source.id,
        "target": e.target.id,
    }
    if e.label:
        d["label"] = e.label
    if e.iri:
        d["iri"] = e.iri
    return d


def _sorted_nodes(nodes):
    return sorted(nodes, key=lambda n: (type(n).__name__, n.id))


def _sorted_edges(edges):
    return sorted(edges, key=lambda e: e.id)
