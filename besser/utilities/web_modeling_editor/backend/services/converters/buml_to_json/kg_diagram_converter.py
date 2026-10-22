"""Knowledge Graph diagram conversion — BUML → JSON.

Inverse of ``process_kg_diagram``: serializes a ``KnowledgeGraph`` metamodel
instance to the JSON shape consumed by the frontend.

Not carried through B-UML: the editor's view state. A node's canvas
``position`` and the model-level ``settings`` (layout, visibility limits, …;
``KnowledgeGraphData`` in the frontend's ``shared/types/project.ts``) have no
counterpart in the ``KnowledgeGraph`` metamodel, so ``process_kg_diagram``
ignores them and this module never emits them. A graph that goes through B-UML
(project export/import, ``/export-buml`` → ``/get-json-model``) comes back with
the editor's default settings and is laid out afresh.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict


from besser.BUML.metamodel.kg import (
    KGAxiom,
    KGBlank,
    KGClass,
    KGEdge,
    KGIndividual,
    KGLiteral,
    KGNode,
    KGNodeConstraint,
    KGProperty,
    KGPropertyConstraint,
    KnowledgeGraph,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json._safe_buml_loader import (
    safe_load_buml,
    strip_buml_imports,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError


__all__ = ["kg_to_json", "kg_buml_to_json"]


_CLASS_TO_NODE_TYPE = {
    KGClass: "class",
    KGIndividual: "individual",
    KGProperty: "property",
    KGLiteral: "literal",
    KGBlank: "blank",
    KGNodeConstraint: "nodeConstraint",
    KGPropertyConstraint: "propertyConstraint",
}


def kg_to_json(kg: KnowledgeGraph) -> Dict[str, Any]:
    """Serialize a ``KnowledgeGraph`` to the KG diagram JSON shape.

    The output preserves :class:`KGNode.metadata`, :class:`KGEdge.metadata`,
    and :class:`KnowledgeGraph.axioms` so the OWL-imported information
    survives a JSON round-trip — without that, the preflight detectors
    (PUNNING, RESTRICTION_UNATTACHED, EQUIVALENT_CLASSES, INVERSE_PROPERTY,
    …) would silently miss issues that depend on those fields.
    """
    nodes = [_node_to_dict(n) for n in _sorted_nodes(kg.nodes)]
    edges = [_edge_to_dict(e) for e in _sorted_edges(kg.edges)]
    axioms = [_axiom_to_dict(a) for a in kg.axioms]
    return {
        "title": kg.name,
        "model": {
            "type": "KnowledgeGraphDiagram",
            "version": "1.0.0",
            "nodes": nodes,
            "edges": edges,
            "axioms": axioms,
        },
    }


def kg_buml_to_json(content: str) -> Dict[str, Any]:
    """Load a BUML Python snippet and serialize the resulting ``KnowledgeGraph``.

    Mirrors ``quantum_buml_to_json`` / ``class_buml_to_json`` — used by the
    project round-trip pipeline, which drops the return value straight into a
    diagram entry's ``model`` key. It therefore returns the **bare model**
    (``{"type", "version", "nodes", "edges", "axioms"}``), not the
    ``{"title", "model"}`` envelope :func:`kg_to_json` builds for the HTTP
    endpoints.

    ``content`` arrives from a user upload (``/get-json-model``,
    ``/get-project-json-model``), so it is loaded through
    :func:`safe_load_buml`, which validates the AST against an allowlist before
    executing anything. A restricted-``__builtins__`` ``exec`` would not do: it
    blocks name lookups but not ``().__class__.__base__.__subclasses__()``,
    which reaches ``os`` through attribute access alone. The KG builder only
    ever emits literals and constructor calls, so the allowlist costs nothing.
    """
    import besser.BUML.metamodel.kg as kg_module

    allowed_names: Dict[str, Any] = {
        "set": set,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        # The builder emits set() for an empty node/edge set and float('nan') /
        # float('inf') for non-finite metadata values, since their repr() is a
        # bare name Python does not define. Both are plain constructors; the
        # allowlist's protection comes from refusing unknown calls and dunder
        # attribute access, not from withholding these.
        "float": float,
    }
    # Classes and functions only. ``dir`` also lists the package's submodules
    # (``knowledge_graph``, ``axioms``, ``constraint_specs``, …), and handing a
    # module to the loader hands over everything it imported: an upload could
    # reach ``knowledge_graph.copy.dispatch_table.clear()`` and mutate the
    # interpreter's global state.
    for name in dir(kg_module):
        if name.startswith("_"):
            continue
        member = getattr(kg_module, name)
        if inspect.isclass(member) or inspect.isfunction(member):
            allowed_names[name] = member

    try:
        local_vars = safe_load_buml(strip_buml_imports(content), allowed_names)
    except Exception as exc:
        raise ConversionError(f"Failed to load Knowledge Graph BUML content: {exc}") from exc

    kg: KnowledgeGraph | None = None
    for value in local_vars.values():
        if isinstance(value, KnowledgeGraph):
            kg = value
            break
    if kg is None:
        raise ConversionError("BUML KG code did not define a KnowledgeGraph instance.")
    return kg_to_json(kg)["model"]


def _node_to_dict(n: KGNode) -> Dict[str, Any]:
    node_type = _CLASS_TO_NODE_TYPE.get(type(n))
    if node_type is None:
        # Exact-type lookup on purpose: a subclass the editor does not know
        # would otherwise be passed off as an individual and lose its kind.
        raise ConversionError(
            f"KG node {n.id!r} has unsupported type {type(n).__name__}; expected one of "
            + ", ".join(sorted(cls.__name__ for cls in _CLASS_TO_NODE_TYPE))
            + "."
        )
    d: Dict[str, Any] = {
        "id": n.id,
        "nodeType": node_type,
        "label": n.label,
    }
    if isinstance(n, KGLiteral):
        d["value"] = n.value
        if n.datatype:
            d["datatype"] = n.datatype
    else:
        if n.iri:
            d["iri"] = n.iri
    if n.metadata:
        d["metadata"] = dict(n.metadata)
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
    if e.metadata:
        d["metadata"] = dict(e.metadata)
    return d


def _axiom_to_dict(a: KGAxiom) -> Dict[str, Any]:
    """Serialise an axiom with its concrete subclass name."""
    return {"kind": type(a).__name__, **a.to_dict()}


def _sorted_nodes(nodes):
    return sorted(nodes, key=lambda n: (type(n).__name__, n.id))


def _sorted_edges(edges):
    return sorted(edges, key=lambda e: e.id)
