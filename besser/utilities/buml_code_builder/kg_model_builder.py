"""
KG Model Builder: Generates Python code for BESSER KnowledgeGraph models.

Counterpart of
``besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json
.kg_diagram_converter.kg_buml_to_json``, which reads the code this module
writes. Two constraints follow from that pairing and shape everything below:

1. **The generated source is executed, not imported.** ``kg_buml_to_json``
   strips the import block and ``exec()``s the rest against a restricted
   namespace whose only builtins are ``set list dict tuple str int float bool
   len range print`` plus the public names of ``besser.BUML.metamodel.kg``.
   The emitted code therefore sticks to literals and constructor calls — no
   comprehensions over helpers like ``sorted`` or ``enumerate``, and ``set()``
   rather than a bare ``{}`` for an empty node/edge set.

2. **The section banner has to be written here.** ``project_to_code`` only
   prepends a numbered ``# KNOWLEDGE_GRAPH MODEL n: "…" #`` header when a
   project holds more than one KG, so a lone KG would otherwise carry no
   header at all and ``project_to_json``'s section regex would never find it.
   ``domain_model_builder``, ``agent_model_builder`` and ``bpmn_model_builder``
   all emit their own banner for the same reason.

Values are rendered with :func:`repr` rather than the usual
``_escape_python_string``. Node and edge ``metadata`` carries arbitrarily
nested constraint specs imported from OWL/SHACL, and hand-escaping nested
structures is error-prone where ``repr`` is correct by construction for every
JSON-safe type. :func:`_literal` guarantees only such types reach it.
"""

import hashlib
from typing import Any, Dict, List, Set, TextIO

from besser.BUML.metamodel.kg import (
    KGBlank,
    KGClass,
    KGIndividual,
    KGLiteral,
    KGNode,
    KGNodeConstraint,
    KGProperty,
    KGPropertyConstraint,
    KnowledgeGraph,
)
from besser.utilities.buml_code_builder.common import _comment_safe, safe_var_name


__all__ = ["kg_model_to_code"]


#: Node class -> the constructor name emitted in the generated code. Mirrors
#: ``_CLASS_TO_NODE_TYPE`` in ``kg_diagram_converter`` so both directions agree
#: on which of the seven node kinds a node belongs to.
_NODE_CLASSES = (
    KGClass,
    KGIndividual,
    KGProperty,
    KGLiteral,
    KGBlank,
    KGNodeConstraint,
    KGPropertyConstraint,
)

#: Every axiom dataclass the metamodel defines, imported by the generated code.
_AXIOM_NAMES = [
    "EquivalentClassesAxiom",
    "EquivalentPropertiesAxiom",
    "DisjointClassesAxiom",
    "DisjointUnionAxiom",
    "SubPropertyOfAxiom",
    "InversePropertiesAxiom",
    "PropertyChainAxiom",
    "HasKeyAxiom",
    "ImportAxiom",
]


def kg_model_to_code(model: KnowledgeGraph, file_path: str, model_var_name: str = "kg_model"):
    """
    Generates Python code for a KnowledgeGraph model.

    Args:
        model (KnowledgeGraph): The knowledge graph model.
        file_path (str): The path to save the generated code.
        model_var_name (str, optional): Name of the KnowledgeGraph variable in
            the generated code. Defaults to ``"kg_model"``, which is the key
            ``project_to_json`` looks up in its ``SECTION_CONFIG`` table —
            any other name and the section is silently skipped on import.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        _write_banner(f)
        _write_imports(f)

        node_vars = _write_nodes(f, model)
        edge_vars = _write_edges(f, model, node_vars)
        _write_graph(f, model, node_vars, edge_vars, model_var_name)


def _write_banner(f: TextIO) -> None:
    f.write("###########################\n")
    f.write("# KNOWLEDGE_GRAPH MODEL   #\n")
    f.write("###########################\n\n")


def _write_imports(f: TextIO) -> None:
    f.write("from besser.BUML.metamodel.kg import (\n")
    f.write("    KnowledgeGraph, KGClass, KGIndividual, KGProperty, KGLiteral,\n")
    f.write("    KGBlank, KGNodeConstraint, KGPropertyConstraint, KGEdge,\n")
    for name in _AXIOM_NAMES:
        f.write(f"    {name},\n")
    f.write(")\n\n")


def _write_nodes(f: TextIO, model: KnowledgeGraph) -> Dict[str, str]:
    """Emit one constructor per node; return a mapping of node id -> variable name."""
    node_vars: Dict[str, str] = {}
    used: Set[str] = set()

    f.write("# Nodes\n")
    for node in _sorted_nodes(model.nodes):
        var = _unique_var("kg_node", node.label or _short_hash(node.id), used)
        node_vars[node.id] = var

        args = [f"id={_literal(node.id)}"]
        if isinstance(node, KGLiteral):
            args.append(f"value={_literal(node.value)}")
            if node.datatype:
                args.append(f"datatype={_literal(node.datatype)}")
            args.append(f"label={_literal(node.label)}")
        else:
            args.append(f"label={_literal(node.label)}")
            if node.iri:
                args.append(f"iri={_literal(node.iri)}")
        if node.metadata:
            args.append(f"metadata={_literal(node.metadata)}")

        f.write(f"{var} = {_node_class_name(node)}({', '.join(args)})\n")

    if not node_vars:
        f.write("# (none)\n")
    f.write("\n")
    return node_vars


def _write_edges(f: TextIO, model: KnowledgeGraph, node_vars: Dict[str, str]) -> List[str]:
    """Emit one constructor per edge; return the variable names in emission order."""
    edge_vars: List[str] = []
    used: Set[str] = set()

    f.write("# Edges\n")
    for edge in _sorted_edges(model.edges):
        var = _unique_var("kg_edge", edge.label or _short_hash(edge.id), used)
        edge_vars.append(var)

        args = [
            f"id={_literal(edge.id)}",
            f"source={node_vars[edge.source.id]}",
            f"target={node_vars[edge.target.id]}",
            f"label={_literal(edge.label)}",
        ]
        if edge.iri:
            args.append(f"iri={_literal(edge.iri)}")
        if edge.metadata:
            args.append(f"metadata={_literal(edge.metadata)}")

        f.write(f"{var} = KGEdge({', '.join(args)})\n")

    if not edge_vars:
        f.write("# (none)\n")
    f.write("\n")
    return edge_vars


def _write_graph(
    f: TextIO,
    model: KnowledgeGraph,
    node_vars: Dict[str, str],
    edge_vars: List[str],
    model_var_name: str,
) -> None:
    f.write(f"# Knowledge Graph: {_comment_safe(model.name)}\n")
    f.write(f"{model_var_name} = KnowledgeGraph(\n")
    f.write(f"    name={_literal(model.name)},\n")

    # ``set()`` rather than ``{}``, which is an empty *dict*. ``set`` is one of
    # the builtins ``kg_buml_to_json`` leaves in the exec namespace.
    ordered_node_vars = [node_vars[n.id] for n in _sorted_nodes(model.nodes)]
    f.write(f"    nodes={_var_set(ordered_node_vars)},\n")
    f.write(f"    edges={_var_set(edge_vars)},\n")

    f.write("    axioms=[\n")
    for axiom in model.axioms:
        f.write(f"        {_axiom_expr(axiom)},\n")
    f.write("    ],\n")
    f.write(")\n")


# ----------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------


def _node_class_name(node: KGNode) -> str:
    for cls in _NODE_CLASSES:
        if type(node) is cls:
            return cls.__name__
    # Defensive: a future node kind should still round-trip as an individual
    # rather than emitting a name the exec namespace does not define.
    return "KGIndividual"


def _axiom_expr(axiom: Any) -> str:
    """Render one axiom dataclass as a keyword-argument constructor call."""
    fields = getattr(axiom, "__dataclass_fields__", {})
    args = [f"{name}={_literal(getattr(axiom, name))}" for name in fields]
    return f"{type(axiom).__name__}({', '.join(args)})"


def _var_set(var_names: List[str]) -> str:
    if not var_names:
        return "set()"
    return "{" + ", ".join(var_names) + "}"


def _literal(value: Any) -> str:
    """Render ``value`` as a Python literal safe to ``exec``.

    Only JSON-safe types are rendered structurally; anything else degrades to
    its ``str()`` form so a stray object can never emit a ``repr`` that is not
    valid Python (``<Foo object at 0x…>``).
    """
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, dict):
        items = ", ".join(f"{_literal(k)}: {_literal(v)}" for k, v in value.items())
        return "{" + items + "}"
    if isinstance(value, (list, tuple, set)):
        rendered = ", ".join(_literal(v) for v in value)
        return "[" + rendered + "]"
    return repr(str(value))


def _unique_var(prefix: str, hint: str, used: Set[str]) -> str:
    """Build a collision-free variable name from an arbitrary label.

    KG identifiers are IRIs, blank-node ids and literal values, none of which
    are Python identifiers, so the label is only ever a *hint*: it is
    sanitised, truncated, and disambiguated with a counter when two nodes
    sanitise to the same thing.
    """
    base = safe_var_name(hint or "node", lowercase=False)[:40] or "node"
    candidate = f"{prefix}_{base}"
    if candidate not in used:
        used.add(candidate)
        return candidate
    counter = 2
    while f"{candidate}_{counter}" in used:
        counter += 1
    unique = f"{candidate}_{counter}"
    used.add(unique)
    return unique


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]


def _sorted_nodes(nodes):
    """Mirror ``kg_diagram_converter._sorted_nodes`` so both halves of the
    round trip agree on ordering, and the output stays deterministic."""
    return sorted(nodes, key=lambda n: (type(n).__name__, n.id))


def _sorted_edges(edges):
    return sorted(edges, key=lambda e: e.id)
