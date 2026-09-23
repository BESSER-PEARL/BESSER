"""Deployment model code builder.

Generates a self-contained Python module from a ``DeploymentModel`` that
re-creates the model on ``exec()``. Implements 03-... §3 / §4.2 / §5
(option-b emission, sort_by_timestamp determinism, layout passthrough,
manifests dot-assignment per Q1=a, ``Multiplicity`` emission with
conditional ``UNLIMITED_MAX_MULTIPLICITY`` import).
"""

from typing import Optional

from besser.BUML.metamodel.structural import UNLIMITED_MAX_MULTIPLICITY
from besser.BUML.metamodel.uml_deployment import (
    Artifact,
    CommunicationPath,
    DeploymentDependency,
    DeploymentModel,
    DeploymentRelation,
    Interface,
    InterfaceProvided,
    InterfaceRequired,
    Node,
)
from besser.utilities.buml_code_builder.common import (
    _escape_python_string,
    safe_var_name,
)
from besser.utilities.utils import sort_by_timestamp


# ---------------------------------------------------------------------------
# Helpers (mirror the Component builder; kept private per-module)
# ---------------------------------------------------------------------------

class _NameDispenser:
    """Mint unique Python variable names keyed by metamodel object identity."""

    def __init__(self):
        self._used: set = set()
        self._for_obj: dict = {}

    def name_for(self, obj) -> str:
        if obj in self._for_obj:
            return self._for_obj[obj]
        prefix = type(obj).__name__.lower()
        base = safe_var_name(obj.name, lowercase=True) or prefix
        candidate = f"{prefix}_{base}" if base != prefix else prefix
        suffix = 1
        while candidate in self._used:
            candidate = f"{prefix}_{base}_{suffix}"
            suffix += 1
        self._used.add(candidate)
        self._for_obj[obj] = candidate
        return candidate


def _emit_str_list(values) -> str:
    if not values:
        return "[]"
    parts = [f"'{_escape_python_string(v)}'" for v in values]
    return f"[{', '.join(parts)}]"


def _emit_layout_line(var_name: str, layout: Optional[dict]) -> Optional[str]:
    if not layout:
        return None
    return f"{var_name}.layout = {layout!r}"


def _emit_multiplicity(multiplicity) -> str:
    """Emit a ``Multiplicity(min, max)`` constructor expression.

    Uses the ``UNLIMITED_MAX_MULTIPLICITY`` symbol when the upper bound is
    unbounded (so the import-emission code can conditionally pull it in).
    """
    if multiplicity is None:
        return "None"
    if multiplicity.max == UNLIMITED_MAX_MULTIPLICITY:
        return f"Multiplicity({multiplicity.min}, UNLIMITED_MAX_MULTIPLICITY)"
    return f"Multiplicity({multiplicity.min}, {multiplicity.max})"


def _needs_unlimited_constant(model: DeploymentModel) -> bool:
    """Check if any DeploymentRelation uses UNLIMITED_MAX_MULTIPLICITY."""
    for rel in model.relationships:
        if isinstance(rel, DeploymentRelation):
            if rel.multiplicity.max == UNLIMITED_MAX_MULTIPLICITY:
                return True
    return False


def _topological_sort_nodes(nodes) -> list:
    """Sort Nodes so every parent precedes its children.

    Within each depth level, sort by Element.timestamp so the emitted
    `.py` is reproducible byte-for-byte for a given model.
    """
    by_depth: dict = {}
    for node in nodes:
        depth = 0
        cursor = node.parent
        while cursor is not None:
            depth += 1
            cursor = cursor.parent
        by_depth.setdefault(depth, []).append(node)
    result: list = []
    for depth in sorted(by_depth):
        result.extend(sort_by_timestamp(by_depth[depth]))
    return result


def _collect_imports(model: DeploymentModel) -> tuple:
    """Return ``(metamodel_imports, structural_imports)`` lists.

    ``Multiplicity`` is included from ``structural`` only if any
    DeploymentRelation exists; ``UNLIMITED_MAX_MULTIPLICITY`` is added on top
    if needed.
    """
    needed = {"DeploymentModel"}
    has_node = False
    has_locality_nondefault = False
    has_nondefault_kind = False

    all_nodes = list(model.all_nodes())
    all_artifacts = list(model.all_artifacts())

    if all_nodes:
        needed.add("Node")
        has_node = True
    if all_artifacts:
        needed.add("Artifact")
    if model.interfaces:
        needed.add("Interface")

    for n in all_nodes:
        if n.kind.name != "GENERIC":
            has_nondefault_kind = True
        if n.locality.name != "LOCAL":
            has_locality_nondefault = True
    for a in all_artifacts:
        if a.locality.name != "LOCAL":
            has_locality_nondefault = True

    has_relation = False
    for rel in model.relationships:
        if isinstance(rel, DeploymentRelation):
            needed.add("DeploymentRelation")
            has_relation = True
        elif isinstance(rel, CommunicationPath):
            needed.add("CommunicationPath")
        elif isinstance(rel, DeploymentDependency):
            needed.add("DeploymentDependency")
        elif isinstance(rel, InterfaceProvided):
            needed.add("InterfaceProvided")
        elif isinstance(rel, InterfaceRequired):
            needed.add("InterfaceRequired")

    enum_imports = []
    if has_nondefault_kind and has_node:
        enum_imports.append("NodeKind")
    if has_locality_nondefault:
        enum_imports.append("Locality")

    ordered_meta = [
        c for c in (
            "DeploymentModel", "Node", "Artifact", "Interface",
            "DeploymentRelation", "CommunicationPath",
            "DeploymentDependency",
            "InterfaceProvided", "InterfaceRequired",
        )
        if c in needed
    ] + enum_imports

    structural = []
    if has_relation:
        structural.append("Multiplicity")
        if _needs_unlimited_constant(model):
            structural.append("UNLIMITED_MAX_MULTIPLICITY")

    return ordered_meta, structural


# ---------------------------------------------------------------------------
# Element emission
# ---------------------------------------------------------------------------

def _emit_node(node: Node, var_name: str) -> list:
    """Emit a Node constructor + post-construction lines."""
    lines = []
    args = [f"name='{_escape_python_string(node.name)}'"]
    if node.kind.name != "GENERIC":
        args.append(f"kind=NodeKind.{node.kind.name}")
    if node.locality.name != "LOCAL":
        args.append(f"locality=Locality.{node.locality.name}")
    if node.stereotypes:
        args.append(f"stereotypes={_emit_str_list(node.stereotypes)}")
    lines.append(f"{var_name} = Node({', '.join(args)})")
    layout_line = _emit_layout_line(var_name, node.layout)
    if layout_line is not None:
        lines.append(layout_line)
    return lines


def _emit_artifact(artifact: Artifact, var_name: str) -> list:
    """Emit an Artifact constructor + post-construction lines.

    ``manifests`` round-trips via dot-assignment (Q1=a), kept off the
    constructor for symmetry with the Component side.
    """
    lines = []
    args = [f"name='{_escape_python_string(artifact.name)}'"]
    if artifact.locality.name != "LOCAL":
        args.append(f"locality=Locality.{artifact.locality.name}")
    if artifact.stereotypes:
        args.append(f"stereotypes={_emit_str_list(artifact.stereotypes)}")
    lines.append(f"{var_name} = Artifact({', '.join(args)})")
    if artifact.manifests:
        lines.append(f"{var_name}.manifests = {_emit_str_list(artifact.manifests)}")
    layout_line = _emit_layout_line(var_name, artifact.layout)
    if layout_line is not None:
        lines.append(layout_line)
    return lines


def _emit_interface(interface: Interface, var_name: str) -> list:
    """Emit an Interface constructor + optional layout line."""
    lines = []
    args = [f"name='{_escape_python_string(interface.name)}'"]
    if interface.stereotypes:
        args.append(f"stereotypes={_emit_str_list(interface.stereotypes)}")
    lines.append(f"{var_name} = Interface({', '.join(args)})")
    layout_line = _emit_layout_line(var_name, interface.layout)
    if layout_line is not None:
        lines.append(layout_line)
    return lines


def _emit_relationship(rel, dispenser: _NameDispenser) -> list:
    """Emit a relationship — inline or via a temp variable when it has a
    layout. Returns the source lines including the
    ``deployment_model.add_relationship(...)`` call."""
    rel_class = type(rel).__name__
    source_var = dispenser.name_for(rel.source)
    target_var = dispenser.name_for(rel.target)

    args = [f"source={source_var}", f"target={target_var}"]
    if isinstance(rel, DeploymentRelation):
        # Only emit non-default multiplicity. Default Multiplicity(1, 1)
        # is omitted for compact round-trip.
        mult = rel.multiplicity
        if not (mult.min == 1 and mult.max == 1):
            args.append(f"multiplicity={_emit_multiplicity(mult)}")
    if rel.name:
        args.append(f"name='{_escape_python_string(rel.name)}'")
    if rel.stereotypes:
        args.append(f"stereotypes={_emit_str_list(rel.stereotypes)}")

    constructor = f"{rel_class}({', '.join(args)})"

    if rel.layout:
        rel_var = dispenser.name_for(rel)
        return [
            f"{rel_var} = {constructor}",
            _emit_layout_line(rel_var, rel.layout),
            f"deployment_model.add_relationship({rel_var})",
        ]
    return [f"deployment_model.add_relationship({constructor})"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def deployment_model_to_code(model: DeploymentModel,
                             file_path: Optional[str] = None,
                             model_var_name: str = "deployment_model") -> str:
    """Generate Python code that re-creates ``model`` on ``exec()``.

    Args:
        model: The ``DeploymentModel`` to serialise.
        file_path: Optional path to write the generated code to.
        model_var_name: Variable name for the model in the generated code.

    Returns:
        The generated Python source as a string.
    """
    if not isinstance(model, DeploymentModel):
        raise TypeError(
            f"deployment_model_to_code expects a DeploymentModel, "
            f"got {type(model).__name__}."
        )

    dispenser = _NameDispenser()
    lines: list = []

    # ----- Header banner + imports -----
    lines.append("####################")
    lines.append("# DEPLOYMENT MODEL #")
    lines.append("####################")
    lines.append("")

    meta_imports, structural_imports = _collect_imports(model)
    if len(meta_imports) <= 4:
        joined = ", ".join(meta_imports)
        lines.append(
            f"from besser.BUML.metamodel.uml_deployment import {joined}"
        )
    else:
        lines.append("from besser.BUML.metamodel.uml_deployment import (")
        for symbol in meta_imports:
            lines.append(f"    {symbol},")
        lines.append(")")

    if structural_imports:
        joined = ", ".join(structural_imports)
        lines.append(
            f"from besser.BUML.metamodel.structural import {joined}"
        )
    lines.append("")

    # ----- Model constructor -----
    lines.append(
        f"{model_var_name} = DeploymentModel("
        f"name='{_escape_python_string(model.name)}')"
    )
    lines.append("")

    # ----- Nodes — topologically sorted (parents before children) so
    # `parent.add_nested_node(child)` has its parent var already in scope.
    # Within each depth level, sort by Element.timestamp for determinism.
    all_nodes_sorted = _topological_sort_nodes(model.all_nodes())
    if all_nodes_sorted:
        lines.append("# --- Nodes ---")
        for node in all_nodes_sorted:
            var = dispenser.name_for(node)
            lines.extend(_emit_node(node, var))
            if node.parent is None:
                lines.append(f"{model_var_name}.add_node({var})")
            else:
                parent_var = dispenser.name_for(node.parent)
                lines.append(f"{parent_var}.add_nested_node({var})")
            lines.append("")

    # ----- Artifacts — all artifacts (root + nested in any node).
    all_artifacts_sorted = sort_by_timestamp(model.all_artifacts())
    if all_artifacts_sorted:
        lines.append("# --- Artifacts ---")
        for artifact in all_artifacts_sorted:
            var = dispenser.name_for(artifact)
            lines.extend(_emit_artifact(artifact, var))
            if artifact.parent is None:
                lines.append(f"{model_var_name}.add_artifact({var})")
            else:
                parent_var = dispenser.name_for(artifact.parent)
                lines.append(f"{parent_var}.add_artifact({var})")
            lines.append("")

    # ----- Interfaces.
    interfaces_sorted = sort_by_timestamp(model.interfaces)
    if interfaces_sorted:
        lines.append("# --- Interfaces ---")
        for interface in interfaces_sorted:
            var = dispenser.name_for(interface)
            lines.extend(_emit_interface(interface, var))
            lines.append(f"{model_var_name}.add_interface({var})")
            lines.append("")

    # ----- Relationships.
    relationships_sorted = sort_by_timestamp(model.relationships)
    if relationships_sorted:
        lines.append("# --- Relationships ---")
        for rel in relationships_sorted:
            lines.extend(_emit_relationship(rel, dispenser))
            lines.append("")

    source = "\n".join(line for line in lines if line is not None) + "\n"

    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(source)
    return source
