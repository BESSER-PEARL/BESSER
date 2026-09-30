"""Reject diagrams saved in the legacy v3 editor format.

The JSON -> B-UML converters read only the v4 (React Flow) wire shape,
``{version: '4.x', nodes: [...], edges: [...]}`` (see
``docs/source/migrations/uml-v4-shape.md``). A v3 payload
(``{version: '3.x', elements: {...}, relationships: {...}}``) walks no node
array, so without this guard it converts to an *empty* model and every
endpoint answers 200 with empty code or ``isValid: true``.

The current editor upgrades v3 diagrams on load, so a v3 payload reaching the
backend means a stale client or a hand-made request. It is rejected with a
:class:`LegacyDiagramFormatError` (a ``ConversionError``, so HTTP 400).

GUI (GrapesJS) and quantum-circuit diagrams keep their own formats and are
never treated as legacy UML.
"""

from typing import Any, Iterable, Mapping, Optional

from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    LegacyDiagramFormatError,
)

# Diagram types whose model is not the UML nodes/edges shape.
NON_UML_DIAGRAM_TYPES = frozenset({"GUINoCodeDiagram", "QuantumCircuitDiagram"})

LEGACY_V3_HINT = (
    "open it in the current BESSER web editor (which upgrades it to the v4 "
    "format automatically) and retry, or re-export it from there."
)


def is_legacy_v3_model(model: Any, diagram_type: Optional[str] = None) -> bool:
    """Return True when ``model`` is a UML diagram in the legacy v3 shape.

    A model is legacy when it declares a UML ``type`` and a ``version``
    starting with ``"3."``, or when it carries the v3 ``elements`` /
    ``relationships`` maps and no v4 ``nodes``. GUI and quantum diagrams (by
    ``diagram_type`` or ``model['type']``) are never legacy; the version rule
    also needs a ``type`` because GrapesJS GUI models carry their own,
    unrelated ``version`` and no ``type``.
    """
    if not isinstance(model, Mapping):
        return False
    model_type = model.get("type")
    if diagram_type in NON_UML_DIAGRAM_TYPES or model_type in NON_UML_DIAGRAM_TYPES:
        return False
    version = model.get("version")
    if (
        isinstance(model_type, str) and model_type
        and isinstance(version, str) and version.strip().startswith("3.")
    ):
        return True
    if "nodes" in model:
        return False
    return "elements" in model or "relationships" in model


def legacy_v3_message(
    model: Mapping[str, Any],
    diagram_type: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """Build the user-facing rejection message for a legacy v3 model."""
    dtype = model.get("type") or diagram_type
    what = "This diagram"
    if title and dtype:
        what = f"Diagram '{title}' ({dtype})"
    elif title:
        what = f"Diagram '{title}'"
    elif dtype:
        what = f"This {dtype}"
    version = model.get("version")
    version_note = f" (version {version})" if isinstance(version, str) and version else ""
    return f"{what} uses the legacy v3 editor format{version_note}, which this backend no longer reads; {LEGACY_V3_HINT}"


def ensure_not_legacy_model(
    model: Any,
    diagram_type: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """Raise :class:`LegacyDiagramFormatError` if ``model`` is a legacy v3 UML model."""
    if is_legacy_v3_model(model, diagram_type):
        raise LegacyDiagramFormatError(legacy_v3_message(model, diagram_type, title))


def _as_diagram_list(value: Any) -> Iterable[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping):
        return [value]
    return []


def ensure_project_not_legacy(diagrams: Any) -> None:
    """Check every diagram of a raw project ``diagrams`` map.

    Accepts both the multi-diagram (``{type: [diagram, ...]}``) and the old
    single-diagram (``{type: diagram}``) layouts. Each entry is a
    ``{title, model, ...}`` dict; entries that are not dicts are ignored.
    """
    if not isinstance(diagrams, Mapping):
        return
    for diagram_type, value in diagrams.items():
        for diagram in _as_diagram_list(value):
            if not isinstance(diagram, Mapping):
                continue
            title = diagram.get("title") if isinstance(diagram.get("title"), str) else None
            ensure_diagram_not_legacy(
                diagram.get("model"),
                diagram_type=diagram_type,
                title=title,
                reference=diagram.get("referenceDiagramData"),
            )


def ensure_diagram_not_legacy(
    model: Any,
    diagram_type: Optional[str] = None,
    title: Optional[str] = None,
    reference: Any = None,
) -> None:
    """Check one diagram: its model, the model's embedded
    ``referenceDiagramData`` (object diagrams) and a sibling ``reference``."""
    ensure_not_legacy_model(model, diagram_type, title)
    if isinstance(model, Mapping):
        ensure_not_legacy_reference(model.get("referenceDiagramData"), title)
    ensure_not_legacy_reference(reference, title)


def ensure_not_legacy_reference(reference: Any, title: Optional[str] = None) -> None:
    """Check a ``referenceDiagramData`` payload (a bare model or a ``{model}`` wrapper)."""
    if not isinstance(reference, Mapping):
        return
    inner = reference.get("model")
    model = inner if isinstance(inner, Mapping) else reference
    ref_title = reference.get("title") if isinstance(reference.get("title"), str) else None
    if ref_title is None and title:
        ref_title = f"reference of {title}"
    ensure_not_legacy_model(model, None, ref_title)
