"""Who added each class and attribute of a project's model.

``project.attribution`` in ``diagrams.json`` maps an element id to
``{"addedBy": <GitHub login>, "addedAt": <UTC time>}``. It sits beside the
diagrams, not inside them: ``ProjectInput`` does not read it, so no converter or
generator sees it, and an element's record merges like any other key. It covers
both editor formats, whose elements keep their ids across a migration.
"""

from datetime import datetime, timezone
from typing import Iterator, Optional

ATTRIBUTION_KEY = "attribution"
_ATTRIBUTED_APOLLON_TYPES = frozenset({
    "Class", "AbstractClass", "Interface", "Enumeration", "ClassAttribute", "ClassMethod",
})


def _class_models(project: dict) -> Iterator[dict]:
    diagrams = (project.get("diagrams") or {}).get("ClassDiagram") or []
    for diagram in diagrams if isinstance(diagrams, list) else [diagrams]:
        model = diagram.get("model") if isinstance(diagram, dict) else None
        if isinstance(model, dict):
            yield model


def _as_list(items: object) -> list:
    return items if isinstance(items, list) else []


def attributed_ids(project: dict) -> set[str]:
    """The ids of the classes and members in ``project``'s class diagrams, in either editor format."""
    found: list = []
    for model in _class_models(project):
        elements = model.get("elements")
        for element_id, element in elements.items() if isinstance(elements, dict) else ():
            if isinstance(element, dict) and element.get("type") in _ATTRIBUTED_APOLLON_TYPES:
                found.append(element_id)
        for node in _as_list(model.get("nodes")):  # React Flow: the members are rows of their class.
            if isinstance(node, dict) and node.get("type") == "class":
                data = node.get("data") if isinstance(node.get("data"), dict) else {}
                rows = _as_list(data.get("attributes")) + _as_list(data.get("methods"))
                found += [node.get("id"), *(row.get("id") for row in rows if isinstance(row, dict))]
    return {element_id for element_id in found if isinstance(element_id, str)}


def _records(project: dict) -> dict:
    records = project.get(ATTRIBUTION_KEY)
    return records if isinstance(records, dict) else {}


def attribute_project(project: dict, base_project: Optional[dict], login: str, now: Optional[datetime] = None) -> dict:
    """``project`` with what it adds over ``base_project`` attributed to ``login``.

    Records of ``base_project`` are kept even when the sender dropped them, and
    records of removed elements go. A model without any record stays without the key.
    """
    current = attributed_ids(project)
    known = attributed_ids(base_project) if base_project else set()
    records = {**_records(base_project or {}), **_records(project)}
    added_at = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    for element_id in current - known:
        records.setdefault(element_id, {"addedBy": login, "addedAt": added_at})
    records = {element_id: records[element_id] for element_id in sorted(records) if element_id in current}
    if not records and ATTRIBUTION_KEY not in project:
        return project
    return {**project, ATTRIBUTION_KEY: records}
