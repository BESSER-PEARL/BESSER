"""Who added each class, attribute and relationship of a project's model.

``project.attribution`` in ``diagrams.json`` maps an element id to
``{"addedBy": <GitHub login>, "addedAt": <UTC time>}``. It sits beside the
diagrams, not inside them: ``ProjectInput`` does not read it, so no converter or
generator sees it, and an element's record merges like any other key. It covers
both editor formats, whose elements keep their ids across a migration.
"""

import re
from datetime import datetime, timezone
from typing import Iterator, Optional

ATTRIBUTION_KEY = "attribution"
# GitHub's managed users have an underscore in their login.
_LOGIN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,38}(\[bot\])?")
_TIME_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
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
    """The ids of the classes, members and relationships in ``project``'s class diagrams, in either editor format."""
    found: list = []
    for model in _class_models(project):
        elements = model.get("elements")
        for element_id, element in elements.items() if isinstance(elements, dict) else ():
            if isinstance(element, dict) and element.get("type") in _ATTRIBUTED_APOLLON_TYPES:
                found.append(element_id)
        relationships = model.get("relationships")
        found += list(relationships) if isinstance(relationships, dict) else []
        found += [edge.get("id") for edge in _as_list(model.get("edges")) if isinstance(edge, dict)]
        for node in _as_list(model.get("nodes")):  # React Flow: the members are rows of their class.
            if isinstance(node, dict) and node.get("type") == "class":
                data = node.get("data") if isinstance(node.get("data"), dict) else {}
                rows = _as_list(data.get("attributes")) + _as_list(data.get("methods"))
                found += [node.get("id"), *(row.get("id") for row in rows if isinstance(row, dict))]
    return {element_id for element_id in found if isinstance(element_id, str)}


def _well_formed(record: object) -> bool:
    return (
        isinstance(record, dict)
        and isinstance(record.get("addedBy"), str) and _LOGIN_RE.fullmatch(record["addedBy"]) is not None
        and isinstance(record.get("addedAt"), str) and _TIME_RE.fullmatch(record["addedAt"]) is not None
    )


def attribution_records(project: dict) -> dict[str, dict[str, str]]:
    """The well-formed records of ``project``, which comes from a repository or a client: only their two fields."""
    records = project.get(ATTRIBUTION_KEY)
    return {
        element_id: {"addedBy": record["addedBy"], "addedAt": record["addedAt"]}
        for element_id, record in (records.items() if isinstance(records, dict) else ())
        if _well_formed(record)
    }


def without_records(project: dict) -> dict:
    """``project`` with no records, for a save whose repo model cannot be read: there is nothing to keep."""
    return {key: value for key, value in project.items() if key != ATTRIBUTION_KEY}


def with_well_formed_records(project: dict) -> dict:
    """``project`` with only well-formed records, for a save GitHub did not answer in time to compare."""
    return {**project, ATTRIBUTION_KEY: attribution_records(project)} if ATTRIBUTION_KEY in project else project


def attribute_project(project: dict, base_project: Optional[dict], login: str, now: Optional[datetime] = None) -> dict:
    """``project`` with what it adds over ``base_project`` attributed to ``login``.

    The records of ``base_project`` stand, whatever the sender says: a sender can
    only add its own login to elements it adds. Records of removed elements go.
    A model without any record, on either side, stays without the key.
    """
    current = attributed_ids(project)
    known = attributed_ids(base_project) if base_project else set()
    base_records = attribution_records(base_project or {})
    records = dict(base_records)
    added_at = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    for element_id in current - known:
        records.setdefault(element_id, {"addedBy": login, "addedAt": added_at})
    records = {element_id: records[element_id] for element_id in sorted(records) if element_id in current}
    if not records and not base_records and ATTRIBUTION_KEY not in project:
        return project
    return {**project, ATTRIBUTION_KEY: records}
