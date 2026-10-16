"""Canonical, order-independent-where-it-should-be dumps of pipeline output.

Rules: sort anything whose order is not meaningful; keep anything whose order
*is* meaningful (the warning list) in order, because shifted warning order is
exactly the symptom of a refactor changing a tie-break.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

import rdflib
from rdflib.compare import to_canonical_graph


def _meta(element) -> Any:
    meta = getattr(element, "metadata", None)
    if meta is None:
        return None
    return {"uri": getattr(meta, "uri", None), "description": getattr(meta, "description", None)}


def _type_name(t) -> str:
    return f"{type(t).__name__}:{getattr(t, 'name', t)}"


def _mult(m) -> List[Any]:
    return [m.min, m.max]


def _prop(p) -> Dict[str, Any]:
    return {
        "name": p.name,
        "type": _type_name(p.type),
        "mult": _mult(p.multiplicity),
        "is_id": bool(getattr(p, "is_id", False)),
        "is_derived": bool(getattr(p, "is_derived", False)),
        "meta": _meta(p),
    }


def dump_class_result(res) -> Dict[str, Any]:
    dm = res.domain_model
    classes = []
    for c in sorted(dm.get_classes(), key=lambda c: c.name):
        classes.append({
            "name": c.name,
            "abstract": bool(c.is_abstract),
            "meta": _meta(c),
            "parents": sorted(p.name for p in c.parents()),
            "attributes": sorted((_prop(a) for a in c.attributes), key=lambda d: d["name"]),
        })
    enums = [
        {"name": e.name, "literals": sorted(l.name for l in e.literals)}
        for e in sorted(dm.get_enumerations(), key=lambda e: e.name)
    ]
    gens = sorted((g.specific.name, g.general.name) for g in dm.generalizations)
    assocs = []
    for a in sorted(dm.associations, key=lambda a: a.name):
        ends = sorted(
            ({
                "name": e.name, "type": e.type.name, "mult": _mult(e.multiplicity),
                "navigable": bool(e.is_navigable), "composite": bool(e.is_composite),
            } for e in a.ends),
            key=lambda d: d["name"],
        )
        assocs.append({"name": a.name, "ends": ends})
    # DomainModel.constraints is a set, so its order is not a contract: sort.
    constraints = sorted(
        ({"name": c.name, "context": c.context.name, "expression": c.expression, "language": c.language}
         for c in dm.constraints),
        key=lambda d: (d["context"], d["name"], d["expression"]),
    )
    assoc_by_id = {id(a): a for a in dm.associations}
    return {
        "name": dm.name,
        "classes": classes,
        "enumerations": enums,
        "generalizations": gens,
        "associations": assocs,
        "constraints": constraints,
        "iri_to_class": {k: v.name for k, v in sorted(res.iri_to_class.items())},
        "property_iri_to_attribute": {
            k: f"{getattr(getattr(v, 'owner', None), 'name', None)}.{v.name}"
            for k, v in sorted(res.property_iri_to_attribute.items())
        },
        "property_iri_to_association": {
            k: v.name for k, v in sorted(res.property_iri_to_association.items())
        },
        "assoc_source_end": {
            assoc_by_id[k].name: v.name
            for k, v in res.assoc_source_end.items() if k in assoc_by_id
        },
        "class_stereotypes": {k: list(v) for k, v in sorted(res.class_stereotypes.items())},
        # In order, on purpose.
        "warnings": [asdict(w) for w in res.warnings],
    }


def dump_graph(graph: rdflib.Graph) -> List[str]:
    """Isomorphism-stable triples: blank nodes relabelled deterministically."""
    canonical = to_canonical_graph(graph)
    return sorted(f"{s.n3()} {p.n3()} {o.n3()}" for s, p, o in canonical)


def _stable_message(message: str) -> str:
    """Keep the part of a pyshacl message that does not depend on hash order.

    pyshacl renders the focus node's blank-node neighbourhood into the message,
    and for self-referential structures that text varies with PYTHONHASHSEED.
    That is third-party output, not something BESSER controls; everything BESSER
    derives (id, order, label, node ids) is compared exactly.
    """
    return message.split(" [", 1)[0]


def dump_consistency(report) -> Dict[str, Any]:
    issues = []
    for issue in report.issues:                            # report order is part of the contract
        record = asdict(issue)
        record["message"] = _stable_message(record["message"])
        issues.append(record)
    return {
        "issues": issues,
        "issue_count": report.issue_count,
        "severity_counts": dict(sorted(report.severity_counts.items())),
        "kg_signature": report.kg_signature,
        "inference_used": report.inference_used,
    }
