"""How the model of a modify run relates to the model its base app was built from.

A modify run edits the previous app in place. Without being told what changed
in the class diagram, it keeps old and new elements side by side; for an
unrelated model it writes the new app next to the old one. This compares the
two models without any LLM call (#659):

* ``model_summary`` is what the recipe stores as ``model.user_model``: the
  model the user sent, before a modify run's model-sync added classes to it.
* ``describe_model_change`` lists every change (classes, attribute types and
  optionality, enumeration literals, association ends) for the modify run.
* ``compare_models`` gives a verdict (small / large / unrelated) so the UI can
  ask "update the existing app / start fresh" before a run.

Elements are matched by ``id`` when both summaries carry one (a rename then
reads as a rename), otherwise by name. Older recipes only have the run's class
names, model-sync classes included: they give a verdict but no change list.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field

SMALL, LARGE, UNRELATED = "small", "large", "unrelated"

# Share of the base model (class and attribute names) gone above which a
# change is large. Measured on the hotel model: add an attribute 0, rename a
# class 0.04, change ~25% of the attributes 0.11, ~50% 0.32, keep the classes
# and change everything else 0.54-0.64.
LARGE_GONE = 0.15
# Another app: most of the base model is gone, the new model mostly reuses no
# base names (so deleting classes is not "another app"), and few base classes
# or names remain.
UNRELATED_GONE = 0.5
UNRELATED_REUSED = 0.5
UNRELATED_CLASSES_KEPT = 0.5
UNRELATED_OVERLAP = 0.25


def _with_id(element, entry: dict) -> dict:
    element_id = getattr(element, "id", None)
    if element_id is not None:
        entry["id"] = str(element_id)
    return entry


def model_summary(domain_model) -> dict:
    """Classes (attribute types, optional/required), enumerations and
    associations of ``domain_model``, as stored in the recipe."""
    from besser.BUML.metamodel.structural.structural import UNLIMITED_MAX_MULTIPLICITY

    classes = {}
    for cls in sorted(domain_model.get_classes(), key=lambda c: c.name):
        attributes = {}
        for attr in sorted(cls.attributes, key=lambda a: a.name):
            entry = {"type": getattr(attr.type, "name", str(attr.type)),
                     "optional": bool(attr.is_optional)}
            if getattr(attr, "is_derived", False):
                entry["derived"] = True
            attributes[attr.name] = _with_id(attr, entry)
        classes[cls.name] = _with_id(cls, {"attributes": attributes})
    enumerations = {
        enum.name: _with_id(enum, {"literals": sorted(lit.name for lit in enum.literals)})
        for enum in sorted(domain_model.get_enumerations(), key=lambda e: e.name)
    }
    associations = {}
    for assoc in sorted(domain_model.associations, key=lambda a: a.name):
        ends = list(assoc.ends)
        rendered = []
        for end in ends:
            # An end is typed by the class it points AT; it belongs to the other end's class.
            owner = next((o for o in ends if o is not end), end).type.name
            upper = "*" if end.multiplicity.max == UNLIMITED_MAX_MULTIPLICITY else end.multiplicity.max
            rendered.append(f"{owner}.{end.name}[{end.multiplicity.min}..{upper}]")
        associations[assoc.name] = _with_id(assoc, {"ends": " - ".join(sorted(rendered))})
    return {"classes": classes, "enumerations": enumerations, "associations": associations}


def is_summary(summary) -> bool:
    """A ``model_summary`` dict (older recipes hold class-name lists)."""
    return isinstance(summary, dict) and isinstance(summary.get("classes"), dict)


def pair_elements(old: dict, new: dict) -> list[tuple[str, str]]:
    """``(old name, new name)`` of the elements present on both sides: same
    ``id`` first, then the same name, then the same name ignoring case."""
    pairs, used_old, used_new = [], set(), set()

    def take(o, n):
        pairs.append((o, n))
        used_old.add(o)
        used_new.add(n)

    old_by_id = {}
    for name, entry in old.items():
        if isinstance(entry, dict) and entry.get("id"):
            old_by_id.setdefault(entry["id"], name)
    for name, entry in new.items():
        match = old_by_id.get(entry.get("id")) if isinstance(entry, dict) and entry.get("id") else None
        if match is not None and match not in used_old:
            take(match, name)
    for name in new:
        if name not in used_new and name in old and name not in used_old:
            take(name, name)
    lowered = {o.lower(): o for o in old if o not in used_old}
    for name in new:
        match = lowered.get(name.lower())
        if name not in used_new and match is not None and match not in used_old:
            take(match, name)
    return sorted(pairs)


def _attributes(entry) -> dict:
    attributes = entry.get("attributes") if isinstance(entry, dict) else None
    return attributes if isinstance(attributes, dict) else {}


def _typed(entry: dict) -> str:
    kind = entry.get("type")
    if not kind:
        return ""
    return f": {kind}{' (optional)' if entry.get('optional') else ''}"


def _attribute_changes(old_cls: str, before: dict, new_cls: str, after: dict) -> list[str]:
    lines = []
    pairs = pair_elements(before, after)
    for name in sorted(set(after) - {n for _, n in pairs}):
        lines.append(f"{new_cls}.{name}{_typed(after[name])} added")
    for name in sorted(set(before) - {o for o, _ in pairs}):
        lines.append(f"{old_cls}.{name} removed")
    for old, new in pairs:
        if old != new:
            lines.append(f"{old_cls}.{old} renamed to {new_cls}.{new}")
        b, a = before[old], after[new]
        if b.get("type") and a.get("type") and b["type"] != a["type"]:
            lines.append(f"{new_cls}.{new} type {b['type']} -> {a['type']}")
        if "optional" in b and "optional" in a and b["optional"] != a["optional"]:
            change = "required -> optional" if a["optional"] else "optional -> required"
            lines.append(f"{new_cls}.{new} {change}")
    return lines


def describe_model_change(old: dict, new: dict) -> list[str]:
    """One line per change between two ``model_summary`` dicts.

    Association ends read ``Owner.end[min..max]``, the owner being the class
    the end belongs to. Sections the old summary lacks are not compared.
    """
    lines = []
    old_c, new_c = old.get("classes") or {}, new.get("classes") or {}
    pairs = pair_elements(old_c, new_c)
    for name in sorted(set(new_c) - {n for _, n in pairs}):
        lines.append(f"class {name} added")
    for name in sorted(set(old_c) - {o for o, _ in pairs}):
        lines.append(f"class {name} removed")
    for o, n in pairs:
        if o != n:
            lines.append(f"class {o} renamed to {n}")
        lines.extend(_attribute_changes(o, _attributes(old_c[o]), n, _attributes(new_c[n])))

    if isinstance(old.get("enumerations"), dict):
        old_e, new_e = old["enumerations"], new.get("enumerations") or {}
        pairs = pair_elements(old_e, new_e)
        for name in sorted(set(new_e) - {n for _, n in pairs}):
            lines.append(f"enumeration {name} added: {', '.join(new_e[name].get('literals', []))}")
        for name in sorted(set(old_e) - {o for o, _ in pairs}):
            lines.append(f"enumeration {name} removed")
        for o, n in pairs:
            if o != n:
                lines.append(f"enumeration {o} renamed to {n}")
            before, after = old_e[o].get("literals", []), new_e[n].get("literals", [])
            if before != after:
                lines.append(f"enumeration {n} literals {', '.join(before)} -> {', '.join(after)}")

    if isinstance(old.get("associations"), dict):
        old_a, new_a = old["associations"], new.get("associations") or {}
        pairs = pair_elements(old_a, new_a)
        for name in sorted(set(new_a) - {n for _, n in pairs}):
            lines.append(f"association {name}: none -> {new_a[name].get('ends')}")
        for name in sorted(set(old_a) - {o for o, _ in pairs}):
            lines.append(f"association {name}: {old_a[name].get('ends')} -> removed")
        for o, n in pairs:
            if o != n:
                lines.append(f"association {o} renamed to {n}")
            if old_a[o].get("ends") != new_a[n].get("ends"):
                lines.append(f"association {n}: {old_a[o].get('ends')} -> {new_a[n].get('ends')}")
    return lines


@dataclass(frozen=True)
class ModelChange:
    verdict: str          # small | large | unrelated
    basis: str            # "attributes", or "classes" for an older recipe
    gone: float           # share of the base model no longer present
    overlap: float        # share of the base model's names still in use
    removed: tuple[str, ...]  # "Class" / "Class.attribute", for display
    added: tuple[str, ...]
    changes: tuple[str, ...] = ()  # describe_model_change; empty for an older recipe
    # Classes ("Room") and attributes ("Room.floor") paired across a name change.
    renamed: tuple[tuple[str, str], ...] = ()
    # The two summaries, for the post-run conformance check. Not serialised.
    base: dict = field(default_factory=dict, repr=False, compare=False)
    new: dict = field(default_factory=dict, repr=False, compare=False)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict, "basis": self.basis,
            "gone": round(self.gone, 3), "overlap": round(self.overlap, 3),
            "removed": list(self.removed), "added": list(self.added),
            "changes": list(self.changes),
            "renamed": [{"from": old, "to": new} for old, new in self.renamed],
        }


def _names(summary: dict) -> tuple[list[str], dict[str, list[str]] | None]:
    """Class names and ``{class: [attribute names]}`` (None when unknown)."""
    classes = summary.get("classes")
    if isinstance(classes, dict):
        return list(classes), {c: list(_attributes(e)) for c, e in classes.items()}
    attributes = summary.get("attributes")
    return list(classes), (attributes if isinstance(attributes, dict) else None)


def _aligned_to(base: dict, new: dict) -> tuple[list[str], dict[str, list[str]]]:
    """``new``'s names, with every element paired to a base element renamed to its base name."""
    base_c, new_c = base["classes"], new["classes"]
    renamed = {n: o for o, n in pair_elements(base_c, new_c)}
    attributes = {}
    for name, entry in new_c.items():
        old = renamed.get(name)
        attrs = _attributes(entry)
        own = {n: o for o, n in pair_elements(_attributes(base_c[old]), attrs)} if old else {}
        attributes[old or name] = [own.get(a, a) for a in attrs]
    return list(attributes), attributes


def _renames(base: dict, new: dict) -> tuple[tuple[str, str], ...]:
    """Class and attribute renames between two ``model_summary`` dicts."""
    out = []
    base_c, new_c = base["classes"], new["classes"]
    for o, n in pair_elements(base_c, new_c):
        if o != n:
            out.append((o, n))
        for oa, na in pair_elements(_attributes(base_c[o]), _attributes(new_c[n])):
            if oa != na:
                out.append((f"{o}.{oa}", f"{n}.{na}"))
    return tuple(out)


def _qualified(classes: list[str], attributes: dict | None) -> set[str]:
    out = set(classes)
    for cls, attrs in (attributes or {}).items():
        out.update(f"{cls}.{a}" for a in attrs)
    return out


def compare_models(base: dict, new: dict) -> ModelChange | None:
    """Compare a recorded base model with ``new`` (a ``model_summary``).

    ``base`` is a ``model_summary``, or an older recipe's ``{"classes": [names],
    "attributes": {...}?}``; ``None`` when it has no classes or is malformed.
    Measured case-insensitively, with elements paired by id first. A class
    whose name alone changed (same non-empty attributes) counts as kept, and
    attributes count without their class, so a class rename costs one element.
    """
    classes = base.get("classes") if isinstance(base, dict) else None
    if not isinstance(classes, (list, dict)) or not classes or not is_summary(new):
        return None
    full = is_summary(base)
    old_c, old_a = _names(base)
    with_attributes = old_a is not None
    new_c, new_a = _aligned_to(base, new) if full else _names(new)
    added_q = _qualified(*_names(new)) if with_attributes else set(_names(new)[0])
    if full:
        # Display names: the new model's own, minus what pairs with the base.
        back = _aligned_to(new, base)
        old_q, cur_q = _qualified(old_c, old_a), _qualified(new_c, new_a)
        removed, added = old_q - cur_q, added_q - _qualified(*back)
    else:
        old_q = _qualified(old_c, old_a if with_attributes else None)
        removed, added = old_q - added_q, added_q - old_q

    lo_c = [str(c).lower() for c in old_c]
    lo_a = {str(c).lower(): [str(a).lower() for a in v] for c, v in (old_a or {}).items()}
    nw_c = [str(c).lower() for c in new_c]
    nw_a = ({str(c).lower(): [str(a).lower() for a in v] for c, v in new_a.items()}
            if with_attributes else {})
    by_attrs = {frozenset(lo_a[c]): c for c in set(lo_c) - set(nw_c) if lo_a.get(c)}
    renamed = {c: by_attrs[frozenset(nw_a.get(c, ()))] for c in set(nw_c) - set(lo_c)
               if frozenset(nw_a.get(c, ())) in by_attrs}
    nw_c = [renamed.get(c, c) for c in nw_c]

    def elements(cls_names, attrs):
        return Counter(f"class {c}" for c in cls_names) + Counter(
            f"attr {a}" for v in attrs.values() for a in v)

    old, cur = elements(lo_c, lo_a), elements(nw_c, nw_a)
    gone = sum((old - cur).values()) / sum(old.values())
    classes_kept = len(set(lo_c) & set(nw_c)) / len(set(lo_c))
    old_names, cur_names = {e.split(" ", 1)[1] for e in old}, {e.split(" ", 1)[1] for e in cur}
    overlap = len(old_names & cur_names) / len(old_names)
    reused = len(old_names & cur_names) / len(cur_names) if cur_names else 0.0
    if (gone > UNRELATED_GONE and reused < UNRELATED_REUSED
            and (classes_kept < UNRELATED_CLASSES_KEPT or overlap < UNRELATED_OVERLAP)):
        verdict = UNRELATED
    elif gone > LARGE_GONE:
        verdict = LARGE
    else:
        verdict = SMALL
    return ModelChange(
        verdict, "attributes" if with_attributes else "classes", gone, overlap,
        tuple(sorted(removed)), tuple(sorted(added)),
        tuple(describe_model_change(base, new)) if full else (),
        _renames(base, new) if full else (),
        base if full else {}, new,
    )


def compare_with_base(workspace: str, domain_model) -> ModelChange | None:
    """Compare ``domain_model`` with the model recorded in ``workspace``'s recipe.

    ``None`` when either side is missing or unreadable: callers then keep
    today's behaviour (edit the app in place).
    """
    if domain_model is None:
        return None
    try:
        with open(os.path.join(workspace, ".besser_recipe.json"), encoding="utf-8") as fh:
            recorded = json.load(fh).get("model") or {}
        base = recorded.get("user_model") or {"classes": recorded.get("classes")}
        return compare_models(base, model_summary(domain_model))
    except (OSError, ValueError, AttributeError, TypeError):
        return None
