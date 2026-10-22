"""How the model of a modify run relates to the model its base app was built from.

A modify run edits the previous app in place. Without being told what changed
in the class diagram, it keeps old and new elements side by side; for an
unrelated model it writes the new app next to the old one. This compares the
two models without any LLM call (#659):

* ``model_summary`` is what the recipe stores as ``model.user_model``: the
  model the user sent, before a modify run's model-sync added classes to it.
* ``describe_model_change`` lists the changes for the modify run: classes
  (added, removed, renamed, abstract, parents), attributes (added, removed,
  renamed, type, optional/required), enumeration literals and association
  ends. Methods and OCL constraints are not compared.
* ``compare_models`` gives a verdict (small / large / unrelated) so the UI can
  ask "update the existing app / start fresh" before a run.

Every comparison pairs elements with ``pair_elements``, so the verdict, the
change list and ``renamed`` always agree. Older recipes only have the run's
class names, model-sync classes included: they give a verdict but no change
list.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from besser.spec_driven_agent.execution.workspace_fs import read_plain_text

SMALL, LARGE, UNRELATED = "small", "large", "unrelated"
# basis: what the base recipe recorded.
FULL_MODEL, CLASS_NAMES = "attributes", "classes"

# Share of the base model (classes and attributes) gone above which a change
# is large. Hotel model: add an attribute 0, change ~25% of the attributes
# 0.11, ~50% 0.32, keep the classes and change everything else 0.54-0.64.
LARGE_GONE = 0.15
# Another app: more than half of the base model is gone, the new model is
# mostly new elements (so deleting classes is not "another app"), and either
# most base classes are gone or three quarters of the base model is.
UNRELATED_GONE = 0.5
UNRELATED_REUSED = 0.5
UNRELATED_CLASSES_KEPT = 0.5
UNRELATED_GONE_WITH_CLASSES_KEPT = 0.75


def _with_id(element, entry: dict) -> dict:
    element_id = getattr(element, "id", None)
    if element_id is not None:
        entry["id"] = str(element_id)
    return entry


def model_summary(domain_model) -> dict:
    """Classes (abstract, parents, attribute types, optional/required),
    enumerations and associations of ``domain_model``, as stored in the recipe."""
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
        classes[cls.name] = _with_id(cls, {
            "abstract": bool(cls.is_abstract),
            "parents": sorted(parent.name for parent in cls.parents()),
            "attributes": attributes,
        })
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
    """A ``model_summary`` dict (older recipes hold a class-name list)."""
    return isinstance(summary, dict) and isinstance(summary.get("classes"), dict)


def _id(entry) -> str | None:
    return entry.get("id") if isinstance(entry, dict) else None


def pair_elements(old: dict, new: dict, signature=None) -> list[tuple[str, str]]:
    """``(old name, new name)`` of the elements of two ``{name: entry}`` maps
    that are the same element. Each element is paired at most once:

    1. the same ``id``, when both entries carry one;
    2. the same name, then the same name ignoring case, unless both entries
       carry ids (then the ids decide);
    3. only with ``signature`` (classes: their attribute names): the one
       unpaired element on each side with the same non-empty signature, i.e.
       an element whose name alone changed.
    """
    pairs: list[tuple[str, str]] = []
    free_old, free_new = dict(old), dict(new)

    def take(o: str, n: str) -> None:
        pairs.append((o, n))
        del free_old[o], free_new[n]

    old_by_id = {}
    for name, entry in old.items():
        old_by_id.setdefault(_id(entry), name)
    for name, entry in new.items():
        match = old_by_id.get(_id(entry)) if _id(entry) else None
        if match in free_old:
            take(match, name)

    def name_match_allowed(o: str, n: str) -> bool:
        return not (_id(old[o]) and _id(new[n]))

    for name in list(free_new):
        if name in free_old and name_match_allowed(name, name):
            take(name, name)
    for name in list(free_new):
        match = next((o for o in free_old if o.lower() == name.lower()), None)
        if match is not None and name_match_allowed(match, name):
            take(match, name)

    if signature is not None:
        def by_signature(entries: dict) -> dict:
            groups: dict = {}
            for name, entry in entries.items():
                groups.setdefault(signature(entry), []).append(name)
            return groups

        old_groups, new_groups = by_signature(free_old), by_signature(free_new)
        for key, new_names in new_groups.items():
            old_names = old_groups.get(key, [])
            if key and len(old_names) == 1 and len(new_names) == 1 \
                    and name_match_allowed(old_names[0], new_names[0]):
                take(old_names[0], new_names[0])
    return sorted(pairs)


def _attributes(entry) -> dict:
    attributes = entry.get("attributes") if isinstance(entry, dict) else None
    return attributes if isinstance(attributes, dict) else {}


def _attribute_names(entry) -> frozenset:
    return frozenset(name.lower() for name in _attributes(entry))


def match_classes(old_classes: dict, new_classes: dict) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """``(old class, new class, attribute pairs)`` for every class in both models."""
    return [
        (o, n, pair_elements(_attributes(old_classes[o]), _attributes(new_classes[n])))
        for o, n in pair_elements(old_classes, new_classes, signature=_attribute_names)
    ]


def _typed(entry: dict) -> str:
    kind = entry.get("type")
    if not kind:
        return ""
    return f": {kind}{' (optional)' if entry.get('optional') else ''}"


def _attribute_changes(old_cls: str, before: dict, new_cls: str, after: dict,
                       pairs: list[tuple[str, str]]) -> list[str]:
    lines = []
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


def _class_changes(old_cls: str, before: dict, new_cls: str, after: dict,
                   new_name_of: dict) -> list[str]:
    lines = []
    if old_cls != new_cls:
        lines.append(f"class {old_cls} renamed to {new_cls}")
    if "abstract" in before and "abstract" in after and before["abstract"] != after["abstract"]:
        lines.append(f"class {new_cls} is {'now' if after['abstract'] else 'no longer'} abstract")
    if "parents" in before and "parents" in after:
        was = {new_name_of.get(p, p) for p in before["parents"]}
        now = set(after["parents"])
        lines.extend(f"class {new_cls}: parent {p} added" for p in sorted(now - was))
        lines.extend(f"class {new_cls}: parent {p} removed" for p in sorted(was - now))
    return lines


def describe_model_change(old: dict, new: dict) -> list[str]:
    """One line per change between two ``model_summary`` dicts.

    Association ends read ``Owner.end[min..max]``, the owner being the class
    the end belongs to. Sections the old summary lacks are not compared.
    """
    lines = []
    old_c, new_c = old.get("classes") or {}, new.get("classes") or {}
    matched = match_classes(old_c, new_c)
    new_name_of = {o: n for o, n, _ in matched}
    for name in sorted(set(new_c) - {n for _, n, _ in matched}):
        lines.append(f"class {name} added")
    for name in sorted(set(old_c) - set(new_name_of)):
        lines.append(f"class {name} removed")
    for o, n, attribute_pairs in matched:
        lines.extend(_class_changes(o, old_c[o], n, new_c[n], new_name_of))
        lines.extend(_attribute_changes(
            o, _attributes(old_c[o]), n, _attributes(new_c[n]), attribute_pairs))

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
    basis: str            # FULL_MODEL, or CLASS_NAMES for an older recipe
    gone: float           # share of the base model's classes and attributes not kept
    removed: tuple[str, ...]  # "Class" / "Class.attribute"; renamed elements excluded
    added: tuple[str, ...]
    # Classes ("Room") and attributes ("Room.floor") kept under a new name.
    renamed: tuple[tuple[str, str], ...] = ()
    changes: tuple[str, ...] = ()  # describe_model_change; empty for an older recipe
    # The two summaries, for the post-run conformance check. Not serialised.
    base: dict = field(default_factory=dict, repr=False, compare=False)
    new: dict = field(default_factory=dict, repr=False, compare=False)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict, "basis": self.basis, "gone": round(self.gone, 3),
            "removed": list(self.removed), "added": list(self.added),
            "renamed": [{"from": old, "to": new} for old, new in self.renamed],
            "changes": list(self.changes),
        }


def _elements(classes: dict) -> int:
    return len(classes) + sum(len(_attributes(entry)) for entry in classes.values())


def _qualified(classes: dict, names) -> list[str]:
    """A class and its attributes, as ``Class`` / ``Class.attribute``."""
    out = []
    for name in names:
        out.append(name)
        out.extend(f"{name}.{a}" for a in _attributes(classes[name]))
    return out


def _verdict(gone: float, reused: float, classes_kept: float) -> str:
    if gone > UNRELATED_GONE and reused < UNRELATED_REUSED and (
            classes_kept < UNRELATED_CLASSES_KEPT or gone > UNRELATED_GONE_WITH_CLASSES_KEPT):
        return UNRELATED
    return LARGE if gone > LARGE_GONE else SMALL


def compare_models(base: dict, new: dict) -> ModelChange | None:
    """Compare a recorded base model with ``new`` (a ``model_summary``).

    ``base`` is a ``model_summary``, or an older recipe's ``{"classes": [names]}``
    (then only class names are compared). ``None`` when the base has no classes
    or is malformed.
    """
    if not is_summary(new) or not isinstance(base, dict):
        return None
    if is_summary(base) and base["classes"]:
        basis = FULL_MODEL
    elif isinstance(base.get("classes"), list) and base["classes"]:
        basis = CLASS_NAMES
        base = {"classes": {str(name): {} for name in base["classes"]}}
        new = {"classes": {name: {} for name in new["classes"]}}
    else:
        return None
    old_c, new_c = base["classes"], new["classes"]
    matched = match_classes(old_c, new_c)

    removed = _qualified(old_c, sorted(set(old_c) - {o for o, _, _ in matched}))
    added = _qualified(new_c, sorted(set(new_c) - {n for _, n, _ in matched}))
    renamed = [(o, n) for o, n, _ in matched if o != n]
    kept = len(matched)
    for o, n, attribute_pairs in matched:
        kept += len(attribute_pairs)
        before, after = _attributes(old_c[o]), _attributes(new_c[n])
        removed += [f"{o}.{a}" for a in sorted(set(before) - {oa for oa, _ in attribute_pairs})]
        added += [f"{n}.{a}" for a in sorted(set(after) - {na for _, na in attribute_pairs})]
        renamed += [(f"{o}.{oa}", f"{n}.{na}") for oa, na in attribute_pairs if oa != na]

    gone = 1 - kept / _elements(old_c)
    new_elements = _elements(new_c)
    reused = kept / new_elements if new_elements else 1.0
    classes_kept = len(matched) / len(old_c)
    full = basis == FULL_MODEL
    return ModelChange(
        _verdict(gone, reused, classes_kept), basis, gone,
        tuple(sorted(removed)), tuple(sorted(added)), tuple(renamed),
        tuple(describe_model_change(base, new)) if full else (),
        base if full else {}, new,
    )


def compare_with_base(workspace: str, domain_model) -> ModelChange | None:
    """Compare ``domain_model`` with the model recorded in ``workspace``'s recipe.

    ``None`` when either side is missing or unreadable: callers then keep
    today's behaviour (edit the app in place).
    """
    if domain_model is None:
        return None
    text = read_plain_text(os.path.join(workspace, ".besser_recipe.json"), root=workspace)
    if text is None:
        return None
    try:
        recorded = json.loads(text).get("model") or {}
        base = recorded.get("user_model") or {"classes": recorded.get("classes")}
        return compare_models(base, model_summary(domain_model))
    except (ValueError, AttributeError, TypeError):
        return None
