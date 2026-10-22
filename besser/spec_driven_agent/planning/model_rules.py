"""Model rules that the user's own model change broke (#659).

When the user removes an attribute or class that an OCL rule uses, the
converter rejects the rule. Recovering it "in favour of the original
specification" made the agent put the removed element back. A rejected rule
is ``dropped_by_model_change`` only when both hold:

1. the same rule text converted in the previous run (the recipe records a
   fingerprint of every converted rule), or it was already dropped then for
   the same missing element;
2. the element it now misses existed in the previous model and the model
   change removed it. A renamed element is not a removal: the rule is kept
   and recovered against the new name (``renamed_in_model``).

Anything else (a new or edited rule, a typo, an element that never existed,
no recorded previous model) keeps the normal recovery path.
"""

from __future__ import annotations

import hashlib
import logging
import re

from besser.spec_driven_agent.planning.model_change import pair_elements

logger = logging.getLogger(__name__)

DROPPED = "dropped_by_model_change"
RENAMED = "renamed_in_model"

# The converter writes a generated name into an unnamed invariant it accepts.
_INVARIANT_NAME = re.compile(r"(\binv)\s+\w+\s*:", re.IGNORECASE)
# "Owner.end[0..1]" in a model_summary association.
_END = re.compile(r"(\w+)\.(\w+)\[")


def is_dropped_by_model_change(issue) -> bool:
    return isinstance(issue, dict) and bool(issue.get(DROPPED))


def rule_fingerprint(expression: str) -> str:
    """Identity of a rule's text, ignoring whitespace and the invariant name."""
    text = _INVARIANT_NAME.sub(r"\1:", " ".join(str(expression or "").split()))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def converted_rule_fingerprints(domain_model) -> list[str]:
    """Fingerprints of every OCL rule the model holds: invariants and method contracts."""
    rules = list(domain_model.constraints)
    for cls in domain_model.get_classes():
        for method in cls.methods:
            rules.extend(method.pre)
            rules.extend(method.post)
    return sorted({rule_fingerprint(r.expression) for r in rules if isinstance(r.expression, str)})


def _ends(summary: dict) -> dict[str, str]:
    """``{"Owner.end": association name}`` in a ``model_summary``."""
    return {f"{owner}.{end}": name
            for name, entry in (summary.get("associations") or {}).items()
            for owner, end in _END.findall(entry.get("ends") or "")}


def _change_for(missing: str, change) -> tuple[str, str] | None:
    """``("removed" | "renamed", old name)``, or None when the model change does not explain it."""
    renamed, removed = dict(change.renamed), set(change.removed)
    if "." not in missing:
        if missing in renamed:
            return "renamed", missing
        return ("removed", missing) if missing in removed else None
    cls, prop = missing.split(".", 1)
    old_cls = next((o for o, n in change.renamed if n == cls and "." not in o), cls)
    # The rule may have reached the attribute through a parent class.
    candidates = [f"{old_cls}.{prop}"] + sorted(e for e in removed | set(renamed) if e.endswith(f".{prop}"))
    for old in candidates:
        if old in renamed:
            return "renamed", old
        if old in removed:
            return "removed", old
    end = f"{old_cls}.{prop}"
    before, after = _ends(change.base), _ends(change.new)
    if end not in before or end in after:
        return None
    kept = {o for o, _ in pair_elements(change.base.get("associations") or {},
                                         change.new.get("associations") or {})}
    # The association survived under another end name: a rename, not a removal.
    return None if before[end] in kept else ("removed", end)


def _expression(issue: dict) -> str:
    return issue.get("expression") or issue.get("original_text") or ""


def mark_rules_broken_by_model_change(issues, change, *, converted_before: set, dropped_before: list) -> None:
    """Mark rejected rules ``change`` (a full ``ModelChange``) removed or renamed.

    ``dropped_before`` holds the previous run's dropped issues; see the module
    docstring for the rule.
    """
    renames = dict(change.renamed)
    still_dropped = {(rule_fingerprint(_expression(i)), i.get("missing_element")): i[DROPPED]
                     for i in dropped_before or []}
    for issue in issues or []:
        missing = issue.get("missing_element") if isinstance(issue, dict) else None
        if not missing:
            continue
        fingerprint = rule_fingerprint(_expression(issue))
        if (fingerprint, missing) in still_dropped:
            issue[DROPPED] = list(still_dropped[fingerprint, missing])
            continue
        if fingerprint not in converted_before:
            continue
        found = _change_for(missing, change)
        if found is None:
            continue
        kind, old = found
        if kind == "renamed":
            issue[RENAMED] = {"from": old, "to": renames[old]}
        else:
            issue[DROPPED] = [old]
        logger.info("modify: rule %s: %s %s by the model change", issue.get("name"), old, kind)
