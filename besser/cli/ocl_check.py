"""``besser validate --ocl``: check the OCL constraints of the models of one file.

The check is the editor's (``check_ocl_constraint``, behind ``/validate-diagram``):
every constraint of a domain model is parsed, and the invariants are evaluated
on each object model of the same file that instantiates it. It needs the
``bocl`` evaluator (``besser[ocl]``) and the editor backend's packages, which
are optional: without them the check is skipped with a warning.

bocl evaluates an invariant by building Python source from the constraint and
the object model's values and running ``eval()`` on it, so a crafted value runs
code. Parsing is safe. Invariants are evaluated only when running the file's
code was already allowed: a ``.py`` model (it was executed to load it), or
``--exec-py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from besser.cli.loading import LoadedModel, missing_backend_dependency_message

# Every OCL message starts with this, then the checker's label: "[Book inv positive_pages]".
OCL_PREFIX = "OCL constraint"
_OCL_EXTRA = 'pip install "besser[ocl]"'
# The modules of the ``ocl`` extra (``bocl`` brings the ANTLR runtime).
_OCL_EXTRA_MODULES = frozenset({"bocl", "antlr4"})
# A message of ``check_ocl_constraint``: an optional marker ("❌", "OCL"), the label, the rest.
_CHECKER_MESSAGE = re.compile(r"^\S*?\s*(\[[^\]]*\])\s*(.*)$", re.DOTALL)


class OclUnavailable(Exception):
    """What the OCL check needs is not installed; the message says what to install."""


@dataclass
class OclOutcome:
    """The OCL findings of one file, by the ``id()`` of the ``LoadedModel`` they are about."""

    errors: dict[int, list[str]] = field(default_factory=dict)
    warnings: dict[int, list[str]] = field(default_factory=dict)
    # Constraints parsed, over all the file's domain models.
    constraints: int = 0
    # (domain model, object model) pairs the invariants were evaluated on.
    evaluations: int = 0
    # (domain model, object model) pairs left unevaluated: evaluating was not allowed.
    unevaluated: int = 0


def load_checker() -> Callable[..., dict[str, Any]]:
    """The editor's ``check_ocl_constraint``.

    Raises:
        OclUnavailable: ``bocl`` or the editor backend's packages are not installed.
    """
    try:
        from besser.utilities.web_modeling_editor.backend.services.validators.ocl_checker import (
            check_ocl_constraint,
        )
    except ModuleNotFoundError as exc:
        module = (exc.name or "").split(".")[0]
        if module in ("", "besser"):  # a module missing from besser itself is a packaging bug
            raise
        if module in _OCL_EXTRA_MODULES:
            raise OclUnavailable(
                f"OCL constraints not checked: the OCL evaluator is not installed (missing module: "
                f"{module!r}); install it with: {_OCL_EXTRA}"
            ) from exc
        raise OclUnavailable(f"OCL constraints not checked: {missing_backend_dependency_message(exc)}") from exc
    return check_ocl_constraint


def _message(entry: str) -> tuple[str, str]:
    """``entry``, a message of the checker, as (its label, the message under the stable prefix)."""
    entry = entry.strip()
    match = _CHECKER_MESSAGE.match(entry)
    if not match:
        return entry, f"{OCL_PREFIX}: {entry}"
    label, rest = match.groups()
    return label, f"{OCL_PREFIX} {label}: {rest}"


def _instantiates(object_model: Any, domain_model: Any) -> bool:
    """Whether some object of ``object_model`` is an instance of a class of ``domain_model``."""
    types = domain_model.types
    return any(getattr(obj, "classifier", None) in types for obj in object_model.objects)


def check_ocl(models: list[LoadedModel], checker: Callable[..., dict[str, Any]], evaluate: bool) -> OclOutcome:
    """Check the OCL constraints of the domain models among ``models`` with ``checker``.

    A syntax error is reported on the domain model, and so is a constraint the
    editor JSON converter dropped because it does not parse. Only with
    ``evaluate`` are the invariants evaluated (see the module docstring). An invariant an object
    model violates is reported on the object model, unless the constraint
    does not parse (it is already reported). An object model is paired with
    the domain models whose classes it instantiates, or, when none match,
    with the file's only domain model.
    """
    outcome = OclOutcome()
    domains = [loaded for loaded in models if loaded.kind == "domain"]
    for domain in domains:
        syntax = checker(domain.model)
        constraints = len(syntax["valid_constraints"]) + len(syntax["invalid_constraints"])
        outcome.constraints += constraints
        if not syntax["success"] and not constraints:  # the checker itself failed
            outcome.errors.setdefault(id(domain), []).append(f"{OCL_PREFIX}s not checked: {syntax['message']}")
            continue
        broken = set()
        for entry in syntax["invalid_constraints"]:
            label, message = _message(entry)
            broken.add(label)
            outcome.errors.setdefault(id(domain), []).append(message)
        dropped = getattr(domain.model, "ocl_warnings", None) or []
        outcome.constraints += len(dropped)
        outcome.errors.setdefault(id(domain), []).extend(f"{OCL_PREFIX} dropped: {reason}" for reason in dropped)
        if constraints:
            for instance in _object_models_of(domain, domains, models):
                if evaluate:
                    _evaluate(domain, instance, broken, checker, outcome)
                else:
                    outcome.unevaluated += 1
    return outcome


def _object_models_of(domain: LoadedModel, domains: list[LoadedModel],
                      models: list[LoadedModel]) -> list[LoadedModel]:
    """The object models among ``models`` to evaluate the invariants of ``domain`` on."""
    paired = []
    for instance in (loaded for loaded in models if loaded.kind == "object"):
        matching = [candidate for candidate in domains if _instantiates(instance.model, candidate.model)]
        if domain in matching or (not matching and len(domains) == 1):
            paired.append(instance)
    return paired


def _evaluate(domain: LoadedModel, instance: LoadedModel, broken: set[str],
              checker: Callable[..., dict[str, Any]], outcome: OclOutcome) -> None:
    """Evaluate the invariants of ``domain`` on ``instance``; skip the ``broken`` (unparsable) ones."""
    outcome.evaluations += 1
    evaluated = checker(domain.model, instance.model)
    if not evaluated["success"] and not (evaluated["invalid_constraints"] or evaluated["valid_constraints"]):
        outcome.errors.setdefault(id(instance), []).append(f"{OCL_PREFIX}s not evaluated: {evaluated['message']}")
        return
    for entry in evaluated["invalid_constraints"]:
        label, message = _message(entry)
        if label not in broken:
            outcome.errors.setdefault(id(instance), []).append(message)
    for entry in evaluated["warnings"]:
        outcome.warnings.setdefault(id(instance), []).append(_message(entry)[1])
