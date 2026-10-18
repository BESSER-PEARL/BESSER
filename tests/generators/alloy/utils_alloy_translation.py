import re
from typing import Dict, List, Optional, Set

from besser.BUML.metamodel.structural.structural import Constraint, DomainModel
from besser.generators.alloy.alloy_translator import BumlToAlloy


# ---------------------------------------------------------------------------
# Minimal parser for the signatures, facts and run command of a generated .als file
# ---------------------------------------------------------------------------

_SIG_RE = re.compile(
    r"^[ \t]*(?P<abstract>abstract\s+)?(?:(?P<mult>one|lone|some)\s+)?sig\s+"
    r"(?P<names>\w+(?:\s*,\s*\w+)*)\s*"
    r"(?:extends\s+(?P<parent>\w+)|in\s+(?P<subset>\w+(?:\s*\+\s*\w+)*))?\s*"
    r"\{(?P<body>[^{}]*)\}",
    re.MULTILINE,
)

# Not anchored to the line start: OCL facts may follow an auxiliary ``fun`` on the same line.
_FACT_RE = re.compile(r"\bfact\b\s*\w*\s*\{")

_RUN_RE = re.compile(r"^[ \t]*run\s+(?P<pred>\w+)\s+for\s+(?P<scopes>.+)$", re.MULTILINE)

# Comparison/implication operators; ``->``, ``<:`` and ``:>`` are left untouched.
_OP_RE = re.compile(r"\s*(<=>|=>|=<|<=|>=|!=|(?<![-:])>|<(?!:)|=)\s*")


def _normalize(expr: str) -> str:
    """Single spaces between tokens and exactly one space around comparison operators."""
    return " ".join(_OP_RE.sub(r" \1 ", expr).split())


def _brace_body(text: str, open_idx: int) -> str:
    """Contents of the ``{...}`` block whose opening brace is at ``open_idx``."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:i]
    raise AssertionError(f"Unbalanced braces after position {open_idx}")


class AlloyTestSpec:
    """Signatures, facts and run command of an Alloy specification.

    - ``sigs``:    normalized declarations, e.g. ``"one sig ENUM_Position_FORWARD extends Position"``
    - ``fields``:  signature name -> normalized field declarations, e.g. ``{"Team_country: one Country"}``
    - ``parents``: signature name -> the signature it extends (``None`` if it extends none)
    - ``facts``:   normalized fact bodies, e.g. ``{"all a: Fan | #(a.Fan_favoriteTeams) >= 2"}``
    - ``run_pred``:   predicate executed by the ``run`` command, e.g. ``"instance_model"``
    - ``run_scopes``: normalized scopes of the ``run`` command in order, e.g. ``["5 Team", "5 Int"]``
    """

    def __init__(self, text: str):
        self.sigs: Set[str] = set()
        self.fields: Dict[str, Set[str]] = {}
        self.parents: Dict[str, Optional[str]] = {}
        self.facts: Set[str] = set()
        self.run_pred: Optional[str] = None
        self.run_scopes: List[str] = []

        for m in _SIG_RE.finditer(text):
            fields = set()
            for decl in m.group("body").split(","):
                if decl.strip():
                    fname, ftype = decl.split(":", 1)
                    fields.add(f"{fname.strip()}: {' '.join(ftype.split())}")

            prefix = []
            if m.group("abstract"):
                prefix.append("abstract")
            if m.group("mult"):
                prefix.append(m.group("mult"))
            suffix = []
            if m.group("parent"):
                suffix += ["extends", m.group("parent")]
            if m.group("subset"):
                suffix += ["in", " + ".join(s.strip() for s in m.group("subset").split("+"))]

            for name in re.split(r"\s*,\s*", m.group("names")):
                assert name not in self.fields, f"Signature {name} declared twice"
                self.sigs.add(" ".join(prefix + ["sig", name] + suffix))
                self.fields[name] = set(fields)
                self.parents[name] = m.group("parent")

        for m in _FACT_RE.finditer(text):
            self.facts.add(_normalize(_brace_body(text, m.end() - 1)))

        runs = list(_RUN_RE.finditer(text))
        assert len(runs) <= 1, "More than one run command"
        if runs:
            self.run_pred = runs[0].group("pred")
            self.run_scopes = [" ".join(sc.split()) for sc in runs[0].group("scopes").split(",")]

    def subsigs_of(self, parent: str) -> Set[str]:
        """Names of the signatures that directly extend ``parent``."""
        return {name for name, p in self.parents.items() if p == parent}


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------

def translate(model: DomainModel, classname: str, constraint: str, scope: int) -> AlloyTestSpec:
    model.constraints = {
        Constraint(
            name="named",
            context=model.get_class_by_name(classname),
            expression=f"context {classname} inv named: {constraint}",
            language="OCL",
        )
    }
    spec = BumlToAlloy(model, scope=scope, seed=0).translate()
    return AlloyTestSpec(spec.render())
