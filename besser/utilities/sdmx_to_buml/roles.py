"""Detect what each SDMX dimension *is*, so that the model documents how to read its data.

Every dimension gets exactly one role:

* ``frequency``  the FREQ dimension (annual, monthly, ...). One value at a time.
* ``unit``       anything that must never be summed across its codes: units of measure,
                 indicators and measures, adjustments, and dimensions with a single code in
                 the data (constants such as ``_Z``). One value at a time.
* ``geo``        countries and areas: the codes are mostly ISO 3166 alpha-2 codes, or the
                 concept is an area, country, partner or reporter.
* ``hierarchy``  a nomenclature with levels: SDMX parent references, a .Stat Suite
                 ``HIER_CONTEXT`` hierarchical codelist, or numeric codes nesting by prefix
                 (e.g. HS2 > HS4 > HS6 > CN8, NACE digits).
* ``breakdown``  every other categorical dimension (sex, age, sector...).

Each dimension also gets a ``total_code`` (the code holding the sum of the others: ``_T``,
``TOTAL``, ``WORLD``... or a label such as "Total") and a ``default_code``: the code a query
should use when the dimension is not being analysed, so that values are never double counted.
"""

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from besser.utilities.sdmx_to_buml.structure import Codelist, Component, DataStructure

FREQUENCY, GEO, HIERARCHY, UNIT, BREAKDOWN = "frequency", "geo", "hierarchy", "unit", "breakdown"
ROLES = (FREQUENCY, GEO, HIERARCHY, UNIT, BREAKDOWN)

_FREQ_IDS = {"FREQ", "FREQUENCY"}
# Concepts that are measures or units: summing across their codes is meaningless.
_UNIT_IDS = re.compile(
    r"^(UNIT|UNIT_MEASURE|UNIT_MEAS|UNIT_MULT|UNITS|CURRENCY|PRICES?|INDICATORS?|INDIC\w*|MEASURE|MEASURES|"
    r"NA_ITEM|STO|S_ADJ|ADJUSTMENT|FLOW|FLOWS|STK_FLOW|TRANSFORMATION|TRANSFORM|BASE_PER|BASE_PERIOD|DECIMALS)$")
_UNIT_WORDS = ("unit", "measure", "indicator", "currency", "adjustment", "flow", "base period", "price")
_GEO_IDS = re.compile(r"(^|_)(GEO|AREA|REF_AREA|COUNTRY|COUNTERPART_AREA|PARTNER|REPORTER|REGION)($|_)")
_GEO_WORDS = ("country", "area", "partner", "reporter", "geopolitical", "region", "geo")
_TOTAL_CODES = ("TOTAL", "_T", "WORLD", "T", "TOT", "ALL", "W0", "W1", "_X")
_TOTAL_LABELS = re.compile(r"^(total|world|all countries( of the world)?|all|all items|all activities|"
                           r"total[ ,].*|world total)$", re.I)
_ISO2 = re.compile(r"^[A-Z]{2}$")
# Two-letter codes that are not countries: aggregates and "not specified" pseudo-areas.
_NOT_COUNTRY = re.compile(r"^(Q[P-Z]|EU|EA|EZ|XX|ZZ)$")
# Preferred default for unit-like dimensions without a total: a level rather than a ratio.
_HEADLINE = [(re.compile(p, re.I), w) for p, w in (
    (r"^(EUR|MIO_EUR|CP_MEUR|CP_MNAC|NR|THS|THS_PER|PER|USD|XDC)$", 8),
    (r"MEUR|MIO_EUR|euro|dollar", 2), (r"value", 2), (r"gross domestic product", 4), (r"total", 2),
    (r"^PC|_PC|^PCH|percent|percentage|^I\d|index|rate|_HAB|per capita|^RT", -6),
)]


@dataclass
class DimensionRole:
    """The role of one dimension, with its total and default code."""
    id: str
    role: str
    label: str
    total_code: Optional[str] = None
    default_code: Optional[str] = None
    hierarchy: Optional[str] = None  # "sdmx" | "prefix" | "hcl"
    used_codes: list[str] = field(default_factory=list)
    reason: str = ""


def infer_prefix_parents(codes: Iterable[str]) -> dict[str, Optional[str]]:
    """Parent of each code: the longest other code that is a strict prefix of it.

    Works for nested numeric nomenclatures (``84`` > ``8471`` > ``847130`` > ``84713000``).
    Numeric codes without a prefix parent are attached to ``TOTAL`` when that code exists.
    """
    codes = list(codes)
    present = set(codes)
    parents: dict[str, Optional[str]] = {}
    for code in codes:
        parent = None
        if code.isdigit():  # only numeric nomenclature codes nest by prefix
            for n in range(len(code) - 1, 0, -1):
                if code[:n] in present:
                    parent = code[:n]
                    break
        if parent is None and code.isdigit() and "TOTAL" in present:
            parent = "TOTAL"
        parents[code] = parent
    return parents


def looks_prefix_hierarchical(codes: Iterable[str]) -> bool:
    """True for numeric codes where most have a strictly shorter code as prefix (a nomenclature)."""
    codes = list(codes)
    digit_codes = [c for c in codes if c.isdigit()]
    if len(digit_codes) < 50:
        return False
    # A run of plain integers (ranks 1..N, years) nests by prefix too, but is not a nomenclature.
    if not any(c.startswith("0") for c in digit_codes):
        ints = {int(c) for c in digit_codes}
        if len(ints) > 0.9 * (max(ints) - min(ints) + 1):
            return False
    parents = infer_prefix_parents(codes)
    with_parent = sum(1 for c in digit_codes if parents.get(c) not in (None, "TOTAL"))
    return with_parent / len(digit_codes) > 0.5


def _label_of(codelist: Optional[Codelist], code: str) -> str:
    if codelist and code in codelist.codes:
        return codelist.codes[code].name or code
    return code


def _iso_share(codes: list[str]) -> float:
    if not codes:
        return 0.0
    return sum(1 for c in codes if _ISO2.match(c)) / len(codes)


def find_total(codelist: Optional[Codelist], used: list[str], parents: Optional[dict] = None) -> Optional[str]:
    """The code holding the total of a dimension, by SDMX conventions, labels, or the hierarchy root."""
    present = set(used)
    for code in _TOTAL_CODES:
        if code in present:
            return code
    for code in used:
        if _TOTAL_LABELS.match(_label_of(codelist, code).strip()):
            return code
    if parents:
        roots = [c for c in used if parents.get(c) is None]
        if len(roots) == 1 and len(used) > 1:
            return roots[0]
    return None


def _headline(codelist: Optional[Codelist], used: list[str]) -> Optional[str]:
    if not used:
        return None
    best, best_score = used[0], 0
    for code in used:
        text = f"{code} {_label_of(codelist, code)}"
        score = sum(w for rx, w in _HEADLINE if rx.search(code) or rx.search(text))
        if score > best_score:
            best, best_score = code, score
    return best


def _hierarchy_of(dsd: DataStructure, dim: Component, codelist: Optional[Codelist],
                  forced_prefix: Optional[set[str]]):
    if dim.id in dsd.hierarchies:
        return "hcl", dsd.hierarchies[dim.id]
    if codelist is not None and codelist.has_parents:
        return "sdmx", {c.id: c.parent for c in codelist.codes.values()}
    if codelist is not None and (
            (forced_prefix is not None and dim.id in forced_prefix)
            or (forced_prefix is None and looks_prefix_hierarchical(codelist.codes))):
        return "prefix", infer_prefix_parents(codelist.codes)
    return None, None


def detect_roles(dsd: DataStructure, prefix_hierarchy: Optional[Iterable[str]] = None) -> dict[str, DimensionRole]:
    """Assign a role, a total and a default code to every dimension of ``dsd`` (deterministic).

    Args:
        dsd: the parsed data structure.
        prefix_hierarchy: dimension ids whose codes nest by prefix; ``None`` detects them.
    """
    forced_prefix = set(prefix_hierarchy) if prefix_hierarchy is not None else None
    out: dict[str, DimensionRole] = {}
    for dim in dsd.dimensions:
        codelist = dsd.codelist_for(dim)
        used = dsd.used_codes(dim)
        label = dim.name or dim.id
        key = (dim.concept or dim.id or "").upper()
        words = f"{dim.id} {dim.concept or ''} {label}".lower()
        hierarchy, parents = _hierarchy_of(dsd, dim, codelist, forced_prefix)
        total = find_total(codelist, used, parents)

        if dim.id.upper() in _FREQ_IDS or key in _FREQ_IDS:
            role, reason = FREQUENCY, "frequency concept"
        elif len(used) == 1:
            role, reason = UNIT, "a single code in the data (constant)"
        elif _UNIT_IDS.match(dim.id.upper()) or _UNIT_IDS.match(key) or (
                any(w in label.lower() for w in _UNIT_WORDS) and not hierarchy and len(used) <= 60):
            role, reason = UNIT, "unit or measure concept: never summed"
        elif (_GEO_IDS.search(dim.id.upper()) or any(w in words for w in _GEO_WORDS)
              or (_iso_share(used) >= 0.6 and len(used) >= 10)):
            role, reason = GEO, f"area concept, {_iso_share(used):.0%} ISO alpha-2 codes"
        elif hierarchy:
            role, reason = HIERARCHY, f"{hierarchy} hierarchy"
        else:
            role, reason = BREAKDOWN, "categorical"

        if role == FREQUENCY:
            default = "A" if "A" in used else (used[0] if used else None)
        elif role == UNIT:
            default = used[0] if len(used) == 1 else (total or _headline(codelist, used))
        elif total:
            default = total
        elif role == GEO:
            # No world total: fall back to the first aggregate (EU27, euro area...) if any.
            default = next((c for c in used if not _ISO2.match(c) or _NOT_COUNTRY.match(c)), None)
        else:
            default = None
        out[dim.id] = DimensionRole(dim.id, role, label, total, default, hierarchy, used, reason)
    return out
