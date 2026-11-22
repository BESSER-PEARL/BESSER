"""Identifier sanitisation and auxiliary-class naming conventions.

The aux-class names follow the tokens used in ``conversion_kg2uml.tex`` exactly:
``_some_p_D``, ``_min_n_p_D``, ``_all_p_D``, ``_not_C``,
``_C1_..._Cn_Union``/``_Intersection``, and ``_hasValue_p_a`` (O08).
"""
from __future__ import annotations

from besser.BUML.metamodel.kg.iri import local_name as _local_name
from besser.utilities.identifiers import sanitize_identifier


# ``local_name`` lives in the metamodel package so every layer can share it.
local_name = _local_name


def sanitize(name: str) -> str:
    """Turn an arbitrary label into a valid UML/PlantUML identifier.

    ASCII only: the names reach OCL, whose identifier token is ASCII (see
    :mod:`besser.utilities.identifiers`). Underscores are kept as they are, because the
    auxiliary class names below start with one.
    """
    return sanitize_identifier(name, ascii_only=True)


def namespace_hint(iri: str) -> str:
    """A short, human-ish token identifying an IRI's namespace.

    Used to disambiguate two IRIs that share a local name
    """
    s = str(iri)
    # strip the local part
    for sep in ("#", "/"):
        if sep in s:
            s = s.rsplit(sep, 1)[0]
    # take the last meaningful path/host token
    token = local_name(s)
    token = token.replace(".owl", "").replace(".rdf", "")
    return sanitize(token) or "ns"


# --- auxiliary-class names ------------------------------------------------

def union_name(operands: list[str]) -> str:
    """Name of the auxiliary class for a union of ``operands``."""
    return "_" + "_".join(operands) + "_Union"


def intersection_name(operands: list[str]) -> str:
    """Name of the auxiliary class for an intersection of ``operands``."""
    return "_" + "_".join(operands) + "_Intersection"


def one_of_name(individuals: list[str]) -> str:
    """Name of the auxiliary class for an enumeration of ``individuals``."""
    return "_" + "_".join(individuals) + "_OneOf"


def domain_name(prop: str) -> str:
    """Name of the abstract class the lowering declares ``prop``'s end on for several classes.

    Not one of the paper's tokens: plain UML lets a class inherit two ends of
    one name, BUML does not, so ``to_buml._hoist_shared_ends`` declares the end
    once, on a parent of every class that declared it.
    """
    return f"_{prop}_Domain"


def inverse_name(prop: str) -> str:
    """Name of the opposite end of ``prop``'s association, which ``[ owl:inverseOf prop ]`` navigates (D16).

    Not one of the paper's tokens: D16 maps an inverse to the opposite end but
    names none, and a restriction on the inverse needs a name to navigate by.
    Used only when the ontology names no inverse of ``prop`` itself.
    """
    return f"inverse_{prop}"


def not_name(cls: str) -> str:
    """Name of the auxiliary class for the complement of ``cls``."""
    return f"_not_{cls}"


def restriction_name(kind: str, prop: str, filler: str, n: int | None = None) -> str:
    """Aux name for a property restriction.

    ``kind`` is one of: some, all, min, max, exact, hasValue, hasSelf.
    """
    if kind == "hasSelf":
        return f"_hasSelf_{prop}"
    if n is not None:
        return f"_{kind}_{n}_{prop}_{filler}"
    return f"_{kind}_{prop}_{filler}"


def regex_literal(text: str) -> str:
    """OCL single-quoted string literal for a regex (``sh:pattern``, ``xsd:pattern``).

    Backslashes are left literal (not doubled) so ``^\\d{9}$`` renders as
    ``'^\\d{9}$'`` -- regex fidelity matters more than OCL string-escape purity here.
    """
    return "'" + text.replace("'", "\\'").replace("\n", " ") + "'"
