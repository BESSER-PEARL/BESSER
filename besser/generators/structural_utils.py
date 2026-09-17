from __future__ import annotations

from typing import Dict, List

from besser.BUML.metamodel.structural import DomainModel


def _sorted_association_ends(association) -> list:
    """Return association ends in a deterministic order."""
    return sorted(association.ends, key=lambda end: (end.type.name, end.name or ""))


def get_foreign_keys(model: DomainModel) -> Dict[str, List[str]]:
    """
    Return a mapping of association name -> [class_name_with_fk, fk_property_name].
    """
    fkeys: Dict[str, List[str]] = {}

    for association in model.associations:
        ends = _sorted_association_ends(association)
        if len(ends) != 2:
            continue

        end0, end1 = ends[0], ends[1]

        # One-to-one
        if end0.multiplicity.max == 1 and end1.multiplicity.max == 1:
            if end0.multiplicity.min > 0 and end1.multiplicity.min == 0:
                fkeys[association.name] = [end1.type.name, end0.name]
            elif end1.multiplicity.min > 0 and end0.multiplicity.min == 0:
                fkeys[association.name] = [end0.type.name, end1.name]
            else:
                fkeys[association.name] = [end0.type.name, end1.name]

        # Many-to-one
        elif end0.multiplicity.max > 1 and end1.multiplicity.max <= 1:
            fkeys[association.name] = [end0.type.name, end1.name]

        elif end0.multiplicity.max <= 1 and end1.multiplicity.max > 1:
            fkeys[association.name] = [end1.type.name, end0.name]

    return fkeys


# Model type name -> python type used for primary keys and the foreign keys
# that reference them. Anything not listed keeps the historical integer
# surrogate. Shared by the SQLAlchemy and backend generators so a ForeignKey
# annotation and its path parameter can never disagree about the PK's type.
_PK_PY_TYPES = {"str": "str", "string": "str", "int": "int", "integer": "int", "float": "float"}


def get_pk_py_types(model: DomainModel) -> Dict[str, str]:
    """Class name -> python type of its primary key (default 'int').

    The PK selection mirrors the SQLAlchemy generator: the ``is_id``
    attribute, else one literally named ``id``. Classes with neither get the
    integer surrogate and are simply absent from the mapping (callers use
    ``.get(name, 'int')``).
    """
    def _own_pk(cls):
        """The PK declared on this class itself, if any."""
        return (next((a for a in cls.attributes if a.is_id), None)
                or next((a for a in cls.attributes if a.name == "id"), None))

    def _resolve(cls, seen):
        """The PK type, following inheritance.

        A subclass in joined-table inheritance has no id attribute of its
        own -- its PK IS the parent's, emitted as a ForeignKey to it. Looking
        only at the class's own attributes therefore missed every subclass,
        which then fell back to the 'int' default while the parent carried a
        string uuid. Every FK pointing at Guest or Employee came out
        Mapped_[int] against a String PK: the schema is inconsistent, the API
        rejects the real id with 422, and nothing catches it because SQLite
        creates the tables anyway.
        """
        if cls.name in seen:          # defensive: a cycle in parents()
            return None
        seen.add(cls.name)
        own = _own_pk(cls)
        if own is not None:
            type_name = (getattr(own.type, "name", "") or "").lower()
            return _PK_PY_TYPES.get(type_name, "int")
        for parent in cls.parents():
            inherited = _resolve(parent, seen)
            if inherited is not None:
                return inherited
        return None

    pk_types: Dict[str, str] = {}
    for cls in model.get_classes():
        resolved = _resolve(cls, set())
        if resolved is not None:
            pk_types[cls.name] = resolved
    return pk_types


def normalize_method_code(code, method_name="method"):
    """Normalize user-written method code so it can be embedded in generated files.

    The editor's code box lets users mix tabs and spaces, which Python rejects
    with an IndentationError the moment the generated module is imported --
    taking the whole application down with it. Tabs are expanded to 4 spaces
    and whitespace-only lines are blanked; if the result still does not
    compile, a stub carrying the original code as comments is emitted instead,
    so one broken method body can never prevent the generated app from
    starting.
    """
    if not code or not code.strip():
        return code or ""

    lines = []
    for line in code.splitlines():
        line = line.expandtabs(4).rstrip()
        lines.append(line if line.strip() else "")
    normalized = "\n".join(lines)

    try:
        compile(normalized, f"<{method_name}>", "exec")
        return normalized
    except SyntaxError:
        pass

    commented = "\n".join(
        ("    # " + line) if line else "    #" for line in normalized.splitlines()
    )
    return (
        f"def {method_name}(self):\n"
        f"    # NOTE: the original code of '{method_name}' does not compile and was\n"
        f"    # commented out so the generated application can still start.\n"
        f"    # Fix the method body in the editor and regenerate.\n"
        f"{commented}\n"
        f"    raise NotImplementedError(\"Method '{method_name}' has invalid code\")"
    )
