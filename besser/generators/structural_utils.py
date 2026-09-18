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


def get_deferred_fk_associations(model: DomainModel) -> set:
    """Associations whose FK must be created nullable to break a table cycle.

    Two tables that each hold a NOT NULL foreign key to the other cannot be
    created, and neither row can be inserted first. SQLAlchemy says so
    outright — ``Can't sort tables; there are unresolvable cycles between
    tables booking, guest`` — and ``create_all`` raises before the app ever
    serves a request. Live 2026-09-17: a hotel model where Booking pointed
    at its contact Guest and Guest pointed back at its Booking.

    A cycle needs at least two associations, since a single association
    produces only one FK column (see ``get_foreign_keys``). So one edge of
    each cycle has to give: its FK becomes nullable, the row is inserted
    without it, and the link is set afterwards from the other side.

    Which edge gives is decided by association name, so the choice is
    stable across regenerations — a generator that picked a different edge
    each run would rewrite unrelated files on every run.

    Returns the set of association names whose FK is deferred.
    """
    fkeys = get_foreign_keys(model)
    if len(fkeys) < 2:
        return set()

    required_by_end = _required_association_ends(model)

    # Edge: the table holding the FK depends on the table it points at.
    # Only NOT NULL FKs constrain creation order, so only they can form a
    # blocking cycle — a nullable one is already deferrable.
    edges: Dict[str, List[tuple]] = {}
    for assoc_name, owner in fkeys.items():
        owner_class, fk_end_name = owner[0], owner[1]
        target = required_by_end.get((assoc_name, fk_end_name))
        if target is None:
            continue  # optional FK: no ordering constraint
        edges.setdefault(owner_class, []).append((target, assoc_name))

    deferred: set = set()
    # Iterate to a fixed point: breaking one cycle can leave another.
    while True:
        cycle = _find_cycle(edges, deferred)
        if not cycle:
            return deferred
        deferred.add(max(cycle))


def _required_association_ends(model: DomainModel) -> Dict[tuple, str]:
    """(association name, end name) -> target class, for REQUIRED ends only."""
    required: Dict[tuple, str] = {}
    for association in model.associations:
        ends = _sorted_association_ends(association)
        if len(ends) != 2:
            continue
        for end in ends:
            if end.multiplicity.min > 0:
                required[(association.name, end.name)] = end.type.name
    return required


def _find_cycle(edges: Dict[str, List[tuple]], deferred: set) -> set:
    """Return the association names on one cycle, or an empty set.

    Only the associations actually ON the cycle are returned, not the whole
    search path — deferring an edge that merely leads to a cycle would not
    break it, and would make a second FK nullable for nothing.
    """
    colour: Dict[str, int] = {}  # 0 = on the current path, 1 = finished
    path_nodes: List[str] = []
    path_assocs: List[str] = []  # edge i goes from path_nodes[i] to [i+1]

    def visit(node: str) -> set:
        colour[node] = 0
        path_nodes.append(node)
        for target, assoc_name in sorted(edges.get(node, ()), key=lambda e: e[1]):
            if assoc_name in deferred:
                continue
            state = colour.get(target)
            if state == 0:
                start = path_nodes.index(target)
                return set(path_assocs[start:]) | {assoc_name}
            if state is None:
                path_assocs.append(assoc_name)
                found = visit(target)
                if found:
                    return found
                path_assocs.pop()
        colour[node] = 1
        path_nodes.pop()
        return set()

    for node in sorted(edges):
        if node not in colour:
            found = visit(node)
            if found:
                return found
    return set()


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
