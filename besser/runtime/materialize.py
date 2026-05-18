"""Materialize a ``DomainModel`` into an executable Python class map.

This module is the bridge between BESSER's design-time metamodel and the
runtime kernel's expected ``class_map`` shape. The generated platform
gets its class_map from emitted ``domain_classes.py`` files; tools that
want to execute methods *without* a generation step (the web modeling
editor's object diagram view, ad-hoc experiments, simulations from a
Python REPL) call ``materialize_classes`` here to get the same class_map
shape on the fly.

We deliberately avoid the platform generator's Jinja template here:
rendering and ``exec``-ing a whole module-string is slow and pulls in
typing imports we don't need at runtime. Instead, we build each Python
class with ``type()`` and attach compiled method bodies as functions —
much smaller surface, identical observable behaviour.
"""
from __future__ import annotations

from collections import namedtuple
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


# Per-class association-end metadata. Keyed by ``(class_name, end_name)`` so
# the engine can: (a) decide singleton vs list when assigning materialized
# neighbours; (b) find the inverse end when walking reverse links to support
# bidirectional navigation (e.g. ``port.incomingStreams`` from the Stream-side
# ``target`` association).
AssociationEndMeta = namedtuple(
    "AssociationEndMeta",
    ["owner_class", "end_name", "target_class", "multiplicity_max", "association_name"],
)

# Returned alongside the class_map. Two layered indices:
#   by_end:      (owner_class, end_name)            -> AssociationEndMeta
#   by_assoc:    (owner_class, association_name)    -> AssociationEndMeta
# The engine uses ``by_end`` for forward assignment and ``by_assoc`` to find
# the inverse end name when walking inbound links.
AssociationsIndex = namedtuple("AssociationsIndex", ["by_end", "by_assoc"])


def materialize_classes(domain_model: Any) -> Dict[str, type]:
    """Build a ``{class_name: PythonClass}`` map from a ``DomainModel``.

    Each emitted class has:

      * ``__init__(**kwargs)`` that stores every attribute as an instance
        slot (defaulting to ``None`` for anything not provided) — keeps
        bodies like ``self.level or 0.0`` predictable.
      * One callable per ``Method`` with ``implementation_type=CODE`` and
        a non-empty ``code`` body. Methods with other implementation
        types are skipped here; the kernel's per-call check still
        surfaces them as AttributeError if a UI tries to dispatch one.

    Args:
        domain_model: A ``besser.BUML.metamodel.structural.DomainModel``.

    Returns:
        Dictionary mapping the diagram's class names to dynamically
        constructed Python classes, ready to hand to ``Engine``.
    """
    classes: Dict[str, type] = {}
    # ``get_classes()`` is the public access point and already drops
    # things like primitive types from the result set.
    for cls in domain_model.get_classes():
        attribute_names = _attribute_names_for(cls)
        namespace: Dict[str, Any] = {"__init__": _make_init(attribute_names)}

        for method in _iter_methods(cls):
            impl = getattr(method, "implementation_type", None)
            code = (getattr(method, "code", "") or "").strip()
            if impl is None or impl.value != "code" or not code:
                continue
            namespace[method.name] = _compile_method(method, code)

        classes[cls.name] = type(cls.name, (), namespace)
    return classes


def build_associations_index(domain_model: Any) -> AssociationsIndex:
    """Walk every association in the domain model and build forward +
    inverse end indices.

    For a bidirectional association ``rel(name='source')`` between
    ``Stream`` (mult ``0..*``, end ``outgoingStreams``) and ``Port``
    (mult ``1``, end ``sourcePort``) we record TWO entries — one per
    end — so navigation works from either side:

      ``by_end[("Stream", "sourcePort")]`` →
          {target_class='Port', multiplicity_max=1, association_name='source'}
      ``by_end[("Port", "outgoingStreams")]`` →
          {target_class='Stream', multiplicity_max=9999, association_name='source'}

    The engine reads ``by_end`` when assigning a materialized neighbour
    to a slot, and ``by_assoc`` when resolving inbound links (the
    ``links`` array on the *other* side of the relationship) into the
    correct attribute name on the current instance.
    """
    by_end: Dict[Tuple[str, str], AssociationEndMeta] = {}
    by_assoc: Dict[Tuple[str, str], AssociationEndMeta] = {}

    for cls in domain_model.get_classes():
        owner_name = cls.name
        for end in _all_association_ends(cls):
            assoc = getattr(end, "owner", None)
            assoc_name = getattr(assoc, "name", "") if assoc is not None else ""
            target_type = getattr(end, "type", None)
            target_name = getattr(target_type, "name", "") if target_type else ""
            multiplicity = getattr(end, "multiplicity", None)
            mult_max = getattr(multiplicity, "max", 1) if multiplicity is not None else 1
            meta = AssociationEndMeta(
                owner_class=owner_name,
                end_name=getattr(end, "name", "") or "",
                target_class=target_name,
                multiplicity_max=int(mult_max) if mult_max is not None else 1,
                association_name=assoc_name,
            )
            if meta.end_name:
                by_end[(owner_name, meta.end_name)] = meta
            if assoc_name:
                # The inverse-lookup table is keyed by (owner_class,
                # association_name). When the engine sees a link
                # ``{association_name: "source", target: this_obj}`` on
                # another instance, it consults this index to find which
                # named slot on ``this_obj`` should receive the reverse
                # reference.
                by_assoc[(owner_name, assoc_name)] = meta

    return AssociationsIndex(by_end=by_end, by_assoc=by_assoc)


# ---------------------------------------------------------------------------
# Helpers — kept private to discourage callers from depending on the precise
# shape of the materialized classes.
# ---------------------------------------------------------------------------
def _attribute_names_for(cls: Any) -> List[str]:
    """Own + inherited attribute names, deterministic order."""
    own = getattr(cls, "attributes", None) or set()
    inherited = (
        cls.inherited_attributes()
        if hasattr(cls, "inherited_attributes")
        else set()
    )
    combined = list(own) + list(inherited)
    # Sorted by name so subsequent ``vars(obj)`` ordering is stable —
    # matters for round-trip tests, not strictly for correctness.
    return [getattr(a, "name", "") for a in combined if getattr(a, "name", None)]


def _all_association_ends(cls: Any) -> Iterable[Any]:
    """Yield this class's association ends. Falls back to a structural-
    walk if ``all_association_ends`` (the BUML helper) isn't present."""
    if hasattr(cls, "all_association_ends"):
        try:
            return list(cls.all_association_ends())
        except Exception:
            pass
    # Fallback: walk every association and pick the ends whose ``type``
    # points to (or is a generalisation of) this class.
    associations = getattr(cls, "associations", None) or set()
    return [end for assoc in associations for end in getattr(assoc, "ends", [])
            if getattr(end, "type", None) is cls]


def _iter_methods(cls: Any) -> Iterable[Any]:
    """Sorted iteration over a class's declared methods."""
    methods = getattr(cls, "methods", None) or set()
    return sorted(methods, key=lambda m: getattr(m, "name", ""))


def _make_init(attribute_names: List[str]) -> Callable[..., None]:
    """Build an __init__ that stores every attribute kwarg as a slot.

    Missing kwargs default to ``None`` — matches the generated-platform
    convention so method bodies like ``(self.level or 0.0)`` behave the
    same in both runtimes.
    """
    def __init__(self, **kwargs):  # noqa: N807 — Python dunder
        for name in attribute_names:
            setattr(self, name, kwargs.get(name))
        # Tolerate extra kwargs the caller may pass (e.g. attributes the
        # diagram knew about but this class no longer declares). Set them
        # as well so method bodies that reference them don't AttributeError.
        for k, v in kwargs.items():
            if k not in attribute_names and not k.startswith("_"):
                setattr(self, k, v)
    return __init__


def _compile_method(method: Any, code: str) -> Callable:
    """Compile a Method's code into a function bound on the class.

    Two shapes supported:

      (a) Full ``def name(self, ...): body`` — the shape the BESSER web
          modeling editor's Python Implementation panel saves. We exec
          it as-is and look the function up by name in the resulting
          namespace.
      (b) Just the body, e.g. ``self.x = self.x + 1``. We wrap with a
          synthetic ``def`` header that uses the Method's parameter
          names from the diagram.

    Anything raised by the body propagates; the kernel turns it into an
    HTTP error at the boundary.
    """
    stripped = code.lstrip()
    method_name = getattr(method, "name", "_anonymous")

    if stripped.startswith("def "):
        # Author authored the whole def block (web-editor shape).
        ns: Dict[str, Any] = {}
        exec(compile(code, f"<method:{method_name}>", "exec"), ns)
        fn = ns.get(method_name)
        if fn is None or not callable(fn):
            # Defensive: fall back to the first callable we find. Covers
            # the case where the def's name in code disagrees with the
            # diagram's Method.name (unusual but possible after a rename).
            for value in ns.values():
                if callable(value):
                    fn = value
                    break
        if fn is None:
            raise ValueError(
                f"Method '{method_name}' code parsed but yielded no callable"
            )
        return fn

    # Bare body — wrap with a synthetic def whose signature mirrors the
    # Method's declared parameters.
    param_names = [getattr(p, "name", "") for p in (getattr(method, "parameters", []) or [])]
    param_names = [p for p in param_names if p]
    signature = ", ".join(["self"] + param_names)
    wrapped_lines = [f"def {method_name}({signature}):"]
    for line in code.splitlines() or [""]:
        wrapped_lines.append("    " + line if line else "")
    wrapped_source = "\n".join(wrapped_lines) + "\n"
    ns_w: Dict[str, Any] = {}
    exec(compile(wrapped_source, f"<method:{method_name}>", "exec"), ns_w)
    return ns_w[method_name]


__all__ = [
    "materialize_classes",
    "build_associations_index",
    "AssociationEndMeta",
    "AssociationsIndex",
]
