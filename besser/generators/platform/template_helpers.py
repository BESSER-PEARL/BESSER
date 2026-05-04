"""Jinja filters for the Platform generator.

These helpers translate `PlatformCustomization` enum values into the strings
the generated React/xyflow code expects, so each piece of conversion logic
lives in exactly one place.
"""

from typing import Optional

from besser.BUML.metamodel.platform_customization import ArrowStyle, LineStyle


# CSS `stroke-dasharray` patterns. None means "default solid line"; an explicit
# "0" forces solid even when something upstream guesses otherwise.
_DASH_PATTERNS = {
    LineStyle.SOLID: "0",
    LineStyle.DASHED: "8 4",
    LineStyle.DOTTED: "2 4",
}


def dash_for(line_style) -> Optional[str]:
    """Return the SVG `stroke-dasharray` for a `LineStyle` value, or None."""
    if line_style is None:
        return None
    if isinstance(line_style, str):
        try:
            line_style = LineStyle(line_style)
        except ValueError:
            return None
    return _DASH_PATTERNS.get(line_style)


# Mapping from our ArrowStyle vocabulary to the marker descriptor consumed by
# the generated React code. xyflow ships only Arrow / ArrowClosed; the other
# entries reference custom <marker> defs injected once into InstanceCanvas.
_MARKER_DESCRIPTORS = {
    ArrowStyle.NONE: None,
    ArrowStyle.FILLED_TRIANGLE: {"type": "arrowclosed"},
    ArrowStyle.OPEN_TRIANGLE: {"type": "arrow"},
    ArrowStyle.DIAMOND: {"type": "url", "id": "platform-diamond"},
    ArrowStyle.OPEN_DIAMOND: {"type": "url", "id": "platform-open-diamond"},
    ArrowStyle.CIRCLE: {"type": "url", "id": "platform-circle"},
}


def marker_for(arrow_style):
    """Return the marker descriptor for an `ArrowStyle`, or None if 'none'."""
    if arrow_style is None:
        return None
    if isinstance(arrow_style, str):
        try:
            arrow_style = ArrowStyle(arrow_style)
        except ValueError:
            return None
    return _MARKER_DESCRIPTORS.get(arrow_style)


def enum_value(value) -> Optional[str]:
    """Return the .value of an enum, or the raw value if it's already a string."""
    if value is None:
        return None
    if hasattr(value, "value"):
        return value.value
    return str(value)


# ---------------------------------------------------------------------------
# Representation resolution (inheritance-aware)
# ---------------------------------------------------------------------------
#
# `is_port`, `is_connection_class` and `port_side` are class-level flags but
# the generated editor must treat subclasses the same as their flagged
# ancestor. These helpers walk the BUML inheritance chain so the template
# code can stay declarative.


def _walk_class_chain(cls):
    """Yield `cls` then its parents recursively (BUML Class.all_parents())."""
    yield cls
    parents = getattr(cls, "all_parents", None)
    if callable(parents):
        for parent in parents():
            yield parent


def build_subclass_registry(classes):
    """Map each class name to itself + all transitive subclasses.

    Used by the generated runtime to do polymorphic instance matching: an
    instance of ``Compressor`` should also be acceptable wherever a class
    typed as ``Equipment`` is expected.
    """
    classes = list(classes)
    name_to_cls = {c.name: c for c in classes}
    children_by_parent = {}
    for cls in classes:
        parents = getattr(cls, "parents", None)
        if not callable(parents):
            continue
        for parent in parents():
            if parent.name in name_to_cls:
                children_by_parent.setdefault(parent.name, []).append(cls.name)

    registry = {}
    for cls in classes:
        seen = {cls.name}
        queue = list(children_by_parent.get(cls.name, []))
        while queue:
            n = queue.pop(0)
            if n in seen:
                continue
            seen.add(n)
            queue.extend(children_by_parent.get(n, []))
        registry[cls.name] = sorted(seen)
    return registry


def resolve_class_representation(cls, customization):
    """Return a dict describing the effective representation of a class.

    Walks the inheritance chain — the first ancestor (including ``cls``) that
    sets one of the representation flags wins. ``port_side`` is taken from
    the same class that contributed ``is_port``.

    Returns:
        ``{'mode': 'node'|'container'|'port'|'connection', 'port_side': str|None}``.
    """
    if customization is None:
        return {"mode": "node", "port_side": None}

    for ancestor in _walk_class_chain(cls):
        cust = customization.get_class_customization(ancestor.name)
        if cust.is_port:
            return {"mode": "port", "port_side": enum_value(cust.port_side) or "auto"}
        if cust.is_connection_class:
            return {"mode": "connection", "port_side": None}
        if cust.is_container:
            return {"mode": "container", "port_side": None}
    return {"mode": "node", "port_side": None}


def resolve_connection_edge_style(cls, customization):
    """For a connection-class, resolve edge styling up the inheritance chain.

    The user often only sets style on the base class (e.g. ``Stream``) and
    expects subclasses (``MaterialStream``, ``EnergyStream``) to inherit it.
    This walks ancestors and merges fields field-wise: each field uses the
    first ancestor that sets it.

    Returns a ``ClassCustomization``-like object with the merged edge fields,
    or ``None`` when neither this class nor any ancestor is a connection
    class — there's nothing to resolve.
    """
    if customization is None:
        return None

    chain = list(_walk_class_chain(cls))
    is_connection = any(
        customization.get_class_customization(c.name).is_connection_class for c in chain
    )
    if not is_connection:
        return None

    # Manually merge each edge field — first non-None ancestor wins.
    fields = (
        "edge_color", "line_width", "line_style", "line_routing",
        "source_arrow_style", "target_arrow_style",
        "label_visible", "label_font_size", "label_font_color",
    )
    merged = type("MergedEdgeStyle", (), {})()
    for f in fields:
        value = None
        for ancestor in chain:
            cust = customization.get_class_customization(ancestor.name)
            v = getattr(cust, f, None)
            if v is not None:
                value = v
                break
        setattr(merged, f, value)
    return merged


def find_connection_endpoints(cls, customization):
    """For a connection-class, locate its source and target endpoint associations.

    Iterates the class's outgoing association ends (own + inherited) and picks
    the ones whose association name has been flagged via
    ``AssociationCustomization.is_source_endpoint`` / ``is_target_endpoint``.

    Returns:
        ``{'sourceAssociation': str|None, 'targetAssociation': str|None,
           'portClass': str|None}``.
        ``portClass`` is the target-class of the source-endpoint association
        (we use the source endpoint's target class because both endpoints
        should target the same port-class anyway).
    """
    if customization is None:
        return {"sourceAssociation": None, "targetAssociation": None, "portClass": None}

    source_assoc = None
    target_assoc = None
    port_class = None

    ends_method = getattr(cls, "all_association_ends", None)
    if not callable(ends_method):
        return {"sourceAssociation": None, "targetAssociation": None, "portClass": None}

    for end in ends_method():
        assoc = end.owner
        assoc_cust = customization.get_association_customization(assoc.name)
        if assoc_cust.is_source_endpoint and source_assoc is None:
            source_assoc = assoc.name
            if hasattr(end, "type") and getattr(end.type, "name", None):
                port_class = end.type.name
        elif assoc_cust.is_target_endpoint and target_assoc is None:
            target_assoc = assoc.name
            # Fall back to target side if no port class found yet
            if port_class is None and hasattr(end, "type") and getattr(end.type, "name", None):
                port_class = end.type.name

    return {
        "sourceAssociation": source_assoc,
        "targetAssociation": target_assoc,
        "portClass": port_class,
    }


def build_representation_registries(classes, customization):
    """Build PORT_CLASSES and CONNECTION_CLASSES registries for the template.

    Resolves inheritance so a subclass of a flagged class is automatically
    included.

    Returns:
        ``{'port_classes': {name: {'portSide': str}},
           'connection_classes': {name: {'sourceAssociation': str|None,
                                         'targetAssociation': str|None,
                                         'portClass': str|None}}}``.
    """
    port_classes = {}
    connection_classes = {}
    for cls in classes:
        rep = resolve_class_representation(cls, customization)
        if rep["mode"] == "port":
            port_classes[cls.name] = {"portSide": rep["port_side"] or "auto"}
        elif rep["mode"] == "connection":
            connection_classes[cls.name] = find_connection_endpoints(cls, customization)
    return {"port_classes": port_classes, "connection_classes": connection_classes}


def validate_representation(classes, customization):
    """Check the customization for connection-class wiring problems.

    Returns:
        list of human-readable error strings. Empty list means the
        configuration is consistent and the editor can be generated.
    """
    issues = []
    if customization is None:
        return issues

    classes_by_name = {c.name: c for c in classes}
    registry = build_representation_registries(classes, customization)
    port_class_names = set(registry["port_classes"].keys())

    # Per connection-class, ensure exactly one source & one target endpoint exist
    # and their target classes are port-classes.
    for conn_name, info in registry["connection_classes"].items():
        cls = classes_by_name.get(conn_name)
        if cls is None:
            continue
        if not info["sourceAssociation"]:
            issues.append(
                f"Connection class '{conn_name}' has no association marked as Source endpoint."
            )
        if not info["targetAssociation"]:
            issues.append(
                f"Connection class '{conn_name}' has no association marked as Target endpoint."
            )
        if info["portClass"] and info["portClass"] not in port_class_names:
            issues.append(
                f"Connection class '{conn_name}' endpoint targets '{info['portClass']}', "
                f"which is not flagged as a Port."
            )

    # Endpoint flags set on associations whose source class isn't a connection
    # (or a connection's subclass). In BUML, BinaryAssociation ends are
    # unordered — a class is "on the source side" of an association if that
    # association appears in its ``all_association_ends()``. So we collect
    # every owner class for the association and require at least one of them
    # (resolved through inheritance) to be a connection-class.
    for assoc_name, assoc_cust in customization.association_overrides.items():
        if not (assoc_cust.is_source_endpoint or assoc_cust.is_target_endpoint):
            continue
        owner_classes = []
        for cls in classes:
            ends_method = getattr(cls, "all_association_ends", None)
            if not callable(ends_method):
                continue
            for end in ends_method():
                if end.owner.name == assoc_name:
                    owner_classes.append(cls)
                    break
        if not owner_classes:
            issues.append(
                f"Association '{assoc_name}' is marked as an endpoint but no class in "
                f"the model has an end on this association."
            )
            continue
        if not any(
            resolve_class_representation(c, customization)["mode"] == "connection"
            for c in owner_classes
        ):
            owner_names = ", ".join(c.name for c in owner_classes)
            issues.append(
                f"Association '{assoc_name}' is marked as an endpoint but no source "
                f"class is a Connection. Source candidates: {owner_names}."
            )

    return issues
