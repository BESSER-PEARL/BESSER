"""Runtime engine — generic method dispatcher for besser4DT.

Goal: execute any user-authored method body against the live instance
state held by an ``instance_manager`` — both inside generated platforms
and inside the web modeling editor's object diagram.

The engine is intentionally *generic*: no generated module is imported
here. Dependencies (the singleton instance manager, the class registry
mapping class names to Python classes) are passed in at construction
time, so this file can be unit-tested against fakes and copied verbatim
into every generated app regardless of domain.

This module covers ``implementation_type == CODE`` only. The generated
domain class already exposes those bodies as real Python methods, so the
engine just looks up by name and calls.

**Link resolution** (Phase 1.0.8): when a method body navigates an
association — ``self.ports``, ``port.incomingStreams``, etc. — the
engine materializes those neighbour instances too, recursively, so the
body sees a live object graph instead of empty defaults. After the call
returns, the engine walks every materialized neighbour and writes any
mutated attributes back to the store, so neighbour-affecting side
effects (e.g. a Compressor writing to its outlet stream's pressure)
persist.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple


class Engine:
    """Method-driven runtime for instance state.

    The contract:

      * Reconstruct a domain Python object — *and its connected
        neighbours* — from the attribute dicts the ``instance_manager``
        is currently holding.
      * Look up ``method_name`` on the root object and call it with
        ``**args``.
      * Read mutated attributes back out of every materialized object
        (root + all reachable neighbours) and ``update_instance`` so the
        store reflects the new state.
      * Return a status dict so the HTTP layer can surface the call
        result + the freshly-mutated root instance to the frontend.

    Failures are propagated as raised exceptions; the FastAPI layer turns
    them into HTTP responses.
    """

    def __init__(
        self,
        instance_manager: Any,
        class_map: Dict[str, type],
        associations_index: Any = None,
    ):
        """Build an Engine wired to a specific platform.

        Args:
            instance_manager: Object with ``get_instance``,
                ``get_all_instances``, and ``update_instance``.
            class_map: ``{class_name: PythonClass}``.
            associations_index: Optional ``AssociationsIndex`` (from
                ``materialize.build_associations_index``) — when present,
                the engine uses it to (a) decide singleton vs list on
                association slots, and (b) resolve inbound links to the
                right reverse-navigation slot. When ``None``, link
                resolution silently no-ops and method bodies see empty
                neighbour slots (the pre-Phase-1.0.8 behaviour).
        """
        self._im = instance_manager
        self._classes = class_map
        self._assoc = associations_index

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def invoke(
        self,
        class_name: str,
        instance_id: str,
        method_name: str,
        args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Invoke ``method_name`` on the named instance with ``args``.

        Raises:
            KeyError: If no instance with that id exists for the class.
            ValueError: If the class isn't registered (typo / wrong domain).
            AttributeError: If the class has no method by that name.
            TypeError: If ``args`` doesn't match the method's signature.
            Exception: Whatever the method body raises is re-raised.
        """
        args = dict(args or {})

        instance_data = self._im.get_instance(class_name, instance_id)
        if instance_data is None:
            raise KeyError(
                f"No instance with id '{instance_id}' for class '{class_name}'"
            )

        if class_name not in self._classes:
            raise ValueError(f"Unknown class '{class_name}' (not in class_map)")

        # Build a reverse-link index once per invoke. Cheap on small
        # plants; for very large plants this can be lifted to an
        # incremental cache later.
        reverse_index = self._build_reverse_index()

        visited: Dict[Tuple[str, str], Any] = {}
        domain_obj = self._materialize_graph(
            class_name, instance_id, visited, reverse_index
        )

        method: Optional[Callable] = getattr(domain_obj, method_name, None)
        if method is None or not callable(method):
            raise AttributeError(
                f"Class '{class_name}' has no callable method '{method_name}' — "
                f"check the implementation_type of the method in the source diagram."
            )

        result = method(**args)

        # Persist mutated state on EVERY visited instance. With link
        # resolution, a method body can write neighbour attributes
        # (e.g. ``Compressor.step`` updating its outlet stream), and
        # those mutations must round-trip back to the store.
        self._writeback_graph(visited)

        updated = self._im.get_instance(class_name, instance_id)
        return {
            "instance_id": instance_id,
            "class_name": class_name,
            "method_name": method_name,
            "args": args,
            "result": result,
            "instance": updated,
        }

    def tick(
        self,
        class_name: str,
        instance_id: str,
        dt: float = 1.0,
    ) -> Dict[str, Any]:
        """Convenience: invoke ``step(dt=...)`` on the named instance.

        Used by the periodic scheduler which ticks every instance with
        the conventional ``step(self, dt)`` method. Manual per-method
        buttons in the UI go through ``invoke`` directly.
        """
        return self.invoke(
            class_name=class_name,
            instance_id=instance_id,
            method_name="step",
            args={"dt": dt},
        )

    # -----------------------------------------------------------------
    # Graph materialization (Phase 1.0.8)
    # -----------------------------------------------------------------
    def _materialize_graph(
        self,
        class_name: str,
        instance_id: str,
        visited: Dict[Tuple[str, str], Any],
        reverse_index: Dict[Tuple[str, str], List[Tuple[str, str, str]]],
    ):
        """Recursively materialize an instance + its connected graph.

        Cycle protection: every (class, id) is registered in ``visited``
        *before* recursing into its neighbours, so a Stream that
        navigates back to a Port that navigates back to the Stream
        terminates instead of spinning.

        Returns the materialized Python object, or ``None`` if the
        instance can't be found in the store (a stale link, say —
        better to no-op than to crash mid-tick).
        """
        key = (class_name, instance_id)
        if key in visited:
            return visited[key]

        instance_data = self._im.get_instance(class_name, instance_id)
        if instance_data is None:
            return None
        cls = self._classes.get(class_name)
        if cls is None:
            return None

        obj = self._materialize_one(cls, instance_data)
        visited[key] = obj   # register BEFORE recursing into links

        # Forward links — the instance's own ``links`` array. Each link
        # says "this instance is related via ``association_name`` to a
        # target instance".
        outgoing_by_end: Dict[str, List[Any]] = defaultdict(list)
        for link in instance_data.get("links", []) or []:
            target_class = link.get("target_class")
            target_id = link.get("target_id")
            assoc_name = link.get("association_name")
            if not (target_class and target_id and assoc_name):
                continue
            neighbour = self._materialize_graph(
                target_class, target_id, visited, reverse_index
            )
            if neighbour is None:
                continue
            end_name = self._forward_end_name(class_name, assoc_name)
            if end_name:
                outgoing_by_end[end_name].append(neighbour)

        # Inbound links — other instances that point to this one. Drives
        # bidirectional navigation: ``stream.source = port`` is stored
        # as an outgoing link on the Stream, but ``port.outgoingStreams``
        # is its reverse-navigation slot on Port.
        for source_class, source_id, assoc_name in reverse_index.get(key, []):
            neighbour = self._materialize_graph(
                source_class, source_id, visited, reverse_index
            )
            if neighbour is None:
                continue
            end_name = self._inverse_end_name(class_name, assoc_name)
            if end_name:
                outgoing_by_end[end_name].append(neighbour)

        # Assign each set of neighbours to the right slot, with the right
        # cardinality. Multiplicity comes from the association index;
        # when we don't know it, default to a list (the safer choice —
        # method bodies can still do ``for x in self.foo``).
        for end_name, neighbours in outgoing_by_end.items():
            mult_max = self._multiplicity_max(class_name, end_name)
            if mult_max == 1:
                setattr(obj, end_name, neighbours[0] if neighbours else None)
            else:
                setattr(obj, end_name, list(neighbours))

        return obj

    def _materialize_one(self, cls: type, instance_data: Dict[str, Any]):
        """Build a single domain Python object from its attribute dict.

        Falls back to ``cls() + setattr`` if the constructor rejects an
        unknown kwarg (e.g. legacy data or a renamed attribute) — better
        to run the method on a partially-rehydrated object than to
        abort on a single bad key.

        Also pre-initializes every association slot known for this
        class to a sensible default (``None`` for singletons, ``[]``
        for many) so method bodies that read ``self.ports`` etc. don't
        AttributeError when the instance has no outgoing links.
        """
        attrs = dict(instance_data.get("attributes") or {})
        try:
            obj = cls(**attrs)
        except TypeError:
            obj = cls()
            for key, value in attrs.items():
                try:
                    setattr(obj, key, value)
                except Exception:
                    continue
        # Pre-initialize association slots
        if self._assoc is not None:
            cls_name = type(obj).__name__
            for (owner, end_name), meta in self._assoc.by_end.items():
                if owner != cls_name or not end_name:
                    continue
                if not hasattr(obj, end_name):
                    default = None if meta.multiplicity_max == 1 else []
                    try:
                        setattr(obj, end_name, default)
                    except Exception:
                        continue
        return obj

    def _build_reverse_index(
        self,
    ) -> Dict[Tuple[str, str], List[Tuple[str, str, str]]]:
        """Scan the store once and build ``{(target_class, target_id):
        [(source_class, source_id, association_name)]}``.

        Lets ``_materialize_graph`` resolve inbound links — without this
        we could only navigate associations in the direction they were
        authored, and bidirectional roles (the common case in DT
        topologies) wouldn't work."""
        index: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = defaultdict(list)
        get_all = getattr(self._im, "get_all_instances", None)
        if get_all is None:
            return index
        try:
            rows = get_all() or []
        except Exception:
            return index
        for row in rows:
            source_class = row.get("class_name")
            source_id = row.get("id")
            if not (source_class and source_id):
                continue
            for link in row.get("links", []) or []:
                target_class = link.get("target_class")
                target_id = link.get("target_id")
                assoc_name = link.get("association_name")
                if not (target_class and target_id and assoc_name):
                    continue
                index[(target_class, target_id)].append(
                    (source_class, source_id, assoc_name)
                )
        return index

    def _forward_end_name(self, owner_class: str, association_name: str) -> Optional[str]:
        """Slot name on ``owner_class`` for an outgoing link. Built by
        looking up the inverse end (the one belonging to the target
        class) and… no — we want the end on the OWNER's side that
        targets the OTHER class. That's the entry in ``by_assoc`` whose
        ``owner_class`` matches the LINK target, because BUML names the
        end after the role on its own side. Falls back to using
        ``association_name`` directly when the index isn't available."""
        if self._assoc is None:
            return association_name or None
        # The link says owner -- assoc_name --> target. The end on owner's
        # side that holds the navigation slot is the one whose target_class
        # is the link's target. We don't have target_class here — but the
        # caller will use the index entry indexed by (owner, assoc_name)
        # which is one of the two ends (could be either direction). For
        # outgoing forward navigation we want the OWNER's end, so it
        # should be by_end keyed by (owner_class, ???) — we don't have the
        # end name. Use ``by_assoc[(owner, assoc)].end_name`` which
        # actually returns the OWNER's own end (the one that points
        # outward from this owner), because that's how we indexed it.
        meta = self._assoc.by_assoc.get((owner_class, association_name))
        if meta is not None and meta.end_name:
            return meta.end_name
        return association_name or None

    def _inverse_end_name(
        self, owner_class: str, association_name: str
    ) -> Optional[str]:
        """Slot name on ``owner_class`` for an *inbound* link of
        ``association_name``. Same lookup as forward — ``by_assoc`` is
        keyed by (owner, assoc), so it already discriminates the right
        end per class."""
        if self._assoc is None:
            return None
        meta = self._assoc.by_assoc.get((owner_class, association_name))
        if meta is not None and meta.end_name:
            return meta.end_name
        return None

    def _multiplicity_max(self, owner_class: str, end_name: str) -> int:
        """Max multiplicity (1 or >1) for an end. ``9999`` reads as 'many'.
        Default to many when unknown so method bodies that iterate
        ``self.<assoc>`` don't blow up on a singleton-shaped slot."""
        if self._assoc is None:
            return 9999
        meta = self._assoc.by_end.get((owner_class, end_name))
        if meta is None:
            return 9999
        return meta.multiplicity_max

    # -----------------------------------------------------------------
    # Writeback (Phase 1.0.8)
    # -----------------------------------------------------------------
    def _writeback_graph(self, visited: Dict[Tuple[str, str], Any]) -> None:
        """Walk every materialized object and persist its mutated
        attributes. Neighbour mutations (e.g. a method that writes a
        downstream stream's pressure) are the whole point of resolving
        the link graph — without this they'd evaporate at call return.
        """
        for (class_name, instance_id), obj in visited.items():
            original = self._im.get_instance(class_name, instance_id)
            if original is None:
                continue
            new_attrs = self._extract_attributes(obj, original)
            self._im.update_instance(class_name, instance_id, new_attrs)

    def _extract_attributes(
        self, domain_obj: Any, original: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Read back mutable attribute state from ``domain_obj``.

        Keys come from the original attribute set *and* any new attribute
        the method body introduced via ``self.foo = ...``. Association
        slots are excluded — the runtime kernel does not own link state
        through the attribute channel.
        """
        original_attrs = dict(original.get("attributes") or {})
        result: Dict[str, Any] = {}
        candidate_keys = set(original_attrs.keys()) | {
            k for k in vars(domain_obj).keys() if not k.startswith("_")
        }
        # Filter out anything that looks like an association slot we set
        # during materialization. If the engine has the associations
        # index, we know the exact end names to skip; otherwise we fall
        # back to skipping list/has-to_dict values.
        association_slots = self._known_association_slots_for(domain_obj)
        for key in candidate_keys:
            if key in association_slots:
                continue
            value = getattr(domain_obj, key, None)
            if isinstance(value, list):
                continue
            if hasattr(value, "to_dict") and callable(value.to_dict):
                continue
            result[key] = value
        return result

    def _known_association_slots_for(self, domain_obj: Any) -> set:
        """Names of attributes on ``domain_obj`` that came from the
        association index (i.e. they hold materialized neighbours and
        should NOT be serialized as attribute values)."""
        if self._assoc is None:
            return set()
        cls_name = type(domain_obj).__name__
        return {
            end_name for (owner, end_name) in self._assoc.by_end.keys()
            if owner == cls_name
        }
