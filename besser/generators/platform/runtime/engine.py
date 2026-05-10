"""Runtime engine — Phase 1.0.

Goal: execute a user-authored ``step(self, dt)`` method body against the
live instance state held by the generated platform's ``instance_manager``.

The engine is intentionally *generic*: no generated module is imported here.
Dependencies (the singleton instance manager, the class registry mapping
class names to Python classes) are passed in at construction time, so this
file can be unit-tested against fakes and copied verbatim into every
generated app regardless of domain.

Phase 1.0 covers ``implementation_type == CODE`` only — the generated
domain class already exposes those bodies as real Python methods, so the
engine just looks up by name and calls. Phase 1.1+ will plug a state-machine
dispatcher and a periodic scheduler into the same surface.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class Engine:
    """Tick-driven runtime for instance state.

    The contract:

      * Reconstruct a domain Python object from the attribute dict the
        ``instance_manager`` is currently holding.
      * Look up ``method_name`` on that object and call it with ``dt``.
      * Read mutated attributes back out and ``update_instance`` so the
        store reflects the new state — including any new attributes the
        method introduced.
      * Return a small status dict so the HTTP layer can surface the call
        result + the freshly-mutated instance to the frontend.

    Failures are propagated as raised exceptions; the FastAPI layer turns
    them into HTTP responses. The engine never silently no-ops — if there
    is no method, no instance, or the body raises, the caller hears about
    it. That matters for a DT: a silent simulation is worse than a stopped
    one.
    """

    def __init__(self, instance_manager: Any, class_map: Dict[str, type]):
        self._im = instance_manager
        self._classes = class_map

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def tick(
        self,
        class_name: str,
        instance_id: str,
        method_name: str = "step",
        dt: float = 1.0,
    ) -> Dict[str, Any]:
        """Run one tick of ``method_name`` on the named instance.

        Args:
            class_name: Domain class of the instance (e.g. ``"Compressor"``).
            instance_id: UUID assigned by ``instance_manager`` at creation.
            method_name: Method to dispatch. Defaults to ``"step"`` —
                Phase 1.0 convention — so the UI can ▶ Tick without picking.
            dt: Simulated time delta passed to the method.

        Returns:
            ``{instance_id, class_name, method_name, dt, result, instance}``
            where ``instance`` is the freshly-mutated row in the store.

        Raises:
            KeyError: If no instance with that id exists for the class.
            ValueError: If the class isn't registered (typo / wrong domain).
            AttributeError: If the class has no method by that name.
            Exception: Whatever the method body raises is re-raised.
        """
        instance_data = self._im.get_instance(class_name, instance_id)
        if instance_data is None:
            raise KeyError(
                f"No instance with id '{instance_id}' for class '{class_name}'"
            )

        cls = self._classes.get(class_name)
        if cls is None:
            raise ValueError(f"Unknown class '{class_name}' (not in class_map)")

        domain_obj = self._materialize(cls, instance_data)

        method: Optional[Callable] = getattr(domain_obj, method_name, None)
        if method is None or not callable(method):
            raise AttributeError(
                f"Class '{class_name}' has no callable method '{method_name}' — "
                f"check the implementation_type of the method in the source diagram."
            )

        result = method(dt)

        # Persist mutated state back. This is the part that makes the tick
        # actually move the world forward instead of running in a vacuum.
        new_attrs = self._extract_attributes(domain_obj, instance_data)
        self._im.update_instance(class_name, instance_id, new_attrs)
        updated = self._im.get_instance(class_name, instance_id)

        return {
            "instance_id": instance_id,
            "class_name": class_name,
            "method_name": method_name,
            "dt": dt,
            "result": result,
            "instance": updated,
        }

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _materialize(self, cls: type, instance_data: Dict[str, Any]):
        """Build a domain Python object from the manager's attribute dict.

        The generated ``__init__`` accepts attribute kwargs by name. If the
        dict has keys the constructor doesn't recognise (e.g. legacy data,
        or attribute names that needed safe_var_name mangling), fall back
        to constructing empty + ``setattr`` so we don't lose state.
        """
        attrs = dict(instance_data.get("attributes") or {})
        try:
            return cls(**attrs)
        except TypeError:
            obj = cls()
            for key, value in attrs.items():
                try:
                    setattr(obj, key, value)
                except Exception:
                    # Ignore attributes the class refuses — better to run
                    # the tick on a partially-rehydrated object than to
                    # abort the simulation on a single bad key.
                    continue
            return obj

    def _extract_attributes(
        self, domain_obj: Any, original: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Read back mutable attribute state from ``domain_obj``.

        Returns a dict suitable for ``instance_manager.update_instance``.
        Keys come from the original attribute set *and* any new attribute
        the method body introduced via ``self.foo = ...``. Association
        slots are excluded — the runtime kernel does not own link state
        in Phase 1.0; that's the link API's job.
        """
        original_attrs = dict(original.get("attributes") or {})
        result: Dict[str, Any] = {}

        # Combine the originally-tracked keys with anything the method may
        # have set on self for the first time (e.g. lazy state). We use
        # vars() for the second source because dataclass-style instance
        # attributes are the common case.
        candidate_keys = set(original_attrs.keys()) | {
            k for k in vars(domain_obj).keys() if not k.startswith("_")
        }

        for key in candidate_keys:
            value = getattr(domain_obj, key, None)
            # Skip association ends (lists of related objects, or single
            # objects with .to_dict). The runtime kernel doesn't propagate
            # link mutations through the attribute channel.
            if isinstance(value, list):
                continue
            if hasattr(value, "to_dict") and callable(value.to_dict):
                continue
            result[key] = value

        return result
