"""Input bindings — maps instance attributes to simulated data sources.

A *binding* declares that a named attribute on a specific instance should be
overwritten before every tick with a value computed by a built-in simulator.
This is the core mechanism that feeds synthetic input data into a Digital Twin
without requiring an external broker (MQTT, REST) at simulation time.

Supported simulator kinds
-------------------------
``constant``
    The attribute holds a fixed value every tick.
    params: ``{"value": <number>}``

``sine``
    Sinusoidal signal: ``offset + amplitude * sin(2π * frequency * t)``
    where ``t = tick_count * dt`` (elapsed simulated time in seconds).
    params: ``{"amplitude": float, "frequency": float, "offset": float}``
    (all optional, defaulting to amplitude=1, frequency=1, offset=0)

``ramp``
    Linear ramp: ``start + rate * t``, optionally clamped at ``max_value``.
    params: ``{"start": float, "rate": float, "max_value": float | None}``

``replay_csv``
    Replay a pre-loaded (tick_index, value) sequence, cycling if exhausted.
    params: ``{"source_id": str}`` — the opaque key returned by
    ``InputBindings.load_csv(path_or_rows)``.

Persistence
-----------
Call ``InputBindings.save(path)`` / ``InputBindings.load(path)`` to persist
the binding table as JSON so bindings survive a server restart.  Replayed CSV
data is stored inline in the same JSON file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class InputBindings:
    """Registry of attribute→simulator bindings for all instances.

    The scheduler calls :meth:`apply_pre_tick` once before every engine tick
    so that simulated values are written into the InstanceManager store *before*
    the step methods read them.
    """

    def __init__(self, instance_manager):
        self._im = instance_manager
        # { (class_name, instance_id, attribute): {"kind": str, "params": dict} }
        self._bindings: dict[tuple[str, str, str], dict] = {}
        # { source_id: [(value, ...), ...] } — CSV replay data
        self._csv_sources: dict[str, list[Any]] = {}

    # ------------------------------------------------------------------ CRUD

    def set_binding(
        self,
        class_name: str,
        instance_id: str,
        attribute: str,
        kind: str,
        params: dict | None = None,
    ) -> None:
        """Set or replace the input binding for a single attribute slot.

        Parameters
        ----------
        class_name, instance_id, attribute:
            Identify the target slot.
        kind:
            Simulator kind — ``"constant"``, ``"sine"``, ``"ramp"``, or
            ``"replay_csv"``.
        params:
            Simulator parameters dict (kind-specific; see module docstring).
        """
        _VALID_KINDS = {"constant", "sine", "ramp", "replay_csv"}
        if kind not in _VALID_KINDS:
            raise ValueError(f"Unknown binding kind {kind!r}. Choose from {_VALID_KINDS}.")
        key = (class_name, instance_id, attribute)
        self._bindings[key] = {"kind": kind, "params": params or {}}

    def clear_binding(self, class_name: str, instance_id: str, attribute: str) -> None:
        """Remove the binding for the given slot (no-op if not set)."""
        self._bindings.pop((class_name, instance_id, attribute), None)

    def get_binding(self, class_name: str, instance_id: str, attribute: str) -> dict | None:
        """Return the binding dict for the slot, or ``None`` if unbound."""
        return self._bindings.get((class_name, instance_id, attribute))

    def list_bindings_for_instance(
        self, class_name: str, instance_id: str
    ) -> list[dict]:
        """Return all bindings for the given instance as serialisable dicts."""
        result = []
        for (cn, iid, attr), cfg in self._bindings.items():
            if cn == class_name and iid == instance_id:
                result.append({"attribute": attr, "kind": cfg["kind"], "params": cfg["params"]})
        return result

    def load_csv(self, source_id: str, rows: list[Any]) -> str:
        """Register a replay-CSV data source.

        Parameters
        ----------
        source_id:
            Caller-chosen stable key (e.g. UUID). Used as ``params.source_id``
            in a ``replay_csv`` binding.
        rows:
            List of scalar values (numbers or strings) to replay in order.

        Returns
        -------
        str
            The ``source_id`` echoed back so callers can write
            ``{"source_id": returned_id}`` into the binding params.
        """
        self._csv_sources[source_id] = list(rows)
        return source_id

    # -------------------------------------------------------------- tick hook

    def apply_pre_tick(self, tick_count: int, dt: float = 1.0) -> None:
        """Write simulated values into the instance manager before the tick.

        Called by :class:`besser.runtime.scheduler.Scheduler` before
        dispatching the engine tick for each instance.
        """
        t = tick_count * dt  # elapsed simulated time in seconds
        for (class_name, instance_id, attribute), cfg in list(self._bindings.items()):
            try:
                value = self._simulate(cfg["kind"], cfg["params"], tick_count, t)
            except Exception:
                continue  # bad binding — skip silently (don't abort the whole tick)
            try:
                inst = self._im.get_instance(class_name, instance_id)
                new_attrs = dict(inst["attributes"])
                new_attrs[attribute] = value
                self._im.update_instance(class_name, instance_id, new_attrs)
            except Exception:
                continue

    # --------------------------------------------------------- simulators

    @staticmethod
    def _simulate(kind: str, params: dict, tick_count: int, t: float) -> Any:
        if kind == "constant":
            return params.get("value", 0)

        if kind == "sine":
            amplitude = float(params.get("amplitude", 1.0))
            frequency = float(params.get("frequency", 1.0))
            offset = float(params.get("offset", 0.0))
            return offset + amplitude * math.sin(2 * math.pi * frequency * t)

        if kind == "ramp":
            start = float(params.get("start", 0.0))
            rate = float(params.get("rate", 1.0))
            value = start + rate * t
            max_v = params.get("max_value")
            if max_v is not None:
                value = min(value, float(max_v))
            return value

        if kind == "replay_csv":
            raise NotImplementedError(
                "replay_csv requires the csv_sources dict — call apply_pre_tick via the instance."
            )

        raise ValueError(f"Unknown kind {kind!r}")

    def _simulate_replay(self, params: dict, tick_count: int) -> Any:
        source_id = params.get("source_id", "")
        rows = self._csv_sources.get(source_id)
        if not rows:
            return 0
        return rows[tick_count % len(rows)]

    # Override apply_pre_tick logic for replay_csv (needs self._csv_sources)
    def _compute_value(self, kind: str, params: dict, tick_count: int, t: float) -> Any:
        if kind == "replay_csv":
            return self._simulate_replay(params, tick_count)
        return self._simulate(kind, params, tick_count, t)

    def apply_pre_tick(self, tick_count: int, dt: float = 1.0) -> None:  # noqa: F811
        """Write simulated values into the instance manager before the tick."""
        t = tick_count * dt
        for (class_name, instance_id, attribute), cfg in list(self._bindings.items()):
            try:
                value = self._compute_value(cfg["kind"], cfg["params"], tick_count, t)
            except Exception:
                continue
            try:
                inst = self._im.get_instance(class_name, instance_id)
                new_attrs = dict(inst["attributes"])
                new_attrs[attribute] = value
                self._im.update_instance(class_name, instance_id, new_attrs)
            except Exception:
                continue

    # ---------------------------------------------------- persistence

    def save(self, path: str | Path) -> None:
        """Persist bindings and CSV data to a JSON file."""
        data = {
            "bindings": [
                {"class_name": cn, "instance_id": iid, "attribute": attr, **cfg}
                for (cn, iid, attr), cfg in self._bindings.items()
            ],
            "csv_sources": self._csv_sources,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        """Load bindings from a JSON file written by :meth:`save`."""
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        self._bindings = {}
        for entry in data.get("bindings", []):
            key = (entry["class_name"], entry["instance_id"], entry["attribute"])
            self._bindings[key] = {"kind": entry["kind"], "params": entry.get("params", {})}
        self._csv_sources = data.get("csv_sources", {})
