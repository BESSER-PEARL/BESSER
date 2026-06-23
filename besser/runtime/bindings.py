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

``mqtt``
    Subscribe to an MQTT topic and cache the latest received value; that
    cached value is written pre-tick, overriding the isStep computation for
    this attribute.  If no message has been received yet the attribute is
    left unchanged (no write).
    params: ``{"host": str, "port": int=1883, "topic": str}``
    Requires ``paho-mqtt`` to be installed.  If paho is not available,
    ``set_binding`` raises ``ImportError`` at bind time.

Persistence
-----------
Call ``InputBindings.save(path)`` / ``InputBindings.load(path)`` to persist
the binding table as JSON so bindings survive a server restart.  Replayed CSV
data is stored inline in the same JSON file.  MQTT subscriptions are
re-established automatically on ``load()``.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Sentinel returned by _compute_value for an mqtt binding that has not yet
# received any payload — instructs apply_pre_tick to skip the write.
_UNSET = object()


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
        # MQTT support (lazy-imported when first mqtt binding is created)
        # { (host, port): paho.Client }
        self._mqtt_clients: dict[tuple[str, int], Any] = {}
        # { (host, port, topic): latest_value }
        self._mqtt_cache: dict[tuple[str, int, str], Any] = {}

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
            Simulator kind — ``"constant"``, ``"sine"``, ``"ramp"``,
            ``"replay_csv"``, or ``"mqtt"``.
        params:
            Simulator parameters dict (kind-specific; see module docstring).
        """
        _VALID_KINDS = {"constant", "sine", "ramp", "replay_csv", "mqtt"}
        if kind not in _VALID_KINDS:
            raise ValueError(f"Unknown binding kind {kind!r}. Choose from {_VALID_KINDS}.")
        params = params or {}

        # If replacing an existing mqtt binding, unsubscribe the old topic first.
        existing = self._bindings.get((class_name, instance_id, attribute))
        if existing and existing["kind"] == "mqtt":
            self._unsubscribe_mqtt(existing["params"])

        key = (class_name, instance_id, attribute)
        self._bindings[key] = {"kind": kind, "params": params}

        if kind == "mqtt":
            self._subscribe_mqtt(params)

    def clear_binding(self, class_name: str, instance_id: str, attribute: str) -> None:
        """Remove the binding for the given slot (no-op if not set)."""
        existing = self._bindings.pop((class_name, instance_id, attribute), None)
        if existing and existing["kind"] == "mqtt":
            self._unsubscribe_mqtt(existing["params"])

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
        """Write simulated values into the instance manager before the tick."""
        t = tick_count * dt
        for (class_name, instance_id, attribute), cfg in list(self._bindings.items()):
            try:
                value = self._compute_value(cfg["kind"], cfg["params"], tick_count, t)
            except Exception:
                continue
            # _UNSET means no MQTT payload received yet — preserve current value.
            if value is _UNSET:
                continue
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

    def _compute_value(self, kind: str, params: dict, tick_count: int, t: float) -> Any:
        if kind == "replay_csv":
            return self._simulate_replay(params, tick_count)
        if kind == "mqtt":
            host = params.get("host", "localhost")
            port = int(params.get("port", 1883))
            topic = params.get("topic", "")
            return self._mqtt_cache.get((host, port, topic), _UNSET)
        return self._simulate(kind, params, tick_count, t)

    # --------------------------------------------------------- MQTT internals

    def _ensure_mqtt(self, host: str, port: int) -> Any:
        """Return (and lazily create) the paho Client for (host, port)."""
        key = (host, port)
        if key in self._mqtt_clients:
            return self._mqtt_clients[key]

        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise ImportError(
                "paho-mqtt is required for mqtt bindings. Install it with: pip install paho-mqtt"
            ) from exc

        client = mqtt.Client(client_id="", clean_session=True)

        # Capture references for the closure.
        cache = self._mqtt_cache
        h, p = host, port

        def on_message(c, userdata, msg):  # noqa: ARG001
            try:
                value = _coerce(msg.payload)
                cache[(h, p, msg.topic)] = value
            except Exception:
                pass

        def on_connect(c, userdata, flags, rc):  # noqa: ARG001
            if rc == 0:
                logger.info("MQTT connected to %s:%d", h, p)
                # Re-subscribe all topics for this broker on reconnect.
                for (bh, bp, btopic) in list(cache.keys()):
                    if bh == h and bp == p:
                        c.subscribe(btopic)
            else:
                logger.warning("MQTT connect failed rc=%d for %s:%d", rc, h, p)

        client.on_message = on_message
        client.on_connect = on_connect

        try:
            client.connect_async(host, port, keepalive=60)
            client.loop_start()
        except Exception as exc:
            logger.warning("MQTT connect_async failed for %s:%d — %s", host, port, exc)

        self._mqtt_clients[key] = client
        return client

    def _subscribe_mqtt(self, params: dict) -> None:
        """Subscribe to the topic described by params."""
        host = params.get("host", "localhost")
        port = int(params.get("port", 1883))
        topic = params.get("topic", "")
        if not topic:
            return
        try:
            client = self._ensure_mqtt(host, port)
            client.subscribe(topic)
            logger.debug("MQTT subscribed: %s:%d %s", host, port, topic)
        except Exception as exc:
            logger.warning("MQTT subscribe failed (%s:%d %s): %s", host, port, topic, exc)

    def _unsubscribe_mqtt(self, params: dict) -> None:
        """Unsubscribe the topic described by params (best-effort)."""
        host = params.get("host", "localhost")
        port = int(params.get("port", 1883))
        topic = params.get("topic", "")
        if not topic:
            return
        key = (host, port)
        client = self._mqtt_clients.get(key)
        if client:
            try:
                client.unsubscribe(topic)
            except Exception:
                pass
        self._mqtt_cache.pop((host, port, topic), None)

    def shutdown(self) -> None:
        """Disconnect all MQTT clients. Call on application shutdown."""
        for (host, port), client in list(self._mqtt_clients.items()):
            try:
                client.loop_stop()
                client.disconnect()
                logger.info("MQTT disconnected from %s:%d", host, port)
            except Exception as exc:
                logger.debug("MQTT disconnect error for %s:%d: %s", host, port, exc)
        self._mqtt_clients.clear()
        self._mqtt_cache.clear()

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
        """Load bindings from a JSON file written by :meth:`save`.

        MQTT subscriptions are re-established automatically.
        """
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        self._bindings = {}
        for entry in data.get("bindings", []):
            key = (entry["class_name"], entry["instance_id"], entry["attribute"])
            self._bindings[key] = {"kind": entry["kind"], "params": entry.get("params", {})}
        self._csv_sources = data.get("csv_sources", {})
        # Re-subscribe all persisted MQTT bindings.
        for cfg in self._bindings.values():
            if cfg["kind"] == "mqtt":
                self._subscribe_mqtt(cfg["params"])


def _coerce(payload: bytes) -> Any:
    """Coerce a raw MQTT payload bytes to a Python scalar.

    Tries JSON first (handles numbers, bools, strings, null), then float,
    then falls back to a decoded UTF-8 string.
    """
    text = payload.decode("utf-8", errors="replace").strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text
