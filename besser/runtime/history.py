"""HistoryStore — time-series snapshot store for Digital Twin platforms.

Every tick the scheduler calls :meth:`HistoryStore.snapshot`, which records
one data point per primitive attribute per steppable instance.  Frontend charts
query the ``/history`` endpoint, which in turn calls :meth:`HistoryStore.query`.

Backend selection
-----------------
When ``influxdb-client`` is installed **and** the environment variable
``INFLUXDB_URL`` is set, the store writes to **InfluxDB 2.x** and queries it
with Flux.  This is the mode used in the Docker-Compose deployment.

When InfluxDB is not available (unit tests, the BESSER editor backend, a
developer's machine without Docker), the store falls back to an **in-memory
ring buffer** that behaves identically from the caller's perspective.  The
``BESSER_HISTORY_MAXPOINTS`` environment variable caps the ring (default 50 000
rows; fine for a demo session).

InfluxDB configuration (environment variables)
-----------------------------------------------
``INFLUXDB_URL``      — e.g. ``http://influxdb:8086``  (required to enable InfluxDB)
``INFLUXDB_TOKEN``    — API token  (default ``"besser-token"``)
``INFLUXDB_ORG``      — organisation name  (default ``"besser"``)
``INFLUXDB_BUCKET``   — bucket name  (default ``"dt_history"``)

Schema (InfluxDB)
-----------------
Measurement = ``class_name``
Tags         = ``instance_id``, ``attribute``
Fields       = ``value`` (float or string), ``tick_id`` (int)
Time         = wall-clock nanoseconds at snapshot time

Schema (in-memory fallback)
----------------------------
A plain list of dicts ``{"tick_id", "ts", "class_name", "instance_id",
"attribute", "value"}``, capped at ``BESSER_HISTORY_MAXPOINTS``.

Public API (identical for both backends)
-----------------------------------------
``snapshot(tick_count)``                                    — write one tick
``query(class_name, instance_id, attribute,
        since_tick=0, limit=1000)``                        — read back
``clear()``                                                 — delete all data
``flush()``                                                 — ensure writes committed
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MAXPOINTS = 50_000


def _use_influxdb() -> bool:
    """Return True if InfluxDB is configured and influxdb-client is installed."""
    if not os.environ.get("INFLUXDB_URL", "").strip():
        return False
    try:
        import influxdb_client  # noqa: F401
        return True
    except ImportError:
        return False


class HistoryStore:
    """Append-only time-series store for per-tick instance attribute snapshots.

    Parameters
    ----------
    instance_manager:
        Platform InstanceManager singleton; used to iterate instances at
        snapshot time.
    steppable_classes:
        Dict ``{class_name: method_name}`` or set of class names whose
        instances should be recorded.  Pass the same value as the scheduler's
        ``steppable_classes`` argument.  When ``None`` (default) every instance
        is recorded (backward-compatible with older generated bootstraps).
    """

    def __init__(
        self,
        instance_manager,
        steppable_classes: dict | set | None = None,
        db_path: str | None = None,  # accepted for backward-compatibility; ignored (InfluxDB / in-memory used instead)
    ):
        self._im = instance_manager
        # Normalise to a set of class names for O(1) membership checks.
        if steppable_classes is None:
            self._steppable: set[str] | None = None  # record everything
        elif isinstance(steppable_classes, dict):
            self._steppable = set(steppable_classes.keys())
        else:
            self._steppable = set(steppable_classes)

        self._use_influx = _use_influxdb()
        if self._use_influx:
            self._init_influxdb()
        else:
            maxpts = int(os.environ.get("BESSER_HISTORY_MAXPOINTS", _DEFAULT_MAXPOINTS))
            self._buffer: list[dict] = []
            self._maxpoints = maxpts

    # --------------------------------------------------------- InfluxDB init

    def _init_influxdb(self) -> None:
        from influxdb_client import InfluxDBClient, WriteOptions
        from influxdb_client.client.write_api import SYNCHRONOUS

        url = os.environ["INFLUXDB_URL"]
        token = os.environ.get("INFLUXDB_TOKEN", "besser-token")
        org = os.environ.get("INFLUXDB_ORG", "besser")
        bucket = os.environ.get("INFLUXDB_BUCKET", "dt_history")

        self._influx_client = InfluxDBClient(url=url, token=token, org=org)
        # Use synchronous writes for simplicity (batching can be added later).
        self._write_api = self._influx_client.write_api(write_options=SYNCHRONOUS)
        self._query_api = self._influx_client.query_api()
        self._delete_api = self._influx_client.delete_api()
        self._influx_org = org
        self._influx_bucket = bucket
        logger.info("HistoryStore: connected to InfluxDB at %s, bucket=%s", url, bucket)

    # --------------------------------------------------------- write

    def snapshot(self, tick_count: int) -> None:
        """Write a snapshot of every steppable instance's primitive attributes."""
        ts = time.time()

        if self._use_influx:
            self._snapshot_influx(tick_count, ts)
        else:
            self._snapshot_buffer(tick_count, ts)

    def _should_record(self, class_name: str) -> bool:
        """Return True if this class's instances should be recorded."""
        if self._steppable is None:
            return True
        return class_name in self._steppable

    def _snapshot_influx(self, tick_count: int, ts: float) -> None:
        from influxdb_client.client.write_api import Point

        points = []
        for inst in self._im.get_all_instances():
            cn = inst["class_name"]
            if not self._should_record(cn):
                continue
            iid = inst["id"]
            for attr, val in inst.get("attributes", {}).items():
                if not (isinstance(val, (int, float, str, bool)) or val is None):
                    continue
                p = (
                    Point(cn)
                    .tag("instance_id", iid)
                    .tag("attribute", attr)
                    .field("tick_id", tick_count)
                )
                # InfluxDB fields must be typed; store numerics as float,
                # booleans as int (Flux handles it), strings as string.
                if isinstance(val, bool):
                    p = p.field("value_bool", int(val))
                elif isinstance(val, (int, float)):
                    p = p.field("value", float(val))
                elif isinstance(val, str):
                    p = p.field("value_str", val)
                else:
                    p = p.field("value_str", "null")
                # Use wall-clock time in nanoseconds.
                p = p.time(int(ts * 1e9))
                points.append(p)

        if points:
            try:
                self._write_api.write(bucket=self._influx_bucket, record=points)
            except Exception as exc:
                logger.warning("HistoryStore: InfluxDB write error at tick %d: %s", tick_count, exc)

    def _snapshot_buffer(self, tick_count: int, ts: float) -> None:
        import json
        for inst in self._im.get_all_instances():
            cn = inst["class_name"]
            if not self._should_record(cn):
                continue
            iid = inst["id"]
            for attr, val in inst.get("attributes", {}).items():
                if isinstance(val, (int, float, str, bool)) or val is None:
                    self._buffer.append({
                        "tick_id": tick_count,
                        "ts": ts,
                        "class_name": cn,
                        "instance_id": iid,
                        "attribute": attr,
                        "value": val,
                    })
        # Trim to keep memory bounded.
        if len(self._buffer) > self._maxpoints:
            self._buffer = self._buffer[-self._maxpoints:]

    # --------------------------------------------------------- read

    def query(
        self,
        class_name: str,
        instance_id: str,
        attribute: str,
        since_tick: int = 0,
        limit: int = 1000,
    ) -> list[dict]:
        """Return up to ``limit`` snapshots for the given attribute slot.

        Each entry: ``{"tick_id": int, "ts": float, "value": Any}``
        """
        if self._use_influx:
            return self._query_influx(class_name, instance_id, attribute, since_tick, limit)
        return self._query_buffer(class_name, instance_id, attribute, since_tick, limit)

    def _query_influx(
        self,
        class_name: str,
        instance_id: str,
        attribute: str,
        since_tick: int,
        limit: int,
    ) -> list[dict]:
        bucket = self._influx_bucket
        # Flux query without pivot - fetch all fields and group by timestamp
        flux = f"""
from(bucket: "{bucket}")
  |> range(start: 0)
  |> filter(fn: (r) => r._measurement == "{class_name}")
  |> filter(fn: (r) => r.instance_id == "{instance_id}")
  |> filter(fn: (r) => r.attribute == "{attribute}")
  |> filter(fn: (r) => r._field == "tick_id" or r._field == "value" or r._field == "value_str" or r._field == "value_bool")
  |> sort(columns: ["_time"])
"""
        try:
            tables = self._query_api.query(flux, org=self._influx_org)
        except Exception as exc:
            logger.warning("HistoryStore: InfluxDB query error: %s", exc)
            return []

        # Group records by timestamp to pair tick_id with value fields
        by_time: dict[float, dict] = {}
        for table in tables:
            for record in table.records:
                ts = record.get_time().timestamp() if record.get_time() else 0.0
                if ts not in by_time:
                    by_time[ts] = {"ts": ts, "tick_id": None, "value": None}
                
                field = record.get_field()
                value = record.get_value()
                
                if field == "tick_id":
                    by_time[ts]["tick_id"] = int(value) if value is not None else 0
                elif field == "value" and value is not None:
                    by_time[ts]["value"] = value
                elif field == "value_bool" and value is not None:
                    by_time[ts]["value"] = bool(value)
                elif field == "value_str":
                    by_time[ts]["value"] = None if value == "null" else value

        # Filter, sort, and apply limit
        results = []
        for entry in by_time.values():
            tick_id = entry.get("tick_id")
            if tick_id is None or tick_id < since_tick:
                continue
            if entry.get("value") is None:
                continue
            results.append({
                "tick_id": tick_id,
                "ts": entry["ts"],
                "value": entry["value"]
            })

        # Sort by tick_id and apply limit
        results.sort(key=lambda x: x["tick_id"])
        return results[-limit:] if len(results) > limit else results

    def _query_buffer(
        self,
        class_name: str,
        instance_id: str,
        attribute: str,
        since_tick: int,
        limit: int,
    ) -> list[dict]:
        rows = [
            r for r in self._buffer
            if (
                r["class_name"] == class_name
                and r["instance_id"] == instance_id
                and r["attribute"] == attribute
                and r["tick_id"] >= since_tick
            )
        ]
        # Return chronological slice capped at limit.
        return [
            {"tick_id": r["tick_id"], "ts": r["ts"], "value": r["value"]}
            for r in rows[-limit:]
        ]

    # --------------------------------------------------------- management

    def clear(self) -> None:
        """Delete all recorded ticks and snapshots (used by /reset)."""
        if self._use_influx:
            try:
                from datetime import datetime, timezone
                start = "1970-01-01T00:00:00Z"
                stop = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                self._delete_api.delete(
                    start, stop,
                    predicate="",
                    bucket=self._influx_bucket,
                    org=self._influx_org,
                )
            except Exception as exc:
                logger.warning("HistoryStore: InfluxDB clear error: %s", exc)
        else:
            self._buffer.clear()

    def flush(self) -> None:
        """Ensure all pending writes are committed (called on shutdown)."""
        if self._use_influx:
            try:
                self._write_api.flush()
            except Exception as exc:
                logger.debug("HistoryStore: InfluxDB flush error: %s", exc)
            try:
                self._write_api.close()
                self._influx_client.close()
            except Exception:
                pass
        # In-memory backend: nothing to flush.

    @property
    def db_path(self) -> str:
        """Kept for backward compatibility. Returns a descriptive string."""
        if self._use_influx:
            return f"influxdb://{os.environ.get('INFLUXDB_URL', '?')}/{self._influx_bucket}"
        return ":memory:"
