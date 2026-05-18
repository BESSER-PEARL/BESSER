"""Phase 1.0.8 — direct tests for the runtime engine's link resolution.

Bypasses the editor/converter layer and exercises the Engine with a
hand-built ``DomainModel`` plus a minimal in-memory instance store.
That's the right test boundary for the kernel changes: it isolates the
graph-materialisation logic from converter quirks (e.g. the editor's
inability to disambiguate two associations connecting the same class
pair without an explicit ``associationId``).

The scenarios mirror what a hydrogen plant DT needs:
    Tank.step(dt) reads neighbour state from connected Ports and Streams
    and integrates inflow − outflow into ``level``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    FloatType,
    Method,
    Multiplicity,
    Parameter,
    Property,
    StringType,
)
from besser.runtime.engine import Engine
from besser.runtime.materialize import build_associations_index, materialize_classes


# ---------------------------------------------------------------------------
# Minimal instance store mirroring the generated platform's contract
# ---------------------------------------------------------------------------
class _Store:
    """Drop-in replacement for ``services.instance_manager`` used by the
    Engine. Same shape (get/get_all/update/links) but without any of the
    generated-platform infrastructure (FastAPI, UUIDs, etc.). Lets tests
    build a clean object graph in three lines."""

    def __init__(self) -> None:
        self._rows: Dict[tuple, Dict[str, Any]] = {}

    def add(
        self,
        class_name: str,
        instance_id: str,
        attributes: Dict[str, Any],
        links: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        self._rows[(class_name, instance_id)] = {
            "id": instance_id,
            "class_name": class_name,
            "instance_name": instance_id,
            "attributes": dict(attributes),
            "links": list(links or []),
        }

    def get_instance(self, class_name: str, instance_id: str) -> Optional[Dict[str, Any]]:
        return self._rows.get((class_name, instance_id))

    def get_all_instances(self) -> List[Dict[str, Any]]:
        return list(self._rows.values())

    def update_instance(
        self,
        class_name: str,
        instance_id: str,
        attributes: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        row = self._rows.get((class_name, instance_id))
        if row is None:
            return None
        row["attributes"].update(attributes)
        return row


# ---------------------------------------------------------------------------
# Tank/Port/Stream domain model — exactly the shape of the user's H2 plant
# (with role names matching what they'll set in the editor for Phase 1.0.8).
# ---------------------------------------------------------------------------
@pytest.fixture
def h2_domain_model() -> DomainModel:
    # Classes
    tank = Class(
        name="Tank",
        attributes={Property(name="level", type=FloatType)},
    )
    port = Class(
        name="Port",
        attributes={Property(name="direction", type=StringType)},
    )
    stream = Class(
        name="MaterialStream",
        attributes={Property(name="massFlow", type=FloatType)},
    )

    # Tank ── equi_ports ──> Port  (mult 1 / * with roles equipment / ports)
    equi_ports = BinaryAssociation(
        name="equi_ports",
        ends={
            Property(
                name="equipment",
                type=tank,
                multiplicity=Multiplicity(min_multiplicity=1, max_multiplicity=1),
            ),
            Property(
                name="ports",
                type=port,
                multiplicity=Multiplicity(min_multiplicity=0, max_multiplicity=9999),
            ),
        },
    )
    # Stream ── source ──> Port (Stream-side role outgoingStreams)
    source_assoc = BinaryAssociation(
        name="source",
        ends={
            Property(
                name="outgoingStreams",
                type=stream,
                multiplicity=Multiplicity(min_multiplicity=0, max_multiplicity=9999),
            ),
            Property(
                name="sourcePort",
                type=port,
                multiplicity=Multiplicity(min_multiplicity=1, max_multiplicity=1),
            ),
        },
    )
    # Stream ── target ──> Port (Stream-side role incomingStreams)
    target_assoc = BinaryAssociation(
        name="target",
        ends={
            Property(
                name="incomingStreams",
                type=stream,
                multiplicity=Multiplicity(min_multiplicity=0, max_multiplicity=9999),
            ),
            Property(
                name="targetPort",
                type=port,
                multiplicity=Multiplicity(min_multiplicity=1, max_multiplicity=1),
            ),
        },
    )

    # Tank.step(dt) — integrates flows from connected streams over dt seconds.
    step = Method(
        name="step",
        parameters=[Parameter(name="dt", type=FloatType)],
        code=(
            "inflow = 0.0\n"
            "outflow = 0.0\n"
            "for p in (self.ports or []):\n"
            "    direction = str(getattr(p, 'direction', '')).strip().lower()\n"
            "    if direction == 'inlet':\n"
            "        for s in (p.incomingStreams or []):\n"
            "            inflow += float(s.massFlow or 0.0)\n"
            "    elif direction == 'outlet':\n"
            "        for s in (p.outgoingStreams or []):\n"
            "            outflow += float(s.massFlow or 0.0)\n"
            "self.level = max(0.0, min(100.0, (self.level or 0.0) + (inflow - outflow) * dt))\n"
        ),
    )
    tank.methods = {step}

    return DomainModel(
        name="TinyH2",
        types={tank, port, stream},
        associations={equi_ports, source_assoc, target_assoc},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestLinkResolution:
    def test_tank_step_integrates_inflow_minus_outflow_via_neighbours(self, h2_domain_model):
        """The headline scenario from the user's hydrogen DT.

        Tank with two ports; inlet port linked to a Stream(massFlow=5),
        outlet port linked to a Stream(massFlow=2). ``step(dt=2)``
        should read both streams via the resolved object graph and
        update ``tank.level`` to ``10 + (5-2)*2 = 16``."""
        store = _Store()
        store.add("Tank", "Tank_A", {"level": 10.0}, [
            {"association_name": "equi_ports", "target_class": "Port", "target_id": "Port_In"},
            {"association_name": "equi_ports", "target_class": "Port", "target_id": "Port_Out"},
        ])
        store.add("Port", "Port_In", {"direction": "Inlet"})
        store.add("Port", "Port_Out", {"direction": "Outlet"})
        # Streams hold the outgoing link (Stream.target = Port_In means
        # the stream flows INTO that port — i.e. the port is an inlet).
        store.add("MaterialStream", "Stream_In", {"massFlow": 5.0}, [
            {"association_name": "target", "target_class": "Port", "target_id": "Port_In"},
        ])
        store.add("MaterialStream", "Stream_Out", {"massFlow": 2.0}, [
            {"association_name": "source", "target_class": "Port", "target_id": "Port_Out"},
        ])

        class_map = materialize_classes(h2_domain_model)
        engine = Engine(
            instance_manager=store,
            class_map=class_map,
            associations_index=build_associations_index(h2_domain_model),
        )
        res = engine.invoke("Tank", "Tank_A", "step", {"dt": 2.0})

        assert res["instance"]["attributes"]["level"] == 16.0
        # Round-trip: re-read the store; the writeback path landed.
        assert store.get_instance("Tank", "Tank_A")["attributes"]["level"] == 16.0

    def test_neighbour_mutation_persists(self, h2_domain_model):
        """When a method mutates a neighbour's attribute, the writeback
        walk must persist it. Adds a method that closes one of the
        streams (sets massFlow to 0) — confirms the change lands."""
        # Add a method that touches the neighbour
        tank_cls = next(c for c in h2_domain_model.get_classes() if c.name == "Tank")
        close_in = Method(
            name="closeInlet",
            parameters=[],
            code=(
                "for p in (self.ports or []):\n"
                "    if str(getattr(p, 'direction', '')).strip().lower() == 'inlet':\n"
                "        for s in (p.incomingStreams or []):\n"
                "            s.massFlow = 0.0\n"
            ),
        )
        tank_cls.methods = tank_cls.methods | {close_in}

        store = _Store()
        store.add("Tank", "Tank_A", {"level": 10.0}, [
            {"association_name": "equi_ports", "target_class": "Port", "target_id": "Port_In"},
        ])
        store.add("Port", "Port_In", {"direction": "Inlet"})
        store.add("MaterialStream", "Stream_In", {"massFlow": 5.0}, [
            {"association_name": "target", "target_class": "Port", "target_id": "Port_In"},
        ])

        class_map = materialize_classes(h2_domain_model)
        engine = Engine(
            instance_manager=store,
            class_map=class_map,
            associations_index=build_associations_index(h2_domain_model),
        )
        engine.invoke("Tank", "Tank_A", "closeInlet", {})

        # Neighbour mutation landed in the store
        assert store.get_instance("MaterialStream", "Stream_In")["attributes"]["massFlow"] == 0.0

    def test_no_links_gives_empty_collections_not_attribute_error(self, h2_domain_model):
        """A Tank with no linked ports must NOT AttributeError when the
        method body iterates ``self.ports``. The engine pre-initialises
        association slots to ``[]`` / ``None`` based on multiplicity."""
        store = _Store()
        store.add("Tank", "Lonely", {"level": 50.0})  # no links

        class_map = materialize_classes(h2_domain_model)
        engine = Engine(
            instance_manager=store,
            class_map=class_map,
            associations_index=build_associations_index(h2_domain_model),
        )
        res = engine.invoke("Tank", "Lonely", "step", {"dt": 1.0})
        # No flows → level unchanged
        assert res["instance"]["attributes"]["level"] == 50.0

    def test_bidirectional_nav_works_from_either_side(self, h2_domain_model):
        """The Tank holds the equi_ports link; from Tank, ``self.ports``
        must resolve. From Port, the reverse navigation ``port.equipment``
        must also resolve back to the Tank via the engine's reverse-link
        index."""
        # Add a Port method that walks back to its equipment
        port_cls = next(c for c in h2_domain_model.get_classes() if c.name == "Port")
        echo_owner = Method(
            name="echoOwnerLevel",
            parameters=[],
            # Use a public attribute name — the engine filters underscore-
            # prefixed slots from writeback (treats them as Python internals).
            code="self.lastSeenLevel = float(getattr(self.equipment, 'level', -1.0))",
        )
        port_cls.methods = port_cls.methods | {echo_owner}

        store = _Store()
        store.add("Tank", "Tank_A", {"level": 42.0}, [
            {"association_name": "equi_ports", "target_class": "Port", "target_id": "P1"},
        ])
        store.add("Port", "P1", {"direction": "Inlet"})

        class_map = materialize_classes(h2_domain_model)
        engine = Engine(
            instance_manager=store,
            class_map=class_map,
            associations_index=build_associations_index(h2_domain_model),
        )
        engine.invoke("Port", "P1", "echoOwnerLevel", {})

        # The Port saw its Tank.level = 42 via reverse navigation
        assert store.get_instance("Port", "P1")["attributes"]["lastSeenLevel"] == 42.0

    def test_cycle_does_not_blow_the_stack(self, h2_domain_model):
        """Bidirectional links form a cycle (Tank → Port via equi_ports;
        Port → Tank via reverse-nav). The visited-set in
        ``_materialize_graph`` must break the cycle, not stack-overflow."""
        store = _Store()
        store.add("Tank", "Tank_A", {"level": 0.0}, [
            {"association_name": "equi_ports", "target_class": "Port", "target_id": "P1"},
        ])
        store.add("Port", "P1", {"direction": "Inlet"})

        class_map = materialize_classes(h2_domain_model)
        engine = Engine(
            instance_manager=store,
            class_map=class_map,
            associations_index=build_associations_index(h2_domain_model),
        )
        # Should not recurse forever. Just confirming it returns at all.
        res = engine.invoke("Tank", "Tank_A", "step", {"dt": 1.0})
        assert res["method_name"] == "step"
