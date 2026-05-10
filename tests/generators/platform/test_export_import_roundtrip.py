"""Round-trip tests for the generated platform's export/import services.

Phase 0 of besser4DT is the persistence bridge: an operator must be able to
save a plant out of the generated app and load it back. This file exercises
the end-to-end loop by generating a real platform tree, dynamically importing
the emitted ``services.instance_manager`` + ``services.export_service``, and
asserting that ``export → clear → import`` is identity for the captured
state (ids, attributes, links).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    Multiplicity,
    Property,
    StringType,
)
from besser.generators.platform.platform_generator import PlatformGenerator


# ---------------------------------------------------------------------------
# Domain model fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def region_sensor_model() -> DomainModel:
    """Tiny two-class model with one association — enough to round-trip both
    instances and links without dragging in the customization machinery."""
    region = Class(name="Region", attributes={Property(name="label", type=StringType)})
    sensor = Class(name="Sensor", attributes={Property(name="kind", type=StringType)})
    has = BinaryAssociation(
        name="has",
        ends={
            Property(
                name="region",
                type=region,
                multiplicity=Multiplicity(min_multiplicity=1, max_multiplicity=1),
            ),
            Property(
                name="sensors",
                type=sensor,
                multiplicity=Multiplicity(min_multiplicity=0, max_multiplicity=9999),
            ),
        },
    )
    return DomainModel(name="RegionSensor", types={region, sensor}, associations={has})


# ---------------------------------------------------------------------------
# Dynamic-import helpers
# ---------------------------------------------------------------------------
# The generated backend lays out ``services/`` and ``metamodel/`` as packages
# under a ``backend/`` directory. To run the real services in-process we put
# that backend directory on sys.path, import the modules, and then carefully
# undo both side effects so subsequent tests don't see stale imports.
_PLATFORM_PACKAGES = ("services", "metamodel")


def _drop_platform_modules() -> None:
    for mod_name in list(sys.modules):
        if mod_name in _PLATFORM_PACKAGES or mod_name.startswith(
            tuple(f"{p}." for p in _PLATFORM_PACKAGES)
        ):
            del sys.modules[mod_name]


def _import_platform_services(backend_dir: Path):
    backend_str = str(backend_dir)
    # Clear any prior tests' cached modules first — the package names collide
    # across generated platforms even though the on-disk paths differ.
    _drop_platform_modules()
    sys.path.insert(0, backend_str)
    try:
        im_mod = importlib.import_module("services.instance_manager")
        exp_mod = importlib.import_module("services.export_service")
        return im_mod, exp_mod
    except Exception:
        sys.path.remove(backend_str)
        _drop_platform_modules()
        raise


def _cleanup_imports(backend_dir: Path) -> None:
    backend_str = str(backend_dir)
    if backend_str in sys.path:
        sys.path.remove(backend_str)
    _drop_platform_modules()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestExportImportRoundTrip:
    def test_roundtrip_preserves_instances_and_links(self, tmp_path, region_sensor_model):
        """export_to_json → clear → import_from_json must be identity for
        ids, attributes, links, AND positions — that's the contract the
        runtime kernel will rely on later."""
        out = tmp_path / "platform"
        PlatformGenerator(region_sensor_model, customization=None, output_dir=str(out)).generate()
        backend_dir = out / "backend"
        try:
            im_mod, exp_mod = _import_platform_services(backend_dir)
            im = im_mod.instance_manager
            exp = exp_mod.export_service

            # Populate: 1 region, 2 sensors at distinct positions, both linked
            # from the region. The positions matter — a DT's layout is part
            # of the plant identity, not just decoration.
            r = im.create_instance("Region", "alpha", {"label": "North"}, position={"x": 120.0, "y": 80.0})
            s1 = im.create_instance("Sensor", "thermo_1", {"kind": "temperature"}, position={"x": 400.0, "y": 220.0})
            s2 = im.create_instance("Sensor", "thermo_2", {"kind": "pressure"}, position={"x": 400.0, "y": 360.0})
            im.create_link("Region", r["id"], "Sensor", s1["id"], "has")
            im.create_link("Region", r["id"], "Sensor", s2["id"], "has")

            def snapshot():
                """Sorted, attribute-comparable view of the current store."""
                rows = sorted(im.get_all_instances(), key=lambda i: i["id"])
                return [
                    {
                        "id": i["id"],
                        "class_name": i["class_name"],
                        "instance_name": i["instance_name"],
                        "attributes": dict(i["attributes"]),
                        "links": sorted(
                            (link["association_name"], link["target_class"], link["target_id"])
                            for link in i["links"]
                        ),
                        "position": i.get("position"),
                    }
                    for i in rows
                ]

            before = snapshot()
            payload = exp.export_to_json()

            im.clear_all()
            assert im.get_all_instances() == []

            result = exp.import_from_json(payload)
            assert result == {"instances": 3, "links": 2}

            after = snapshot()
            assert after == before
            # Spot-check: positions specifically survived the round trip.
            positions_after = {row["id"]: row["position"] for row in after}
            assert positions_after[r["id"]] == {"x": 120.0, "y": 80.0}
            assert positions_after[s1["id"]] == {"x": 400.0, "y": 220.0}

        finally:
            _cleanup_imports(backend_dir)

    def test_update_instance_position(self, tmp_path, region_sensor_model):
        """The drag-end PATCH path: update_instance_position mutates the
        stored position in place, returning the updated row."""
        out = tmp_path / "platform_dragstop"
        PlatformGenerator(region_sensor_model, customization=None, output_dir=str(out)).generate()
        backend_dir = out / "backend"
        try:
            im_mod, _ = _import_platform_services(backend_dir)
            im = im_mod.instance_manager

            inst = im.create_instance("Region", "alpha", {"label": "North"})
            assert inst["position"] is None

            updated = im.update_instance_position("Region", inst["id"], {"x": 555.5, "y": 222.25})
            assert updated is not None
            assert updated["position"] == {"x": 555.5, "y": 222.25}

            # Same object in storage was mutated — no clone.
            stored = im.get_instance("Region", inst["id"])
            assert stored["position"] == {"x": 555.5, "y": 222.25}

            # Unknown id → None, no exception.
            assert im.update_instance_position("Region", "nonexistent", {"x": 0.0, "y": 0.0}) is None

        finally:
            _cleanup_imports(backend_dir)

    def test_import_rejects_unknown_class(self, tmp_path, region_sensor_model):
        """A typo in a class name should fail loudly and leave the store
        untouched (validation happens before any mutation)."""
        out = tmp_path / "platform_reject"
        PlatformGenerator(region_sensor_model, customization=None, output_dir=str(out)).generate()
        backend_dir = out / "backend"
        try:
            im_mod, exp_mod = _import_platform_services(backend_dir)
            im = im_mod.instance_manager
            exp = exp_mod.export_service

            # Seed something to prove import_state doesn't wipe it on failure.
            im.create_instance("Region", "alpha", {"label": "North"})

            bad_payload = {
                "model_name": "RegionSensor",
                "instances": {
                    "NotAClass": [
                        {
                            "id": "deadbeef",
                            "class_name": "NotAClass",
                            "instance_name": "x",
                            "attributes": {},
                            "links": [],
                        }
                    ]
                },
            }

            with pytest.raises(ValueError, match="Unknown class"):
                exp.import_from_json(bad_payload)

            # Pre-existing instance should still be there.
            remaining = im.get_all_instances()
            assert len(remaining) == 1
            assert remaining[0]["instance_name"] == "alpha"

        finally:
            _cleanup_imports(backend_dir)

    def test_export_to_buml_emits_links_section(self, tmp_path, region_sensor_model):
        """The BUML export previously skipped associations; the new section
        must mirror what export_to_python emits."""
        out = tmp_path / "platform_buml"
        PlatformGenerator(region_sensor_model, customization=None, output_dir=str(out)).generate()
        backend_dir = out / "backend"
        try:
            im_mod, exp_mod = _import_platform_services(backend_dir)
            im = im_mod.instance_manager
            exp = exp_mod.export_service

            r = im.create_instance("Region", "alpha", {"label": "North"})
            s = im.create_instance("Sensor", "thermo", {"kind": "temperature"})
            im.create_link("Region", r["id"], "Sensor", s["id"], "has")

            buml = exp.export_to_buml()
            # Header for the new section
            assert "Create associations (links) between objects" in buml
            # The link itself, using the buml var-name convention (_obj suffix)
            assert "alpha_obj.has = thermo_obj" in buml

        finally:
            _cleanup_imports(backend_dir)


class TestImportEndpointWired:
    """Confirm the router exposes POST /import/json — the wire contract
    between frontend Load and backend import_from_json."""

    def test_import_route_present_in_generated_router(self, tmp_path, region_sensor_model):
        out = tmp_path / "platform_router"
        PlatformGenerator(region_sensor_model, customization=None, output_dir=str(out)).generate()
        router_src = (out / "backend" / "routers" / "export.py").read_text(encoding="utf-8")
        assert '@router.post("/import/json")' in router_src
        assert "async def import_as_json" in router_src

    def test_frontend_import_helper_present(self, tmp_path, region_sensor_model):
        out = tmp_path / "platform_frontend"
        PlatformGenerator(region_sensor_model, customization=None, output_dir=str(out)).generate()
        api_src = (out / "frontend" / "src" / "api" / "instances.ts").read_text(encoding="utf-8")
        assert "export const importFromJson" in api_src
        assert "/export/import/json" in api_src

    def test_frontend_load_button_present(self, tmp_path, region_sensor_model):
        out = tmp_path / "platform_load_button"
        PlatformGenerator(region_sensor_model, customization=None, output_dir=str(out)).generate()
        app_src = (out / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
        assert "handleImportJsonFile" in app_src
        assert "Load a previously saved plant" in app_src
        # Hidden file input that drives the picker
        assert 'type="file"' in app_src


class TestPositionPersistence:
    """Phase 0.5: canvas position is a first-class part of an instance —
    persisted server-side so layouts survive refresh and round-trip."""

    def test_position_endpoint_present_in_router(self, tmp_path, region_sensor_model):
        out = tmp_path / "platform_position_router"
        PlatformGenerator(region_sensor_model, customization=None, output_dir=str(out)).generate()
        router_src = (out / "backend" / "routers" / "instances.py").read_text(encoding="utf-8")
        assert '@router.patch("/{class_name}/{instance_id}/position"' in router_src
        assert "async def update_instance_position" in router_src

    def test_schema_includes_position(self, tmp_path, region_sensor_model):
        out = tmp_path / "platform_position_schema"
        PlatformGenerator(region_sensor_model, customization=None, output_dir=str(out)).generate()
        schemas_src = (out / "backend" / "schemas" / "instance_schemas.py").read_text(encoding="utf-8")
        assert "class Position(BaseModel)" in schemas_src
        # InstanceCreate and InstanceResponse both carry optional Position
        assert "position: Optional[Position]" in schemas_src

    def test_frontend_types_include_position(self, tmp_path, region_sensor_model):
        out = tmp_path / "platform_position_types"
        PlatformGenerator(region_sensor_model, customization=None, output_dir=str(out)).generate()
        models_src = (out / "frontend" / "src" / "types" / "models.ts").read_text(encoding="utf-8")
        assert "export interface Position" in models_src
        assert "position?: Position" in models_src

    def test_frontend_api_has_update_position_helper(self, tmp_path, region_sensor_model):
        out = tmp_path / "platform_position_api"
        PlatformGenerator(region_sensor_model, customization=None, output_dir=str(out)).generate()
        api_src = (out / "frontend" / "src" / "api" / "instances.ts").read_text(encoding="utf-8")
        assert "export const updateInstancePosition" in api_src
        assert "/position" in api_src

    def test_canvas_wires_drag_stop_handler(self, tmp_path, region_sensor_model):
        out = tmp_path / "platform_position_canvas"
        PlatformGenerator(region_sensor_model, customization=None, output_dir=str(out)).generate()
        canvas_src = (out / "frontend" / "src" / "components" / "InstanceCanvas.tsx").read_text(encoding="utf-8")
        assert "onNodeDragStop" in canvas_src
        assert "updateInstancePosition" in canvas_src
        # Backend position is consulted on render so refresh restores layout
        assert "backendPosition" in canvas_src

    def test_canvas_prefers_backend_position_over_cache(self, tmp_path, region_sensor_model):
        """Regression: the in-memory position cache must NOT win over a
        freshly-imported backend position. Otherwise, after dragging nodes,
        loading a saved JSON renders objects at their pre-import positions
        instead of the imported ones."""
        out = tmp_path / "platform_position_priority"
        PlatformGenerator(region_sensor_model, customization=None, output_dir=str(out)).generate()
        canvas_src = (out / "frontend" / "src" / "components" / "InstanceCanvas.tsx").read_text(encoding="utf-8")
        # Backend position must come first in the fallback chain.
        assert "backendPosition ?? storedPosition ?? defaultPosition" in canvas_src
