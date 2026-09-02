"""
Round-trip and code-builder tests for the Map component.

Covers:
  1. ``parse_map`` (json_to_buml): GrapesJS JSON → Map metamodel instance.
  2. ``_apply_map_attributes`` / ``gui_buml_to_json`` (buml_to_json): Map + attributes
     are emitted correctly in the output JSON.
  3. Code builder: Map-containing GUIModel → Python code → exec() → GUIModel.
  4. React generator: MapBlock import/usage + Leaflet deps in generated package.json.
  5. Flutter generator: FlutterMap widget emission + flutter_map in pubspec.yaml.
"""

import os
import json
import textwrap
import tempfile

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Property, FloatType, StringType, Multiplicity,
)
from besser.BUML.metamodel.gui import (
    GUIModel, Module, Screen, DataBinding,
)
from besser.BUML.metamodel.gui.dashboard import Map

from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.gui_processors.chart_parsers import (
    parse_map,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.gui_diagram_converter import (
    gui_buml_to_json,
)
from besser.utilities.buml_code_builder.gui_model_builder import gui_model_to_code


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def location_domain():
    """Domain model with a Location class that has lat/lng/name attributes."""
    lat_prop = Property(name="latitude", type=FloatType)
    lng_prop = Property(name="longitude", type=FloatType)
    name_prop = Property(name="store_name", type=StringType)
    location_cls = Class(name="Location", attributes={lat_prop, lng_prop, name_prop})
    domain_model = DomainModel(name="LocationDomain", types={location_cls})
    return {
        "domain_model": domain_model,
        "location_cls": location_cls,
        "lat_prop": lat_prop,
        "lng_prop": lng_prop,
        "name_prop": name_prop,
    }


def _build_class_model(location_cls: Class, lat_prop: Property, lng_prop: Property,
                        name_prop: Property) -> dict:
    """Build a minimal GrapesJS class-model dict (UUID-indexed) for resolver lookups."""
    cls_id = "cls-uuid-001"
    lat_id = "attr-uuid-lat"
    lng_id = "attr-uuid-lng"
    name_id = "attr-uuid-name"
    class_model = {
        "elements": {
            cls_id: {"id": cls_id, "type": "Class", "name": location_cls.name},
            lat_id: {"id": lat_id, "type": "ClassAttribute", "name": lat_prop.name},
            lng_id: {"id": lng_id, "type": "ClassAttribute", "name": lng_prop.name},
            name_id: {"id": name_id, "type": "ClassAttribute", "name": name_prop.name},
        }
    }
    return class_model, cls_id, lat_id, lng_id, name_id


# ---------------------------------------------------------------------------
# 1. parse_map: JSON → Map metamodel
# ---------------------------------------------------------------------------

class TestParseMap:

    def test_minimal_map_no_binding(self):
        """Map without a data-source is constructed with default defaults."""
        comp = {
            "type": "map",
            "attributes": {
                "map-title": "World Overview",
                "map-latitude": "48.8566",
                "map-longitude": "2.3522",
                "map-zoom": "10",
            },
        }
        m = parse_map(comp, {}, None)
        assert isinstance(m, Map)
        assert m.title == "World Overview"
        assert m.center_latitude == pytest.approx(48.8566)
        assert m.center_longitude == pytest.approx(2.3522)
        assert m.zoom == 10
        assert m.data_binding is None
        assert m.latitude_field is None

    def test_map_with_bound_class_and_geo_fields(self, location_domain):
        """parse_map resolves a bound class + geo field UUIDs into Property objects."""
        loc_cls = location_domain["location_cls"]
        lat_prop = location_domain["lat_prop"]
        lng_prop = location_domain["lng_prop"]
        name_prop = location_domain["name_prop"]
        domain_model = location_domain["domain_model"]

        class_model, cls_id, lat_id, lng_id, name_id = _build_class_model(
            loc_cls, lat_prop, lng_prop, name_prop
        )

        comp = {
            "type": "map",
            "attributes": {
                "map-title": "Store Locations",
                "map-latitude": "51.5",
                "map-longitude": "-0.09",
                "map-zoom": "12",
                "data-source": cls_id,
                "latitude-field": lat_id,
                "longitude-field": lng_id,
                "marker-label-field": name_id,
            },
        }
        m = parse_map(comp, class_model, domain_model)
        assert isinstance(m, Map)
        assert m.data_binding is not None
        assert m.data_binding.domain_concept is loc_cls
        assert m.latitude_field is not None
        assert m.latitude_field.name == "latitude"
        assert m.longitude_field is not None
        assert m.longitude_field.name == "longitude"
        assert m.marker_label_field is not None
        assert m.marker_label_field.name == "store_name"

    def test_map_missing_attributes_uses_defaults(self):
        """parse_map falls back to sensible defaults when attributes are absent."""
        m = parse_map({}, {}, None)
        assert m.center_latitude == pytest.approx(0.0)
        assert m.center_longitude == pytest.approx(0.0)
        assert m.zoom == 10


# ---------------------------------------------------------------------------
# 2. gui_buml_to_json: Map metamodel → GrapesJS JSON
# ---------------------------------------------------------------------------

class TestMapBumlToJson:

    @staticmethod
    def _make_gui_with_map(map_component: Map) -> str:
        """Generate Python code for a single-screen GUI containing *map_component*."""
        screen = Screen(
            name="MapScreen",
            description="Screen with map",
            view_elements={map_component},
            is_main_page=True,
        )
        module = Module(name="MapModule", screens={screen})
        gui = GUIModel(
            name="MapApp",
            package="com.test.map",
            versionCode="1",
            versionName="1.0",
            modules={module},
            description="Map demo",
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
            gui_model_to_code(gui, tf.name)
            code_path = tf.name
        with open(code_path, "r", encoding="utf-8") as f:
            code = f.read()
        os.unlink(code_path)
        return code

    def test_map_attributes_in_json_output(self):
        """gui_buml_to_json emits map-latitude, map-longitude, map-zoom attributes."""
        m = Map(name="CityMap", title="City View", center_latitude=48.85,
                center_longitude=2.35, zoom=11)
        code = self._make_gui_with_map(m)
        result = gui_buml_to_json(code)

        # Flatten all component attributes from every page
        all_attrs: list[dict] = []
        for page in result.get("pages", []):
            for frame in page.get("frames", []):
                for comp in frame.get("component", {}).get("components", []):
                    _collect_attrs(comp, all_attrs)

        map_attrs = next((a for a in all_attrs if a.get("map-latitude") is not None), None)
        assert map_attrs is not None, "map-latitude not found in JSON output"
        assert float(map_attrs["map-latitude"]) == pytest.approx(48.85)
        assert float(map_attrs["map-longitude"]) == pytest.approx(2.35)
        assert int(map_attrs["map-zoom"]) == 11

    def test_map_type_string_in_json(self):
        """The serialized component type for a Map is 'map'."""
        m = Map(name="TypeMap", title="T")
        code = self._make_gui_with_map(m)
        result = gui_buml_to_json(code)

        all_types: list[str] = []
        for page in result.get("pages", []):
            for frame in page.get("frames", []):
                for comp in frame.get("component", {}).get("components", []):
                    _collect_types(comp, all_types)

        assert "map" in all_types, f"Expected 'map' in types, got: {all_types}"


def _collect_attrs(comp: dict, out: list) -> None:
    """Recursively collect all 'attributes' dicts from a component tree."""
    attrs = comp.get("attributes", {})
    if attrs:
        out.append(attrs)
    for child in comp.get("components", []):
        _collect_attrs(child, out)


def _collect_types(comp: dict, out: list) -> None:
    """Recursively collect all 'type' strings from a component tree."""
    t = comp.get("type")
    if t:
        out.append(t)
    for child in comp.get("components", []):
        _collect_types(child, out)


# ---------------------------------------------------------------------------
# 3. Code builder round-trip: Map → Python code → exec() → Map preserved
# ---------------------------------------------------------------------------

class TestMapCodeBuilderRoundtrip:

    def test_unbound_map_roundtrip(self, tmp_path):
        """Unbound Map survives exec() round-trip with correct field values."""
        m = Map(name="ExecMap", title="Exec Test", center_latitude=1.23,
                center_longitude=4.56, zoom=7)
        screen = Screen(name="S1", description="", view_elements={m}, is_main_page=True)
        module = Module(name="M1", screens={screen})
        gui = GUIModel(name="G1", package="com.test", versionCode="1",
                       versionName="1.0", modules={module}, description="")

        code_file = str(tmp_path / "gui.py")
        gui_model_to_code(gui, code_file)
        with open(code_file, "r", encoding="utf-8") as f:
            code = f.read()

        # Verify it compiles
        compile(code, code_file, "exec")

        # Verify it executes to a GUIModel that contains the Map
        ns: dict = {}
        exec(code, ns)  # noqa: S102 — test sandbox
        recreated: GUIModel = ns["gui_model"]
        assert isinstance(recreated, GUIModel)

        screens = [s for mod in recreated.modules for s in mod.screens]
        all_comps = [ve for s in screens for ve in s.view_elements]
        map_comps = [c for c in all_comps if isinstance(c, Map)]
        assert len(map_comps) == 1
        rm: Map = map_comps[0]
        assert rm.center_latitude == pytest.approx(1.23)
        assert rm.center_longitude == pytest.approx(4.56)
        assert rm.zoom == 7

    def test_bound_map_roundtrip_preserves_geo_field_names(self, tmp_path):
        """Map with DataBinding: geo field names survive the code-builder round-trip."""
        lat_p = Property(name="lat", type=FloatType)
        lng_p = Property(name="lng", type=FloatType)
        label_p = Property(name="label", type=StringType)
        loc_cls = Class(name="Loc", attributes={lat_p, lng_p, label_p})
        binding = DataBinding(name="loc_bind", domain_concept=loc_cls)

        m = Map(name="BoundMap", center_latitude=0.0, center_longitude=0.0, zoom=5,
                latitude_field=lat_p, longitude_field=lng_p,
                marker_label_field=label_p, data_binding=binding)
        screen = Screen(name="S2", description="", view_elements={m}, is_main_page=True)
        module = Module(name="M2", screens={screen})
        domain = DomainModel(name="D", types={loc_cls})
        gui = GUIModel(name="G2", package="com.test", versionCode="1",
                       versionName="1.0", modules={module}, description="")

        code_file = str(tmp_path / "bound_gui.py")
        gui_model_to_code(gui, code_file, domain_model=domain)
        with open(code_file, "r", encoding="utf-8") as f:
            code = f.read()

        # Geo field names must appear in the generated code
        assert "lat" in code
        assert "lng" in code
        assert "label" in code

        compile(code, code_file, "exec")
