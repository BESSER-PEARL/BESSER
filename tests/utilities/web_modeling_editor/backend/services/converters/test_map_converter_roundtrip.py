"""
Round-trip and code-builder tests for the multi-layer Map component.

Covers:
  1. ``parse_map`` (json_to_buml): JSON with ``map-layers`` → Map with MapLayer list.
  2. ``_apply_map_attributes`` / ``gui_buml_to_json`` (buml_to_json): Map with
     layers is serialised to JSON and ``map-layers`` attribute is present.
  3. Code builder: Map-containing GUIModel → Python code → exec() → GUIModel
     with correct layers, types, and field names.
"""

import json
import os
import tempfile

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Property, FloatType, StringType, Multiplicity,
)
from besser.BUML.metamodel.gui import GUIModel, Module, Screen, DataBinding
from besser.BUML.metamodel.gui.dashboard import Map, MapLayer, MapLayerType

from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.gui_processors.chart_parsers import (
    parse_map,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.gui_diagram_converter import (
    gui_buml_to_json,
    _serialize_gui_model,
)
from besser.utilities.buml_code_builder.gui_model_builder import gui_model_to_code


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store_domain():
    """Store class with lat/lng/name attributes (points layer)."""
    lat  = Property(name="latitude",   type=FloatType)
    lng  = Property(name="longitude",  type=FloatType)
    name = Property(name="store_name", type=StringType)
    cls  = Class(name="Store", attributes={lat, lng, name})
    domain = DomainModel(name="StoreDomain", types={cls})
    return {"domain": domain, "cls": cls, "lat": lat, "lng": lng, "name": name}


@pytest.fixture()
def canton_domain():
    """Canton class with geometry/population attributes (choropleth layer)."""
    geom = Property(name="geometry",   type=StringType)
    pop  = Property(name="population", type=FloatType)
    lbl  = Property(name="canton_name", type=StringType)
    cls  = Class(name="Canton", attributes={geom, pop, lbl})
    domain = DomainModel(name="CantonDomain", types={cls})
    return {"domain": domain, "cls": cls, "geom": geom, "pop": pop, "lbl": lbl}


def _build_class_model(*classes_and_props):
    """
    Build a minimal GrapesJS class-model dict from (class, prop...) tuples.
    Returns the model dict and a mapping of object → UUID.
    """
    elements: dict = {}
    id_map: dict = {}
    counter = [0]

    def new_id(prefix: str):
        counter[0] += 1
        return f"{prefix}-{counter[0]:03d}"

    for cls, *props in classes_and_props:
        cls_id = new_id("cls")
        elements[cls_id] = {"id": cls_id, "type": "Class", "name": cls.name}
        id_map[cls] = cls_id
        for p in props:
            attr_id = new_id("attr")
            elements[attr_id] = {"id": attr_id, "type": "ClassAttribute", "name": p.name}
            id_map[p] = attr_id

    return {"elements": elements}, id_map


def _collect_attrs(comp: dict, out: list) -> None:
    attrs = comp.get("attributes", {})
    if attrs:
        out.append(attrs)
    for child in comp.get("components", []):
        _collect_attrs(child, out)


def _make_gui(layers: list, center_lat=49.0, center_lng=6.0, zoom=10) -> tuple:
    """Wrap *layers* in a minimal GUIModel/Screen/Module. Returns (gui, map_comp)."""
    map_comp = Map(
        name="TestMap",
        title="Test",
        center_latitude=center_lat,
        center_longitude=center_lng,
        zoom=zoom,
        layers=layers,
    )
    screen = Screen(name="S", description="", view_elements={map_comp}, is_main_page=True)
    module = Module(name="M", screens={screen})
    gui = GUIModel(name="G", package="com.test", versionCode="1",
                   versionName="1.0", modules={module}, description="")
    return gui, map_comp


# ---------------------------------------------------------------------------
# 1. parse_map: JSON with map-layers → Map with MapLayer list
# ---------------------------------------------------------------------------

class TestParseMapLayered:

    def test_map_no_layers(self):
        """Map without map-layers attribute produces an empty layers list."""
        comp = {
            "type": "map",
            "attributes": {
                "map-title": "World",
                "map-latitude": "0",
                "map-longitude": "0",
                "map-zoom": "5",
            },
        }
        m = parse_map(comp, {}, None)
        assert isinstance(m, Map)
        assert m.layers == []

    def test_map_with_points_layer_by_uuid(self, store_domain):
        """parse_map resolves a points layer from UUID references."""
        cls = store_domain["cls"]
        lat = store_domain["lat"]
        lng = store_domain["lng"]
        name = store_domain["name"]
        class_model, id_map = _build_class_model((cls, lat, lng, name))

        layers_json = json.dumps([{
            "name": "stores",
            "type": "points",
            "dataSource": id_map[cls],
            "latitudeField": id_map[lat],
            "longitudeField": id_map[lng],
            "labelField": id_map[name],
        }])
        comp = {
            "type": "map",
            "attributes": {
                "map-title": "Stores",
                "map-latitude": "49.0",
                "map-longitude": "6.0",
                "map-zoom": "10",
                "map-layers": layers_json,
            },
        }
        m = parse_map(comp, class_model, store_domain["domain"])

        assert len(m.layers) == 1
        layer = m.layers[0]
        assert isinstance(layer, MapLayer)
        assert layer.layer_type is MapLayerType.points
        assert layer.data_binding is not None
        assert layer.data_binding.domain_concept is cls
        assert layer.latitude_field.name  == "latitude"
        assert layer.longitude_field.name == "longitude"
        assert layer.label_field.name     == "store_name"

    def test_map_with_choropleth_layer_by_uuid(self, canton_domain):
        """parse_map resolves a choropleth layer from UUID references."""
        cls  = canton_domain["cls"]
        geom = canton_domain["geom"]
        pop  = canton_domain["pop"]
        class_model, id_map = _build_class_model((cls, geom, pop))

        layers_json = json.dumps([{
            "name": "cantons",
            "type": "choropleth",
            "dataSource": id_map[cls],
            "geojsonField": id_map[geom],
            "valueField": id_map[pop],
        }])
        comp = {
            "type": "map",
            "attributes": {
                "map-latitude": "49.8",
                "map-longitude": "6.1",
                "map-zoom": "9",
                "map-layers": layers_json,
            },
        }
        m = parse_map(comp, class_model, canton_domain["domain"])

        assert len(m.layers) == 1
        layer = m.layers[0]
        assert layer.layer_type is MapLayerType.choropleth
        assert layer.geojson_field.name == "geometry"
        assert layer.value_field.name   == "population"

    def test_map_with_two_layers(self, store_domain, canton_domain):
        """Two layers in one map-layers JSON → both MapLayer objects resolved."""
        store_cls  = store_domain["cls"]
        canton_cls = canton_domain["cls"]
        lat  = store_domain["lat"]
        lng  = store_domain["lng"]
        geom = canton_domain["geom"]
        pop  = canton_domain["pop"]

        # Build a combined domain
        combined_domain = DomainModel(
            name="Combined", types={store_cls, canton_cls}
        )
        class_model, id_map = _build_class_model(
            (store_cls, lat, lng),
            (canton_cls, geom, pop),
        )

        layers_json = json.dumps([
            {
                "name": "stores",
                "type": "points",
                "dataSource": id_map[store_cls],
                "latitudeField": id_map[lat],
                "longitudeField": id_map[lng],
            },
            {
                "name": "cantons",
                "type": "choropleth",
                "dataSource": id_map[canton_cls],
                "geojsonField": id_map[geom],
                "valueField": id_map[pop],
            },
        ])
        comp = {
            "type": "map",
            "attributes": {
                "map-latitude": "49.8",
                "map-longitude": "6.1",
                "map-zoom": "9",
                "map-layers": layers_json,
            },
        }
        m = parse_map(comp, class_model, combined_domain)
        assert len(m.layers) == 2
        types = {l.layer_type for l in m.layers}
        assert types == {MapLayerType.points, MapLayerType.choropleth}

    def test_map_layers_name_fallback(self, store_domain):
        """
        When a field value is a name rather than a UUID (BUML→JSON round-trip),
        parse_map resolves it via name-based fallback.
        """
        cls = store_domain["cls"]
        lat = store_domain["lat"]
        lng = store_domain["lng"]

        # Use names instead of UUIDs (the BUML→JSON converter emits names)
        layers_json = json.dumps([{
            "name": "stores",
            "type": "points",
            "dataSource": cls.name,        # class name, not UUID
            "latitudeField": lat.name,     # field name, not UUID
            "longitudeField": lng.name,
        }])
        comp = {
            "type": "map",
            "attributes": {
                "map-latitude": "49.0",
                "map-longitude": "6.0",
                "map-zoom": "10",
                "map-layers": layers_json,
            },
        }
        # class_model is empty — name fallback must kick in
        m = parse_map(comp, {"elements": {}}, store_domain["domain"])

        assert len(m.layers) == 1
        layer = m.layers[0]
        assert layer.data_binding.domain_concept is cls
        assert layer.latitude_field.name  == "latitude"
        assert layer.longitude_field.name == "longitude"

    def test_static_props_still_parse(self):
        """Static map props (title, lat, lng, zoom) still parse correctly."""
        comp = {
            "type": "map",
            "attributes": {
                "map-title": "Test Title",
                "map-latitude": "51.5",
                "map-longitude": "-0.1",
                "map-zoom": "13",
            },
        }
        m = parse_map(comp, {}, None)
        assert m.title == "Test Title"
        assert m.center_latitude  == pytest.approx(51.5)
        assert m.center_longitude == pytest.approx(-0.1)
        assert m.zoom == 13


# ---------------------------------------------------------------------------
# 2. gui_buml_to_json: Map with layers → JSON with map-layers attribute
# ---------------------------------------------------------------------------

class TestMapBumlToJsonLayered:

    @staticmethod
    def _gui_with_layers_to_json(layers: list, **kwargs) -> dict:
        """Serialize a GUIModel directly via _serialize_gui_model.

        Bypasses the code-builder intermediate step intentionally: that round-trip
        is covered by TestMapCodeBuilderRoundtrip.  Here we want to test that
        _apply_map_attributes / _serialize_gui_model emit the correct map-layers
        JSON given a BUML Map whose layers are already fully configured.
        """
        gui, _ = _make_gui(layers, **kwargs)
        return _serialize_gui_model(gui)

    def test_map_layers_attribute_emitted(self, store_domain):
        """gui_buml_to_json emits a map-layers JSON attribute."""
        cls = store_domain["cls"]
        lat = store_domain["lat"]
        lng = store_domain["lng"]
        layer = MapLayer(
            name="stores",
            layer_type=MapLayerType.points,
            data_binding=DataBinding(name="b", domain_concept=cls),
            latitude_field=lat,
            longitude_field=lng,
        )
        result = self._gui_with_layers_to_json([layer])

        all_attrs: list = []
        for page in result.get("pages", []):
            for frame in page.get("frames", []):
                for comp in frame.get("component", {}).get("components", []):
                    _collect_attrs(comp, all_attrs)

        map_attrs = next((a for a in all_attrs if "map-layers" in a), None)
        assert map_attrs is not None, "map-layers attribute not found in JSON output"

        layers_data = json.loads(map_attrs["map-layers"])
        assert isinstance(layers_data, list)
        assert len(layers_data) == 1
        l0 = layers_data[0]
        assert l0["type"]   == "points"
        assert l0["name"]   == "stores"
        assert l0["dataSource"] == "Store"  # BUML→JSON uses class name

    def test_choropleth_layer_roundtrip_fields(self, canton_domain):
        """Choropleth layer fields are preserved in buml_to_json output."""
        cls  = canton_domain["cls"]
        geom = canton_domain["geom"]
        pop  = canton_domain["pop"]
        layer = MapLayer(
            name="cantons",
            layer_type=MapLayerType.choropleth,
            data_binding=DataBinding(name="b", domain_concept=cls),
            geojson_field=geom,
            value_field=pop,
        )
        result = self._gui_with_layers_to_json([layer])

        all_attrs: list = []
        for page in result.get("pages", []):
            for frame in page.get("frames", []):
                for comp in frame.get("component", {}).get("components", []):
                    _collect_attrs(comp, all_attrs)

        map_attrs = next((a for a in all_attrs if "map-layers" in a), None)
        assert map_attrs is not None

        layers_data = json.loads(map_attrs["map-layers"])
        l0 = layers_data[0]
        assert l0["type"]          == "choropleth"
        assert l0["geojsonField"]  == "geometry"
        assert l0["valueField"]    == "population"

    def test_static_map_attrs_still_emitted(self):
        """map-latitude, map-longitude, map-zoom are still present in the JSON."""
        gui, _ = _make_gui([], center_lat=48.85, center_lng=2.35, zoom=11)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tf:
            gui_model_to_code(gui, tf.name)
            code_path = tf.name
        with open(code_path, "r", encoding="utf-8") as f:
            code = f.read()
        os.unlink(code_path)
        result = gui_buml_to_json(code)

        all_attrs: list = []
        for page in result.get("pages", []):
            for frame in page.get("frames", []):
                for comp in frame.get("component", {}).get("components", []):
                    _collect_attrs(comp, all_attrs)

        map_attrs = next((a for a in all_attrs if "map-latitude" in a), None)
        assert map_attrs is not None, "map-latitude not found in JSON output"
        assert float(map_attrs["map-latitude"])  == pytest.approx(48.85)
        assert float(map_attrs["map-longitude"]) == pytest.approx(2.35)
        assert int(map_attrs["map-zoom"])        == 11


# ---------------------------------------------------------------------------
# 3. Code builder: exec() round-trip
# ---------------------------------------------------------------------------

class TestMapCodeBuilderRoundtrip:

    def test_unbound_map_compiles_and_executes(self, tmp_path):
        """A Map with no layers produces valid, executable Python code."""
        gui, _ = _make_gui([], center_lat=1.23, center_lng=4.56, zoom=7)
        code_file = str(tmp_path / "gui.py")
        gui_model_to_code(gui, code_file)

        with open(code_file, encoding="utf-8") as f:
            code = f.read()

        compile(code, code_file, "exec")
        ns: dict = {}
        exec(code, ns)  # noqa: S102 — test sandbox
        recreated: GUIModel = ns["gui_model"]
        assert isinstance(recreated, GUIModel)

        screens = [s for mod in recreated.modules for s in mod.screens]
        all_comps = [ve for s in screens for ve in s.view_elements]
        map_comps = [c for c in all_comps if isinstance(c, Map)]
        assert len(map_comps) == 1
        rm = map_comps[0]
        assert rm.center_latitude  == pytest.approx(1.23)
        assert rm.center_longitude == pytest.approx(4.56)
        assert rm.zoom == 7

    def test_map_with_layers_field_names_in_code(self, store_domain, canton_domain, tmp_path):
        """Field names for all layer types appear in the generated code."""
        store_cls  = store_domain["cls"]
        canton_cls = canton_domain["cls"]
        lat  = store_domain["lat"]
        lng  = store_domain["lng"]
        geom = canton_domain["geom"]
        pop  = canton_domain["pop"]

        combined_domain = DomainModel(
            name="Combined", types={store_cls, canton_cls}
        )

        layers = [
            MapLayer(
                name="stores",
                layer_type=MapLayerType.points,
                data_binding=DataBinding(name="b1", domain_concept=store_cls),
                latitude_field=lat,
                longitude_field=lng,
            ),
            MapLayer(
                name="cantons",
                layer_type=MapLayerType.choropleth,
                data_binding=DataBinding(name="b2", domain_concept=canton_cls),
                geojson_field=geom,
                value_field=pop,
            ),
        ]
        gui, _ = _make_gui(layers)
        code_file = str(tmp_path / "multi_layer.py")
        gui_model_to_code(gui, code_file, domain_model=combined_domain)

        with open(code_file, encoding="utf-8") as f:
            code = f.read()

        # Field names from both layers must appear in the generated code
        assert "latitude"   in code
        assert "longitude"  in code
        assert "geometry"   in code
        assert "population" in code
        assert "MapLayerType.points"     in code
        assert "MapLayerType.choropleth" in code

        # The code must compile
        compile(code, code_file, "exec")

    def test_map_with_layers_executes(self, store_domain, tmp_path):
        """A Map with a single points layer executes and layers are preserved."""
        cls = store_domain["cls"]
        lat = store_domain["lat"]
        lng = store_domain["lng"]

        layer = MapLayer(
            name="stores",
            layer_type=MapLayerType.points,
            data_binding=DataBinding(name="b", domain_concept=cls),
            latitude_field=lat,
            longitude_field=lng,
        )
        gui, _ = _make_gui([layer])
        code_file = str(tmp_path / "exec_layers.py")
        gui_model_to_code(gui, code_file, domain_model=store_domain["domain"])

        with open(code_file, encoding="utf-8") as f:
            code = f.read()

        compile(code, code_file, "exec")
        ns: dict = {}
        exec(code, ns)  # noqa: S102 — test sandbox
        recreated: GUIModel = ns["gui_model"]
        assert isinstance(recreated, GUIModel)

        screens = [s for mod in recreated.modules for s in mod.screens]
        map_comps = [
            ve for s in screens for ve in s.view_elements if isinstance(ve, Map)
        ]
        assert len(map_comps) == 1
        rm = map_comps[0]
        assert len(rm.layers) == 1
        assert rm.layers[0].layer_type is MapLayerType.points
