"""
Tests for the MapLayer metamodel additions:
  - MapLayerType enum
  - MapLayer construction, setter validation, effective_layer_type()
  - Map.layers setter validation
"""
import pytest

from besser.BUML.metamodel.structural import (
    Class, Property, DomainModel, FloatType, StringType,
)
from besser.BUML.metamodel.gui import DataBinding
from besser.BUML.metamodel.gui.dashboard import Map, MapLayer, MapLayerType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def geo_class():
    """A simple Location class with lat/lng/name attributes."""
    lat  = Property(name="latitude",  type=FloatType)
    lng  = Property(name="longitude", type=FloatType)
    name = Property(name="store_name", type=StringType)
    cls  = Class(name="Location", attributes={lat, lng, name})
    return {"cls": cls, "lat": lat, "lng": lng, "name": name}


@pytest.fixture()
def geojson_class():
    """A Canton class with geometry (str) and population (float) attributes."""
    geom  = Property(name="geometry",   type=StringType)
    pop   = Property(name="population", type=FloatType)
    label = Property(name="canton_name", type=StringType)
    cls   = Class(name="Canton", attributes={geom, pop, label})
    return {"cls": cls, "geom": geom, "pop": pop, "label": label}


# ---------------------------------------------------------------------------
# MapLayerType
# ---------------------------------------------------------------------------

class TestMapLayerType:

    def test_enum_values_exist(self):
        assert MapLayerType.points.value    == "points"
        assert MapLayerType.geojson.value   == "geojson"
        assert MapLayerType.choropleth.value == "choropleth"
        assert MapLayerType.heatmap.value   == "heatmap"

    def test_all_four_types_present(self):
        values = {t.value for t in MapLayerType}
        assert values == {"points", "geojson", "choropleth", "heatmap"}


# ---------------------------------------------------------------------------
# MapLayer construction and setter validation
# ---------------------------------------------------------------------------

class TestMapLayerConstruction:

    def test_minimal_construction(self):
        layer = MapLayer(name="points_layer")
        assert layer.name == "points_layer"
        assert layer.layer_type is None
        assert layer.data_binding is None
        assert layer.latitude_field is None
        assert layer.longitude_field is None
        assert layer.label_field is None
        assert layer.weight_field is None
        assert layer.geojson_field is None
        assert layer.value_field is None

    def test_explicit_layer_type(self):
        layer = MapLayer(name="heat", layer_type=MapLayerType.heatmap)
        assert layer.layer_type is MapLayerType.heatmap

    def test_all_fields_set(self, geo_class, geojson_class):
        layer = MapLayer(
            name="full_layer",
            layer_type=MapLayerType.choropleth,
            latitude_field=geo_class["lat"],
            longitude_field=geo_class["lng"],
            label_field=geo_class["name"],
            weight_field=geo_class["lat"],   # reuse for test brevity
            geojson_field=geojson_class["geom"],
            value_field=geojson_class["pop"],
        )
        assert layer.latitude_field.name == "latitude"
        assert layer.geojson_field.name  == "geometry"
        assert layer.value_field.name    == "population"

    def test_field_setter_rejects_non_property(self, geo_class):
        """All field setters must raise TypeError when given a non-Property."""
        layer = MapLayer(name="bad")
        with pytest.raises(TypeError):
            layer.latitude_field = "latitude"  # string, not Property
        with pytest.raises(TypeError):
            layer.longitude_field = 0.0
        with pytest.raises(TypeError):
            layer.label_field = 42
        with pytest.raises(TypeError):
            layer.weight_field = ["a"]
        with pytest.raises(TypeError):
            layer.geojson_field = True
        with pytest.raises(TypeError):
            layer.value_field = {}

    def test_field_setter_accepts_none(self, geo_class):
        """All field setters must accept None to clear a previously set value."""
        layer = MapLayer(name="nullable", latitude_field=geo_class["lat"])
        layer.latitude_field = None
        assert layer.latitude_field is None


# ---------------------------------------------------------------------------
# effective_layer_type() auto-detection
# ---------------------------------------------------------------------------

class TestEffectiveLayerType:

    def test_explicit_type_wins(self, geo_class):
        layer = MapLayer(name="l", layer_type=MapLayerType.geojson,
                         latitude_field=geo_class["lat"],
                         longitude_field=geo_class["lng"])
        # Even though lat/lng are set (would suggest points), explicit type wins.
        assert layer.effective_layer_type() is MapLayerType.geojson

    def test_geojson_and_value_field_gives_choropleth(self, geojson_class):
        layer = MapLayer(
            name="l",
            geojson_field=geojson_class["geom"],
            value_field=geojson_class["pop"],
        )
        assert layer.effective_layer_type() is MapLayerType.choropleth

    def test_geojson_field_only_gives_geojson(self, geojson_class):
        layer = MapLayer(name="l", geojson_field=geojson_class["geom"])
        assert layer.effective_layer_type() is MapLayerType.geojson

    def test_weight_field_gives_heatmap(self, geo_class):
        layer = MapLayer(name="l",
                         latitude_field=geo_class["lat"],
                         longitude_field=geo_class["lng"],
                         weight_field=geo_class["lat"])  # reuse for test brevity
        assert layer.effective_layer_type() is MapLayerType.heatmap

    def test_lat_lng_fields_give_points(self, geo_class):
        layer = MapLayer(name="l",
                         latitude_field=geo_class["lat"],
                         longitude_field=geo_class["lng"])
        assert layer.effective_layer_type() is MapLayerType.points

    def test_bound_class_with_geometry_attribute_gives_geojson(self, geojson_class):
        """When no field refs are set but the bound class has a 'geometry' str attribute,
        auto-detect should return geojson."""
        binding = DataBinding(name="b", domain_concept=geojson_class["cls"])
        layer = MapLayer(name="l", data_binding=binding)
        result = layer.effective_layer_type()
        # The class has 'geometry:str' and 'population:float' → choropleth or geojson
        assert result in (MapLayerType.geojson, MapLayerType.choropleth)

    def test_bound_class_with_lat_lng_attributes_gives_points(self, geo_class):
        """Bound class with 'latitude' and 'longitude' float attributes → points."""
        binding = DataBinding(name="b", domain_concept=geo_class["cls"])
        layer = MapLayer(name="l", data_binding=binding)
        result = layer.effective_layer_type()
        assert result is MapLayerType.points

    def test_no_hints_defaults_to_points(self):
        """With no field refs and no bound class, default is points."""
        layer = MapLayer(name="l")
        assert layer.effective_layer_type() is MapLayerType.points


# ---------------------------------------------------------------------------
# Map.layers setter validation
# ---------------------------------------------------------------------------

class TestMapLayers:

    def test_empty_layers(self):
        m = Map(name="M", layers=[])
        assert m.layers == []

    def test_single_layer(self, geo_class):
        layer = MapLayer(name="p", layer_type=MapLayerType.points)
        m = Map(name="M", layers=[layer])
        assert len(m.layers) == 1
        assert m.layers[0] is layer

    def test_multiple_layers(self, geo_class, geojson_class):
        l1 = MapLayer(name="points", layer_type=MapLayerType.points)
        l2 = MapLayer(name="regions", layer_type=MapLayerType.choropleth)
        m = Map(name="M", layers=[l1, l2])
        assert len(m.layers) == 2

    def test_layers_setter_rejects_non_list(self):
        m = Map(name="M")
        with pytest.raises(TypeError):
            m.layers = MapLayer(name="bad")  # single item, not a list

    def test_layers_setter_rejects_non_maplayer_items(self):
        m = Map(name="M")
        with pytest.raises(TypeError):
            m.layers = ["not_a_layer"]

    def test_layers_default_is_empty(self):
        m = Map(name="M")
        assert m.layers == []

    def test_map_static_props(self):
        m = Map(name="M", title="T", center_latitude=1.1,
                center_longitude=2.2, zoom=8)
        assert m.title == "T"
        assert m.center_latitude == pytest.approx(1.1)
        assert m.center_longitude == pytest.approx(2.2)
        assert m.zoom == 8
