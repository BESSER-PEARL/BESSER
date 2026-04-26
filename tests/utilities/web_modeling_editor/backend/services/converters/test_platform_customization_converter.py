"""Round-trip tests for PlatformCustomization JSON <-> BUML converters."""

import pytest

from besser.BUML.metamodel.platform_customization import (
    ArrowStyle,
    AssociationCustomization,
    ClassCustomization,
    DiagramCustomization,
    FontWeight,
    LabelPosition,
    LineStyle,
    NodeShape,
    PlatformCustomizationModel,
    Theme,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json import (
    platform_customization_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml import (
    process_platform_customization_diagram,
)


@pytest.fixture
def full_json_payload():
    return {
        "classOverrides": {
            "Region": {"isContainer": True, "defaultWidth": 400, "defaultHeight": 300},
            "Sensor": {"defaultWidth": 80, "defaultHeight": 80},
        },
        "associationOverrides": {
            "has": {"edgeColor": "#22c55e"},
        },
    }


class TestJsonToBuml:
    def test_empty_json_produces_empty_model(self):
        model = process_platform_customization_diagram(None)
        assert isinstance(model, PlatformCustomizationModel)
        assert model.class_overrides == {}
        assert model.association_overrides == {}

    def test_empty_dict_produces_empty_model(self):
        model = process_platform_customization_diagram({})
        assert model.class_overrides == {}
        assert model.association_overrides == {}

    def test_full_payload(self, full_json_payload):
        model = process_platform_customization_diagram(full_json_payload)
        region = model.get_class_customization("Region")
        assert region.is_container is True
        assert region.default_width == 400
        assert region.default_height == 300
        sensor = model.get_class_customization("Sensor")
        assert sensor.is_container is False  # default
        assert sensor.default_width == 80
        assert model.get_association_customization("has").edge_color == "#22c55e"

    def test_nested_under_model_key(self, full_json_payload):
        """The frontend may wrap payload under a 'model' key — tolerate that."""
        wrapped = {"model": full_json_payload}
        model = process_platform_customization_diagram(wrapped)
        assert model.get_class_customization("Region").is_container is True

    def test_ignores_unknown_extra_fields(self):
        payload = {
            "classOverrides": {"Region": {"isContainer": True, "futureField": "ignored"}},
            "associationOverrides": {},
            "someTopLevelExtra": 42,
        }
        model = process_platform_customization_diagram(payload)
        assert model.get_class_customization("Region").is_container is True

    def test_skips_non_string_keys(self):
        payload = {
            "classOverrides": {"": {"isContainer": True}, "Valid": {"isContainer": True}},
        }
        model = process_platform_customization_diagram(payload)
        assert "" not in model.class_overrides
        assert model.get_class_customization("Valid").is_container is True

    def test_skips_non_dict_values(self):
        payload = {"classOverrides": {"Region": "not-a-dict"}}
        model = process_platform_customization_diagram(payload)
        assert "Region" not in model.class_overrides

    @pytest.mark.parametrize("bad_width", [0, -5, "not-a-number", ""])
    def test_invalid_width_falls_back_to_none(self, bad_width):
        payload = {"classOverrides": {"Region": {"defaultWidth": bad_width}}}
        model = process_platform_customization_diagram(payload)
        assert model.get_class_customization("Region").default_width is None

    def test_whitespace_color_treated_as_none(self):
        payload = {"associationOverrides": {"has": {"edgeColor": "   "}}}
        model = process_platform_customization_diagram(payload)
        assert model.get_association_customization("has").edge_color is None

    def test_string_width_that_parses_is_accepted(self):
        payload = {"classOverrides": {"Region": {"defaultWidth": "400"}}}
        model = process_platform_customization_diagram(payload)
        assert model.get_class_customization("Region").default_width == 400


class TestBumlToJson:
    def test_empty_model_emits_empty_buckets(self):
        model = PlatformCustomizationModel(name="Empty")
        result = platform_customization_to_json(model)
        assert result == {"classOverrides": {}, "associationOverrides": {}}

    def test_full_model_emits_expected_shape(self):
        model = PlatformCustomizationModel(
            name="Full",
            class_overrides={
                "Region": ClassCustomization(is_container=True, default_width=400, default_height=300),
                "Sensor": ClassCustomization(default_width=80, default_height=80),
            },
            association_overrides={"has": AssociationCustomization(edge_color="#22c55e")},
        )
        result = platform_customization_to_json(model)
        assert result["classOverrides"]["Region"] == {
            "isContainer": True,
            "defaultWidth": 400,
            "defaultHeight": 300,
        }
        assert result["classOverrides"]["Sensor"] == {
            "defaultWidth": 80,
            "defaultHeight": 80,
        }
        assert result["associationOverrides"] == {"has": {"edgeColor": "#22c55e"}}

    def test_elides_default_values(self):
        """Classes with all-default customization should not appear in output."""
        model = PlatformCustomizationModel(
            name="Minimal",
            class_overrides={"NoOverride": ClassCustomization()},
            association_overrides={"NoColor": AssociationCustomization()},
        )
        result = platform_customization_to_json(model)
        assert result == {"classOverrides": {}, "associationOverrides": {}}

    def test_elides_only_the_fields_that_are_defaults(self):
        """Partial customization: container=True but no sizes -> emit only isContainer."""
        model = PlatformCustomizationModel(
            name="Partial",
            class_overrides={"Region": ClassCustomization(is_container=True)},
        )
        result = platform_customization_to_json(model)
        assert result["classOverrides"] == {"Region": {"isContainer": True}}


class TestRoundTrip:
    def test_json_to_buml_to_json_is_identity(self, full_json_payload):
        model = process_platform_customization_diagram(full_json_payload)
        result = platform_customization_to_json(model)
        assert result == full_json_payload

    def test_buml_to_json_to_buml_preserves_overrides(self):
        original = PlatformCustomizationModel(
            name="Orig",
            class_overrides={
                "Region": ClassCustomization(is_container=True, default_width=400, default_height=300),
                "Sensor": ClassCustomization(default_width=80, default_height=80),
            },
            association_overrides={"has": AssociationCustomization(edge_color="#22c55e")},
        )
        as_json = platform_customization_to_json(original)
        reparsed = process_platform_customization_diagram(as_json)
        assert reparsed.get_class_customization("Region").is_container is True
        assert reparsed.get_class_customization("Region").default_width == 400
        assert reparsed.get_class_customization("Sensor").default_width == 80
        assert reparsed.get_association_customization("has").edge_color == "#22c55e"


# ---------------------------------------------------------------------------
# v2 — Sirius-Web-style fields
# ---------------------------------------------------------------------------


@pytest.fixture
def v2_full_payload():
    return {
        "classOverrides": {
            "Region": {
                "isContainer": True,
                "defaultWidth": 400,
                "defaultHeight": 300,
                "nodeShape": "rounded_rect",
                "fillColor": "#fef3c7",
                "borderColor": "#a16207",
                "borderWidth": 4,
                "borderStyle": "dashed",
                "borderRadius": 24,
                "fontSize": 18,
                "fontWeight": "bold",
                "fontColor": "#7c2d12",
                "labelPosition": "inside",
            },
        },
        "associationOverrides": {
            "has": {
                "edgeColor": "#22c55e",
                "lineWidth": 3,
                "lineStyle": "dotted",
                "sourceArrowStyle": "none",
                "targetArrowStyle": "diamond",
                "labelVisible": True,
                "labelFontSize": 14,
                "labelFontColor": "#111",
            },
        },
        "diagramCustomization": {
            "backgroundColor": "#fafafa",
            "gridVisible": True,
            "gridSize": 32,
            "snapToGrid": True,
            "theme": "dark",
        },
    }


class TestJsonToBumlV2:
    def test_full_v2_payload_parses(self, v2_full_payload):
        model = process_platform_customization_diagram(v2_full_payload)
        region = model.get_class_customization("Region")
        assert region.node_shape is NodeShape.ROUNDED_RECT
        assert region.fill_color == "#fef3c7"
        assert region.border_style is LineStyle.DASHED
        assert region.border_width == 4
        assert region.font_weight is FontWeight.BOLD
        assert region.label_position is LabelPosition.INSIDE

        has = model.get_association_customization("has")
        assert has.line_style is LineStyle.DOTTED
        assert has.target_arrow_style is ArrowStyle.DIAMOND
        assert has.label_visible is True
        assert has.label_font_size == 14

        diagram = model.diagram_customization
        assert diagram is not None
        assert diagram.theme is Theme.DARK
        assert diagram.grid_size == 32
        assert diagram.snap_to_grid is True

    def test_invalid_enum_falls_back_to_none(self):
        payload = {"classOverrides": {"X": {"nodeShape": "triangle"}}}
        model = process_platform_customization_diagram(payload)
        assert model.get_class_customization("X").node_shape is None

    @pytest.mark.parametrize("bad_width", [-1, 9, "huge"])
    def test_border_width_out_of_range_falls_back_to_none(self, bad_width):
        payload = {"classOverrides": {"X": {"borderWidth": bad_width}}}
        model = process_platform_customization_diagram(payload)
        assert model.get_class_customization("X").border_width is None

    def test_empty_diagram_block_yields_no_diagram_customization(self):
        payload = {"diagramCustomization": {}}
        model = process_platform_customization_diagram(payload)
        assert model.diagram_customization is None

    def test_diagram_block_with_invalid_only_fields_yields_none(self):
        payload = {"diagramCustomization": {"theme": "solarized", "gridSize": 9999}}
        model = process_platform_customization_diagram(payload)
        assert model.diagram_customization is None


class TestBumlToJsonV2:
    def test_v2_full_model_round_trips(self, v2_full_payload):
        model = process_platform_customization_diagram(v2_full_payload)
        emitted = platform_customization_to_json(model)
        assert emitted == v2_full_payload

    def test_v1_payload_round_trips_unchanged(self):
        """Critical regression: a v1-only payload must be byte-identical after
        going through the v2 converter."""
        v1_payload = {
            "classOverrides": {
                "Region": {"isContainer": True, "defaultWidth": 400, "defaultHeight": 300},
                "Sensor": {"defaultWidth": 80, "defaultHeight": 80},
            },
            "associationOverrides": {
                "has": {"edgeColor": "#22c55e"},
            },
        }
        model = process_platform_customization_diagram(v1_payload)
        emitted = platform_customization_to_json(model)
        assert emitted == v1_payload  # no diagramCustomization key, no new fields

    def test_diagram_only_model_emits_diagram_block(self):
        model = PlatformCustomizationModel(
            name="DiagOnly",
            diagram_customization=DiagramCustomization(theme=Theme.DARK, grid_size=32),
        )
        emitted = platform_customization_to_json(model)
        assert emitted["classOverrides"] == {}
        assert emitted["associationOverrides"] == {}
        assert emitted["diagramCustomization"] == {"theme": "dark", "gridSize": 32}

    def test_partial_v2_class_emits_only_set_fields(self):
        model = PlatformCustomizationModel(
            name="Partial",
            class_overrides={
                "X": ClassCustomization(node_shape=NodeShape.HEXAGON, font_size=16),
            },
        )
        emitted = platform_customization_to_json(model)
        assert emitted["classOverrides"] == {"X": {"nodeShape": "hexagon", "fontSize": 16}}


class TestRoundTripV2:
    def test_full_v2_identity(self, v2_full_payload):
        model = process_platform_customization_diagram(v2_full_payload)
        re_emitted = platform_customization_to_json(model)
        re_parsed = process_platform_customization_diagram(re_emitted)
        assert platform_customization_to_json(re_parsed) == v2_full_payload
