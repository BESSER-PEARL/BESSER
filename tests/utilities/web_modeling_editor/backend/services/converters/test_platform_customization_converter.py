"""Round-trip tests for PlatformCustomization JSON <-> BUML converters."""

import pytest

from besser.BUML.metamodel.platform_customization import (
    AssociationCustomization,
    ClassCustomization,
    PlatformCustomizationModel,
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
