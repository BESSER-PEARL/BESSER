"""Tests for the PlatformCustomization metamodel."""

import pytest

from besser.BUML.metamodel.platform_customization import (
    AssociationCustomization,
    ClassCustomization,
    PlatformCustomizationModel,
)


class TestClassCustomization:
    def test_defaults(self):
        c = ClassCustomization()
        assert c.is_container is False
        assert c.default_width is None
        assert c.default_height is None

    def test_explicit_values(self):
        c = ClassCustomization(is_container=True, default_width=400, default_height=300)
        assert c.is_container is True
        assert c.default_width == 400
        assert c.default_height == 300

    @pytest.mark.parametrize("width", [0, -1, -100])
    def test_rejects_non_positive_width(self, width):
        with pytest.raises(ValueError, match="default_width"):
            ClassCustomization(default_width=width)

    @pytest.mark.parametrize("height", [0, -1, -100])
    def test_rejects_non_positive_height(self, height):
        with pytest.raises(ValueError, match="default_height"):
            ClassCustomization(default_height=height)


class TestAssociationCustomization:
    def test_defaults(self):
        a = AssociationCustomization()
        assert a.edge_color is None

    @pytest.mark.parametrize(
        "color",
        ["#22c55e", "hsl(142 76% 35%)", "red", "rgb(34, 197, 94)"],
    )
    def test_accepts_any_css_color_string(self, color):
        a = AssociationCustomization(edge_color=color)
        assert a.edge_color == color

    def test_rejects_empty_color(self):
        with pytest.raises(ValueError, match="edge_color cannot be empty"):
            AssociationCustomization(edge_color="")

    def test_rejects_non_string_color(self):
        with pytest.raises(ValueError, match="edge_color must be a string"):
            AssociationCustomization(edge_color=123)  # type: ignore[arg-type]


class TestPlatformCustomizationModel:
    def test_constructs_empty(self):
        m = PlatformCustomizationModel(name="Empty")
        assert m.name == "Empty"
        assert m.class_overrides == {}
        assert m.association_overrides == {}

    def test_constructs_with_overrides(self):
        m = PlatformCustomizationModel(
            name="RegionSensor",
            class_overrides={
                "Region": ClassCustomization(is_container=True, default_width=400, default_height=300),
                "Sensor": ClassCustomization(default_width=80, default_height=80),
            },
            association_overrides={"has": AssociationCustomization(edge_color="#22c55e")},
        )
        assert m.get_class_customization("Region").is_container is True
        assert m.get_class_customization("Sensor").default_width == 80
        assert m.get_association_customization("has").edge_color == "#22c55e"

    def test_missing_lookup_returns_defaults(self):
        m = PlatformCustomizationModel(name="X")
        unknown_class = m.get_class_customization("Nope")
        assert unknown_class.is_container is False
        assert unknown_class.default_width is None
        unknown_assoc = m.get_association_customization("Nope")
        assert unknown_assoc.edge_color is None

    def test_rejects_non_dict_class_overrides(self):
        with pytest.raises(ValueError, match="class_overrides must be a dict"):
            PlatformCustomizationModel(name="X", class_overrides=[])  # type: ignore[arg-type]

    def test_rejects_wrong_value_type(self):
        with pytest.raises(ValueError, match="ClassCustomization"):
            PlatformCustomizationModel(
                name="X",
                class_overrides={"Region": {"is_container": True}},  # type: ignore[dict-item]
            )

    def test_rejects_empty_key(self):
        with pytest.raises(ValueError, match="non-empty class names"):
            PlatformCustomizationModel(name="X", class_overrides={"": ClassCustomization()})

    def test_name_rules_inherited_from_named_element(self):
        # NamedElement's name setter rejects spaces and hyphens
        with pytest.raises(ValueError, match="spaces"):
            PlatformCustomizationModel(name="has a space")
        with pytest.raises(ValueError, match="Hyphens"):
            PlatformCustomizationModel(name="has-hyphen")

    def test_repr_is_readable(self):
        m = PlatformCustomizationModel(
            name="Pretty",
            class_overrides={"A": ClassCustomization()},
            association_overrides={"b": AssociationCustomization()},
        )
        r = repr(m)
        assert "Pretty" in r
        assert "class_overrides=1" in r
        assert "association_overrides=1" in r
