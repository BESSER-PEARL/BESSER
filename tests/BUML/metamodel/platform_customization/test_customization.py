"""Tests for the PlatformCustomization metamodel."""

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


class TestClassCustomization:
    def test_defaults(self):
        c = ClassCustomization()
        assert c.is_container is False
        assert c.default_width is None
        assert c.default_height is None
        # v2 defaults
        assert c.node_shape is None
        assert c.fill_color is None
        assert c.border_color is None
        assert c.border_width is None
        assert c.border_style is None
        assert c.border_radius is None
        assert c.font_size is None
        assert c.font_weight is None
        assert c.font_color is None
        assert c.label_position is None

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

    # ---- v2: enums --------------------------------------------------------

    def test_accepts_enum_member(self):
        c = ClassCustomization(node_shape=NodeShape.HEXAGON, border_style=LineStyle.DASHED)
        assert c.node_shape is NodeShape.HEXAGON
        assert c.border_style is LineStyle.DASHED

    def test_coerces_string_to_enum(self):
        c = ClassCustomization(
            node_shape="ellipse",
            border_style="dotted",
            font_weight="bold",
            label_position="inside",
        )
        assert c.node_shape is NodeShape.ELLIPSE
        assert c.border_style is LineStyle.DOTTED
        assert c.font_weight is FontWeight.BOLD
        assert c.label_position is LabelPosition.INSIDE

    def test_rejects_invalid_enum_value(self):
        with pytest.raises(ValueError, match="node_shape"):
            ClassCustomization(node_shape="triangle")

    # ---- v2: numeric ranges -----------------------------------------------

    @pytest.mark.parametrize("border_width", [0, 4, 8])
    def test_border_width_in_range(self, border_width):
        c = ClassCustomization(border_width=border_width)
        assert c.border_width == border_width

    @pytest.mark.parametrize("border_width", [-1, 9, 100])
    def test_border_width_out_of_range(self, border_width):
        with pytest.raises(ValueError, match="border_width"):
            ClassCustomization(border_width=border_width)

    @pytest.mark.parametrize("font_size", [8, 14, 24])
    def test_font_size_in_range(self, font_size):
        c = ClassCustomization(font_size=font_size)
        assert c.font_size == font_size

    @pytest.mark.parametrize("font_size", [7, 25, 100])
    def test_font_size_out_of_range(self, font_size):
        with pytest.raises(ValueError, match="font_size"):
            ClassCustomization(font_size=font_size)

    @pytest.mark.parametrize("border_radius", [0, 16, 32])
    def test_border_radius_in_range(self, border_radius):
        c = ClassCustomization(border_radius=border_radius)
        assert c.border_radius == border_radius

    @pytest.mark.parametrize("border_radius", [-1, 33, 100])
    def test_border_radius_out_of_range(self, border_radius):
        with pytest.raises(ValueError, match="border_radius"):
            ClassCustomization(border_radius=border_radius)

    # ---- v2: colors -------------------------------------------------------

    @pytest.mark.parametrize(
        "color",
        ["#fef3c7", "hsl(45 90% 80%)", "rgb(120, 80, 200)", "rebeccapurple"],
    )
    def test_accepts_color_strings(self, color):
        c = ClassCustomization(fill_color=color, border_color=color, font_color=color)
        assert c.fill_color == color
        assert c.border_color == color
        assert c.font_color == color

    def test_rejects_empty_color(self):
        with pytest.raises(ValueError, match="fill_color"):
            ClassCustomization(fill_color="")


class TestAssociationCustomization:
    def test_defaults(self):
        a = AssociationCustomization()
        assert a.edge_color is None
        assert a.line_width is None
        assert a.line_style is None
        assert a.source_arrow_style is None
        assert a.target_arrow_style is None
        assert a.label_visible is None
        assert a.label_font_size is None
        assert a.label_font_color is None

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

    # ---- v2 ---------------------------------------------------------------

    def test_v2_full_construction(self):
        a = AssociationCustomization(
            edge_color="#22c55e",
            line_width=3,
            line_style="dashed",
            source_arrow_style="none",
            target_arrow_style="diamond",
            label_visible=True,
            label_font_size=14,
            label_font_color="#111",
        )
        assert a.line_style is LineStyle.DASHED
        assert a.source_arrow_style is ArrowStyle.NONE
        assert a.target_arrow_style is ArrowStyle.DIAMOND
        assert a.label_visible is True
        assert a.label_font_size == 14

    @pytest.mark.parametrize("line_width", [0, 7, -1])
    def test_line_width_out_of_range(self, line_width):
        with pytest.raises(ValueError, match="line_width"):
            AssociationCustomization(line_width=line_width)

    @pytest.mark.parametrize("font_size", [7, 19])
    def test_label_font_size_out_of_range(self, font_size):
        with pytest.raises(ValueError, match="label_font_size"):
            AssociationCustomization(label_font_size=font_size)

    def test_invalid_arrow_style(self):
        with pytest.raises(ValueError, match="target_arrow_style"):
            AssociationCustomization(target_arrow_style="square")

    def test_label_visible_must_be_bool(self):
        with pytest.raises(ValueError, match="label_visible"):
            AssociationCustomization(label_visible="yes")  # type: ignore[arg-type]


class TestDiagramCustomization:
    def test_defaults(self):
        d = DiagramCustomization()
        assert d.background_color is None
        assert d.grid_visible is None
        assert d.grid_size is None
        assert d.snap_to_grid is None
        assert d.theme is None

    def test_full(self):
        d = DiagramCustomization(
            background_color="#fafafa",
            grid_visible=True,
            grid_size=24,
            snap_to_grid=True,
            theme="dark",
        )
        assert d.theme is Theme.DARK
        assert d.grid_size == 24

    @pytest.mark.parametrize("size", [7, 65])
    def test_grid_size_out_of_range(self, size):
        with pytest.raises(ValueError, match="grid_size"):
            DiagramCustomization(grid_size=size)

    def test_invalid_theme(self):
        with pytest.raises(ValueError, match="theme"):
            DiagramCustomization(theme="solarized")


class TestPlatformCustomizationModel:
    def test_constructs_empty(self):
        m = PlatformCustomizationModel(name="Empty")
        assert m.name == "Empty"
        assert m.class_overrides == {}
        assert m.association_overrides == {}
        assert m.diagram_customization is None

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

    def test_constructs_with_diagram_customization(self):
        m = PlatformCustomizationModel(
            name="Themed",
            diagram_customization=DiagramCustomization(theme="dark", grid_size=32),
        )
        assert m.diagram_customization is not None
        assert m.diagram_customization.theme is Theme.DARK
        # get_diagram_customization always returns a value
        assert m.get_diagram_customization().grid_size == 32

    def test_get_diagram_customization_returns_default_when_unset(self):
        m = PlatformCustomizationModel(name="X")
        d = m.get_diagram_customization()
        assert isinstance(d, DiagramCustomization)
        assert d.theme is None

    def test_diagram_customization_must_be_dataclass(self):
        with pytest.raises(ValueError, match="diagram_customization"):
            PlatformCustomizationModel(name="X", diagram_customization={"theme": "dark"})  # type: ignore[arg-type]

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
        assert "diagram_customization=none" in r
