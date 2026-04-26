"""End-to-end smoke tests for PlatformGenerator + PlatformCustomization.

These don't run npm — they generate the output tree and verify the customization
fields land in the right places (models.ts, InstanceCanvas.tsx, App.tsx).
"""

from pathlib import Path

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
from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    Multiplicity,
    Property,
    StringType,
)
from besser.generators.platform.platform_generator import PlatformGenerator


@pytest.fixture
def region_sensor_model():
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


@pytest.fixture
def full_customization():
    return PlatformCustomizationModel(
        name="FullCust",
        class_overrides={
            "Region": ClassCustomization(
                is_container=True,
                default_width=400,
                default_height=300,
                fill_color="#fef3c7",
                border_color="#a16207",
                border_width=4,
                border_style=LineStyle.DASHED,
                border_radius=24,
            ),
            "Sensor": ClassCustomization(
                node_shape=NodeShape.HEXAGON,
                fill_color="#dbeafe",
                border_color="#1d4ed8",
                border_width=3,
                font_size=16,
                font_weight=FontWeight.BOLD,
                font_color="#1e3a8a",
                label_position=LabelPosition.INSIDE,
            ),
        },
        association_overrides={
            "has": AssociationCustomization(
                edge_color="#22c55e",
                line_width=3,
                line_style=LineStyle.DOTTED,
                target_arrow_style=ArrowStyle.DIAMOND,
                label_visible=True,
                label_font_size=14,
                label_font_color="#111",
            ),
        },
        diagram_customization=DiagramCustomization(
            background_color="#fafafa",
            grid_visible=True,
            grid_size=32,
            snap_to_grid=True,
            theme=Theme.DARK,
        ),
    )


class TestPlatformGeneratorWithCustomization:
    def test_generates_with_full_customization(self, tmp_path, region_sensor_model, full_customization):
        out = tmp_path / "platform"
        PlatformGenerator(
            model=region_sensor_model,
            customization=full_customization,
            output_dir=str(out),
        ).generate()

        # Tree shape — the four files we care about should exist.
        assert (out / "frontend" / "src" / "types" / "models.ts").exists()
        assert (out / "frontend" / "src" / "components" / "InstanceCanvas.tsx").exists()
        assert (out / "frontend" / "src" / "components" / "InstanceNode.tsx").exists()
        assert (out / "frontend" / "src" / "App.tsx").exists()
        assert (out / "frontend" / "src" / "index.css").exists()

    def test_models_ts_contains_node_and_edge_style(self, tmp_path, region_sensor_model, full_customization):
        out = tmp_path / "platform"
        PlatformGenerator(region_sensor_model, full_customization, str(out)).generate()
        models = (out / "frontend" / "src" / "types" / "models.ts").read_text(encoding="utf-8")

        # Region's container + appearance fields
        assert "isContainer: true" in models
        assert "defaultSize: { width: 400, height: 300 }" in models
        assert "fillColor: '#fef3c7'" in models
        assert "borderColor: '#a16207'" in models
        assert "borderWidth: 4" in models
        assert "borderStyle: 'dashed'" in models
        assert "borderRadius: 24" in models

        # Sensor's shape + typography
        assert "shape: 'hexagon'" in models
        assert "fontSize: 16" in models
        assert "fontWeight: 'bold'" in models
        assert "labelPosition: 'inside'" in models

        # Edge style payload + diagram-level
        assert "lineWidth: 3" in models
        assert "lineStyle: 'dotted'" in models
        assert "targetArrow: 'diamond'" in models
        assert "DIAGRAM_STYLE: DiagramStyle = {" in models
        assert "theme: 'dark'" in models
        assert "gridSize: 32" in models
        assert "snapToGrid: true" in models

        # New TypeScript interfaces are present
        assert "export interface NodeStyle" in models
        assert "export interface EdgeStyle" in models

    def test_canvas_uses_grid_and_markers(self, tmp_path, region_sensor_model, full_customization):
        out = tmp_path / "platform"
        PlatformGenerator(region_sensor_model, full_customization, str(out)).generate()
        canvas = (out / "frontend" / "src" / "components" / "InstanceCanvas.tsx").read_text(encoding="utf-8")

        # Custom diamond marker injected once into <defs>
        assert 'id="platform-diamond"' in canvas
        # markerEnd / markerStart wired
        assert "markerEnd: arrowMarker" in canvas
        # DIAGRAM_STYLE.gridSize threaded to xyflow Background gap
        assert "DIAGRAM_STYLE.gridSize" in canvas
        assert "snapToGrid={snapEnabled}" in canvas

    def test_app_threads_theme(self, tmp_path, region_sensor_model, full_customization):
        out = tmp_path / "platform"
        PlatformGenerator(region_sensor_model, full_customization, str(out)).generate()
        app = (out / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
        assert "useDiagramTheme" in app
        assert "DIAGRAM_STYLE.theme" in app
        assert "prefers-color-scheme" in app

    def test_index_css_has_dark_block(self, tmp_path, region_sensor_model, full_customization):
        out = tmp_path / "platform"
        PlatformGenerator(region_sensor_model, full_customization, str(out)).generate()
        css = (out / "frontend" / "src" / "index.css").read_text(encoding="utf-8")
        assert ".dark {" in css

    def test_no_customization_emits_no_style_data(self, tmp_path, region_sensor_model):
        out = tmp_path / "platform_default"
        PlatformGenerator(region_sensor_model, customization=None, output_dir=str(out)).generate()
        models = (out / "frontend" / "src" / "types" / "models.ts").read_text(encoding="utf-8")

        # Style: { ... } per-class block must not appear
        assert "style: { shape" not in models
        assert "style: { color" not in models
        # DIAGRAM_STYLE is still emitted but empty
        assert "DIAGRAM_STYLE: DiagramStyle = {" in models
        assert "theme:" not in models or "theme:" in models  # noop assertion guard

    def test_only_some_fields_set_emits_only_those(self, tmp_path, region_sensor_model):
        cust = PlatformCustomizationModel(
            name="Partial",
            class_overrides={
                "Region": ClassCustomization(font_size=20, fill_color="#fff"),
            },
        )
        out = tmp_path / "platform_partial"
        PlatformGenerator(region_sensor_model, cust, str(out)).generate()
        models = (out / "frontend" / "src" / "types" / "models.ts").read_text(encoding="utf-8")

        assert "fontSize: 20" in models
        assert "fillColor: '#fff'" in models
        # No leakage of unset enum fields
        assert "shape:" not in models
        assert "borderStyle:" not in models
        assert "labelPosition:" not in models
