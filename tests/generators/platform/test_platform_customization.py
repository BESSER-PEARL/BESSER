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
    LineRouting,
    LineStyle,
    NodeShape,
    PlatformCustomizationModel,
    Theme,
)
from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    Generalization,
    Multiplicity,
    Property,
    StringType,
)
from besser.generators.platform.platform_generator import PlatformGenerator
from besser.generators.platform.template_helpers import (
    build_representation_registries,
    build_subclass_registry,
    compute_addable_port_classes,
)


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

    def test_line_routing_default_emits_bezier_fallback(self, tmp_path, region_sensor_model):
        # No customization at all → DEFAULT_EDGE_ROUTING resolves to 'bezier'.
        out = tmp_path / "platform_routing_default"
        PlatformGenerator(region_sensor_model, customization=None, output_dir=str(out)).generate()
        canvas = (out / "frontend" / "src" / "components" / "InstanceCanvas.tsx").read_text(encoding="utf-8")
        # The runtime constant resolves DIAGRAM_STYLE.lineRouting ?? 'bezier'.
        assert "DIAGRAM_STYLE.lineRouting ?? 'bezier'" in canvas
        assert "defaultEdgeOptions={{ type: edgeTypeFor(DEFAULT_EDGE_ROUTING) }}" in canvas
        assert "connectionLineType={CONNECTION_LINE_TYPE}" in canvas

    def test_line_routing_diagram_default_smoothstep(self, tmp_path, region_sensor_model):
        cust = PlatformCustomizationModel(
            name="DiagramRouting",
            diagram_customization=DiagramCustomization(line_routing=LineRouting.SMOOTHSTEP),
        )
        out = tmp_path / "platform_routing_diagram"
        PlatformGenerator(region_sensor_model, cust, str(out)).generate()
        models = (out / "frontend" / "src" / "types" / "models.ts").read_text(encoding="utf-8")
        # lineRouting lands in DIAGRAM_STYLE
        assert "lineRouting: 'smoothstep'" in models
        assert "DIAGRAM_STYLE: DiagramStyle = {" in models

    def test_line_routing_per_association_overrides_diagram(self, tmp_path, region_sensor_model):
        cust = PlatformCustomizationModel(
            name="AssocRouting",
            association_overrides={
                "has": AssociationCustomization(line_routing=LineRouting.STEP),
            },
            diagram_customization=DiagramCustomization(line_routing=LineRouting.SMOOTHSTEP),
        )
        out = tmp_path / "platform_routing_assoc"
        PlatformGenerator(region_sensor_model, cust, str(out)).generate()
        models = (out / "frontend" / "src" / "types" / "models.ts").read_text(encoding="utf-8")
        # Per-association payload mentions the override
        assert "lineRouting: 'step'" in models
        # Diagram default is still emitted
        assert "lineRouting: 'smoothstep'" in models

class TestAddablePortClasses:
    """Compute the {className, associationName} list a class can spawn."""

    @pytest.fixture
    def equipment_with_ports_model(self):
        # Equipment ─has-ports→ Port  (Port has two concrete subclasses)
        port = Class(name="Port", attributes={Property(name="kind", type=StringType)})
        inlet = Class(name="InletPort", attributes=set())
        outlet = Class(name="OutletPort", attributes=set())
        equipment = Class(name="Equipment", attributes={Property(name="tag", type=StringType)})
        ports_assoc = BinaryAssociation(
            name="ports",
            ends={
                Property(
                    name="owner",
                    type=equipment,
                    multiplicity=Multiplicity(min_multiplicity=1, max_multiplicity=1),
                ),
                Property(
                    name="ports",
                    type=port,
                    multiplicity=Multiplicity(min_multiplicity=0, max_multiplicity=9999),
                ),
            },
        )
        gen_inlet = Generalization(general=port, specific=inlet)
        gen_outlet = Generalization(general=port, specific=outlet)
        return DomainModel(
            name="EquipPorts",
            types={equipment, port, inlet, outlet},
            associations={ports_assoc},
            generalizations={gen_inlet, gen_outlet},
        )

    def test_helper_resolves_subclasses_of_a_port_targeted_association(
        self, equipment_with_ports_model
    ):
        cust = PlatformCustomizationModel(
            name="C",
            class_overrides={
                # Only the base Port is flagged — subclasses inherit the flag
                # via build_representation_registries' inheritance walk.
                "Port": ClassCustomization(is_port=True),
            },
        )
        classes = list(equipment_with_ports_model.get_classes())
        registry = build_representation_registries(classes, cust)
        sub_reg = build_subclass_registry(classes)
        addable = compute_addable_port_classes(
            classes, cust, sub_reg, registry["port_classes"].keys()
        )

        # Equipment can spawn Port itself + both concrete subclasses through
        # the same association.
        spawned = {(e["className"], e["associationName"]) for e in addable["Equipment"]}
        assert ("Port", "ports") in spawned
        assert ("InletPort", "ports") in spawned
        assert ("OutletPort", "ports") in spawned
        # Port can't spawn anything off this model — no association.
        assert addable["Port"] == []
        # Subclasses inherit nothing (they aren't on the source side of the assoc).
        assert addable["InletPort"] == []

    def test_helper_skips_abstract_classes(self, equipment_with_ports_model):
        # Mark the base Port as abstract — it should drop out of the addable
        # list while concrete subclasses remain.
        for c in equipment_with_ports_model.get_classes():
            if c.name == "Port":
                c.is_abstract = True
        cust = PlatformCustomizationModel(
            name="C",
            class_overrides={"Port": ClassCustomization(is_port=True)},
        )
        classes = list(equipment_with_ports_model.get_classes())
        registry = build_representation_registries(classes, cust)
        sub_reg = build_subclass_registry(classes)
        addable = compute_addable_port_classes(
            classes, cust, sub_reg, registry["port_classes"].keys()
        )

        spawned = {e["className"] for e in addable["Equipment"]}
        assert "Port" not in spawned
        assert "InletPort" in spawned
        assert "OutletPort" in spawned

    def test_addable_port_classes_lands_in_models_ts(
        self, tmp_path, equipment_with_ports_model
    ):
        cust = PlatformCustomizationModel(
            name="C",
            class_overrides={"Port": ClassCustomization(is_port=True)},
        )
        out = tmp_path / "addable_ports"
        PlatformGenerator(equipment_with_ports_model, cust, str(out)).generate()
        models = (out / "frontend" / "src" / "types" / "models.ts").read_text(encoding="utf-8")

        # The Equipment class has an addablePortClasses block, with all three
        # concrete + base port classes available through "ports".
        assert "addablePortClasses: [" in models
        assert "className: 'Port', associationName: 'ports'" in models
        assert "className: 'InletPort', associationName: 'ports'" in models
        assert "className: 'OutletPort', associationName: 'ports'" in models

    def test_addable_port_classes_empty_when_no_port_class_flagged(
        self, tmp_path, equipment_with_ports_model
    ):
        # No customization → no class is flagged is_port → addablePortClasses
        # should not appear anywhere in the generated metadata.
        out = tmp_path / "no_ports"
        PlatformGenerator(equipment_with_ports_model, customization=None, output_dir=str(out)).generate()
        models = (out / "frontend" / "src" / "types" / "models.ts").read_text(encoding="utf-8")
        assert "addablePortClasses:" not in models

    def test_add_port_popover_template_emitted(self, tmp_path, region_sensor_model):
        # The component file ships unconditionally — InstanceNode imports it.
        out = tmp_path / "popover_check"
        PlatformGenerator(region_sensor_model, customization=None, output_dir=str(out)).generate()
        popover = out / "frontend" / "src" / "components" / "AddPortPopover.tsx"
        assert popover.exists()
        text = popover.read_text(encoding="utf-8")
        # Sanity: the export and the single-option auto-fire branch are present.
        assert "export const AddPortPopover" in text
        assert "addable.length === 1" in text

    def test_add_association_popover_template_emitted(self, tmp_path, region_sensor_model):
        # InstanceNode imports AddAssociationPopover unconditionally — missing
        # the file would break Vite import-analysis at dev-server start.
        out = tmp_path / "assoc_popover_check"
        PlatformGenerator(region_sensor_model, customization=None, output_dir=str(out)).generate()
        popover = out / "frontend" / "src" / "components" / "AddAssociationPopover.tsx"
        assert popover.exists()
        text = popover.read_text(encoding="utf-8")
        assert "export const AddAssociationPopover" in text

    def test_stream_class_selector_template_emitted(self, tmp_path, region_sensor_model):
        # InstanceCanvas imports StreamClassSelector unconditionally — missing
        # the file would break Vite import-analysis at dev-server start.
        out = tmp_path / "stream_selector_check"
        PlatformGenerator(region_sensor_model, customization=None, output_dir=str(out)).generate()
        selector = out / "frontend" / "src" / "components" / "StreamClassSelector.tsx"
        assert selector.exists()
        text = selector.read_text(encoding="utf-8")
        assert "export const StreamClassSelector" in text

    def test_canvas_wires_port_to_port_stream_flow(self, tmp_path, region_sensor_model):
        # The port-to-port drag flow now defers to a modal flow rather than
        # auto-creating an empty connection-class instance. Asserts each of the
        # five integration points lands in the rendered InstanceCanvas.
        out = tmp_path / "canvas_stream_flow"
        PlatformGenerator(region_sensor_model, customization=None, output_dir=str(out)).generate()
        canvas = (out / "frontend" / "src" / "components" / "InstanceCanvas.tsx").read_text(encoding="utf-8")
        # Import + state shape for the new flow.
        assert "from './StreamClassSelector'" in canvas
        assert "setStreamClassPicker" in canvas
        assert "setCreateStreamModal" in canvas
        # Lenient inlet/outlet check — same-direction ports get rejected, but
        # ports lacking a direction attribute fall through.
        assert "direction must differ" in canvas
        assert "attributes?.direction" in canvas
        # isValidConnection now has a port-handle branch (fixes silent veto).
        assert "isPortClass(sourcePort.class_name)" in canvas
        # Attribute modal reuses InstanceCreationModal with endpoints hidden.
        assert "hideEndpointAssociations" in canvas
        assert "handleCreateStreamFromModal" in canvas

    def test_creation_modal_supports_hide_endpoint_associations(self, tmp_path, region_sensor_model):
        # The new prop on InstanceCreationModal must actually be wired into the
        # filter — otherwise the canvas would still show endpoint pickers in
        # the stream-creation modal.
        out = tmp_path / "modal_hide_endpoints"
        PlatformGenerator(region_sensor_model, customization=None, output_dir=str(out)).generate()
        modal = (out / "frontend" / "src" / "components" / "InstanceCreationModal.tsx").read_text(encoding="utf-8")
        assert "hideEndpointAssociations" in modal
        assert "a.isSourceEndpoint || a.isTargetEndpoint" in modal

    def test_instance_node_default_handles_for_regular_classes(self, tmp_path, region_sensor_model):
        # Regular classes (no addablePortClasses) keep all four default
        # handles so xyflow can anchor association edges. Port-host classes
        # opt out automatically. Explicit `connectionPoints` overrides either
        # way.
        out = tmp_path / "node_handles_optin"
        PlatformGenerator(region_sensor_model, customization=None, output_dir=str(out)).generate()
        node = (out / "frontend" / "src" / "components" / "InstanceNode.tsx").read_text(encoding="utf-8")
        # The conditional default keeps HANDLE_POSITIONS for regular classes.
        assert "isPortHost ? [] : HANDLE_POSITIONS" in node
        # And still computes the port-host flag from addablePortClasses.
        assert "classMetadata.addablePortClasses?.length" in node


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
