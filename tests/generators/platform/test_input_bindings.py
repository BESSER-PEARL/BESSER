"""Tests for the is_input metadata flag round-trip and the
generated platform's binding UI.

Verifies:
  - Metadata.is_input accepted and stored.
  - domain_model_builder emits metadata=Metadata(is_input=True) for is_input props.
  - Converter reads isInput from editor JSON into BUML model.
  - Platform generator emits INPUT_ATTRIBUTES_REGISTRY constant in PropertyEditor.
  - PropertyEditor.tsx renders BindingSection for is_input attributes.
"""

from __future__ import annotations

import io
import textwrap

import pytest

from besser.BUML.metamodel.structural import (
    Class,
    DomainModel,
    FloatType,
    IntegerType,
    Metadata,
    Method,
    Parameter,
    Property,
    StringType,
)
from besser.generators.platform.platform_generator import PlatformGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pump_model() -> DomainModel:
    """Pump with one is_input attribute (flowrate) and one normal attribute (name)."""
    flowrate = Property(
        name="flowrate",
        type=FloatType,
        metadata=Metadata(is_input=True),
    )
    label = Property(name="label", type=StringType)
    pump = Class(name="Pump", attributes={flowrate, label})
    step = Method(
        name="step",
        parameters=[Parameter(name="dt", type=FloatType)],
        code="pass",
    )
    pump.methods = {step}
    return DomainModel(name="PumpDomain", types={pump})


@pytest.fixture
def no_input_model() -> DomainModel:
    widget = Class(name="Widget", attributes={Property(name="size", type=IntegerType)})
    return DomainModel(name="WidgetDomain", types={widget})


def _generate(model, tmp_path, name="platform"):
    out = tmp_path / name
    PlatformGenerator(model, customization=None, output_dir=str(out)).generate()
    return out


# ---------------------------------------------------------------------------
# Metadata.is_input field
# ---------------------------------------------------------------------------

class TestMetadataIsInput:
    def test_is_input_default_false(self):
        m = Metadata()
        assert m.is_input is False

    def test_is_input_set_true(self):
        m = Metadata(is_input=True)
        assert m.is_input is True

    def test_is_input_coerced_to_bool(self):
        m = Metadata(is_input=1)
        assert m.is_input is True
        m2 = Metadata(is_input=0)
        assert m2.is_input is False

    def test_property_metadata_is_input(self):
        p = Property(name="flowrate", type=FloatType, metadata=Metadata(is_input=True))
        assert p.metadata.is_input is True

    def test_property_without_metadata_has_no_is_input(self):
        p = Property(name="label", type=StringType)
        assert p.metadata is None or not getattr(p.metadata, "is_input", False)

    def test_repr_includes_is_input(self):
        m = Metadata(is_input=True)
        assert "is_input=True" in repr(m)


# ---------------------------------------------------------------------------
# domain_model_builder emits metadata=Metadata(is_input=True)
# ---------------------------------------------------------------------------

class TestCodeBuilder:
    def test_is_input_emitted_for_input_property(self, pump_model, tmp_path):
        from besser.utilities.buml_code_builder.domain_model_builder import domain_model_to_code
        output_file = tmp_path / "out.py"
        domain_model_to_code(pump_model, str(output_file))
        code = output_file.read_text(encoding="utf-8")
        # The emitted Property() call for 'flowrate' must include Metadata(is_input=True)
        assert "Metadata(is_input=True)" in code

    def test_non_input_property_not_emitted(self, pump_model, tmp_path):
        from besser.utilities.buml_code_builder.domain_model_builder import domain_model_to_code
        output_file = tmp_path / "out.py"
        domain_model_to_code(pump_model, str(output_file))
        code = output_file.read_text(encoding="utf-8")
        # 'label' has no metadata — must not emit a spurious Metadata(is_input=...)
        # Check the label Property line does not contain is_input
        label_line = next(
            (l for l in code.splitlines() if '"label"' in l or "'label'" in l), None
        )
        assert label_line is not None
        assert "is_input" not in label_line


# ---------------------------------------------------------------------------
# Converter round-trip: isInput JSON → BUML
# ---------------------------------------------------------------------------

class TestConverter:
    def _make_class_diagram_json(self, is_input=True):
        """Minimal class-diagram JSON shape that the converter accepts."""
        return {
            "title": "PumpDiagram",
            "model": {
                "elements": {
                    "c1": {
                        "type": "Class",
                        "name": "Pump",
                        "attributes": ["a1"],
                        "methods": [],
                        "bounds": {"x": 0, "y": 0, "width": 200, "height": 100},
                    },
                    "a1": {
                        "type": "ClassAttribute",
                        "visibility": "public",
                        "name": "flowrate",
                        "attributeType": "float",
                        "isOptional": False,
                        "isId": False,
                        "isExternalId": False,
                        "isDerived": False,
                        "isInput": is_input,
                    },
                },
                "relationships": {},
            },
        }

    def test_is_input_true_sets_metadata(self):
        from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
            process_class_diagram,
        )
        data = self._make_class_diagram_json(is_input=True)
        model = process_class_diagram(data)
        pump = model.get_class_by_name("Pump")
        assert pump is not None
        prop = next((p for p in pump.attributes if p.name == "flowrate"), None)
        assert prop is not None
        assert prop.metadata is not None
        assert prop.metadata.is_input is True

    def test_is_input_false_keeps_metadata_clean(self):
        from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
            process_class_diagram,
        )
        data = self._make_class_diagram_json(is_input=False)
        model = process_class_diagram(data)
        pump = model.get_class_by_name("Pump")
        prop = next((p for p in pump.attributes if p.name == "flowrate"), None)
        assert prop is not None
        # is_input=False → no metadata set (or metadata.is_input = False)
        is_input = prop.metadata.is_input if prop.metadata else False
        assert is_input is False


# ---------------------------------------------------------------------------
# Platform generator — INPUT_ATTRIBUTES_REGISTRY in PropertyEditor
# ---------------------------------------------------------------------------

class TestPlatformPropertyEditor:
    def test_input_registry_constant_emitted(self, pump_model, tmp_path):
        out = _generate(pump_model, tmp_path)
        src = (out / "frontend" / "src" / "components" / "PropertyEditor.tsx").read_text(encoding="utf-8")
        assert "INPUT_ATTRIBUTES_REGISTRY" in src
        assert '"Pump"' in src
        assert '"flowrate"' in src

    def test_binding_section_used(self, pump_model, tmp_path):
        out = _generate(pump_model, tmp_path)
        src = (out / "frontend" / "src" / "components" / "PropertyEditor.tsx").read_text(encoding="utf-8")
        assert "BindingSection" in src

    def test_no_input_registry_for_non_input_attrs(self, no_input_model, tmp_path):
        out = _generate(no_input_model, tmp_path, "noui")
        src = (out / "frontend" / "src" / "components" / "PropertyEditor.tsx").read_text(encoding="utf-8")
        # Registry should be empty {} since no is_input attrs
        assert "INPUT_ATTRIBUTES_REGISTRY" in src
        # The Widget class should not appear as a key
        assert '"Widget"' not in src

    def test_history_chart_imported(self, pump_model, tmp_path):
        out = _generate(pump_model, tmp_path)
        src = (out / "frontend" / "src" / "components" / "PropertyEditor.tsx").read_text(encoding="utf-8")
        assert "HistoryChart" in src
