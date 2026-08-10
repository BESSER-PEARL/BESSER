"""Smoke tests for the shipped NN editor templates.

Every template JSON under
``besser/utilities/web_modeling_editor/frontend/packages/webapp2/src/main/
templates/pattern/nn/`` must parse cleanly through ``process_nn_diagram``
— otherwise a schema/field drift silently breaks "Load Template" in the
editor. These tests also exercise the full builder + converter round-trip
on each template so any downstream regression shows up with the template
name attached.
"""

import json
from pathlib import Path

import pytest

from besser.BUML.metamodel.nn import NN
from besser.utilities.web_modeling_editor.backend.services.converters import (
    nn_model_to_json,
    process_nn_diagram,
)


_TEMPLATE_DIR = (
    Path(__file__).resolve().parents[5]
    / "besser"
    / "utilities"
    / "web_modeling_editor"
    / "frontend"
    / "packages"
    / "webapp2"
    / "src"
    / "main"
    / "templates"
    / "pattern"
    / "nn"
)

_TEMPLATE_FILES = sorted(_TEMPLATE_DIR.glob("*.json")) if _TEMPLATE_DIR.is_dir() else []


@pytest.mark.skipif(
    not _TEMPLATE_FILES,
    reason=f"NN template directory {_TEMPLATE_DIR} not found (frontend submodule not checked out)",
)
@pytest.mark.parametrize(
    "template_path",
    _TEMPLATE_FILES,
    ids=[p.name for p in _TEMPLATE_FILES],
)
class TestNNTemplates:
    @staticmethod
    def _wrap(template_model: dict) -> dict:
        """process_nn_diagram expects the outer diagram wrapper, templates
        are the flat UMLModel — wrap to match the editor payload shape."""
        return {
            "title": template_model.get("type", "NNDiagram"),
            "model": template_model,
        }

    def test_template_is_flat_umlmodel(self, template_path: Path):
        """Every NN template is a flat UMLModel (no ``project`` wrapper).

        A regression that re-introduces the wrapped-project format would
        make the template load as an empty canvas in the editor — this
        test locks in the fix from iteration 1.
        """
        model = json.loads(template_path.read_text(encoding="utf-8"))
        assert model.get("type") == "NNDiagram", (
            f"{template_path.name} must have top-level type=NNDiagram, got {model.get('type')!r}"
        )
        assert "elements" in model, f"{template_path.name} missing 'elements'"
        assert "relationships" in model, f"{template_path.name} missing 'relationships'"
        assert "project" not in model, (
            f"{template_path.name} is a wrapped-project export — templates must be flat UMLModel"
        )

    def test_template_parses_via_process_nn_diagram(self, template_path: Path):
        """Each template must build a valid NN metamodel instance."""
        model = json.loads(template_path.read_text(encoding="utf-8"))
        nn = process_nn_diagram(self._wrap(model))
        assert isinstance(nn, NN), (
            f"process_nn_diagram on {template_path.name} should return an NN, got {type(nn).__name__}"
        )

    def test_template_roundtrips_through_converter(self, template_path: Path):
        """Round-trip JSON → NN → JSON must succeed and preserve diagram type."""
        model = json.loads(template_path.read_text(encoding="utf-8"))
        nn = process_nn_diagram(self._wrap(model))
        out = nn_model_to_json(nn)
        assert out["type"] == "NNDiagram"
        assert out["version"] == "3.0.0"
        assert isinstance(out.get("elements"), dict)
        assert isinstance(out.get("relationships"), dict)


@pytest.mark.skipif(
    not _TEMPLATE_FILES,
    reason=f"NN template directory {_TEMPLATE_DIR} not found",
)
def test_nn_model_to_json_byte_identical_raw_dumps():
    """Back-to-back ``nn_model_to_json`` calls must return byte-identical
    JSON *without* ``sort_keys`` — stricter than the existing
    ``test_nn_model_to_json_is_deterministic`` (which sorts keys and
    therefore masks dict insertion-order regressions).

    Guard: if anyone reintroduces uuid.uuid4() for IDs or switches to an
    unordered set for element emission, raw-dump equality breaks even
    though sorted-key equality may still hold.
    """
    template_path = next(p for p in _TEMPLATE_FILES if p.name == "alexnet_nn.json")
    model = json.loads(template_path.read_text(encoding="utf-8"))
    nn = process_nn_diagram({"title": "Alexnet", "model": model})
    first = json.dumps(nn_model_to_json(nn))
    second = json.dumps(nn_model_to_json(nn))
    assert first == second, "nn_model_to_json output must be byte-identical across calls"
