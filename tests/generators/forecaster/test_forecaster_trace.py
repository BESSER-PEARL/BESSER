"""TimeSeriesForecasterGenerator records which lines of the forecaster module come from which layer, though
its own tidy-up removes duplicated imports and blank lines after the render."""

import os

from besser.generators.nn.forecaster import ForecastConfig, TimeSeriesForecasterGenerator
from tests.forecaster_networks import HORIZON, WINDOW, gru, mlp

CONFIG = ForecastConfig(window=WINDOW, horizon=HORIZON, seeds=(0,))


def _identify(model):
    model.id = "id-model"
    for layer in model.layers:
        layer.id = f"id-{layer.name}"
    return model


def _generate(model, out):
    generator = TimeSeriesForecasterGenerator(model, config=CONFIG, output_dir=str(out))
    path = generator.generate()
    return generator, os.path.basename(path), open(path, encoding="utf-8").read()


def _lines(code, region):
    return "\n".join(code.split("\n")[region.start_line - 1:region.end_line]).strip()


def test_layers_of_the_forecaster_network_are_traced_to_their_lines(tmp_path):
    generator, name, code = _generate(_identify(mlp()), tmp_path)
    assert name == "forecaster_mlp.py" and set(generator.traced_files) == {name}
    regions = {}
    for region in generator.traced_files[name].regions:
        regions.setdefault((region.element_id, region.rule.split("#")[1]), []).append(region)

    assert _lines(code, regions[("id-hidden", "layer")][0]) == (
        "self.hidden = nn.Linear(in_features=26, out_features=32, bias=True)")
    assert _lines(code, regions[("id-dropout", "layer")][0]) == "self.dropout = nn.Dropout(p=0.1)"
    assert [_lines(code, r) for r in regions[("id-hidden", "layer_forward")]] == [
        "x = self.hidden(x)", "x = self.actv_func_relu(x)"]
    assert _lines(code, regions[("id-output", "layer_forward")][0]) == "x = self.output(x)"
    (source,) = generator.traced_files[name].sources
    assert (source.element.element_id, source.rule) == ("id-model", "forecaster.py.j2#network")


def test_a_recurrent_layer_traces_both_of_its_forward_lines(tmp_path):
    generator, name, code = _generate(_identify(gru()), tmp_path)
    (call,) = [r for r in generator.traced_files[name].regions
               if r.element_id == "id-recurrent" and r.rule.endswith("#layer_forward")]
    assert _lines(code, call).split("\n")[0].startswith("state, _ = self.recurrent(")


def test_elements_without_ids_render_untraced_and_the_output_is_unchanged(tmp_path):
    traced, name, traced_code = _generate(_identify(mlp()), tmp_path / "traced")
    untraced, _, untraced_code = _generate(mlp(), tmp_path / "untraced")
    assert traced.traced_files[name].regions
    assert traced_code == untraced_code
    assert untraced.traced_files[name].regions == () and untraced.traced_files[name].sources == ()
    assert "\x02" not in traced_code
