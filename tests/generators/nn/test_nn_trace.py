"""The PyTorch and TensorFlow generators record which lines of the generated network come from which
layer, tensor operation, sub-network or dataset, though the code is cleaned of blank lines after it is rendered."""

import pytest

from besser.BUML.metamodel.nn import (
    NN, Configuration, Conv2D, Dataset, DropoutLayer, FlattenLayer, Image, LinearLayer, LSTMLayer, PoolingLayer,
    TensorOp,
)
from besser.generators.nn.nn_code_generator import ModulesDetails, remap_regions
from besser.generators.nn.pytorch.pytorch_code_generator import PytorchGenerator
from besser.generators.nn.tf.tf_code_generator import TFGenerator
from besser.utilities.generation_trace import TraceRegion


def _network(with_data=True, sub_nn=True, tensor_op=True) -> NN:
    """A network whose every element has an id: the element is returned by its name through ``ids``."""
    model = NN(name="my_model")
    model.add_layer(Conv2D(name="l1", actv_func="relu", in_channels=3, out_channels=32, kernel_dim=[3, 3]))
    model.add_layer(PoolingLayer(name="l2", pooling_type="max", dimension="2D", kernel_dim=[2, 2]))
    model.add_layer(FlattenLayer(name="l6"))
    model.add_layer(LinearLayer(name="l7", actv_func="relu", in_features=1024, out_features=64))
    model.add_layer(DropoutLayer(name="d1", rate=0.2))
    if tensor_op:
        model.add_tensor_op(TensorOp(name="r1", tns_type="reshape", reshape_dim=[-1, 64]))
    model.add_layer(LinearLayer(name="l8", in_features=64, out_features=10))
    if sub_nn:
        encoder = NN(name="enc")
        encoder.add_layer(LinearLayer(name="s1", in_features=10, out_features=5, actv_func="relu"))
        encoder.add_layer(LinearLayer(name="s2", in_features=5, out_features=3))
        model.add_sub_nn(encoder)
    model.add_configuration(Configuration(batch_size=32, epochs=10, learning_rate=0.001, optimizer="adam",
                                          metrics=["f1-score"], loss_function="crossentropy"))
    if with_data:
        model.add_train_data(Dataset(name="train_data", path_data="data/train", task_type="multi_class",
                                     input_format="images", image=Image(shape=[32, 32, 3], normalize=False)))
        model.add_test_data(Dataset(name="test_data", path_data="data/test"))
    return model


def _elements(model: NN):
    """Every element of the network that can carry an id."""
    elements = [model, *model.modules]
    for module in model.modules:
        if isinstance(module, NN):
            elements.extend([*module.layers])
    elements.extend(data for data in (model.train_data, model.test_data) if data is not None)
    return elements


def _identify(model: NN) -> NN:
    for element in _elements(model):
        element.id = f"id-{element.name}"
    return model


def _generate(generator_class, model, out, **options):
    generator = generator_class(model, output_dir=str(out), **options)
    generator.generate()
    (name,) = [path for path in out.iterdir() if path.suffix == ".py"]
    return generator, name.name, name.read_text(encoding="utf-8")


def _regions(generator, name):
    result = {}
    for region in generator.traced_files[name].regions:
        result.setdefault((region.element_id, region.rule.split("#")[1]), []).append(region)
    return result


def _line(text, region):
    return "\n".join(text.split("\n")[region.start_line - 1:region.end_line]).strip()


def test_pytorch_subclassing_traces_layers_calls_tensor_ops_sub_networks_and_datasets(tmp_path):
    generator, name, code = _generate(PytorchGenerator, _identify(_network()), tmp_path)
    regions = _regions(generator, name)
    assert name == "pytorch_nn_subclassing.py" and set(generator.traced_files) == {name}

    assert _line(code, regions[("id-l1", "layer")][0]).startswith("self.l1 = nn.Conv2d(in_channels=3")
    assert _line(code, regions[("id-l7", "layer")][0]).startswith("self.l7 = nn.Linear(in_features=1024")
    assert _line(code, regions[("id-d1", "layer")][0]) == "self.d1 = nn.Dropout(p=0.2)"
    # A layer is called in forward, and so is its activation (regions of the same layer).
    assert [_line(code, r) for r in regions[("id-l1", "layer_forward")]] == ["x = self.l1(x)", "x = self.actv_func_relu(x)"]
    assert [_line(code, r) for r in regions[("id-l8", "layer_forward")]] == ["x_1 = self.l8(x_1)"]
    assert _line(code, regions[("id-r1", "tensorop")][0]) == "x_1 = x.reshape(-1, 64)"
    sub = regions[("id-enc", "sub_nn")][0]
    assert _line(code, sub).startswith("self.enc = nn.Sequential(") and _line(code, sub).endswith(")")
    inner = [r for r in regions[("id-s1", "layer")]]
    assert all(sub.start_line < r.start_line <= r.end_line < sub.end_line for r in inner)
    assert [_line(code, r) for r in inner] == ["nn.Linear(in_features=10, out_features=5, bias=True),", "nn.ReLU(),"]
    assert _line(code, regions[("id-enc", "sub_nn_forward")][0]) == "x_1 = self.enc(x_1)"
    train = regions[("id-train_data", "train_dataset")][0]
    assert _line(code, train).startswith("def prepare_datasets():") and _line(code, train).endswith("return train_loader, test_loader")
    assert (regions[("id-test_data", "test_dataset")][0].start_line, regions[("id-test_data", "test_dataset")][0].end_line) \
        == (train.start_line, train.end_line)
    assert [s.element.element_id for s in generator.traced_files[name].sources] == ["id-my_model"]


def test_a_layer_that_calls_an_rnn_has_a_region_of_several_lines(tmp_path):
    model = NN(name="rn")
    model.add_layer(LSTMLayer(name="l2", return_type="full", input_size=3, hidden_size=8, batch_first=True))
    model.add_layer(LSTMLayer(name="l4", return_type="last", input_size=8, hidden_size=8, batch_first=True))
    model.add_layer(LinearLayer(name="l6", in_features=8, out_features=2))
    model.add_configuration(Configuration(batch_size=8, epochs=1, learning_rate=0.01, optimizer="adam",
                                          metrics=["mae"], loss_function="mse"))
    generator, name, code = _generate(PytorchGenerator, _identify(model), tmp_path)
    (call,) = _regions(generator, name)[("id-l4", "layer_forward")]
    assert (call.end_line - call.start_line + 1) == 2
    assert _line(code, call).startswith("x, _ = self.l4(x)")


def test_pytorch_sequential_traces_each_entry_of_the_sequential(tmp_path):
    generator, name, code = _generate(PytorchGenerator, _identify(_network(sub_nn=False)), tmp_path,
                                      generation_type="sequential")
    regions = _regions(generator, name)
    assert name == "pytorch_nn_sequential.py"
    assert [_line(code, r) for r in regions[("id-l1", "layer")]] == [
        "nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=0, dilation=1, "
        "groups=1, bias=True),", "nn.ReLU(),"]
    assert _line(code, regions[("id-r1", "tensorop")][0]) == "Lambda(lambda x: x.reshape(-1, 64)),"
    assert _line(code, regions[("id-train_data", "train_dataset")][0]).startswith("def prepare_datasets():")


def test_tensorflow_subclassing_and_sequential_are_traced(tmp_path):
    generator, name, code = _generate(TFGenerator, _identify(_network()), tmp_path / "sub")
    regions = _regions(generator, name)
    assert name == "tf_nn_subclassing.py"
    assert _line(code, regions[("id-l7", "layer")][0]).startswith("self.l7 = layers.Dense(units=64")
    assert _line(code, regions[("id-l7", "layer_forward")][0]) == "x = self.l7(x)"
    assert _line(code, regions[("id-r1", "tensorop")][0]) == "x_1 = tf.reshape(x, [-1, 64])"
    assert _line(code, regions[("id-enc", "sub_nn")][0]).startswith("self.enc = Sequential([")
    assert _line(code, regions[("id-enc", "sub_nn_forward")][0]) == "x_1 = self.enc(x_1)"
    assert _line(code, regions[("id-s2", "layer")][0]) == "layers.Dense(units=3, use_bias=True),"

    generator, name, code = _generate(TFGenerator, _identify(_network(sub_nn=False)), tmp_path / "seq",
                                      generation_type="sequential")
    regions = _regions(generator, name)
    assert _line(code, regions[("id-l8", "layer")][0]) == "layers.Dense(units=10, use_bias=True),"
    assert _line(code, regions[("id-r1", "tensorop")][0]).startswith("layers.Lambda(lambda x: tf.reshape(")


@pytest.mark.parametrize("generator_class, options", [
    (PytorchGenerator, {}), (PytorchGenerator, {"generation_type": "sequential"}), (PytorchGenerator, {"channel_last": True}),
    (TFGenerator, {}), (TFGenerator, {"generation_type": "sequential"}),
])
def test_elements_without_ids_render_untraced_and_the_output_is_unchanged(tmp_path, generator_class, options):
    sub_nn = options.get("generation_type") != "sequential"
    traced_model = _identify(_network(sub_nn=sub_nn))
    traced, name, traced_code = _generate(generator_class, traced_model, tmp_path / "traced", **options)
    untraced, _, untraced_code = _generate(generator_class, _network(sub_nn=sub_nn), tmp_path / "untraced", **options)
    assert traced.traced_files[name].regions
    assert traced_code == untraced_code
    assert untraced.traced_files[name].regions == () and untraced.traced_files[name].sources == ()
    assert "\x02" not in traced_code and "\x03" not in traced_code
    # Every region lies inside the file and starts on a line with code.
    lines = traced_code.split("\n")
    for region in traced.traced_files[name].regions:
        assert 1 <= region.start_line <= region.end_line <= len(lines)
        assert lines[region.start_line - 1].strip() and lines[region.end_line - 1].strip()


def test_modules_details_remember_the_element_of_each_key():
    generator = PytorchGenerator(_identify(_network()), output_dir=None)
    details = generator.modules_details
    assert isinstance(details, ModulesDetails)
    assert details.elements["l1_layer"].id == "id-l1" and details.elements["l1_activ"].id == "id-l1"
    assert details.elements["r1_op"].id == "id-r1" and details.elements["enc_nn"].id == "id-enc"
    assert details["enc_nn"].elements["s1_layer"].id == "id-s1"


def test_remap_regions_follows_the_lines_a_clean_up_keeps():
    before = "a\n\n\n\nb\nc\n\nd\n"
    after = "a\n\nb\nc\nd\n"
    regions = [TraceRegion("e1", "T", "e1", "r", 1, 1), TraceRegion("e2", "T", "e2", "r", 5, 6),
               TraceRegion("e3", "T", "e3", "r", 8, 8)]
    assert [(r.element_id, r.start_line, r.end_line) for r in remap_regions(regions, before, after)] == [
        ("e1", 1, 1), ("e2", 3, 4), ("e3", 5, 5)]
    # The lines are matched by their content, whitespace aside; a text that is not made of them gives nothing.
    assert [(r.start_line, r.end_line) for r in remap_regions(
        [TraceRegion("e", "T", "e", "r", 1, 2)], "  a\n b\n", "a\nb\n")] == [(1, 2)]
    assert remap_regions(regions, before, "a\nz\n") == []
    # Duplicated lines the clean-up removed do not shift what follows.
    assert [(r.start_line, r.end_line) for r in remap_regions(
        [TraceRegion("e", "T", "e", "r", 4, 4)], "import x\nimport x\n\ny = 1\n", "import x\n\ny = 1\n")] == [(3, 3)]
