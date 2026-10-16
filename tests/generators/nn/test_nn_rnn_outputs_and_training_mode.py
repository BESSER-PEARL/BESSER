"""Recurrent layers that return one tensor, and the training / evaluation mode of generated PyTorch code."""
import os

import pytest

from besser.BUML.metamodel.nn import NN, Configuration, Dataset, DropoutLayer, LinearLayer
from besser.BUML.metamodel.nn.neural_network import GRULayer, LSTMLayer, SimpleRNNLayer
from besser.generators.nn.pytorch.pytorch_code_generator import PytorchGenerator
from besser.generators.nn.tf.tf_code_generator import TFGenerator


def rnn_then_linear(layer_cls, return_type):
    model = NN(name="forecaster")
    model.add_layer(layer_cls(name="rnn", input_size=3, hidden_size=8, return_type=return_type))
    model.add_layer(LinearLayer(name="out", in_features=8, out_features=2))
    return model


@pytest.mark.parametrize("layer_cls", [GRULayer, LSTMLayer, SimpleRNNLayer])
@pytest.mark.parametrize("return_type", ["last", "full"])
def test_rnn_returning_one_tensor_needs_no_input_var(layer_cls, return_type, tmp_path):
    model = rnn_then_linear(layer_cls, return_type)
    assert model.validate(raise_exception=False)["success"]
    PytorchGenerator(model, output_dir=str(tmp_path / "torch")).generate()
    TFGenerator(rnn_then_linear(layer_cls, return_type), output_dir=str(tmp_path / "tf")).generate()
    code = (tmp_path / "torch" / "pytorch_nn_subclassing.py").read_text(encoding="utf-8")
    assert "x, _ = self.rnn(x)" in code
    assert "x = self.out(x)" in code
    if return_type == "last":
        assert "x = x[:, -1, :]" in code

    tf_code = (tmp_path / "tf" / "tf_nn_subclassing.py").read_text(encoding="utf-8")
    assert "x = self.rnn(x)" in tf_code     # one tensor: no unpacking of hidden states
    assert "x = self.out(x)" in tf_code
    assert ("return_sequences=True" in tf_code) == (return_type == "full")
    assert "return_state=True" not in tf_code


@pytest.mark.parametrize("return_type", ["hidden", "both"])
def test_rnn_returning_several_tensors_still_needs_input_var(return_type):
    result = rnn_then_linear(LSTMLayer, return_type).validate(raise_exception=False)
    assert not result["success"]
    assert any("input_var is required" in e for e in result["errors"])


def regression_model(outputs):
    model = NN(name="regressor")
    model.add_layer(LinearLayer(name="hidden", in_features=4, out_features=16, actv_func="relu"))
    model.add_layer(DropoutLayer(name="drop", rate=0.5))
    model.add_layer(LinearLayer(name="out", in_features=16, out_features=outputs))
    model.add_configuration(Configuration(batch_size=8, epochs=2, learning_rate=0.01, optimizer="adam",
                                          loss_function="mse", metrics=["mae"]))
    model.add_train_data(Dataset(name="train", path_data="train.csv", task_type="regression", input_format="csv"))
    model.add_test_data(Dataset(name="test", path_data="test.csv"))
    return model


def test_generated_training_switches_modes_and_keeps_multi_target_labels(tmp_path):
    PytorchGenerator(regression_model(2), output_dir=str(tmp_path)).generate()
    code = (tmp_path / "pytorch_nn_subclassing.py").read_text(encoding="utf-8")
    assert "model.train()" in code
    assert "model.eval()" in code
    assert "if labels.dim() == 1:" in code


def test_generated_training_runs_with_two_targets(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    pytest.importorskip("pandas")
    PytorchGenerator(regression_model(2), output_dir=str(tmp_path)).generate()
    namespace = {"__name__": "generated"}
    path = os.path.join(str(tmp_path), "pytorch_nn_subclassing.py")
    with open(path, encoding="utf-8") as f:
        exec(compile(f.read(), path, "exec"), namespace)
    torch.manual_seed(0)
    x, y = torch.randn(32, 4), torch.randn(32, 2)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=8)
    network = namespace["NeuralNetwork"]()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    with _no_broadcast_warning():
        namespace["train"](network, loader, torch.nn.MSELoss(), optimizer, 2)
    namespace["evaluate"](network, loader, torch.nn.MSELoss())
    assert not network.training                       # evaluate left it in evaluation mode
    with torch.no_grad():
        assert torch.equal(network(x), network(x))    # so dropout is off: the same input, the same output


class _no_broadcast_warning:
    """Fail when MSELoss broadcasts targets of the wrong shape (the bug this fixes)."""

    def __enter__(self):
        import warnings
        self._catch = warnings.catch_warnings()
        self._catch.__enter__()
        warnings.filterwarnings("error", message=".*target size.*")
        return self

    def __exit__(self, *exc):
        return self._catch.__exit__(*exc)
