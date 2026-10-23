"""TimeSeriesForecasterGenerator: a B-UML NN model becomes a runnable time-series forecaster.

Generation and validation need neither torch nor numpy; the tests that run the generated code skip
when torch is not installed. Synthetic series only.
"""
import importlib.util
from pathlib import Path

import pytest

from besser.BUML.metamodel.nn import (
    NN,
    Configuration,
    Conv1D,
    DropoutLayer,
    EmbeddingLayer,
    FlattenLayer,
    GRULayer,
    LinearLayer,
    LSTMLayer,
    PoolingLayer,
    SimpleRNNLayer,
    TensorOp,
)
from besser.generators.nn.forecaster import (
    ForecastConfig,
    TimeSeriesForecasterGenerator,
)

WINDOW, HORIZON = 24, 12
CONFIG = ForecastConfig(window=WINDOW, horizon=HORIZON, seeds=(0,))


def configuration(epochs: int = 2, loss: str = "mse") -> Configuration:
    return Configuration(batch_size=32, epochs=epochs, learning_rate=0.003, optimizer="adamW",
                         loss_function=loss, metrics=["mae"], weight_decay=0.01)


# --------------------------------------------------------------------------- networks
def mlp() -> NN:
    model = NN(name="mlp")
    model.add_layer(LinearLayer(name="hidden", in_features=WINDOW + 2, out_features=32, actv_func="relu"))
    model.add_layer(DropoutLayer(name="dropout", rate=0.1))
    model.add_layer(LinearLayer(name="output", in_features=32, out_features=HORIZON))
    model.add_configuration(configuration())
    return model


def gru() -> NN:
    model = NN(name="gru")
    model.add_layer(GRULayer(name="recurrent", input_size=3, hidden_size=16, return_type="last",
                             batch_first=True, output_var="state"))
    model.add_layer(LinearLayer(name="output", in_features=16, out_features=HORIZON, input_var="state"))
    model.add_configuration(configuration())
    return model


def stacked_lstm() -> NN:
    """The architecture of BESSER's lstm.py example, without its embedding (a series has no tokens)."""
    model = NN(name="stacked_lstm")
    model.add_layer(LSTMLayer(name="l2", return_type="full", input_size=3, hidden_size=8, bidirectional=True,
                              batch_first=True))
    model.add_layer(DropoutLayer(name="l3", rate=0.2))
    model.add_layer(LSTMLayer(name="l4", return_type="last", input_size=16, hidden_size=8, batch_first=True))
    model.add_layer(LinearLayer(name="l5", actv_func="relu", in_features=8, out_features=8))
    model.add_layer(LinearLayer(name="l6", in_features=8, out_features=HORIZON))
    model.add_configuration(configuration())
    return model


def simple_rnn() -> NN:
    model = NN(name="simple_rnn")
    model.add_layer(SimpleRNNLayer(name="rnn", input_size=3, hidden_size=8, return_type="last", batch_first=True))
    model.add_layer(LinearLayer(name="out", in_features=8, out_features=HORIZON))
    model.add_configuration(configuration())
    return model


def cnn1d() -> NN:
    """Conv1D -> max pool -> Conv1D -> flatten -> linear, over (batch, channels, window)."""
    model = NN(name="cnn1d")
    model.add_layer(Conv1D(name="c1", actv_func="relu", in_channels=3, out_channels=8, kernel_dim=[3]))
    model.add_layer(PoolingLayer(name="p1", pooling_type="max", dimension="1D", kernel_dim=[2]))
    model.add_layer(Conv1D(name="c2", actv_func="relu", in_channels=8, out_channels=8, kernel_dim=[3]))
    model.add_layer(FlattenLayer(name="flat"))
    model.add_layer(LinearLayer(name="out", in_features=8 * 9, out_features=HORIZON))
    model.add_configuration(configuration())
    return model


def cnn_rnn() -> NN:
    """The architecture of BESSER's cnn_rnn.py example, without its embedding: a dropout, two
    convolution branches over the same input, concatenated, permuted into a sequence for a GRU."""
    model = NN(name="cnn_rnn")
    model.add_layer(DropoutLayer(name="l2", rate=0.1))
    model.add_layer(Conv1D(name="l3", actv_func="relu", in_channels=3, out_channels=8, kernel_dim=[4],
                           input_reused=True, name_module_input="l2"))
    model.add_layer(PoolingLayer(name="l4", pooling_type="max", dimension="1D", kernel_dim=[2]))
    model.add_layer(Conv1D(name="l5", actv_func="relu", in_channels=3, out_channels=8, kernel_dim=[5],
                           input_reused=True, name_module_input="l2"))
    model.add_layer(PoolingLayer(name="l6", pooling_type="max", dimension="1D", kernel_dim=[2]))
    model.add_tensor_op(TensorOp(name="op1", tns_type="concatenate", layers_of_tensors=["l4", "l6"],
                                 concatenate_dim=1))
    model.add_tensor_op(TensorOp(name="op2", tns_type="permute", permute_dim=[0, 2, 1]))
    model.add_layer(GRULayer(name="l8", input_size=16, hidden_size=8, batch_first=True, return_type="last"))
    model.add_layer(LinearLayer(name="l11", in_features=8, out_features=HORIZON))
    model.add_configuration(configuration())
    return model


NETWORKS = {"mlp": (mlp, "flat"), "gru": (gru, "sequence"), "stacked_lstm": (stacked_lstm, "sequence"),
            "simple_rnn": (simple_rnn, "sequence"), "cnn1d": (cnn1d, "channels"), "cnn_rnn": (cnn_rnn, "channels")}


def generate(build, out: Path, config: ForecastConfig = CONFIG) -> Path:
    return Path(TimeSeriesForecasterGenerator(build(), config, output_dir=str(out)).generate())


# --------------------------------------------------------------------------- generation
@pytest.mark.parametrize("name", NETWORKS)
def test_every_layout_generates_valid_python(name, tmp_path):
    build, layout = NETWORKS[name]
    path = generate(build, tmp_path)
    assert path.name == f"forecaster_{name}.py"
    code = path.read_text(encoding="utf-8")
    compile(code, str(path), "exec")
    assert f'LAYOUT = "{layout}"' in code
    assert "class NeuralNetwork(nn.Module):" in code
    for function in ("def fit(", "def backtest(", "def mase(", "def save(", "def load(", "def main("):
        assert function in code


def test_the_network_is_written_by_the_stock_pytorch_templates(tmp_path):
    mlp_code = generate(mlp, tmp_path).read_text(encoding="utf-8")
    assert "self.hidden = nn.Linear(in_features=26, out_features=32, bias=True)" in mlp_code
    assert "self.dropout = nn.Dropout(p=0.1)" in mlp_code
    assert "torch.optim.AdamW(network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)" in mlp_code
    gru_code = generate(gru, tmp_path).read_text(encoding="utf-8")
    assert "nn.GRU(input_size=3, hidden_size=16" in gru_code
    assert "def forward(self, x):" in gru_code


def test_the_pipeline_settings_are_written_into_the_module(tmp_path):
    config = ForecastConfig(window=WINDOW, horizon=HORIZON, season=4, log=False, difference=1, calendar=True,
                            standardize=False, seeds=(3, 5), description="Quarterly series.")
    code = generate(mlp, tmp_path, config).read_text(encoding="utf-8")
    for line in ("WINDOW = 24", "HORIZON = 12", "SEASON = 4", "LOG = False", "DIFFERENCE = 1",
                 "STANDARDIZE = False", "SEEDS = [3, 5]"):
        assert line in code
    assert "Quarterly series." in code


def test_generation_is_deterministic(tmp_path):
    for name, (build, _) in NETWORKS.items():
        a = generate(build, tmp_path / "a").read_bytes()
        b = generate(build, tmp_path / "b").read_bytes()
        assert a == b, name


def test_without_calendar_features_the_window_alone_is_the_input(tmp_path):
    config = ForecastConfig(window=WINDOW, horizon=HORIZON, calendar=False)
    model = NN(name="plain")
    model.add_layer(LinearLayer(name="out", in_features=WINDOW, out_features=HORIZON))
    model.add_configuration(configuration())
    assert "CALENDAR = False" in Path(
        TimeSeriesForecasterGenerator(model, config, output_dir=str(tmp_path)).generate()).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- refusals
def linear_net(in_features: int, out_features: int, loss: str = "mse", actv=None) -> NN:
    model = NN(name="net")
    model.add_layer(LinearLayer(name="a", in_features=in_features, out_features=8, actv_func="relu"))
    model.add_layer(LinearLayer(name="b", in_features=8, out_features=out_features, actv_func=actv))
    model.add_configuration(configuration(loss=loss))
    return model


@pytest.mark.parametrize("model, message", [
    (linear_net(10, HORIZON), "in_features must be 26, not 10"),
    (linear_net(26, 6), "returns 6 values; the horizon is 12"),
    (linear_net(26, 1), "returns 1 value; the horizon is 12"),
    (linear_net(26, HORIZON, loss="crossentropy"), "'mse' loss, not 'crossentropy'"),
    (linear_net(26, HORIZON, actv="softmax"), "must be linear, not 'softmax'"),
])
def test_a_network_that_does_not_fit_the_pipeline_is_refused(model, message, tmp_path):
    with pytest.raises(ValueError, match=message):
        TimeSeriesForecasterGenerator(model, CONFIG, output_dir=str(tmp_path))


def test_recurrent_and_convolutional_first_layers_must_read_one_value_and_the_calendar(tmp_path):
    rnn = NN(name="rnn")
    rnn.add_layer(LSTMLayer(name="lstm", input_size=1, hidden_size=8, return_type="last", batch_first=False))
    rnn.add_layer(LinearLayer(name="out", in_features=8, out_features=HORIZON))
    rnn.add_configuration(configuration())
    with pytest.raises(ValueError) as info:
        TimeSeriesForecasterGenerator(rnn, CONFIG, output_dir=str(tmp_path))
    assert "input_size must be 3, not 1" in str(info.value)
    assert "must be batch_first" in str(info.value)
    conv = NN(name="conv")
    conv.add_layer(Conv1D(name="c", in_channels=1, out_channels=4, kernel_dim=[3]))
    conv.add_layer(FlattenLayer(name="f"))
    conv.add_layer(LinearLayer(name="out", in_features=88, out_features=HORIZON))
    conv.add_configuration(configuration())
    with pytest.raises(ValueError, match="in_channels must be 3, not 1"):
        TimeSeriesForecasterGenerator(conv, CONFIG, output_dir=str(tmp_path))


def test_a_classifier_is_refused_with_every_reason_at_once(tmp_path):
    """BESSER's lstm.py example (a text classifier): embedding first, two softmax outputs, no
    configuration. Every problem is listed, not only the first."""
    model = NN(name="lstm")
    model.add_layer(EmbeddingLayer(name="l1", num_embeddings=100, embedding_dim=8))
    model.add_layer(LSTMLayer(name="l2", return_type="last", input_size=8, hidden_size=8, batch_first=True))
    model.add_layer(LinearLayer(name="l6", actv_func="softmax", in_features=8, out_features=2))
    with pytest.raises(ValueError) as info:
        TimeSeriesForecasterGenerator(model, CONFIG, output_dir=str(tmp_path))
    text = str(info.value)
    assert text.startswith("NN 'lstm' cannot be generated as a forecaster:")
    for reason in ("needs a Configuration", "not EmbeddingLayer", "returns 2 values", "not 'softmax'"):
        assert reason in text


def test_a_network_made_of_sub_networks_is_refused(tmp_path):
    features = NN(name="features")
    features.add_layer(LinearLayer(name="f", in_features=26, out_features=8))
    model = NN(name="outer")
    model.add_sub_nn(features)
    model.add_layer(LinearLayer(name="out", in_features=8, out_features=HORIZON))
    model.add_configuration(configuration())
    with pytest.raises(ValueError, match=r"not a sub-network \(NN\)"):
        TimeSeriesForecasterGenerator(model, CONFIG, output_dir=str(tmp_path))


def test_an_empty_network_is_refused(tmp_path):
    model = NN(name="empty")
    model.add_configuration(configuration())
    with pytest.raises(ValueError, match="has no layers"):
        TimeSeriesForecasterGenerator(model, CONFIG, output_dir=str(tmp_path))


@pytest.mark.parametrize("kwargs, message", [
    ({"window": 0}, "window must be at least 1"),
    ({"horizon": 0}, "horizon must be at least 1"),
    ({"season": 0}, "season must be at least 1"),
    ({"difference": -1}, "difference must be"),
    ({"seeds": ()}, "seeds must name at least one seed"),
])
def test_forecast_config_is_checked(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ForecastConfig(**{"window": WINDOW, "horizon": HORIZON, **kwargs}).validate()


# --------------------------------------------------------------------------- the generated code
try:
    import numpy as np
    import torch
except ImportError:  # the generation tests above still run
    np = torch = None
needs_torch = pytest.mark.skipif(torch is None, reason="the generated forecaster runs on PyTorch")


def synthetic(n: int = 120, seed: int = 1):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 1e9 * np.exp(0.004 * t + 0.15 * np.sin(2 * np.pi * t / 12) - 0.2 * (t % 12 == 7)
                        + 0.03 * rng.standard_normal(n))


def load(path: Path):
    spec = importlib.util.spec_from_file_location(f"generated_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@needs_torch
@pytest.mark.parametrize("name", NETWORKS)
def test_every_layout_trains_predicts_and_reloads(name, tmp_path):
    fc = load(generate(NETWORKS[name][0], tmp_path))
    values = synthetic()
    a = fc.fit(values)
    forecast = a.predict(values, 15)          # beyond the horizon: forecasts feed the next window
    assert forecast.shape == (15,) and np.isfinite(forecast).all() and (forecast > 0).all()
    assert np.array_equal(fc.fit(values).predict(values, 15), forecast)     # seeded
    fc.save(a, str(tmp_path / "m.pt"))
    assert np.array_equal(fc.load(str(tmp_path / "m.pt")).predict(values, 15), forecast)


@needs_torch
def test_training_leaves_the_global_random_state_alone(tmp_path):
    fc = load(generate(mlp, tmp_path))
    torch.manual_seed(123)
    before = torch.rand(1)
    torch.manual_seed(123)
    fc.fit(synthetic(), epochs=1)
    assert torch.equal(torch.rand(1), before)


@needs_torch
def test_a_global_model_trains_on_several_series(tmp_path):
    fc = load(generate(gru, tmp_path))
    series = [synthetic(80, seed) for seed in (1, 2, 3)]
    out = fc.fit(series, starts=[0, 3, 6], epochs=1).predict(series[0], HORIZON)
    assert out.shape == (HORIZON,) and np.isfinite(out).all()


@needs_torch
def test_backtest_mase_and_the_seasonal_naive_benchmark(tmp_path):
    fc = load(generate(mlp, tmp_path))
    values = synthetic(96)
    forecasts = fc.backtest(values, [70, 73], seeds=[0])
    assert set(forecasts) == {70, 73}
    scores = fc.mase(values, forecasts)
    assert set(scores) == set(range(1, HORIZON + 1)) and all(np.isfinite(list(scores.values())))
    naive = fc.mase(values, {o: fc.seasonal_naive(values, o) for o in forecasts})
    # on a series that repeats its season exactly but for noise, the seasonal naive scores near 1
    assert 0.2 < np.mean(list(naive.values())) < 5


@needs_torch
def test_the_command_line_forecasts_a_csv(tmp_path, capsys):
    fc = load(generate(mlp, tmp_path))
    values = synthetic(96)
    csv = tmp_path / "series.csv"
    csv.write_text("period,value\n" + "\n".join(f"{2010 + i // 12}-{i % 12 + 1:02d},{v}"
                                                   for i, v in enumerate(values)), encoding="utf-8")
    fc.main([str(csv), "--steps", "3", "--save", str(tmp_path / "cli.pt")])
    assert capsys.readouterr().out.count("t+") == 3
    assert (tmp_path / "cli.pt").exists()


@needs_torch
def test_too_short_a_series_is_refused(tmp_path):
    fc = load(generate(mlp, tmp_path))
    with pytest.raises(ValueError, match="needs series of at least"):
        fc.fit(synthetic(30))


@needs_torch
def test_a_series_with_zeros_is_forecast_in_levels(tmp_path):
    """The logarithm is skipped when a value is not positive."""
    fc = load(generate(mlp, tmp_path))
    values = synthetic()
    values[5] = 0.0
    assert np.isfinite(fc.fit(values, epochs=1).predict(values)).all()


@needs_torch
def test_the_saved_file_names_the_model(tmp_path):
    fc = load(generate(mlp, tmp_path))
    fc.save(fc.fit(synthetic(), epochs=1), str(tmp_path / "m.pt"))
    assert torch.load(str(tmp_path / "m.pt"), weights_only=True)["model"] == "mlp"
    assert load(generate(mlp, tmp_path / "again")).MODEL_NAME == "mlp"
