"""TimeSeriesForecasterGenerator: a B-UML NN model becomes a runnable time-series forecaster.

Generation and validation need neither torch nor numpy; the tests that run the generated code skip
when torch is not installed. Synthetic series only.
"""
import ast
import importlib.util
from pathlib import Path

import pytest

from besser.BUML.metamodel.nn import (
    NN,
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
from tests.forecaster_networks import HORIZON, WINDOW, configuration, gru, mlp

CONFIG = ForecastConfig(window=WINDOW, horizon=HORIZON, seeds=(0,))


# --------------------------------------------------------------------------- networks


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


def channel_last_cnn() -> NN:
    """A first Conv1D marked permute_in: with channel_last it reads (batch, window, channels), the
    layout of an RNN."""
    model = NN(name="channel_last_cnn")
    model.add_layer(Conv1D(name="c1", actv_func="relu", in_channels=3, out_channels=8, kernel_dim=[3],
                           permute_in=True))
    model.add_layer(FlattenLayer(name="flat"))
    model.add_layer(LinearLayer(name="out", in_features=8 * 22, out_features=HORIZON))
    model.add_configuration(configuration())
    return model


def dropout_op_net() -> NN:
    """A TensorOp dropout: the stock templates write it with torch.nn.functional (F)."""
    model = NN(name="dropout_op")
    model.add_layer(LinearLayer(name="hidden", in_features=WINDOW + 2, out_features=16, actv_func="relu"))
    model.add_tensor_op(TensorOp(name="drop", tns_type="dropout", dropout_rate=0.1))
    model.add_layer(LinearLayer(name="output", in_features=16, out_features=HORIZON))
    model.add_configuration(configuration())
    return model


NETWORKS = {"mlp": (mlp, "flat"), "gru": (gru, "sequence"), "stacked_lstm": (stacked_lstm, "sequence"),
            "simple_rnn": (simple_rnn, "sequence"), "cnn1d": (cnn1d, "channels"), "cnn_rnn": (cnn_rnn, "channels"),
            "channel_last_cnn": (channel_last_cnn, "sequence"), "dropout_op": (dropout_op_net, "flat")}
CHANNEL_LAST = {"channel_last_cnn"}


def generate(build, out: Path, config: ForecastConfig = CONFIG) -> Path:
    model = build()
    return Path(TimeSeriesForecasterGenerator(model, config, output_dir=str(out),
                                              channel_last=model.name in CHANNEL_LAST).generate())


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
                 "CALENDAR = True", "STANDARDIZE = False", "SEEDS = [3, 5]"):
        assert line in code
    assert "Quarterly series." in code


def test_the_default_difference_is_the_season(tmp_path):
    config = ForecastConfig(window=WINDOW, horizon=HORIZON, season=4)
    code = generate(mlp, tmp_path, config).read_text(encoding="utf-8")
    assert "SEASON = 4" in code and "DIFFERENCE = 4" in code
    assert ForecastConfig().lag == 12 and ForecastConfig(season=7).lag == 7


def test_without_a_config_the_defaults_are_used(tmp_path):
    path = TimeSeriesForecasterGenerator(mlp(), output_dir=str(tmp_path)).generate()
    code = Path(path).read_text(encoding="utf-8")
    for line in ("WINDOW = 24", "HORIZON = 12", "SEASON = 12", "DIFFERENCE = 12", "SEEDS = [0]"):
        assert line in code


@pytest.mark.parametrize("text", [
    '"""\nimport os; os.system("echo PWNED")\n"""',
    'a\\" \r import os \x00 \u2028 import sys \\',
    'ends with a quote"',
])
def test_free_text_cannot_escape_the_docstring(text, tmp_path):
    """The description and the regenerate command are text in the module's docstring, never code."""
    config = ForecastConfig(window=WINDOW, horizon=HORIZON, description=text)
    path = TimeSeriesForecasterGenerator(mlp(), config, output_dir=str(tmp_path), regenerate=text).generate()
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    assert text in ast.get_docstring(tree, clean=False)
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert "os" not in imported and "sys" in imported      # sys: the module's own import, not the text's
    assert sum(isinstance(node, ast.Import) and node.names[0].name == "sys" for node in tree.body) == 1
    assert not any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) for node in tree.body)


def test_the_stock_imports_cover_functional_tensor_ops(tmp_path):
    code = generate(dropout_op_net, tmp_path).read_text(encoding="utf-8")
    assert "F.dropout(" in code
    assert "import torch.nn.functional as F" in code
    assert code.count("\nimport torch\n") == 1


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


def test_a_last_module_that_is_not_linear_is_refused(tmp_path):
    model = NN(name="ends_recurrent")
    model.add_layer(LinearLayer(name="a", in_features=WINDOW + 2, out_features=8))
    model.add_layer(GRULayer(name="b", input_size=8, hidden_size=HORIZON, return_type="last", batch_first=True))
    model.add_configuration(configuration())
    with pytest.raises(ValueError, match="the last module 'b' must be a LinearLayer returning the 12 forecasts, "
                                         "not GRULayer"):
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
    ({"difference": -1}, r"difference must be 0 \(levels\), 1 \(first difference\) or the season \(12\), not -1"),
    ({"difference": 5}, r"or the season \(12\), not 5"),
    ({"season": 4, "difference": 12}, r"or the season \(4\), not 12"),
    ({"seeds": ()}, "seeds must name at least one seed"),
    ({"seeds": tuple(range(101))}, "at most 100 seeds, not 101"),
])
def test_forecast_config_is_checked(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ForecastConfig(**{"window": WINDOW, "horizon": HORIZON, **kwargs}).validate()


@pytest.mark.parametrize("data, message", [
    ({"seed": 1, "seeds": [2]}, "send either seed or seeds, not both"),
    ({"window": "1" * 400}, r"window must be an integer, got '1{40}"),
    ({"difference": True}, "difference must be an integer, got True"),
])
def test_from_dict_names_the_problem_briefly(data, message):
    with pytest.raises(ValueError, match=message) as info:
        ForecastConfig.from_dict(data)
    assert len(str(info.value)) < 200


def test_from_dict_takes_null_as_the_season():
    assert ForecastConfig.from_dict({"season": 4, "difference": None}).lag == 4
    assert ForecastConfig.from_dict({}).difference is None


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
    """The logarithm is skipped when a value is not positive, with a warning, and the forecaster
    remembers it: a series without the zero is then forecast in levels too (not on the log scale)."""
    fc = load(generate(mlp, tmp_path))
    values = synthetic()
    values[5] = 0.0
    with pytest.warns(UserWarning, match="forecast in levels"):
        forecaster = fc.fit(values, epochs=1)
    assert forecaster.log is False
    clean = synthetic()
    out = forecaster.predict(clean)
    assert np.isfinite(out).all()
    # in levels the forecast stays on the series' scale; read as logs it was exp(1e9)-sized or tiny
    assert 0.01 * clean.max() < np.abs(out).max() < 100 * clean.max()


@needs_torch
def test_a_model_trained_on_the_logarithm_refuses_a_value_that_is_not_positive(tmp_path):
    fc = load(generate(mlp, tmp_path))
    forecaster = fc.fit(synthetic(), epochs=1)
    assert forecaster.log is True
    values = synthetic()
    values[-3] = 0.0
    with pytest.raises(ValueError, match="trained on the logarithm"):
        forecaster.predict(values)


@needs_torch
def test_pooled_series_share_one_log_decision(tmp_path):
    fc = load(generate(gru, tmp_path))
    positive, with_zero = synthetic(80, 1), synthetic(80, 2)
    with_zero[10] = 0.0
    with pytest.warns(UserWarning, match="forecast in levels"):
        forecaster = fc.fit([positive, with_zero], epochs=1)
    assert forecaster.log is False


@needs_torch
def test_difference_and_undifference_are_inverse(tmp_path):
    fc = load(generate(mlp, tmp_path))          # DIFFERENCE = 12
    z = np.arange(30, dtype=float) ** 1.5
    d = fc.difference(z)
    assert np.isnan(d[:12]).all()
    assert np.allclose(d[12:], z[12:] - z[:-12])
    assert np.allclose(fc.undifference(z[:20], d[20:30]), z[20:30])


@needs_torch
def test_the_calendar_features_give_the_position_of_the_first_period_forecast(tmp_path):
    flat = load(generate(mlp, tmp_path / "flat"))
    lags = np.zeros((2, WINDOW))
    x = flat.network_inputs(lags, np.array([0, 11]), start=3)
    # origin 0 of a series starting in April: the first value forecast is May (position 4)
    assert np.allclose(x[0, -2:], [np.sin(2 * np.pi * 4 / 12), np.cos(2 * np.pi * 4 / 12)])
    assert np.allclose(x[1, -2:], [np.sin(2 * np.pi * 3 / 12), np.cos(2 * np.pi * 3 / 12)])
    seq = load(generate(gru, tmp_path / "seq"))
    x = seq.network_inputs(np.zeros((1, WINDOW)), np.array([WINDOW - 1]), start=0)
    assert x.shape == (1, WINDOW, 3)
    # step k of the window is the k-th period of the series: its own position, not the next one
    assert np.allclose(x[0, :, 1], np.sin(2 * np.pi * (np.arange(WINDOW) % 12) / 12))


@needs_torch
def test_training_rows_mask_the_targets_that_are_not_observed(tmp_path):
    config = ForecastConfig(window=WINDOW, horizon=HORIZON, calendar=False)
    fc = load(generate(lambda: linear_net(WINDOW, HORIZON), tmp_path, config))
    d = np.arange(40, dtype=float)
    d[:3] = np.nan                                   # undefined differences at the start
    inputs, targets = fc.training_rows(d, start=0)
    # windows must be complete: the first origin is 3 + WINDOW - 1; the last one is n - 2
    assert inputs.shape == (40 - 1 - (3 + WINDOW - 1), WINDOW)
    assert np.array_equal(inputs[0], d[3:3 + WINDOW])
    # the last origin sees one observed target; the rest are NaN (left out of the loss)
    assert targets[-1, 0] == d[-1] and np.isnan(targets[-1, 1:]).all()
    assert not np.isnan(targets[0]).any()


@needs_torch
def test_a_constant_season_leaves_mase_undefined_with_a_warning(tmp_path):
    fc = load(generate(mlp, tmp_path))
    values = np.tile(np.arange(1.0, 13.0), 8)        # repeats its season exactly
    with pytest.warns(UserWarning, match="no in-sample error"):
        assert fc.mase(values, {60: fc.seasonal_naive(values, 60)}) == {}


@needs_torch
def test_a_quarterly_csv_gives_the_season_position(tmp_path):
    fc = load(generate(mlp, tmp_path, ForecastConfig(window=WINDOW, horizon=HORIZON, season=4)))
    assert fc.season_start("2020-Q3") == 2 and fc.season_start("2020Q1") == 0
    assert fc.season_start("2020-03") is None        # a month says nothing about a season of 4


@needs_torch
def test_the_saved_file_names_the_model(tmp_path):
    fc = load(generate(mlp, tmp_path))
    fc.save(fc.fit(synthetic(), epochs=1), str(tmp_path / "m.pt"))
    assert torch.load(str(tmp_path / "m.pt"), weights_only=True)["model"] == "mlp"
    assert load(generate(mlp, tmp_path / "again")).MODEL_NAME == "mlp"
    other = load(generate(gru, tmp_path / "gru"))
    with pytest.raises(ValueError, match="holds the 'mlp' forecaster, not 'gru'"):
        other.load(str(tmp_path / "m.pt"))
