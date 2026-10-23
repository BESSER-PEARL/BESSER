"""TimeSeriesForecasterGenerator: a B-UML neural network model becomes a runnable time-series forecaster.

BESSER's ``PytorchGenerator`` turns a B-UML ``NN`` model into a PyTorch ``nn.Module`` plus a
training script for images or a CSV with one target column. A forecaster needs more than that, and
this generator adds it around the same network code (it reuses the stock generator's layer and
forward-pass templates, so the network is written exactly as ``PytorchGenerator`` writes it):

- **windowing**: the last ``window`` values of a series are the input, the next ``horizon`` values
  the output (one network for every horizon, the "MIMO" strategy);
- **transforms**: logarithm, differencing (first or seasonal), standardisation, and their inverse;
- **season position features** (sine and cosine of the month, for a monthly series);
- **deterministic training**: seeds, a seeded shuffle, an ensemble of seeds, a masked loss so the
  last origins (whose future is only partly observed) still train the network;
- **inference**: ``predict`` for any number of steps (recursively beyond ``horizon``), ``save`` and
  ``load`` without timestamps;
- **evaluation**: a rolling-origin ``backtest`` and the MASE against the seasonal naive forecast;
- **a command line**: ``python forecaster_<model>.py series.csv --steps 12 --backtest 20``.

Nothing here is specific to a dataset: the series, its period and the pipeline are parameters
(``ForecastConfig``), and the network is whatever the B-UML model describes, as long as its first
layer (after any dropout) takes the window and its last layer, a linear one, returns ``horizon``
values. With the calendar features on, the first layer is one of:

- a ``LinearLayer`` with ``window + 2`` inputs: the window, flat, plus the sine and cosine;
- an RNN layer (LSTM, GRU, SimpleRNN), ``batch_first``, with ``input_size=3``: one step per value;
- a ``Conv1D`` with ``in_channels=3``: the window as a sequence of three channels.
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from besser.BUML.metamodel.nn import NN
from besser.BUML.metamodel.nn.neural_network import (
    RNN,
    Conv1D,
    DropoutLayer,
    LinearLayer,
)
from besser.generators.nn.pytorch.pytorch_code_generator import PytorchGenerator

TEMPLATES = Path(__file__).parent / "templates"
# the stock PyTorch templates: their macros write the layers and the forward pass
PYTORCH_TEMPLATES = Path(__file__).parent.parent / "pytorch" / "templates"


@dataclass(frozen=True)
class ForecastConfig:
    """How a series becomes training examples for the network, and back.

    Args:
        window: number of past values the network sees.
        horizon: number of future values it returns (the size of its last layer).
        season: the period of the series (12 for monthly data, 4 for quarterly, 7 for daily).
        log: forecast the logarithm when every value is positive (multiplicative season and growth).
        difference: the lag of the difference: 0 (levels), 1 (first difference) or ``season``
            (seasonal difference: the network forecasts the change on the same period a season
            earlier).
        calendar: add the sine and cosine of the position in the season as inputs.
        standardize: centre and scale the differenced series on the training data.
        seeds: one network is trained per seed and their forecasts are averaged.
        description: free text copied into the generated module's docstring.
    """
    window: int
    horizon: int
    season: int = 12
    log: bool = True
    difference: int = 12
    calendar: bool = True
    standardize: bool = True
    seeds: tuple[int, ...] = field(default=(0,))
    description: str = ""

    def validate(self) -> None:
        errors = []
        if self.window < 1:
            errors.append("window must be at least 1")
        if self.horizon < 1:
            errors.append("horizon must be at least 1")
        if self.season < 1:
            errors.append("season must be at least 1")
        if self.difference < 0:
            errors.append("difference must be 0 (levels) or a positive lag (1, or the season)")
        if not self.seeds:
            errors.append("seeds must name at least one seed")
        if errors:
            raise ValueError("Invalid ForecastConfig: " + "; ".join(errors))

    @property
    def calendar_features(self) -> int:
        return 2 if self.calendar else 0


class TimeSeriesForecasterGenerator(PytorchGenerator):
    """Generate ``forecaster_<model name>.py``: the network of a B-UML ``NN`` model, wrapped in a
    time-series pipeline (windowing, transforms, seeded training, prediction, backtest, CLI).

    Args:
        model: the B-UML neural network. It needs a ``Configuration`` (epochs, batch size, optimizer,
            learning rate, weight decay; the loss must be ``mse``). ``train_data`` / ``test_data``
            are not needed: the series is given at run time.
        config: the time-series pipeline (``ForecastConfig``).
        output_dir: where the module is written (default: ``<cwd>/output``).
        regenerate: the command that regenerates the module, quoted in its docstring.
        channel_last: as in ``PytorchGenerator``: the convolutions marked ``permute_in`` /
            ``permute_out`` read and return (batch, length, channels). A first ``Conv1D`` marked
            ``permute_in`` then reads the window as a sequence (the layout of an RNN) instead of
            (batch, channels, length).
    """

    def __init__(self, model: NN, config: ForecastConfig, output_dir: str | None = None,
                 regenerate: str = "", channel_last: bool = False):
        config.validate()
        self.config = config
        self.regenerate = regenerate
        self.layout = self._check_shapes(model, config, channel_last)
        super().__init__(model, output_dir=output_dir, generation_type="subclassing",
                         channel_last=channel_last)

    # ------------------------------------------------------------------ validation
    @staticmethod
    def _check_shapes(model: NN, config: ForecastConfig, channel_last: bool = False) -> str:
        """The input layout ("flat", "sequence" or "channels"), after checking the network fits the
        pipeline. Every problem is reported at once."""
        errors = []
        if model.configuration is None:
            errors.append(f"NN '{model.name}' needs a Configuration (epochs, batch size, optimizer, learning rate)")
        elif model.configuration.loss_function != "mse":
            errors.append(f"NN '{model.name}': a forecaster is trained with the 'mse' loss, "
                          f"not '{model.configuration.loss_function}'")
        layers = list(model.modules)
        if not layers:
            errors.append(f"NN '{model.name}' has no layers")
            raise ValueError(f"NN '{model.name}' cannot be generated as a forecaster:\n  - " + "\n  - ".join(errors))
        # a dropout changes no shape: the window reaches the first layer after it
        first = next((m for m in layers if not isinstance(m, DropoutLayer)), layers[0])
        last = layers[-1]
        layout = "flat"
        channels = 1 + config.calendar_features
        per_step = (f"one value per time step{' plus the sine and cosine of the season position' if config.calendar else ''}")

        def kind(module) -> str:
            return "a sub-network (NN)" if isinstance(module, NN) else type(module).__name__
        if isinstance(first, RNN):
            layout = "sequence"
            if first.input_size != channels:
                errors.append(f"the first layer '{first.name}' is recurrent: it reads {per_step},"
                              f" so input_size must be {channels}, not {first.input_size}")
            if not first.batch_first:
                errors.append(f"the first layer '{first.name}' must be batch_first")
        elif isinstance(first, Conv1D):
            layout = "sequence" if channel_last and first.permute_in else "channels"
            if first.in_channels != channels:
                errors.append(f"the first layer '{first.name}' is a 1D convolution over the window: it reads {per_step},"
                              f" so in_channels must be {channels}, not {first.in_channels}")
        elif isinstance(first, LinearLayer):
            need = config.window + config.calendar_features
            if first.in_features != need:
                errors.append(f"the first layer '{first.name}' reads the window of {config.window} values"
                              f"{' plus 2 calendar features' if config.calendar else ''}, so in_features must be "
                              f"{need}, not {first.in_features}")
        else:
            errors.append(f"the first module '{first.name}' must be a LinearLayer (flat window), an RNN layer "
                          f"(LSTM, GRU, SimpleRNN) or a Conv1D (one channel per input), "
                          f"not {kind(first)}")
        if not isinstance(last, LinearLayer):
            errors.append(f"the last module '{last.name}' must be a LinearLayer returning the {config.horizon} "
                          f"forecasts, not {kind(last)}")
        else:
            if last.out_features != config.horizon:
                values = "value" if last.out_features == 1 else "values"
                errors.append(f"the last layer '{last.name}' returns {last.out_features} {values}; "
                              f"the horizon is {config.horizon}, so out_features must be {config.horizon}")
            if last.actv_func not in (None, "", "linear"):
                errors.append(f"the last layer '{last.name}' must be linear, not '{last.actv_func}': "
                              f"forecasts are unbounded (and can be negative)")
        if errors:
            raise ValueError(f"NN '{model.name}' cannot be generated as a forecaster:\n  - " + "\n  - ".join(errors))
        return layout

    # ------------------------------------------------------------------ generation
    def _extra_imports(self) -> list[str]:
        """Imports the stock layer syntax may need (the same rule as the stock template)."""
        text = repr(self.modules_details)
        out = []
        if "get_activation_function" in text:
            out.append("from besser.generators.nn.pytorch.utils_pytorch import get_activation_function")
        if "Permute(" in text:
            out.append("from besser.generators.nn.utils_nn import Permute")
        return out

    def describe_network(self) -> list[str]:
        """One line per module, for the docstring of the generated module."""
        lines = []
        for m in self.model.modules:
            attrs = []
            for key in ("in_features", "out_features", "input_size", "hidden_size", "rate", "actv_func",
                        "return_type"):
                v = getattr(m, key, None)
                if v not in (None, "", False):
                    attrs.append(f"{key}={v}")
            lines.append(f"{m.name}: {type(m).__name__}({', '.join(attrs)})")
        return lines

    @staticmethod
    def _tidy(code: str) -> str:
        """No blank lines inside ``forward`` (the stock macros leave some), two between top-level
        definitions, none at the ends."""
        out, inside = [], False
        for line in code.split("\n"):
            stripped = line.strip()
            if line.startswith("    def forward("):
                inside = True
            elif inside and (stripped.startswith("return ") or line.startswith(("def ", "class "))):
                inside = False
            if inside and not stripped:
                continue
            out.append(line.rstrip())
        code = re.sub(r"\n{4,}", "\n\n\n", "\n".join(out))
        return code.strip() + "\n"

    def file_name_for(self) -> str:
        return f"forecaster_{self.model.name}.py"

    def generate(self, *args) -> str:
        env = Environment(loader=FileSystemLoader([str(TEMPLATES), str(PYTORCH_TEMPLATES)]))
        env.filters["strip_counter_suffix"] = lambda name: re.sub(r"_c\d+$", "", name)
        conf = self.model.configuration
        code = env.get_template("forecaster.py.j2").render(
            model=self.model,
            modules_details=self.modules_details,
            strip_layer_counter_suffix=self.strip_layer_counter_suffix,
            forward_input_var=self.forward_input_var(),
            cfg=asdict(self.config),
            layout=self.layout,
            conf=conf,
            extra_imports=self._extra_imports(),
            network=self.describe_network(),
            regenerate=self.regenerate,
        )
        code = self._tidy(code)
        path = self.build_generation_path(file_name=self.file_name_for())
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(code)
        return path
