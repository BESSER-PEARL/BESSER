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
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from besser.BUML.metamodel.nn import NN
from besser.BUML.metamodel.nn.neural_network import (
    RNN,
    Conv1D,
    DropoutLayer,
    LinearLayer,
)
from besser.generators.nn.pytorch.pytorch_code_generator import PytorchGenerator

TEMPLATES = Path(__file__).parent / "templates"
# the stock PyTorch templates: their macros write the imports, the layers and the forward pass
PYTORCH_TEMPLATES = Path(__file__).parent.parent / "pytorch" / "templates"

# Defaults: two years of a monthly series in, one year out.
DEFAULT_WINDOW = 24
DEFAULT_HORIZON = 12
DEFAULT_SEASON = 12
# An ensemble beyond this many seeds only multiplies the training time.
MAX_SEEDS = 100
# Window, horizon and season become layer sizes and array lengths: no network has a dimension
# beyond a 32-bit index.
MAX_SIZE = 2 ** 31 - 1
# How much of a rejected value an error message repeats.
MAX_SHOWN = 60


def _shown(value) -> str:
    """``repr(value)``, cut to MAX_SHOWN characters for an error message."""
    text = repr(value)
    return text if len(text) <= MAX_SHOWN else text[:MAX_SHOWN - 3] + "..."


def docstring_text(text) -> str:
    """``text`` made safe inside a triple-quoted docstring of the generated module: backslashes and
    quotes are escaped, and every character that is not printable (except newline and tab) is
    written as its escape, so no text can close the docstring or start a new line of code."""
    out = []
    for ch in str(text):
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch in "\n\t" or ch.isprintable():
            out.append(ch)
        else:
            out.append(ch.encode("unicode_escape").decode("ascii"))
    return "".join(out)


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
            earlier). ``None``, the default, means ``season``.
        calendar: add the sine and cosine of the position in the season as inputs.
        standardize: centre and scale the differenced series on the training data.
        seeds: one network is trained per seed and their forecasts are averaged.
        description: free text copied into the generated module's docstring (escaped).
    """
    window: int = DEFAULT_WINDOW
    horizon: int = DEFAULT_HORIZON
    season: int = DEFAULT_SEASON
    log: bool = True
    difference: int | None = None
    calendar: bool = True
    standardize: bool = True
    seeds: tuple[int, ...] = field(default=(0,))
    description: str = ""

    @property
    def lag(self) -> int:
        """The difference actually applied: ``difference``, or ``season`` when it is ``None``."""
        return self.season if self.difference is None else self.difference

    def validate(self) -> None:
        """Raise ``ValueError`` naming every setting out of range."""
        errors = []
        for name in ("window", "horizon", "season"):
            value = getattr(self, name)
            if value < 1:
                errors.append(f"{name} must be at least 1")
            elif value > MAX_SIZE:
                errors.append(f"{name} must be at most {MAX_SIZE}")
        if self.season >= 1 and self.lag not in (0, 1, self.season):
            errors.append(f"difference must be 0 (levels), 1 (first difference) or the season "
                          f"({self.season}), not {self.lag}")
        if not self.seeds:
            errors.append("seeds must name at least one seed")
        elif len(self.seeds) > MAX_SEEDS:
            errors.append(f"at most {MAX_SEEDS} seeds, not {len(self.seeds)}")
        if errors:
            raise ValueError("Invalid ForecastConfig: " + "; ".join(errors))

    def resolved(self) -> ForecastConfig:
        """The same config with ``difference`` set to the lag it stands for."""
        return replace(self, difference=self.lag)

    @property
    def calendar_features(self) -> int:
        """The number of season-position inputs: 2 (sine and cosine) with ``calendar``, else 0."""
        return 2 if self.calendar else 0

    @classmethod
    def from_dict(cls, data: dict | None) -> ForecastConfig:
        """Build a config from a JSON object (the web editor's generation settings).

        Every key is optional and takes the dataclass default. ``seed`` (one int) is accepted
        instead of ``seeds``; ``difference`` may be ``null`` (the season). Unknown keys and values
        of the wrong type raise ``ValueError`` naming every problem at once.
        """
        data = dict(data or {})
        errors = []
        if "seed" in data and "seeds" in data:
            errors.append("send either seed or seeds, not both")
            data.pop("seed")
        elif "seed" in data:
            data["seeds"] = [data.pop("seed")]
        ints = ("window", "horizon", "season", "difference")
        bools = ("log", "calendar", "standardize")
        unknown = sorted(set(data) - set(ints) - set(bools) - {"seeds", "description"})
        if unknown:
            names = ", ".join(str(k)[:MAX_SHOWN] for k in unknown[:10])
            errors.append(f"unknown setting(s): {names}{', ...' if len(unknown) > 10 else ''}")
        values = {}
        for key in ints:
            if key not in data or (key == "difference" and data[key] is None):
                continue
            value = data[key]
            # bool is an int subclass: True must not pass as a window of 1
            if isinstance(value, bool) or not isinstance(value, int):
                errors.append(f"{key} must be an integer, got {_shown(value)}")
            else:
                values[key] = value
        for key in bools:
            if key in data:
                if not isinstance(data[key], bool):
                    errors.append(f"{key} must be true or false, got {_shown(data[key])}")
                else:
                    values[key] = data[key]
        if "seeds" in data:
            seeds = data["seeds"]
            if (not isinstance(seeds, (list, tuple))
                    or any(isinstance(s, bool) or not isinstance(s, int) for s in seeds)):
                errors.append(f"seeds must be a list of integers, got {_shown(seeds)}")
            else:
                values["seeds"] = tuple(seeds)
        if "description" in data:
            if not isinstance(data["description"], str):
                errors.append("description must be text")
            else:
                values["description"] = data["description"]
        if errors:
            raise ValueError("Invalid ForecastConfig: " + "; ".join(errors))
        config = cls(**values)
        config.validate()
        return config


class TimeSeriesForecasterGenerator(PytorchGenerator):
    """Generate ``forecaster_<model name>.py``: the network of a B-UML ``NN`` model, wrapped in a
    time-series pipeline (windowing, transforms, seeded training, prediction, backtest, CLI).

    Args:
        model: the B-UML neural network. It needs a ``Configuration`` (epochs, batch size, optimizer,
            learning rate, weight decay; the loss must be ``mse``). ``train_data`` / ``test_data``
            are not needed: the series is given at run time.
        config: the time-series pipeline (``ForecastConfig``); the defaults when omitted.
        output_dir: where the module is written (default: ``<cwd>/output``).
        regenerate: the command that regenerates the module, quoted in its docstring.
        channel_last: as in ``PytorchGenerator``: the convolutions marked ``permute_in`` /
            ``permute_out`` read and return (batch, length, channels). A first ``Conv1D`` marked
            ``permute_in`` then reads the window as a sequence (the layout of an RNN) instead of
            (batch, channels, length).
    """

    def __init__(self, model: NN, config: ForecastConfig | None = None, output_dir: str | None = None,
                 regenerate: str = "", channel_last: bool = False):
        config = ForecastConfig() if config is None else config
        config.validate()
        self.config = config.resolved()
        self.regenerate = regenerate
        self.layout = self._check_shapes(model, self.config, channel_last)
        super().__init__(model, output_dir=output_dir, generation_type="subclassing",
                         channel_last=channel_last)

    # ------------------------------------------------------------------ validation
    @staticmethod
    def _check_shapes(model: NN, config: ForecastConfig, channel_last: bool = False) -> str:
        """The input layout ("flat", "sequence" or "channels"), after checking the network fits the
        pipeline. Every problem is reported at once."""
        errors = []
        if model.configuration is None:
            errors.append(f"NN '{model.name}' needs a Configuration (epochs, batch size, optimizer, "
                          f"learning rate)")
        elif model.configuration.loss_function != "mse":
            errors.append(f"NN '{model.name}': a forecaster is trained with the 'mse' loss, "
                          f"not '{model.configuration.loss_function}'")
        layers = list(model.modules)
        if not layers:
            errors.append(f"NN '{model.name}' has no layers")
            raise ValueError(f"NN '{model.name}' cannot be generated as a forecaster:\n  - "
                             + "\n  - ".join(errors))
        # a dropout changes no shape: the window reaches the first layer after it
        first = next((m for m in layers if not isinstance(m, DropoutLayer)), layers[0])
        last = layers[-1]
        layout = "flat"
        extra = config.calendar_features
        channels = 1 + extra
        per_step = "one value per time step" + (
            " plus the sine and cosine of the season position" if config.calendar else "")

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
                errors.append(f"the first layer '{first.name}' is a 1D convolution over the window: "
                              f"it reads {per_step}, so in_channels must be {channels}, "
                              f"not {first.in_channels}")
        elif isinstance(first, LinearLayer):
            need = config.window + extra
            if first.in_features != need:
                errors.append(f"the first layer '{first.name}' reads the window of {config.window} values"
                              f"{f' plus {extra} calendar features' if extra else ''}, so in_features must be "
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
            raise ValueError(f"NN '{model.name}' cannot be generated as a forecaster:\n  - "
                             + "\n  - ".join(errors))
        return layout

    # ------------------------------------------------------------------ generation
    def _describe_network(self) -> list[str]:
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

    def _tidy(self, code: str) -> str:
        """No blank lines inside ``forward`` (the stock clean-up), no duplicated import line, two
        blank lines between top-level definitions (PEP 8), none at the ends."""
        seen, lines = set(), []
        for line in self._strip_forward_blank_lines(code).split("\n"):
            line = line.rstrip()
            if line.startswith(("import ", "from ")) and not line.startswith("from __future__"):
                if line in seen:
                    continue
                seen.add(line)
            lines.append(line)
        code = re.sub(r"\n{4,}", "\n\n\n", "\n".join(lines))
        # the stock import macro leaves blank lines between its imports: one at most there
        head, sep, body = code.partition("\nMODEL_NAME = ")
        code = re.sub(r"\n{3,}", "\n\n", head).rstrip("\n") + "\n\n" + sep.lstrip("\n") + body
        return code.strip() + "\n"

    def file_name_for(self) -> str:
        """The generated file's name: ``forecaster_<model name>.py``."""
        return f"forecaster_{self.model.name}.py"

    def generate(self, *args) -> str:
        """Write the forecaster module and return its path."""
        env = self._make_env([str(TEMPLATES), str(PYTORCH_TEMPLATES)])
        env.filters["docstring_text"] = docstring_text
        env.filters["py"] = repr
        code = env.get_template("forecaster.py.j2").render(
            model=self.model,
            modules_details=self.modules_details,
            strip_layer_counter_suffix=self.strip_layer_counter_suffix,
            forward_input_var=self.forward_input_var(),
            cfg=asdict(self.config),
            layout=self.layout,
            conf=self.model.configuration,
            network=self._describe_network(),
            regenerate=self.regenerate,
        )
        code = self._tidy(code)
        path = self.build_generation_path(file_name=self.file_name_for())
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(code)
        return path
