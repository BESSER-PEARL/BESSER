"""``besser.toml``: the outputs ``besser build`` generates, so their command lines are written once."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_FILENAME = "besser.toml"

# What ``besser schema config`` prints: the whole format, by example.
CONFIG_EXAMPLE = '''\
# besser.toml: what 'besser build' generates. Paths are relative to this file.

model = "model.py"              # the model file every output reads, unless it names its own

[output.api]                    # one table per output; 'besser build api' builds this one alone
generator = "backend"           # a generator id ('besser generators' lists them)
dir = "generated/api"           # where its code goes
options = { http_methods = ["GET", "POST"] }   # the generator's options, typed

[output.db]
generator = "sqlalchemy"
dir = "generated/db"
model = "other_model.py"        # optional: this output's own model file
use = ["library_model"]         # optional: the model to read when the file holds several of a kind
options = { dbms = "postgresql" }
overwrite = false               # optional: true regenerates from scratch, discarding edits made in dir
'''

_OUTPUT_KEYS = {"generator": str, "dir": str, "model": str, "use": list, "options": dict, "overwrite": bool}
_TYPE_NAMES = {str: "a string", list: "a list of strings", dict: "a table", bool: "true or false"}


class ConfigError(Exception):
    """The configuration file is missing or not what ``besser build`` reads."""


@dataclass(frozen=True)
class OutputConfig:
    """One ``[output.NAME]`` table: a generator, the model it reads and where its code goes."""

    name: str
    generator: str
    # As the commands take them: relative to the working directory when the file's folder is.
    model: str
    output_dir: str
    options: dict[str, Any] = field(default_factory=dict)
    use: list[str] = field(default_factory=list)
    overwrite: bool = False


def load_config(path: str) -> list[OutputConfig]:
    """The outputs of the configuration file at ``path``, in the file's order.

    Raises:
        ConfigError: The file is missing, is not TOML, or holds a key or a value ``build`` does not read.
    """
    file = Path(path)
    try:
        data = tomllib.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"{path}: no such file; 'besser schema config' prints an example to start from") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path}: cannot be read: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: not a TOML file: {exc}") from exc

    unknown = sorted(set(data) - {"model", "output"})
    if unknown:
        raise ConfigError(f"{path}: unknown key {', '.join(unknown)}; the file takes 'model' and [output.NAME] tables")
    default_model = data.get("model")
    if default_model is not None and not isinstance(default_model, str):
        raise ConfigError(f"{path}: model must be a string, the path of a model file")
    tables = data.get("output", {})
    if not isinstance(tables, dict) or not tables or not all(isinstance(table, dict) for table in tables.values()):
        raise ConfigError(f"{path}: no output to build: add an [output.NAME] table with 'generator' and 'dir'")

    outputs = []
    for name, table in tables.items():
        where = f"{path}: [output.{name}]"
        unknown = sorted(set(table) - set(_OUTPUT_KEYS))
        if unknown:
            raise ConfigError(f"{where}: unknown key {', '.join(unknown)}; an output takes {', '.join(_OUTPUT_KEYS)}")
        for key, value in table.items():
            expected = _OUTPUT_KEYS[key]
            if not isinstance(value, expected) or (expected is list and not all(isinstance(v, str) for v in value)):
                raise ConfigError(f"{where}: {key} must be {_TYPE_NAMES[expected]}")
        missing = [key for key in ("generator", "dir") if key not in table]
        model = table.get("model", default_model)
        if model is None:
            missing.append("model (or a top-level model)")
        if missing:
            raise ConfigError(f"{where}: missing {', '.join(missing)}")
        outputs.append(OutputConfig(
            name=name, generator=table["generator"],
            model=(file.parent / model).as_posix(), output_dir=(file.parent / table["dir"]).as_posix(),
            options=table.get("options", {}), use=table.get("use", []), overwrite=table.get("overwrite", False),
        ))
    return outputs
