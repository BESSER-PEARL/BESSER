"""``besser.toml``: the outputs ``besser build`` generates, so their command lines are written once."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_FILENAME = "besser.toml"

# What ``besser schema config`` prints: the whole format, by example.
CONFIG_EXAMPLE = '''\
# besser.toml: what 'besser build' generates. Paths are relative to this file.

model = "model.py"              # the model file every output reads, unless it names its own
test = "python -m pytest -q"    # optional: what 'besser apply' runs after the build (no shell)
test_timeout = 600              # optional: seconds before 'besser apply' stops the tests (default 600)

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

# Seconds 'besser apply' lets the test command run: minutes for a real suite, not forever for one that
# starts a server or waits for input.
DEFAULT_TEST_TIMEOUT = 600

_OUTPUT_KEYS = {"generator": str, "dir": str, "model": str, "use": list, "options": dict, "overwrite": bool}
_TYPE_NAMES = {str: "a string", list: "a list of strings", dict: "a table", bool: "true or false"}


class ConfigError(Exception):
    """The configuration file is missing or not what ``besser build`` reads."""


@dataclass(frozen=True)
class TestConfig:
    """What ``besser apply`` runs after the build: the ``test`` command line and its time limit."""

    __test__ = False  # not a pytest test class, whatever its name

    command: str | None
    timeout: float = DEFAULT_TEST_TIMEOUT


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


def _read_config(path: str) -> dict[str, Any]:
    """The configuration file at ``path`` as TOML data, with its top-level keys checked.

    Raises:
        ConfigError: The file is missing, is not TOML, or holds a top-level key or value no command reads.
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

    unknown = sorted(set(data) - {"model", "test", "test_timeout", "output"})
    if unknown:
        raise ConfigError(f"{path}: unknown key {', '.join(unknown)}; the file takes 'model', 'test', "
                          "'test_timeout' and [output.NAME] tables")
    for key, what in (("model", "the path of a model file"), ("test", "a command line")):
        if key in data and not isinstance(data[key], str):
            raise ConfigError(f"{path}: {key} must be a string, {what}")
    timeout = data.get("test_timeout", DEFAULT_TEST_TIMEOUT)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not timeout > 0:
        raise ConfigError(f"{path}: test_timeout must be a positive number of seconds")
    return data


def load_test_config(path: str) -> TestConfig:
    """The ``test`` command line of the configuration file at ``path`` (None when it has none) and its time limit.

    Raises:
        ConfigError: The file is missing, is not TOML, or holds a top-level key or value no command reads.
    """
    data = _read_config(path)
    return TestConfig(data.get("test") or None, data.get("test_timeout", DEFAULT_TEST_TIMEOUT))


def load_default_model(path: str) -> str | None:
    """The top-level ``model`` of the configuration file at ``path``, relative to the working directory.

    Raises:
        ConfigError: The file is missing, is not TOML, or holds a top-level key or value no command reads.
    """
    model = _read_config(path).get("model")
    return (Path(path).parent / model).as_posix() if model else None


def load_config(path: str) -> list[OutputConfig]:
    """The outputs of the configuration file at ``path``, in the file's order.

    Raises:
        ConfigError: The file is missing, is not TOML, or holds a key or a value ``build`` does not read.
    """
    file = Path(path)
    data = _read_config(path)
    default_model = data.get("model")
    tables = data.get("output", {})
    if not isinstance(tables, dict) or not tables or not all(isinstance(table, dict) for table in tables.values()):
        raise ConfigError(f"{path}: no output to build: add an [output.NAME] table with 'generator' and 'dir'")

    outputs = []
    # A directory holds one generator's output and its record: a second would overwrite or refuse it.
    dir_owners: dict[str, str] = {}
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
        # As the file system names it: "out", "./out/" and the absolute path of out are one directory.
        owner = dir_owners.setdefault(os.path.normcase(os.path.abspath(file.parent / table["dir"])), name)
        if owner != name:
            raise ConfigError(f"{where}: dir {table['dir']!r} is also the dir of [output.{owner}]: "
                              "give each output its own directory")
        outputs.append(OutputConfig(
            name=name, generator=table["generator"],
            model=(file.parent / model).as_posix(), output_dir=(file.parent / table["dir"]).as_posix(),
            options=table.get("options", {}), use=table.get("use", []), overwrite=table.get("overwrite", False),
        ))
    return outputs
