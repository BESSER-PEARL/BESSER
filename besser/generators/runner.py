"""Run a registered generator: one calling contract for every caller.

A generator's registry entry says what its constructor takes:
``GeneratorSpec.models`` are its model arguments, in order, and
``GeneratorSpec.options`` its keyword arguments. ``run_generator`` builds the
generator from them, runs it and returns the files it wrote, optionally with the
record ``regenerate`` merges against; ``regenerate_generator`` builds it the same
way and merges its output into a folder that holds edits; ``preview_generator``
says what a run would write, writing nothing. The web editor backend
(``services/generation/dispatch.py``) and the ``besser`` command line both
generate through this module.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Mapping

from besser.generators import regeneration
from besser.generators.generator_interface import GeneratorInterface
from besser.generators.registry import GENERATORS, GeneratorSpec, get_generator


class GeneratorInputError(ValueError):
    """The caller's input cannot be run: an unknown generator, a missing model or a bad option."""


def _type_name(value: Any) -> str:
    return type(value).__name__


def _is_integer(value: Any) -> bool:
    # bool is an int subclass: True must not pass as 1
    return isinstance(value, int) and not isinstance(value, bool)


# JSON Schema type -> (check, how an error names the type)
_TYPE_CHECKS: dict[str, tuple[Callable[[Any], bool], str]] = {
    "boolean": (lambda value: isinstance(value, bool), "true or false"),
    "integer": (_is_integer, "an integer"),
    "string": (lambda value: isinstance(value, str), "a string"),
    "array": (lambda value: isinstance(value, (list, tuple)), "a list"),
}


def _option_problem(name: str, schema: Mapping[str, Any], value: Any) -> str | None:
    """Why ``value`` does not fit the option's schema, or ``None`` when it does."""
    check, expected = _TYPE_CHECKS[schema["type"]]
    if not check(value):
        return f"{name} must be {expected}, got {_type_name(value)}"
    if schema["type"] == "array":
        for item in value:
            problem = _option_problem(f"each value of {name}", schema["items"], item)
            if problem:
                return problem
    allowed = schema.get("enum")
    if allowed and value not in allowed:
        return f"{name} must be one of {', '.join(map(str, allowed))}, got {value!r}"
    return None


def check_options(spec: GeneratorSpec, options: Mapping[str, Any] | None) -> dict[str, Any]:
    """``options`` checked against the generator's schema, with its defaults filled in.

    An option that is ``None`` is passed to the generator as it is.

    Raises:
        GeneratorInputError: Naming every unknown option and every value of the wrong type.
    """
    schema = spec.options
    options = dict(options or {})
    problems = []
    unknown = sorted(set(options) - set(schema))
    if unknown:
        available = f"it takes {', '.join(schema)}" if schema else "it takes no options"
        problems.append(f"unknown option(s) for the '{spec.id}' generator: {', '.join(unknown)} ({available})")
    for name, value in options.items():
        if name in schema and value is not None:
            problem = _option_problem(name, schema[name], value)
            if problem:
                problems.append(problem)
    if problems:
        raise GeneratorInputError("; ".join(problems))
    for name, option in schema.items():
        if name not in options and "default" in option:
            options[name] = option["default"]
    return options


def _build(spec: GeneratorSpec, models: list, output_dir: str, options: Mapping[str, Any]) -> GeneratorInterface:
    """The generator, built as its registry entry describes."""
    return spec.load_class()(*models, output_dir=output_dir, **check_options(spec, options))


def _build_flutter(spec: GeneratorSpec, models: list, output_dir: str,
                   options: Mapping[str, Any]) -> GeneratorInterface:
    """The Flutter generator also takes the GUI model's main page and its module."""
    gui_model = models[1]
    main_pages = [
        (module, screen)
        for module in gui_model.modules for screen in module.screens if screen.is_main_page
    ]
    if len(main_pages) != 1:
        raise GeneratorInputError(
            f"The 'flutter' generator needs a GUI model with exactly one main page "
            f"(a Screen with is_main_page=True); this one has {len(main_pages)}."
        )
    module, main_page = main_pages[0]
    return spec.load_class()(
        *models, main_page, module=module, output_dir=output_dir, **check_options(spec, options),
    )


def _build_forecaster(spec: GeneratorSpec, models: list, output_dir: str,
                      options: Mapping[str, Any]) -> GeneratorInterface:
    """The forecaster takes its settings as one ``ForecastConfig``.

    ``ForecastConfig.from_dict`` names every invalid setting at once, so it
    checks them instead of ``check_options``.
    """
    # the generator class is only registered when this module imports
    from besser.generators.nn.forecaster import ForecastConfig

    settings = dict(options)
    channel_last = settings.pop("channel_last", False)
    if not isinstance(channel_last, bool):
        raise GeneratorInputError(f"channel_last must be true or false, got {_type_name(channel_last)}")
    return spec.load_class()(
        *models, ForecastConfig.from_dict(settings), output_dir=output_dir, channel_last=channel_last,
    )


# Generators whose constructor takes more than ``(*models, output_dir=..., **options)``.
_BUILDERS: dict[str, Callable[[GeneratorSpec, list, str, Mapping[str, Any]], GeneratorInterface]] = {
    "flutter": _build_flutter,
    "forecaster": _build_forecaster,
}


def _file_states(directory: str) -> dict[str, tuple[int, int]]:
    """Every file under ``directory``, by path relative to it (with ``/``), with what a rewrite changes."""
    states = {}
    for root, _, files in os.walk(directory):
        for name in files:
            path = os.path.join(root, name)
            stat = os.stat(path)
            states[os.path.relpath(path, directory).replace(os.sep, "/")] = (stat.st_mtime_ns, stat.st_size)
    return states


def build_generator(
    generator_id: str,
    models: Mapping[str, Any],
    output_dir: str,
    options: Mapping[str, Any] | None = None,
) -> GeneratorInterface:
    """The generator ``generator_id``, built on ``models`` to write to ``output_dir``.

    Args:
        generator_id: The ``id`` of an entry of ``besser.generators.registry.GENERATORS``.
        models: The B-UML models to generate from, by kind (``GeneratorSpec.models``:
            ``{"domain": domain_model}``, ``{"domain": ..., "gui": ...}``, ...). Kinds the
            generator does not read are ignored.
        output_dir: The directory it writes to.
        options: The generator's options (``GeneratorSpec.options``).

    Raises:
        GeneratorInputError: Unknown generator, a model the generator reads is missing,
            or an unknown or invalid option.
    """
    try:
        spec = get_generator(generator_id)
    except KeyError:
        known = ", ".join(spec.id for spec in GENERATORS)
        raise GeneratorInputError(f"Unknown generator '{generator_id}'. Available generators: {known}") from None

    missing = [kind for kind in spec.models if models.get(kind) is None]
    if missing:
        raise GeneratorInputError(
            f"The '{spec.id}' generator reads a {' and a '.join(spec.models)} model; "
            f"missing: {', '.join(missing)}."
        )

    build = _BUILDERS.get(spec.id, _build)
    try:
        return build(spec, [models[kind] for kind in spec.models], os.fspath(output_dir), options or {})
    except GeneratorInputError:
        raise
    except ValueError as exc:
        # A constructor refuses a value the schema cannot express (a Java package name, a forecast window).
        raise GeneratorInputError(str(exc)) from exc


def run_generator(
    generator_id: str,
    models: Mapping[str, Any],
    output_dir: str,
    options: Mapping[str, Any] | None = None,
    record: bool = False,
) -> list[str]:
    """Run the generator ``generator_id`` and return the files it wrote.

    The arguments are those of :func:`build_generator`; ``output_dir`` is created if missing.
    With ``record``, the run also writes the record :meth:`GeneratorInterface.regenerate`
    merges against (:func:`besser.generators.regeneration.record_output`), so a later
    regeneration keeps the edits made in ``output_dir``.

    Returns:
        The files the run created or rewrote, sorted, relative to ``output_dir`` with ``/``.

    Raises:
        GeneratorInputError: Unknown generator, a model the generator reads is missing,
            or an unknown or invalid option. Nothing was written.
        NotRecordableError: With ``record``, the generator or the folder cannot take a record;
            nothing was written.
        RegenerationError: With ``record``, a folder or link is where a file goes; nothing was written.
    """
    generator = build_generator(generator_id, models, output_dir, options)
    if record:
        return regeneration.record_output(generator)
    output_dir = os.fspath(output_dir)
    before = _file_states(output_dir)
    generator.generate()
    return sorted(path for path, state in _file_states(output_dir).items() if before.get(path) != state)


def regenerate_generator(
    generator_id: str,
    models: Mapping[str, Any],
    output_dir: str,
    options: Mapping[str, Any] | None = None,
    adopt: bool = False,
    dry_run: bool = False,
) -> regeneration.RegenerationReport:
    """Bring ``output_dir`` up to ``models`` with the generator ``generator_id``, keeping edits made in it.

    The arguments are those of :func:`build_generator`; ``adopt`` and the report are those of
    :meth:`GeneratorInterface.regenerate`. With ``dry_run``, the report is of the same run on a
    copy of ``output_dir``, which is left as it is.

    Raises:
        GeneratorInputError: As :func:`build_generator`. Nothing was written.
        RegenerationError: The folder cannot be regenerated safely; no generated file was changed.
    """
    generator = build_generator(generator_id, models, output_dir, options)
    if dry_run:
        return regeneration.preview_regeneration(generator, adopt)
    return generator.regenerate(adopt=adopt)


def preview_generator(
    generator_id: str,
    models: Mapping[str, Any],
    output_dir: str,
    options: Mapping[str, Any] | None = None,
    record: bool = False,
) -> dict[str, str]:
    """What :func:`run_generator` with these arguments would write, writing nothing.

    Returns:
        Path relative to ``output_dir`` (with ``/``) -> ``"new"``, ``"changed"`` or ``"unchanged"``,
        sorted; with ``record``, the record is among them.

    Raises:
        GeneratorInputError: As :func:`build_generator`.
        NotRecordableError, RegenerationError: As :func:`besser.generators.regeneration.preview_output`.
    """
    generator = build_generator(generator_id, models, output_dir, options)
    return regeneration.preview_output(generator, record=record)
