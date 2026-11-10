"""The ``besser`` commands: each takes parsed arguments and returns a ``Result``."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from besser.cli.loading import (
    LoadedModel,
    LoadedModels,
    MissingDependencyError,
    ModelLoadError,
    load_models,
    missing_backend_dependency_message,
)
from besser.cli.output import EXIT_FAILED, EXIT_OK, EXIT_USAGE, Result, failure, stdout_to_stderr
from besser.generators.registry import GENERATORS, GeneratorSpec, get_generator
from besser.generators.runner import GeneratorInputError, run_generator

# Model kinds the editor has no diagram for, so ``convert`` cannot write them to JSON.
_KINDS_WITHOUT_DIAGRAM = frozenset({"deployment", "feature"})
_BOOLEAN_WORDS = {"true": True, "false": False}


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


# ── validate ────────────────────────────────────────────────────────────────


def _validate_model(loaded: LoadedModel) -> dict[str, Any]:
    """The outcome of ``validate()`` on one model, in the JSON output's shape."""
    report = {"name": loaded.name, "kind": loaded.kind, "type": type(loaded.model).__name__,
              "validated": hasattr(loaded.model, "validate"), "errors": [], "warnings": []}
    if not report["validated"]:
        return report
    try:
        outcome = loaded.model.validate(raise_exception=False)
    except Exception as exc:  # a model too broken for its own validate()
        report["errors"].append(f"validate() raised {type(exc).__name__}: {exc}")
        return report
    report["errors"].extend(str(error) for error in outcome.get("errors", []))
    report["warnings"].extend(str(warning) for warning in outcome.get("warnings", []))
    return report


def _validation_text(path: str, reports: list[dict[str, Any]], project_errors: list[str], verdict: str) -> str:
    lines = [f"{path}: {_plural(len(reports), 'model')}"]
    for report in reports:
        if not report["validated"]:
            status = "no validation rules"
        elif report["errors"] or report["warnings"]:
            status = f"{_plural(len(report['errors']), 'error')}, {_plural(len(report['warnings']), 'warning')}"
        else:
            status = "ok"
        lines.append(f"  {report['name']} ({report['type']}): {status}")
        lines.extend(f"    error: {error}" for error in report["errors"])
        lines.extend(f"    warning: {warning}" for warning in report["warnings"])
    lines.extend(f"  error: {error}" for error in project_errors)
    lines.append(verdict)
    return "\n".join(lines)


def validate(model_path: str, strict: bool = False) -> Result:
    """Run ``validate()`` on every model of the file."""
    try:
        loaded = load_models(model_path)
    except ModelLoadError as exc:
        return failure("validate", str(exc), data={"model": model_path})

    reports = [_validate_model(model) for model in loaded.models]
    errors = [f"{report['name']}: {error}" for report in reports for error in report["errors"]]
    errors.extend(loaded.errors)
    warnings = [f"{report['name']}: {warning}" for report in reports for warning in report["warnings"]]

    failed = bool(errors) or (strict and bool(warnings))
    counts = f"{_plural(len(errors), 'error')}, {_plural(len(warnings), 'warning')}"
    if errors:
        verdict = f"FAILED: {counts}"
    elif failed:
        verdict = f"FAILED: {counts} (--strict)"
    else:
        verdict = f"OK: {counts}"
    return Result(
        command="validate",
        exit_code=EXIT_FAILED if failed else EXIT_OK,
        errors=errors,
        warnings=warnings,
        data={"model": model_path, "models": reports},
        text=_validation_text(model_path, reports, loaded.errors, verdict),
        text_has_diagnostics=True,
    )


# ── generators ──────────────────────────────────────────────────────────────


def _option_value_hint(option: Mapping[str, Any]) -> str:
    """How to write the option's value on the command line."""
    choices = option.get("enum") or option.get("items", {}).get("enum")
    if option["type"] == "boolean":
        return "true|false"
    if option["type"] == "array":
        return ",".join(choices) if choices else f"<{option['items']['type']}>,..."
    return "|".join(map(str, choices)) if choices else f"<{option['type']}>"


def _generator_entry(spec: GeneratorSpec) -> dict[str, Any]:
    return {
        "id": spec.id, "summary": spec.summary, "category": spec.category,
        "models": list(spec.models), "options": spec.options,
    }


def _generator_text(spec: GeneratorSpec) -> list[str]:
    lines = [f"{spec.id}  [{', '.join(spec.models)}]  {spec.summary}"]
    for name, option in spec.options.items():
        line = f"    --option {name}={_option_value_hint(option)}"
        if "default" in option:
            default = option["default"]
            if isinstance(default, bool):
                default = str(default).lower()
            elif isinstance(default, list):
                default = ",".join(map(str, default))
            line += f"  (default: {default})"
        lines.append(line)
        if option.get("description"):
            lines.append(f"        {option['description']}")
    return lines


def generators() -> Result:
    """List the registered generators with the models they read and their options."""
    lines = ["Generators (id, the models it reads, what it generates):", ""]
    for spec in GENERATORS:
        lines.extend(_generator_text(spec))
    return Result(
        command="generators",
        data={"generators": [_generator_entry(spec) for spec in GENERATORS]},
        text="\n".join(lines),
    )


# ── generate ────────────────────────────────────────────────────────────────


def _parse_option_value(name: str, option: Mapping[str, Any], text: str) -> Any:
    """``text`` as the option's type; a comma-separated list for an array."""
    if option["type"] == "boolean":
        if text.lower() not in _BOOLEAN_WORDS:
            raise GeneratorInputError(f"{name} must be true or false, got {text!r}")
        return _BOOLEAN_WORDS[text.lower()]
    if option["type"] == "integer":
        try:
            return int(text)
        except ValueError:
            raise GeneratorInputError(f"{name} must be an integer, got {text!r}") from None
    if option["type"] == "array":
        return [_parse_option_value(name, option["items"], item.strip()) for item in text.split(",") if item.strip()]
    return text


def parse_options(spec: GeneratorSpec, assignments: list[str]) -> dict[str, Any]:
    """``--option name=value`` arguments as the generator's options.

    A value is read as the type the option's schema gives it. An unknown name
    keeps its text: ``run_generator`` refuses it, naming the options there are.

    Raises:
        GeneratorInputError: An argument is not ``name=value`` or a value does not parse.
    """
    schema = spec.options
    options: dict[str, Any] = {}
    for assignment in assignments:
        name, separator, text = assignment.partition("=")
        if not separator or not name:
            raise GeneratorInputError(f"--option takes name=value, got {assignment!r}")
        options[name] = _parse_option_value(name, schema[name], text) if name in schema else text
    return options


def _models_for(spec: GeneratorSpec, loaded: LoadedModels) -> dict[str, LoadedModel]:
    """The one model of each kind the generator reads.

    Raises:
        GeneratorInputError: The file holds none, or several, of a kind.
    """
    chosen = {}
    for kind in spec.models:
        candidates = loaded.of_kind(kind)
        if not candidates:
            held = ", ".join(sorted({model.kind for model in loaded.models}))
            raise GeneratorInputError(
                f"the '{spec.id}' generator reads a {kind} model and {loaded.path} holds none (it holds: {held})"
            )
        if len(candidates) > 1:
            names = ", ".join(model.name for model in candidates)
            raise GeneratorInputError(
                f"{loaded.path} holds {len(candidates)} {kind} models ({names}) and the '{spec.id}' "
                "generator reads one: keep one in the file, or name the ones to use in a Project(...)"
            )
        chosen[kind] = candidates[0]
    return chosen


def generate(generator_id: str, model_path: str, output_dir: str, option_assignments: list[str]) -> Result:
    """Validate the models the generator reads, then run it into ``output_dir``."""
    data: dict[str, Any] = {"generator": generator_id, "model": model_path, "output_dir": output_dir}
    try:
        spec = get_generator(generator_id)
    except KeyError:
        known = ", ".join(spec.id for spec in GENERATORS)
        return failure(
            "generate", f"unknown generator '{generator_id}'; available generators: {known}", data=data,
        )

    try:
        options = parse_options(spec, option_assignments)
        models = _models_for(spec, load_models(model_path))
    except (GeneratorInputError, ModelLoadError) as exc:
        return failure("generate", str(exc), data=data)

    reports = [_validate_model(model) for model in models.values()]
    warnings = [f"{report['name']}: {warning}" for report in reports for warning in report["warnings"]]
    errors = [f"{report['name']}: {error}" for report in reports for error in report["errors"]]
    if errors:
        errors.append("nothing was generated: the model has validation errors")
        return Result(command="generate", exit_code=EXIT_FAILED, errors=errors, warnings=warnings, data=data)

    try:
        with stdout_to_stderr():
            written = run_generator(
                spec.id, {kind: model.model for kind, model in models.items()}, output_dir, options,
            )
    except GeneratorInputError as exc:
        return Result(command="generate", exit_code=EXIT_USAGE, errors=[str(exc)], warnings=warnings, data=data)
    except Exception as exc:
        message = f"the '{spec.id}' generator failed: {type(exc).__name__}: {exc}"
        return Result(command="generate", exit_code=EXIT_FAILED, errors=[message], warnings=warnings, data=data)

    files = [Path(output_dir, path).as_posix() for path in written]
    lines = [f"Generated {_plural(len(files), 'file')} in {output_dir} with the '{spec.id}' generator:"]
    lines.extend(f"  {path}" for path in files)
    return Result(command="generate", warnings=warnings, files=files, data=data, text="\n".join(lines))


# ── convert ─────────────────────────────────────────────────────────────────


def _write_buml(loaded: LoadedModels, output_path: Path) -> list[str]:
    """Write the models as a B-UML project script, as the editor's export does; it has no warnings."""
    from besser.utilities.buml_code_builder.project_builder import project_to_code

    project_to_code(project=loaded.as_project(), file_path=str(output_path))
    return []


def _write_editor_json(loaded: LoadedModels, output_path: Path) -> list[str]:
    """Write the models as an editor project export; returns the warnings.

    The editor's importer reads the B-UML code the code builder writes, so the
    models go through it: models -> ``project_to_code`` -> ``project_to_json``.
    """
    from besser.utilities.buml_code_builder.project_builder import project_to_code

    try:
        from besser.utilities.web_modeling_editor.backend.constants.constants import API_VERSION, TEMP_DIR_PREFIX
        from besser.utilities.web_modeling_editor.backend.services.converters import project_to_json
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] in ("", "besser"):
            raise
        raise MissingDependencyError(missing_backend_dependency_message(exc)) from exc

    warnings = [
        f"{model.label} was left out: the editor has no diagram for a {model.kind} model"
        for model in loaded.models if model.kind in _KINDS_WITHOUT_DIAGRAM
    ]
    with tempfile.TemporaryDirectory(prefix=TEMP_DIR_PREFIX) as temp_dir:
        code_path = os.path.join(temp_dir, "project.py")
        project_to_code(project=loaded.as_project(), file_path=code_path)
        repairs: list[str] = []
        project = project_to_json(Path(code_path).read_text(encoding="utf-8"), repairs)
    warnings.extend(repairs)

    export = {"project": project, "exportedAt": datetime.now(timezone.utc).isoformat(), "version": API_VERSION}
    output_path.write_text(json.dumps(export, indent=2) + "\n", encoding="utf-8")
    return warnings


# Output suffix -> (the input suffix it converts from, the writer).
_CONVERSIONS = {
    ".json": (".py", _write_editor_json),
    ".py": (".json", _write_buml),
}


def convert(input_path: str, output_path: str) -> Result:
    """Convert a B-UML ``.py`` file to an editor JSON project, or the reverse."""
    data = {"input": input_path, "output": output_path}
    source, target = Path(input_path), Path(output_path)
    expected_source, write = _CONVERSIONS.get(target.suffix.lower(), (None, None))
    if source.suffix.lower() != expected_source:
        return failure(
            "convert",
            "convert turns a B-UML .py file into an editor .json project, or a .json export into "
            f"a .py file; got {source.name} -> {target.name}",
            data=data,
        )

    try:
        loaded = load_models(source)
        if loaded.errors:
            return Result(command="convert", exit_code=EXIT_FAILED, errors=loaded.errors, data=data)
        target.parent.mkdir(parents=True, exist_ok=True)
        with stdout_to_stderr():
            warnings = write(loaded, target)
    except ModelLoadError as exc:
        return failure("convert", str(exc), data=data)
    except Exception as exc:
        return failure("convert", f"cannot convert {source}: {type(exc).__name__}: {exc}", EXIT_FAILED, data)

    counts = _plural(len(loaded.models), "model")
    return Result(
        command="convert", warnings=warnings, files=[target.as_posix()], data=data,
        text=f"Converted {source} ({counts}) to {target}",
    )
