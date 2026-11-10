"""The ``besser`` commands: each takes parsed arguments and returns a ``Result``."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from besser.cli.loading import (
    STDIN,
    LoadedModel,
    LoadedModels,
    MissingDependencyError,
    ModelLoadError,
    load_models,
    missing_backend_dependency_message,
)
from besser.cli.output import (
    EXIT_FAILED,
    EXIT_OK,
    EXIT_USAGE,
    Result,
    failure,
    format_traceback,
    stdout_to_stderr,
)
from besser.cli.starter import STARTER_MODEL
from besser.cli.summary import describe_model, summary_lines
from besser.generators import regeneration
from besser.generators.registry import GENERATORS, GeneratorSpec, get_generator
from besser.generators.runner import GeneratorInputError, regenerate_generator, run_generator
from besser.utilities.three_way_merge import merge_available

# Model kinds the editor has no diagram for, so ``convert`` cannot write them to JSON.
_KINDS_WITHOUT_DIAGRAM = frozenset({"deployment", "feature"})
_BOOLEAN_WORDS = {"true": True, "false": False}
_REGEN_EXTRA = 'pip install "besser[regen]"'


class _Stopped(Exception):
    """A command cannot go on; carries the ``Result`` to return."""

    def __init__(self, result: Result):
        super().__init__(*result.errors)
        self.result = result


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


def _validation_text(path: str, reports: list[dict[str, Any]], project_errors: list[str]) -> list[str]:
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
    return lines


def validate(model_paths: Sequence[str], strict: bool = False, exec_py: bool = False, ocl: bool = False) -> Result:
    """Run ``validate()`` on every model of each file."""
    # With one file a message starts with the model's name; with several, with the file's too.
    several = len(model_paths) > 1
    reports: list[dict[str, Any]] = []
    load_errors: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    lines: list[str] = []
    traceback = ""
    for model_path in model_paths:
        try:
            loaded = load_models(model_path)
        except ModelLoadError as exc:
            load_errors.append(str(exc))
            traceback += format_traceback(exc)
            continue
        prefix = f"{loaded.path}: " if several else ""
        file_reports = [{"file": str(model_path), **_validate_model(model)} for model in loaded.models]
        reports.extend(file_reports)
        errors.extend(f"{prefix}{report['name']}: {error}" for report in file_reports for error in report["errors"])
        errors.extend(f"{prefix}{error}" for error in loaded.errors)
        warnings.extend(
            f"{prefix}{report['name']}: {warning}" for report in file_reports for warning in report["warnings"]
        )
        lines.extend(_validation_text(str(loaded.path), file_reports, loaded.errors))

    failed = bool(errors or load_errors) or (strict and bool(warnings))
    counts = f"{_plural(len(errors) + len(load_errors), 'error')}, {_plural(len(warnings), 'warning')}"
    if errors or load_errors:
        verdict = f"FAILED: {counts}"
    elif failed:
        verdict = f"FAILED: {counts} (--strict)"
    else:
        verdict = f"OK: {counts}"
    if load_errors:
        exit_code = EXIT_USAGE
    else:
        exit_code = EXIT_FAILED if failed else EXIT_OK
    return Result(
        command="validate",
        exit_code=exit_code,
        errors=load_errors + errors,
        warnings=warnings,
        data={"model_files": [str(path) for path in model_paths], "models": reports},
        # No report for a lone file that does not load: its error says it all.
        text="\n".join([*lines, verdict]) if lines else "",
        shown_in_text=frozenset(errors + warnings),
        traceback=traceback,
    )


# ── info ────────────────────────────────────────────────────────────────────


def info(model_path: str) -> Result:
    """Summarise the models of the file and name the generators that can run on it."""
    data: dict[str, Any] = {"model": model_path}
    try:
        loaded = load_models(model_path)
    except ModelLoadError as exc:
        return failure("info", str(exc), data=data, cause=exc)

    warnings = list(loaded.errors)
    models = []
    lines = [f"{loaded.path}: {_plural(len(loaded.models), 'model')}"]
    for model in loaded.models:
        try:
            entry = describe_model(model)
        except Exception as exc:  # a model too broken to walk: say so and keep its name and kind
            entry = {"name": model.name, "kind": model.kind, "type": type(model.model).__name__, "counts": {}}
            warnings.append(f"{model.name}: cannot be summarised: {type(exc).__name__}: {exc}")
        models.append(entry)
        lines.extend(summary_lines(entry))
    runnable = [spec.id for spec in GENERATORS if all(loaded.of_kind(kind) for kind in spec.models)]
    lines.append(f"Generators that read {'these models' if len(models) > 1 else 'this model'}: "
                 f"{', '.join(runnable) or 'none'}")
    return Result(
        command="info", warnings=warnings, data={**data, "models": models, "generators": runnable},
        text="\n".join(lines),
    )


# ── new ─────────────────────────────────────────────────────────────────────


def new(path: str, force: bool = False) -> Result:
    """Write a starter B-UML model to ``path``."""
    data = {"model": path}
    target = Path(path)
    if target.suffix.lower() != ".py":
        return failure("new", f"a B-UML model is a .py file; got {target.name}", data=data)
    if target.exists() and not force:
        return failure("new", f"{target} exists: pass --force to overwrite it", data=data)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(STARTER_MODEL.replace("MODEL_FILE", target.name), encoding="utf-8")
    except OSError as exc:
        return failure("new", f"cannot write {target}: {exc}", EXIT_FAILED, data, cause=exc)
    lines = [
        f"Wrote a starter model to {target}. Next:",
        f"  besser validate {target}",
        f"  besser info {target}",
        f"  besser generate python {target} -o out",
    ]
    return Result(command="new", files=[target.as_posix()], data=data, text="\n".join(lines))


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
        "supports_regenerate": spec.supports_regenerate,
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
    regenerable = ", ".join(spec.id for spec in GENERATORS if spec.supports_regenerate)
    lines.extend(["", f"'besser regenerate' keeps hand edits with: {regenerable}"])
    return Result(
        command="generators",
        data={"generators": [_generator_entry(spec) for spec in GENERATORS]},
        text="\n".join(lines),
    )


# ── generate and regenerate ─────────────────────────────────────────────────


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


def _models_for(spec: GeneratorSpec, loaded: LoadedModels, use: Sequence[str] = ()) -> dict[str, LoadedModel]:
    """The one model of each kind the generator reads; ``use`` names the one to take among several.

    Raises:
        GeneratorInputError: The file holds none of a kind, several with none or several of
            them named, or no model with a name of ``use``.
    """
    unknown = [name for name in use if not any(name in model.names for model in loaded.models)]
    if unknown:
        held = ", ".join(model.name for model in loaded.models)
        raise GeneratorInputError(f"{loaded.path} holds no model named {', '.join(unknown)} (it holds: {held})")
    chosen = {}
    for kind in spec.models:
        candidates = loaded.of_kind(kind)
        if not candidates:
            held = ", ".join(sorted({model.kind for model in loaded.models}))
            raise GeneratorInputError(
                f"the '{spec.id}' generator reads a {kind} model and {loaded.path} holds none (it holds: {held})"
            )
        if len(candidates) > 1:
            named = [model for model in candidates if model.names.intersection(use)]
            if len(named) != 1:
                names = ", ".join(model.name for model in candidates)
                raise GeneratorInputError(
                    f"{loaded.path} holds {len(candidates)} {kind} models ({names}) and the '{spec.id}' "
                    "generator reads one: name it with --use NAME"
                )
            candidates = named
        chosen[kind] = candidates[0]
    return chosen


def _find_generator(command: str, generator_id: str, data: dict[str, Any]) -> GeneratorSpec:
    try:
        return get_generator(generator_id)
    except KeyError:
        known = ", ".join(spec.id for spec in GENERATORS)
        raise _Stopped(failure(
            command, f"unknown generator '{generator_id}'; available generators: {known}", data=data,
        )) from None


def _generator_input(
    command: str, spec: GeneratorSpec, model_path: str, option_assignments: list[str], use: Sequence[str],
    data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """The models the generator reads, by kind, its options and the validation warnings.

    Raises:
        _Stopped: The file does not load, an option does not parse, or a model has validation errors.
    """
    try:
        options = parse_options(spec, option_assignments)
        models = _models_for(spec, load_models(model_path), use)
    except (GeneratorInputError, ModelLoadError) as exc:
        raise _Stopped(failure(command, str(exc), data=data, cause=exc)) from exc

    reports = [_validate_model(model) for model in models.values()]
    warnings = [f"{report['name']}: {warning}" for report in reports for warning in report["warnings"]]
    errors = [f"{report['name']}: {error}" for report in reports for error in report["errors"]]
    if errors:
        errors.append("nothing was generated: the model has validation errors")
        raise _Stopped(Result(command=command, exit_code=EXIT_FAILED, errors=errors, warnings=warnings, data=data))
    return {kind: model.model for kind, model in models.items()}, options, warnings


def _generator_failure(command: str, spec: GeneratorSpec, exc: Exception, warnings: list[str],
                       data: dict[str, Any]) -> Result:
    """What a raising generator is reported as: its input refused (2), or a failure (1)."""
    if isinstance(exc, GeneratorInputError):
        return Result(command=command, exit_code=EXIT_USAGE, errors=[str(exc)], warnings=warnings, data=data)
    message = f"the '{spec.id}' generator failed: {type(exc).__name__}: {exc}"
    return Result(command=command, exit_code=EXIT_FAILED, errors=[message], warnings=warnings, data=data,
                  traceback=format_traceback(exc))


def _require_no_output(spec: GeneratorSpec, model_path: str, output_dir: str, data: dict[str, Any]) -> None:
    """``generate`` overwrites: refuse a folder that holds something, unless ``--force`` says to.

    Raises:
        _Stopped: The folder has a regeneration record, or is not empty.
    """
    folder = Path(output_dir)
    if (folder / regeneration.RECORD_FILENAME).exists():
        message = (
            f"{output_dir} holds generated code and the record of it ({regeneration.RECORD_FILENAME}): "
            f"'besser regenerate {spec.id} {model_path} -o {output_dir}' brings it up to the model and keeps "
            "hand edits; --force overwrites them"
        )
    elif folder.is_dir() and any(folder.iterdir()):
        message = (
            f"{output_dir} is not empty and generating overwrites what is there: pass --force to overwrite, "
            "or keep hand edits with 'besser regenerate' (see 'besser regenerate --help')"
        )
    else:
        return
    raise _Stopped(failure("generate", message, data=data))


def generate(generator_id: str, model_path: str, output_dir: str, option_assignments: list[str],
             use: Sequence[str] = (), force: bool = False, dry_run: bool = False) -> Result:
    """Validate the models the generator reads, then run it into ``output_dir``."""
    data: dict[str, Any] = {"generator": generator_id, "model": model_path, "output_dir": output_dir}
    try:
        spec = _find_generator("generate", generator_id, data)
        if not force:
            _require_no_output(spec, model_path, output_dir, data)
        models, options, warnings = _generator_input("generate", spec, model_path, option_assignments, use, data)
    except _Stopped as stopped:
        return stopped.result

    # --force over a regenerated folder: its record describes the output this run overwrites.
    record = Path(output_dir, regeneration.RECORD_FILENAME)
    had_record = record.is_file()
    try:
        with stdout_to_stderr():
            written = run_generator(spec.id, models, output_dir, options)
    except Exception as exc:
        return _generator_failure("generate", spec, exc, warnings, data)

    if had_record:
        record.unlink(missing_ok=True)
        warnings.append(
            f"removed {record.as_posix()}: it recorded the output this run overwrote. To keep later edits, "
            f"run 'besser regenerate {spec.id} {model_path} -o {output_dir} --adopt'"
        )
    files = [Path(output_dir, path).as_posix() for path in written]
    lines = [f"Generated {_plural(len(files), 'file')} in {output_dir} with the '{spec.id}' generator:"]
    lines.extend(f"  {path}" for path in files)
    return Result(command="generate", warnings=warnings, files=files, data=data, text="\n".join(lines))


@contextlib.contextmanager
def _without_regeneration_log() -> Iterator[None]:
    """Mute the warnings ``regenerate()`` logs about its report: the command prints the report itself."""
    logger = logging.getLogger(regeneration.__name__)
    level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(level)


# Report list -> what the command says about the files in it; conflicts are errors instead.
_REPORT_WARNINGS = {
    "merged": "the model's changes were merged into your edits; a clean merge can still break, so run your tests",
    "kept_removed_from_model": "the model no longer produces these files and they may hold your work, so they were kept",
    "not_restored": "you deleted these files and the model changed them; they stay deleted",
    "not_merged": "these files clash with the model and cannot be merged as text; they were left as they were",
    "differs_from_model": "these files differ from this model's output and count as your edits from now on "
                          "(a long list usually means it is not the model they were generated from)",
}
# Report list -> its label in the text report.
_REPORT_LABELS = {"written": "written", "merged": "merged", "conflicted": "conflict", "deleted": "deleted"}


def regenerate(generator_id: str, model_path: str, output_dir: str, option_assignments: list[str],
               use: Sequence[str] = (), adopt: bool = False, dry_run: bool = False) -> Result:
    """Bring ``output_dir`` up to the model with the generator, keeping the edits made in it."""
    data: dict[str, Any] = {"generator": generator_id, "model": model_path, "output_dir": output_dir}
    try:
        spec = _find_generator("regenerate", generator_id, data)
        if not spec.supports_regenerate:
            supported = ", ".join(spec.id for spec in GENERATORS if spec.supports_regenerate)
            raise _Stopped(failure(
                "regenerate",
                f"the '{spec.id}' generator does not support regenerate: its output differs from run to run for "
                f"one model, so a merge could not tell a model change from noise. 'besser generate {spec.id} ... "
                f"--force' overwrites the folder instead. Generators that support it: {supported}",
                data=data,
            ))
        if not merge_available():
            raise _Stopped(failure(
                "regenerate", f"regenerate needs the pygit2 package, which is not installed: {_REGEN_EXTRA}",
                data=data,
            ))
        models, options, warnings = _generator_input("regenerate", spec, model_path, option_assignments, use, data)
    except _Stopped as stopped:
        return stopped.result

    try:
        with stdout_to_stderr(), _without_regeneration_log():
            report = regenerate_generator(spec.id, models, output_dir, options, adopt=adopt)
    except regeneration.RegenerationError as exc:
        message = str(exc)
        if isinstance(exc, regeneration.UnrecordedOutputError):
            message += (
                f" With this command: 'besser regenerate {spec.id} <that model> -o {output_dir} --adopt', "
                "then regenerate with the changed model."
            )
        return Result(command="regenerate", exit_code=EXIT_USAGE, errors=[message], warnings=warnings, data=data,
                      traceback=format_traceback(exc))
    except Exception as exc:
        return _generator_failure("regenerate", spec, exc, warnings, data)

    # As ``files``: paths under the -o given, not relative to it.
    lists = {
        name: [Path(output_dir, path).as_posix() for path in paths]
        for name, paths in dataclasses.asdict(report).items()
    }
    data.update(lists, adopted=adopt)
    warnings.extend(
        f"{reason}: {', '.join(lists[name])}" for name, reason in _REPORT_WARNINGS.items() if lists[name]
    )
    errors = [
        f"{path}: your edit and the model changed the same lines. Resolve the conflict markers "
        "('<<<<<<< your edit' ... '>>>>>>> generated from your model'), then run regenerate again"
        for path in lists["conflicted"]
    ]
    changed = [(label, path) for name, label in _REPORT_LABELS.items() for path in lists[name]]
    if adopt:
        lines = [f"Recorded {output_dir} as the '{spec.id}' generator's output for {model_path}; no file was "
                 "changed. 'besser regenerate' now keeps the edits made in it."]
    elif changed:
        counts = ", ".join(f"{len(lists[name])} {label}" for name, label in _REPORT_LABELS.items() if lists[name])
        lines = [f"Regenerated {output_dir} with the '{spec.id}' generator ({counts}):"]
        width = max(len(label) for label, _ in changed)
        lines.extend(f"  {label.ljust(width)}  {path}" for label, path in changed)
    else:
        lines = [f"{output_dir} is up to date with {model_path}: no file changed."]
    return Result(
        command="regenerate", exit_code=EXIT_FAILED if errors else EXIT_OK, errors=errors, warnings=warnings,
        files=sorted(lists["written"] + lists["merged"] + lists["conflicted"]), data=data, text="\n".join(lines),
    )


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


def convert(input_path: str, output_path: str, dry_run: bool = False) -> Result:
    """Convert a B-UML ``.py`` file to an editor JSON project, or the reverse."""
    data = {"input": input_path, "output": output_path}
    source, target = Path(input_path), Path(output_path)
    expected_source, write = _CONVERSIONS.get(target.suffix.lower(), (None, None))
    # Standard input has no name to tell what it holds: that is known once it is loaded.
    from_stdin = input_path == STDIN
    wrong_direction = (
        "convert turns a B-UML .py file into an editor .json project, or a .json export into "
        f"a .py file; got {{held}} -> {target.name}"
    )
    if write is None or (not from_stdin and source.suffix.lower() != expected_source):
        return failure("convert", wrong_direction.format(held=source.name), data=data)

    try:
        loaded = load_models(input_path)
        source = loaded.path
        if loaded.suffix != expected_source:
            held = "a B-UML script" if loaded.suffix == ".py" else "editor JSON"
            return failure("convert", wrong_direction.format(held=f"{held} on standard input"), data=data)
        if loaded.errors:
            return Result(command="convert", exit_code=EXIT_FAILED, errors=loaded.errors, data=data)
        target.parent.mkdir(parents=True, exist_ok=True)
        with stdout_to_stderr():
            warnings = write(loaded, target)
    except ModelLoadError as exc:
        return failure("convert", str(exc), data=data, cause=exc)
    except Exception as exc:
        return failure("convert", f"cannot convert {source}: {type(exc).__name__}: {exc}", EXIT_FAILED, data, exc)

    counts = _plural(len(loaded.models), "model")
    return Result(
        command="convert", warnings=warnings, files=[target.as_posix()], data=data,
        text=f"Converted {source} ({counts}) to {target}",
    )
