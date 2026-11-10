"""The ``besser`` commands: each takes parsed arguments and returns a ``Result``."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import shlex
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, TypeVar

from besser.cli import loading
from besser.cli.discovery import SKIP_NO_MODEL, SKIP_PY, ModelFiles, SkippedFile, find_model_files
from besser.cli.loading import (
    STDIN,
    LoadedModel,
    LoadedModels,
    MissingDependencyError,
    ModelLoadError,
    NoModelError,
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
    plural,
    stdout_to_stderr,
)
from besser.cli.ocl_check import OclOutcome, OclUnavailable, check_ocl, load_checker
from besser.cli.starter import STARTER_MODEL
from besser.cli.summary import describe_model, summary_lines
from besser.generators import regeneration
from besser.generators.registry import GENERATORS, GeneratorSpec, get_generator
from besser.generators.runner import (
    GeneratorInputError,
    preview_generator,
    regenerate_generator,
    run_generator,
)
from besser.utilities.three_way_merge import merge_available

# Model kinds the editor has no diagram for, so ``convert`` cannot write them to JSON.
_KINDS_WITHOUT_DIAGRAM = frozenset({"deployment", "feature"})
_BOOLEAN_WORDS = {"true": True, "false": False}
_REGEN_EXTRA = 'pip install "besser[regen]"'
# The first line of a dry run's report.
_DRY_RUN = "Dry run: nothing was written."
# What a dry run says of each file, in the order its counts are given.
_PREVIEW_STATUSES = (regeneration.NEW, regeneration.CHANGED, regeneration.UNCHANGED)
_Outcome = TypeVar("_Outcome")


class _Stopped(Exception):
    """A command cannot go on; carries the ``Result`` to return."""

    def __init__(self, result: Result):
        super().__init__(*result.errors)
        self.result = result


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
    lines = [f"{path}: {plural(len(reports), 'model')}"]
    for report in reports:
        if not report["validated"] and not (report["errors"] or report["warnings"]):
            status = "no validation rules"
        elif report["errors"] or report["warnings"]:
            status = f"{plural(len(report['errors']), 'error')}, {plural(len(report['warnings']), 'warning')}"
        else:
            status = "ok"
        lines.append(f"  {report['name']} ({report['type']}): {status}")
        lines.extend(f"    error: {error}" for error in report["errors"])
        lines.extend(f"    warning: {warning}" for warning in report["warnings"])
    lines.extend(f"  error: {error}" for error in project_errors)
    return lines


@dataclasses.dataclass
class _FileCheck:
    """The outcome of ``validate`` on one model file, in the JSON output's ``results`` shape."""

    file: str
    exit_code: int = EXIT_OK
    # Each starts with the model's name, except a load error and the project's own errors.
    errors: list[str] = dataclasses.field(default_factory=list)
    warnings: list[str] = dataclasses.field(default_factory=list)
    models: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    load_error: str = ""

    def as_json(self) -> dict[str, Any]:
        return {"file": self.file, "ok": self.exit_code == EXIT_OK, "exit_code": self.exit_code,
                "errors": self.errors, "warnings": self.warnings, "models": self.models}


def _check_file(path: str, loaded: LoadedModels, strict: bool,
                checker: Any | None, exec_py: bool) -> tuple[_FileCheck, OclOutcome | None]:
    """Run ``validate()``, and the OCL check when ``checker`` is given, on the models of ``loaded``.

    OCL invariants are evaluated when running the file's code is allowed: it is a
    ``.py`` file, or ``exec_py`` (see ``besser.cli.ocl_check``).
    """
    check = _FileCheck(file=path, models=[{"file": path, **_validate_model(model)} for model in loaded.models])
    outcome = None
    if checker is not None:
        outcome = check_ocl(loaded.models, checker, evaluate=exec_py or loaded.suffix == ".py")
        for model, report in zip(loaded.models, check.models):
            report["errors"].extend(outcome.errors.get(id(model), []))
            report["warnings"].extend(outcome.warnings.get(id(model), []))
    check.errors = [f"{report['name']}: {error}" for report in check.models for error in report["errors"]]
    check.errors.extend(loaded.errors)
    check.warnings = [f"{report['name']}: {warning}" for report in check.models for warning in report["warnings"]]
    check.exit_code = EXIT_FAILED if check.errors or (strict and check.warnings) else EXIT_OK
    return check, outcome


# How the run says that .py files found in a directory were not loaded.
_EXEC_PY_HINT = "pass --exec-py to load the .py files found in a directory"


def _nothing_to_validate(model_paths: Sequence[str], found: ModelFiles, exec_py: bool) -> Result:
    where = ", ".join(model_paths)
    skipped_py = sum(skipped.reason == SKIP_PY for skipped in found.skipped)
    if exec_py:
        message = f"nothing to validate: no model file found in {where}"
    elif skipped_py:
        message = (f"nothing to validate: no editor .json export found in {where} "
                   f"({plural(skipped_py, '.py file')} skipped; {_EXEC_PY_HINT})")
    else:
        message = (f"nothing to validate: no editor .json export found in {where} "
                   "(.py models in a directory are loaded with --exec-py)")
    skipped = [{"file": skipped.path, "reason": skipped.reason} for skipped in found.skipped]
    return failure("validate", message, data={"model_files": [], "models": [], "results": [], "skipped": skipped})


def _ocl_note(constraints: int, evaluations: int, unevaluated: int) -> str:
    """The run's one note on the OCL check, from its counts over every file (see ``OclOutcome``)."""
    if not constraints:
        return "OCL: the models have no OCL constraints"
    note = f"OCL: {plural(constraints, 'constraint')} checked"
    if evaluations or not unevaluated:
        note += f", invariants evaluated on {plural(evaluations, 'object model')}"
    if unevaluated:
        note += (f"; invariants not evaluated on {plural(unevaluated, 'object model')} of editor JSON: the OCL "
                 "evaluator runs Python built from the models' values; pass --exec-py to evaluate them")
    return note


def _load_error_text(path: str, message: str) -> str:
    """A file that does not load, as the report lists it among the files checked."""
    return f"{path}: does not load: {message.removeprefix(f'{Path(path)}: ').removeprefix(f'{path}: ')}"


def validate(model_paths: Sequence[str], strict: bool = False, exec_py: bool = False, ocl: bool = False) -> Result:
    """Run ``validate()`` on every model of each file, and of each model file found in a directory.

    With ``ocl``, also check the domain models' OCL constraints and evaluate
    them on the object models. The exit status is the worst file's.
    """
    found = find_model_files(model_paths, exec_py=exec_py)
    skipped = list(found.skipped)
    checker, warnings, notes = None, [], []
    if ocl:
        try:
            checker = load_checker()
        except OclUnavailable as exc:
            warnings.append(str(exc))

    checks: list[_FileCheck] = []
    lines: list[str] = []
    traceback = ""
    constraints = evaluations = unevaluated = 0
    for model_file in found.files:
        load_error = None
        try:
            loaded = load_models(model_file.path)
        except NoModelError as exc:
            # A directory holds files that are not models (helpers, an empty project): skip what it holds;
            # a file named on the command line must be a model.
            if not model_file.explicit:
                skipped.append(SkippedFile(model_file.path, SKIP_NO_MODEL))
                continue
            load_error = exc
        except ModelLoadError as exc:
            load_error = exc
        if load_error is not None:
            checks.append(_FileCheck(file=model_file.path, exit_code=EXIT_USAGE, errors=[str(load_error)],
                                     load_error=str(load_error)))
            lines.append(_load_error_text(model_file.path, str(load_error)))
            traceback += format_traceback(load_error)
            continue
        check, outcome = _check_file(model_file.path, loaded, strict, checker, exec_py)
        checks.append(check)
        if outcome is not None:
            constraints += outcome.constraints
            evaluations += outcome.evaluations
            unevaluated += outcome.unevaluated
        lines.extend(_validation_text(str(loaded.path), check.models, loaded.errors))
    if not checks:
        return _nothing_to_validate(model_paths, dataclasses.replace(found, skipped=skipped), exec_py)

    # With one file a message starts with the model's name; with several, or a directory, with the file's too.
    several = found.searched_directory or len(checks) > 1
    load_errors = [check.load_error for check in checks if check.load_error]
    errors = [f"{check.file}: {error}" if several else error
              for check in checks if not check.load_error for error in check.errors]
    file_warnings = [f"{check.file}: {warning}" if several else warning
                     for check in checks for warning in check.warnings]
    # The report lists the files' messages; the run's own (OCL not installed) go to standard error.
    # A lone file that does not load has no report: its error is printed on its own.
    shown_in_text = frozenset(errors + file_warnings + (load_errors if several else []))
    if not several and load_errors:
        lines = []
    warnings = file_warnings + warnings

    skipped_py = sum(entry.reason == SKIP_PY for entry in skipped)
    if skipped_py:
        notes.append(f"{plural(skipped_py, '.py file')} skipped; {_EXEC_PY_HINT}")
    if checker is not None and len(load_errors) < len(checks):
        notes.append(_ocl_note(constraints, evaluations, unevaluated))

    exit_code = max(check.exit_code for check in checks)
    if strict and warnings:
        exit_code = max(exit_code, EXIT_FAILED)
    counts = f"{plural(len(errors) + len(load_errors), 'error')}, {plural(len(warnings), 'warning')}"
    if errors or load_errors:
        verdict = f"FAILED: {counts}"
    elif exit_code != EXIT_OK:
        verdict = f"FAILED: {counts} (--strict)"
    else:
        verdict = f"OK: {counts}"
    skipped_lines = [f"skipped: {entry.path}: {entry.reason}" for entry in skipped]
    return Result(
        command="validate",
        exit_code=exit_code,
        errors=load_errors + errors,
        warnings=warnings,
        notes=notes,
        data={
            "model_files": [check.file for check in checks],
            "models": [report for check in checks for report in check.models],
            "results": [check.as_json() for check in checks],
            "skipped": [{"file": entry.path, "reason": entry.reason} for entry in skipped],
        },
        text="\n".join([*lines, *skipped_lines, verdict]) if lines or skipped_lines else "",
        shown_in_text=shown_in_text,
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
    notes = [
        f"{loaded.path} holds {plural(len(loaded.of_kind(kind)), kind + ' model')} and names no current one: "
        f"{reason}; generators need --use NAME"
        for kind, reason in loaded.current_unresolved.items()
    ]
    current = {id(marked.model) for marked in loaded.current.values()}
    models = []
    lines = [f"{loaded.path}: {plural(len(loaded.models), 'model')}"]
    for model in loaded.models:
        is_current = id(model) in current
        try:
            entry = describe_model(model, current=is_current)
        except Exception as exc:  # a model too broken to walk: say so and keep its name and kind
            entry = {"name": model.name, "kind": model.kind, "type": type(model.model).__name__,
                     "current": is_current, "counts": {}}
            warnings.append(f"{model.name}: cannot be summarised: {type(exc).__name__}: {exc}")
        models.append(entry)
        lines.extend(summary_lines(entry))
    runnable = [spec.id for spec in GENERATORS if all(loaded.of_kind(kind) for kind in spec.models)]
    lines.append(f"Generators that read {'these models' if len(models) > 1 else 'this model'}: "
                 f"{', '.join(runnable) or 'none'}")
    return Result(
        command="info", warnings=warnings, notes=notes, data={**data, "models": models, "generators": runnable},
        text="\n".join(lines), text_is_result=True,
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
        text="\n".join(lines), text_is_result=True,
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


def _one_named(loaded: LoadedModels, kind: str, named: list[LoadedModel], use: Sequence[str],
               current: loading.CurrentModel | None) -> tuple[LoadedModel, str]:
    """The one of ``named``, the models of ``kind`` that ``use`` names, to read, and a note when the
    project's current diagram told models of one name apart (empty otherwise).

    Raises:
        GeneratorInputError: ``use`` names several of them, or one name names several and none is current.
    """
    if len(named) == 1:
        return named[0], ""
    used = sorted({name for model in named for name in model.names.intersection(use)})
    which = "; ".join(model.diagram or model.label for model in named)
    if len(used) > 1:
        raise GeneratorInputError(
            f"--use names {len(named)} {kind} models of {loaded.path} ({', '.join(used)}) and a generator reads one"
        )
    if current is not None and any(model is current.model for model in named):
        return current.model, (
            f"{loaded.path} holds {len(named)} {kind} models named {used[0]} ({which}); using "
            f"{current.model.diagram}, the project's current {current.diagram_type}"
        )
    reason = loaded.current_unresolved.get(kind, "it is none of them")
    raise GeneratorInputError(
        f"--use {used[0]} is ambiguous: {loaded.path} holds {len(named)} {kind} models of that name ({which}) "
        f"and the project names no current one to choose ({reason}); rename the diagrams in the editor"
    )


def _models_for(
    spec: GeneratorSpec, loaded: LoadedModels, use: Sequence[str] = (),
) -> tuple[dict[str, LoadedModel], list[str]]:
    """The one model of each kind the generator reads, and notes on the ones picked among several.

    ``use`` names the model to take among several; without it, an editor project's current
    diagram of that kind (``currentDiagramIndices``) is taken, and a note says so.

    Raises:
        GeneratorInputError: The file holds none of a kind, several with none or several of
            them named and no current diagram, or no model with a name of ``use``.
    """
    unknown = [name for name in use if not any(name in model.names for model in loaded.models)]
    if unknown:
        held = ", ".join(model.name for model in loaded.models)
        raise GeneratorInputError(f"{loaded.path} holds no model named {', '.join(unknown)} (it holds: {held})")
    chosen, notes = {}, []
    for kind in spec.models:
        candidates = loaded.of_kind(kind)
        if not candidates:
            held = ", ".join(sorted({model.kind for model in loaded.models}))
            raise GeneratorInputError(
                f"the '{spec.id}' generator reads a {kind} model and {loaded.path} holds none (it holds: {held})"
            )
        if len(candidates) > 1:
            names = ", ".join(model.name for model in candidates)
            named = [model for model in candidates if model.names.intersection(use)]
            current = loaded.current.get(kind)
            if named:
                model, note = _one_named(loaded, kind, named, use, current)
                named = [model]
                notes.extend([note] if note else [])
            elif current is not None:
                named = [current.model]
                notes.append(
                    f"{loaded.path} holds {len(candidates)} {kind} models ({names}); using {current.model.name}, "
                    f"the project's current {current.diagram_type} (--use NAME picks another)"
                )
            if len(named) != 1:
                message = (
                    f"{loaded.path} holds {len(candidates)} {kind} models ({names}) and the '{spec.id}' "
                    "generator reads one: name it with --use NAME"
                )
                if kind in loaded.current_unresolved:
                    message += f" (the project names no current one: {loaded.current_unresolved[kind]})"
                raise GeneratorInputError(message)
            candidates = named
        chosen[kind] = candidates[0]
    return chosen, notes


def _find_generator(command: str, generator_id: str, data: dict[str, Any]) -> GeneratorSpec:
    try:
        return get_generator(generator_id)
    except KeyError:
        known = ", ".join(spec.id for spec in GENERATORS)
        raise _Stopped(failure(
            command, f"unknown generator '{generator_id}'; available generators: {known}", data=data,
        )) from None


@dataclasses.dataclass
class _GeneratorInput:
    """What the generator runs on, as read from the command line."""

    models: dict[str, Any]  # kind -> the model
    options: dict[str, Any]
    warnings: list[str]
    notes: list[str]
    # The model names a later run passes as --use to read the same models (see ``_rerun_use``).
    use: list[str]


def _rerun_use(chosen: dict[str, LoadedModel], loaded: LoadedModels, use: Sequence[str]) -> list[str]:
    """The ``--use`` names that make another run read the models of ``chosen``.

    A kind needs one when ``--use`` was given or the file holds several models of it (the
    project's current diagram, picked without ``--use``, may change before the next run), and
    gets one when the name tells the model from the others of its kind.
    """
    names = []
    for kind, model in chosen.items():
        candidates = loaded.of_kind(kind)
        if (use or len(candidates) > 1) and sum(model.name in other.names for other in candidates) == 1:
            names.append(model.name)
    return names


def _generator_input(
    command: str, spec: GeneratorSpec, model_path: str, option_assignments: list[str], use: Sequence[str],
    data: dict[str, Any],
) -> _GeneratorInput:
    """The models the generator reads and its options, refused when the models have errors.

    Adds ``used`` to ``data``: kind -> the name of the model read.

    Raises:
        _Stopped: The file does not load, an option does not parse, a model has validation errors,
            or the diagrams of an editor project reference ones it does not hold.
    """
    try:
        options = parse_options(spec, option_assignments)
        loaded = load_models(model_path)
        models, notes = _models_for(spec, loaded, use)
    except (GeneratorInputError, ModelLoadError) as exc:
        raise _Stopped(failure(command, str(exc), data=data, cause=exc)) from exc
    data["used"] = {kind: model.name for kind, model in models.items()}

    reports = [_validate_model(model) for model in models.values()]
    warnings = [f"{report['name']}: {warning}" for report in reports for warning in report["warnings"]]
    errors = [f"{report['name']}: {error}" for report in reports for error in report["errors"]]
    if errors:
        errors.append("nothing was generated: the model has validation errors")
    elif loaded.errors:
        errors = [*loaded.errors, "nothing was generated: the project's diagrams reference elements or diagrams "
                                  "it does not hold"]
    if errors:
        raise _Stopped(Result(command=command, exit_code=EXIT_FAILED, errors=errors, warnings=warnings,
                              notes=notes, data=data))
    return _GeneratorInput({kind: model.model for kind, model in models.items()}, options, warnings, notes,
                           _rerun_use(models, loaded, use))


def _regenerate_command(spec: GeneratorSpec, model_path: str | None, output_dir: str,
                        option_assignments: Sequence[str], use: Sequence[str], adopt: bool = False) -> str:
    """The ``besser regenerate`` command line for this output, quoted for the shell and for the message.

    ``model_path`` ``None`` is a placeholder for the model to give.
    """
    tail = ["-o", output_dir, *(word for assignment in option_assignments for word in ("--option", assignment)),
            *(word for name in use for word in ("--use", name))]
    if adopt:
        tail.append("--adopt")
    model = "<that model>" if model_path is None else shlex.quote(model_path)
    command = f"{shlex.join(['besser', 'regenerate', spec.id])} {model} {shlex.join(tail)}"
    # In single quotes as the other commands named in messages, unless shell quoting is in it.
    return f'"{command}"' if "'" in command else f"'{command}'"


def _needs_regen_extra() -> str:
    """What a suggested ``regenerate`` needs installed first, when it is not."""
    return "" if merge_available() else f" (needs {_REGEN_EXTRA})"


def _file_in_the_way(path: Path) -> Path | None:
    """The nearest of ``path``'s parents that exists, when it is not a directory."""
    for parent in path.parents:
        if os.path.lexists(parent):
            return None if parent.is_dir() else parent
    return None


def _output_dir_problem(output_dir: str) -> str | None:
    """Why ``-o`` cannot be the generator's output directory, or ``None`` when it can."""
    folder = Path(output_dir)
    if os.path.lexists(folder) and not folder.is_dir():
        return f"{output_dir} is a file: -o takes a directory"
    blocking = _file_in_the_way(folder)
    if blocking is not None:
        return f"{output_dir} cannot be created: {blocking.as_posix()} is a file"
    return None


def _require_output_dir(command: str, output_dir: str, data: dict[str, Any]) -> None:
    """Raises: _Stopped: ``-o`` names a file, or a path under one."""
    problem = _output_dir_problem(output_dir)
    if problem:
        raise _Stopped(failure(command, problem, data=data))


def _generator_failure(command: str, spec: GeneratorSpec, exc: Exception, warnings: list[str],
                       data: dict[str, Any], notes: list[str]) -> Result:
    """What a raising generator is reported as: its input refused (2), or a failure (1)."""
    if isinstance(exc, GeneratorInputError):
        return Result(command=command, exit_code=EXIT_USAGE, errors=[str(exc)], warnings=warnings, notes=notes,
                      data=data)
    message = f"the '{spec.id}' generator failed: {type(exc).__name__}: {exc}"
    return Result(command=command, exit_code=EXIT_FAILED, errors=[message], warnings=warnings, notes=notes,
                  data=data, traceback=format_traceback(exc))


def _holds_a_file(folder: Path) -> bool:
    return any(path.is_file() for path in folder.rglob("*"))


def _require_no_output(spec: GeneratorSpec, model_path: str, output_dir: str, option_assignments: list[str],
                       use: Sequence[str], data: dict[str, Any]) -> None:
    """``generate`` overwrites: refuse a folder that holds something, unless ``--force`` says to.

    The models are not loaded yet, so the ``regenerate`` command named gives ``--use`` as it was given.

    Raises:
        _Stopped: The folder has a regeneration record, or is not empty.
    """
    folder = Path(output_dir)
    if (folder / regeneration.RECORD_FILENAME).exists():
        command = _regenerate_command(spec, model_path, output_dir, option_assignments, use)
        message = (
            f"{output_dir} holds generated code and the record of it ({regeneration.RECORD_FILENAME}): "
            f"{command} brings it up to the model and keeps hand edits{_needs_regen_extra()}; --force overwrites them"
        )
    elif folder.is_dir() and any(folder.iterdir()):
        message = f"{output_dir} is not empty and generating overwrites what is there: pass --force to overwrite"
        if not spec.supports_regenerate:
            message += f" (the '{spec.id}' generator does not support 'besser regenerate', which keeps hand edits)"
        elif _holds_a_file(folder):
            adopt = _regenerate_command(spec, model_path, output_dir, option_assignments, use, adopt=True)
            message += (f", or keep hand edits with 'besser regenerate' (a folder generated without a record needs "
                        f"{adopt} first; see 'besser regenerate --help')")
        else:
            message += ", or keep hand edits with 'besser regenerate' (see 'besser regenerate --help')"
    else:
        return
    raise _Stopped(failure("generate", message, data=data))


def _with_record(spec: GeneratorSpec, output_dir: str, warnings: list[str],
                 run: Callable[[bool], _Outcome]) -> tuple[_Outcome, bool]:
    """``run(record)`` with the record ``regenerate`` merges against, when the generator and the folder
    can take one, and whether it was; a refused record is a warning in ``warnings``."""
    if spec.supports_regenerate:
        try:
            return run(True), True
        except regeneration.NotRecordableError as exc:
            warnings.append(f"{output_dir} gets no {regeneration.RECORD_FILENAME}, so 'besser regenerate' cannot "
                            f"keep the edits made in it: {exc}")
    return run(False), False


def _record_note(spec: GeneratorSpec, model_path: str, output_dir: str, option_assignments: list[str],
                 given: _GeneratorInput, dry_run: bool) -> str:
    """The note that ``generate`` records its output, naming the ``regenerate`` command that merges into it."""
    record = Path(output_dir, regeneration.RECORD_FILENAME).as_posix()
    command = _regenerate_command(spec, model_path, output_dir, option_assignments, given.use)
    if dry_run:
        return (f"would record the output in {record}: {command} would then bring it up to a changed model "
                f"and keep your edits{_needs_regen_extra()}")
    return (f"recorded the output in {record}: {command} brings it up to a changed model and keeps your edits"
            f"{_needs_regen_extra()}")


def _generate_dry_run(spec: GeneratorSpec, model_path: str, output_dir: str, option_assignments: list[str],
                      given: _GeneratorInput, data: dict[str, Any]) -> Result:
    """What ``generate`` would write, by file: new, changed or unchanged. Nothing is written.

    The record is bookkeeping, not output: it is not listed, and ``recorded`` says whether it would be written.
    """
    warnings, notes = given.warnings, given.notes
    try:
        with stdout_to_stderr():
            plan, data["recorded"] = _with_record(spec, output_dir, warnings, lambda record: preview_generator(
                spec.id, given.models, output_dir, given.options, record=record))
    except regeneration.RegenerationError as exc:
        return Result(command="generate", exit_code=EXIT_USAGE, errors=[str(exc)], warnings=warnings, notes=notes,
                      data=data, traceback=format_traceback(exc))
    except Exception as exc:
        return _generator_failure("generate", spec, exc, warnings, data, notes)

    if data["recorded"]:
        plan.pop(regeneration.RECORD_FILENAME, None)
        notes.append(_record_note(spec, model_path, output_dir, option_assignments, given, dry_run=True))
    data["would_write"] = {Path(output_dir, rel).as_posix(): status for rel, status in plan.items()}
    if not spec.supports_regenerate and regeneration.CHANGED in plan.values():
        notes.append(f"the '{spec.id}' generator's output differs from run to run for one model, "
                     "so a file can read as changed when the model is not")
    counts = ", ".join(f"{list(plan.values()).count(status)} {status}" for status in _PREVIEW_STATUSES
                       if status in plan.values())
    lines = [_DRY_RUN, f"The '{spec.id}' generator would write {plural(len(plan), 'file')} in {output_dir}"
                       f"{f' ({counts})' if counts else ''}:"]
    width = max((len(status) for status in plan.values()), default=0)
    lines.extend(f"  {status.ljust(width)}  {path}" for path, status in data["would_write"].items())
    return Result(command="generate", warnings=warnings, notes=notes, data=data, text="\n".join(lines))


def generate(generator_id: str, model_path: str, output_dir: str, option_assignments: list[str],
             use: Sequence[str] = (), force: bool = False, dry_run: bool = False) -> Result:
    """Validate the models the generator reads, then run it into ``output_dir``, recording its output when
    it supports ``regenerate``; with ``dry_run``, say what it would write instead."""
    data: dict[str, Any] = {"generator": generator_id, "model": model_path, "output_dir": output_dir,
                            "dry_run": dry_run, "recorded": False}
    try:
        spec = _find_generator("generate", generator_id, data)
        # A dry run says what the run would do, and that includes these refusals.
        _require_output_dir("generate", output_dir, data)
        if not force:
            _require_no_output(spec, model_path, output_dir, option_assignments, use, data)
        given = _generator_input("generate", spec, model_path, option_assignments, use, data)
    except _Stopped as stopped:
        return stopped.result
    if dry_run:
        return _generate_dry_run(spec, model_path, output_dir, option_assignments, given, data)

    warnings, notes = given.warnings, given.notes
    try:
        with stdout_to_stderr():
            written, data["recorded"] = _with_record(spec, output_dir, warnings, lambda record: run_generator(
                spec.id, given.models, output_dir, given.options, record=record))
    except regeneration.RegenerationError as exc:
        return Result(command="generate", exit_code=EXIT_USAGE, errors=[str(exc)], warnings=warnings, notes=notes,
                      data=data, traceback=format_traceback(exc))
    except Exception as exc:
        return _generator_failure("generate", spec, exc, warnings, data, notes)

    if data["recorded"]:
        notes.append(_record_note(spec, model_path, output_dir, option_assignments, given, dry_run=False))
    files = [Path(output_dir, path).as_posix() for path in written]
    lines = [f"Generated {plural(len(files), 'file')} in {output_dir} with the '{spec.id}' generator:"]
    lines.extend(f"  {path}" for path in files)
    return Result(command="generate", warnings=warnings, notes=notes, files=files, data=data,
                  text="\n".join(lines))


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


# Report list -> what the command says about the files in it, after the run and in a dry run;
# conflicts are errors instead.
_REPORT_WARNINGS = {
    "merged": (
        "the model's changes were merged into your edits; a clean merge can still break, so run your tests",
        "the model's changes would be merged into your edits; a clean merge can still break, so run your tests",
    ),
    "kept_removed_from_model": (
        "the model no longer produces these files and they may hold your work, so they were kept",
        "the model no longer produces these files and they may hold your work, so they would be kept",
    ),
    "not_restored": (
        "you deleted these files and the model changed them; they stay deleted",
        "you deleted these files and the model changes them; they would stay deleted",
    ),
    "not_merged": (
        "these files clash with the model and cannot be merged as text; they were left as they were",
        "these files clash with the model and cannot be merged as text; they would be left as they are",
    ),
    "differs_from_model": (
        "these files differ from this model's output and count as your edits from now on "
        "(a long list usually means it is not the model they were generated from)",
        "these files differ from this model's output and would count as your edits from then on "
        "(a long list usually means it is not the model they were generated from)",
    ),
}
# Report list -> its label in the text report. These lists are the files the run changes.
_REPORT_LABELS = {"written": "written", "merged": "merged", "conflicted": "conflict", "deleted": "deleted"}
_CONFLICT_MARKERS = "('<<<<<<< your edit' ... '>>>>>>> generated from your model')"


def _regeneration_refusal(exc: regeneration.RegenerationError, spec: GeneratorSpec, output_dir: str,
                          option_assignments: list[str], given: _GeneratorInput) -> str:
    """What the command says of a folder ``regenerate()`` refuses: in its own words, with the next step."""
    if isinstance(exc, regeneration.UnrecordedOutputError):
        # 'besser generate' records its output: such a folder was generated by an older BESSER, or otherwise.
        adopt = _regenerate_command(spec, None, output_dir, option_assignments, given.use, adopt=True)
        return (f"{output_dir} holds generated files ({exc.named_files()}) but no {regeneration.RECORD_FILENAME}, "
                "so your edits cannot be told from generated code. Record it once with the model those files came "
                f"from: {adopt}, then regenerate with the changed model.")
    if isinstance(exc, regeneration.ConflictsPendingError):
        paths = ", ".join(Path(output_dir, rel).as_posix() for rel in exc.files)
        holds = "holds" if len(exc.files) == 1 else "hold"
        return (f"{paths} still {holds} the conflict markers of the last regenerate: resolve them "
                f"{_CONFLICT_MARKERS}, then run regenerate again.")
    if isinstance(exc, regeneration.AlreadyRecordedError):
        return (f"{output_dir} already has {regeneration.RECORD_FILENAME}: there is nothing to adopt; "
                "run it without --adopt")
    return str(exc)


def regenerate(generator_id: str, model_path: str, output_dir: str, option_assignments: list[str],
               use: Sequence[str] = (), adopt: bool = False, dry_run: bool = False) -> Result:
    """Bring ``output_dir`` up to the model with the generator, keeping the edits made in it."""
    data: dict[str, Any] = {"generator": generator_id, "model": model_path, "output_dir": output_dir,
                            "dry_run": dry_run}
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
        _require_output_dir("regenerate", output_dir, data)
        given = _generator_input("regenerate", spec, model_path, option_assignments, use, data)
    except _Stopped as stopped:
        return stopped.result
    warnings, notes = given.warnings, given.notes

    try:
        with stdout_to_stderr(), _without_regeneration_log():
            report = regenerate_generator(spec.id, given.models, output_dir, given.options, adopt=adopt,
                                          dry_run=dry_run)
    except regeneration.RegenerationError as exc:
        message = _regeneration_refusal(exc, spec, output_dir, option_assignments, given)
        return Result(command="regenerate", exit_code=EXIT_USAGE, errors=[message], warnings=warnings, notes=notes,
                      data=data, traceback=format_traceback(exc))
    except Exception as exc:
        return _generator_failure("regenerate", spec, exc, warnings, data, notes)

    # As ``files``: paths under the -o given, not relative to it.
    lists = {
        name: [Path(output_dir, path).as_posix() for path in paths]
        for name, paths in dataclasses.asdict(report).items()
    }
    data.update(lists, adopted=adopt)
    if dry_run:
        # Every file the run would change, with what it would do to it: the key every dry run has.
        data["would_write"] = dict(sorted((path, name) for name in _REPORT_LABELS for path in lists[name]))
    # A dry run reports what the run would do, in the run's own words.
    warnings.extend(
        f"dry run: {would}: {', '.join(lists[name])}" if dry_run else f"{done}: {', '.join(lists[name])}"
        for name, (done, would) in _REPORT_WARNINGS.items() if lists[name]
    )
    if dry_run:
        errors = [f"{path}: your edit and the model change the same lines: regenerate would write conflict "
                  "markers into it" for path in lists["conflicted"]]
    else:
        errors = [
            f"{path}: your edit and the model changed the same lines. Resolve the conflict markers "
            f"{_CONFLICT_MARKERS}, then run regenerate again"
            for path in lists["conflicted"]
        ]
    files = [] if dry_run else sorted(lists["written"] + lists["merged"] + lists["conflicted"])
    return Result(
        command="regenerate", exit_code=EXIT_FAILED if errors else EXIT_OK, errors=errors, warnings=warnings,
        notes=notes, files=files, data=data,
        text="\n".join(_regeneration_text(spec, model_path, output_dir, lists, adopt, dry_run)),
    )


def _regeneration_text(spec: GeneratorSpec, model_path: str, output_dir: str, lists: dict[str, list[str]],
                       adopt: bool, dry_run: bool) -> list[str]:
    """The report of a regeneration, or of what it would do."""
    changed = [(label, path) for name, label in _REPORT_LABELS.items() for path in lists[name]]
    counts = ", ".join(f"{len(lists[name])} {label}" for name, label in _REPORT_LABELS.items() if lists[name])
    if adopt and dry_run:
        lines = [_DRY_RUN, f"Adopting would record {output_dir} as the '{spec.id}' generator's output for "
                           f"{model_path}; no file would change."]
    elif adopt:
        lines = [f"Recorded {output_dir} as the '{spec.id}' generator's output for {model_path}; no file was "
                 "changed. 'besser regenerate' now keeps the edits made in it."]
    elif changed and dry_run:
        lines = [_DRY_RUN, f"Regenerating {output_dir} with the '{spec.id}' generator would change "
                           f"{plural(len(changed), 'file')} ({counts}):"]
    elif changed:
        lines = [f"Regenerated {output_dir} with the '{spec.id}' generator ({counts}):"]
    elif dry_run:
        lines = [_DRY_RUN, f"{output_dir} is up to date with {model_path}: no file would change."]
    else:
        lines = [f"{output_dir} is up to date with {model_path}: no file changed."]
    if changed and not adopt:
        width = max(len(label) for label, _ in changed)
        lines.extend(f"  {label.ljust(width)}  {path}" for label, path in changed)
    return lines


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
    data: dict[str, Any] = {"input": input_path, "output": output_path, "dry_run": dry_run}
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
    problem = _output_file_problem(target)
    if problem:
        return failure("convert", problem, data=data)

    try:
        loaded = load_models(input_path)
        source = loaded.path
        if loaded.suffix != expected_source:
            held = "a B-UML script" if loaded.suffix == ".py" else "editor JSON"
            return failure("convert", wrong_direction.format(held=f"{held} on standard input"), data=data)
        if loaded.errors:
            return Result(command="convert", exit_code=EXIT_FAILED, errors=loaded.errors, data=data)
        if dry_run:
            status, warnings = _convert_dry_run(loaded, target, write)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with stdout_to_stderr():
                warnings = write(loaded, target)
    except ModelLoadError as exc:
        return failure("convert", str(exc), data=data, cause=exc)
    except Exception as exc:
        return failure("convert", f"cannot convert {source}: {type(exc).__name__}: {exc}", EXIT_FAILED, data, exc)

    counts = plural(len(loaded.models), "model")
    if dry_run:
        data["would_write"] = {target.as_posix(): status}
        return Result(command="convert", warnings=warnings, data=data,
                      text=f"{_DRY_RUN}\nConverting {source} ({counts}) {_CONVERT_PREVIEW[status].format(target)}")
    return Result(
        command="convert", warnings=warnings, files=[target.as_posix()], data=data,
        text=f"Converted {source} ({counts}) to {target}",
    )


def _output_file_problem(target: Path) -> str | None:
    """Why ``-o`` cannot be the converted file, or ``None`` when it can."""
    if target.is_dir():
        return f"{target.as_posix()} is a directory: -o takes a file"
    blocking = _file_in_the_way(target)
    if blocking is not None:
        return f"{target.as_posix()} cannot be created: {blocking.as_posix()} is a file"
    return None


# A dry run's status of the target -> what converting would do to it.
_CONVERT_PREVIEW = {
    regeneration.NEW: "would create {}",
    regeneration.CHANGED: "would overwrite {}",
    regeneration.UNCHANGED: "would leave {} as it is: it holds the same",
}


def _convert_dry_run(loaded: LoadedModels, target: Path, write: Callable[[LoadedModels, Path], list[str]],
                     ) -> tuple[str, list[str]]:
    """Convert into a scratch file, so the conversion's failures and warnings are those of a real run.

    Returns:
        What the run would do to ``target`` (new, changed or unchanged), and the warnings.
    """
    from besser.utilities.web_modeling_editor.backend.constants.constants import TEMP_DIR_PREFIX

    with tempfile.TemporaryDirectory(prefix=TEMP_DIR_PREFIX) as scratch:
        converted = Path(scratch, target.name)
        with stdout_to_stderr():
            warnings = write(loaded, converted)
        content = converted.read_bytes()
    if not target.exists():
        return regeneration.NEW, warnings
    unchanged = target.is_file() and target.read_bytes() == content
    return (regeneration.UNCHANGED if unchanged else regeneration.CHANGED), warnings
