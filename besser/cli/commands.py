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
from typing import Any, Callable, Iterator, Mapping, Sequence, TypeVar

from besser.cli import loading
from besser.cli.discovery import SKIP_NO_MODEL, SKIP_PY, ModelFiles, SkippedFile, find_model_files
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
        if not report["validated"] and not (report["errors"] or report["warnings"]):
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


def _load_found_python(path: str) -> LoadedModels | None:
    """The models of a ``.py`` file found in a directory (with ``--exec-py``), or ``None`` when it holds none.

    A directory holds Python files that are not models (helpers, generated
    code): unlike a file named on the command line, one without a model is
    skipped rather than failing the run.
    """
    loaded = loading._load_python(Path(path), loading._execute(Path(path)))
    return loaded if loaded.models else None


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


def _nothing_to_validate(model_paths: Sequence[str], found: ModelFiles) -> Result:
    message = f"nothing to validate: no model file found in {', '.join(model_paths)}"
    skipped_py = sum(skipped.reason == SKIP_PY for skipped in found.skipped)
    if skipped_py:
        message += f" ({_plural(skipped_py, '.py file')} skipped; pass --exec-py to load them)"
    skipped = [{"file": skipped.path, "reason": skipped.reason} for skipped in found.skipped]
    return failure("validate", message, data={"model_files": [], "models": [], "results": [], "skipped": skipped})


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
        try:
            if model_file.explicit or not model_file.path.lower().endswith(".py"):
                loaded = load_models(model_file.path)
            elif (loaded := _load_found_python(model_file.path)) is None:
                skipped.append(SkippedFile(model_file.path, SKIP_NO_MODEL))
                continue
        except ModelLoadError as exc:
            checks.append(_FileCheck(file=model_file.path, exit_code=EXIT_USAGE, errors=[str(exc)],
                                     load_error=str(exc)))
            traceback += format_traceback(exc)
            continue
        check, outcome = _check_file(model_file.path, loaded, strict, checker, exec_py)
        checks.append(check)
        if outcome is not None:
            constraints += outcome.constraints
            evaluations += outcome.evaluations
            unevaluated += outcome.unevaluated
        lines.extend(_validation_text(str(loaded.path), check.models, loaded.errors))
    if not checks:
        return _nothing_to_validate(model_paths, dataclasses.replace(found, skipped=skipped))

    # With one file a message starts with the model's name; with several, or a directory, with the file's too.
    several = found.searched_directory or len(checks) > 1
    load_errors = [check.load_error for check in checks if check.load_error]
    errors = [f"{check.file}: {error}" if several else error
              for check in checks if not check.load_error for error in check.errors]
    file_warnings = [f"{check.file}: {warning}" if several else warning
                     for check in checks for warning in check.warnings]
    # The report lists the files' messages; the run's own (OCL not installed) go to standard error.
    shown_in_text = frozenset(errors + file_warnings)
    warnings = file_warnings + warnings

    skipped_py = sum(entry.reason == SKIP_PY for entry in skipped)
    if skipped_py:
        notes.append(f"{_plural(skipped_py, '.py file')} found in a directory skipped: loading a .py model "
                     "executes it; pass --exec-py to load them")
    if checker is not None and len(load_errors) < len(checks):
        notes.append(
            f"OCL: {_plural(constraints, 'constraint')} parsed, invariants evaluated on "
            f"{_plural(evaluations, 'object model')}" if constraints else "OCL: the models have no OCL constraints"
        )
    if unevaluated:
        notes.append(
            f"OCL: invariants not evaluated on {_plural(unevaluated, 'object model')} of editor JSON: the OCL "
            "evaluator runs Python built from the models' values; pass --exec-py to evaluate them"
        )

    exit_code = max(check.exit_code for check in checks)
    if strict and warnings:
        exit_code = max(exit_code, EXIT_FAILED)
    counts = f"{_plural(len(errors) + len(load_errors), 'error')}, {_plural(len(warnings), 'warning')}"
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
        # No report for a lone file that does not load: its error says it all.
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
        f"{loaded.path} holds {_plural(len(loaded.of_kind(kind)), kind + ' model')} and names no current one: "
        f"{reason}; generators need --use NAME"
        for kind, reason in loaded.current_unresolved.items()
    ]
    current = {id(marked.model) for marked in loaded.current.values()}
    models = []
    lines = [f"{loaded.path}: {_plural(len(loaded.models), 'model')}"]
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
            if not named and current is not None:
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


def _generator_input(
    command: str, spec: GeneratorSpec, model_path: str, option_assignments: list[str], use: Sequence[str],
    data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str]]:
    """The models the generator reads, by kind, its options, the validation warnings and notes on the models.

    Adds ``used`` to ``data``: kind -> the name of the model read.

    Raises:
        _Stopped: The file does not load, an option does not parse, or a model has validation errors.
    """
    try:
        options = parse_options(spec, option_assignments)
        models, notes = _models_for(spec, load_models(model_path), use)
    except (GeneratorInputError, ModelLoadError) as exc:
        raise _Stopped(failure(command, str(exc), data=data, cause=exc)) from exc
    data["used"] = {kind: model.name for kind, model in models.items()}

    reports = [_validate_model(model) for model in models.values()]
    warnings = [f"{report['name']}: {warning}" for report in reports for warning in report["warnings"]]
    errors = [f"{report['name']}: {error}" for report in reports for error in report["errors"]]
    if errors:
        errors.append("nothing was generated: the model has validation errors")
        raise _Stopped(Result(command=command, exit_code=EXIT_FAILED, errors=errors, warnings=warnings,
                              notes=notes, data=data))
    return {kind: model.model for kind, model in models.items()}, options, warnings, notes


def _generator_failure(command: str, spec: GeneratorSpec, exc: Exception, warnings: list[str],
                       data: dict[str, Any], notes: list[str]) -> Result:
    """What a raising generator is reported as: its input refused (2), or a failure (1)."""
    if isinstance(exc, GeneratorInputError):
        return Result(command=command, exit_code=EXIT_USAGE, errors=[str(exc)], warnings=warnings, notes=notes,
                      data=data)
    message = f"the '{spec.id}' generator failed: {type(exc).__name__}: {exc}"
    return Result(command=command, exit_code=EXIT_FAILED, errors=[message], warnings=warnings, notes=notes,
                  data=data, traceback=format_traceback(exc))


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


def _generate_dry_run(spec: GeneratorSpec, models: dict[str, Any], output_dir: str, options: dict[str, Any],
                      warnings: list[str], notes: list[str], data: dict[str, Any]) -> Result:
    """What ``generate`` would write, by file: new, changed or unchanged. Nothing is written."""
    try:
        with stdout_to_stderr():
            plan, _ = _with_record(spec, output_dir, warnings, lambda record: preview_generator(
                spec.id, models, output_dir, options, record=record))
    except regeneration.RegenerationError as exc:
        return Result(command="generate", exit_code=EXIT_USAGE, errors=[str(exc)], warnings=warnings, notes=notes,
                      data=data, traceback=format_traceback(exc))
    except Exception as exc:
        return _generator_failure("generate", spec, exc, warnings, data, notes)

    data["would_write"] = {Path(output_dir, rel).as_posix(): status for rel, status in plan.items()}
    if not spec.supports_regenerate and regeneration.CHANGED in plan.values():
        notes.append(f"the '{spec.id}' generator's output differs from run to run for one model, "
                     "so a file can read as changed when the model is not")
    counts = ", ".join(f"{list(plan.values()).count(status)} {status}" for status in _PREVIEW_STATUSES
                       if status in plan.values())
    lines = [_DRY_RUN, f"The '{spec.id}' generator would write {_plural(len(plan), 'file')} in {output_dir}"
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
        # A dry run says what the run would do, and that includes this refusal.
        if not force:
            _require_no_output(spec, model_path, output_dir, data)
        models, options, warnings, notes = _generator_input("generate", spec, model_path, option_assignments, use, data)
    except _Stopped as stopped:
        return stopped.result
    if dry_run:
        return _generate_dry_run(spec, models, output_dir, options, warnings, notes, data)

    try:
        with stdout_to_stderr():
            written, data["recorded"] = _with_record(spec, output_dir, warnings, lambda record: run_generator(
                spec.id, models, output_dir, options, record=record))
    except regeneration.RegenerationError as exc:
        return Result(command="generate", exit_code=EXIT_USAGE, errors=[str(exc)], warnings=warnings, notes=notes,
                      data=data, traceback=format_traceback(exc))
    except Exception as exc:
        return _generator_failure("generate", spec, exc, warnings, data, notes)

    if data["recorded"]:
        notes.append(
            f"recorded the output in {Path(output_dir, regeneration.RECORD_FILENAME).as_posix()}: "
            f"'besser regenerate {spec.id} {model_path} -o {output_dir}' brings it up to a changed model "
            "and keeps your edits"
        )
    files = [Path(output_dir, path).as_posix() for path in written]
    lines = [f"Generated {_plural(len(files), 'file')} in {output_dir} with the '{spec.id}' generator:"]
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
        models, options, warnings, notes = _generator_input("regenerate", spec, model_path, option_assignments, use, data)
    except _Stopped as stopped:
        return stopped.result

    try:
        with stdout_to_stderr(), _without_regeneration_log():
            report = regenerate_generator(spec.id, models, output_dir, options, adopt=adopt, dry_run=dry_run)
    except regeneration.RegenerationError as exc:
        message = str(exc)
        if isinstance(exc, regeneration.UnrecordedOutputError):
            # 'besser generate' records its output: such a folder was generated by an older BESSER, or otherwise.
            message += (
                f" Record it once with the model those files came from: 'besser regenerate {spec.id} <that model> "
                f"-o {output_dir} --adopt', then regenerate with the changed model."
            )
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
    # A dry run reports what the run would do, in the run's own words.
    prefix = "dry run: " if dry_run else ""
    warnings.extend(
        f"{prefix}{reason}: {', '.join(lists[name])}" for name, reason in _REPORT_WARNINGS.items() if lists[name]
    )
    if dry_run:
        errors = [f"{path}: your edit and the model change the same lines: regenerate would write conflict "
                  "markers into it" for path in lists["conflicted"]]
    else:
        errors = [
            f"{path}: your edit and the model changed the same lines. Resolve the conflict markers "
            "('<<<<<<< your edit' ... '>>>>>>> generated from your model'), then run regenerate again"
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
                           f"{_plural(len(changed), 'file')} ({counts}):"]
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

    counts = _plural(len(loaded.models), "model")
    if dry_run:
        data["would_write"] = {target.as_posix(): status}
        return Result(command="convert", warnings=warnings, data=data,
                      text=f"{_DRY_RUN}\nConverting {source} ({counts}) {_CONVERT_PREVIEW[status].format(target)}")
    return Result(
        command="convert", warnings=warnings, files=[target.as_posix()], data=data,
        text=f"Converted {source} ({counts}) to {target}",
    )


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
