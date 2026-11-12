"""The ``besser`` commands: each takes parsed arguments and returns a ``Result``."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, TypeVar

from besser.cli import loading
from besser.cli import schema as schema_module
from besser.cli.agent_files import GUIDE, GeneratedDir, sync_ignore, write_agent_files
from besser.cli.config import (
    CONFIG_EXAMPLE,
    CONFIG_FILENAME,
    ConfigError,
    OutputConfig,
    load_config,
    load_default_model,
    load_test_command,
)
from besser.cli.importing import IMPORT_FORMATS, ImportFormat, format_of, import_format, import_models
from besser.cli.discovery import SKIP_NO_MODEL, SKIP_PY, ModelFiles, SkippedFile, find_model_files
from besser.cli.loading import (
    STDIN,
    LoadedModel,
    LoadedModels,
    MissingDependencyError,
    ModelLoadError,
    NoModelError,
    load_models,
    load_python_text,
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
from besser.cli.model_edit import EditError, Edited, Operation, pick_domain_model, run_operation
from besser.cli.ocl_check import OclOutcome, OclUnavailable, check_ocl, load_checker, parse_problems
from besser.cli.starters import DEFAULT_KIND, STARTERS, starter_source
from besser.cli.summary import class_lines, describe_class, describe_model, summary_lines
from besser.generators import regeneration
from besser.generators.registry import GENERATORS, GeneratorSpec, get_generator
from besser.generators.runner import (
    GeneratorInputError,
    preview_generator,
    regenerate_generator,
    run_generator,
)
from besser.utilities.provenance import get_besser_version
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
    if loaded.kind == "domain":
        # Generators read the constraints, so one that names nothing of the model breaks the output.
        ocl_errors, ocl_warnings = parse_problems(loaded.model)
        report["errors"].extend(ocl_errors)
        report["warnings"].extend(ocl_warnings)
    return report


def _validation_text(path: str, reports: list[dict[str, Any]], project_errors: list[str],
                     diagnostics: Sequence[Mapping[str, Any]] = ()) -> list[str]:
    """The report of one file; a message whose line is known (see ``diagnostics``) ends with it."""
    lines_of = {(entry.get("model"), entry["message"]): entry["line"] for entry in diagnostics if "line" in entry}

    def located(model: str, message: str) -> str:
        line = lines_of.get((model, message))
        return f"{message} ({path}, line {line})" if line else message

    lines = [f"{path}: {plural(len(reports), 'model')}"]
    for report in reports:
        if not report["validated"] and not (report["errors"] or report["warnings"]):
            status = "no validation rules"
        elif report["errors"] or report["warnings"]:
            status = f"{plural(len(report['errors']), 'error')}, {plural(len(report['warnings']), 'warning')}"
        else:
            status = "ok"
        lines.append(f"  {report['name']} ({report['type']}): {status}")
        lines.extend(f"    error: {located(report['name'], error)}" for error in report["errors"])
        lines.extend(f"    warning: {located(report['name'], warning)}" for warning in report["warnings"])
    lines.extend(f"  error: {error}" for error in project_errors)
    return lines


_QUOTED_NAME = re.compile(r"'([^']+)'")
_LINE_IN_MESSAGE = re.compile(r": line (\d+): ")


def _definition_line(source: str, message: str) -> int | None:
    """The line of a B-UML script where an element the message names is defined (``name="X"``).

    The first quoted name defined on exactly one line counts: "Attribute 'title' of class 'Book'"
    gives the attribute's line when one attribute is called title, else the class's.
    """
    lines = source.splitlines()
    for name in _QUOTED_NAME.findall(message):
        definition = re.compile(r"""\bname\s*=\s*(['"])""" + re.escape(name) + r"\1")
        found = [number for number, line in enumerate(lines, start=1) if definition.search(line)]
        if len(found) == 1:
            return found[0]
    return None


def _diagnostic(severity: str, file: str, message: str, model: str | None = None,
                source: str | None = None) -> dict[str, Any]:
    """One error or warning with where it is: the file, the model, and the line when it is known."""
    entry: dict[str, Any] = {"severity": severity, "file": file}
    if model is not None:
        entry["model"] = model
    entry["message"] = message
    in_message = _LINE_IN_MESSAGE.search(message)
    line = int(in_message.group(1)) if in_message else _definition_line(source, message) if source else None
    if line is not None:
        entry["line"] = line
    return entry


def _script_source(loaded: LoadedModels) -> str | None:
    """The text of a ``.py`` model file, to find lines in; ``None`` for JSON and standard input."""
    if loaded.suffix != ".py" or not loaded.path.is_file():
        return None
    try:
        return loaded.path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


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
    # Every error and warning of the file, with where it is (see ``_diagnostic``).
    diagnostics: list[dict[str, Any]] = dataclasses.field(default_factory=list)

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
    source = _script_source(loaded)
    for report in check.models:
        for severity in ("error", "warning"):
            check.diagnostics.extend(_diagnostic(severity, path, message, report["name"], source)
                                     for message in report[f"{severity}s"])
    check.diagnostics.extend(_diagnostic("error", path, message) for message in loaded.errors)
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
    return failure("validate", message, data={"model_files": [], "models": [], "results": [], "skipped": skipped,
                                               "diagnostics": []})


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
                                     load_error=str(load_error),
                                     diagnostics=[_diagnostic("error", model_file.path, str(load_error))]))
            lines.append(_load_error_text(model_file.path, str(load_error)))
            traceback += format_traceback(load_error)
            continue
        check, outcome = _check_file(model_file.path, loaded, strict, checker, exec_py)
        checks.append(check)
        if outcome is not None:
            constraints += outcome.constraints
            evaluations += outcome.evaluations
            unevaluated += outcome.unevaluated
        lines.extend(_validation_text(str(loaded.path), check.models, loaded.errors, check.diagnostics))
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
            "diagnostics": [diagnostic for check in checks for diagnostic in check.diagnostics],
        },
        text="\n".join([*lines, *skipped_lines, verdict]) if lines or skipped_lines else "",
        shown_in_text=shown_in_text,
        traceback=traceback,
        # A failure needs its report; a success is its verdict, with the warnings printed on their own.
        brief=verdict if exit_code == EXIT_OK else "",
    )


# ── info ────────────────────────────────────────────────────────────────────


def info(model_path: str, class_name: str | None = None) -> Result:
    """Summarise the models of the file and name the generators that can run on it; or, with
    ``class_name``, describe that class of its domain models."""
    data: dict[str, Any] = {"model": model_path}
    try:
        loaded = load_models(model_path)
    except ModelLoadError as exc:
        return failure("info", str(exc), data=data, cause=exc)
    if class_name is not None:
        return _class_info(loaded, class_name, data)

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
    generators_line = (f"Generators that read {'these models' if len(models) > 1 else 'this model'}: "
                       f"{', '.join(runnable) or 'none'}")
    return Result(
        command="info", warnings=warnings, notes=notes, data={**data, "models": models, "generators": runnable},
        # A coding agent reads the model, not the list of generators besser.toml already chose from.
        text="\n".join([*lines, generators_line]), brief="\n".join(lines), text_is_result=True,
    )


def _class_info(loaded: LoadedModels, class_name: str, data: dict[str, Any]) -> Result:
    """``info --class``: one class of the file's domain models, in full."""
    found = [(model, model.model.get_class_by_name(class_name)) for model in loaded.of_kind("domain")]
    found = [(model, cls) for model, cls in found if cls is not None]
    if not found:
        known = sorted({cls.name for model in loaded.of_kind("domain") for cls in model.model.get_classes()})
        return failure("info", f"{loaded.path} has no class named '{class_name}' (it has: {', '.join(known) or 'none'})",
                       EXIT_FAILED, data)
    entries = [{"model": model.name, **describe_class(model.model, cls)} for model, cls in found]
    lines = []
    for entry in entries:
        if len(entries) > 1:
            lines.append(f"{entry['model']}:")
        lines.extend(class_lines(entry))
    return Result(command="info", data={**data, "classes": entries}, text="\n".join(lines), text_is_result=True)


# ── new ─────────────────────────────────────────────────────────────────────


def _starter_list() -> Result:
    """The kinds ``new`` writes a starter for, with what each models and the generators that read it."""
    kinds = []
    for kind, starter in STARTERS.items():
        generators = [spec.id for spec in GENERATORS if kind in spec.models]
        kinds.append({"kind": kind, "summary": starter.summary, "generators": generators,
                      "editor_diagram": schema_module.editor_diagram_type(kind)})
    width = max(len(kind) for kind in STARTERS)
    lines = ["Kinds (besser new FILE --kind KIND; FILE is a .py file, or a .json file for the editor):"]
    lines.extend(f"  {entry['kind'].ljust(width)}  {entry['summary']}" for entry in kinds)
    return Result(command="new", data={"kinds": kinds}, text="\n".join(lines), text_is_result=True)


def _write_starter_json(kind: str, source: str, target: Path) -> list[str]:
    """Write the starter as an editor project: the starter is loaded and converted, as ``convert`` does."""
    from besser.utilities.web_modeling_editor.backend.constants.constants import TEMP_DIR_PREFIX

    with tempfile.TemporaryDirectory(prefix=TEMP_DIR_PREFIX) as scratch:
        script = Path(scratch, f"{target.stem}.py")
        script.write_text(source, encoding="utf-8")
        loaded = load_models(script)
        with stdout_to_stderr():
            return _write_editor_json(loaded, target)


def _generated_dirs(config_path: Path) -> list[GeneratedDir]:
    """The output folders the configuration file at ``config_path`` lists, relative to its folder; none
    when there is no such file or it does not load (``build`` reports that)."""
    try:
        outputs = load_config(str(config_path))
    except ConfigError:
        return []
    dirs = []
    for output in outputs:
        try:
            path = Path(output.output_dir).relative_to(config_path.parent).as_posix()
        except ValueError:  # outside the project: nothing of the project to keep out of searches
            continue
        try:
            extension_file = get_generator(output.generator).load_class().extension_file
        except (KeyError, ImportError):  # an unknown generator: build reports it
            extension_file = None
        dirs.append(GeneratedDir(path, extension_file))
    return dirs


def _write_agent_files(folder: Path, warnings: list[str], notes: list[str]) -> list[str]:
    """Write the agent guide and ignore rules in ``folder``; a file that cannot be written is a warning.

    Only in a project: the guide sends the agent to ``besser apply`` and ``besser model``, which read
    ``besser.toml``, so a folder without one gets no guide (and a note saying so).
    """
    if not (folder / CONFIG_FILENAME).is_file():
        shown = folder.as_posix()
        notes.append(f"no agent guide: {shown} has no {CONFIG_FILENAME} for it to describe ('besser init' sets one up)")
        return []
    try:
        return write_agent_files(folder, _generated_dirs(folder / CONFIG_FILENAME))
    except (OSError, UnicodeDecodeError) as exc:
        warnings.append(f"the agent guide was not written in {folder}: {exc}")
        return []


def new(path: str | None, kind: str = DEFAULT_KIND, force: bool = False, list_kinds: bool = False,
        agent_files: bool = True) -> Result:
    """Write the starter model of ``kind`` to ``path``: a B-UML script, or an editor project for ``.json``.

    With ``agent_files``, also write the agent guide (``AGENTS.md``, imported by ``CLAUDE.md``) and the
    ``.ignore`` rules that keep generated code out of an agent's searches, beside the model, when that
    folder has a ``besser.toml``.
    """
    if list_kinds:
        return _starter_list()
    data: dict[str, Any] = {"model": path, "kind": kind}
    if path is None:
        return failure("new", "name the file to write (besser new model.py), or pass --list for the kinds",
                       data=data)
    if kind not in STARTERS:
        return failure("new", f"unknown kind '{kind}'; kinds: {', '.join(STARTERS)}", data=data)
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix not in (".py", ".json"):
        return failure("new", f"a model file is a B-UML .py file or an editor .json project; got {target.name}",
                       data=data)
    if suffix == ".json" and kind in _KINDS_WITHOUT_DIAGRAM:
        return failure("new", f"the editor has no diagram for a {kind} model: write a .py file", data=data)
    if target.exists() and not force:
        return failure("new", f"{target} exists: pass --force to overwrite it", data=data)
    warnings: list[str] = []
    notes: list[str] = []
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".py":
            target.write_text(starter_source(kind, target.name), encoding="utf-8")
        else:
            warnings = _write_starter_json(kind, starter_source(kind, target.with_suffix(".py").name), target)
    except OSError as exc:
        return failure("new", f"cannot write {target}: {exc}", EXIT_FAILED, data, cause=exc)
    except ModelLoadError as exc:  # editor JSON needs the backend's packages
        return failure("new", str(exc), data=data, cause=exc)
    guide_files = _write_agent_files(target.parent, warnings, notes) if agent_files else []
    shown = shlex.quote(str(target))
    brief = f"Wrote a starter {kind} model to {target}"
    if guide_files:
        brief += f", and the agent guide: {', '.join(guide_files)}"
    lines = [
        f"{brief}. Next:",
        f"  besser validate {shown}",
        f"  besser info {shown}",
    ]
    generator = STARTERS[kind].generator
    if generator:
        lines.append(f"  besser generate {generator} {shown} -o out")
    lines.append(f"  besser schema {kind}          the classes to write it with")
    return Result(command="new", warnings=warnings, notes=notes, files=[target.as_posix(), *guide_files],
                  data=data, text="\n".join(lines), brief=f"{brief}.")


# The output 'besser init' configures for a domain model: an API a coding agent can test.
_INIT_GENERATOR = {DEFAULT_KIND: "backend"}
_INIT_MODEL = "model.py"


def _init_config(kind: str) -> str | None:
    """The ``besser.toml`` of a new project whose model is of ``kind``; None when no generator reads it."""
    generator = _INIT_GENERATOR.get(kind) or STARTERS[kind].generator
    if generator is None:
        return None
    return (f'model = "{_INIT_MODEL}"\n'
            '# test = "python -m pytest -q"   # what \'besser apply\' runs after the build\n\n'
            f"[output.{generator}]\n"
            f'generator = "{generator}"\n'
            f'dir = "{generator}"\n')


def init(folder: str = ".", kind: str = DEFAULT_KIND) -> Result:
    """Set ``folder`` up as a BESSER project: a starter model, a ``besser.toml`` that builds it, and the
    agent guide and ignore rules. A model or configuration file that exists is kept. A kind no generator
    reads gets no ``besser.toml``, and so no guide."""
    root = Path(folder)
    data: dict[str, Any] = {"folder": folder, "kind": kind}
    if kind not in STARTERS:
        return failure("init", f"unknown kind '{kind}'; kinds: {', '.join(STARTERS)}", data=data)
    files: list[str] = []
    notes: list[str] = []
    warnings: list[str] = []
    try:
        root.mkdir(parents=True, exist_ok=True)
        planned = {_INIT_MODEL: starter_source(kind, _INIT_MODEL), CONFIG_FILENAME: _init_config(kind)}
        for name, text in planned.items():
            target = root / name
            if text is None:
                notes.append(f"no {name}: no generator reads a {kind} model")
            elif target.exists():
                notes.append(f"kept {target.as_posix()}: it exists")
            else:
                target.write_text(text, encoding="utf-8")
                files.append(target.as_posix())
    except OSError as exc:
        return failure("init", f"cannot write in {root}: {exc}", EXIT_FAILED, data, cause=exc)
    files.extend(_write_agent_files(root, warnings, notes))
    brief = f"Set up {root.as_posix()}: wrote {', '.join(files) or 'nothing (all up to date)'}"
    if (root / CONFIG_FILENAME).is_file():
        text = f"{brief}.\nNext: edit {_INIT_MODEL}, then run 'besser apply' (validate, build, test)."
    else:
        text = f"{brief}.\nNext: edit {_INIT_MODEL}, then run 'besser validate {_INIT_MODEL}'."
    return Result(command="init", warnings=warnings, notes=notes, files=files, data=data, text=text,
                  brief=f"{brief}.")


# ── guide ───────────────────────────────────────────────────────────────────


def guide() -> Result:
    """The guide a coding agent needs to work in a BESSER project: what 'new' and 'init' write to AGENTS.md."""
    return Result(command="guide", data={"guide": GUIDE}, text=GUIDE.rstrip("\n"), text_is_result=True)


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

    if not written:
        # A generator that skips what it cannot generate logs why and returns: that is not a success.
        return Result(command="generate", exit_code=EXIT_FAILED, warnings=warnings, notes=notes, data=data,
                      errors=[f"the '{spec.id}' generator wrote no file; what it logged, on standard error, "
                              "says why"])
    if data["recorded"]:
        notes.append(_record_note(spec, model_path, output_dir, option_assignments, given, dry_run=False))
    files = [Path(output_dir, path).as_posix() for path in written]
    summary = f"Generated {plural(len(files), 'file')} in {output_dir} with the '{spec.id}' generator"
    lines = [f"{summary}:", *(f"  {path}" for path in files)]
    return Result(command="generate", warnings=warnings, notes=notes, files=files, data=data,
                  text="\n".join(lines), brief=f"{summary}.")


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
                          option_assignments: list[str], given: _GeneratorInput, rerun: str) -> str:
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
                f"{_CONFLICT_MARKERS}, then run {rerun} again.")
    if isinstance(exc, regeneration.AlreadyRecordedError):
        return (f"{output_dir} already has {regeneration.RECORD_FILENAME}: there is nothing to adopt; "
                "run it without --adopt")
    return str(exc)


def regenerate(generator_id: str, model_path: str, output_dir: str, option_assignments: list[str],
               use: Sequence[str] = (), adopt: bool = False, dry_run: bool = False,
               rerun: str = "regenerate") -> Result:
    """Bring ``output_dir`` up to the model with the generator, keeping the edits made in it.

    ``rerun`` is the command a conflict message says to run again: ``build`` and ``apply`` name themselves.
    """
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
    recorded_by, running = regeneration.recorded_version(output_dir), get_besser_version()
    if recorded_by is not None and recorded_by != running:
        notes.append(f"{output_dir} was generated with BESSER {recorded_by} and this is BESSER {running}: "
                     "a file can change because the generator did, not only the model")

    try:
        with stdout_to_stderr(), _without_regeneration_log():
            report = regenerate_generator(spec.id, given.models, output_dir, given.options, adopt=adopt,
                                          dry_run=dry_run)
    except regeneration.RegenerationError as exc:
        message = _regeneration_refusal(exc, spec, output_dir, option_assignments, given, rerun)
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
            f"{_CONFLICT_MARKERS}, then run {rerun} again"
            for path in lists["conflicted"]
        ]
    files = [] if dry_run else sorted(lists["written"] + lists["merged"] + lists["conflicted"])
    lines = _regeneration_text(spec, model_path, output_dir, lists, adopt, dry_run)
    return Result(
        command="regenerate", exit_code=EXIT_FAILED if errors else EXIT_OK, errors=errors, warnings=warnings,
        notes=notes, files=files, data=data, text="\n".join(lines),
        # The files merged or in conflict are named by the warnings and errors; a dry run is read for its list.
        brief="" if dry_run else lines[0].rstrip(":"),
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


def _write_rdf(loaded: LoadedModels, output_path: Path) -> list[str]:
    """Write the file's knowledge graph as RDF: Turtle, or RDF/XML for ``.owl`` and ``.rdf``."""
    from besser.BUML.notations.kg_rdf import serialize_knowledge_graph

    graphs = loaded.of_kind("kg")
    if len(graphs) != 1:
        held = ", ".join(model.label for model in loaded.models)
        raise _Stopped(failure(
            "convert", f"{output_path.name} is RDF, which is written from one knowledge graph, and {loaded.path} "
                       f"holds {len(graphs) or 'none'} (it holds: {held})",
        ))
    text = serialize_knowledge_graph(graphs[0].model, fmt=_RDF_SERIALIZATIONS[output_path.suffix.lower()])
    output_path.write_text(text, encoding="utf-8")
    return [f"{model.label} was left out: RDF is written from the knowledge graph"
            for model in loaded.models if model.kind != "kg"]


# Output suffix of an RDF file -> its rdflib serialization.
_RDF_SERIALIZATIONS = {".ttl": "turtle", ".owl": "xml", ".rdf": "xml"}
# Output suffix -> (the format written, the writer). A format is what ``LoadedModels.suffix`` names.
_WRITERS: dict[str, tuple[str, Callable[[LoadedModels, Path], list[str]]]] = {
    ".json": (".json", _write_editor_json),
    ".py": (".py", _write_buml),
    **{suffix: ("rdf", _write_rdf) for suffix in _RDF_SERIALIZATIONS},
}
_SAME_FORMAT = (
    "convert turns a B-UML .py file into an editor .json project, or a .json export into "
    "a .py file; got {held} -> {target}"
)


def _import_formats_text() -> str:
    return ", ".join(fmt.name for fmt in IMPORT_FORMATS)


def _source_format(inputs: Sequence[str], source_format: str | None) -> ImportFormat | None:
    """The import format to read ``inputs`` as; ``None`` for a model file (``.py``, ``.json``, standard input).

    Raises:
        ModelLoadError: The inputs are of no format ``convert`` reads, or of several.
    """
    if source_format is not None:
        return import_format(source_format)
    formats = []
    for value in inputs:
        path = Path(value)
        if value == STDIN or path.suffix.lower() in (".py", ".json"):
            formats.append(None)
        elif path.is_dir():
            raise ModelLoadError(f"{value} is a directory: name what it holds with --from "
                                 f"({', '.join(fmt.name for fmt in IMPORT_FORMATS if fmt.directory)})")
        elif not path.exists():
            raise ModelLoadError(f"{value}: no such file")
        else:
            fmt = format_of(path)
            if fmt is None:
                known = ", ".join(suffix for fmt in IMPORT_FORMATS for suffix in fmt.suffixes)
                raise ModelLoadError(
                    f"{value}: convert reads a B-UML .py file, an editor .json export, or a file to import "
                    f"({known}); name another file's format with --from ({_import_formats_text()})"
                )
            formats.append(fmt)
    if len(inputs) > 1 and (formats[0] is None or not formats[0].several or len(set(formats)) > 1):
        several = ", ".join(fmt.name for fmt in IMPORT_FORMATS if fmt.several)
        raise ModelLoadError(f"convert reads one input; only these formats take several files: {several}")
    return formats[0]


def _derive_class_diagrams(loaded: LoadedModels) -> tuple[LoadedModels, list[str]]:
    """``loaded`` with each knowledge graph replaced by the class diagram derived from it, and the warnings.

    Raises:
        _Stopped: The file holds no knowledge graph.
    """
    from besser.BUML.notations.kg_to_buml import kg_to_class_diagram

    if not loaded.of_kind("kg"):
        held = ", ".join(model.label for model in loaded.models)
        raise _Stopped(failure(
            "convert", f"--class-diagram derives a class diagram from a knowledge graph and {loaded.path} "
                       f"holds none (it holds: {held})",
        ))
    models, warnings = [], []
    for model in loaded.models:
        if model.kind != "kg":
            models.append(model)
            continue
        result = kg_to_class_diagram(model.model)
        warnings.extend(f"{model.name}: {warning.message}" for warning in result.warnings)
        models.append(LoadedModel(name=result.domain_model.name, kind="domain", model=result.domain_model))
    # The file's own Project holds the graphs: the models are wrapped in a new one.
    return dataclasses.replace(loaded, models=models, project=None), warnings


def convert(inputs: str | Sequence[str], output_path: str, dry_run: bool = False,
            source_format: str | None = None, class_diagram: bool = False) -> Result:
    """Convert a model file to another format, or import a file of another notation as a model.

    ``source_format`` names the import format of ``inputs`` (see ``besser.cli.importing``) when
    their suffix does not; ``class_diagram`` writes the class diagram derived from each
    knowledge graph instead of the graph.
    """
    inputs = [inputs] if isinstance(inputs, str) else list(inputs)
    data: dict[str, Any] = {"input": inputs[0], "inputs": inputs, "output": output_path, "dry_run": dry_run}
    source, target = Path(inputs[0]), Path(output_path)
    written_format, write = _WRITERS.get(target.suffix.lower(), (None, None))
    if write is None:
        return failure(
            "convert", f"convert writes a B-UML .py file, an editor .json project or, from a knowledge graph, "
                       f"RDF ({', '.join(_RDF_SERIALIZATIONS)}); got {target.name}", data=data,
        )
    problem = _output_file_problem(target)
    if problem:
        return failure("convert", problem, data=data)

    try:
        fmt = _source_format(inputs, source_format)
        # Refused before the file is loaded (a .py model is executed): there is nothing to convert.
        if fmt is None and not class_diagram and source.suffix.lower() == written_format:
            return failure("convert", _SAME_FORMAT.format(held=source.name, target=target.name), data=data)
        if fmt is None:
            loaded, warnings = load_models(inputs[0]), []
        else:
            loaded, warnings = import_models(fmt, inputs)
        source = loaded.path
        if class_diagram:
            loaded, derived = _derive_class_diagrams(loaded)
            warnings.extend(derived)
        elif fmt is None and loaded.suffix == written_format:  # standard input: known once it is loaded
            held = "a B-UML script" if loaded.suffix == ".py" else "editor JSON"
            return failure("convert", _SAME_FORMAT.format(held=f"{held} on standard input", target=target.name),
                           data=data)
        if loaded.errors:
            return Result(command="convert", exit_code=EXIT_FAILED, errors=loaded.errors, data=data)
        if dry_run:
            status, written_warnings = _convert_dry_run(loaded, target, write)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with stdout_to_stderr():
                written_warnings = write(loaded, target)
        warnings.extend(written_warnings)
    except _Stopped as stopped:
        return dataclasses.replace(stopped.result, data=data)
    except ModelLoadError as exc:
        return failure("convert", str(exc), data=data, cause=exc)
    except Exception as exc:
        return failure("convert", f"cannot convert {source}: {type(exc).__name__}: {exc}", EXIT_FAILED, data, exc)

    data["models"] = [{"name": model.name, "kind": model.kind, "type": type(model.model).__name__}
                      for model in loaded.models]
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


# ── build ───────────────────────────────────────────────────────────────────


def option_assignments(options: Mapping[str, Any] | Sequence[str] | None) -> list[str]:
    """Typed options ({name: value}, as a configuration file or a tool call gives them) as the
    ``name=value`` arguments ``parse_options`` reads; a list of such arguments is returned as it is."""
    if options is None:
        return []
    if not isinstance(options, Mapping):
        return list(options)

    def text(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, list):
            return ",".join(text(item) for item in value)
        return str(value)

    return [f"{name}={text(value)}" for name, value in options.items()]


def _build_output(output: OutputConfig, dry_run: bool, rerun: str) -> Result:
    """Generate one output of the configuration: into an empty directory, or over it when the
    output says ``overwrite``; otherwise bring the directory up to the model, keeping the edits made in it."""
    assignments = option_assignments(output.options)
    folder = Path(output.output_dir)
    empty = not os.path.lexists(folder) or (folder.is_dir() and not any(folder.iterdir()))
    try:
        regenerable = get_generator(output.generator).supports_regenerate
    except KeyError:  # generate reports it, naming the generators there are
        regenerable = False
    if empty or output.overwrite or not regenerable:
        result = generate(output.generator, output.model, output.output_dir, assignments, output.use,
                          force=output.overwrite, dry_run=dry_run)
        if result.exit_code == EXIT_USAGE and not empty and not output.overwrite:
            result.errors.append(f"to overwrite it on every build, set 'overwrite = true' in [output.{output.name}]")
        return result
    return regenerate(output.generator, output.model, output.output_dir, assignments, output.use, dry_run=dry_run,
                      rerun=rerun)


def build(names: Sequence[str] = (), config_path: str = CONFIG_FILENAME, dry_run: bool = False,
          rerun: str = "besser build") -> Result:
    """Generate the outputs the configuration file lists, or those of ``names``.

    Each is generated as ``generate`` does the first time and as ``regenerate`` does afterwards.
    The exit status is the worst output's; an output that fails does not stop the others.
    ``rerun`` is the command a conflict message says to run again once the markers are resolved.
    """
    data: dict[str, Any] = {"config": config_path, "dry_run": dry_run, "outputs": []}
    try:
        outputs = load_config(config_path)
    except ConfigError as exc:
        return failure("build", str(exc), data=data, cause=exc)
    known = [output.name for output in outputs]
    unknown = [name for name in names if name not in known]
    if unknown:
        return failure("build", f"{config_path} has no output named {', '.join(unknown)} (it has: {', '.join(known)})",
                       data=data)
    selected = [output for output in outputs if not names or output.name in names]

    result = Result(command="build", data=data)
    lines: list[str] = []
    briefs: list[str] = []
    traceback = ""
    for output in selected:
        built = _build_output(output, dry_run, rerun)
        data["outputs"].append({"name": output.name, **built.as_json()})
        result.exit_code = max(result.exit_code, built.exit_code)
        result.errors.extend(f"{output.name}: {error}" for error in built.errors)
        result.warnings.extend(f"{output.name}: {warning}" for warning in built.warnings)
        result.notes.extend(f"{output.name}: {note}" for note in built.notes)
        result.files.extend(built.files)
        traceback += built.traceback
        status = "ok" if built.exit_code == EXIT_OK else "FAILED"
        lines.append(f"[{output.name}] {output.generator}: {output.model} -> {output.output_dir}: {status}")
        lines.extend(f"  {line}" for line in built.text.splitlines())
        ok_brief = built.brief.rstrip(".") if built.exit_code == EXIT_OK else ""
        briefs.append(f"[{output.name}] {ok_brief or status}")
    failed = sum(entry["exit_code"] != EXIT_OK for entry in data["outputs"])
    verdict = f"{'FAILED' if failed else 'OK'}: {plural(len(selected), 'output')}, {failed} failed"
    lines.append(verdict)
    if not dry_run:
        config_file = Path(config_path)
        updated = sync_ignore(config_file.parent, _generated_dirs(config_file))
        result.files.extend(updated)
        result.notes.extend(f"updated {path}: the generated folders of {config_path}" for path in updated)
    result.text = "\n".join(lines)
    # A dry run is read for its list of files.
    result.brief = "" if dry_run else f"{verdict}: {' '.join(briefs)}"
    result.traceback = traceback
    return result


# ── apply ───────────────────────────────────────────────────────────────────

# The lines of the test command's output that 'apply' keeps in its JSON, and prints when the tests fail
# without a summary it knows: the end, where pytest, unittest and npm print their summary.
_TEST_TAIL_LINES = 40
# Where pytest lists the failing tests, one line each with its first error line.
_PYTEST_SUMMARY = re.compile(r"^=+ short test summary info =+$")
# The lines that name a failing test, in the output of a runner that prints no such section.
_FAILURE_LINE = re.compile(r"^(FAILED|ERROR|FAIL:|ERROR:|not ok\b)")
# pytest's error lines ('E   SyntaxError: ...'): the cause, which its one-line summary cuts or leaves out.
_PYTEST_ERROR_LINE = re.compile(r"^E( |$)")
# How much of those error lines 'apply' prints: the first error or two, not every traceback.
_ERROR_LINES_BYTES = 600
# A test that fails or errors in pytest's short summary: 'FAILED path::test - reason', 'ERROR path'.
_PYTEST_FAILING = re.compile(r"^(?:FAILED|ERROR) (\S+)")
# Colour codes, which a runner can print even to a pipe: matched lines must not carry them.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# pytest's exit status when it collected no test.
_PYTEST_NO_TESTS = 5


def _error_lines(output: list[str]) -> list[str]:
    """pytest's ``E`` lines, the first ones up to ``_ERROR_LINES_BYTES``: what went wrong and where."""
    kept: list[str] = []
    size = 0
    for line in output:
        if not _PYTEST_ERROR_LINE.match(line):
            continue
        size += len(line.encode("utf-8")) + 1
        if size > _ERROR_LINES_BYTES:
            kept.append("E   ...")
            break
        kept.append(line)
    return kept


def _pytest_summary_start(output: list[str]) -> int | None:
    return next((index for index, line in enumerate(output) if _PYTEST_SUMMARY.match(line.strip())), None)


def _failing_lines(output: list[str]) -> list[str]:
    """What to print of the output of tests that fail: pytest's short summary (one line per failing test
    and the counts), after its first error lines when it names an error (a module that does not import:
    the summary line has no cause), or the lines that name a failure and the last line, or else the end
    of the output."""
    start = _pytest_summary_start(output)
    if start is not None:
        summary = output[start + 1:]
        errored = any(line.startswith("ERROR ") for line in summary)
        return [*(_error_lines(output[:start]) if errored else []), *summary]
    failures = [line for line in output if _FAILURE_LINE.match(line.strip())]
    if failures:
        return [*failures, output[-1]] if output[-1] not in failures else failures
    return output[-_TEST_TAIL_LINES:]


def _failing_tests(output: list[str]) -> list[str]:
    """The tests pytest's short summary names as failed or in error, in its order."""
    start = _pytest_summary_start(output)
    if start is None:
        return []
    return [match.group(1) for line in output[start + 1:] if (match := _PYTEST_FAILING.match(line))]


def _test_argv(command: str) -> list[str]:
    """The test command as arguments; a leading ``python`` is the interpreter running besser, which has
    BESSER installed and is there when no ``python`` is on the path."""
    argv = shlex.split(command)
    if argv and argv[0] in ("python", "python3"):
        argv[0] = sys.executable
    return argv


def apply(config_path: str = CONFIG_FILENAME) -> Result:
    """Build every output of the configuration file, then run its ``test`` command: one call per model change.

    The build validates each model first and generates nothing from one with errors. The tests run
    only after a build that succeeded, from the configuration file's folder, without a shell and
    without besser's standard input. The one-line report always ends with what the tests did.
    """
    built = build((), config_path=config_path, rerun="besser apply")
    result = dataclasses.replace(built, command="apply")
    result.data = {**built.data, "test": None}
    if built.exit_code != EXIT_OK:
        return result
    try:
        command = load_test_command(config_path)
    except ConfigError as exc:  # build read the same file: only a change in between gets here
        return failure("apply", str(exc), data=result.data, cause=exc)
    if command is None:
        result.notes.append(f"no tests run: {config_path} has no 'test' command")
        return _with_tests(result, built, f"none (set 'test' in {Path(config_path).name})")
    try:
        process = subprocess.run(_test_argv(command), cwd=Path(config_path).parent, capture_output=True,
                                 stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                                 check=False)
    except (OSError, ValueError) as exc:
        return failure("apply", f"cannot run the test command '{command}': {exc}", EXIT_FAILED, result.data, exc)
    output = _ANSI.sub("", process.stdout + process.stderr).strip().splitlines()
    tail = output[-_TEST_TAIL_LINES:]
    result.data["test"] = {"command": command, "exit_code": process.returncode, "output_tail": tail,
                           "failing": _failing_tests(output)}
    if process.returncode == _PYTEST_NO_TESTS and "pytest" in command:
        return _with_tests(result, built, "none collected")
    if process.returncode != 0:
        result.exit_code = EXIT_FAILED
        result.errors.append(f"the tests failed (exit status {process.returncode}): {command}")
        result.text = "\n".join([*_failing_lines(output),
                                 f"FAILED: tests (exit status {process.returncode}); build {built.brief}"])
        result.brief = ""
        return result
    return _with_tests(result, built, tail[-1].strip() if tail else "no output")


def _with_tests(result: Result, built: Result, outcome: str) -> Result:
    """``result`` whose report, and its one line, end with what the tests did."""
    result.text = f"{built.text}\ntests: {outcome}"
    result.brief = f"{built.brief}; tests: {outcome}"
    return result


# ── model ───────────────────────────────────────────────────────────────────

# The model file 'besser model' edits when none is named and there is no besser.toml naming one.
DEFAULT_MODEL_FILE = "model.py"


def default_model_path(config_path: str = CONFIG_FILENAME) -> str:
    """The model file of ``besser.toml`` when it names one, else ``model.py``."""
    try:
        return load_default_model(config_path) or DEFAULT_MODEL_FILE
    except ConfigError:
        return DEFAULT_MODEL_FILE


def _read_model_text(path: Path) -> str:
    if not path.exists():
        raise _Stopped(failure("model", f"{path}: no such file; name the model file with -f FILE"))
    try:
        return path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _Stopped(failure("model", f"{path}: cannot be read as UTF-8 text: {exc}")) from exc


def model(operations: Sequence[Operation], model_path: str | None = None, use: Sequence[str] = ()) -> Result:
    """Make ``operations`` on the domain model of a ``.py`` model file, and write it if the result is valid.

    The file is written once, after every operation: when the new text loads, holds each change
    and has no validation error the old text did not have. Otherwise nothing is written.
    """
    path = Path(model_path or default_model_path())
    data: dict[str, Any] = {"model": path.as_posix(), "changes": []}
    if path.suffix.lower() != ".py":
        return failure("model", f"{path}: 'besser model' edits a B-UML .py file; for editor JSON, "
                                f"'besser convert {path} -o model.py' first", data=data)
    try:
        raw = _read_model_text(path)
    except _Stopped as stopped:
        return stopped.result
    # Edits are made on "\n" lines; a file written with "\r\n" gets them back.
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    try:
        before = _validate_model(pick_domain_model(load_python_text(path, text)[0], use))
        edits: list[Edited] = []
        for operation in operations:
            try:
                edited = run_operation(path, text, operation, use)
            except EditError as exc:
                raise EditError(f"{operation.line}: {exc}") from exc
            edits.append(edited)
            text = edited.text
    except (ModelLoadError, EditError) as exc:
        return failure("model", f"{exc} ({path} not changed)", data=data, cause=exc)
    try:
        loaded_after = pick_domain_model(load_python_text(path, text)[0], use)
    except (ModelLoadError, EditError) as exc:
        return failure("model", f"the change would break {path} ({exc}); not changed: make it by hand",
                       EXIT_FAILED, data, exc)
    missed = [problem for edited in edits if (problem := edited.check(loaded_after.model))]
    if missed:
        return failure("model", f"the change did not land as expected ({'; '.join(missed)}); {path} not changed: "
                                "make it by hand", EXIT_FAILED, data)
    after = _validate_model(loaded_after)
    new_errors = [error for error in after["errors"] if error not in before["errors"]]
    data["changes"] = [edited.summary for edited in edits]
    if new_errors:
        return Result(command="model", exit_code=EXIT_FAILED, data=data,
                      errors=[*new_errors, f"{path} not changed: the change makes the model invalid"])
    path.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    warnings = [warning for edited in edits for warning in edited.warnings]
    warnings += [f"still invalid: {error}" for error in after["errors"]]
    line = f"{path}: {'; '.join(edited.summary for edited in edits)}"
    return Result(command="model", files=[path.as_posix()], data=data, warnings=warnings, text=line, brief=line)


# ── schema ──────────────────────────────────────────────────────────────────

# Topic -> what it describes, besides the model kinds.
_SCHEMA_TOPICS = {
    "project": "an editor project file (.json), as a JSON Schema, and the Project class of a .py file",
    "result": "the object every command prints with --json, as a JSON Schema",
    "config": f"{CONFIG_FILENAME}, the file 'besser build' reads",
}


def schema_topics() -> list[str]:
    """What ``besser schema`` takes: the model kinds, then the other formats."""
    return [*STARTERS, *(topic for topic in _SCHEMA_TOPICS if topic not in STARTERS)]


def _schema_overview() -> Result:
    topics = [{"topic": kind, "summary": f"the B-UML classes of {STARTERS[kind].summary.partition(':')[0]} "
                                         f"({root.__name__})"} for kind, root in loading.MODEL_KINDS]
    topics.extend({"topic": topic, "summary": summary} for topic, summary in _SCHEMA_TOPICS.items())
    width = max(len(entry["topic"]) for entry in topics)
    lines = [
        "A model file is a B-UML Python script (.py), which is what to write and edit, or a Web",
        "Modeling Editor project (.json), which is what the editor opens. 'besser convert' turns",
        "one into the other. 'besser new FILE --kind KIND' writes a working example of a kind.",
        "",
        "Topics (besser schema TOPIC):",
        *(f"  {entry['topic'].ljust(width)}  {entry['summary']}" for entry in topics),
    ]
    return Result(command="schema", data={"topic": "", "topics": topics}, text="\n".join(lines),
                  text_is_result=True)


def schema(topic: str | None = None) -> Result:
    """Describe a format the command reads or writes: a model kind, a project, the JSON output, the configuration."""
    if not topic:
        return _schema_overview()
    data: dict[str, Any] = {"topic": topic}
    if topic == "result":
        data["schema"] = schema_module.result_schema()
        text = json.dumps(data["schema"], indent=2)
    elif topic == "config":
        data["example"] = CONFIG_EXAMPLE
        text = CONFIG_EXAMPLE.rstrip("\n")
    elif topic == "project":
        try:
            data["schema"] = schema_module.project_schema()
        except ModuleNotFoundError as exc:
            if (exc.name or "").split(".")[0] in ("", "besser"):
                raise
            return failure("schema", missing_backend_dependency_message(exc), data=data, cause=exc)
        data["description"] = {
            "kind": "project", "root_class": "Project",
            "doc": "In a .py file, a Project names the models the file holds; without one, every model bound "
                   "to a top-level name counts.",
            **schema_module.describe_package(schema_module.PROJECT_PACKAGE),
        }
        text = "\n".join([*schema_module.kind_text(data["description"]), "",
                          "An editor project file (.json), as a JSON Schema:", json.dumps(data["schema"], indent=2)])
    elif topic in schema_module.KIND_PACKAGES:
        data["description"] = schema_module.describe_kind(topic)
        text = "\n".join(schema_module.kind_text(data["description"]))
    else:
        return failure("schema", f"unknown topic '{topic}'; topics: {', '.join(schema_topics())}", data=data)
    return Result(command="schema", data=data, text=text, text_is_result=True)
